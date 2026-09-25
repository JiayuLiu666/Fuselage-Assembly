import torch
import os
from os import path
from shutil import copyfile
from bo_env import ClassicFuselageEnv
import numpy as np
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    NoisyExpectedImprovement,
    UpperConfidenceBound,
)

import botorch.settings as botorch_settings
from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import RFFKernel, RBFKernel
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")


def save_data(actions, response, num_of_queries, eps, true_response, filename=None):
    torch.save({
        'actions': actions,
        'response': response,
        'queries': num_of_queries,
        'true_response': true_response,
        'uncertainty': eps,
    }, filename)
    
def initialize_model(actions, response, device, M_target, yvar, state_dict=None):
        
    kernel_kwargs = {"nu": 2.5, "ard_num_dims": actions.shape[-1]}
    ls_init = torch.tensor([[0.2, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931,
         0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.2]],
                    dtype=torch.double, device=device)
    
    base_kernel = RFFKernel(**kernel_kwargs, num_samples=M_target)
    covar_module = ScaleKernel(base_kernel).to(device)
    model = FixedNoiseGP(
    actions, response, yvar,
    covar_module=covar_module).to(device)
    with torch.no_grad():
        model.covar_module.base_kernel.lengthscale.copy_(ls_init) 
        # model.covar_module.outputscale.copy_(torch.tensor(11.3646, dtype=torch.double, device=device))
        model.covar_module.outputscale.copy_(torch.tensor(1, dtype=torch.double, device=device))
    model.eval()
    return model, model.covar_module


def sample_sobol_on_grid(n=1, device=None, seed=123):
    device = device or torch.device("cpu")
    D = 18
    var_dims = [0, 17]  # variable dimensions
    V = len(var_dims)

    # Sobol in [0,1]^V
    sobol = torch.quasirandom.SobolEngine(dimension=V, scramble=True, seed=seed)
    u = sobol.draw(n).to(device)

    # Map to continuous range [-1, 1]
    x = 2.0 * u - 1.0  # u in [0,1] -> [-1,1]

    # Snap to nearest grid in {-1.0, -0.9, ..., 1.0}
    values = torch.linspace(-0.5, 0.5, 21, device=device)           # 21 points
    step = values[1] - values[0]                                    # 0.1
    idx = torch.round((x - values[0]) / step).clamp(0, values.numel()-1).long()

    out = torch.zeros(n, D, device=device)
    out[:, var_dims] = values[idx]
    return out

def W_GP_UCB(model, x_D, response, V_t, Unweighted_Phi, eps_list, beta, lam):
    eps_list = torch.tensor(eps_list, device=response.device, dtype=response.dtype)
    W = torch.diag(1 / torch.pow(eps_list, 2)).to(response)
    V_t_inv = torch.linalg.inv(V_t)

    Y_weighted = torch.matmul(W, response.reshape(-1, 1))
    nu_t = V_t_inv @ Unweighted_Phi.T @ Y_weighted  # [2M,1]

    features_matrix = model.covar_module.base_kernel.get_features(x_D, 18, normalize=True)  # [|x_D|,2M]
    mean = (model.covar_module.outputscale * (features_matrix.to(response) @ nu_t)).reshape(-1)
    variance = model.covar_module.outputscale * torch.diagonal(
        lam * features_matrix.to(response) @ V_t_inv @ features_matrix.T.to(response)
    )
    sigma = variance.clamp_min(1e-12).sqrt().reshape(-1)

    score = mean + beta.sqrt() * sigma
    idx_local = torch.argmax(score)                         # index inside x_D
    x_next = x_D[idx_local].reshape(1, -1)
    return x_next, idx_local, V_t_inv, mean[idx_local], variance[idx_local]

@dataclass
class TRState:
    var_dims: list            # 哪些维度参与 TR 距离（你这里 [0,17]）
    step: float               # 网格步长（你这里 0.05）
    radius: float             # 当前 TR 半径（L∞ 半径）
    r_min: float
    r_max: float
    succ_tol: int = 3         # 连续成功多少次 -> 扩大
    fail_tol: int = 10        # 连续失败多少次 -> 缩小
    inc: float = 2.0          # 扩大倍数
    dec: float = 2.0          # 缩小倍数
    succ: int = 0
    fail: int = 0
    center: torch.Tensor = None  # [1, D]

    def snap_radius(self):
        # 把半径对齐到网格步长，避免出现半径比步长更细却选不到点
        m = max(1, int(round(self.radius / self.step)))
        self.radius = float(m * self.step)
        self.radius = float(min(max(self.radius, self.r_min), self.r_max))

    def update(self, improved: bool):
        if improved:
            self.succ += 1
            self.fail = 0
        else:
            self.fail += 1
            self.succ = 0

        if self.succ >= self.succ_tol:
            self.radius = min(self.radius * self.inc, self.r_max)
            self.succ = 0
        if self.fail >= self.fail_tol:
            self.radius = max(self.radius / self.dec, self.r_min)
            self.fail = 0

        self.snap_radius()

    def should_restart(self):
        return self.radius <= self.r_min + 1e-12
    
def tr_filter_candidates(search_space: torch.Tensor, center: torch.Tensor, radius: float, var_dims: list):
    """
    search_space: [M, D]
    center:      [1, D]
    return:
      X_tr:       [M_tr, D]
      idx_tr:     [M_tr] indices in original search_space
    """
    # 只对 var_dims 做 L∞ 距离
    diff = (search_space[:, var_dims] - center[:, var_dims]).abs()   # [M, V]
    dist_inf = diff.max(dim=-1).values                               # [M]
    mask = dist_inf <= radius + 1e-12
    idx_tr = mask.nonzero(as_tuple=True)[0]
    X_tr = search_space[idx_tr]
    return X_tr, idx_tr


def pick_random_center_from_space(search_space: torch.Tensor, device=None):
    device = device or search_space.device
    ridx = torch.randint(0, search_space.shape[0], (1,), device=device).item()
    return search_space[ridx:ridx+1], ridx
        
        
if __name__ == "__main__":    
    obs_noise = 0.2 ** 2
    M_target = 400
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Discrete GP-UCB, 2 actuators")
    ###
    __file__ = 'FuselageActuators'
    folder = path.join(__file__, 'AnsysFiles', "Test") 
    # file = random.choice(os.listdir(folder))
    file = 'SolutionInputDP52.inp'
    filepath = path.join(folder, file)
    original_input_filename = filepath
    print("Initial shape from", file.split('.')[0])
    ###

    log_dir = "Experiments"
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
    env_name = "Classic_TurBO_Discrete"
    log_dir = log_dir + '/' + env_name + '/' + 'exp_set_1' + '/'
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    input_filename = log_dir + file
    copyfile(original_input_filename, input_filename)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    #### create new log file for each run
    log_f_name = log_dir + 'PPO_' + env_name + "_log_" + str(run_num) + ".txt"

    ### BO LOOP
    trails = np.arange(5)
    for tri in trails:
        seed = int(tri)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_default_dtype(torch.float64)
        botorch_settings.debug._set_state(True)
        env = ClassicFuselageEnv(ip='129.161.91.97', obs_noise=obs_noise)  
        _ = env.reset(input_filename, seed=seed)
            
        # Initialize random data
        print("noise level: ", obs_noise)
        
        initial_response = []
        ### create search space
        eta = 1
        lam = 1
        search_space = sample_sobol_on_grid(n=441, seed=seed, device=device)
        random_index = torch.randint(0, search_space.shape[0], (1,)).item()
        actions = search_space[random_index].to(device).reshape(1,-1)
        method = 'chebyshev'
        response, true_response, num_of_queries = env.step_surrogate(action=actions, device=device, eps=eta * 1, method=method)
        yvar = torch.tensor([[obs_noise]], dtype=torch.float64, device=device)
        eps_list = [1]
        
        true_response = torch.tensor(true_response).reshape(1,1).to(actions)
        response = response.reshape(1,1).to(actions) 
        num_of_queries_num = num_of_queries.item()
        
        Weighted_Phi = torch.zeros((actions.shape[0], 2 * M_target)).to(actions) #t,2048
        Unweighted_Phi = torch.zeros((actions.shape[0], 2 * M_target)).to(actions)
        model_ei, kernel = initialize_model(actions=actions, response=response, M_target=M_target, yvar=yvar, device=device)
         
        for i in range(actions[0:1].shape[0]):
            x = actions[0:1][i,:].reshape(1,-1)
            features = model_ei.covar_module.base_kernel.get_features(x, 18, normalize=True) #1,400
            # features = features / math.sqrt(2 * M_target)
            Unweighted_Phi[i, :] = features
            weighted_features = features * (1 / eps_list[i])
            Weighted_Phi[i, :] = weighted_features
        
        V_t = model_ei.covar_module.outputscale * torch.matmul(Weighted_Phi[0:1,:].T, Weighted_Phi[0:1,:]) + lam * torch.eye(2 * M_target).to(actions)

        # V_t = torch.matmul(Unweighted_Phi[0:1,:].T, Unweighted_Phi[0:1,:]) + lam * torch.eye(2 * M_target).to(actions)
        
        n_iters = 3001
        _ = env.reset(input_filename, seed=seed)
        ts = torch.arange(1, n_iters + 1)
        beta_t =  (1 + np.sqrt(np.log(ts) ** 2)) ** 2
        model_ei.eval()
        iterations = 1
        
        #### TurBO
        grid_values = torch.linspace(-0.5, 0.5, 21, device=device)
        grid_step = float((grid_values[1] - grid_values[0]).item())

        var_dims = [0, 17]       # 你当前只在 2 个 actuator 上动
        tr = TRState(
            var_dims=var_dims,
            step=grid_step,
            radius=0.20,         # 初始半径：建议 0.10 (=2 steps)
            r_min=grid_step,     # 最小半径至少一个 step
            r_max=0.50,          # 最大半径覆盖全域
            succ_tol=3,
            fail_tol=5,
            inc=3.0,
            dec=2.0,
        )
        tr.center = actions[-1:].detach().clone()  # 初始中心：用当前点
        tr.snap_radius()
        
        # best according to the observed noisy response
        best_idx = torch.argmax(response.view(-1))  # response: [t,1]
        best_y = response[best_idx].item()          # float
        best_x = actions[best_idx:best_idx+1].detach().clone()  # [1,18]
        
        available = search_space.clone()
        
        while num_of_queries_num < n_iters:
            torch.cuda.empty_cache()
            model_ei.eval()
            #----------------------------------
            # new_actions, V_t_inv = GP_UCB(model=model_ei, \
            #     x_D=search_space, response=response, V_t=V_t, Unweighted_Phi=Unweighted_Phi, eps_list=eps_list)
            #----------------------------------
            X_tr, idx_tr = tr_filter_candidates(available, tr.center, tr.radius, tr.var_dims)
            if X_tr.shape[0] == 0:
                # fallback：用全域
                X_tr = available
                idx_tr = torch.arange(available.shape[0], device=device)
            
            new_actions, idx_local, V_t_inv, posterior_mean, var = W_GP_UCB(model=model_ei, \
                x_D=X_tr, response=response, V_t=V_t, Unweighted_Phi=Unweighted_Phi, eps_list=eps_list, beta=beta_t[len(response)-1], lam=lam)
            #----------------------------------
            # picked_global_idx = int(idx_tr[idx_local].item())
            
            eps = (torch.sqrt(var) / np.sqrt(lam).item()).item()
            new_response, true_responses, num_oracle_queries = env.step_surrogate(action=new_actions, device=device, eps=eta * eps, method=method)
            ####
            sigma_obs = eta * eps_list[-1]
            yvar_new = torch.tensor([[max(obs_noise, eps**2)]], dtype=torch.float64, device=device)
            yvar = torch.cat([yvar, yvar_new], dim=0)
            ####
            eps_list.append(eps)
            num_of_queries_num += num_oracle_queries.item()
            actions = torch.cat([actions, new_actions.to(torch.float64).to(device)]) #input
            response = torch.cat([response, new_response.to(torch.float64).to(device)]) #output
            true_response = torch.cat([true_response, true_responses.to(true_response)])

            x_max_feature = model_ei.covar_module.base_kernel.get_features(new_actions, 18, normalize=True)
            weighted_feature = x_max_feature * (1 / eps_list[-1])
            V_t = V_t + torch.matmul(weighted_feature.T, weighted_feature)
            Unweighted_Phi = torch.cat([Unweighted_Phi, x_max_feature])
            
            
            _ = env.reset(input_filename, seed=seed)
            num_of_queries = torch.cat([num_of_queries, num_oracle_queries.to(device)])
            
            y_new = response[-1].item()
            improved = (y_new > best_y + 1e-6)
            
            if improved:
                best_y = y_new
                best_x = new_actions.detach().clone()
            
            tr.center = best_x.detach().clone()
            tr.update(improved)
            
            # 半径太小就 restart（换中心、重置半径）
            if tr.should_restart():
                tr.radius = 0.10
                tr.succ = 0
                tr.fail = 0
                # 选一个新的中心（也可选 best_x，这里给你随机重启）
                tr.center, _ = pick_random_center_from_space(available, device=device)
                tr.snap_radius()
            
            
            
            save_data(actions, response, num_of_queries, eps_list, true_response, log_dir + str(obs_noise) + 'training_data_' + str(tri) + '_.pth')
            # if iterations % 100 == 0:
            iterations += 1
            print('iterstion {0},  y: {1}  f(x): {2} posterior_mean: {3} eps: {4}'.format(iterations, round(response[-1].item(), 3), 
                                                                                     round(true_response[-1].item(), 3), 
                                                                                     round(posterior_mean.item(), 3),
                                                                                     round(eps_list[-1], 3)))


