from joblib import dump, load
from datetime import datetime
import os
from os import path
import gpytorch
from ansys.mapdl.core import launch_mapdl, launcher, Mapdl
from ansys.mapdl import reader as mapdl_reader
from shutil import copyfile
import torch
from quantum_bo_env import QuantumFuselageEnv
import botorch.settings as botorch_settings
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP, FixedNoiseGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import MaternKernel, RFFKernel, ScaleKernel, LinearKernel, RBFKernel
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    NoisyExpectedImprovement,
    UpperConfidenceBound,
)
import warnings
warnings.filterwarnings("ignore")
import numpy as np

def save_data(actions, response, true_response, num_of_queries, eps, file_path):
    # print("Saving data to:", file_path)  # Debug: Print the file path
    torch.save({
        'actions': actions,
        'response': response,
        'queries': num_of_queries,
        'uncertainty': eps,
        'true_response': true_response
    }, file_path)
        
def initialize_model(actions, response, device, M_target, yvar, state_dict=None):
    # --- FORCE consistent dtype ---
    dtype = torch.float32
    actions = actions.to(device=device, dtype=dtype)
    response = response.to(device=device, dtype=dtype)
    yvar = yvar.to(device=device, dtype=dtype)

    kernel_kwargs = {"nu": 2.5, "ard_num_dims": actions.shape[-1]}
    ls_init = torch.tensor([[0.1, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931,
                             0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.1]],
                           dtype=dtype, device=device)

    base_kernel = RFFKernel(**kernel_kwargs, num_samples=M_target)
    covar_module = ScaleKernel(base_kernel).to(device=device, dtype=dtype)

    model = FixedNoiseGP(actions, response, yvar, covar_module=covar_module).to(device)
    with torch.no_grad():
        model.covar_module.base_kernel.lengthscale.copy_(ls_init)
        model.covar_module.outputscale.copy_(torch.tensor(1.0, dtype=dtype, device=device))
    model.eval()
    return model, model.covar_module

def sample_sobol_on_grid(n=1, device=None, seed=123): # 21*21 = 441
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
    values = torch.linspace(-0.5, 0.5, 41, device=device)           # 21 points
    step = values[1] - values[0]                                    # 0.1
    idx = torch.round((x - values[0]) / step).clamp(0, values.numel()-1).long()

    out = torch.zeros(n, D, device=device)
    out[:, var_dims] = values[idx]
    return out

from dataclasses import dataclass

@dataclass
class TRState:
    var_dims: list
    step: float
    radius: float
    r_min: float
    r_max: float
    succ_tol: int = 3
    fail_tol: int = 10
    inc: float = 2.0
    dec: float = 2.0
    succ: int = 0
    fail: int = 0
    center: torch.Tensor = None  # [1,D]

    def snap_radius(self):
        # align radius to grid step
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
    L-infinity trust region on selected var_dims.
    search_space: [M,D], center: [1,D]
    returns: X_tr [M_tr,D], idx_tr [M_tr] indices in original search_space
    """
    diff = (search_space[:, var_dims] - center[:, var_dims]).abs()
    dist_inf = diff.max(dim=-1).values
    mask = dist_inf <= radius + 1e-12
    idx_tr = mask.nonzero(as_tuple=True)[0]
    return search_space[idx_tr], idx_tr


def pick_random_center(search_space: torch.Tensor):
    ridx = torch.randint(0, search_space.shape[0], (1,), device=search_space.device).item()
    return search_space[ridx:ridx+1], ridx

def W_GP_UCB(
    model,
    x_D,
    response,
    V_t,
    Unweighted_Phi,
    eps_list,
    beta,
    lam,
    *,
    # --- stability knobs ---
    eps_floor=None,          # e.g. sqrt(obs_noise) or MC standard error floor
    jitter=1e-6,
    max_jitter=1e-2,
):
    """
    Stable weighted GP-UCB with Cholesky solves (no explicit inverse).

    Args:
        model: GP model with RFFKernel providing get_features
        x_D:    candidate set [N, d]
        response: y_t [t,1] or [t]
        V_t:    Gram-like matrix [2M,2M] (already includes lam*I and weighted features)
        Unweighted_Phi: [t, 2M]
        eps_list: length t list/1D tensor, per-observation noise scale (or uncertainty proxy)
        beta: scalar tensor/float
        lam: scalar (your ridge / regularization factor used in variance expression)
        eps_floor: if not None, clamp eps >= eps_floor to prevent huge weights
        jitter: initial diagonal jitter for Cholesky
    Returns:
        x_next: [1,d]
        idx_local: int index in x_D
        mean_at: scalar tensor
        var_at: scalar tensor
        (optional) info dict
    """
    device = V_t.device
    dtype = V_t.dtype

    # ---- format y ----
    y = response
    if y.dim() == 1:
        y = y.view(-1, 1)
    else:
        y = y.reshape(-1, 1)
    y = y.to(device=device, dtype=dtype)

    # ---- eps weights (with floor) ----
    eps = torch.as_tensor(eps_list, device=device, dtype=dtype).view(-1)
    if eps_floor is not None:
        eps_floor_t = torch.tensor(float(eps_floor), device=device, dtype=dtype)
        eps = torch.clamp(eps, min=eps_floor_t)
    # guard against zeros / negative
    eps = torch.clamp(eps, min=torch.finfo(dtype).eps)

    # weighted targets: y_i / eps_i^2
    # (equivalent to diag(1/eps^2) @ y, but avoids building W explicitly)
    y_weighted = y / (eps.view(-1, 1) ** 2)

    # ---- move Phi ----
    Phi = Unweighted_Phi.to(device=device, dtype=dtype)  # [t, 2M]

    # ---- Cholesky with adaptive jitter ----
    I = torch.eye(V_t.size(0), device=device, dtype=dtype)
    V = V_t  # assume already on device/dtype

    used_jitter = float(jitter)
    while True:
        try:
            L = torch.linalg.cholesky(V + used_jitter * I)
            break
        except RuntimeError:
            used_jitter *= 10.0
            if used_jitter > max_jitter:
                raise RuntimeError(
                    f"Cholesky failed even with jitter={used_jitter}. "
                    "V_t is likely not PSD / numerically unstable."
                )

    def solve_V(b):
        # solves (V + jitter I) x = b
        return torch.cholesky_solve(b, L)

    # ---- nu_t = (V)^{-1} Phi^T y_weighted ----
    rhs = Phi.transpose(-2, -1) @ y_weighted  # [2M,1]
    nu_t = solve_V(rhs)                       # [2M,1]

    # ---- features for candidates ----
    # get_features returns [N, 2M] for RFFKernel (as you used)
    F = model.covar_module.base_kernel.get_features(x_D, 18, normalize=True)
    F = F.to(device=device, dtype=dtype)      # [N, 2M]

    outscale = model.covar_module.outputscale.to(device=device, dtype=dtype)

    # ---- posterior mean: outscale * F nu ----
    mean = (outscale * (F @ nu_t)).view(-1)   # [N]

    # ---- posterior variance diag: outscale * lam * diag(F V^{-1} F^T) ----
    # compute diag(F V^{-1} F^T) without forming NxN:
    # diag = sum_j F_ij * (V^{-1} F^T)_j,i
    Vinv_FT = solve_V(F.transpose(-2, -1))    # [2M, N]
    diag = (F * Vinv_FT.transpose(-2, -1)).sum(dim=-1)  # [N]
    variance = outscale * (lam * diag)        # [N]

    # numerical guard
    variance = torch.clamp(variance, min=1e-12)

    sigma = variance.sqrt()
    beta_t = beta if torch.is_tensor(beta) else torch.tensor(beta, device=device, dtype=dtype)
    score = mean + beta_t.sqrt() * sigma

    # ---- pick best ----
    idx_local = int(torch.argmax(score).item())
    x_next = x_D[idx_local:idx_local+1].clone()

    mean_at = mean[idx_local]
    var_at = variance[idx_local]
    
    return x_next, idx_local, mean_at, var_at   

from dataclasses import dataclass

@dataclass
class MultiTR:
    tr: TRState
    best_x: torch.Tensor    # [1,D] 该TR内部最优
    best_y: float           # 该TR内部最优值
    mask: torch.Tensor      # [|search_space|] bool, 该TR自己的可用点
    
if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    print("Quantum Discrete GP-UCB, 2 actuators")

    obs_noise =  0.2 ** 2
    M_target = 400
    print('noise_level: ', obs_noise)
    ###
    __file__ = 'FuselageActuators'
    folder = path.join(__file__, 'AnsysFiles', "Test") 
    # file = random.choice(os.listdir(folder))
    file = 'SolutionInputDP52.inp'
    filepath = path.join(folder, file)
    original_input_filename = filepath
    print("Initial shape from", file.split('.')[0])

    log_dir = "Experiments"
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
    env_name = "Quantum_TurBo_Discrete"
    log_dir = log_dir + '/' + env_name + '/' + 'exp_set_1' + '/' # 1: real Machine ; 0: Simulator
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    input_filename = log_dir + file
    copyfile(original_input_filename, input_filename)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    #### create new log file for each run
    
    trails = np.arange(5)
    for tri in trails:
        seed = int(tri)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_default_dtype(torch.float64)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        botorch_settings.debug._set_state(True)
        env = QuantumFuselageEnv(ip='129.161.91.97', obs_noise=obs_noise) 
        _ = env.reset(input_filename, seed=seed)   #tensor. 1,1

        search_space = sample_sobol_on_grid(n=1681, seed=seed, device=device)

        lam = 1
        eta = 1
        eps_list = [1]
        random_index = torch.randint(0, search_space.shape[0], (1,)).item()
        actions = search_space[random_index].to(device).reshape(1,-1)
        response, true_responses, num_of_queries = env.step_surrogate(actions, eps=1)
        num_of_queries = num_of_queries.to(actions)
        true_response = torch.tensor(true_responses).reshape(1,1).to(actions)
        response = response.reshape(1,1).to(actions)
        num_of_queries_num = num_of_queries.item()
        
        Weighted_Phi = torch.zeros((actions.shape[0], 2 * M_target)).to(actions) #t,2048
        Unweighted_Phi = torch.zeros((actions.shape[0], 2 * M_target)).to(actions)
        
        
        yvar = torch.tensor([[obs_noise]], dtype=torch.float64, device=device)
        model_ei, kernel = initialize_model(actions=actions, response=response, M_target=M_target, yvar=yvar, device=device)

        for i in range(actions[0:1].shape[0]):
            x = actions[0:1][i,:].reshape(1,-1)
            features = model_ei.covar_module.base_kernel.get_features(x, 18, normalize=True) #1,400
            # features = features / math.sqrt(2 * M_target)
            Unweighted_Phi[i, :] = features
            weighted_features = features * (1 / eps_list[i])
            Weighted_Phi[i, :] = weighted_features
        
        V_t = model_ei.covar_module.outputscale * torch.matmul(Weighted_Phi[0:1,:].T, Weighted_Phi[0:1,:]) + lam * torch.eye(2 * M_target).to(actions)

        iterations = 1
        n_iter = 10001
        ts = torch.arange(1, n_iter)
        beta_t = (1 + np.sqrt(np.log(ts) ** 2))**2
        
        _ = env.reset(input_filename, seed=seed)
        
        # ----- best init -----
        best_idx = torch.argmax(response.view(-1))
        best_y = response[best_idx].item()
        best_x = actions[best_idx:best_idx+1].detach().clone()

        # ----- TuRBO init -----
        grid_values = torch.linspace(-0.5, 0.5, 41, device=device)
        grid_step = float((grid_values[1] - grid_values[0]).item())

        var_dims = [0, 17]
        
        K = 3  # 你要几个中心（几个TurBO）
        trs = []

        # 全局最优（可选）
        global_best_y = best_y
        global_best_x = best_x.detach().clone()

        for k in range(K):
            trk = TRState(
                var_dims=var_dims,
                step=grid_step,
                radius=0.20,
                r_min=grid_step,
                r_max=0.50,
                succ_tol=3,
                fail_tol=5,
                inc=2,
                dec=2,
            )

            # 方案1：随机中心
            c0, _ = pick_random_center(search_space)
            trk.center = c0.clone()
            trk.snap_radius()

            # 也可以用：让第0个中心从全局best开始，其余随机
            # if k == 0:
            #     trk.center = global_best_x.clone()
            # else:
            #     trk.center, _ = pick_random_center(search_space)

            m = torch.ones(search_space.shape[0], dtype=torch.bool, device=device)

            trs.append(MultiTR(
                tr=trk,
                best_x=trk.center.clone(),
                best_y=best_y,          # 用全局初始 best_y
                mask=m,
            ))
        
        tol_improve = 1e-4

        while num_of_queries_num < n_iter:
            model_ei.eval()
            iterations += 1
            torch.cuda.empty_cache()

            # ========= 1) 每个TR各自提出一个候选 =========
            proposals = []  # (score_proxy, k, new_actions, picked_global_idx, posterior_mean, var, TR_size)

            for k in range(K):
                mk = trs[k]
                trk = mk.tr

                # TR k 的 available
                available_k = search_space[mk.mask]

                # TR过滤
                X_tr, idx_tr_local = tr_filter_candidates(available_k, trk.center, trk.radius, trk.var_dims)
                if X_tr.shape[0] == 0:
                    X_tr = available_k
                    idx_tr_local = torch.arange(available_k.shape[0], device=device)

                # 在 TR 内选点
                new_actions_k, idx_local, post_mean_k, var_k = W_GP_UCB(
                    model=model_ei,
                    x_D=X_tr,
                    response=response,
                    V_t=V_t,
                    Unweighted_Phi=Unweighted_Phi,
                    eps_list=eps_list,
                    beta=beta_t[len(response)-1],
                    lam=lam
                )

                # idx_local -> TR_k available 的局部索引
                picked_in_available_k = int(idx_tr_local[idx_local].item())

                # 需要把它映射回 search_space 的全局索引，用 mask 做一次映射
                global_indices_k = mk.mask.nonzero(as_tuple=True)[0]  # [n_avail_k]
                picked_global_idx = int(global_indices_k[picked_in_available_k].item())

                # 用 UCB 值当作比较的 score proxy（不一定等于真实score，但足够选冠军）
                score_proxy = (post_mean_k + torch.sqrt(beta_t[len(response)-1]) * torch.sqrt(var_k)).item()

                proposals.append((score_proxy, k, new_actions_k, picked_global_idx, post_mean_k, var_k, X_tr.shape[0]))

            # ========= 2) 选冠军（哪个TR的候选最好）=========
            proposals.sort(key=lambda t: t[0], reverse=True)
            score_proxy, k_star, new_actions, picked_global_idx, posterior_mean, var, tr_size = proposals[0]

            # ========= 3) 真实评估冠军点 =========
            eps = (torch.sqrt(var) / np.sqrt(lam).item()).item()
            new_response, true_responses, num_oracle_queries = env.step_surrogate(new_actions, eps, device)

            eps_list.append(eps)
            num_of_queries_num += num_oracle_queries.item()

            actions = torch.cat([actions, new_actions.to(torch.float64).to(device)])
            response = torch.cat([response, new_response.to(torch.float64).to(device)])
            true_response = torch.cat([true_response, true_responses.to(true_response)])
            num_of_queries = torch.cat([num_of_queries, num_oracle_queries.to(device)])

            # ========= 4) 更新 V_t / Phi =========
            x_feat = model_ei.covar_module.base_kernel.get_features(new_actions, 18, normalize=True)
            weighted_feat = x_feat * (1 / eps_list[-1])
            V_t = V_t + (weighted_feat.T @ weighted_feat)
            Unweighted_Phi = torch.cat([Unweighted_Phi, x_feat])

            # ========= 5) 更新冠军 TR（只更新这个TR的 succ/fail/radius/center/mask/best）=========
            y_new = float(new_response[-1].item())

            mk = trs[k_star]
            trk = mk.tr

            # TR 内部 best_old 用来判断 improved（局部成功）
            best_y_old_local = mk.best_y
            improved_local = (y_new > best_y_old_local + tol_improve)

            if improved_local:
                mk.best_y = y_new
                mk.best_x = new_actions.detach().clone()

            # 全局 best（可选）
            if y_new > global_best_y + tol_improve:
                global_best_y = y_new
                global_best_x = new_actions.detach().clone()

            # center 建议用“该TR内部best”，不要强行用 global_best（否则所有TR会挤到一个地方）
            trk.center = mk.best_x.clone() if mk.best_y > float("-inf") else new_actions.detach().clone()
            trk.update(improved_local)

            # 失败就 mask 掉冠军点（你原本逻辑）
            if not improved_local:
                mk.mask[picked_global_idx] = False

                # 如果这个TR没点了，直接重启
                if mk.mask.sum().item() == 0:
                    mk.mask[:] = True

            # restart if too small
            if trk.should_restart():
                trk.radius = 0.20
                trk.succ = 0
                trk.fail = 0
                trk.center, _ = pick_random_center(search_space)
                trk.snap_radius()
                mk.best_x = trk.center.clone()
                mk.best_y = float("-inf")
                mk.mask[:] = True

            # reset env if needed
            _ = env.reset(input_filename, seed=seed)

            print(
                f"Stage {iterations} | y={y_new:.3f} true={true_responses[-1].item():.3f} "
                f"| global_best={global_best_y:.3f} | TR#{k_star} r={trk.radius:.3f} succ={trk.succ} fail={trk.fail} "
                f"| post_mean={posterior_mean.item():.3f} | eps={eps_list[-1]:.3f} | TR_size={tr_size}"
            )

            save_data(
                actions, response, true_response, num_of_queries, eps_list,
                log_dir + str(obs_noise) + 'quan_training_data_' + str(tri) + '_.pth'
            )
                    
    
            