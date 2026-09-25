from joblib import dump, load
from datetime import datetime
import os
import sys
from os import path
import gpytorch
from ansys.mapdl.core import launch_mapdl, launcher, Mapdl
from ansys.mapdl import reader as mapdl_reader
from shutil import copyfile
import torch
from quantum_bo_env_constraint import QuantumFuselageEnv
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

from utils import build_train_gp_with_rff, sample_actions
import math

from torch.distributions import Normal

@torch.no_grad()
def compute_pof_from_posterior(mu_c: torch.Tensor, sig_c: torch.Tensor, eps: float = 1e-12):
    z = mu_c / sig_c.clamp_min(eps)
    return Normal(0.0, 1.0).cdf(z)   # [N]

def minmax_norm(a: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12):
    # a: [N], mask: [N] bool
    v = a[mask]
    if v.numel() == 0:
        return a * 0.0
    lo = v.min()
    hi = v.max()
    return (a - lo) / (hi - lo + eps)

def lambda_by_stage(stage, lam0=1, t0=20, p=1.0):
    return float(lam0 * (t0 / (t0 + max(1, stage)))**p)

def lambda_by_eps(eps, eps0=0.2, lam0=0.8):
    # eps 大：更偏扩张；eps 小：更偏优化
    return float(lam0 * (eps / (eps + eps0)))

def save_data(actions, response, true_response, num_of_queries, eps, file_path):
    # print("Saving data to:", file_path)  # Debug: Print the file path
    torch.save({
        'actions': actions,
        'response': response,
        'queries': num_of_queries,
        'uncertainty': eps,
        'true_response': true_response
    }, file_path)
    
def initialize_c_model(actions, response, device):
    yvar = torch.full_like(response, 1e-6, device=device)
    
    model = FixedNoiseGP(
    actions, response, yvar).to(device)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll

def refit_with_warm_start(train_X, train_Y, device, M_target, old_model=None):
    state = None if old_model is None else old_model.state_dict()
    model, _ = initialize_model(actions=train_X, response=train_Y, device=device, M_target=M_target)
    if state is not None:
        model.load_state_dict(state, strict=False)  # warm start（不一定所有参数都匹配）

    model.eval(); model.likelihood.eval()
    return model
    
def initialize_model(actions, response, device, M_target, state_dict=None):
    yvar = torch.full_like(response, 1e-6, device=device)
    kernel_kwargs = {"nu": 2.5, "ard_num_dims": actions.shape[-1]}
    
    # ls_init = torch.tensor([[0.1, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931,
    #      0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.6931, 0.1]],
    #                 dtype=torch.double, device=device)
    
    base_kernel = RFFKernel(**kernel_kwargs, num_samples=M_target)
    covar_module = ScaleKernel(base_kernel).to(device)

    model = FixedNoiseGP(actions, response, yvar, covar_module=covar_module,).to(device)
    # with torch.no_grad():
    #     model.covar_module.base_kernel.lengthscale.copy_(ls_init) 
    #     model.covar_module.outputscale.copy_(torch.tensor(1, dtype=torch.double, device=device))
        
    model.eval()
    return model, model.covar_module

def fit_rff_fixednoise_gp(model: FixedNoiseGP):
    model.train()
    model.likelihood.train()

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # BoTorch 会用 LBFGS 或 Adam 之类优化超参数（取决于版本/设置）
    fit_gpytorch_mll(mll)

    model.eval()
    model.likelihood.eval()
    return model

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
    values = torch.linspace(-1, 1, 41, device=device)           # 21 points
    step = values[1] - values[0]                                    # 0.1
    idx = torch.round((x - values[0]) / step).clamp(0, values.numel()-1).long()

    out = torch.zeros(n, D, device=device)
    out[:, var_dims] = values[idx]
    return out

@torch.no_grad()
def compute_lcb_c_botorch(c_model, X, beta_c: float):
    """
    X: [N, d]
    return: mu [N], sigma [N], lcb [N], ucb [N]
    """
    post = c_model.posterior(X)
    mu = post.mean.view(-1)
    var = post.variance.view(-1).clamp_min(1e-12)
    sigma = var.sqrt()
    rad = math.sqrt(beta_c)
    lcb = mu - rad * sigma
    ucb = mu + rad * sigma
    return mu, sigma, lcb, ucb

def W_GP_UCB_scores(model, x_D, response, V_t, Unweighted_Phi, eps_list, beta, lam):
    eps_list = torch.tensor(eps_list, device=V_t.device, dtype=V_t.dtype)
    W = torch.diag(1.0 / (eps_list**2)).to(V_t)
    Unweighted_Phi = Unweighted_Phi.to(V_t)
    response = response.to(V_t)

    V_t_inv = torch.linalg.inv(V_t)

    Y_weighted = (W @ response.reshape(-1, 1))
    
    
    nu_t = V_t_inv @ Unweighted_Phi.T @ Y_weighted   # [2M,1]

    features = model.covar_module.base_kernel.get_features(x_D, 18, normalize=True)  # [N,2M]
    mean = (model.covar_module.outputscale/ model.covar_module.outputscale* (features @ nu_t)).reshape(-1)          # [N]
    var = model.covar_module.outputscale / model.covar_module.outputscale * torch.diagonal(lam * features @ V_t_inv @ features.T)
    sigma = var.clamp_min(1e-12).sqrt()

    ucb = mean + beta.sqrt() * sigma
    return ucb, mean, sigma, var  
    
    
if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    print("Quantum Discrete GP-UCB, 2 actuators")

    noise_std = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    obs_noise = noise_std ** 2
    M_target = 1024
    print('noise_level: ', obs_noise)
    ###
    __file__ = 'FuselageActuators'
    folder = path.join(__file__, 'AnsysFiles', "Test") 
    # file = random.choice(os.listdir(folder))
    file = 'SolutionInputDP52.inp'
    filepath = path.join(folder, file)
    original_input_filename = filepath
    print("Initial shape from", file.split('.')[0])

    log_dir = "Experiments_constraints"
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
    env_name = "Quantum_Discrete_cUCB_POF"
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
        
        N = search_space.shape[0]  # 1681
        s = torch.zeros(N, device=device, dtype=torch.float64)

        #----------------------------------------------------------------------------------------------------------
        # actions = torch.zeros(1, 18, device=device, dtype=torch.float64)# 10 initial points
        # true_response = response
        #----------------------------------------------------------------------------------------------------------
        lam = 1
        eta = 1
        ## Initialize
        #----------------------------------------------------------------------------------------------------------
        train_actions = sample_actions(N=10, d=18, low=-1, high=1, active_idx=(0,17), seed=10086+tri)
        train_response = []
        for i in range(train_actions.shape[0]):
            env.reset(input_filename, seed=seed)
            train_response.append(env.step_tsai_wu(train_actions[i]))
        train_actions_tensor = torch.tensor(train_actions)
        train_response_tensor = torch.tensor(train_response)


        
        c_model, c_model_mll = initialize_c_model(actions=train_actions_tensor, response=train_response_tensor, device=device)
        fit_gpytorch_mll(c_model_mll)
        c_model.eval()
        
        c_X = train_actions_tensor.clone().to(device)
        c_Y = train_response_tensor.clone().to(device)
        

       #----------------------------------------------------------------------------------------------------------
       
        initial_num_points = 50
        random_index = torch.randint(0, search_space.shape[0], (initial_num_points,))
        eps_list = [1] * initial_num_points
        actions = search_space[random_index].to(device)
        
        f_X = actions.clone().to(device)
        
        init_train_response = []
        for i in range(f_X.shape[0]):
            env.reset(input_filename, seed=seed)
            response, true_response, num_of_queries, c_val = env.step_surrogate(action=f_X[i,:], device=device, eps=1)
            init_train_response.append(response)
        f_Y = torch.tensor(init_train_response).reshape(-1,1).to(device)
        
        
        Weighted_Phi = torch.zeros((f_X.shape[0], 2 * M_target)).to(actions) #t,2048
        Unweighted_Phi = torch.zeros((f_X.shape[0], 2 * M_target)).to(actions)
        

        
        model_ei, kernel = initialize_model(actions=f_X, response=f_Y, M_target=M_target, device=device)

        model_ei = fit_rff_fixednoise_gp(model_ei)
        
        # 例子：在 441 点上算
        N = search_space.shape[0]

        t = max(1, len(response))  # 或你用的迭代计数
        
        for i in range(f_X.shape[0]):
            x = f_X[i,:].reshape(1,-1)
            features = model_ei.covar_module.base_kernel.get_features(x, 18, normalize=True) #1,400
            # features = features / math.sqrt(2 * M_target)
            Unweighted_Phi[i, :] = features
            weighted_features = features
            Weighted_Phi[i, :] = weighted_features
        
        V_t = model_ei.covar_module.outputscale / model_ei.covar_module.outputscale * torch.matmul(Weighted_Phi.T, Weighted_Phi) + lam * torch.eye(2 * M_target).to(actions)

        iterations = 1
        n_iter = 30001
        ts = torch.arange(1, n_iter)
        beta_t = (1 + 1*np.sqrt(np.log(ts) ** 2))**2
        
        _ = env.reset(input_filename, seed=seed)
        
        num_of_queries_num = num_of_queries.item()
        
        while num_of_queries_num < n_iter:
            model_ei.eval()
            iterations += 1
            torch.cuda.empty_cache()
            #----------------------------------MANUAL UCB gram update----------------------------------
            # if iterations >= 100000:
            #     seed = 10086
            #     search_space = sample_sobol_on_grid(n=441, seed=seed, device=device) #subset 
            
            
            #----------------------------------Skip search space---------------------------------------
            # ===== 1) 目标UCB：在候选集上得到 ucb_f =====
            ucb_f, mean_f, sig_f, var_f = W_GP_UCB_scores(
                model=model_ei,
                x_D=search_space,
                response=f_Y,
                V_t=V_t,
                Unweighted_Phi=Unweighted_Phi,
                eps_list=eps_list,
                beta=beta_t[len(response)-1],
                lam=lam
            )

            # ===== 2) 约束的 PoF =====
            beta_c = 3.0  # 这里 beta_c 不再用于筛 safe，只是你仍需要 mu/sigma
            mu_c, sig_c, lcb_c, ucb_c = compute_lcb_c_botorch(c_model, search_space, beta_c)
            pof = compute_pof_from_posterior(mu_c, sig_c)  # [N] in [0,1]

            # ===== 3) 合并：UCB * PoF =====
            # 可选：用一个很小的 gate 防止 pof≈0 的点被选（不加也行）
            pof_min = 1e-6
            acq = ucb_f * pof.clamp_min(pof_min)

            idx = torch.argmax(acq)
            new_actions = search_space[idx].reshape(1, -1)

            # ===== 4) eps 用 var_f 对应 idx =====
            var = var_f[idx].item()
            eps = (math.sqrt(var) / math.sqrt(lam))

            new_response, true_responses, num_oracle_queries, c_val = env.step_surrogate(new_actions, eps*eta, device)
            
            num_of_queries_num += num_oracle_queries.item()
            # c_val 要是标量，变成 shape [1,1] 或 [1]
            c_X = torch.cat([c_X, new_actions.to(c_X)], dim=0)
            c_Y = torch.cat([c_Y, torch.tensor([[float(c_val)]], device=device, dtype=torch.float64)], dim=0)

            c_model, c_mll = initialize_c_model(actions=c_X, response=c_Y, device=device)
            fit_gpytorch_mll(c_mll)
            c_model.eval()
                        
            eps_list.append(eps)
            num_of_queries_num += num_oracle_queries.item()
            
            actions = torch.cat([actions, new_actions.to(torch.float64).to(device)]) #input
            response = torch.cat([response, new_response.to(torch.float64).to(device)]) #output
            true_response = torch.cat([true_response, true_responses.to(true_response)])
            f_Y = torch.cat([f_Y, new_response.to(torch.float64).to(device)]) 
            x_max_feature = model_ei.covar_module.base_kernel.get_features(new_actions, 18, normalize=True)
            weighted_feature = x_max_feature * (1 / eps_list[-1])
            V_t = V_t + torch.matmul(weighted_feature.T, weighted_feature)
            Unweighted_Phi = torch.cat([Unweighted_Phi, x_max_feature])
            
            
            _ = env.reset(input_filename, seed=seed)
            
            num_of_queries = torch.cat([num_of_queries, num_oracle_queries.to(device)])

            print('Stage {0} y: {1} f(x): {2} eps: {3} PoF>0.9: {4}/{5}'.format(iterations, round(new_response[-1].item(), 3), 
                                                                          round(true_responses[-1].item(), 3), 
                                                                            round(eps_list[-1], 3),
                                                                            int((pof > 0.9).sum().item()), 
                                                                            N))
            save_data(actions, response, true_response, num_of_queries, eps_list, log_dir + str(obs_noise) + 'quan_training_data_' + str(tri)+'_.pth')
            _ = env.reset(input_filename, seed=seed)
            
