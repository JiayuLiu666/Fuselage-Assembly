from datetime import datetime
import os
from os import path
from gpytorch.kernels import MaternKernel, RFFKernel, ScaleKernel, LinearKernel, RBFKernel
from ansys.mapdl.core import launch_mapdl, launcher, Mapdl
from ansys.mapdl import reader as mapdl_reader
from shutil import copyfile
import torch
from botorch.models.transforms.input import Normalize
from bo_env_constraints import ClassicFuselageEnv
from botorch.models import FixedNoiseGP
from utils import build_train_gp_with_rff, sample_actions_variable_active
from botorch.models.transforms import Normalize, Standardize
from botorch.generation.gen import gen_candidates_scipy, gen_candidates_torch, TGenCandidates
import botorch
import numpy as np
import warnings
import argparse
import random
import gpytorch
from botorch.optim import optimize_acqf
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize

from botorch.acquisition.analytic import (
    WeightGramUpperConfidenceBound,
    ExpectedImprovement,
    NoisyExpectedImprovement,
    UpperConfidenceBound,
    GramUpperConfidenceBound,
    GramExpectedImprovement
)

warnings.filterwarnings("ignore")

def get_fixed_features_from_z(z: torch.Tensor, threshold: float = 1e-3):
    if not isinstance(z, torch.Tensor):
        raise TypeError("z must be a torch.Tensor")

    z = z.detach().cpu().view(-1)
    idx = torch.where(torch.abs(z) < threshold)[0].tolist()
    d = z.numel()

    # If you'd fix ALL dims to 0, BO can't move. So instead "fix" nothing by setting to 1.
    if len(idx) == d:
        return {i: 1 for i in range(d)}  # your requested "all one" behavior

    return {int(i): 0 for i in idx}

def nnz_count(F, tol=1e-2):
    return int(np.sum(np.abs(F) > tol))


def soft_threshold_np(v, kappa):
    return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)

def nnz_count_np(F, tol=1e-2):
    return int(np.sum(np.abs(F) > tol))

def project_box(v, lower, upper):
    """
    Π_C(v) for box C = {F: lower <= F <= upper}
    """
    return np.minimum(np.maximum(v, lower), upper)


def penalty_scaled_dual(force, z, u, rho):
    """
    penalty(x) = rho/2 * || x - z + u ||^2   (scaled dual u)
    X: [n,d]
    z,u: [d]
    returns: [n]
    """
    diff = force - (z - u)           # because x - z + u = x - (z - u)
    return 0.5 * rho * (diff.pow(2).sum(dim=-1))


def get_fixed_features_from_z(z: torch.Tensor, threshold: float = 1e-3):
    z = z.detach().view(-1)
    idx = torch.where(torch.abs(z) < threshold)[0].tolist()
    d = z.numel()

    # if you'd fix everything, don't fix anything
    if len(idx) == d:
        return None

    return {int(i): 0 for i in idx}


def algorithm2_search_lambda_admmbo(
    env, input_filename, seed,
    X_init, y_init,                 # initial data for OPT
    M,                              # target nnz
    outputscale,
    ls_init,
    rho=0.5,
    e3=1e-4,
    e4=1e-2,
    alpha_opt=100,
    max_admm_iter=50,
    max_bs_iter=30,
    zero_tol=1e-3,
    lam_min=0.0,
    lam_max=0.5,                   # choose a starting guess; will expand if needed
    bounds_low=-0.5,
    bounds_high=0.5,
    LN=1e2,
    dtype=torch.float32,
    device=None,
    verbose=False,
):
    """
    Algorithm 2 implemented with ADMM-BO as Algorithm 1 subroutine.
    Returns: lam_best, F_best, (z_best,u_best), info
    """

    if device is None:
        device = X_init.device

    def solve(lam, action, response, ls_init, outputscale, LN=None, z0=None, u0=None):
        F, (Xall, yall) = admmbo_solve_F_for_lambda(
            env=env, input_filename=input_filename, seed=seed,
            X=action, y=response,
            outputscale=outputscale,
            ls_init=ls_init,
            lam=lam, rho=rho, e3=e3, e4=e4,
            alpha_opt=alpha_opt,
            max_admm_iter=max_admm_iter,
            bounds_low=bounds_low, bounds_high=bounds_high,
            dtype=dtype, device=device, z0=z0, u0=u0, LN=LN,
            verbose=False,
        )
        # hard-zero small entries for nnz counting
        Fz = np.where(np.abs(F) > zero_tol, F, 0.0)
        nnz = nnz_count_np(Fz, tol=zero_tol)
        
        return Fz, nnz, Xall, yall

    # # ---- evaluate at lam_min ----
    z0 = torch.zeros(d, device=device, dtype=dtype)
    u0 = torch.zeros(d, device=device, dtype=dtype)
    
    F, nnz0, Xall, yall = solve(lam_min, X_init, y_init, ls_init=ls_init, outputscale=outputscale, z0=z0, u0=u0)
    print('force', np.array2string(F, separator=", "))
    if verbose:
        print(f"[init] lam_min={lam_min:.3e} nnz={nnz0}")

        
    # # ---- binary search ----
    force_list = [F]
    lo, hi = lam_min, lam_max
    for it in range(max_bs_iter):
        lam = 0.5 * (lo + hi)
        F, nnz, Xall, yall = solve(lam, X_init, y_init, ls_init=ls_init, outputscale=outputscale)


        if verbose:
            print(f"[bs {it:02d}] lam={lam:.6e} nnz={nnz}")

        print('force', np.array2string(F, separator=", "))
        force_list.append(F)
        
        if nnz == M:
            return force_list, F, (Xall, yall)

        # Algorithm 2 decision:
        if nnz < M:
            hi = lam     # too sparse => λ too big
        else:
            lo = lam     # too dense  => λ too small
            
    return force_list, F, (Xall, yall)
    
def admmbo_solve_F_for_lambda(
    env, input_filename, seed,
    X, y,                 # torch: [n0,d], [n0] or [n0,1]
    lam,
    outputscale,
    ls_init,
    rho=1.0,
    e3=1e-4,
    e4=1e-2,
    alpha_opt=1000,                   # OPT inner steps each ADMM iteration
    max_admm_iter=50,
    bounds_low=-0.5,
    bounds_high=0.5,
    tol=1e-2,
    z0=None, u0=None,  
    LN=None,  # warm start (numpy arrays length d)
    dtype=torch.float32,
    device=None,
    verbose=False,
):
    """
    ADMM-BO for fixed lambda:
      F-update: OPT (Algorithm 3.2) minimizing f(x)+ rho/2||x - z + u||^2
      z-update: soft_threshold(F+u, lam/rho)
      u-update: u += F - z

    Returns: F_np, z_np, u_np, history, (X_all, y_all)
    """
    if device is None:
        device = X.device

    # keep expanding dataset inside OPT; initialize from given data
    X = X.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype).view(-1, 1)

    d = X.shape[1]
    m = d
    bounds = torch.stack([
        torch.full((d,), float(bounds_low), device=device, dtype=dtype),
        torch.full((d,), float(bounds_high), device=device, dtype=dtype),
    ])

    # ADMM state (single vector, not list)
    if z0 is None:
        z = torch.ones(d, device=device, dtype=dtype) * 1e-6
    else:
        z = torch.as_tensor(z0, device=device, dtype=dtype).view(-1)

    if u0 is None:
        u = torch.ones(d, device=device, dtype=dtype) * 1e-6
    else:
        u = torch.as_tensor(u0, device=device, dtype=dtype).view(-1)

    kappa = lam / rho

    hist = {"r_norm": [], "s_norm": [], "e1": [], "e2": []}

    def f_blackbox(x_t, eps=1e-3):
        env.reset(input_filename, seed=seed)
        return env.step_surrogate(x_t, eps)

    if isinstance(z, torch.Tensor):
        z_prev_np = z.detach().cpu().numpy().copy()
    else:
        z_prev_np = np.asarray(z, dtype=float).copy()

    for k in range(max_admm_iter):
        # ---- F step via OPT (Algorithm 3.2) ----
        F_t, y_new = opt_algorithm_3_2_botorch(
            force_init=X,
            res_init=y,
            f_blackbox=f_blackbox,
            z=z,
            u=u,
            bounds=bounds,
            outputscale=outputscale,
            ls_init=ls_init,
            rho=rho,
            alpha=alpha_opt,
            dtype=dtype,
            device=device,
            verbose=False,
        )
        
        X = torch.cat([X, F_t.view(1, -1)], dim=0)
        y = torch.cat([y, y_new.view(1, -1)], dim=0)
        
        
        F_np = F_t.detach().cpu().numpy().reshape(-1)
        
        F_np[np.abs(F_np) < tol] = 0.0
        # ---- z,u updates (numpy) ----

        z_prev = z_prev_np.copy()
        u_np = u.detach().cpu().numpy().reshape(-1)

        z_np = soft_threshold_np(F_np + u_np, kappa)

        
        print(
        f"lam={lam:.3g}, kappa={kappa:.3g}, "
        f"nnz(F)={np.sum(np.abs(F_np)>1e-3)}, nnz(z)={np.sum(np.abs(z_np)>1e-3)}, "
        f"||F-z||={np.linalg.norm(F_np-z_np):.3e}"
        )
        
        u_np = u_np + (F_np - z_np)


        # ---- residuals ----
        r = F_np - z_np
        s = rho * (z_np - z_prev)

        r_norm = np.linalg.norm(r, 2)
        s_norm = np.linalg.norm(s, 2)

        e1 = np.sqrt(m) * e3 + e4 * max(np.linalg.norm(z_np, 2), np.linalg.norm(F_np, 2))
        e2 = np.sqrt(m) * e3 + e4 * np.linalg.norm(rho * u_np, 2)

        hist["r_norm"].append(r_norm)
        hist["s_norm"].append(s_norm)
        hist["e1"].append(e1)
        hist["e2"].append(e2)

        if verbose:
            print(f"  [ADMM-BO k={k:02d}] r={r_norm:.3e} (<= {e1:.3e}) | s={s_norm:.3e} (<= {e2:.3e})")

        # write back
        z = torch.tensor(z_np, device=device, dtype=dtype)
        u = torch.tensor(u_np, device=device, dtype=dtype)
        z_prev_np = z_np.copy()

        if (r_norm <= e1) and (s_norm <= e2):
            break

    return F_np, (X, y)

def opt_algorithm_3_2_botorch(
    force_init, 
    res_init,
    f_blackbox,         # callable: takes x (torch [d]) and returns scalar (float or 0-d tensor)
    z, 
    u,
    bounds,
    outputscale,
    ls_init,
    rho=1.0,
    alpha=20,
    M_target=1024,
    dtype=torch.float32,
    device=None,
    verbose=False,
):
    """
    BoTorch version of Algorithm 3.2 OPT.

    Inputs:
      X_init:      [n, d] torch or numpy
      y_init:      [n] or [n,1]
      B_candidates:[Ncand, d] candidate set B
      f_blackbox:  function(x)->float (black-box, noisy ok)
      z_list:      [N, d]  (ADMM z_i)
      mae_list:      [N, d]  (ADMM dual y_i, UNscaled)
      rho:         scalar
      alpha:       number of OPT iterations

    Outputs:
      x_min:       [d] torch tensor
      (X_all,y_all): final dataset after alpha additions
      info: dict
    """
    if device is None:
        device = force_init.device if torch.is_tensor(force_init) else torch.device("cpu")

    # ---- make tensors and model ----
    force = torch.as_tensor(force_init, dtype=dtype, device=device)
    res_init = torch.as_tensor(res_init, dtype=dtype, device=device).view(-1, 1)
    penalty = penalty_scaled_dual(force, z, u, rho).view(-1, 1)  #0-0.5
    train_Y = res_init + penalty
    
    yvar = torch.full_like(res_init, 1e-6, device=device)
    kernel_kwargs = {"nu": 2.5, "ard_num_dims": force.shape[-1]}
    ls_init = ls_init.to(device)
    base_kernel = RFFKernel(**kernel_kwargs, num_samples=M_target)
    covar_module = ScaleKernel(base_kernel).to(device)
    model = FixedNoiseGP(
    force, train_Y, yvar,
    covar_module=covar_module).to(device)
    
    # model.train()
    # mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # with gpytorch.settings.cholesky_jitter(1e-4):
    #     fit_gpytorch_mll(mll)

    # model.eval()
    
    with torch.no_grad():
        model.covar_module.base_kernel.lengthscale.copy_(ls_init) 
        model.covar_module.outputscale.copy_(outputscale.to(device))
    model.eval()
    
    Phi = torch.zeros((force.shape[0], 2 * M_target)).to(force)
    
    for i in range(force.shape[0]):
        x = force[i,:].reshape(1,-1)
        features = model.covar_module.base_kernel.get_features(x, 18, normalize=True) #1,400
        # features = features / math.sqrt(2 * M_target)
        Phi[i, :] = features
    lamb = 1

    V_t = model.covar_module.outputscale * torch.matmul(Phi[0:1,:].T, Phi[0:1,:]) + lamb * torch.eye(2 * M_target).to(force)


    if isinstance(z, torch.Tensor):
        z = z.to(device=device, dtype=dtype).view(-1)          # [d]
    else:
        z = torch.tensor(z, dtype=dtype).to(device)
        
    if isinstance(u, torch.Tensor):
        u = u.to(device=device, dtype=dtype).view(-1)          # [d]
    else:
        u = torch.tensor(u, dtype=dtype).to(device)
    # Algorithm line 3: for t=1..alpha
    
    beta_t =  np.sqrt(2)
    
    for t in range(1, alpha + 1):
        # line 4: build u_targets on current data points
        
        penalty_train = penalty_scaled_dual(force, z, u, rho).view(-1, 1)  #0-0.5

        y_gp = res_init + 1.0 * penalty_train

        best_f = y_gp.min()     
        acq_func = GramExpectedImprovement(model=model, best_f=best_f, response=y_gp, lamb=lamb, V_t=V_t, Phi=Phi, maximize=False)
        
        # acq_func = GramUpperConfidenceBound(model=model, \
        #                                                 response=y_gp, lamb=lamb, \
        #                                                 beta=beta_t, \
        #                                                 V_t=V_t, Phi=Phi, maximize=False)
        
        fixed_features = get_fixed_features_from_z(z, threshold=1e-3)
        
        candidates, mean, _ , _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,
            fixed_features=fixed_features,
            num_restarts=5,
            raw_samples=1500, 
            timeout_sec=5
        )

        eps = 1e-3
        # line 7: evaluate true f(x^t) and append to F
        y_t,_,_ = f_blackbox(candidates, eps)  # float or tensor 18,
        y_t = torch.as_tensor(y_t, dtype=dtype, device=device).view(1, 1)

        force = torch.cat([force, candidates.view(1, -1)], dim=0)
        res_init = torch.cat([res_init, y_t], dim=0)
        
        x_max_feature = model.covar_module.base_kernel.get_features(candidates, 18, normalize=True)
        V_t = V_t + torch.matmul(x_max_feature.T, x_max_feature)
        Phi = torch.cat([Phi, x_max_feature])
        
        if verbose:
            print(f"[OPT t={t}/{alpha}] EI={float(mean.item()):.3e}  f(x_t)={float(y_t):.6g}")

    # line 10: choose x_min among sampled points under augmented objective
    pen_all = penalty_scaled_dual(force, z, u, rho).view(-1, 1)
    aug_obj = res_init + 1.0 * pen_all
    best_idx = torch.argmin(aug_obj).item()
    force_min = force[best_idx]
    response_min = res_init[best_idx]

    return force_min, response_min 
    
if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    dtype = torch.float32
    botorch.settings.debug(True)
    parser = argparse.ArgumentParser(description='Example of using argparse to adjust parameters.')
    parser.add_argument('--M_features', type=int, default=1024, help='Number of features used in RFF.')
    parser.add_argument('--max_iteration', type=int, default=10000, help='Maximum iterations.')
    parser.add_argument('--obs_noise', type=float, default=0.2**2, help='noise level of observation') #0.05 and 0.1
    parser.add_argument('--eta', type=float, default=1, help='precision trade-off parameters')
    parser.add_argument('--lam', type=float, default=0.5, help='lambda hyperparameters')   
    parser.add_argument('--B', type=float, default=1, help='exploit hyperparameters')
    args = parser.parse_args()
    seed = 0
    random.seed(seed)
    __file__ = 'FuselageActuators'
    folder = path.join(__file__, 'AnsysFiles', "Test") 
    file = random.choice(os.listdir(folder))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filepath = path.join(folder, file)
    original_input_filename = filepath
    print("Initial shape from", file.split('.')[0])
    ###

    log_dir = "Experiments"
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
    env_name = "Classic_EXP_ADMM"
    log_dir = log_dir + '/' + env_name + '/' + 'exp_set_' + str(0) + '/'
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    input_filename = log_dir + file
    copyfile(original_input_filename, input_filename)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    obs_noise = 0.1 ** 2

    trails = np.arange(1)
    for tri in trails:
        # print(tri)
        torch.cuda.empty_cache()
        torch.manual_seed(tri)
        np.random.seed(tri)
        n_iters = args.max_iteration
        M_target = args.M_features
        env = ClassicFuselageEnv(ip='129.161.91.97', obs_noise=obs_noise)
        shape_deviation = env.reset(input_filename, seed=seed)

        ##get data
        test_actions = sample_actions_variable_active(
            N=1000, d=18, k_range=(16, 18), seed=0, return_active_idx=False
            )
        test_response = []
        for i in range(test_actions.shape[0]):
            env.reset(input_filename, seed=seed)
            obs ,_ , _ = env.step_surrogate(test_actions[i], eps=1e-3)
            test_response.append(obs)
        test_response_tensor = torch.tensor(test_response, dtype=torch.float32).to(device)
        test_actions_tensor = torch.tensor(test_actions).to(device)
        
        _, lengthscale, outputscale = build_train_gp_with_rff(actions = test_actions_tensor, response=test_response_tensor, test_actions=test_actions_tensor, test_response=test_response_tensor)
        env.reset(input_filename, seed=seed)
        
        # outputscale=None
        # lengthscale=None
        #### 
        d = test_actions_tensor.shape[1]
        z0_init = torch.zeros(d, device=device, dtype=dtype)
        u0_init = torch.zeros(d, device=device, dtype=dtype)
        
        LN = 1e2
        rho = 1
        M = 10                 # target nonzeros
        lam_min = 0.0
        lam_max = 0.5 # starting guess; will expand automatically

        Force_list, F, (Xall, yall) = algorithm2_search_lambda_admmbo(
            env=env,
            input_filename=input_filename,
            seed=seed,
            X_init=test_actions_tensor,       # [n0,18]
            y_init=test_response_tensor,      # [n0]
            M=M,
            outputscale=outputscale,
            ls_init=lengthscale,
            rho=rho,
            alpha_opt=300,
            e3=1e-4,
            e4=1e-2,
            max_admm_iter=30,
            max_bs_iter=30,
            zero_tol=1e-3,
            lam_min=lam_min,
            lam_max=lam_max,
            bounds_low=-0.5,
            bounds_high=0.5,
            LN=LN,
            verbose=True,
        )
        
        log_dir = "Experiments_constraints"
        if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
        env_name = "Classic_ADMMBO_EXP_10"
        log_dir = log_dir + '/' + env_name + '/' + 'exp_set_' + str(i) + '/'
        if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        torch.save({
            'actions': Xall,
            'response': yall
        }, log_dir + 'training_data_' + str(tri) + '_.pth')
        



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
