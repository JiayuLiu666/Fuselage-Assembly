import argparse
import os
from os import path
from shutil import copyfile
import torch
from quantum_bo_env_constraint import QuantumFuselageEnv
import botorch.settings as botorch_settings
from botorch.models import FixedNoiseGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RFFKernel, ScaleKernel
import warnings
warnings.filterwarnings("ignore")
import numpy as np

import math

NOISE_FREE_VARIANCE = 1e-6

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


def stage_epsilon(var_value, lam, eps_max):
    """epsilon_t = min(epsilon_max, sqrt(var(x) / lambda))."""
    return min(float(eps_max), math.sqrt(max(float(var_value), 1e-12) / float(lam)))

def save_data(
    actions,
    response,
    true_response,
    num_of_queries,
    eps,
    file_path,
    active_records=None,
    metadata=None,
):
    # print("Saving data to:", file_path)  # Debug: Print the file path
    payload = {
        'actions': actions,
        'response': response,
        'queries': num_of_queries,
        'uncertainty': eps,
        'true_response': true_response,
    }
    if active_records is not None:
        payload['active_records'] = active_records
    if metadata is not None:
        payload['metadata'] = metadata
    torch.save(payload, file_path)
    

def initialize_c_model(actions, response, device, ls_init=None, outputscale=None):
    yvar = torch.full_like(response, NOISE_FREE_VARIANCE, device=device)
    
    model = FixedNoiseGP(
    actions, response, yvar).to(device)
    # Apply manual hyperparameters if provided
    if ls_init is not None or outputscale is not None:
        with torch.no_grad():
            if ls_init is not None:
                model.covar_module.base_kernel.lengthscale.copy_(ls_init.to(device))
            if outputscale is not None:
                model.covar_module.outputscale.copy_(outputscale.to(device))
        model.eval()
        return model, None  # no MLL needed when using manual hyperparams
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll

def initialize_f_model(actions, response, device, M_target,
                     obs_noise, ls_init=None):
    noise_tensor = torch.as_tensor(obs_noise, dtype=response.dtype, device=device)
    if noise_tensor.ndim == 0:
        yvar = torch.full_like(response, float(noise_tensor.item()), device=device)
    else:
        yvar = noise_tensor.reshape(-1, 1) if noise_tensor.ndim == 1 else noise_tensor
        yvar = yvar.expand_as(response).clone()
    kernel_kwargs = {"nu": 2.5, "ard_num_dims": actions.shape[-1]}
    
    base_kernel = RFFKernel(**kernel_kwargs, num_samples=M_target)
    covar_module = ScaleKernel(base_kernel).to(device)

    model = FixedNoiseGP(actions, response, yvar, covar_module=covar_module,).to(device)
    with torch.no_grad():
        if ls_init is not None:
            model.covar_module.base_kernel.lengthscale.copy_(ls_init.to(device))
        model.covar_module.outputscale.copy_(
            torch.tensor(1.0, dtype=torch.double, device=device)
        )
        
    model.eval()
    return model, model.covar_module


def sample_sobol_on_grid(grid_size=21, seed=None, device=None): # 21*21 = 441
    device = device or torch.device("cpu")

    D = 18

    var_dims = [0, 17]

    values = torch.linspace(-1, 1, grid_size, device=device, dtype=torch.float64)

    pairs = torch.cartesian_prod(values, values)

    X = torch.zeros(pairs.shape[0], D, device=device, dtype=torch.float64)

    X[:, var_dims] = pairs

    return X


def format_safe_initial_points(search_space, safety_margin, indices, active_dims=(0, 17)):
    lines = []
    for rank, idx in enumerate(indices.tolist(), start=1):
        a0 = float(search_space[idx, active_dims[0]].item())
        a1 = float(search_space[idx, active_dims[1]].item())
        margin = float(safety_margin[idx].item())
        lines.append(
            f"  {rank:02d}. idx={idx:03d}, dim{active_dims[0]}={a0:+.2f}, "
            f"dim{active_dims[1]}={a1:+.2f}, 1-FI={margin:+.4f}"
        )
    return lines

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


def evaluate_initial_points(env, actions, device, obs_precision: float = 0.04):
    """Initial objective evaluations using fixed observation precision."""
    if isinstance(actions, torch.Tensor):
        actions_np = actions.detach().cpu().numpy()
    else:
        actions_np = np.asarray(actions)

    actions_np = actions_np.reshape(-1, 18).astype(np.float64)
    response_list = []
    true_response_list = []
    query_list = []
    variance_list = []

    for action_np in actions_np:
        env.forces = np.zeros(18, dtype=np.float64)
        y_obs, y_true, n_q, _, empirical_variance = env.step_surrogate(
            torch.tensor(action_np, dtype=torch.float64, device=device).reshape(1, -1),
            eps=obs_precision,
            device=device,
        )
        response_list.append(y_obs.reshape(1, 1))
        true_response_list.append(y_true.reshape(1, 1))
        query_list.append(n_q.reshape(1, 1))
        variance_list.append(empirical_variance.reshape(1, 1))

    response_tensor = torch.cat(response_list, dim=0)
    true_response_tensor = torch.cat(true_response_list, dim=0)
    query_tensor = torch.cat(query_list, dim=0)
    variance_tensor = torch.cat(variance_list, dim=0).clamp_min(NOISE_FREE_VARIANCE)
    return response_tensor, true_response_tensor, query_tensor, variance_tensor

def W_GP_UCB_scores(model, x_D, response, V_t, Unweighted_Phi, eps_list, beta, lam, chunk_size=1024):
    eps_list = torch.as_tensor(eps_list, device=V_t.device, dtype=V_t.dtype).view(-1, 1)
    Unweighted_Phi = Unweighted_Phi.to(device=V_t.device, dtype=V_t.dtype)
    response = response.to(device=V_t.device, dtype=V_t.dtype).reshape(-1, 1)
    x_D = x_D.to(device=V_t.device, dtype=V_t.dtype)
    beta = torch.as_tensor(beta, device=V_t.device, dtype=V_t.dtype)

    # W @ y where W = diag(1 / eps_i^2), without materializing the full diagonal matrix.
    Y_weighted = response / (eps_list ** 2)

    L = torch.linalg.cholesky(V_t)
    rhs = Unweighted_Phi.T @ Y_weighted
    nu_t = torch.cholesky_solve(rhs, L)

    mean_chunks = []
    var_chunks = []
    for start in range(0, x_D.shape[0], chunk_size):
        x_chunk = x_D[start : start + chunk_size]
        features = model.covar_module.base_kernel.get_features(x_chunk, x_chunk.shape[-1], normalize=True)
        mean_chunks.append((features @ nu_t).reshape(-1))

        # Diagonal-only variance via Cholesky solve: proj = V_t^{-1} phi(x)^T
        proj = torch.cholesky_solve(features.T, L).T
        chunk_var = (lam * (proj * features).sum(dim=1)).clamp_min(1e-12)
        var_chunks.append(chunk_var)

    mean = torch.cat(mean_chunks, dim=0)
    var = torch.cat(var_chunks, dim=0)
    sigma = var.sqrt()

    ucb = mean + beta.sqrt() * sigma
    return ucb, mean, sigma, var
    
    
if __name__ == "__main__":
    print("Quantum Discrete GP-UCB, 2 actuators")

    parser = argparse.ArgumentParser(
        description="Quantum Discrete Safe-Set cUCB.")
    parser.add_argument("--query_budget", type=int, default=20000,
                        help="Oracle query budget.")
    parser.add_argument("--eps_max", type=float, default=0.04,
                        help="Maximum per-stage epsilon before eta scaling.")
    parser.add_argument("--obs_noise", type=float, default=0.1**2,
                        help="Observation noise variance.")
    parser.add_argument("--min_shots", type=int, default=20,
                        help="Minimum shots for Monte Carlo Estimation.")
    parser.add_argument("--M_features", type=int, default=400,
                        help="Number of RFF features.")
    parser.add_argument("--B", type=float, default=3.0,
                        help="Exploration coefficient B in beta_t.")
    parser.add_argument("--obj_ls", type=float, default=0.2,
                        help="Objective lengthscale for active dimensions.")
    parser.add_argument("--lam0", type=float, default=0.5,
                        help="Initial lambda weight for boundary expansion.")
    parser.add_argument("--t0", type=float, default=10.0,
                        help="Decay timescale for lambda schedule.")
    parser.add_argument("--lam_p", type=float, default=2,
                        help="Decay power for lambda schedule.")
    parser.add_argument("--init_num_points", type=int, default=5,
                        help="Number of safe warmup points for both objective and constraint models.")
    args = parser.parse_args()

    query_budget = args.query_budget
    init_num_points = args.init_num_points
    Big_B = args.B
    min_shots = args.min_shots
    obs_noise = args.obs_noise
    M_target = args.M_features
    print('noise_level: ', obs_noise)
    print('eps_max: ', args.eps_max)
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
            
    env_name = "Quantum_Discrete_cUCB"
    log_dir = log_dir + '/' + env_name + '/' + 'exp_set_1' + '/' # 0: real Machine ; 1: Simulator
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    input_filename = log_dir + file
    copyfile(original_input_filename, input_filename)

    trials = np.arange(5)
    for tri in trials:
        seed = int(tri)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_default_dtype(torch.float64)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        botorch_settings.debug._set_state(True)
        env = QuantumFuselageEnv(obs_noise=obs_noise, min_shots=min_shots) 
        _ = env.reset(input_filename, seed=seed)   #tensor. 1,1

        search_space = sample_sobol_on_grid(grid_size=21, seed=seed, device=device)
        
        #----------------------------------------------------------------------------------------------------------
        # actions = torch.zeros(1, 18, device=device, dtype=torch.float64)# 10 initial points
        # true_response = response
        #----------------------------------------------------------------------------------------------------------
        lam = 1
        eta = 1
        ## Initialize — noise-free warmup
        #----------------------------------------------------------------------------------------------------------
        # Select warmup points from discrete grid — only safe points (1 - FI >= 0)
        grid_np = search_space.detach().cpu().numpy().reshape(-1, 18)
        FI_all = env.tsai_wu.predict(grid_np, return_std=False)
        safety_margin = torch.tensor(1.0 - FI_all, dtype=torch.float64, device=device)
        safe_idx = torch.where(safety_margin >= 0.0)[0]
        print(f"[Init] {len(safe_idx)}/{search_space.shape[0]} grid points are safe")

        init_pool_size = init_num_points

        # Seed-dependent random init from safe grid points (different per trial)
        safe_margins = safety_margin[safe_idx]
        weights = safe_margins.clamp_min(0.0).cpu().numpy().astype(float)
        weights /= weights.sum() + 1e-12
        rng = np.random.RandomState(seed)
        chosen_safe = rng.choice(safe_idx.cpu().numpy(), size=min(init_pool_size, len(safe_idx)),
                                 replace=False, p=weights)
        init_idx = torch.tensor(chosen_safe, dtype=torch.long, device=device)

        print(f"[Init] Selected {init_idx.numel()} safe warmup point(s) (seed={seed}):")
        for line in format_safe_initial_points(search_space, safety_margin, init_idx):
            print(line)

        chosen = init_idx[:init_num_points]
        actions = search_space[chosen].to(device)
        f_X = actions.clone()

        # Initial objective evaluation with fixed observation precision.
        env.reset(input_filename, seed=seed)
        f_Y, true_response, init_queries, init_obs_var = evaluate_initial_points(
            env, f_X, device=device, obs_precision=0.04
        )
        response = f_Y.clone()

        # Noise-free constraint evaluation — sample from safe set only
        c_chosen = init_idx[:init_num_points]
        c_train_tensor = search_space[c_chosen].to(device)
        c_train_actions_np = c_train_tensor.detach().cpu().numpy().reshape(-1, 18)
        FI_batch = env.tsai_wu.predict(c_train_actions_np, return_std=False)
        constraint_response = torch.tensor(
            (1 - FI_batch), dtype=torch.float64, device=device).reshape(-1, 1)

        # Manual constraint GP hyperparameters (validated by smoke_test_constraint_ls.py)
        # Constraint is smoother than objective → needs longer LS (0.50 vs 0.10)
        c_ls = torch.full((1, 18), 0.6931, dtype=torch.float64, device=device)
        c_ls[0, 0]  = 0.50   # active dim
        c_ls[0, 17] = 0.50   # active dim
        c_os = torch.tensor(1.0, dtype=torch.float64, device=device)

        # Fit constraint GP with manual hyperparameters (no MLL optimization)
        c_X = c_train_tensor.clone()
        c_Y = constraint_response.clone()
        c_model, _ = initialize_c_model(
            actions=c_X, response=c_Y, device=device,
            ls_init=c_ls, outputscale=c_os)
        c_model.eval()
        print(f"[Constraint GP] Manual LS: active=0.50, inactive=0.6931, "
              f"fitted on {c_X.shape[0]} points.")

        # Focused exact sweep winner: M=400, B=3.0, active_LS=0.20,
        # lam0=0.5, t0=10, lam_p=2.0.
        lengthscale = torch.full((1, 18), 0.6931, dtype=torch.float64, device=device)
        lengthscale[0, 0]  = args.obj_ls   # active dim
        lengthscale[0, 17] = args.obj_ls   # active dim
        outputscale = torch.tensor(1.0, dtype=torch.float64, device=device)
        print(f"[Manual LS] active={lengthscale[0,0].item():.2f}, "
              f"inactive={lengthscale[0,1].item():.4f}, "
              f"outputscale={outputscale.item():.2f}")

        # Initialise objective GP kernel with manual hyperparameters.
        # Use a single warmup point just to satisfy FixedNoiseGP constructor.
        init_X = f_X[:init_num_points]
        init_Y = f_Y[:init_num_points]
        init_yvar = init_obs_var[:init_num_points].clone()
        f_model, _ = initialize_f_model(
            actions=init_X, response=init_Y, M_target=M_target, device=device,
            obs_noise=init_yvar, ls_init=lengthscale)
        f_model.eval()

        # Seed objective GP with first 2 warmup observations for initial posterior structure
        n_seed = init_num_points
        f_X = f_X[:n_seed].clone()
        f_Y = f_Y[:n_seed].clone()
        actions = f_X.clone()
        response = f_Y.clone()
        true_response = true_response[:n_seed].clone()
        eps_list = [args.eps_max] * n_seed
        num_of_queries = init_queries[:n_seed].clone()
        Weighted_Phi = torch.zeros((n_seed, 2 * M_target), device=device, dtype=torch.float64)
        Unweighted_Phi = torch.zeros((n_seed, 2 * M_target), device=device, dtype=torch.float64)
        
        # 例子：在 441 点上算
        N = search_space.shape[0]

        # t = max(1, len(response))  # 或你用的迭代计数
        
        beta_c = 3.0

        mu_c, sig_c, lcb_c, ucb_c = compute_lcb_c_botorch(c_model, search_space, beta_c)
        safe_mask = lcb_c >= 0.0
        print("verified safe:", int(safe_mask.sum().item()), "/", N)

        
        for i in range(f_X.shape[0]):
            x = f_X[i,:].reshape(1,-1)
            features = f_model.covar_module.base_kernel.get_features(x, 18, normalize=True) #1,400
            Unweighted_Phi[i, :] = features
            weighted_features = features * (1.0 / args.eps_max)
            Weighted_Phi[i, :] = weighted_features
        
        V_t = 1 * torch.matmul(Weighted_Phi.T, Weighted_Phi) + lam * torch.eye(2 * M_target).to(actions)

        iterations = 1
        n_iter = query_budget + 10  # used only for beta_t scheduling
        ts = torch.arange(1, n_iter + 1, device=device, dtype=torch.float64)
        beta_t = (1.0 + Big_B * torch.log(ts).abs()).pow(2)
        _ = env.reset(input_filename, seed=seed)
        
        num_of_queries_num = 0   # oracle sample cost (for logging)

        # Separate lists for ACTIVE (post-init) queried points only.
        active_actions_list   = []
        active_response_list  = []
        active_true_resp_list = []
        active_queries_list   = []
        active_eps_list_only  = []
        active_records = []
        selected_counts = {}
        print(f"Initial total queries: {num_of_queries_num}")
        while num_of_queries_num < query_budget:
            iterations += 1
            #----------------------------------MANUAL UCB gram update----------------------------------
            # if iterations >= 100000:
            #     seed = 10086
            #     search_space = sample_sobol_on_grid(n=441, seed=seed, device=device) #subset 
            
            
            #----------------------------------Skip search space---------------------------------------
            # ===== 1) 目标UCB：在 441 点上得到一条 ucb 曲线 =====
            t_acq = max(1, len(response) + 1)
            beta = beta_t[t_acq - 1]
            ucb_f, mean_f, sig_f, var_f = W_GP_UCB_scores(
                model=f_model,
                x_D=search_space,
                response=f_Y,
                V_t=V_t,
                Unweighted_Phi=Unweighted_Phi,
                eps_list=eps_list,
                beta=beta,
                lam=lam
            )

            # ===== 2) 约束安全集：c(x) ~ GP(m_c, sig_c)，用 LCB >= 0 =====
            
            beta_c = 3.0  # 你可以 schedule
            mu_c, sig_c, lcb_c, ucb_c = compute_lcb_c_botorch(c_model, search_space, beta_c)
            safe_mask = lcb_c >= 0.0
            safe_mask_sum_before = int(safe_mask.sum().item())

            # 如果安全集为空：可以降 beta_c 或退回最安全点（ucb_c最大/ lcb最大）
            if safe_mask.sum().item() == 0:
                idx = torch.argmax(lcb_c)  # 最不危险的
                new_actions = search_space[idx].reshape(1, -1)
            else:
                # ===== 3) 边界扩张项：a_bnd = -|m_c/sig_c|（只在安全集内）=====
                eps_sigma = 1e-9
                a_bnd = -torch.abs(mu_c / (sig_c + eps_sigma))   # [N]

                # ===== 4) 归一化 + lambda_t 递减合并 =====
                ucb_n  = minmax_norm(ucb_f, mask=safe_mask)
                bnd_n  = minmax_norm(a_bnd, mask=safe_mask)

                lam_t = lambda_by_stage(int(iterations), lam0=args.lam0, t0=args.t0, p=args.lam_p)
                score = (1.0 - lam_t) * ucb_n + lam_t * bnd_n
                score[~safe_mask] = -1e18

                idx = torch.argmax(score)
                new_actions = search_space[idx].reshape(1, -1)

            # ===== 5) 按你原来的方式评估新点 =====
            # 注意：eps 用目标模型的 posterior 方差来算（你原来就是这么做）
            selected_idx = int(idx.item())
            mu_c_idx = float(mu_c[selected_idx].item())
            sig_c_idx = float(sig_c[selected_idx].item())
            lcb_c_idx = float(lcb_c[selected_idx].item())
            ucb_c_idx = float(ucb_c[selected_idx].item())
            gp_safe = bool(safe_mask[selected_idx].item())

            var = var_f[idx].item() if safe_mask.sum().item() > 0 else var_f[idx].item()
            eps = stage_epsilon(var, lam, args.eps_max)

            new_response, true_responses, num_oracle_queries, c_val, empirical_variance = env.step_surrogate(new_actions, eps*eta, device)
            c_val_scalar = float(c_val.reshape(-1)[0].item())
            fi_scalar = float(1.0 - c_val_scalar)
            actual_safe = bool(c_val_scalar >= 0.0)

            num_of_queries_num += int(num_oracle_queries.item())


            # Incremental constraint GP update: use fantasy model
            # (fixed hyperparameters, O(1) update instead of O(n³) refit)
            with torch.no_grad():
                c_fantasy_noise = torch.full((1,), 1e-6, device=device, dtype=torch.float64)
                c_model = c_model.get_fantasy_model(
                    new_actions,
                    torch.tensor([[float(c_val)]], device=device, dtype=torch.float64).view(-1),
                    noise=c_fantasy_noise,
                )
                c_model.eval()

            _, _, lcb_c_next, _ = compute_lcb_c_botorch(
                c_model, search_space, beta_c
            )
            safe_mask_next = lcb_c_next >= 0.0
            safe_mask_sum_after = int(safe_mask_next.sum().item())
            safe_set_expanded = bool(safe_mask_sum_after > safe_mask_sum_before)

            eps_list.append(eps)
            
            actions = torch.cat([actions, new_actions.to(torch.float64).to(device)]) #input
            response = torch.cat([response, new_response.to(torch.float64).to(device)]) #output
            true_response = torch.cat([true_response, true_responses.to(true_response)])
            f_Y = torch.cat([f_Y, new_response.to(torch.float64).to(device)]) 
            x_max_feature = f_model.covar_module.base_kernel.get_features(new_actions, 18, normalize=True)
            weighted_feature = x_max_feature * (1 / eps_list[-1])
            V_t = V_t + torch.matmul(weighted_feature.T, weighted_feature)
            Unweighted_Phi = torch.cat([Unweighted_Phi, x_max_feature])
            
            
            _ = env.reset(input_filename, seed=seed)
            
            num_of_queries = torch.cat([num_of_queries, num_oracle_queries.to(device)])

            # Accumulate active-only records
            active_actions_list.append(new_actions.to(torch.float64).to(device))
            active_response_list.append(new_response.to(torch.float64).to(device))
            active_true_resp_list.append(true_responses.to(torch.float64).to(device))
            active_queries_list.append(num_oracle_queries.to(device))
            active_eps_list_only.append(eps)

            selection_count_for_idx = selected_counts.get(selected_idx, 0) + 1
            selected_counts[selected_idx] = selection_count_for_idx
            active_records.append(
                {
                    "trial": int(tri),
                    "seed": int(seed),
                    "iteration": int(iterations),
                    "active_step": int(len(active_records) + 1),
                    "selected_index": selected_idx,
                    "selected_idx": selected_idx,
                    "c_val": c_val_scalar,
                    "FI": fi_scalar,
                    "safe": actual_safe,
                    "unsafe": bool(not actual_safe),
                    "actual_safe": actual_safe,
                    "actual_unsafe": bool(not actual_safe),
                    "gp_safe": gp_safe,
                    "gp_unsafe": bool(not gp_safe),
                    "gp_safe_but_actual_unsafe": bool(gp_safe and (not actual_safe)),
                    "mu_c": mu_c_idx,
                    "sig_c": sig_c_idx,
                    "lcb_c": lcb_c_idx,
                    "ucb_c": ucb_c_idx,
                    "mu_c_idx": mu_c_idx,
                    "sig_c_idx": sig_c_idx,
                    "lcb_c_idx": lcb_c_idx,
                    "ucb_c_idx": ucb_c_idx,
                    "safe_mask_sum": safe_mask_sum_before,
                    "safe_mask_sum_before": safe_mask_sum_before,
                    "safe_mask_sum_after": safe_mask_sum_after,
                    "safe_set_expanded": safe_set_expanded,
                    "num_oracle_queries": int(num_oracle_queries.item()),
                    "eps": float(eps),
                    "selection_count_for_idx": int(selection_count_for_idx),
                    "repeated_sampling_count": int(selection_count_for_idx - 1),
                    "total_budget_after_step": int(num_of_queries_num),
                    "budget_overspent": bool(num_of_queries_num > query_budget),
                    "objective_observation": float(new_response.reshape(-1)[0].item()),
                    "objective_true": float(true_responses.reshape(-1)[0].item()),
                    "empirical_variance": float(empirical_variance.reshape(-1)[0].item()),
                }
            )

            num_safe = safe_mask_sum_before
            safe_rate = num_safe / N
            print('Stage {0} y: {1} f(x): {2} eps: {3} safeset rate: {4:.3f} ({5}/{6})'.format(iterations, round(new_response[-1].item(), 3), 
                                                                          round(true_responses[-1].item(), 3), 
                                                                            round(eps_list[-1], 3),
                                                                            safe_rate, num_safe, N))
            # Save ONLY the actively queried points (no init mixing)
            if iterations % 100 == 0 or num_of_queries_num >= n_iter:
                def _cat(lst):
                    return torch.cat(lst, dim=0) if lst else torch.empty(0)
                save_data(_cat(active_actions_list), _cat(active_response_list),
                          _cat(active_true_resp_list), _cat(active_queries_list),
                          active_eps_list_only,
                          log_dir + str(obs_noise) + 'quan_training_data_' + str(tri)+'_.pth',
                          active_records=active_records,
                          metadata={
                              "query_budget": int(query_budget),
                              "trial": int(tri),
                              "seed": int(seed),
                              "num_active_steps": int(len(active_records)),
                              "final_total_budget": int(num_of_queries_num),
                          })
            _ = env.reset(input_filename, seed=seed)

        def _cat(lst):
            return torch.cat(lst, dim=0) if lst else torch.empty(0)
        save_data(_cat(active_actions_list), _cat(active_response_list),
                    _cat(active_true_resp_list), _cat(active_queries_list),
                    active_eps_list_only,
                    log_dir + str(obs_noise) + 'quan_training_data_' + str(tri)+'_.pth',
                    active_records=active_records,
                    metadata={
                        "query_budget": int(query_budget),
                        "trial": int(tri),
                        "seed": int(seed),
                        "num_active_steps": int(len(active_records)),
                        "final_total_budget": int(num_of_queries_num),
                    })
        print(f"[Trial {tri}] Saved  ({int(num_of_queries_num)} oracle queries,  {len(active_actions_list)} active steps)")
