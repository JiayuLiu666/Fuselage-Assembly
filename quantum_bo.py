from joblib import dump, load
from datetime import datetime
import os
from os import path
from ansys.mapdl.core import launch_mapdl, launcher, Mapdl
from ansys.mapdl import reader as mapdl_reader
from shutil import copyfile
import torch
from quantum_bo_env import QuantumFuselageEnv
from bo_env import ClassicFuselageEnv
import botorch.settings as botorch_settings
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from botorch.models import SingleTaskGP, FixedNoiseGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import MaternKernel, RFFKernel, ScaleKernel, LinearKernel
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    WeightGramUpperConfidenceBound,
    UpperConfidenceBound,
    WeightedUpperConfidenceBound
)
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from utils import build_train_gp_with_rff, sample_actions

from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qUpperConfidenceBound
)
from botorch.sampling.normal import SobolQMCNormalSampler
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import gc
import argparse
from scipy.stats import qmc

import random


def save_model(model, filename="model_state.pth"):
    """Save the model's state dictionary to a file."""
    torch.save(model.state_dict(), filename)

def save_data(actions, true_response, response, num_of_queries, eps, error_init, file_path):
    # print("Saving data to:", file_path)  # Debug: Print the file path
    torch.save({
        'actions': actions,
        'true_response': true_response,
        'response': response,
        'queries': num_of_queries,
        'uncertainty': eps,
        'error_init': error_init,
    }, file_path)
    
def load_data_wo_constraint(filename, device):
    data = torch.load(filename)
    actions = data['actions'].to(torch.float32).to(device)
    response = data['response'].to(torch.float32).to(device)
    # num_of_queries =  data['queries'].to(torch.float32).to(device)
    return actions, response
    
def initialize_model(actions, response, device, M_target, ls_init=None, outputscale=None):
    yvar = torch.full_like(response, 1e-6, device=device)
    
    kernel_kwargs = {"nu": 2.5, "ard_num_dims": actions.shape[-1]}
    ls_init = ls_init.to(device)
    
    base_kernel = RFFKernel(**kernel_kwargs, num_samples=M_target)
    covar_module = ScaleKernel(base_kernel).to(device)
    model = FixedNoiseGP(
    actions, response, yvar,
    covar_module=covar_module).to(device)
    with torch.no_grad():
        model.covar_module.base_kernel.lengthscale.copy_(ls_init) 
        # model.covar_module.outputscale.copy_(torch.tensor(11.3646, dtype=torch.double, device=device))
        model.covar_module.outputscale.copy_(outputscale.to(device))
    model.eval()
    return model, model.covar_module

def optimize_acqf_and_get_observation(acq_func, bounds, target_func_quantum, target_func_classic=None, lam=None, gp_model=None, device=None, obs_noise=None, eta=None):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize, continuous
    candidates, posterior, V_t_inv, V_t  = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        fixed_features={4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0},
        q=1,
        num_restarts=10,
        raw_samples=1500, 
        timeout_sec=5
    )
    gp_model.eval()
    lam = lam
    features = gp_model.covar_module.base_kernel.get_features(candidates, 18, normalize=True)

    var = lam * features @ V_t_inv @ features.T
    if var.item() < 1e-12:
        var = torch.tensor([1e-12])
    eps = (torch.sqrt(var) / np.sqrt(lam).item()).item()
    
    new_response, rmse, num_oracle_queries = target_func_quantum(candidates, eps=eps*eta, device=device)

    return candidates, new_response, torch.tensor(rmse).reshape(1,1), num_oracle_queries, eps, V_t, posterior

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example of using argparse to adjust parameters.')
    parser.add_argument('--M_target', type=int, default=1024, help='Number of targers.')
    parser.add_argument('--max_iteration', type=int, default=10000, help='Maximum iterations.')
    parser.add_argument('--obs_noise', type=float, default=0.2**2, help='noise level of observation')
    parser.add_argument('--eta', type=float, default=1, help='precision trade-off parameters')
    parser.add_argument('--lam', type=float, default=0.5, help='lambda hyperparameters')
    parser.add_argument('--B', type=float, default=2, help='exploit hyperparameters')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    botorch_settings.debug._set_state(True)
    # timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    bounds = torch.tensor([[-0.5] * 18, [0.5] * 18], device=device, dtype=torch.float32)
    obs_noise =  args.obs_noise
    print('noise_level: ', obs_noise)
    eta = args.eta
    print('trade off parameters: ', eta)
    lam = args.lam
    print('lambda: ', lam)
    Big_B = args.B
    print('Big B: ', Big_B)
    ###
    
    for i in range(10):
        seed = i
        random.seed(seed)
        __file__ = 'FuselageActuators'
        folder = path.join(__file__, 'AnsysFiles', "Test") 
        file = random.choice(os.listdir(folder))
        # file = 'SolutionInputDP52.inp'
        # file = 'SolutionInputDP60.inp' for i in range(8,9): 
        # file = 'SolutionInputDP45.inp' for i in range(4,5):
        filepath = path.join(folder, file)
        original_input_filename = filepath
        print("Initial shape from", file.split('.')[0])

        log_dir = "Experiments_unconstraint_continuous"
        if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
        env_name = "Quantum_EXP_10_" + str(eta) + '_' + str(lam) + '_' + str(Big_B)
        log_dir = log_dir + '/' + env_name + '/' + 'exp_set_' + str(i) + '/'
        if not os.path.exists(log_dir):
                os.makedirs(log_dir)

        input_filename = log_dir + file
        copyfile(original_input_filename, input_filename)
        trails = np.arange(5)
        for tri in trails:
            print(tri)
            torch.cuda.empty_cache()
            torch.manual_seed(tri)
            np.random.seed(tri)
            env = QuantumFuselageEnv(ip='129.161.91.97', obs_noise=obs_noise) 
            error_init, file_2 = env.reset(input_filename, seed=seed)
            print("target shape:", file_2)
            n_iter = args.max_iteration
            M_target = args.M_target
            # Initialize random data
        #######----------------------------------train hyperparameters---------------------------------
            # env2 = ClassicFuselageEnv(ip='129.161.91.97', obs_noise=obs_noise) 
            # _ = env2.reset(input_filename, seed=seed)
            actions_tensor = sample_actions(N=1800, d=18, low=-0.5, high=0.5, active_idx=(0,1,2,3,14,15,16,17), seed=10086+tri)
            response_list = []
            for i in range(actions_tensor.shape[0]):
                env.reset(input_filename, seed=seed)
                response_list.append(env.cal_deviation(actions_tensor[i]))
            response_tensor = torch.tensor(response_list)
            test_actions = sample_actions(N=100, active_idx=(0,1,2,3,14,15,16,17), seed=0)
            test_response = []
            for i in range(test_actions.shape[0]):
                env.reset(input_filename, seed=seed)
                test_response.append(env.cal_deviation(test_actions[i]))
            test_response_tensor = torch.tensor(test_response, dtype=torch.double).to(device)
            test_actions_tensor = torch.tensor(test_actions).to(device)
            
            _, lengthscale, outputscale = build_train_gp_with_rff(actions = actions_tensor, response=response_tensor, test_actions=test_actions_tensor, test_response=test_response_tensor)
            env.reset(input_filename, seed=seed)
            #-------------------------------------
            actions = torch.zeros((n_iter+10, 18), dtype=torch.float32, device=device)
            response = torch.zeros((n_iter+10, 1), dtype=torch.float32, device=device)
            true_response = torch.zeros((n_iter+10, 1), dtype=torch.float32, device=device)
            stages = torch.zeros((n_iter+10, 1), dtype=torch.float32, device=device)
            #-------------------------------------
            initial_response = []      
            init_action = torch.zeros(1, 18, device=device, dtype=torch.float32)
            
            init_response, init_true_response, num_of_queries = env.step_surrogate(init_action, eta*1, device=device)
            num_of_queries_num = num_of_queries.item()
            #-------------------------------------
            actions[0], response[0], true_response[0], stages[0] = init_action, init_response, init_true_response, num_of_queries   

            iterations = 1
            ts = torch.arange(1, n_iter)
            
            model_ei, kernel = initialize_model(actions[0:1], response[0:1], device, M_target, ls_init=lengthscale, outputscale=outputscale)
            beta_t = (1 + Big_B * torch.sqrt(torch.log(ts.float()) ** 2)) ** 2
            eps_list = [1]
            model_ei.eval()
            
            Weighted_Phi = torch.zeros((actions.shape[0], 2 * M_target)).to(actions) #t,2048
            Unweighted_Phi = torch.zeros((actions.shape[0], 2 * M_target)).to(actions)
            for i in range(actions[0:1].shape[0]):
                x = actions[0:1][i,:].reshape(1,-1)
                features = model_ei.covar_module.base_kernel.get_features(x, 18, normalize=True) #1,400
                Unweighted_Phi[i, :] = features
                weighted_features = features * (1 / eps_list[i])
                Weighted_Phi[i, :] = weighted_features
            
            V_t = torch.matmul(Weighted_Phi[0:1,:].T, Weighted_Phi[0:1,:]) + lam * torch.eye(2 * M_target).to(actions)
            
            error_init, file_name = env.reset(input_filename, seed=seed)

            while num_of_queries_num < n_iter:
                qei = WeightGramUpperConfidenceBound(model=model_ei, \
                                                response=response[:iterations], \
                                                beta=beta_t[iterations-1], lam=lam, \
                                                eps_list = eps_list[:iterations], Unweighted_Phi=Unweighted_Phi[:iterations, :], V_t=V_t)
                    
                new_actions, new_response, true_responses, num_oracle_queries, eps, V_t, posterior = optimize_acqf_and_get_observation(acq_func=qei, bounds=bounds, gp_model=model_ei, \
                    target_func_quantum=env.step_surrogate, device=device, obs_noise=obs_noise, eta=eta, lam=lam)
                
                new_actions = new_actions.detach()
                new_response = new_response.detach()
                true_responses = true_responses.detach()
                
                eps_list.append(eps)
                actions[iterations] = new_actions.to(device)
                response[iterations] = new_response.to(device)
                true_response[iterations] = true_responses.to(device)
                stages[iterations] = num_oracle_queries.to(device)
                num_of_queries_num += num_oracle_queries.item()
                
                # Update V_t and Phi--------------------------------------------------
                x_max_feature = model_ei.covar_module.base_kernel.get_features(new_actions, 18, normalize=True)
                weighted_feature = x_max_feature * (1 / eps_list[-1])
                V_t = V_t + torch.matmul(weighted_feature.T, weighted_feature)
                Unweighted_Phi[iterations, :] = x_max_feature            
                # Update end--------------------------------------------------

                # Save data periodically (every 100 stages) instead of every stage
                if iterations % 100 == 0 or num_of_queries_num >= n_iter:
                    save_data(actions[:iterations+1], true_response[:iterations+1], response[:iterations+1], stages[:iterations+1], eps_list, error_init, log_dir + str(obs_noise) + 'quan_training_data_'+str(tri) +'_.pth')
                if iterations % 100 == 0:
                    print('Stage {0} y: {1} f(x): {2} eps: {3} posterior: {4}'.format(iterations, round(new_response[-1].item(), 3), round(true_responses[-1].item(), 3), round(eps, 3),\
                        round(posterior.item(), 3)))
                error_init, file_name = env.reset(input_filename, seed=seed)
                iterations += 1
            