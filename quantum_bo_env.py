from qiskit import QuantumCircuit
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import LinearAmplitudeFunction

# from qiskit_aer.primitives import Sampler
from qiskit.primitives import Sampler

from qiskit_finance.circuit.library import NormalDistribution
# from qiskit_aer.noise import NoiseModel
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from joblib import dump, load
from datetime import datetime
import os
from os import path
import time
import random
from ansys.mapdl.core import launch_mapdl, launcher, Mapdl
from ansys.mapdl import reader as mapdl_reader
from shutil import copyfile
import torch

from circuit_utils import iterative_amplitude_estimation
from qiskit_ibm_runtime import QiskitRuntimeService

class BernoulliA(QuantumCircuit):
    def __init__(self, probability):
        super().__init__(1)
        # 原始的 theta
        theta_p = 2 * np.arcsin(np.sqrt(probability))
        # 施加总旋转：theta + 2*offset_phi
        # 注意：Ry(alpha) 旋转的角度是 alpha/2，所以这里传入 2*(theta/2 + offset_phi)
        self.ry(theta_p, 0)

# 2. 定义 Q 算子（它必须与 A 保持一致的角度步长）
class BernoulliQ(QuantumCircuit):
    def __init__(self, probability):
        super().__init__(1)
        # Q 算子的步长现在是基于 (theta_p + 2*offset_phi) 的
        self._theta_total = 2 * np.arcsin(np.sqrt(probability))
        self.ry(2 * self._theta_total, 0)

    def power(self, k):
        q_k = QuantumCircuit(1)
        q_k.ry(2 * k * self._theta_total, 0)
        return q_k

class QuantumFuselageEnv(object):
    def __init__(self, obs_noise, ip=None, n_actuators=10):
        self.ip = ip
        self.n_actuators = n_actuators
        torch.set_default_dtype(torch.float32)
        # self.mapdl = Mapdl(ip=self.ip, port=8800)
        # self.mapdl = launch_mapdl(ip=self.ip, port=8800, loglevel='ERROR', override=True, cleanup_on_exit=True, nproc=4)

        self.surrogate = load('surrogate_likeDu_v22.joblib').coef_
        
        self.tsai_wu = load('surrogate_tsaiwu.joblib')

        # self.device_quantum = FakeWashington()
        # self.coupling_map = self.device_quantum.configuration().coupling_map
        # self.noise_model = NoiseModel.from_backend(self.device_quantum)
        self.obs_noise = obs_noise
        
        real_machine = False
        
        if real_machine == True:
            service = QiskitRuntimeService()   # assumes you already saved your account
            self.backend = service.least_busy(
                operational=True, simulator=False, min_num_qubits=127
            )
            print(self.backend)
        

        
    def reset(self, filepath, seed):
        random.seed(seed)
        # np.random.seed(0)
        # torch.manual_seed(0)
        self.forces = np.zeros(18, dtype=np.float32) 
        # file = random.choice(os.listdir(folder))
        # file = input_filename
        # filepath = path.join(folder, file)
        # print("Initial shape from: ", input_filename.split('/')[-1])
        # Parse the Ansys file
        file = filepath.split('/')[-1]
        
        with open(filepath, 'r') as f:
            text = f.read()
            new_text = text.split('/com,******************* SOLVE FOR LS 1 OF 1 ****************')
            self.setup_text = new_text[0]
            new_text = text.split('! *********** WB SOLVE COMMAND ***********')
            self.finish_text = new_text[1]
            f.close()
        
        __file__ = 'FuselageActuators'
        folder = path.join(__file__, 'Shapes', 'Test') 
        file1 = file.split(".")[0] + ".npy"
        filepath = path.join(folder, file1)
        self.initPos = np.load(filepath)
        self.displacements = np.zeros((177, 2))
        
        folder = path.join(__file__, 'Shapes', 'Test') 
        # file2 = random.choice(os.listdir(folder))
        file2 = 'SolutionInputDP53.npy'

        # file2 = 'SolutionInputDP44.npy' for i in range(8,9): manual set
        # file2 = 'SolutionInputDP58.npy' for i in range(4,5): manual set
        
        while file1 == file2:   # make sure files are not the same
            file2 = random.choice(os.listdir(folder))
            
            
        # Load precalculated target positions
        filepath = path.join(folder, file2)
        # print("Target shape from", file2.split('.')[0])
        self.targetPos = np.load(filepath).astype(np.float32)
          
        self.deviations = self._get_deviation().astype(np.float32)
        self.initDev = self.deviations # for recording
        # self.deviations = self._get_deviation()
        # Assemble observation
        # obs = self.deviations
        # obs = obs.flatten()

        # Initialize the error
        self.error_init, self.maxDev = self._get_errors()
        
        return torch.tensor(self.error_init, dtype=torch.float32).reshape(1, 1), file2

    def cal_deviation(self, action):
        angles = np.linspace(12, -192, 18)
        action = action.reshape(-1)

        self.forces += action*1000 # Action space (-1,1) scaled to (-1000lb, 1000lb)
        
        self.forces = np.array(self.forces, dtype=np.float32)
        
        self.forces_Y = self.forces*np.cos(np.deg2rad(angles))
        self.forces_Z = self.forces*np.sin(np.deg2rad(angles))
        
        # Predict deviations from surrogate model
        u = np.dot(self.surrogate, self.forces).flatten()
        self.displacements = u.reshape((-1,2))

        # Track 
        p_init = self.initPos[:,0:2].flatten()
        p_final = p_init + u
        p_target = self.targetPos[:,0:2].flatten()
        self.deviations = p_final - p_target
        # Assemble observation
        
        self.error, self.maxDev = self._get_errors() #rmse, max_e
        
        return self.error
        # return -self.error/self.error_init
    
    
    def step_surrogate(self, action, eps, device=None):
        
        # torch.manual_seed(0)
        # np.random.seed(0)
        angles = np.linspace(12, -192, 18)
        action = action.detach().cpu().numpy().reshape(-1)

        # n = self.n_actuators
        self.forces += action*1000 # Action space (-1,1) scaled to (-1000lb, 1000lb)
        # idx = (abs(action)).argsort()[:9-n]
        # self.forces[idx] = 0
        
        self.forces = np.array(self.forces, dtype=np.float32)
        
        self.forces_Y = self.forces*np.cos(np.deg2rad(angles))
        self.forces_Z = self.forces*np.sin(np.deg2rad(angles))
        
        # Predict deviations from surrogate model
        u = np.dot(self.surrogate, self.forces).flatten()
        self.displacements = u.reshape((-1,2))

        # Track 
        p_init = self.initPos[:,0:2].flatten()
        p_final = p_init + u
        p_target = self.targetPos[:,0:2].flatten()
        self.deviations = p_final - p_target
        # Assemble observation
        variance = self.obs_noise 
        
        stddev = np.sqrt(variance)

        self.error, self.maxDev = self._get_errors() #rmse, max_e
        # Terminate after one time step
        mean = -self.error 

        # Quantum
        low = mean - 3 * stddev
        high = mean + 3 * stddev  
        
        num_uncertainty_qubits = 6
        uncertainty_model = NormalDistribution(num_uncertainty_qubits, mu=mean, sigma=stddev**2, bounds=(low, high))    

        c_approx = 1
        slopes = 1
        offsets = 0
        
        f_min = low
        f_max = high #change

        # The LinearAmplitudeFunction is a piecewise linear function
        linear_payoff = LinearAmplitudeFunction(
            num_uncertainty_qubits,
            slopes,
            offsets,
            domain=(low, high),
            image=(f_min, f_max),
            rescaling_factor=c_approx,
        )

        # construct A operator for QAE for the payoff function by
        # composing the uncertainty model and the objective
        num_qubits = linear_payoff.num_qubits
        monte_carlo = QuantumCircuit(num_qubits)
        monte_carlo.append(uncertainty_model, range(num_uncertainty_qubits))
        monte_carlo.append(linear_payoff, range(num_qubits))

        
        objective_qubits = [0]
        seed = 0
        epsilon = np.clip(eps / (1 * stddev), 1e-6, 0.5)

        alpha = 0.05
        
        # max_shots = 32 * np.log(2/alpha*np.log2(np.pi/(4*epsilon)))   

        # construct estimation problem. post_processing is the inverse of the rescaling, i.e., it maps the [0, 1] interval to the original one.
        # objective_qubits is the list of qubits that are used to encode the objective function.
        # problem is the estimation problem that is passed to the QAE algorithm.
        problem = EstimationProblem(state_preparation=monte_carlo, objective_qubits=objective_qubits, post_processing=linear_payoff.post_processing,)

        # # construct amplitude estimation
        ae = IterativeAmplitudeEstimation(epsilon_target=epsilon, alpha=alpha, sampler=Sampler(options={
            "shots": int(np.ceil(100)), "seed":seed}))
        result = ae.estimate(problem)
        

        scale_obs = result.estimation_processed / self.error_init
        num_oracle_queries = result.num_oracle_queries
        
        
        #######
        # estimation, scale_obs, oracle_queries_complexity, num_oracle_queries = iterative_amplitude_estimation(problem=problem, \
        #     backend=self.backend, \
        #     shots=int(np.ceil(max_shots)), \
        #     epsilon=epsilon, \
        #     alpha=alpha
        #     )
        # scale_obs = scale_obs / self.error_init
        #######
        
        if num_oracle_queries == 0:
            num_oracle_queries = 10
            ae = IterativeAmplitudeEstimation(epsilon_target=epsilon, alpha=alpha, sampler=Sampler(options={
                "shots": int(10), "seed":seed}))
            result = ae.estimate(problem)
            scale_obs = result.estimation_processed / self.error_init
            
        # return torch.tensor(obs, dtype=torch.float32).reshape(1,1).to(device), self.error, torch.tensor(num_oracle_queries).reshape(1,1).to(device)  #numpy
        return torch.tensor(scale_obs, dtype=torch.float32).reshape(1,1).to(device), torch.tensor(self.error, dtype=torch.float32).reshape(1,1).to(device), torch.tensor(num_oracle_queries).reshape(1,1).to(device)
                        
    def _run_ansys(self):
        try:
            self.mapdl.clear()
            log1 = self.mapdl.input_strings(self.setup_text) # run setup
            log2 = self._set_actuator_forces(self.mapdl, self.forces) # apply forces
            log3 = self.mapdl.input_strings(self.finish_text) # complete solution
            self.result = self.mapdl.result # store result
            self.mapdl.finish()
        except:
            print("exit ansys and try to reconnect 5 times")
            
            try:
                self.mapdl.exit()
                print("remote exit")
                time.sleep(30)
            except:
                time.sleep(30)
                
            i = 0
            while i <= 5:
                i += 1
                try: 
                    self.mapdl = Mapdl(ip=self.ip, port=8800)
                    print("sucessfully reconnect")
                    self.mapdl.clear()

                    log1 = self.mapdl.input_strings(self.setup_text) # run setup
                    log2 = self._set_actuator_forces(self.mapdl, self.forces) # apply forces
                    log3 = self.mapdl.input_strings(self.finish_text) # complete solution
                    self.result = self.mapdl.result # store result
                    self.mapdl.finish()
            
                    break
                except:
                    try:
                        print("reconnect fails and remote exit again")
                        self.mapdl.exit()
                        time.sleep(30)
                    except:
                        time.sleep(30)
        return self.result
    
    def _get_deviation(self):
        '''
        Calculate the distance of the current node positions from their ideal positions
        ''' 
        finalPos = self.initPos + self.displacements
        deviations = finalPos - self.targetPos[:,0:2]
        return deviations.flatten()
    
    def _get_displacement(self):
        '''
        Get the displacement of nodes on the fuselage edge after forces have been applied. 
        The displacements are relative to the initial positions of the nodes.
        '''
        self.mapdl.cmsel(name='CM_FUSELAGE_EDGE') # select nodes on the edge of the fuselage
        displacements = self.mapdl.post_processing.nodal_displacement('ALL') # get displacements of nodes on the edge
        self.mapdl.allsel()
        return displacements[:,1:3]
            
    def _get_initPos(self):
        '''
        Get the initial positions of nodes on the fuselage edge before any forces are applied
        '''
        self.mapdl.cmsel(name='CM_FUSELAGE_EDGE') # select nodes on the edge of the fuselage
        initPos = self.mapdl.mesh.nodes # initial positions of nodes
        nnum = self.mapdl.mesh.nnum # corresponding node numbers
        self.mapdl.allsel()
        return initPos[:,1:3]
        
    def _get_obs(self):
        # Get displacements from simulation
        self.displacements = self._get_displacement()
        # Calculate deviations
        self.deviations = self._get_deviation()
        obs = self.deviations
        obs = obs.flatten() #np.expand_dims(obs, -1)
        return np.array(obs, dtype=np.float32)

    def _get_errors(self):
        # Needs to be called after getting observations so that data is up to date
        # Calculate error relative to perfect circle with r=288
        n = len(self.deviations)
        dev_total = np.sqrt(np.square(self.deviations[:177]) + np.square(self.deviations[177:]))
        max_e = max(dev_total) # maximum error
        mae = sum(np.abs(self.deviations))/n # mean absolute error
        rmse = np.sqrt(sum((self.deviations)**2)/n) # root mean squared error
        mse = sum((self.deviations)**2)/n # mean squared error
        se = sum((self.deviations)**2) # sum of squared errors
        return mae, max_e

    def _set_actuator_forces(self, mapdl, forces):
        # Calculate y and z components of the forces from desired magnitudes
        angles = np.linspace(12, -192, 18)
        self.forces_Y = forces*np.cos(np.deg2rad(angles))
        self.forces_Z = forces*np.sin(np.deg2rad(angles))
        
        # Apply the forces as surface force on selected elements
        for i in range(0,18):
            # Set x component of force (practically zero)
            mapdl.esel("s", "real", "", 27+3*i)
            mapdl.sfe("all", 1, "pres", 1, 2.24808943074769e-009)
            # Set y component of force
            mapdl.esel("s", "real", "", 28+3*i)
            mapdl.sfe("all", 1, "pres", 1, self.forces_Y[i])
            # Set z component of force
            mapdl.esel("s", "real", "", 29+3*i)
            mapdl.sfe("all", 1, "pres", 1, self.forces_Z[i])
        mapdl.esel("all")   # make sure everything is selected before running solve 
        
        # Run the solution
        mapdl._run("/nopr")
        mapdl.run("/gopr")
        mapdl.run("nsub,1,1,1")
        mapdl.time(1.)
        mapdl.outres("erase")
        mapdl.outres("all", "none")
        mapdl.outres("nsol", "all")
        mapdl.outres("rsol", "all")
        mapdl.outres("eangl", "all")
        mapdl.outres("etmp", "all")
        mapdl.outres("veng", "all")
        mapdl.outres("strs", "all")
        mapdl.outres("epel", "all")
        mapdl.outres("eppl", "all")
        mapdl.outres("cont", "all")

    def _record(self):
        # Build dataframes
        df1 = pd.DataFrame(self.initDev, self.h1).T
        df2 = pd.DataFrame(self.forces, self.h2).T
        df3 = pd.DataFrame(self.finalDev, self.h3).T
        # Join them together
        df = pd.concat([df1, df2, df3], axis=1)
        # Write csv file
        df.to_csv(self.recordPath, mode='a', header=not os.path.exists(self.recordPath))
        
    # def step(self, action): #argmax UCB
    #     '''
    #     action: select from action_sapce: it is an array \in R^(18), with (18,)
    #     '''
    #     # n = self.n_actuators
    #     # if type(action) == torch.Tensor():
    #     #     action = action.detach().numpy()
    #     action = action.detach().cpu().numpy().reshape(-1)
        
    #     self.forces += action*1000 # Action space (-1,1) scaled to (-1000lb, 1000lb)
    #     self.forces = np.array(self.forces, dtype=np.float32)
    #     # # print('Input X:', self.forces)
        
    #     # idx = (abs(action)).argsort()[:18-n]
        
    #     # self.forces[idx] = 0  #eliminate minimum forces
        
    #     # Run the Ansys simulation with forces
    #     self._run_ansys()
    #     # Get displacements from Ansys
    #     self.displacements = self._get_displacement()
    #     u = self.displacements.flatten()
    #     # Track 
    #     p_init = self.initPos[:,0:2].flatten() #Phi
    #     p_final = p_init + u #Yc + Y(F)
    #     p_target = self.targetPos[:,0:2].flatten()
        
    #     self.deviations = p_final - p_target #Yc + Y(F) - Y^*
    #     # Assemble observation
    #     obs = self.deviations
    #     obs = obs.flatten()
    #     self.error, self.maxDev = self._get_errors() #rmse, max_e
    #     # Terminate after one time step
    #     return torch.tensor(self.error, dtype=torch.float32).reshape(1,1), self.error


    # def step_ansys(self, action, eps, device=None): #calculate the system response
    #     '''
    #     action: select from action_sapce: it is an array \in R^(9), with (9,)
    #     '''
    #     quantum_noise = False
    #     variance = self.obs_noise
    #     stddev = np.sqrt(variance)
    #     action = action.detach().cpu().numpy().reshape(-1)
        
    #     self.forces += action*1000 # Action space (-1,1) scaled to (-1000lb, 1000lb)
    #     self.forces = np.array(self.forces, dtype=np.float32)
        
    #     # Run the Ansys simulation with forces
    #     self._run_ansys()
    #     # Get displacements from Ansys
    #     self.displacements = self._get_displacement()
    #     u = self.displacements.flatten()
    #     # Track 
    #     p_init = self.initPos[:,0:2].flatten() #Phi
    #     p_final = p_init + u #Yc + Y(F)
    #     p_target = self.targetPos[:,0:2].flatten()
        
    #     self.deviations = p_final - p_target #Yc + Y(F) - Y^*
    #     # Assemble observation
    #     observation = self.deviations
    #     observation = observation.flatten()
    #     self.error, self.maxDev = self._get_errors() #rmse, max_e
    #     # Terminate after one time step

    #     mean = self.error
    #     # Quantum
    #     low = mean - 3 * stddev
    #     high = mean + 3 * stddev  
    #     num_uncertainty_qubits = 6
    #     uncertainty_model = NormalDistribution(num_uncertainty_qubits, mu=mean, sigma=stddev**2, bounds=(low, high))    

    #     c_approx = 1
    #     slopes = 1
    #     offsets = 0
        
    #     f_min = low
    #     f_max = high #change

    #     # The LinearAmplitudeFunction is a piecewise linear function
    #     linear_payoff = LinearAmplitudeFunction(
    #         num_uncertainty_qubits,
    #         slopes,
    #         offsets,
    #         domain=(low, high),
    #         image=(f_min, f_max),
    #         rescaling_factor=c_approx,
    #     )

    #     # construct A operator for QAE for the payoff function by
    #     # composing the uncertainty model and the objective
    #     num_qubits = linear_payoff.num_qubits
    #     monte_carlo = QuantumCircuit(num_qubits)
    #     monte_carlo.append(uncertainty_model, range(num_uncertainty_qubits))
    #     monte_carlo.append(linear_payoff, range(num_qubits))

    #     # set target precision and confidence level
    #     epsilon = eps / (3 * stddev)

    #     objective_qubits = [0]
    #     seed = 0

    #     epsilon = np.clip(epsilon, 1e-6, 0.5)

    #     alpha = 0.05
    #     # max_shots = (32 / (1 - 2 * np.sin(np.pi/14))**2) * np.log(2/alpha*np.log2(np.pi/(4*epsilon)))
    #     # max_shots = 32 * np.log(2/alpha*np.log2(np.pi/(4*epsilon)))
    #     # max_shots = 100
    #     # construct estimation problem. post_processing is the inverse of the rescaling, i.e., it maps the [0, 1] interval to the original one.
    #     # objective_qubits is the list of qubits that are used to encode the objective function.
    #     # problem is the estimation problem that is passed to the QAE algorithm.
    #     problem = EstimationProblem(state_preparation=monte_carlo, objective_qubits=objective_qubits, post_processing=linear_payoff.post_processing, )
    #     # construct amplitude estimation

    #     if quantum_noise == True:
    #         ae = IterativeAmplitudeEstimation(
    #         epsilon_target=epsilon, alpha=alpha, sampler=Sampler(backend_options={
    #             "method": "density_matrix",
    #             "coupling_map": self.coupling_map,
    #             "noise_model": self.noise_model,
    #         },run_options={"shots": int(np.ceil(max_shots)),"seed_simulator":seed},
    #         transpile_options={"seed_transpiler": seed},)
    #     )
    #     else:
    #         ae = IterativeAmplitudeEstimation(epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={"shots": int(np.ceil(max_shots)), "seed_simulator":seed}))

    #     # Running result
    #     result = ae.estimate(problem)
    #     obs = result.estimation_processed
        
    #     num_oracle_queries = result.num_oracle_queries

    #     if num_oracle_queries == 0:
    #         # print("number of oracle is 0!!")
    #         # use the number of oracle calls given by the paper if num_oracle_queries == 0 average oracles
    #         num_oracle_queries = int(np.ceil((0.8 / epsilon) * np.log((2 / alpha) * np.log2(np.pi / (4 * epsilon)))))
        
    #     return torch.tensor(obs, dtype=torch.float32).reshape(1,1).to(device), self.error, torch.tensor(num_oracle_queries).reshape(1,1).to(device)  #numpy

