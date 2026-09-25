import numpy as np
import pandas as pd
from joblib import dump, load
from datetime import datetime
import os
from os import path
import time
import random
import ansys
from ansys.mapdl.core import launch_mapdl, launcher, Mapdl
from ansys.mapdl import reader as mapdl_reader
from shutil import copyfile
import torch

from math import log, ceil, sqrt
from statistics import mean, pstdev
from typing import Callable, Tuple, Optional, List
import random

class ClassicFuselageEnv(object):
    def __init__(self, obs_noise, ip=None, min_shots=100, n_actuators=18, eps_max=0.04, force_scale=1000.0):
        self.ip = ip
        self.n_actuators = n_actuators
        self.surrogate = load('surrogate_likeDu_v22.joblib').coef_
        # self.mapdl = launch_mapdl(ip=self.ip, port=8800, loglevel='ERROR', override=True, cleanup_on_exit=True, nproc=4)
        self.obs_noise = obs_noise
        self.min_shots = min_shots
        self.eps_max = eps_max
        self.force_scale = force_scale  # action in (-1,1) scaled to (-force_scale, +force_scale) lb
        
    def reset(self, filepath, seed=None, target_npy=None):
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
        self.displacements = np.zeros((177,2))
        
        folder = path.join(__file__, 'Shapes', 'Test') 
        if target_npy is not None:
            file2 = target_npy.split('/')[-1]
        else:
            file2 = 'SolutionInputDP53.npy'
        
        while file1 == file2:   # make sure files are not the same
            file2 = random.choice(os.listdir(folder))
            
        # Load precalculated target positions
        filepath = path.join(folder, file2)
        # print("Target shape from", file2.split('.')[0])
        self.targetPos = np.load(filepath)  
          
        self.deviations = self._get_deviation()
        self.initDev = self.deviations # for recording
        
        # self.deviations = self._get_deviation()
        # Assemble observation
        # obs = self.deviations
        # obs = obs.flatten()

        # Initialize the error
        self.error_init, self.maxDev = self._get_errors()
        
        p_init = self.initPos[:,0:2].flatten() #Phi

        p_target = self.targetPos[:,0:2].flatten()
        
        shape_deviation = p_init - p_target
        
        return file2, torch.tensor(self.error_init, dtype=torch.float32).reshape(1,1)
    
    def cal_deviation(self, action):
        angles = np.linspace(12, -192, 18)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy().reshape(-1)
        else:
            action = action.reshape(-1)

        self.forces += action*self.force_scale # Action space (-1,1) scaled to (-force_scale, +force_scale) lb
        
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
        self.error, self.maxDev = self._get_errors() #mae, max_e
        
        return self.error
        #return -self.error/self.error_init
    
    def _set_actuator_forces(self, mapdl, forces):
        # Calculate y and z components of the forces from desired magnitudes
        angles = np.linspace(12, -192, 18)
        self.forces_Y = forces*np.cos(np.deg2rad(angles))
        self.forces_Z = forces*np.sin(np.deg2rad(angles))
        
        # Apply the forces as surface force on selected elements
        for i in range(0 , 18):
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

    def step_surrogate(self, action, method=None, device=None, eps=1e-3):
        
        action = action.detach().cpu().numpy().reshape(-1)
        angles = np.linspace(12, -192, 18)
        
        self.forces += action*self.force_scale # Action space (-1,1) scaled to (-force_scale, +force_scale) lb
        
        self.forces = np.array(self.forces, dtype=np.float32)
        
        self.forces_Y = self.forces*np.cos(np.deg2rad(angles))
        self.forces_Z = self.forces*np.sin(np.deg2rad(angles))
        
        u = np.dot(self.surrogate, self.forces).flatten()
        self.displacements = u.reshape((-1,2))

        p_init = self.initPos[:,0:2].flatten() #Phi
        p_final = p_init + u #Yc + Y(F)
        p_target = self.targetPos[:,0:2].flatten()
        
        self.deviations = p_final - p_target
        
        self.error, self.maxDev = self._get_errors() #rmse, max_e
        
        # n = len(self.deviations)
        # noise_level = (self.obs_noise / (self.error_init**2 * n))
        # Terminate after one time step
        variance = self.obs_noise
        stddev = np.sqrt(variance)

        
        relative_obs, num_oracle_queries = self._monte_carlo_estimate(draw=lambda: self.draw_gaussian(mean=-self.error, std=stddev), var=variance, \
            epsilon=eps, delta=0.05, max_samples=29999, method=method)     
        
        relative_error = relative_obs / self.error_init
        
        return torch.tensor(relative_error, dtype=torch.float32).to(device).reshape(1,1),  torch.tensor(self.error, dtype=torch.float32).to(device).reshape(1,1), torch.tensor(num_oracle_queries).reshape(1,1).to(device)
    
    def draw_gaussian(self, mean, std):
        return random.gauss(mean, std)
    
    def make_trunc_normal_draw(self, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        thr = 3.0 * sigma
        def draw():
            while True:
                x = np.random.normal(loc=mu, scale=sigma)
                if abs(x - mu) <= thr:   # keep only within ±3σ
                    return x
        return draw
    
    def step(self, action, device): #argmax UCB
        '''
        action: select from action_sapce: it is an array \in R^(18), with (18,)
        '''
        # n = self.n_actuators
        # if type(action) == torch.Tensor():
        #     action = action.detach().numpy()
        action = action.detach().cpu().numpy().reshape(-1)
        
        self.forces += action*self.force_scale # Action space (-1,1) scaled to (-force_scale, +force_scale) lb
        self.forces = np.array(self.forces, dtype=np.float32)
        # # print('Input X:', self.forces)
        
        # idx = (abs(action)).argsort()[:18-n]
        
        # self.forces[idx] = 0  #eliminate minimum forces
        
        # Run the Ansys simulation with forces
        self._run_ansys()
        # Get displacements from Ansys
        self.displacements = self._get_displacement()
        u = self.displacements.flatten()
        # Track 
        p_init = self.initPos[:,0:2].flatten() #Phi
        p_final = p_init + u #Yc + Y(F)
        p_target = self.targetPos[:,0:2].flatten()
        
        self.deviations = p_final - p_target #Yc + Y(F) - Y^*
        # Assemble observation

        self.error, self.maxDev = self._get_errors() #rmse, max_e
        # Terminate after one time step
        obs = np.random.normal(self.error, np.sqrt(self.obs_noise))

        return torch.tensor(obs, dtype=torch.float32).to(device).reshape(1,1), self.error


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
        mae = sum(np.abs(self.deviations))/n # mean absolute error 0.3
        rmse = np.sqrt(sum((self.deviations)**2)/n) # root mean squared error
        mse = sum((self.deviations)**2)/n # mean squared error
        se = sum((self.deviations)**2) # sum of squared errors
        return mae, max_e

    def _record(self):
        # Build dataframes
        df1 = pd.DataFrame(self.initDev, self.h1).T
        df2 = pd.DataFrame(self.forces, self.h2).T
        df3 = pd.DataFrame(self.finalDev, self.h3).T
        # Join them together
        df = pd.concat([df1, df2, df3], axis=1)
        # Write csv file
        df.to_csv(self.recordPath, mode='a', header=not os.path.exists(self.recordPath))
        
    def _monte_carlo_estimate(self,
        draw: Callable[[], float],
        epsilon: float,
        var: float,
        delta: float = 0.05,
        method: str = "chebyshev",
        min_batch: int = 100, 
        max_samples: Optional[int] = None,
    ) -> Tuple[float, int]:
        """
        Estimate E[X] with additive error <= epsilon with probability >= 1 - delta.

        Parameters
        ----------
        draw : Callable[[], float]
            A function that returns one i.i.d. sample X.
            For 'hoeffding' you should ensure X ∈ [0, 1] (or scale it).
        epsilon : float
            Target absolute error tolerance.
        delta : float, default 0.05
            Failure probability (1 - confidence).
        method : {'hoeffding', 'clt', 'chebyshev'}, default 'hoeffding'
            - 'hoeffding': fixed-sample bound assuming X ∈ [0, 1].
            - 'clt': sequential stopping using a (robust) CI from the central limit theorem.
        min_batch : int, default 100
            Batch size to sample at a time for the 'clt' method.
        max_samples : Optional[int], default None
            Safety cap on the total samples. If None, unlimited.

        Returns
        -------
        mu_hat : float
            The Monte Carlo estimate of the mean.
        n : int
            The total number of queries (samples) used.
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0,1)")
        if method not in {"hoeffding", "clt", "chebyshev", "non_monte_carlo"}:
            raise ValueError("method must be 'hoeffding' or 'clt'")
        
        if method == 'non_monte_carlo':
            n_required_precision = self.eps_max

            n_required = ceil((var/n_required_precision**2)*(1/delta))
            if n_required < self.min_shots:
                n_required = self.min_shots

            samples = [draw() for _ in range(n_required)]
            return (sum(samples) / n_required, n_required)

        # Method 1: Distribution-free fixed-N via Hoeffding (assumes X ∈ [0,1])
        if method == "hoeffding":
            # n >= (1/(2*epsilon^2)) * ln(2/delta)
            n_required = ceil((1.0 / (2.0 * (epsilon ** 2))) * log(2.0 / delta))
            if max_samples is not None:
                n_required = min(n_required, max_samples)
            samples = [draw() for _ in range(n_required)]
            return (sum(samples) / n_required, n_required)
        
        if method == 'chebyshev':            
            n_required = ceil((var/epsilon**2)*(1/delta))
            if max_samples is not None:
                n_required = min(n_required, max_samples)
            n_required = max(n_required, self.min_shots)
                
            samples = [draw() for _ in range(n_required)]
            return (sum(samples) / n_required, n_required)
            
        # Method 2: Sequential CLT-based stopping (works beyond [0,1], but needs finite variance)
        if method == 'clt':
            samples: List[float] = []
            # z for two-sided (1 - delta) CI; use normal quantile ~1.96 for 95%,
            # or a conservative upper bound via 2.0 for simplicity.
            # If you want exact, import from scipy.stats: z = st.norm.ppf(1 - delta/2)
            def z_two_sided(d: float) -> float:
                # crude piecewise without SciPy
                if d <= 0.02:
                    return 2.326347874  # ~98% CI
                if d <= 0.05:
                    return 1.959963985  # ~95% CI
                if d <= 0.10:
                    return 1.644853627  # ~90% CI
                if d == 0.05:
                    return 1.959963985  
                return 1.281551566      # ~80% CI
            z = z_two_sided(delta)
            
            n_required = ceil(var*((z)/epsilon)**2)
            if max_samples is not None:
                n_required = min(n_required, max_samples)
            samples = [draw() for _ in range(n_required)]
            return (sum(samples) / n_required, n_required)
        
