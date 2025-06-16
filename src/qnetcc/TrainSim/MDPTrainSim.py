import os

from qnetcc.environments.MDPEnv import Environment
from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network
from stable_baselines3 import PPO
from qnetcc.TrainSim.MyCallbacks import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
import re # for getting integers from strings
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tqdm
import math
import time
import random
import torch

from matplotlib import rc
import style
rc('font',**style.fontstyle)
rc('text', usetex=True)

class training_and_policy_simulation(object):
    def __init__(self, nodes, t_cut, p, p_s, 
                do_training, do_further_training, training_steps,
                do_simulation, simulation_eps, time_out_mult = 2,
                new_training_version = 1, training_version_ = None,
                ent_reg = 0.001, agent_verbose = 1,
                callback=1, save_times = 10,
                cluster = 0,
                cc_effects = 1,
                swap_asap_check = 0,
                seed = 42,
                indSim = 0):
        """The initial settings defining the scenario's that we do our training and 
        simulations on. 

        Also for setting some other stuff, such as where to save/get model/data, the verbosity of the functions etc. 
        """
        # network params
        self.nodes = nodes
        self.t_cut = t_cut
        self.p = p
        self.p_s = p_s

        self.cc_effects = cc_effects
        self.swap_asap_check = swap_asap_check

        self.abs_path = os.path.abspath(os.getcwd())
        self.save_path = os.path.join(self.abs_path, '..', '..', 'data/global_agent_swap_sum')
        self.sub_proj_rel_path = '/env_cc_a_alt_o_hist'

        # forcing independent traning simulations
        self.indSim = indSim

        # if running on a cluster, use the paths for the cluster
        self.cluster = cluster
        if self.cluster == 1:
            self.abs_path = os.path.abspath(os.getcwd())
            self.save_path = '/home/lijt/data1/quantum_network'
        if self.cc_effects == 0:
            self.sub_proj_rel_path = '/env_a_alt_o_hist'
                
        # make new version if there is already a trained version, and if the training_version has not been set explicitly
        self.new_training_version = new_training_version
        self.training_version = 0
        if self.new_training_version == 1:
            temp_model_path = self.save_path+self.sub_proj_rel_path+f'/Training{self.training_version}'+f'/PPO_cc_nodes_{nodes}_t_cut_{t_cut}_p_{p:.02f}_p_s_{p_s:.02f}.zip'
            while os.path.exists(temp_model_path): # temp_model_path is only used to check if model with training_version already exists
                self.training_version += 1
                temp_model_path = self.save_path+self.sub_proj_rel_path+f'/Training{self.training_version}'+f'/PPO_cc_nodes_{nodes}_t_cut_{t_cut}_p_{p:.02f}_p_s_{p_s:.02f}.zip'
        # overwrite what ever version we are at and go to fixed training version
        if training_version_ != None:
            self.training_version = training_version_
        self.all_model_folder = self.save_path+self.sub_proj_rel_path+f'/Training{self.training_version}'
        self.data_folder = self.save_path+self.sub_proj_rel_path+f'/sim_dat{self.training_version}'
        if self.indSim == 1:
            self.data_folder = self.save_path+self.sub_proj_rel_path+f'/sim_datInd{self.training_version}'
        self.fig_folder = self.save_path+self.sub_proj_rel_path+f'/figures'
        self.model_path = self.all_model_folder+f'/PPO_cc_nodes_{nodes}_t_cut_{t_cut}_p_{p:.02f}_p_s_{p_s:.02f}.zip'
        # path for delivery times
        self.file_name_template = f'/sim_dat_cc_'+f'n_{self.nodes}_t_cut_{self.t_cut}_p_{self.p:.02f}_p_s_{self.p_s:.02f}'
        self.sim_dat_path_template = self.data_folder+self.file_name_template
        # paths for delivery times and fidelity
        self.fid_data_folder = self.save_path+self.sub_proj_rel_path+f'/fid_dat{self.training_version}'
        if self.indSim == 1:
            self.fid_data_folder = self.save_path+self.sub_proj_rel_path+f'/fid_datInd{self.training_version}'
        self.fidelity_file_name_template = f'/fid_dat_cc_fidelity_'+f'n_{self.nodes}_t_cut_{self.t_cut}_p_{self.p:.02f}_p_s_{self.p_s:.02f}'
        self.sim_fidelity_dat_path_template = self.fid_data_folder+self.fidelity_file_name_template
        #model log path
        self.model_log_path = self.all_model_folder+f'/PPO_cc_nodes_{nodes}_t_cut_{t_cut}_p_{p:.02f}_p_s_{p_s:.02f}/'
        if not os.path.exists(self.model_log_path):
            os.makedirs(self.model_log_path)
        self.learning_curve_path = self.model_log_path+'/learn_curve/'

        # training_params
        self.do_training = do_training
        self.do_further_training = do_further_training
        self.training_steps = training_steps
        self.eval_freq = int(1e5)

        # PPO params
        self.ent_reg = ent_reg
        self.agent_verbose = agent_verbose

        # simulation params
        self.do_simulation = do_simulation
        self.sim_eps = simulation_eps
        self.time_out_mult = time_out_mult

        # call backs
        self.callback = callback
        self.save_times = save_times
        self.save_freq = int(self.training_steps/self.save_times) # save every this many time steps

        # seeds
        if seed != None: # when a random seed has been set, by default it is 42
            if training_version_ != None:
                seed = training_version_ # setting the seed to be equal to the training version
            self.env_seed = seed
            self.np_seed = seed
            self.rand_seed = seed
            self.torch_seed = seed
            np.random.seed(self.np_seed)
            random.seed(self.rand_seed)
            torch.manual_seed(self.torch_seed)
            torch.cuda.manual_seed(self.torch_seed)
            torch.cuda.manual_seed_all(self.torch_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.model_seed = seed # attribute has to exists, even if None, because passed into PPO(...)


    def do_training_and_simulation(self):
        """does the training for the RL agent if applicable and does the simulations with the chosen policies
        """

        self.train_model()
        simulation_delivery_times =  self.simulate_policy()
        return simulation_delivery_times 
    
    def simulate_policy(self):
        """does simulation and saves the simulated delivery times
        """
        # simulate policy; if set to 0, it returns None
        if self.do_simulation == 1:
            # lists for saving the simulation data
            T_list = []
            micro_T_list = []
            swap_asap_ratio_list = []

            # loading model and environment
            env = Environment(self.get_pos_center_agent(), self.nodes, self.t_cut, self.p, self.p_s, cc_effects=self.cc_effects)
            model = self.get_model()
            # if no model, give env.max_moves/2 as delivery time by default
            if model == 'no model yet':
                average_regular_time = [env.max_moves/2 for i in range(self.sim_eps)] # should be divided by two actually, one mdp step is a round, not time step
                std_regular_time = [0 for i in range(self.sim_eps)]
                average_micro_time = [env.max_moves for i in range(self.sim_eps)]
                std_micro_time = [0 for i in range(self.sim_eps)]
                simulation_delivery_times = [
                                average_regular_time, std_regular_time,
                                average_micro_time, std_micro_time]
            # if there is a trained model for the specified parameters
            else:
                model.set_env(env)
                if self.indSim == 1:
                    np.random.seed(None)
                    random.seed(None)
                start_time = time.time()
                # looping over different episodes and storing the delivery time of each of them
                for episode in tqdm.tqdm(range(self.sim_eps),leave=False):
                    obs, _ = env.reset()
                    done = False
                    score = 0
                    while not done:
                        action, _ = model.predict(obs)
                        if self.swap_asap_check:
                            swap_asap_ratio = self.get_swap_asap_ratio(action=action, env=env)
                        obs, reward, done, _, info = env.step(action)
                        score+=reward
                        stop_time = time.time()
                        # if simulation takes too long, break it off
                        if (stop_time-start_time) > self.sim_eps*self.time_out_mult:
                            break
                    # Have to break twice and then set delivery time to env.max_moves/2
                    if (stop_time-start_time) > self.sim_eps*self.time_out_mult:
                        T_list = [env.max_moves/2 for i in range(self.sim_eps)]
                        micro_T_list = [env.max_moves for i in range(self.sim_eps)]
                        break
                    # if simulation didn't take too long
                    # simulation data of every episode is appended
                    T_list.append(env.quantum_network.time_slot)
                    micro_T_list.append(env.quantum_network.micro_time_slot)
                    if self.swap_asap_check:
                        swap_asap_ratio_list.append(swap_asap_ratio)
                env.close()
                # saving the simulation data
                self.save_and_load_simulation_data(T_list, micro_T_list)
                
                # saving the delivery times (at different time scales) with their std
                average_regular_time = np.average(T_list)
                std_regular_time = np.std(T_list)/np.sqrt(self.sim_eps)
                print(f' average regular time = {average_regular_time} with std {std_regular_time}')
                average_micro_time = np.average(micro_T_list)
                std_micro_time = np.std(micro_T_list)/np.sqrt(self.sim_eps)
                print(f' average micro time = {average_micro_time} with std {std_micro_time}')
            
                simulation_delivery_times = [
                                    average_regular_time, std_regular_time,
                                    average_micro_time, std_micro_time]
            if self.swap_asap_check:
                return simulation_delivery_times, swap_asap_ratio_list
            else:
                return simulation_delivery_times, T_list, micro_T_list
        else:
            return None
        

    def simulate_policy_w_fidelity(self):
        """does simulation and saves the simulated delivery times
        """
        # simulate policy; if set to 0, it returns None
        if self.do_simulation == 1:
            # lists for saving the simulation data
            T_list = []
            micro_T_list = []
            fidelity_list = []
            swap_asap_ratio_list = []

            # loading model and environment
            env = Environment(self.get_pos_center_agent(), self.nodes, self.t_cut, self.p, self.p_s, cc_effects=self.cc_effects)
            model = self.get_model()
            # if no model, give env.max_moves/2 as delivery time by default
            if model == 'no model yet':
                average_regular_time = [env.max_moves/2 for i in range(self.sim_eps)] # should be divided by two actually, one mdp step is a round, not time step
                std_regular_time = [0 for i in range(self.sim_eps)]
                average_micro_time = [env.max_moves for i in range(self.sim_eps)]
                std_micro_time = [0 for i in range(self.sim_eps)]
                simulation_delivery_times = [
                                average_regular_time, std_regular_time,
                                average_micro_time, std_micro_time]
            # if there is a trained model for the specified parameters
            else:
                model.set_env(env)
                if self.indSim == 1:
                    np.random.seed(None)
                    random.seed(None)
                start_time = time.time()
                # looping over different episodes and storing the delivery time of each of them
                for episode in tqdm.tqdm(range(self.sim_eps),leave=False):
                    obs, _ = env.reset()
                    done = False
                    score = 0
                    while not done:
                        action, _ = model.predict(obs)
                        if self.swap_asap_check:
                            swap_asap_ratio = self.get_swap_asap_ratio(action=action, env=env)
                        obs, reward, done, _, info = env.step(action)
                        score+=reward
                        stop_time = time.time()
                        # if simulation takes too long, break it off
                        if (stop_time-start_time) > self.sim_eps*self.time_out_mult:
                            break
                    # Have to break twice and then set delivery time to env.max_moves/2
                    if (stop_time-start_time) > self.sim_eps*self.time_out_mult:
                        T_list = [env.max_moves/2 for i in range(self.sim_eps)]
                        micro_T_list = [env.max_moves for i in range(self.sim_eps)]
                        fidelity_list = [np.nan for i in range(self.sim_eps)]
                        break
                    # if simulation didn't take too long
                    # simulation data of every episode is appended
                    T_list.append(env.quantum_network.time_slot)
                    micro_T_list.append(env.quantum_network.micro_time_slot)
                    fidelity_list.append(env.quantum_network.get_fidelity())
                    assert env.quantum_network.get_fidelity() <= self.t_cut, "link is older than cutoff time"
                    if self.swap_asap_check:
                        swap_asap_ratio_list.append(swap_asap_ratio)
                env.close()
                # saving the simulation data
                self.save_and_load_simulation_fidelity_data(T_list, micro_T_list, fidelity_list)
                
                # saving the delivery times (at different time scales) with their std
                average_regular_time = np.average(T_list)
                std_regular_time = np.std(T_list)/np.sqrt(self.sim_eps)
                print(f' average regular time = {average_regular_time} with std {std_regular_time}')
                average_micro_time = np.average(micro_T_list)
                std_micro_time = np.std(micro_T_list)/np.sqrt(self.sim_eps)
                print(f' average micro time = {average_micro_time} with std {std_micro_time}')
            
                simulation_delivery_times = [
                                    average_regular_time, std_regular_time,
                                    average_micro_time, std_micro_time]
            if self.swap_asap_check:
                return simulation_delivery_times, swap_asap_ratio_list
            else:
                return simulation_delivery_times, T_list, micro_T_list, fidelity_list
        else:
            return None
    
        
    def simulate_policy_w_print(self):
        """does simulation and saves the simulated delivery times
        """
        if self.do_simulation == 1:
            T_list = []
            micro_T_list = []
            swap_asap_ratio_list = []

            if self.nodes == 4:
                assert self.get_pos_center_agent() == 2

            env = Environment(self.get_pos_center_agent(), self.nodes, self.t_cut, self.p, self.p_s, cc_effects=self.cc_effects)
            model = self.get_model()
            model.set_env(env)
            if self.indSim == 1:
                np.random.seed(None)
                random.seed(None)
            start_time = time.time()
            for episode in tqdm.tqdm(range(1, self.sim_eps), leave=False):
                obs, _ = env.reset()
                done = False
                score = 0
                while not done:
                    action, _ = model.predict(obs)
                    print(f'---------------------------------------------')
                    print(f'time step = {env.mdp_time_step}')
                    print(f'action round = {env.action_time_step}')
                    print(f'sent actions = {action}')
                    if self.swap_asap_check:
                        swap_asap_ratio = self.get_swap_asap_ratio(action=action, env=env)
                    obs, reward, done, _, info = env.step(action)
                    score+=reward
                    stop_time = time.time()
                    print(f'------ printed after env.step() ---------')
                    # print(f'history = {env.info_hist[0:4,:]}')
                    # print(f'observation = {obs[0:5,:]}') 
                    print(f'consec end-to-end time steps {env.consec_A_B_ent_time_steps}')
                    print(f'reward, terminated {reward, done}')
                    env.update_action_time_step()
                    print(f'applied actions {env.get_actions()}')
                    print(f'results {env.get_results_for_info_hist()}')
                    env.update_action_time_step()
                    print(f'quantum network state = {env.quantum_network.get_link_config()}')

                    if (stop_time-start_time) > self.sim_eps*self.time_out_mult:
                        break
                if (stop_time-start_time) > self.sim_eps*self.time_out_mult:
                    T_list = [env.max_moves for i in range(self.sim_eps)]
                    micro_T_list = [2*env.max_moves for i in range(self.sim_eps)]
                    break
                T_list.append(env.quantum_network.time_slot)
                micro_T_list.append(env.quantum_network.micro_time_slot)
                if self.swap_asap_check:
                    swap_asap_ratio_list.append(swap_asap_ratio)
            env.close()
            self.save_and_load_simulation_data(T_list, micro_T_list)
            
            # saving the delivery times (at different time scales) with their std
            average_regular_time = np.average(T_list)
            std_regular_time = np.std(T_list)/np.sqrt(self.sim_eps)
            print(f' average regular time = {average_regular_time} with std {std_regular_time}')
            average_micro_time = np.average(micro_T_list)
            std_micro_time = np.std(micro_T_list)/np.sqrt(self.sim_eps)
            print(f' average micro time = {average_micro_time} with std {std_micro_time}')
        
            simulation_delivery_times = [
                                average_regular_time, std_regular_time,
                                average_micro_time, std_micro_time]
            if self.swap_asap_check:
                return simulation_delivery_times, swap_asap_ratio_list
            else:
                return simulation_delivery_times
        else:
            return None, None

    def save_and_load_simulation_data(self, T_list, micro_T_list):
        """saving the simulated delivery times
        """

        if self.do_simulation == 1:
            version = 0                 
            sim_dat_path = self.sim_dat_path_template+f'_v{version}.pkl'
            sim_dat = np.array([T_list, micro_T_list])
            if os.path.exists(self.data_folder) == False:
                os.makedirs(self.data_folder)
            while os.path.exists(sim_dat_path) == True:
                version += 1
                sim_dat_path = self.sim_dat_path_template+f'_v{version}.pkl'
            with open(sim_dat_path, "wb") as file:   # Pickling
                pickle.dump(sim_dat, file)
            with open(sim_dat_path, "rb") as file:   # Unpickling
                sim_dat = pickle.load(file)

    def save_and_load_simulation_fidelity_data(self, T_list, micro_T_list, fidelity_list):
        """saving the simulated delivery times
        """

        if self.do_simulation == 1:
            version = 0                 
            fid_dat_path = self.sim_fidelity_dat_path_template+f'_v{version}.pkl'
            fid_dat = np.array([T_list, micro_T_list, fidelity_list])
            if os.path.exists(self.fid_data_folder) == False:
                os.makedirs(self.fid_data_folder)
            while os.path.exists(fid_dat_path) == True:
                version += 1
                fid_dat_path = self.sim_fidelity_dat_path_template+f'_v{version}.pkl'
            with open(fid_dat_path, "wb") as file:   # Pickling
                pickle.dump(fid_dat, file)
            with open(fid_dat_path, "rb") as file:   # Unpickling
                fid_dat = pickle.load(file)
        
    def get_swap_asap_ratio(self, action, env):
        """To check what percentage of the action align with the swap asap protocol
        """
        equiv_count = 0
        eps_count = 0
        swap_asap_action = env.quantum_network.instant_comm_swap_asap_actions()
        if env.quantum_network.swap_action_time_step():
            if np.array_equal(swap_asap_action[1:], action[1:]):
                equiv_count += 1
        elif env.quantum_network.ent_gen_time_step():
            if np.array_equal(swap_asap_action, action):
                equiv_count += 1
        else:
            # print('----------------------------------')
            # print(f'time slot {env.quantum_network.time_slot}')
            # if env.quantum_network.swap_action_time_step():
            #     print(f'swap time step')
            # else:
            #     print(f'ent gen time step')
            # print(f'RL action {action}')
            # print(f'swap asap action {swap_asap_action}')
            pass
        eps_count += 1
        equiv_ratio =  equiv_count/eps_count
        return equiv_ratio
    
    def get_model_after_n_steps(self, trained_steps):
        """Getting the RL model after having trained n steps
        """
        model_path = self.all_model_folder+f'/PPO_cc_nodes_{nodes}_t_cut_{t_cut}_p_{p:.02f}_p_s_{p_s:.02f}/rl_model_{int(trained_steps)}_steps.zip'
        if os.path.exists(model_path):
            model = PPO.load(model_path)
        else:
            assert False, 'no model with these training steps'
        return model
    
    def get_model(self):
        """Getting previously trained and saved RL model
        """
        model_log_path = self.model_log_path
        model_names = []
        if os.path.exists(model_log_path):
            model_names = [f for f in os.listdir(model_log_path) if f.endswith('.zip')]

        model_numbers = []
        for names in model_names:
            match = re.search(r'\d+', names)
            if match:
                number = int(match.group())  # Convert the extracted string to an integer
                model_numbers.append(number)
        
        if model_numbers != []:
            best_model_path = self.model_log_path + f"best_model_after_{max(model_numbers)}.zip"
            model = PPO.load(best_model_path)
        else:
            model = 'no model yet'

        return model
    
    def get_pos_center_agent(self):
        """calculates the position of an agent at the center of the linear network depending on the number of nodes
        """
        return math.floor(self.nodes/2)

    def train_model(self):
        first_training = False # To keep track if it is the first time training the model 
        if self.do_training == 1:
            model = self.get_model() # always get model, because we always save model
            # callback_ = CheckpointCallback(save_freq=self.save_freq,
            #                                         save_path=self.model_log_path,
            #                                         name_prefix="rl_model")
            callback_ = SaveOnBestTrainingRewardCallback(check_freq = int(self.save_freq), log_dir = self.model_log_path, verbose=0)
            env = Environment(self.get_pos_center_agent(), self.nodes, self.t_cut, self.p, self.p_s, cc_effects=self.cc_effects)
            # setting the call back
            # eval_callback = EvalCallback(eval_env, best_model_save_path=self.model_log_path,
            #                  log_path=self.model_log_path, eval_freq=int(self.eval_freq),
            #                  deterministic=False, render=False)
            # always training
            if self.do_further_training == 1:
                if model == 'no model yet':
                    first_training = True
                    env = Monitor(env, filename=self.model_log_path)
                    model = PPO(policy = "MlpPolicy", env=env, seed=self.model_seed, verbose=self.agent_verbose, ent_coef=self.ent_reg) # reassign to untrained PPO model 
                else: # model already gotten at the start
                    model.set_env(env) # is this sent env really necessary?
                if self.callback == 1:
                    model.learn(total_timesteps=self.training_steps, progress_bar=True, callback=callback_)
                else:
                    model.learn(total_timesteps=self.training_steps, progress_bar=True)
            # only train if there is no model yet
            else:
                if model == 'no model yet':
                    first_training = True
                    env = Monitor(env, filename=self.model_log_path)
                    model = PPO(policy = "MlpPolicy", env=env, seed=self.model_seed, verbose=self.agent_verbose, ent_coef=self.ent_reg) # reassign to untrained PPO model 
                    if self.callback == 1:
                        model.learn(total_timesteps=self.training_steps, progress_bar=True, callback=callback_)
                    else:
                        model.learn(total_timesteps=self.training_steps, progress_bar=True)
                else:
                    # no training
                    pass
            # for saving the model
            if not os.path.exists(self.all_model_folder):
                os.makedirs(self.all_model_folder)
            model.save(self.model_path)

        # if first_training == True and self.cluster == 0: # only plot the learning curve of the first training sesh
        #     self.return_best_models_so_far_plot()
        #     plt.clf()

    def moving_average(self, x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    def plot_log_results(self, title="Average training reward"):
        """
        plot the results
        """
        x, y = ts2xy(load_results(self.model_log_path), "timesteps")
        if not x:
            assert False, f"ts2xy x is empty for nodes {self.nodes} t_cut {self.t_cut} p {self.p:.02f} p_s {self.p_s:.02f}"
        if not y:
            assert False, f"ts2xy y is empty for nodes {self.nodes} t_cut {self.t_cut} p {self.p:.02f} p_s {self.p_s:.02f}"
        window_length = math.ceil(len(x)/50)
        print(f'length = {len(x)}')
        # y_smooth = savgol_filter(y, window_length=math.ceil(window_length), polyorder=1) 
        y_smooth = self.moving_average(y, window_length)
        print(f'model log path = {self.model_log_path}')

        assert len(x) == len(y)
        # assert len(x) == len(y_smooth)

        fig = plt.figure(title)
        plt.plot(x, y, label='reward', color = 'blue', linewidth=0.9, alpha = 0.5)
        plt.plot(x, y_smooth, label='smoothened reward', color='orange')
        plt.yscale("linear")
        plt.xlabel("Number of training timesteps")
        plt.ylabel("Reward")
        plt.grid()
        ax = plt.gca()
        ax.set_ylim([1.5*np.min(y_smooth), np.max(y_smooth)+5])
        plt.legend()
        plt.savefig(self.model_log_path+'/training_curve.jpg', dpi=1200)
        plt.savefig(self.model_log_path+'/training_curve.pdf', dpi=1200)
        plt.clf()

    def moving_average(self, values, window):
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, "valid")
    
    def return_best_models_so_far(self):
        """getting the best trained model so far
        The model save steps don't necessarily match the evaluation steps. 
        """
        env = Environment(self.get_pos_center_agent(), self.nodes, self.t_cut, self.p, self.p_s, cc_effects=self.cc_effects)
        save_steps_arr = []
        best_reward_so_far_arr = []
        prev_reward = -int(env.max_moves) # lowest possible reward
        model_to_save = self.get_model_after_n_steps(self.save_freq) # the first save model
        monitor_steps_arr, reward_arr = ts2xy(load_results(self.model_log_path), "timesteps") # the monitored time steps and rewards
        print(monitor_steps_arr)
        for i in tqdm.tqdm(range(1,self.save_times)): # loop over all the time steps where a model has been saved. start at 1, because at checkpoint 0, 0 is always the best model
            save_step = int(i*self.save_freq)
            nearest_monitor_step, nearest_monitor_step_idx = self.find_nearest(monitor_steps_arr, save_step) # should actually find the nearest that is also lower
            # print(f'current step = {nearest_monitor_step}')
            reward = reward_arr[nearest_monitor_step_idx]
            # print(f'reward = {reward}')
            save_steps_arr.append(save_step)
            if reward > prev_reward:
                print(save_step)
                prev_reward = reward
                model_to_save = self.get_model_after_n_steps(save_step)
            best_reward_so_far_arr.append(prev_reward)
            if i> 1:
                assert best_reward_so_far_arr[-1] >= best_reward_so_far_arr[-2], f"{best_reward_so_far_arr[-1]} and {best_reward_so_far_arr[-2]}"
            model_name = self.all_model_folder+f'/PPO_cc_nodes_{nodes}_t_cut_{t_cut}_p_{p:.02f}_p_s_{p_s:.02f}/best_rl_model_after_{save_step}_steps.zip'
            model_to_save.save(model_name)

        print(best_reward_so_far_arr)

        print(f'rewards = {best_reward_so_far_arr}')
        self.plot_log_results(title="Average training reward")
        plt.plot(save_steps_arr, best_reward_so_far_arr,label='best model reward', color = 'black', linewidth=0.9)
        plt.legend()
        plt.savefig(self.model_log_path+'/training_curve_with_best_agent.jpg', dpi=1200)
        plt.clf()

    def find_nearest(self, array, value):
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx


    def return_best_models_so_far_plot(self):
        """getting the best trained model so far
        The model save steps don't necessarily match the evaluation steps. 
        """

        fig_ratio = (12.8, 4.8)
        # getting the check list of time steps where the best model was found and list of the corresponding average rewards
        try:
            with open(self.model_log_path+f'/best_model_check_points', "rb") as file:   # Unpickling
                best_model_check_points = pickle.load(file)
                # print(f'best model check point {best_model_check_points}')
            with open(self.model_log_path+f'/best_model_check_points_rewards', "rb") as file:   # Unpickling
                best_model_check_points_rewards = pickle.load(file)
                # print(f'best model check point rewards {best_model_check_points_rewards}')

            monitor_steps_arr, reward_arr = ts2xy(load_results(self.model_log_path), "timesteps")
            if np.array_equal(monitor_steps_arr,[]) or np.array_equal(reward_arr,[]): # is true if emtpy
                plt.figure(figsize=fig_ratio)
                plt.title("ts2xy couldn't load the needed date")
                if (not monitor_steps_arr):
                    plt.xlabel("monitor arr empty")
                if (not reward_arr):
                    plt.ylabel("reward arr empty")
                plt.savefig(self.model_log_path+'/training_curve_with_best_agent.jpg', dpi=400)
                plt.savefig(self.model_log_path+'/training_curve_with_best_agent.svg', dpi=400)
                # plt.clf()
                plt.close()
            else:
                window_length = math.ceil(len(reward_arr)/200)
                # reward_arr_smooth = savgol_filter(reward_arr, window_length=window_length, polyorder=3) 
                reward_arr_smooth = self.moving_average(reward_arr, window_length) 

                # filling in the best rewards to that we have the best reward so far for every training step
                current_idx = 0
                filled_best_model_check_points_rewards = []
                for i in range(self.training_steps):
                    # for jumping to the next best model once the training step has been reached
                    if i >= best_model_check_points[current_idx]:
                        current_idx += 1
                        if current_idx >= len(best_model_check_points):
                            current_idx = len(best_model_check_points)-1

                    # if no best model has been reached yet, give worst possible reward
                    if current_idx == 0:
                        best_reward_so_far = None # don't get te reward for the first model RL finds
                        filled_best_model_check_points_rewards.append(best_reward_so_far)
                    # other wise give reward of previous best model
                    else:
                        filled_best_model_check_points_rewards.append(best_model_check_points_rewards[current_idx-1])
                # best_model_check_points_idx_arr = [self.find_nearest(monitor_steps_arr, best_model_check_points[i])[1] for i in range(len(best_model_check_points))]
                # print(f'best model idx {best_model_check_points_idx_arr}')
                # best_model_rewards = [reward_arr[i] for i in best_model_check_points_idx_arr]
                # print(f'rewards = {best_model_rewards}')

                # fig = plt.figure()
                # fig, ax = plt.subplots()
                plt.rcParams.update({'font.size': 14})
                plt.figure(figsize=fig_ratio)
                plt.plot(monitor_steps_arr, reward_arr, label='cumulative reward', color='#5fbcd3ff', linewidth=0.9, alpha = 0.8)
                smooth_start = int((len(monitor_steps_arr)-len(reward_arr_smooth))/2)
                smooth_end = int((len(monitor_steps_arr)+len(reward_arr_smooth))/2)
                trimmed_monitor_steps_arr = monitor_steps_arr[smooth_start:smooth_end]
                assert len(trimmed_monitor_steps_arr) == len(reward_arr_smooth)
                plt.plot(trimmed_monitor_steps_arr, reward_arr_smooth, label='smoothened c. reward', color='#d38d5fff', linewidth=2)
                # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.yscale("linear")
                plt.xlabel("Number of training steps")
                plt.ylabel("Cumulative reward")
                plt.grid()
                ax = plt.gca()
                ax.set_ylim([1.5*np.min(reward_arr_smooth), np.max(reward_arr_smooth)+5])
                ax.set_xlim([trimmed_monitor_steps_arr[0], trimmed_monitor_steps_arr[-1]])
                plt.plot(list(range(self.training_steps)), filled_best_model_check_points_rewards, label='best model c. reward', color = 'black', linewidth=2)
                legend = plt.legend(loc='lower right', framealpha=1)
                plt.savefig(self.model_log_path+'/training_curve_with_best_agent.jpg', dpi=400)
                plt.savefig(self.model_log_path+'/training_curve_with_best_agent.svg', dpi=400)
                # plt.clf()
                plt.close()
        except FileNotFoundError: # Make empty plot when there are no best model {training steps} zip
            plt.figure(figsize=fig_ratio)
            plt.title("no saved best models zip files found")
            plt.savefig(self.model_log_path+'/training_curve_with_best_agent.jpg', dpi=400)
            plt.savefig(self.model_log_path+'/training_curve_with_best_agent.svg', dpi=400)
            # plt.clf()
            plt.close()

# this is for simulate a few episode and going through the specific policy. 
if __name__ == "__main__":
    # training params
    do_training = 1
    do_further_training = 0
    training_steps = int(3e5)

    # simulation params
    do_simulation = 0
    simulation_eps = int(2)

    nodes, t_cut_no_cc, p, p_s = 4, 2, 1, 1
    # nodes, t_cut_no_cc = 4, 2
    # p, p_s = 0.8, 0.6
    t_cut_cc = t_cut_no_cc*Quantum_network(nodes, t_cut_no_cc, p, p_s).t_cut_cc_multiplier() # this assumes agent is at the center of the network
    t_cut = t_cut_cc
    train_and_sim = training_and_policy_simulation(nodes, t_cut, p, p_s, 
                                                    do_training, do_further_training, training_steps,
                                                    do_simulation, simulation_eps,
                                                    ent_reg= 0.0001,
                                                    agent_verbose=1, training_version_=25,
                                                    callback=1, save_times=100)
    
    
    # model_names = train_and_sim.get_model()
    sim_dat = train_and_sim.do_training_and_simulation()
    # train_and_sim.simulate_policy_w_print()
    # action_plot = train_and_sim.action_plot(trials=20)
    # train_and_sim.return_best_models_so_far()
    train_and_sim.return_best_models_so_far_plot()
    # train_and_sim.plot_log_results()
