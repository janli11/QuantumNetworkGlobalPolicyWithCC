import os
import sys

from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network 
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tqdm
import math
import time
import random
from copy import deepcopy

class swap_asap_simulation(object):
    def __init__(self, nodes, t_cut, p, p_s, simulation_eps, time_out_mult=2,
                 cluster = 0):

        # network params
        self.nodes = nodes
        self.t_cut = t_cut
        self.p = p
        self.p_s = p_s

        # simulation params
        self.sim_eps = simulation_eps
        self.time_out_mult = time_out_mult 
        self.max_steps = int(10**5)

        # paths
        self.abs_path = os.path.abspath(os.getcwd())
        self.save_path = os.path.join(self.abs_path, '../../../..', 'data/global_agent_swap_sum')# path for saving simulation data, Trained models, etc. 
        self.cluster = cluster
        if self.cluster == 1:
            self.abs_path = os.path.abspath(os.getcwd())
            self.save_path = '/home/lijt/data1/quantum_network'        
        self.sub_proj_rel_path = '/swap_asap_with_pred'
        self.data_folder = self.save_path+self.sub_proj_rel_path+f'/sim_dat'
        self.file_name_template = f'/sim_dat_'+f'n_{self.nodes}_t_cut_{self.t_cut}_p_{self.p:.02f}_p_s_{self.p_s:.02f}'
        self.sim_dat_path_template = self.data_folder+self.file_name_template

    def simulate_policy(self):
        """simulating the swap asap policy with predictions
        After the prediction thinks end to end has been reached, don't do any thing for self.nodes time steps. This is 
        automatically enforced by the doing the swap asap actions using the prediction as the observation. Now if, the actual 
        quantum network doesn't have end to end entanglement, set the prediction to the actual network and continue again from there. 
        Returns:
            _type_: _description_
        """
        T_estimate = []
        T_list = []
        micro_T_list = []

        start = time.time()
        for run in tqdm.tqdm(range(self.sim_eps),leave=False):
            # Initialize two empty networks, one that represents the true network (quantum), and the other is for the representation for an 'agent' (who will base their actions on this network's state)
            # For example, ... 
            pred_network = Quantum_network(self.nodes, self.t_cut, self.p, self.p_s)
            quantum_network = Quantum_network(self.nodes, self.t_cut, self.p, self.p_s)

            # Set maximum runtime for this episode
            end_eps_point = int(10**5)
            # ...
            max_runs_for_avg_T = 0
            
            global_info_wait_timer = 0

            # As long as neither network has an end-to-end link...
            end_to_end_reach = False
            end_to_end_comm_time = 2*(self.nodes-1) # multiplied by 2 to convert time steps to rounds. n-1, because those are the number of segments
            while not ((quantum_network.A_B_entangled() and pred_network.A_B_entangled()) and global_info_wait_timer>=end_to_end_comm_time): 

                # If agent thinks end-to-end entanglement achieved...
                if pred_network.A_B_entangled(): # always wait for global information when end to end entangled
                    # ... wait until we are sure that the agent can also see the quantum network (i.e. has global info) (which happens always at timestep 2*N - 1)
                    
                    if global_info_wait_timer < end_to_end_comm_time: # need to (2*self.nodes)-1 timestep for global information, because the agents are local. (need to communicate from one end to the other)
                        global_info_wait_timer += 1
                    else:
                        assert global_info_wait_timer == end_to_end_comm_time
                        assert not quantum_network.A_B_entangled()
                        pred_network = deepcopy(quantum_network) # setting the prediction to the actual state. 
                        global_info_wait_timer = 0
                    
                # using the predicted network to get the swap actions    
                node_actions = pred_network.instant_comm_swap_asap_actions() 

                # if it thinks it is end-to-end entangled, don't perform actions that might destroy end-to-end link
                if pred_network.A_B_entangled():
                    if quantum_network.ent_gen_time_step():
                        assert node_actions[0] == 0, f'no EG at last segment'
                        assert node_actions[-1] == 0, f'no EG at first segment'
                pred_network.local_actions_update_network(node_actions) # update the predicted state using the selected action
                pred_network.update_time_slots()
                quantum_network.local_actions_update_network(node_actions) # applying the same actions on the actual quantum network 
                quantum_network.update_time_slots()

                if quantum_network.micro_time_slot > end_eps_point: # break episode if too large
                    end_to_end_reach = False
                    break
                
                stop = time.time()
                # don't allow entire simulation to take longer than certain amount of time
                if (stop-start) > self.sim_eps*self.time_out_mult:
                    break

            # if simulations took to long, just make every episode length equal to max steps
            if (stop-start) > self.sim_eps*self.time_out_mult:
                T_list = [self.max_steps for i in range(self.sim_eps)]
                micro_T_list = [2*self.max_steps for i in range(self.sim_eps)]
                break

            # We can get to this line either because both networks are end-to-end-entangled, or because the episode was too long or the timer ran out
            end_to_end_reach = (quantum_network.A_B_entangled() and pred_network.A_B_entangled())

            # assert self.sim_eps >= 10*max_runs_for_avg_T 
            if end_to_end_reach or run > max_runs_for_avg_T: # if end-to-end has been reached before the max time step or if enough samples have been collected to estimate the avg T
                # end_eps_point = 2*np.average(T_list)
                T_list.append(quantum_network.time_slot)
                micro_T_list.append(quantum_network.micro_time_slot)
        self.save_and_load_simulation_data(T_list, micro_T_list)

        # saving the delivery times (at different time scales) with their std

        average_regular_time = np.average(T_list)
        std_regular_time = np.std(T_list)/np.sqrt(self.sim_eps)
        print(f' average regular time = {average_regular_time} with std {std_regular_time}')
        average_micro_time = np.average(micro_T_list)
        std_micro_time = np.std(micro_T_list)/np.sqrt(self.sim_eps)
        print(f' average micro time = {average_micro_time} with std {std_micro_time}')


    def simulate_policy_w_print(self, seed):
        """simulating the swap asap policy with predictions
        After the prediction thinks end to end has been reached, don't do any thing for self.nodes time steps. This is 
        automatically enforced by the doing the swap asap actions using the prediction as the observation. Now if, the actual 
        quantum network doesn't have end to end entanglement, set the prediction to the actual network and continue again from there. 
        Returns:
            _type_: _description_
        """
        T_estimate = []
        T_list = []
        micro_T_list = []

        # setting the random seed to same eps each time
        random.seed(seed)
        np.random.seed(seed)

        start = time.time()
        for run in tqdm.tqdm(range(self.sim_eps),leave=False):
            # Initialize two empty networks, one that represents the true network (quantum), and the other is for the representation for an 'agent' (who will base their actions on this network's state)
            # For example, ... 
            pred_network = Quantum_network(self.nodes, self.t_cut, self.p, self.p_s)
            quantum_network = Quantum_network(self.nodes, self.t_cut, self.p, self.p_s)

            # Set maximum runtime for this episode
            end_eps_point = int(1e5)
            # ...
            max_runs_for_avg_T = 0
            
            global_info_wait_timer = 0

            # As long as neither network has an end-to-end link...
            end_to_end_reach = False
            end_to_end_comm_time = 2*(self.nodes-1) # multiplied by 2 to convert time steps to rounds. n-1, because those are the number of segments
            while not ((quantum_network.A_B_entangled() and pred_network.A_B_entangled()) and global_info_wait_timer>=end_to_end_comm_time): 

                # If agent thinks end-to-end entanglement achieved...
                if pred_network.A_B_entangled(): # always wait for global information when end to end entangled
                    # ... wait until we are sure that the agent can also see the quantum network (i.e. has global info) (which happens always at timestep 2*N - 1)
                    
                    if global_info_wait_timer < end_to_end_comm_time: # need to (2*self.nodes)-1 timestep for global information, because the agents are local. (need to communicate from one end to the other)
                        global_info_wait_timer += 1
                    else:
                        assert global_info_wait_timer == end_to_end_comm_time
                        assert not quantum_network.A_B_entangled()
                        pred_network = deepcopy(quantum_network) # setting the prediction to the actual state. 
                        global_info_wait_timer = 0
                    
                # using the predicted network to get the swap actions    
                node_actions = pred_network.instant_comm_swap_asap_actions() 

                # priting after actions are selected but before they are applied
                print('--------------------------------------------------')
                print(f'time step {quantum_network.time_slot}')
                print(f'action step {quantum_network.micro_time_slot%2}')
                print(f'actions: {node_actions}')

                print(f'predicted state: {pred_network.get_link_config()}')
                print(f'pred end to end: {pred_network.A_B_entangled()}')
                print(f'actual state: {quantum_network.get_link_config()}')
                print(f'state end to end: {quantum_network.A_B_entangled()}')
                # print(f'end eps {not ( quantum_network.A_B_entangled() and pred_network.A_B_entangled())}')
                print(f'wait for global info {pred_network.A_B_entangled() and not quantum_network.A_B_entangled()}')
                print(f'global info timer {global_info_wait_timer}')

                # if it thinks it is end-to-end entangled, don't perform actions that might destroy end-to-end link
                if pred_network.A_B_entangled():
                    if quantum_network.ent_gen_time_step():
                        assert node_actions[0] == 0, f'no EG at last segment'
                        assert node_actions[-1] == 0, f'no EG at first segment'
                pred_network.local_actions_update_network(node_actions) # update the predicted state using the selected action
                pred_network.update_time_slots()
                quantum_network.local_actions_update_network(node_actions) # applying the same actions on the actual quantum network 
                quantum_network.update_time_slots()

                if quantum_network.micro_time_slot > end_eps_point: # break episode if too large
                    end_to_end_reach = False
                    break
                
                stop = time.time()
                # don't allow entire simulation to take longer than certain amount of time
                if (stop-start) > self.sim_eps*self.time_out_mult:
                    break

            # if simulations took to long, just make every episode length equal to max steps
            if (stop-start) > self.sim_eps*self.time_out_mult:
                T_list = [self.max_steps for i in range(self.sim_eps)]
                micro_T_list = [2*self.max_steps for i in range(self.sim_eps)]
                break

            # We can get to this line either because both networks are end-to-end-entangled, or because the episode was too long or the timer ran out
            end_to_end_reach = (quantum_network.A_B_entangled() and pred_network.A_B_entangled())

            # assert self.sim_eps >= 10*max_runs_for_avg_T 
            if end_to_end_reach or run > max_runs_for_avg_T: # if end-to-end has been reached before the max time step or if enough samples have been collected to estimate the avg T
                # end_eps_point = 2*np.average(T_list)
                T_list.append(quantum_network.time_slot)
                micro_T_list.append(quantum_network.micro_time_slot)

                # printing one more time after end-to-end has been reached
                print('---------printing one more time after end-to-end has been reached----------')
                print(f'time step {quantum_network.time_slot}')
                print(f'action step {quantum_network.micro_time_slot%2}')
                print(f'actions: {node_actions}')
                print(f'predicted state: {pred_network.get_link_config()}')
                print(f'pred end to end: {pred_network.A_B_entangled()}')
                print(f'actual state: {quantum_network.get_link_config()}')
                print(f'state end to end: {quantum_network.A_B_entangled()}')
                # print(f'end eps {not ( quantum_network.A_B_entangled() and pred_network.A_B_entangled())}')
                print(f'wait for global info {pred_network.A_B_entangled() and not quantum_network.A_B_entangled()}')
                print(f'global info timer {global_info_wait_timer}')
        self.save_and_load_simulation_data(T_list, micro_T_list)

        # saving the delivery times (at different time scales) with their std

        average_regular_time = np.average(T_list)
        std_regular_time = np.std(T_list)/np.sqrt(self.sim_eps)
        print(f' average regular time = {average_regular_time} with std {std_regular_time}')
        average_micro_time = np.average(micro_T_list)
        std_micro_time = np.std(micro_T_list)/np.sqrt(self.sim_eps)
        print(f' average micro time = {average_micro_time} with std {std_micro_time}')
        
    def save_and_load_simulation_data(self, T_list, micro_T_list):
        """saving the simulated delivery times
        """

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

if __name__ == "__main__":
    nodes, t_cut, simulation_eps = 4, 8, int(1)
    p, p_s = 1, 1
    swap_sim = swap_asap_simulation(nodes, t_cut, p, p_s, simulation_eps).simulate_policy_w_print()