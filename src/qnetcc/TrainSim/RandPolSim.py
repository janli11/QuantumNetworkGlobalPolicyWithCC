import os
from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network 
import os
import numpy as np
import pickle
import tqdm
import time

class rand_pol_simulation(object):
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
        self.max_steps = int(1e6)

        # paths
        self.abs_path = os.path.abspath(os.getcwd())
        self.save_path = '/Users/janli/PhD_projects/RL_quantum_network/data/global_agent_swap_sum/'# path for saving simulation data, Trained models, etc. 
        self.cluster = cluster
        if self.cluster == 1:
            self.abs_path = os.path.abspath(os.getcwd())
            self.save_path = '/home/lijt/data1/quantum_network'        
        self.sub_proj_rel_path = '/rand_pol'
        self.data_folder = self.save_path+self.sub_proj_rel_path+f'/sim_dat'
        self.file_name_template = f'/sim_dat_'+f'n_{self.nodes}_t_cut_{self.t_cut}_p_{self.p:.02f}_p_s_{self.p_s:.02f}'
        self.sim_dat_path_template = self.data_folder+self.file_name_template

    def simulate_policy(self):
        """simulating the swap asap policy
        """
        T_list = []
        micro_T_list = []

        start = time.time()
        for run in tqdm.tqdm(range(self.sim_eps),leave=False):
            quantum_network = Quantum_network(self.nodes,self.t_cut,self.p,self.p_s)
            while not quantum_network.A_B_entangled():
                node_actions = quantum_network.random_policy_actions()
                quantum_network.local_actions_update_network(node_actions)
                quantum_network.update_time_slots()
                stop = time.time()
                # don't allow entire simulation to take longer than certain amount of time
                if (stop-start) > self.sim_eps*self.time_out_mult:
                    break
            # if simulations takes to long, just make every episode length equal to max steps
            if (stop-start) > self.sim_eps*self.time_out_mult:
                T_list = [self.max_steps for i in range(self.sim_eps)]
                micro_T_list = [2*self.max_steps for i in range(self.sim_eps)]
                break

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
    nodes, t_cut, p, p_s, simulation_eps = 4, 16, 0.1, 0.1, int(1e2)
    swap_sim = rand_pol_simulation(nodes, t_cut, p, p_s, simulation_eps).simulate_policy()