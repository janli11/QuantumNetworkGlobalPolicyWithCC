# file for retrieving the best RL versions at each parameter

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network 

from matplotlib import rc
import style
rc('font',**style.fontstyle)
rc('text', usetex=True)

# paths sim dat

trained_version_start, trained_version_end = 20, 40 # the the range of trained versions to be considered
MC_runs = int(5e3-1)
nodes = 4
t_cut = 2

p_list = np.linspace(1, 0.1, 10)
p_s_list = [0.5, 0.75, 1]

def load_RL_data(setting):
    save_path = '/home/lijt/data1/quantum_network'

    data = {} 
    # create entry in data dictionary for each value of p_s 
    for p_s in p_s_list:
        data[p_s] = -1*np.ones( shape=(len(p_list), MC_runs) )
    # Loop over the swap proabbilities
    for d, p_s in enumerate(p_s_list):
        partial_data_vs_p = []
        # Loop over the entanglement generation probabilites
        for c, p in enumerate(p_list):
            # loading all the different versions at fixed parameter
            partial_data_vs_p_at_trained_version_x = []
            for trained_version in range(trained_version_start, trained_version_end):
            
                partial_data = []
                
                # Version keeps track of several subsets (parts) of the full run
                version = 0
                
                # Construct full path to relevant dataset
                t_cut_cc = t_cut*Quantum_network(nodes=nodes, t_cut=t_cut, p_s=p_s, p_e=p).t_cut_cc_multiplier()

                sim_dat_path = save_path+f'/env_cc_a_alt_o_hist/sim_datInd{trained_version}/sim_dat_cc_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
                # If it exists...
                if os.path.exists(sim_dat_path) == True:
                    if version == 0:
                        assert os.path.exists(sim_dat_path) == True, f"no simulation data found for n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}"
                    while os.path.exists(sim_dat_path) == True:
                        # ... read the file
                        with open(sim_dat_path, "rb") as file: # Unpickling
                            sim_dat = pickle.load(file)
                        
                        partial_data += sim_dat[0].tolist() # First index to get the regular time steps

                        version += 1

                        # Go to next version
                        sim_dat_path = save_path+f'/env_cc_a_alt_o_hist/sim_datInd{trained_version}/sim_dat_cc_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
                else: # when there's not simulation for these parameters yet
                    print(f'no simulation data found for n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f} at training version {trained_version}')
                    max_steps = int(1e6)
                    partial_data = max_steps*np.ones(MC_runs)

                partial_data_vs_p_at_trained_version_x.append(partial_data[:MC_runs])
                print(f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f} at training version {trained_version}')
                print(f'lenght dat = {len(partial_data[:MC_runs])}')
                assert len(partial_data[:MC_runs]) == MC_runs
            print(f'shape = {np.shape(np.array(partial_data_vs_p_at_trained_version_x))} for n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}')
            partial_data_vs_p.append(np.array(partial_data_vs_p_at_trained_version_x))
        data[p_s] = np.array(partial_data_vs_p)

    return data

if __name__ == "__main__":
    # load RL data
    setting = 'RL hist (cc effects)'
    data_RL = load_RL_data(setting)

    save_path = '/home/lijt/data1/quantum_network/env_cc_a_alt_o_hist'
    save_directory = save_path+f'/bestAgents/'
    os.makedirs(save_directory, exist_ok=True)

    for i, p_s in enumerate(p_s_list):
        expected_T_mult_versions = np.average(data_RL[p_s], axis=2)
        bestAgentIndexPs = np.argmax(-1*np.array(expected_T_mult_versions), axis=1)
        for j, p in enumerate(p_list):
            bestAgent = bestAgentIndexPs[j]+trained_version_start
            print(f'best agent for p = {p:.2f} and p_s = {p_s:.2f} is {bestAgent}')
            with open(save_directory+f'bestAgentNodes{nodes}Tcut{t_cut}P{p:.2f}Ps{p_s:.2f}', "wb") as f:
                pickle.dump(bestAgent, f)

    


