# For figure 7 of the main text

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

trained_version_start, trained_version_end = 20, 40
# plot params
MC_runs = int(5e3-1)
nodes = 4
t_cut = 2
p = 1
p_s = 1

def load_RL_sim_data(nodes, t_cut, p, p_s, MC_runs, trained_version=203):
    current_path = os.getcwd()
    save_path = os.path.join(current_path, '../..', 'data/global_agent_swap_sum')

    version = 0
    
    t_cut_cc = t_cut*Quantum_network(nodes=nodes, t_cut=t_cut, p_s=p_s, p_e=p).t_cut_cc_multiplier()

    partial_data = []
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

    partial_data[:MC_runs]
    data = partial_data[:MC_runs]
    assert len(data) == MC_runs, f"data length {len(data)} does not match MC_runs {MC_runs}"
    assert not np.isnan(data).any(), "Array contains NaN"
    assert all(isinstance(x, (int, float)) and not np.isnan(x) for x in data)

    return data

def load_swap_asap_sim_data(setting, nodes, t_cut, p, p_s, MC_runs):
    current_path = os.getcwd()
    save_path = os.path.join(current_path, '../..', 'data/global_agent_swap_sum')

    if setting == "swap-asap (predictive)" or setting == 'swap-asap (no cc effects)' or setting == "random":
        t_cut_cc = t_cut*Quantum_network(nodes=nodes, t_cut=t_cut, p_s=p_s, p_e=p).t_cut_cc_multiplier()
    else:
        t_cut_cc = t_cut

    partial_data = []
    version = 0
    if setting == 'swap-asap (cc effects)':
        sim_dat_path = save_path+f'/swap_asap_cc/sim_dat/sim_dat_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
    elif setting == 'swap-asap (no cc effects)':
        sim_dat_path = save_path+f'/swap_asap/sim_dat/sim_dat_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
    elif setting == 'swap-asap (predictive)':
        sim_dat_path = save_path+f'/swap_asap_with_pred/sim_dat/sim_dat_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'

    if setting == "swap-asap (predictive)" and version == 0:
        assert os.path.exists(sim_dat_path) == True, f"no simulation data found for n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f} at {sim_dat_path}"
    while os.path.exists(sim_dat_path) == True:
        with open(sim_dat_path, "rb") as file: # Unpickling
            sim_dat = pickle.load(file)
        partial_data += sim_dat[0].tolist()
        version += 1

        # preparing the path to load the next version
        if setting == 'swap-asap (cc effects)':
            sim_dat_path = save_path+f'/swap_asap_cc/sim_dat/sim_dat_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
        elif setting == 'swap-asap (no cc effects)':
            sim_dat_path = save_path+f'/swap_asap/sim_dat/sim_dat_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
        elif setting == 'swap-asap (predictive)':
            sim_dat_path = save_path+f'/swap_asap_with_pred/sim_dat/sim_dat_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'

    data = partial_data[:MC_runs]
    assert len(data) == MC_runs, f"data length {len(data)} does not match MC_runs {MC_runs}"
    assert not np.isnan(data).any(), "Array contains NaN"
    assert all(isinstance(x, (int, float)) and not np.isnan(x) for x in data)

    return data


if __name__ == "__main__":
    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')

    # Load all the data
    data_RL = load_RL_sim_data(nodes, t_cut, p, p_s, MC_runs, trained_version=25)
    data_swap_asap = load_swap_asap_sim_data('swap-asap (cc effects)', nodes, t_cut, p, p_s, MC_runs)
    data_swap_asap_pred = load_swap_asap_sim_data('swap-asap (predictive)', nodes, t_cut, p, p_s, MC_runs)

    print('WB swap-asap: ',np.average(data_swap_asap))
    print('Pres swap-asap: ', np.average(data_swap_asap_pred))
    # print('RL data', data_RL)
    print('RL :',np.average(data_RL))
