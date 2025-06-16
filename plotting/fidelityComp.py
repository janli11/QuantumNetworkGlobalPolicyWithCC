# For addressing the comment of Ref 3
# Do faster policies also have higher fidelity?

# Just created the function simulate_policy_w_fidelity in MDPTrainSim.py
# Need to look into how to add the age of the end-to-end link into the saved data
# and rename the save data to sim_dat_fidelity
# need to check how to get the fidelity of the end-to-end link

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network 
from qnetcc.TrainSim.MDPTrainSim import training_and_policy_simulation as env_alt_hist_cc_train_sim
from qnetcc.TrainSim.SwapAsapInstantSim import swap_asap_simulation
from qnetcc.TrainSim.SwapAsapPredObsSim import swap_asap_simulation as swap_asap_w_pred_simulation
from qnetcc.TrainSim.SwapAsapVanillaSim import swap_asap_simulation as swap_asap_simulation_cc

from stable_baselines3 import PPO 
from stable_baselines3.common.utils import set_random_seed

from matplotlib import rc
import style
rc('font',**style.fontstyle)
rc('text', usetex=True)

# To address the comment from Ref 3. 
# What is the percentage increase of the RL agent over best swap-asap

trained_version_start, trained_version_end = 20, 40
# plot params
MC_runs = int(1e2-1)
nodes = 4
t_cut = 2
p_list   = np.linspace(1, 0.5, 6) # [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
p_s_list = [1, 0.75, 0.5]

def get_model(nodes, t_cut, p, p_s, training_steps, training_version=203):
    """Getting previously trained and saved RL model
    """

    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')
    model_path = data_path+'/global_agent_swap_sum/env_cc_a_alt_o_hist'+f'/Training{training_version}'+f'/PPO_cc_nodes_{nodes}_t_cut_{t_cut}_p_{p:.02f}_p_s_{p_s:.02f}/best_model_after_{training_steps}.zip'
    print(model_path)
    if os.path.exists(model_path):
        model = PPO.load(model_path)
    else:
        model = 'no model yet'
    return model

def load_RL_fid_data(nodes, t_cut, p, p_s, MC_runs, trained_version=203):
    current_path = os.getcwd()
    save_path = os.path.join(current_path, '../..', 'data/global_agent_swap_sum')

    version = 0
    
    t_cut_cc = t_cut*Quantum_network(nodes=nodes, t_cut=t_cut, p_s=p_s, p_e=p).t_cut_cc_multiplier()

    partial_data = []
    sim_dat_path = save_path+f'/env_cc_a_alt_o_hist/sim_dat{trained_version}/sim_dat_cc_fidelity_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
    # If it exists...
    if os.path.exists(sim_dat_path) == True:
        if version == 0:
            assert os.path.exists(sim_dat_path) == True, f"no simulation data found for n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}"
        while os.path.exists(sim_dat_path) == True:
            # ... read the file
            with open(sim_dat_path, "rb") as file: # Unpickling
                sim_dat = pickle.load(file)
            
            partial_data += sim_dat[-1].tolist() # First index to get the regular time steps

            version += 1

            # Go to next version
            sim_dat_path = save_path+f'/env_cc_a_alt_o_hist/sim_dat{trained_version}/sim_dat_cc_fidelity_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
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

def load_swap_asap_fid_data(setting, nodes, t_cut, p, p_s, MC_runs):
    current_path = os.getcwd()
    save_path = os.path.join(current_path, '../..', 'data/global_agent_swap_sum')

    if setting == "swap-asap (predictive)" or setting == 'swap-asap (no cc effects)' or setting == "random":
        t_cut_cc = t_cut*Quantum_network(nodes=nodes, t_cut=t_cut, p_s=p_s, p_e=p).t_cut_cc_multiplier()
    else:
        t_cut_cc = t_cut

    partial_data = []
    version = 0
    if setting == 'swap-asap (cc effects)':
        sim_dat_path = save_path+f'/swap_asap_cc/sim_dat/sim_dat_cc_fidelity_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
    elif setting == 'swap-asap (no cc effects)':
        sim_dat_path = save_path+f'/swap_asap/sim_dat/sim_dat_cc_fidelity_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
    elif setting == 'swap-asap (predictive)':
        sim_dat_path = save_path+f'/swap_asap_with_pred/sim_dat/sim_dat_cc_fidelity_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'

    if setting == "swap-asap (predictive)" and version == 0:
        assert os.path.exists(sim_dat_path) == True, f"no simulation data found for n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f} at {sim_dat_path}"
    while os.path.exists(sim_dat_path) == True:
        with open(sim_dat_path, "rb") as file: # Unpickling
            sim_dat = pickle.load(file)
        partial_data += sim_dat[-1].tolist()
        version += 1

        # preparing the path to load the next version
        if setting == 'swap-asap (cc effects)':
            sim_dat_path = save_path+f'/swap_asap_cc/sim_dat/sim_dat_cc_fidelity_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
        elif setting == 'swap-asap (no cc effects)':
            sim_dat_path = save_path+f'/swap_asap/sim_dat/sim_dat_cc_fidelity_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
        elif setting == 'swap-asap (predictive)':
            sim_dat_path = save_path+f'/swap_asap_with_pred/sim_dat/sim_dat_cc_fidelity_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'

    data = partial_data[:MC_runs]
    assert len(data) == MC_runs, f"data length {len(data)} does not match MC_runs {MC_runs}"
    assert not np.isnan(data).any(), "Array contains NaN"
    assert all(isinstance(x, (int, float)) and not np.isnan(x) for x in data)

    return data

def fideliyComparison(nodes, t_cut, p, p_s, samples, training_version=203):
    RLFid = load_RL_fid_data(nodes, t_cut, p, p_s, samples, training_version)
    RLMeanFid = np.average(RLFid)
    assert RLMeanFid <= t_cut*Quantum_network(nodes=nodes, t_cut=t_cut, p_s=p_s, p_e=p).t_cut_cc_multiplier(), f"fidelity has to be smaller than t_cut but is {RLFid}"
    SwapAsapWBFid = load_swap_asap_fid_data('swap-asap (cc effects)', nodes, t_cut, p, p_s, samples)
    SAWBMeanFid = np.average(SwapAsapWBFid)
    SwapAsapPredFid = load_swap_asap_fid_data('swap-asap (predictive)', nodes, t_cut, p, p_s, samples)
    SwapAsapPredMeanFid = np.average(SwapAsapPredFid)

    policies = ['RL', 'Swap-asap WB', 'swap-asap Predictive']
    fidelities = [RLMeanFid, SAWBMeanFid, SwapAsapPredMeanFid]
    colors = ['#d38d5fff',  # RL agent
    '#338000ff',  # Swap-asap WB (wait-for-broadcast)
    '#2c5aa0ff']   # swap-asap Predictive]

    plt.bar(policies, fidelities, color=colors)
    plt.xlabel('Policies')
    plt.ylabel('Average Link Age')
    plt.title(r'End-to-end Link Age at $p_s$'+f'={p_s}'+r' and $p$'+f'={p}')

    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')
    save_directory = data_path+f'/figures/figFidComp'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory) 
    plt.savefig(save_directory+f'/figFidCompNodes{nodes}Tcut{t_cut}P{p}Ps{p_s}.pdf',dpi=1200)
    plt.savefig(save_directory+f'/figFidCompNodes{nodes}Tcut{t_cut}P{p}Ps{p_s}.jpg',dpi=1200)

def combinedFideliyComparison(nodes, t_cut, problist, samples):
    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')
    best_agent_directory = data_path+f'/bestAgents/'
    RLMeanFidList = []
    SAWBMeanFidList = []
    SwapAsapPredMeanFidList = []
    for (p_s, p) in problist:
        bestAgentPath = best_agent_directory+f'bestAgentNodes{nodes}Tcut{t_cut}P{p:.2f}Ps{p_s:.2f}'
        with open(bestAgentPath, "rb") as f:
            bestAgent = pickle.load(f)
            print(f'Ps {p_s}, P {p}, best agent is {bestAgent}')

        RLFid = load_RL_fid_data(nodes, t_cut, p, p_s, samples, bestAgent)
        RLMeanFidList.append(np.average(RLFid))
        SwapAsapWBFid = load_swap_asap_fid_data('swap-asap (cc effects)', nodes, t_cut, p, p_s, samples)
        SAWBMeanFidList.append(np.average(SwapAsapWBFid))
        SwapAsapPredFid = load_swap_asap_fid_data('swap-asap (predictive)', nodes, t_cut, p, p_s, samples)
        SwapAsapPredMeanFidList.append(np.average(SwapAsapPredFid))

    # x-axis labels showing (p_s, p) for each group
    labels = [f'{p:.2f}' for (p_s, p) in problist]

    x = np.arange(len(labels))  # [0, 1, 2]
    width = 0.2

    fig, ax = plt.subplots()  

    # Bar plots: each shifted slightly so they don’t overlap
    ax.bar(x - width, RLMeanFidList, width, label='RL agent', color='#d38d5fff')
    ax.bar(x, SAWBMeanFidList, width, label='WB swap-asap', color='#338000ff')
    ax.bar(x + width, SwapAsapPredMeanFidList, width, label='Predictive swap-asap', color='#2c5aa0ff')

    # Label formatting
    ax.set_xlabel(r'$p_e$')
    ax.set_ylabel('End-to-end link age')
    p_s = problist[0][0]  # Assuming p_s is the same for all combinations in this plot
    ax.set_title(r'End-to-end Link age for $p_s$='+f'{p_s}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')
    save_directory = data_path+f'/figures/figFidComp'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory) 
    prob_str = "_".join(str(val) for tup in problist for val in tup)
    plt.savefig(save_directory+f'/combFigFidCompNodes{nodes}Tcut{t_cut}Prob{prob_str}.pdf',dpi=1200)
    plt.savefig(save_directory+f'/combFigFidCompNodes{nodes}Tcut{t_cut}Prob{prob_str}.jpg',dpi=1200)

    
if __name__ == "__main__":
    ############################################
    # For generating on fidelity comparison plot for each combination of p and p_s
    ############################################
    nodes = 4
    t_cut = 2
    p = 0.7
    p_s = 0.5
    samples = int(1e2-1)

    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')
    save_directory = data_path+f'/bestAgents/'

    # for i, p_s in enumerate(p_s_list):
    #     for j, p in enumerate(p_list):

    #         bestAgentPath = save_directory+f'bestAgentNodes{nodes}Tcut{t_cut}P{p:.2f}Ps{p_s:.2f}'
    #         with open(bestAgentPath, "rb") as f:
    #             bestAgent = pickle.load(f)
    #             print(f'Ps {p_s}, P {p}, best agent is {bestAgent}')

    #         fideliyComparison(nodes, t_cut, p, p_s, samples, training_version=bestAgent)

    ############################################
    # For plotting three comparisons in one figure
    ############################################

    grouped_tuples = [[(ps, p) for p in p_list] for ps in p_s_list] # grouping the probabiliyt tuples by p_s
    for i in range(len(grouped_tuples)):
        problist = grouped_tuples[i]
        combinedFideliyComparison(nodes, t_cut, problist, samples)



