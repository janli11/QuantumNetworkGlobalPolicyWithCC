# To compare the partial RL agent to the WB RL agent
# Makes a plot similar to the Fig. 7 of the main text

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

p_list = np.linspace(1, 0.5, 6) # only the higher values, with lower delivery times
# p_s_list = [0.3, 0.5, 0.9] # for training versions 300, 305
p_s_list =  [1, 0.75, 0.5]

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

def load_InstantRL_data(nodes, t_cut, p, p_s, MC_runs, trained_version=203):
    current_path = os.getcwd()
    save_path = os.path.join(current_path, '../..', 'data/global_agent_swap_sum')

    version = 0
    
    t_cut_cc = t_cut*Quantum_network(nodes=nodes, t_cut=t_cut, p_s=p_s, p_e=p).t_cut_cc_multiplier()

    partial_data = []
    sim_dat_path = save_path+f'/env_a_alt_o_hist/sim_dat{trained_version}/sim_dat_cc_fidelity_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
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
            sim_dat_path = save_path+f'/env_a_alt_o_hist/sim_dat{trained_version}/sim_dat_cc_fidelity_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
    else: # when there's not simulation for these parameters yet
        print(f'no simulation data found for n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f} at training version {trained_version}')
        max_steps = int(1e6)
        partial_data = max_steps*np.ones(MC_runs)

    partial_data[:MC_runs]
    data = partial_data[:MC_runs]
    return data

def load_PartialRL_data(nodes, t_cut, p, p_s, MC_runs, trained_version=203):
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
            
            partial_data += sim_dat[0].tolist() # First index to get the regular time steps

            version += 1

            # Go to next version
            sim_dat_path = save_path+f'/env_cc_a_alt_o_hist/sim_dat{trained_version}/sim_dat_cc_fidelity_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
    else: # when there's not simulation for these parameters yet
        print(f'no simulation data found for n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f} at training version {trained_version}')
        max_steps = int(1e6)
        partial_data = max_steps*np.ones(MC_runs)

    partial_data[:MC_runs]
    data = partial_data[:MC_runs]
    return data

def load_swap_asap_data(setting, nodes, t_cut, p, p_s, MC_runs):
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
        partial_data += sim_dat[0].tolist()
        version += 1

        # preparing the path to load the next version
        if setting == 'swap-asap (cc effects)':
            sim_dat_path = save_path+f'/swap_asap_cc/sim_dat/sim_dat_cc_fidelity_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
        elif setting == 'swap-asap (no cc effects)':
            sim_dat_path = save_path+f'/swap_asap/sim_dat/sim_dat_cc_fidelity_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
        elif setting == 'swap-asap (predictive)':
            sim_dat_path = save_path+f'/swap_asap_with_pred/sim_dat/sim_dat_cc_fidelity_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'

    data = partial_data[:MC_runs]

    return data

def TComparison(nodes, t_cut, p, p_s, samples, training_version=203):
    TWBRL = np.array(load_InstantRL_data(nodes, t_cut, p, p_s, samples, training_version))*6
    avgTWBRL = np.average(TWBRL)
    TPartialRL = load_PartialRL_data(nodes, t_cut, p, p_s, samples, training_version)
    avgTPartialRL= np.average(TPartialRL)
    TSAWB = load_swap_asap_data('swap-asap (cc effects)', nodes, t_cut, p, p_s, samples)
    avgTSAWB = np.average(TSAWB)
    assert np.shape(TWBRL) == np.shape(TPartialRL) == np.shape(TSAWB) == MC_runs, f'shape mismatch: {np.shape(TWBRL)}, {np.shape(TPartialRL)}, {np.shape(TSAWB)}'

    policies = ['Partial RL', 'WB Swap-asap', 'WB Rl']
    deliveryTimes = [avgTPartialRL, avgTSAWB, avgTWBRL]
    colors = ['#d38d5fff',  #  Partial RL agent
    '#338000ff',  # Swap-asap WB (wait-for-broadcast)
    '#cc79a7ff']   # WB RL agent]

    plt.bar(policies, deliveryTimes, color=colors)
    plt.xlabel('Policies')
    plt.ylabel(r'$\mathbb{E}[T]$')
    plt.title(r'End-to-end Link Age at $p_s$'+f'={p_s}'+r' and $p$'+f'={p}')

    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')
    save_directory = data_path+f'/figures/RLComparison'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory) 
    plt.savefig(save_directory+f'/RLComparisonNodes{nodes}Tcut{t_cut}P{p}Ps{p_s}.pdf',dpi=1200)
    plt.savefig(save_directory+f'/RLComparisonNodes{nodes}Tcut{t_cut}P{p}Ps{p_s}.jpg',dpi=1200)

def combinedTComparison(nodes, t_cut, problist, samples):
    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')
    best_agent_directory = data_path+f'/bestAgents/'
    PartialRLavgTList = []
    WBRLavgTList = []
    SAWBavgTList = []
    for (p_s, p) in problist:
        bestAgentPath = best_agent_directory+f'bestAgentNodes{nodes}Tcut{t_cut}P{p:.2f}Ps{p_s:.2f}'
        with open(bestAgentPath, "rb") as f:
            bestAgent = pickle.load(f)
            print(f'Ps {p_s}, P {p}, best agent is {bestAgent}')

        TWBRL = np.array(load_InstantRL_data(nodes, t_cut, p, p_s, samples, trained_version=bestAgent))*6
        WBRLavgTList.append(np.average(TWBRL))
        TPartialRL = load_PartialRL_data(nodes, t_cut, p, p_s, samples, trained_version=bestAgent)
        PartialRLavgTList.append(np.average(TPartialRL))
        TSAWB = load_swap_asap_data('swap-asap (cc effects)', nodes, t_cut, p, p_s, samples)
        SAWBavgTList.append(np.average(TSAWB))

    # x-axis labels showing (p_s, p) for each group
    labels = [f'({p_s:.2f}, {p:.2f})' for (p_s, p) in problist]

    x = np.arange(len(labels))  # [0, 1, 2]
    width = 0.2

    fig, ax = plt.subplots()  

    # Bar plots: each shifted slightly so they don’t overlap
    ax.bar(x - width, PartialRLavgTList, width, label='RL agent', color='#d38d5fff')
    ax.bar(x, WBRLavgTList, width, label='WB RL agent', color='#cc79a7ff')
    ax.bar(x + width, SAWBavgTList, width, label='WB swap-asap', color='#cc79a7ff')

    # Label formatting
    ax.set_xlabel(r'$(p_s, p)$')
    ax.set_ylabel(r'$\mathbb{E}[T]$')
    ax.set_title(r'$\mathbb{E}[T]$ across different $(p_s, p)$ combinations')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')
    save_directory = data_path+f'/figures/figFidComp'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory) 
    plt.savefig(save_directory+f'/combFigFidCompNodes{nodes}Tcut{t_cut}Prob{problist}.pdf',dpi=1200)
    plt.savefig(save_directory+f'/combFigFidCompNodes{nodes}Tcut{t_cut}Prob{problist}.jpg',dpi=1200)

if __name__ == "__main__":
    ############################################
    # For generating fidelity comparison plot for each combination of p and p_s
    ############################################
    nodes = 4
    t_cut = 2
    p = 0.7
    p_s = 0.5
    samples = int(1e2-1)

    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')
    save_directory = data_path+f'/bestAgents/'

    for i, p_s in enumerate(p_s_list):
        for j, p in enumerate(p_list):

            bestAgentPath = save_directory+f'bestAgentNodes{nodes}Tcut{t_cut}P{p:.2f}Ps{p_s:.2f}'
            with open(bestAgentPath, "rb") as f:
                bestAgent = pickle.load(f)
                print(f'Ps {p_s}, P {p}, best agent is {bestAgent}')

            # # RL agent
            # t_cut_cc = t_cut*Quantum_network(nodes, t_cut, p, p_s).t_cut_cc_multiplier()
            # cc_train_and_sim = env_alt_hist_cc_train_sim(nodes, t_cut_cc, p, p_s, 
            #                                             do_training=0, do_further_training=0, training_steps=1,
            #                                             do_simulation=1, simulation_eps = samples, time_out_mult=2,
            #                                             new_training_version = 0, training_version_= bestAgent,
            #                                             callback=1)
            # sim_dat = cc_train_and_sim.simulate_policy_w_fidelity()

            # # WB swap asap
            # t_cut_cc = t_cut
            # sim = swap_asap_simulation_cc(nodes, t_cut_cc, p, p_s, simulation_eps=samples, time_out_mult=2)
            # sim_dat = sim.simulate_policy_w_fidelity()
            
            # # predictive swap asap
            # t_cut_cc = t_cut_cc*Quantum_network(nodes, t_cut_cc, p, p_s).t_cut_cc_multiplier() # this assumes agent is at the center of the network
            # sim = swap_asap_w_pred_simulation(nodes, t_cut_cc, p, p_s, simulation_eps=samples, time_out_mult=2)
            # sim_dat = sim.simulate_policy_w_fidelity()

            TComparison(nodes, t_cut, p, p_s, samples, training_version=bestAgent)

    ############################################
    # For plotting three comparisons in one figure
    ############################################
    problist = [(1, 1), (0.5, 0.9), (0.5, 0.5)]
    combinedTComparison(nodes, t_cut, problist, samples)