# For Appendix I
# to see what the relative improvement of the RL agent is

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

# trained_version_start, trained_version_end = 300, 305
# trained_version_start, trained_version_end = 615, 620
trained_version_start, trained_version_end = 20, 40
# plot params
MC_runs = int(5e3-1)
nodes = 4
t_cut = 2

p_list = np.linspace(1, 0.1, 10)
p_s_list =  [0.5, 0.75, 1]

def load_RL_data(setting):
    current_path = os.getcwd()
    save_path = os.path.join(current_path, '../..', 'data/global_agent_swap_sum')

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

def load_swap_asap_data(setting):
    current_path = os.getcwd()
    save_path = os.path.join(current_path, '../..', 'data/global_agent_swap_sum')

    data = {} 
    for p_s in p_s_list:
        data[p_s] = -1*np.ones( shape=(len(p_list), MC_runs) )
    for d, p_s in enumerate(p_s_list):
        partial_data_vs_p = []
        for c, p in enumerate(p_list):
            t_cut_cc = t_cut
            if setting == "swap-asap (predictive)" or setting == 'swap-asap (no cc effects)' or setting == "random":
                t_cut_cc = t_cut*Quantum_network(nodes=nodes, t_cut=t_cut, p_s=p_s, p_e=p).t_cut_cc_multiplier()

            partial_data = []
            version = 0
            if setting == 'swap-asap (cc effects)':
                sim_dat_path = save_path+f'/swap_asap_cc/sim_dat/sim_dat_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
            elif setting == 'swap-asap (no cc effects)':
                sim_dat_path = save_path+f'/swap_asap/sim_dat/sim_dat_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
            elif setting == 'swap-asap (predictive)':
                sim_dat_path = save_path+f'/swap_asap_with_pred/sim_dat/sim_dat_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'

            if setting == "swap-asap (predictive)" and version == 0:
                assert os.path.exists(sim_dat_path) == True
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

            print(f'setting {setting}')
            print(f' n {nodes}, t_cut {t_cut_cc}, p {p:.02f}, p_s {p_s:.02f} version = {version}')
            partial_data_vs_p.append(np.array(partial_data[:MC_runs]))
            assert np.shape(partial_data[:MC_runs])[0] == MC_runs, f'not enough {setting} sim dat at n {nodes}, t_cut {t_cut_cc}, p {p:.02f}, p_s {p_s:.02f}, data shape {len(partial_data[:MC_runs])}, location {sim_dat_path}'
        data[p_s] = np.array(partial_data_vs_p)

    return data

def compute_relative_improvement(a, b, c):
    """Return list of relative improvements where a < min(b, c), else 0."""
    rel_improve = []
    for a_i, b_i, c_i in zip(a, b, c):
        ref = min(b_i, c_i)
        if a_i < ref:
            rel_improve.append((ref - a_i) / ref)
        else:
            rel_improve.append(0)
    return rel_improve

def compute_relative_improvement_signed(a, b, c):
    """Return list of relative improvements relative to min(b, c), signed."""
    rel_improve = []
    for a_i, b_i, c_i in zip(a, b, c):
        ref = min(b_i, c_i)
        rel_improve.append((ref - a_i) / ref)
    return rel_improve

def plot_improvements(rel_improve, p_list, p_s):
    """Plot the bar chart of relative improvements."""
    paired = sorted(zip(p_list, rel_improve), key=lambda pair: pair[0])
    sorted_ps, sorted_rel = zip(*paired)
    labels_str = [f"{ps:.2f}" for ps in sorted_ps]
    indices = range(len(rel_improve))
    plt.figure()
    plt.bar(indices, sorted_rel)
    plt.xticks(indices, labels_str, rotation=45, ha='right')
    plt.xticks(indices)
    plt.ylabel('Relative Improvement')
    plt.xlabel(r'$p_{e}$')
    plt.title('RL agent advantage at '+r'$p_s$'+f'={p_s}')
    plt.tight_layout()

    # Save the figure
    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')
    save_directory = data_path+f'/figures/figRelPer'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory) 
    plt.savefig(save_directory+f'/figRelPerPs{p_s}.pdf',dpi=1200)
    plt.savefig(save_directory+f'/figRelPerPs{p_s}.jpg',dpi=1200)

def plot_improvements_log(rel_improve, p_list, p_s):
    """Plot bar chart of relative improvements with a symmetric log y-axis."""
    paired = sorted(zip(p_list, rel_improve), key=lambda pair: pair[0])
    sorted_ps, sorted_rel = zip(*paired)
    labels_str = [f"{ps:.2f}" for ps in sorted_ps]
    indices = range(len(rel_improve))

    plt.figure()
    plt.bar(indices, sorted_rel)
    plt.yscale('symlog', linthresh=1e-3)  # Use linthresh to control linear range near 0
    plt.xticks(indices, labels_str, rotation=45, ha='right')
    plt.ylabel('Relative Improvement')
    plt.ylim(-2e1, 5e-1)
    plt.xlabel(r'$p_{e}$')
    # plt.title('RL agent advantage at ' + r'$p_s$' + f'={p_s}')
    plt.tight_layout()

    # Save the figure
    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')
    save_directory = os.path.join(data_path, 'figures/figRelPer')
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    plt.savefig(os.path.join(save_directory, f'figRelPerLogPs{p_s}.pdf'), dpi=1200)
    plt.savefig(os.path.join(save_directory, f'figRelPerLogPs{p_s}.jpg'), dpi=1200)


if __name__ == "__main__":
    # Load all the data
    data_RL = load_RL_data('RL hist (cc effects)')
    data_swap_asap = load_swap_asap_data('swap-asap (cc effects)')
    data_swap_asap_no_cc = load_swap_asap_data('swap-asap (no cc effects)')
    data_swap_asap_pred = load_swap_asap_data('swap-asap (predictive)')

    # Ax 0 will have p_s = 0.9
    for i, p_s in enumerate(p_s_list):

        # naive swap asap with cc
        SAWB_expected_T = np.average(data_swap_asap[p_s], axis=1)
        SAWB_expected_T = np.where(SAWB_expected_T >= 5*1e4, np.nan, SAWB_expected_T)
        # ax[i].plot(p_list, expected_T, label='WB swap asap', marker='D', markersize=markersize, color='#338000ff') 

        # predictive swap asap with cc
        SAPred_expected_T = np.average(data_swap_asap_pred[p_s], axis=1)
        SAPred_expected_T = np.where(SAPred_expected_T >= 5*1e4, np.nan, SAPred_expected_T)
        # ax[i].plot(p_list, expected_T, label='Predictive swap asap', marker='D', markersize=markersize, color='#2c5aa0ff')

        # RL
        RL_expected_T_mult_versions = np.average(data_RL[p_s], axis=2)
        RL_expected_T = -1*np.amax(-1*np.array(RL_expected_T_mult_versions), axis=1)
        RL_expected_T = np.where(RL_expected_T >= 5*1e4, np.nan, RL_expected_T)
        # ax[i].plot(p_list, expected_T, label='RL agent', marker='D', markersize=markersize, color='#d38d5fff')

        relAd = compute_relative_improvement_signed(RL_expected_T, SAWB_expected_T, SAPred_expected_T)
        plot_improvements(relAd, p_list, p_s)
        plot_improvements_log(relAd, p_list, p_s)


 