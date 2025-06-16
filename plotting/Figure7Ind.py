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

# trained_version_start, trained_version_end = 300, 305
# trained_version_start, trained_version_end = 615, 620
trained_version_start, trained_version_end = 20, 40
# plot params
MC_runs = int(5e3-1)
nodes = 4
t_cut = 2

p_list = np.linspace(1, 0.1, 10)
# p_s_list = [0.3, 0.5, 0.9] # for training versions 300, 305
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

# if __name__ == "__main__":
#     # looking for std anomalies 
#     # p_s = 0.5, p idx = 3, version = -19
#     # p_s = 0.5, p idx = 4, version = -18
#     # p_s = 0.75, p idx = 4, version = -13
#     # p_s = 1, p idx = 6, version = -18

#     data_RL = load_RL_data('RL hist (cc effects)')
#     for p_s in p_s_list:
#         expected_T_mult_versions = np.average(data_RL[p_s], axis=2)
#         idx_max_T = -1*np.argmax(-1*np.array(expected_T_mult_versions), axis=1)
#         print(f'best indices for ps = {p_s} is {idx_max_T}')


#     for p_s in p_s_list:
#         expected_T_mult_versions = np.average(data_RL[p_s], axis=2)
#         expected_T = -1*np.amax(-1*np.array(expected_T_mult_versions), axis=1)
#         expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
#         idx_max_T = -1*np.argmax(-1*np.array(expected_T_mult_versions), axis=1)
#         std_T = []
#         T_dat = []
#         assert len(idx_max_T) == len(p_list) # For each p_e, we have a best version
#         for p_idx, version_idx in enumerate(idx_max_T):
#             std_T.append(np.std(data_RL[p_s][p_idx][version_idx]/np.sqrt(MC_runs)))
#             T_dat.append([expected_T[p_idx] ,np.std(data_RL[p_s][p_idx][version_idx]/np.sqrt(MC_runs))])
#         print(f"standard deviation at ps = {p_s} is {std_T}")
#         print(f"T data at ps = {p_s} is {T_dat}")

#     print(data_RL[0.5][3][-19])
#     # print(data_RL[0.5][4][-18])
#     # print(data_RL[0.75][4][-13])
#     # print(data_RL[1][6][-18])






if __name__ == "__main__":
    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')

    # Load all the data
    data_RL = load_RL_data('RL hist (cc effects)')
    data_swap_asap = load_swap_asap_data('swap-asap (cc effects)')
    data_swap_asap_no_cc = load_swap_asap_data('swap-asap (no cc effects)')
    data_swap_asap_pred = load_swap_asap_data('swap-asap (predictive)')
    # data_rand_pol = load_swap_asap_data('random')

    # print(data_swap_asap[0.9].shape)

    #####################################
    # Make the plot
    #####################################
    fig, ax = plt.subplots(1, 3, figsize=(2*4.14,2.66), sharex=True, sharey=True)
    # fig, ax = plt.subplots(1, 3, figsize=style.figsize, sharex=True, sharey=True)

    # Ax 0 will have p_s = 0.9
    for i, p_s in enumerate(p_s_list):
        markersize = 3

        # swap asap no cc
        expected_T = np.average(data_swap_asap_no_cc[p_s], axis=1)
        expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
        ax[i].plot(p_list, expected_T, label='Instantaneous swap-asap',  marker='D', markersize=markersize, color='#5fbcd3ff')

        # naive swap asap with cc
        expected_T = np.average(data_swap_asap[p_s], axis=1)
        expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
        ax[i].plot(p_list, expected_T, label='WB swap-asap', marker='D', markersize=markersize, color='#338000ff') 

        # predictive swap asap with cc
        expected_T = np.average(data_swap_asap_pred[p_s], axis=1)
        expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
        ax[i].plot(p_list, expected_T, label='Predictive swap-asap', marker='D', markersize=markersize, color='#2c5aa0ff')

        # RL
        expected_T_mult_versions = np.average(data_RL[p_s], axis=2)
        expected_T = -1*np.amax(-1*np.array(expected_T_mult_versions), axis=1)
        expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
        ax[i].plot(p_list, expected_T, label='RL agent', marker='D', markersize=markersize, color='#d38d5fff')

        # lines to check optimal delivery time unit probability
        # if p_s == 1:
        #     ax[i].axhline(y=6, color='r', linestyle='-') # for checking swap asap vanilla at unit prob
        #     ax[i].axhline(y=5, color='b', linestyle='-') # optimal global agent policy
        #     ax[i].axhline(y=4, color='k', linestyle='-') # optimal global policy


        ax[i].set_title(r'$p_s$'+f'={p_s}')
        ax[i].set_xlabel(r'$p_{e}$')
        ax[i].set_ylabel(r'$\mathbb{E}[T]$')
        ax[i].set_yscale('log')
        ax[i].label_outer()

    handles, labels = ax[0].get_legend_handles_labels()  # Collect from the first subplot
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1))

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    # plt.subplots_adjust(top=0.6)

    save_directory = data_path+f'/figures/fig7'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory) 
    plt.savefig(save_directory+'/fig7.pdf',dpi=1200)
    plt.savefig(save_directory+'/fig7.jpg',dpi=1200)

    #####################################
    # Make the plot with error bars
    #####################################
    fig, ax = plt.subplots(1, 3, figsize=(2*4.14,2.66), sharex=True, sharey=True)
    # fig, ax = plt.subplots(1, 3, figsize=style.figsize, sharex=True, sharey=True)

    # Ax 0 will have p_s = 0.9
    RelMedianErrIntSA = []
    RelMedianErrWBSA = []
    RelMedianErrPredSA = []
    RelMedianErrRL = []
    for i, p_s in enumerate(p_s_list):
        markersize = 3

        # swap asap no cc
        expected_T = np.average(data_swap_asap_no_cc[p_s], axis=1)
        expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
        std_T = np.std(data_swap_asap_no_cc[p_s]/np.sqrt(MC_runs), axis=1)
        mask = ~np.isnan(expected_T)
        RelMedianErrIntSA.append(np.median(std_T[mask]/expected_T[mask]))
        ax[i].errorbar(p_list, expected_T, yerr = std_T,label='Instantaneous swap-asap',  marker='D', markersize=markersize, capsize=5, capthick=1, elinewidth=1, color='#5fbcd3ff')

        # naive swap asap with cc
        expected_T = np.average(data_swap_asap[p_s], axis=1)
        expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
        std_T = np.std(data_swap_asap[p_s]/np.sqrt(MC_runs), axis=1)
        mask = ~np.isnan(expected_T)
        RelMedianErrWBSA.append(np.median(std_T[mask]/expected_T[mask]))
        ax[i].errorbar(p_list, expected_T, yerr = std_T, label='WB swap-asap', marker='D', markersize=markersize, capsize=5, capthick=1, elinewidth=1, color='#338000ff') 

        # predictive swap asap with cc
        expected_T = np.average(data_swap_asap_pred[p_s], axis=1)
        expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
        std_T = np.std(data_swap_asap_pred[p_s]/np.sqrt(MC_runs), axis=1)
        mask = ~np.isnan(expected_T)
        RelMedianErrPredSA.append(np.median(std_T[mask]/expected_T[mask]))
        ax[i].errorbar(p_list, expected_T, yerr = std_T, label='Predictive swap-asap', marker='D', markersize=markersize, capsize=5, capthick=1, elinewidth=1, color='#2c5aa0ff')

        # RL
        expected_T_mult_versions = np.average(data_RL[p_s], axis=2)
        expected_T = -1*np.amax(-1*np.array(expected_T_mult_versions), axis=1)
        expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
        idx_max_T = -1*np.argmax(-1*np.array(expected_T_mult_versions), axis=1)
        std_T = []
        assert len(idx_max_T) == len(p_list) # For each p_e, we have a best version
        for p_idx, version_idx in enumerate(idx_max_T):
            std_T.append(np.std(data_RL[p_s][p_idx][version_idx]/np.sqrt(MC_runs)))
        print(f'RL std at ps = {p_s} is {std_T}')
        std_T = np.array(std_T)
        mask = ~np.isnan(expected_T)
        RelMedianErrRL.append(np.median(std_T[mask]/expected_T[mask]))
        ax[i].errorbar(p_list, expected_T, yerr = std_T, label='RL agent', marker='D', markersize=markersize, capsize=5, capthick=1, elinewidth=1, color='#d38d5fff')

        # # lines to check optimal delivery time unit probability
        # if p_s == 1:
        #     ax[i].axhline(y=6, color='r', linestyle='-') # for checking swap asap vanilla at unit prob
        #     ax[i].axhline(y=5, color='b', linestyle='-') # optimal global agent policy
        #     ax[i].axhline(y=4, color='k', linestyle='-') # optimal global policy


        ax[i].set_title(r'$p_s$'+f'={p_s}')
        ax[i].set_xlabel(r'$p_{e}$')
        ax[i].set_ylabel(r'$\mathbb{E}[T]$')
        ax[i].set_yscale('log')
        ax[i].label_outer()

    handles, labels = ax[0].get_legend_handles_labels()  # Collect from the first subplot
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1))

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    # plt.subplots_adjust(top=0.6)

    save_directory = data_path+f'/figures/fig7'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory) 
    plt.savefig(save_directory+'/fig7errbar.pdf',dpi=1200)
    plt.savefig(save_directory+'/fig7errbar.jpg',dpi=1200)

    print(f'prob order for p_s = {p_s_list}')
    print(f'Median error for Instantaneous swap-asap = {RelMedianErrIntSA}')
    print(f'Median error for WB swap-asap = {RelMedianErrWBSA}')
    print(f'Median error for predictive swap-asap = {RelMedianErrPredSA}')
    print(f'Median error for RL agent = {RelMedianErrRL}')

    #####################################
    # Make the relative plot
    #####################################
    fig, ax = plt.subplots(1, 3, figsize=(2*4.14,2.66), sharex=True, sharey=True)
    # fig, ax = plt.subplots(1, 3, figsize=style.figsize, sharex=True, sharey=True)

    # Ax 0 will have p_s = 0.9
    for i, p_s in enumerate(p_s_list):
        markersize = 3

        baseline = np.average(data_swap_asap_no_cc[p_s], axis=1)

        # swap asap no cc
        expected_T = np.average(data_swap_asap_no_cc[p_s], axis=1)
        expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
        ax[i].plot(p_list, expected_T/baseline, label='Instantaneous swap-asap',  marker='D', markersize=markersize, color='#5fbcd3ff')

        # naive swap asap with cc
        expected_T = np.average(data_swap_asap[p_s], axis=1)
        expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
        ax[i].plot(p_list, expected_T/baseline, label='WB swap-asap', marker='D', markersize=markersize, color='#338000ff') 

        # predictive swap asap with cc
        expected_T = np.average(data_swap_asap_pred[p_s], axis=1)
        expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
        ax[i].plot(p_list, expected_T/baseline, label='Predictive swap-asap', marker='D', markersize=markersize, color='#2c5aa0ff')

        # RL
        expected_T_mult_versions = np.average(data_RL[p_s], axis=2)
        expected_T = -1*np.amax(-1*np.array(expected_T_mult_versions), axis=1)
        expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
        ax[i].plot(p_list, expected_T/baseline, label='RL agent', marker='D', markersize=markersize, color='#d38d5fff')

        ax[i].set_title(r'$p_s$'+f'={p_s}')
        ax[i].set_xlabel(r'$p_{e}$')
        ax[i].set_ylabel(r'$\mathbb{E}[T]$')
        ax[i].set_yscale('log')
        ax[i].label_outer()

    handles, labels = ax[0].get_legend_handles_labels()  # Collect from the first subplot
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1))

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    # plt.subplots_adjust(top=0.6)

    save_directory = data_path+f'/figures/fig7'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory) 
    plt.savefig(save_directory+'/fig7_relative.pdf',dpi=1200)
    plt.savefig(save_directory+'/fig7_relative.jpg',dpi=1200)


# #############################
# # RL only plot
# ############################

# if __name__ == "__main__":
#     current_path = os.getcwd()
#     data_path = os.path.join(current_path, '../..', 'data')

#     # Load all the data
#     data_RL = load_RL_data('RL hist (cc effects)')


#     #####################################
#     # Make the plot
#     #####################################
#     fig, ax = plt.subplots(1, 3, figsize=(2*4.14,2.66), sharex=True, sharey=True)
#     # fig, ax = plt.subplots(1, 3, figsize=style.figsize, sharex=True, sharey=True)

#     # Ax 0 will have p_s = 0.9
#     for i, p_s in enumerate(p_s_list):
#         markersize = 3

#         # RL
#         expected_T_mult_versions = np.average(data_RL[p_s], axis=2)
#         expected_T = -1*np.amax(-1*np.array(expected_T_mult_versions), axis=1)
#         expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
#         ax[i].plot(p_list, expected_T, label='RL agent', marker='D', markersize=markersize, color='#d38d5fff')

#         # lines to check optimal delivery time unit probability
#         # if p_s == 1:
#         #     ax[i].axhline(y=6, color='r', linestyle='-') # for checking swap asap vanilla at unit prob
#         #     ax[i].axhline(y=5, color='b', linestyle='-') # optimal global agent policy
#         #     ax[i].axhline(y=4, color='k', linestyle='-') # optimal global policy


#         ax[i].set_title(r'$p_s$'+f'={p_s}')
#         ax[i].set_xlabel(r'$p_{e}$')
#         ax[i].set_ylabel(r'$\mathbb{E}[T]$')
#         ax[i].set_yscale('log')
#         ax[i].label_outer()

#     handles, labels = ax[0].get_legend_handles_labels()  # Collect from the first subplot
#     fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1))

#     plt.tight_layout(rect=[0, 0, 1, 0.9])
#     # plt.subplots_adjust(top=0.6)

#     save_directory = data_path+f'/figures/fig7'
#     if not os.path.exists(save_directory):
#         os.makedirs(save_directory) 
#     plt.savefig(save_directory+'/fig7.pdf',dpi=1200)
#     plt.savefig(save_directory+'/fig7.jpg',dpi=1200)

#     #####################################
#     # Make the plot with error bars
#     #####################################
#     fig, ax = plt.subplots(1, 3, figsize=(2*4.14,2.66), sharex=True, sharey=True)
#     # fig, ax = plt.subplots(1, 3, figsize=style.figsize, sharex=True, sharey=True)

#     # Ax 0 will have p_s = 0.9
#     RelMedianErrRL = []
#     for i, p_s in enumerate(p_s_list):
#         markersize = 3

#         # RL
#         expected_T_mult_versions = np.average(data_RL[p_s], axis=2)
#         expected_T = -1*np.amax(-1*np.array(expected_T_mult_versions), axis=1)
#         expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
#         idx_max_T = -1*np.argmax(-1*np.array(expected_T_mult_versions), axis=1)
#         std_T = []
#         assert len(idx_max_T) == len(p_list) # For each p_e, we have a best version
#         for p_idx, version_idx in enumerate(idx_max_T):
#             std_T.append(np.std(data_RL[p_s][p_idx][version_idx]/np.sqrt(MC_runs)))
#         print(f'RL std at ps = {p_s} is {std_T}')
#         std_T = np.array(std_T)
#         mask = ~np.isnan(expected_T)
#         RelMedianErrRL.append(np.median(std_T[mask]/expected_T[mask]))
#         ax[i].errorbar(p_list, expected_T, yerr = std_T, label='RL agent', marker='D', markersize=markersize, capsize=5, capthick=1, elinewidth=1, color='#d38d5fff')

#         # lines to check optimal delivery time unit probability
#         # if p_s == 1:
#         #     ax[i].axhline(y=6, color='r', linestyle='-') # for checking swap asap vanilla at unit prob
#         #     ax[i].axhline(y=5, color='b', linestyle='-') # optimal global agent policy
#         #     ax[i].axhline(y=4, color='k', linestyle='-') # optimal global policy


#         ax[i].set_title(r'$p_s$'+f'={p_s}')
#         ax[i].set_xlabel(r'$p_{e}$')
#         ax[i].set_ylabel(r'$\mathbb{E}[T]$')
#         ax[i].set_yscale('log')
#         ax[i].label_outer()

#     handles, labels = ax[0].get_legend_handles_labels()  # Collect from the first subplot
#     fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1))

#     plt.tight_layout(rect=[0, 0, 1, 0.9])
#     # plt.subplots_adjust(top=0.6)

#     save_directory = data_path+f'/figures/fig7'
#     if not os.path.exists(save_directory):
#         os.makedirs(save_directory) 
#     plt.savefig(save_directory+'/fig7errbar.pdf',dpi=1200)
#     plt.savefig(save_directory+'/fig7errbar.jpg',dpi=1200)

#     print(f'prob order for p_s = {p_s_list}')
#     print(f'Median error for RL agent = {RelMedianErrRL}')