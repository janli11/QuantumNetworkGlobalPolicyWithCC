import os

from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network 
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

# importing the various environments to be compared with each other
from qnetcc.TrainSim.MDPTrainSim import training_and_policy_simulation as env_alt_hist_cc_train_sim
from qnetcc.TrainSim.SwapAsapInstantSim import swap_asap_simulation
from qnetcc.TrainSim.SwapAsapPredObsSim import swap_asap_simulation as swap_asap_w_pred_simulation
from qnetcc.TrainSim.SwapAsapVanillaSim import swap_asap_simulation as swap_asap_simulation_cc

# paths for saving things
abs_path = os.getcwd()
save_path = os.path.join(abs_path, '..', 'data')
proj_fig_folder = '/figures'

############################################################################
# The training and simulation part
############################################################################

# setting of the script 
# Running this script will produce plots of the delivery time vs the EG succes probability at fixed values for
# the number of nodes, swap succes probability and cut-off time. 

# training params 
do_training, do_further_training, training_steps, train_new_model = 1, 0, int(1e6), 0
# trained_versions_start, trained_versions_stop = 0, 5 # locally trained models
trained_versions_start, trained_versions_stop = 40, 45 # cluster trained models
# trained_versions_start, trained_versions_stop = 100, 101 # To estimtate how long training takes
# simulation params
do_simulation, simulation_eps = 0, int(1e3)
MC_runs = int(simulation_eps)
time_out_mult = 2 # how many seconds and episode on average is allowed to take, before the Monte Carlo simulation is aborted
# Callback
Callback=1 

# Quantum Network parameters
nodes_list = [4] 
t_cut_list = [2] # t_cut is this factor multiplied by the number of nodes, this is for taking the sum
# p_list = np.linspace(1, 0.4, 7)
# p_s_list = np.linspace(1, 0.4, 7)
# p_list = [1, 0.9, 0.8, 0.6, 0.5, 0.4]
# p_s_list = [1, 0.9, 0.8, 0.6, 0.5, 0.4]

# local trained model parameters
# p_list = [1, 0.8, 0.6, 0.4]
# p_s_list = [1, 0.8, 0.6, 0.4]

# HPC trained model parameters
# p_list = np.linspace(1, 0.1, 10)
p_list = [1, 0.7, 0.5]
# p_s_list = [1, 0.9, 0.8, 0.7, 0.5]
p_s_list = [1, 0.75, 0.5]
# p_s_list = [0.5]

if __name__ == "__main__":

    scenario_list = ['RL', 'swap-asap (vanilla)', 'swap-asap (instant)', 'swap-asap (predictive)']
    # scenario_list = ['RL']
    # scenario_list = ['swap-asap (vanilla)']
    # scenario_list = ['swap-asap (cc effects)', 'swap-asap (no cc effects)', 'swap-asap (predictive)']
    # scenario_list = ['swap-asap (cc effects)']
    # scenario_list = ['RL hist (cc effects)', 'swap-asap (predictive)']
    # scenario_list = ['swap-asap (cc effects)', 'swap-asap (no cc effects)']

    # RL alt hist WITH CC
    slow_policy_list = []
    if 'RL' in scenario_list:
        for nodes in nodes_list:
            for t_cut_no_cc in t_cut_list:
                for p in p_list:
                    for p_s in p_s_list:
                        print(f'alt history cc')
                        t_cut_cc = t_cut_no_cc*Quantum_network(nodes, t_cut_no_cc, p, p_s).t_cut_cc_multiplier() # this assumes agent is at the center of the network
                        print(f'nodes {nodes}, t_cut {t_cut_cc}, p_e {p}, p_s {p_s}')
                        # if we're only simulating without training
                        if do_training == 0 and do_further_training ==0 and do_simulation == 1: 
                            for training_verion in range(trained_versions_start, trained_versions_stop):
                                print(f'training version = {training_verion}')
                                cc_train_and_sim = env_alt_hist_cc_train_sim(nodes, t_cut_cc, p, p_s, 
                                                                                do_training, do_further_training, training_steps,
                                                                                do_simulation, simulation_eps, time_out_mult,
                                                                                new_training_version = train_new_model, training_version_=training_verion,
                                                                                callback=1)
                                sim_dat = cc_train_and_sim.do_training_and_simulation()
                        else:
                            for training_verion in range(trained_versions_start, trained_versions_stop):
                                print(f'training version = {training_verion}')
                                cc_train_and_sim = env_alt_hist_cc_train_sim(nodes, t_cut_cc, p, p_s, 
                                                                                do_training, do_further_training, training_steps,
                                                                                do_simulation, simulation_eps, time_out_mult,
                                                                                new_training_version = train_new_model, training_version_=training_verion,
                                                                                callback=1)
                                sim_dat = cc_train_and_sim.do_training_and_simulation()
                        # make the best model so far plot
                        cc_train_and_sim.return_best_models_so_far_plot()

    # swap asap WIHTOUT CC
    if 'swap-asap (instant)' in scenario_list:
        for nodes in nodes_list:
            for t_cut_no_cc in t_cut_list:
                for p in p_list:
                    for p_s in p_s_list:
                        print(f'swap asap no cc')
                        print(f'nodes {nodes}, t_cut {t_cut_no_cc}, p_e {p}, p_s {p_s}')
                        sim = swap_asap_simulation(nodes, t_cut_no_cc, p, p_s, simulation_eps, time_out_mult=time_out_mult)
                        start = time.time()
                        if do_simulation == 1:
                            sim_dat = sim.simulate_policy()
                        stop =time.time()
                        print(f'no cc simulation time = {stop-start}')

    # swap asap with CC
    if 'swap-asap (vanilla)' in scenario_list:
        for nodes in nodes_list:
            for t_cut_no_cc in t_cut_list:
                for p in p_list:
                    for p_s in p_s_list:
                        print(f'swap asap cc')
                        print(f'nodes {nodes}, t_cut {t_cut_no_cc}, p_e {p}, p_s {p_s}')
                        sim = swap_asap_simulation_cc(nodes, t_cut_no_cc, p, p_s, simulation_eps, time_out_mult=time_out_mult)
                        start = time.time()
                        if do_simulation == 1:
                            sim_dat = sim.simulate_policy()
                        stop =time.time()
                        print(f'cc simulation time = {stop-start}')

    # swap asap with cc using PREDICTED observations 
    if 'swap-asap (predictive)' in scenario_list:
        for nodes in nodes_list:
            for t_cut_no_cc in t_cut_list:
                for p in p_list:
                    for p_s in p_s_list:
                        print(f'swap asap with pred')
                        t_cut_cc = t_cut_no_cc*Quantum_network(nodes, t_cut_no_cc, p, p_s).t_cut_cc_multiplier() # this assumes agent is at the center of the network
                        print(f'nodes {nodes}, t_cut {t_cut_cc}, p_e {p}, p_s {p_s}')
                        sim = swap_asap_w_pred_simulation(nodes, t_cut_cc, p, p_s, simulation_eps, time_out_mult=time_out_mult)
                        start = time.time()
                        if do_simulation == 1:
                            sim_dat = sim.simulate_policy()
                            stop =time.time()
                            print(f'pred simulation time = {stop-start}')

    ############################################################################
    # plotting the simulation data of the different policies of different MDPs
    ############################################################################

    #########################################################################
    # loading simulation data of various policies into an array each, 
    # each dimension corresponds to a different param: nodes, t_cut, p, p_s, MC_runs

    # data shape for the simulation data of various policies
    sim_dat_shape = (len(nodes_list),len(t_cut_list),len(p_list),len(p_s_list),MC_runs)

    trained_versions = trained_versions_stop-trained_versions_start
    RL_sim_dat_shape = (trained_versions, len(nodes_list),len(t_cut_list),len(p_list),len(p_s_list),MC_runs)

    # loading RL alt hist data of case WITH CC
    if 'RL' in scenario_list:
        sim_dat_cc_alt_hist_arr = -1*np.ones((RL_sim_dat_shape))
        for x, trained_verion in enumerate(range(trained_versions_start, trained_versions_stop)):
            for a, nodes in enumerate(nodes_list):
                for b, t_cut_no_cc in enumerate(t_cut_list):
                    for c, p in enumerate(p_list):
                        for d, p_s in enumerate(p_s_list):
                            # loading all the different versions at fixed parameter
                            t_cut_cc = t_cut_no_cc*Quantum_network(nodes, t_cut_no_cc, p, p_s).t_cut_cc_multiplier() # this assumes agent is at the center of the network
                            train_and_sim = env_alt_hist_cc_train_sim(nodes, t_cut_cc, p, p_s, 
                                                                    do_training, do_further_training, training_steps,
                                                                    do_simulation, simulation_eps, time_out_mult, 
                                                                    new_training_version=train_new_model, training_version_ = trained_verion,
                                                                    callback=1)
                            sim_dat_cc_alt_hist = []
                            version = 0
                            sim_dat_path = train_and_sim.sim_dat_path_template+f'_v{version}.pkl'
                            if os.path.exists(sim_dat_path): # if there is at least on simulation data file
                                while os.path.exists(sim_dat_path) == True:
                                    with open(sim_dat_path, "rb") as file: # Unpickling
                                        sim_dat = pickle.load(file)
                                    sim_dat_cc_alt_hist += sim_dat[0].tolist() # 1 to get the regular time steps
                                    version += 1
                                    sim_dat_path = train_and_sim.sim_dat_path_template+f'_v{version}.pkl'
                                assert sim_dat_cc_alt_hist != [], f"simulation data is empty for nodes {nodes}, t_cut {t_cut_cc}, p_e {p}, p_s {p_s}"

                                # putting the data at fixed nodes, t_cut, p, p_s into one array
                                if len(sim_dat_cc_alt_hist) < MC_runs:
                                    print(f"not enough RL sim dat for nodes {nodes}, t_cut {t_cut_cc}, p_e {p}, p_s {p_s}")
                                    sim_dat_cc_alt_hist_arr[x,a,b,c,d] = [int(1e7) for i in range(MC_runs)]
                                else:
                                    sim_dat_cc_alt_hist_arr[x,a,b,c,d] = sim_dat_cc_alt_hist[:MC_runs] 
                            else: # This is when there is no simulation data yet! Just set it to some maximum that will be filtered out in the plots anyways
                                print(f"no valid RL sim dat for nodes {nodes}, t_cut {t_cut_cc}, p_e {p}, p_s {p_s}")
                                sim_dat_cc_alt_hist_arr[x,a,b,c,d] = [int(1e7) for i in range(MC_runs)]


    # loading data from swap asap WIHTOUT CC
    if 'swap-asap (instant)' in scenario_list:
        sim_dat_swap_asap_arr = -1*np.ones((sim_dat_shape))
        for a, nodes in enumerate(nodes_list):
            for b, t_cut_no_cc in enumerate(t_cut_list):
                for c, p in enumerate(p_list):
                    for d, p_s in enumerate(p_s_list):
                        # loading all the different versions at fixed parameter
                        sim = swap_asap_simulation(nodes, t_cut_no_cc, p, p_s, simulation_eps, time_out_mult=time_out_mult)
                        sim_dat_swap_asap = []
                        version = 0
                        sim_dat_path = sim.sim_dat_path_template+f'_v{version}.pkl'
                        assert os.path.exists(sim_dat_path), f"no sim dat file found for {nodes}, t_cut {t_cut_no_cc}, p_e {p}, p_s {p_s} at path {sim_dat_path}"
                        while os.path.exists(sim_dat_path) == True:
                            with open(sim_dat_path, "rb") as file: # Unpickling
                                sim_dat = pickle.load(file)
                            sim_dat_swap_asap += sim_dat[0].tolist() # 1 to get the regular time steps
                            version += 1
                            sim_dat_path = sim.sim_dat_path_template+f'_v{version}.pkl'

                        # putting the data at fixed nodes, t_cut, p, p_s into one array
                        sim_dat_swap_asap_arr[a,b,c,d] = sim_dat_swap_asap[:MC_runs]


    # loading data from swap asap WITH CC
    if 'swap-asap (vanilla)' in scenario_list:
        sim_dat_swap_asap_cc_arr = -1*np.ones((sim_dat_shape))
        for a, nodes in enumerate(nodes_list):
            for b, t_cut_no_cc in enumerate(t_cut_list):
                for c, p in enumerate(p_list):
                    for d, p_s in enumerate(p_s_list):
                        # loading all the different versions at fixed parameter
                        sim = swap_asap_simulation_cc(nodes, t_cut_no_cc, p, p_s, simulation_eps, time_out_mult=time_out_mult)
                        sim_dat_swap_asap_cc = []
                        version = 0
                        sim_dat_path = sim.sim_dat_path_template+f'_v{version}.pkl'
                        assert os.path.exists(sim_dat_path), f"no sim dat file found for {nodes}, t_cut {t_cut_no_cc}, p_e {p}, p_s {p_s} at path {sim_dat_path}"
                        while os.path.exists(sim_dat_path) == True:
                            with open(sim_dat_path, "rb") as file: # Unpickling
                                sim_dat = pickle.load(file)
                            sim_dat_swap_asap_cc += sim_dat[0].tolist() # 1 to get the regular time steps
                            version += 1
                            sim_dat_path = sim.sim_dat_path_template+f'_v{version}.pkl'

                        # putting the data at fixed nodes, t_cut, p, p_s into one array
                        sim_dat_swap_asap_cc_arr[a,b,c,d] = sim_dat_swap_asap_cc[:MC_runs] 


    # loading data from swap asap using PREDICTED observations WITH CC
    if 'swap-asap (predictive)' in scenario_list:
        sim_dat_swap_asap_w_pred_arr = -1*np.ones((sim_dat_shape))
        for a, nodes in enumerate(nodes_list):
            for b, t_cut_no_cc in enumerate(t_cut_list):
                for c, p in enumerate(p_list):
                    for d, p_s in enumerate(p_s_list):
                        # loading all the different versions at fixed parameter
                        t_cut_cc = t_cut_no_cc*Quantum_network(nodes, t_cut_no_cc, p, p_s).t_cut_cc_multiplier()
                        sim = swap_asap_w_pred_simulation(nodes, t_cut_cc, p, p_s, simulation_eps, time_out_mult=time_out_mult)
                        sim_dat_swap_asap_w_pred = []
                        version = 0
                        sim_dat_path = sim.sim_dat_path_template+f'_v{version}.pkl'
                        if os.path.exists(sim_dat_path) == True:
                            assert os.path.exists(sim_dat_path), f"no sim dat file found for {nodes}, t_cut {t_cut_cc}, p_e {p}, p_s {p_s} at path {sim_dat_path}"
                            while os.path.exists(sim_dat_path) == True:
                                with open(sim_dat_path, "rb") as file: # Unpickling
                                    sim_dat = pickle.load(file)
                                sim_dat_swap_asap_w_pred += sim_dat[0].tolist() # 1 to get the regular time steps
                                version += 1
                                sim_dat_path = sim.sim_dat_path_template+f'_v{version}.pkl'

                            # putting the data at fixed nodes, t_cut, p, p_s into one array
                            sim_dat_swap_asap_w_pred_arr[a,b,c,d] = sim_dat_swap_asap_w_pred[:MC_runs] 
                        else: # when there's no simulation for these parameters yet
                            sim_dat_swap_asap_w_pred_arr[a,b,c,d] = [int(1e7) for i in range(MC_runs)]


    #########################################################################
    # plotting delivery time vs p; nodes, t_cut, p_s fixed

    line_styles = [
                ('dotted',                (0, (1, 1))),
                ('densely dotted',        (0, (1, 1))),
                ('long dash with offset', (5, (10, 3))),
                ('dashed',                (0, (5, 5))),
                ('densely dashed',        (0, (5, 1))),

                ('dashdotted',            (0, (3, 5, 1, 5))),
                ('densely dashdotted',    (0, (3, 1, 1, 1))),

                ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
                ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


    sim_dat_arr_dict = {'RL': sim_dat_cc_alt_hist_arr,
                        'swap-asap (instant)': sim_dat_swap_asap_arr,
                        'swap-asap (vanilla)': sim_dat_swap_asap_cc_arr, 
                        'swap-asap (predictive)': sim_dat_swap_asap_w_pred_arr}
    
    # sim_dat_arr_dict = {
    #                     'swap-asap (no cc effects)': sim_dat_swap_asap_arr,
    #                     'swap-asap (cc effects)': sim_dat_swap_asap_cc_arr, 
    #                     'swap-asap (predictive)': sim_dat_swap_asap_w_pred_arr}


    color_list = ['#d38d5fff', '#5fbcd3ff', '#338000ff', '#2c5aa0ff']


    plt.figure(figsize=(6, 4)) # choose size of figures
    for a, nodes in enumerate(nodes_list):
        for b, t_cut_no_cc in enumerate(t_cut_list):
            for d, p_s in enumerate(p_s_list):
                line_styles_idx = 0
                for scenario ,sim_dat_arr in sim_dat_arr_dict.items():
                    if scenario in scenario_list and scenario != 'RL': # because sim_dat_arr_dict could contain different possible scenario's than the ones selects at the top
                        expected_T = np.average(sim_dat_arr[a,b,:,d], axis=1)
                        expected_T = np.where(expected_T > 5*1e4, np.nan, expected_T) # currint
                        T_std = np.std(sim_dat_arr[a,b,:,d], axis=1)/np.sqrt(MC_runs) # dividing by np.sqrt(MC_runs) to get std error
                        plt.errorbar(p_list, expected_T, T_std, color=color_list[line_styles_idx], linestyle=line_styles[line_styles_idx][1], label=f'{scenario}', capsize=5, capthick=2, ecolor='red', elinewidth=2)
                        line_styles_idx += 1 # so that each scenario has a different line style
                    if scenario in scenario_list and scenario == 'RL': 
                        expected_T_mult_versions = []
                        T_std_mult_versions = []
                        for x in range(trained_versions):
                            expected_T_mult_versions.append(np.average(sim_dat_arr[x,a,b,:,d], axis=1))
                            T_std_mult_versions.append(np.std(sim_dat_arr[x,a,b,:,d], axis=1)/np.sqrt(MC_runs)) # dividing by np.sqrt(MC_runs) to get std error
                        expected_T = -1*np.amax(-1*np.array(expected_T_mult_versions), axis=0)
                        expected_T = np.where(expected_T > 5*1e4, np.nan, expected_T) # currint
                        indices_best_model = np.argmax(-1*np.array(expected_T_mult_versions), axis=0)
                        T_std = [T_std_mult_versions[indices_best_model[i]][i] for i in range(len(expected_T))]
                        plt.errorbar(p_list, expected_T, T_std, color=color_list[line_styles_idx], linestyle=line_styles[line_styles_idx][1], label=f'{scenario}', capsize=5, capthick=2, ecolor='red', elinewidth=2)
                        if p_s == 1:
                            plt.axhline(y=6, color='r', linestyle='-') # for checking swap asap vanilla at unit prob
                            plt.axhline(y=5, color='b', linestyle='-') # optimal global agent policy
                        line_styles_idx += 1 # so that each scenario has a different line style

                plt.title(r'$\mathbb{E}[T_{end}]$'+' for various policies at '+r'$n$'+f'={nodes}, '+r'$p_s$'+f'={p_s}')
                # plt.title(f'{nodes} nodes: '+r'$\mathbb{E}[t]$'+f' for {policy_list}')
                plt.legend(loc="upper right")
                plt.yscale('log')
                plt.xlabel(r'$p_{ent. gen.}$')
                plt.ylabel(r'$\mathbb{E}[T]$')
                # ax = plt.gca()
                # ax.set_ylim([1, ymax])
                fig_folder = save_path+proj_fig_folder
                if os.path.exists(fig_folder) == False:
                    os.makedirs(fig_folder)
                fig_path = fig_folder+f'/expected_delivery_time_n_{nodes}_t_cut_{t_cut_no_cc}_p_s_{p_s}.jpg'
                plt.savefig(fig_path,dpi=300)
                plt.clf()


