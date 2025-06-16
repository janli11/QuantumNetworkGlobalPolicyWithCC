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
trained_versions_start, trained_versions_stop = 20, 21 
do_simulation, simulation_eps = 0, int(1e3)
MC_runs = int(simulation_eps)
time_out_mult = 2 # how many seconds and episode on average is allowed to take, before the Monte Carlo simulation is aborted
# Callback
Callback=1 

# Quantum Network parameters
nodes_list = [4] 
t_cut_list = [2] # t_cut is this factor multiplied by the number of nodes, this is for taking the sum
prob_list = [(1, 0.8),(0.75, 0.7), (0.5, 1)]
# prob_list = [(1, 1)]

# explicitly turning off the cc effects
cc_effects = 0

if __name__ == "__main__":

    for nodes in nodes_list:
        for t_cut_no_cc in t_cut_list:
            for p_s, p in prob_list:
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
                                                                        callback=1, cc_effects=cc_effects)
                        sim_dat = cc_train_and_sim.do_training_and_simulation()
                else:
                    for training_verion in range(trained_versions_start, trained_versions_stop):
                        print(f'training version = {training_verion}')
                        cc_train_and_sim = env_alt_hist_cc_train_sim(nodes, t_cut_cc, p, p_s, 
                                                                        do_training, do_further_training, training_steps,
                                                                        do_simulation, simulation_eps, time_out_mult,
                                                                        new_training_version = train_new_model, training_version_=training_verion,
                                                                        callback=1, cc_effects=cc_effects)
                        sim_dat = cc_train_and_sim.do_training_and_simulation()
                # make the best model so far plot
                cc_train_and_sim.return_best_models_so_far_plot()

    

    