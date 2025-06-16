# fidelity simulation, manually simulation last point at p_s, p_e = 0.5, 0.5 

import os
from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network 
import numpy as np
import pickle

import argparse
from itertools import product

from qnetcc.TrainSim.MDPTrainSim import training_and_policy_simulation as env_alt_hist_cc_train_sim

if __name__ == "__main__":

    # No training, just simulation
    do_training = 0
    do_further_training = 0
    training_steps = int(1e3)
    train_new_model = 0
    training_version_start = 20
    training_version_stop = 40

    # simulation params
    do_simulation = 1
    simulation_eps = int(5e3)
    time_out_mult = 2

    nodes = 4
    t_cut = 2
    p_list   = [0.5]
    p_s_list = [0.5]

    for p in p_list:
        for p_s in p_s_list:
            # getting best training version for this p and p_s
            current_path = os.getcwd()
            data_path = os.path.join(current_path, '../..', 'data')
            best_agent_directory = data_path+f'/bestAgents/'
            bestAgentPath = best_agent_directory+f'bestAgentNodes{nodes}Tcut{t_cut}P{p:.2f}Ps{p_s:.2f}'
            with open(bestAgentPath, "rb") as f:
                bestAgent = pickle.load(f)
                print(f'Ps {p_s}, P {p}, best agent is {bestAgent}')
            assert bestAgent >= training_version_start
            assert bestAgent <= training_version_stop, f'best agent {bestAgent} is greater than training version stop {training_version_stop}'

            assert bestAgent==39

            # RL alt hist WITH CC
            t_cut_cc = t_cut*Quantum_network(nodes, t_cut, p, p_s).t_cut_cc_multiplier() # this assumes agent is at the center of the network
            cc_train_and_sim = env_alt_hist_cc_train_sim(nodes, t_cut_cc, p, p_s, 
                                                            do_training, do_further_training, training_steps,
                                                            do_simulation, simulation_eps, time_out_mult, 
                                                            new_training_version=train_new_model, training_version_=int(bestAgent),
                                                            callback=1, cluster=0, save_times=int(training_steps/1000), indSim=1)
            cc_train_and_sim.simulate_policy_w_fidelity()
            # cc_train_and_sim.return_best_models_so_far_plot()