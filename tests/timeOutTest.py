# To test the time out multiplier for the RL simulation

import os
from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network 
import numpy as np
from qnetcc.TrainSim.MDPTrainSim import training_and_policy_simulation as env_alt_hist_cc_train_sim

######################################
# test if two consecutive LOCAL runs of the simulation yield different results
#####################################

nodes = 4
t_cut = 2
p = 0.3
p_s = 0.5
do_training, do_further_training, training_steps = 0, 0, 10000
do_simulation, simulation_eps, time_out_mult = 1, int(1e1), 2
train_new_model, training_version = 1, 23

# RL alt hist WITH CC
t_cut_cc = t_cut*Quantum_network(nodes, t_cut, p, p_s).t_cut_cc_multiplier() # this assumes agent is at the center of the network
cc_train_and_sim = env_alt_hist_cc_train_sim(nodes, t_cut_cc, p, p_s, 
                                                do_training, do_further_training, training_steps,
                                                do_simulation, simulation_eps, time_out_mult, 
                                                new_training_version=train_new_model, training_version_=training_version,
                                                callback=1, cluster=0, save_times=int(training_steps/1000),
                                                swap_asap_check=0,
                                                indSim=1)
import time
start_time = time.time()
cc_train_and_sim.simulate_policy()
end_time = time.time()
print(f"Simulation took {end_time - start_time} seconds with time_out_mult = {time_out_mult} and simulation_eps = {simulation_eps}")
