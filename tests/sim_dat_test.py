# To test if indepdenent runs of the simulation yield different results

import os
from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network 
import numpy as np
from qnetcc.TrainSim.MDPTrainSim import training_and_policy_simulation as env_alt_hist_cc_train_sim

######################################
# test if two consecutive LOCAL runs of the simulation yield different results
#####################################

nodes = 4
t_cut = 2
p = 0.7
p_s = 0.75
do_training, do_further_training, training_steps = 0, 0, 10000
do_simulation, simulation_eps, time_out_mult = 1, int(1e2), 2
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
for i in range(1):
    sim_dat_1, T_list1, micro_T_list1 = cc_train_and_sim.do_training_and_simulation()
    print('first simulation finished')
    print(f'delivery times : {T_list1}')
    sim_dat_2, T_list2, micro_T_list2 = cc_train_and_sim.do_training_and_simulation()
    print('second simulation finished')
    print(f'delivery times : {T_list2}')
    if np.average(T_list1) >= 5e4:
        print('average is greater than 5e4, reached max runs')
    else:
        assert np.array_equal(T_list1, T_list2) == False, "independent runs are equal"
        print('independent runs are not equal, as expected')

# ######################################
# # test if two consecutive cluster runs of the simulation yield different results
# #####################################
# import pickle

# nodes = 4
# t_cut = 2
# p = 0.9
# p_s = 0.5
# trained_version = print(np.random.randint(20,40))

# current_path = os.getcwd()
# save_path = os.path.join(current_path, '../..', 'data/global_agent_swap_sum')

            
# partial_data = []

# # Version keeps track of several subsets (parts) of the full run
# version = 0

# # Construct full path to relevant dataset
# t_cut_cc = t_cut*Quantum_network(nodes=nodes, t_cut=t_cut, p_s=p_s, p_e=p).t_cut_cc_multiplier()

# sim_dat_path = save_path+f'/env_cc_a_alt_o_hist/sim_datInd{trained_version}/sim_dat_cc_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
# # If it exists...
# if os.path.exists(sim_dat_path) == True:
#     if version == 0:
#         assert os.path.exists(sim_dat_path) == True, f"no simulation data found for n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}"
#     while os.path.exists(sim_dat_path) == True:
#         # ... read the file
#         if version > 0:
#             previous_sim_dat = sim_dat
#         with open(sim_dat_path, "rb") as file: # Unpickling
#             sim_dat = pickle.load(file)
#         if version > 0:
#             assert np.array_equal(previous_sim_dat[0], sim_dat[0]) == False, "independent runs are equal"

#         version += 1

#         # Go to next version
#         sim_dat_path = save_path+f'/env_cc_a_alt_o_hist/sim_datInd{trained_version}/sim_dat_cc_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f}_v{version}.pkl'
# else: # when there's not simulation for these parameters yet
#     print(f'no simulation data found for n_{nodes}_t_cut_{t_cut_cc}_p_{p:.02f}_p_s_{p_s:.02f} at training version {trained_version}')

