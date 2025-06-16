import os
from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network 
import numpy as np

import argparse
from itertools import product

from qnetcc.TrainSim.MDPTrainSim import training_and_policy_simulation as env_alt_hist_cc_train_sim

# paths for saving things
abs_path = os.path.abspath(os.getcwd())
proj_fig_folder = '/cluster/figures'

############################################################################
# The training and simulation part
############################################################################

parser = argparse.ArgumentParser(description='hpc test')
parser.add_argument('-train','--do_training',type=int,required=True, help='1 to train model, 0 to not train model')
parser.add_argument('-train_more','--do_further_training',type=int,required=True, help='1 to train model, even it has already been trained before, 0 for not doing it')
parser.add_argument('-train_steps','--training_steps',type=str,required=True, help='number of training steps')
parser.add_argument('-sim','--do_simulation',type=int,required=True, help='1 for simulating the policy, 0 for not doing simulating the policy')
parser.add_argument('-sim_eps','--simulation_eps',type=str,required=True, help='how many episodes to do the simulation')
parser.add_argument('-train_new_model','--train_new_model',type=int,required=True, help='Whether to train new model from scratch for given parameters or not')
parser.add_argument('-training_version_start','--training_version_start',type=str,required=True, help='For explicitly choosing the training version start')
parser.add_argument('-training_version_stop','--training_version_stop',type=str,required=True, help='For explicitly choosing the training version stop')
parser.add_argument('-idx','--idx',type=int,required=True, help='the index of the product of nodes_list, t_cut_list, p_list, p_s_list')
parser.add_argument('-N_idx','--N_idx',type=int,required=True, help='the total number of indices in the slurm array')
args = parser.parse_args()

# training params
do_training = args.do_training
do_further_training = args.do_further_training
training_steps = int(float(args.training_steps))
train_new_model = args.train_new_model
training_version_start = int(float(args.training_version_start))
training_version_stop = int(float(args.training_version_stop))

# simulation params
do_simulation = args.do_simulation
simulation_eps = int(float(args.simulation_eps))
time_out_mult = 2

scenario_list = ['RL hist (cc effects)']
nodes_list = [4] 
t_cut_factor_list = [2] # t_cut is this factor multiplied by the number of nodes, this is for taking the sum
p_list = np.linspace(1, 0.5, 6)
p_s_list = [1, 0.5]
if training_version_stop == -1:
    training_version_list = [None for i in range(training_version_stop-training_version_start)]
else:
    training_version_list = list(range(training_version_start, training_version_stop))

# taking the cartesian product of all the network parameters we will iterate over
prod_lists =[]
for el in product(*[scenario_list, nodes_list, t_cut_factor_list, p_list, p_s_list, training_version_list]):
    prod_lists.append(el)
assert len(prod_lists) == len(scenario_list)*len(nodes_list)*len(t_cut_factor_list)*len(p_list)*len(p_s_list)*len(training_version_list)
print(len(prod_lists))
print(args.N_idx)
assert len(prod_lists) == args.N_idx, f'len prod_list is {len(prod_lists)} and N_idx is {args.N_idx}'

idx = args.idx
scenario = prod_lists[idx][0]
nodes = prod_lists[idx][1]
# t_cut = prod_lists[idx][2]*nodes # make t_cut scale with the number of nodes
t_cut = prod_lists[idx][2] # t_cut doesn't scale with number of nodes
p = prod_lists[idx][3]
p_s = prod_lists[idx][4]
training_version = prod_lists[idx][5]

print('simulation parameters are :')
print(f"do training = {do_training}")
print(f"do further training = {do_further_training}")
print(f"train steps = {training_steps}")
print(f"simulate policy = {do_simulation}")
print(f"simulation episodes = {simulation_eps}")
print(f"nodes = {nodes}")
print(f"t_cut = {t_cut}")
print(f"p = {p}")
print(f"p_s = {p_s}")
print(f"training version = {training_version}")


if scenario == 'RL hist (cc effects)':
    # RL alt hist WITH CC
    t_cut_cc = t_cut*Quantum_network(nodes, t_cut, p, p_s).t_cut_cc_multiplier() # this assumes agent is at the center of the network
    cc_train_and_sim = env_alt_hist_cc_train_sim(nodes, t_cut_cc, p, p_s, 
                                                    do_training, do_further_training, training_steps,
                                                    do_simulation, simulation_eps, time_out_mult, 
                                                    new_training_version=train_new_model, training_version_=training_version,
                                                    callback=1, cluster=1, save_times=int(training_steps/1000))
    sim_dat = cc_train_and_sim.do_training_and_simulation()
    cc_train_and_sim.return_best_models_so_far_plot()

