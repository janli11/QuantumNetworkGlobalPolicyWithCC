import os

from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network 
import os
import numpy as np
import argparse
from itertools import product
# importing the various environments to be compared with each other
from qnetcc.TrainSim.SwapAsapInstantSim import swap_asap_simulation
from qnetcc.TrainSim.SwapAsapPredObsSim import swap_asap_simulation as swap_asap_w_pred_simulation
from qnetcc.TrainSim.SwapAsapVanillaSim import swap_asap_simulation as swap_asap_simulation_cc
from qnetcc.TrainSim.RandPolSim import rand_pol_simulation as rand_pol_sim

# paths for saving things
abs_path = os.path.abspath(os.getcwd())
proj_fig_folder = '/cluster/figures'

############################################################################
# The training and simulation part
############################################################################

parser = argparse.ArgumentParser(description='hpc test')
parser.add_argument('-sim','--do_simulation',type=int,required=True, help='1 for simulating the policy, 0 for not doing simulating the policy')
parser.add_argument('-sim_eps','--simulation_eps',type=str,required=True, help='how many episodes to do the simulation')
parser.add_argument('-idx','--idx',type=int,required=True, help='the index of the product of nodes_list, t_cut_list, p_list, p_s_list')
parser.add_argument('-N_idx','--N_idx',type=int,required=True, help='the total number of indices in the slurm array')
args = parser.parse_args()

# simulation params
do_simulation = args.do_simulation
simulation_eps = int(float(args.simulation_eps))
time_out_mult = 2

scenario_list = ['swap-asap (cc effects)', 'swap-asap (no cc effects)', 'swap-asap (predictive)']

nodes_list = [4] 
t_cut_factor_list = [2] # t_cut is this factor multiplied by the number of nodes, this is for taking the sum
p_list = np.linspace(1, 0.1, 10)
# p_s_list = [0.9, 0.5, 0.3]
# p_s_list = [0.9, 0.7, 0.5]
p_s_list = [1, 0.75, 0.5]

# taking the cartesian product of all the network parameters we will iterate over
prod_lists =[]
for el in product(*[scenario_list, nodes_list, t_cut_factor_list, p_list, p_s_list]):
    prod_lists.append(el)
assert len(prod_lists) == len(scenario_list)*len(nodes_list)*len(t_cut_factor_list)*len(p_list)*len(p_s_list)
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

print('simulation parameters are :')
print(f"simulate policy = {do_simulation}")
print(f"simulation episodes = {simulation_eps}")
print(f"nodes = {nodes}")
print(f"t_cut = {t_cut}")
print(f"p = {p}")
print(f"p_s = {p_s}")   

if scenario == 'swap-asap (no cc effects)':
    # swap asap WIHTOUT CC
    t_cut_cc = t_cut*Quantum_network(nodes, t_cut, p, p_s).t_cut_cc_multiplier()
    sim = swap_asap_simulation(nodes, t_cut_cc, p, p_s, simulation_eps, time_out_mult=time_out_mult, cluster=1)
    sim_dat = sim.simulate_policy()

if scenario == 'swap-asap (cc effects)':
    # swap asap WITH CC
    sim = swap_asap_simulation_cc(nodes, t_cut, p, p_s, simulation_eps, time_out_mult=time_out_mult, cluster=1)
    sim_dat = sim.simulate_policy()

if scenario == 'swap-asap (predictive)':
    # swap asap (predictive)
    t_cut_cc = t_cut*Quantum_network(nodes, t_cut, p, p_s).t_cut_cc_multiplier()
    sim = swap_asap_w_pred_simulation(nodes, t_cut_cc, p, p_s, simulation_eps, time_out_mult=time_out_mult, cluster=1)
    sim_dat = sim.simulate_policy()

if scenario == 'random':
    # random policy 
    t_cut_cc = t_cut*Quantum_network(nodes, t_cut, p, p_s).t_cut_cc_multiplier()
    sim = rand_pol_sim(nodes, t_cut_cc, p, p_s, simulation_eps, time_out_mult=time_out_mult, cluster=1)
    sim_dat = sim.simulate_policy()