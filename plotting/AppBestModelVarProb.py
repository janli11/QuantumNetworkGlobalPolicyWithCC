# This script makes the action color plots for the best model for a list of different p_s, p_e paremeters.
# To see where the structure breaks down

import os

from qnetcc.TrainSim.MDPTrainSim import training_and_policy_simulation as env_alt_hist_cc_train_sim
from qnetcc.environments.MDPEnv import Environment
from stable_baselines3.common.utils import set_random_seed

from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network 
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import math
import matplotlib.colors as mcolors
import re # for getting integers from strings

# importing the various environments to be compared with each other
from qnetcc.TrainSim.MDPTrainSim import training_and_policy_simulation as env_alt_hist_cc_train_sim

# paths for saving things
abs_path = os.getcwd()
save_path = os.path.join(abs_path, '../..', 'data')
proj_fig_folder = '/figures'

def get_model(nodes, t_cut, p, p_s, training_version):
    """Getting previously trained and saved RL model
    """
    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')
    model_log_path = data_path+'/global_agent_swap_sum/env_cc_a_alt_o_hist'+f'/Training{training_version}'+f'/PPO_cc_nodes_{nodes}_t_cut_{t_cut}_p_{p:.02f}_p_s_{p_s:.02f}/'
    
    model_names = []
    if os.path.exists(model_log_path):
        model_names = [f for f in os.listdir(model_log_path) if f.endswith('.zip')]

    model_numbers = []
    for names in model_names:
        match = re.search(r'\d+', names)
        if match:
            number = int(match.group())  # Convert the extracted string to an integer
            model_numbers.append(number)
    
    if model_numbers != []:
        best_model_path = model_log_path + f"best_model_after_{max(model_numbers)}.zip"
        model = PPO.load(best_model_path)
    else:
        model = 'no model yet'

    return model

def action_plot(trials, nodes, t_cut, prob_arr, training_version):
    """Make actions plot for an array of different success probabilities. 
    It makes sure that all of the action plots are the same size. 

    trim is for what percantage of the action plot to trim off
    """
    
    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')

    longest_eps = 0 # number of steps in the longest eps
    action_hist_arr = []

    for prob in prob_arr:
        p, p_s = prob, prob
        # do the simulation
        pos_center_agent = math.floor(nodes/2)
        env = Environment(pos_center_agent, nodes, t_cut, p, p_s)
        model = get_model(nodes, t_cut, p, p_s, training_version)
        if model == 'no model yet':
            assert False, f"No trained models for params nodes_{nodes}_t_cut_{t_cut}_p_{p:.02f}_p_s_{p_s:.02f}, so can't do simulation"
        model.set_env(env)
        set_random_seed(seed=training_version) # setting global stable baselines seed 

        num_elements = 2**(nodes-1)+2**(nodes-2)+1 # ent gen action + swap actions + 1 extra for when the episode is over

        action_hist = (num_elements-1)*np.ones((trials, env.max_moves)) # the highest value, i.e. num_elements is for when episodes is already over and no more actions are taken

        step_longest_ep = 0 # to find number of step in the longest episode
        for episode in tqdm.tqdm(range(0, trials),leave=False):
            obs, _ = env.reset()
            done = False
            step = 0
            last_action = 'swap'
            while not done:
                action, _ = model.predict(obs)
                if env.action_time_step == 'swap':
                    action_as_int = binary_array_to_decimal(action[1:]) + 2**(nodes-1)
                    assert action_as_int >= 2**(nodes-1)-1
                elif env.action_time_step == 'ent_gen':
                    action_as_int = binary_array_to_decimal(action)  
                action_hist[episode, step] = action_as_int
                if step > 0:
                    assert last_action != env.action_time_step
                last_action = env.action_time_step
                step += 1
                if step_longest_ep < step:
                    step_longest_ep = step
                obs, reward, done, _, info = env.step(action)

        action_hist_arr.append(action_hist)
        if step_longest_ep > longest_eps:
            longest_eps = step_longest_ep

    action_hist = action_hist[:,:step_longest_ep]

    cmap = mcolors.ListedColormap(['#e6194b', '#3cb44b', '#ffe119', '#0082c8', 
                            '#f58231', '#911eb4', '#46f0f0', '#f032e6', 
                            '#d2f53c', '#fabebe', '#008080', '#e6beff', 
                            '#000000'])

    # Normalize the data to map each element to a color
    norm = mcolors.BoundaryNorm(boundaries=np.arange(num_elements+1)-0.5, ncolors=num_elements)

    # custom labels
    labels_arr = []
    n_ent_gen = nodes-1
    for j in range(2**n_ent_gen):
        action_bin = int_to_binary(j, n_ent_gen)
        label = 'EG : '+''.join(map(str, action_bin))
        labels_arr.append(label)
    n_swap = nodes-2
    for j in range(2**n_swap):
        action_bin = int_to_binary(j, n_swap)
        label = 'Swap: '+''.join(map(str, action_bin))
        labels_arr.append(label)
    labels_arr.append('Episode over')
    assert len(labels_arr) == num_elements

    for i, prob in enumerate(prob_arr):
        p, p_s = prob, prob

        action_hist = action_hist_arr[i][:,:longest_eps]

        # Plotting the data
        fig, ax = plt.subplots()
        cax = ax.imshow(action_hist, cmap=cmap, norm=norm)
        ax.axis('off') 
        cbar = fig.colorbar(cax, ticks=np.arange(num_elements))
        cbar.ax.set_yticklabels(labels_arr)

        # Show the plot
        save_directory = data_path+f'/figures/app_best_models'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        plt.savefig(save_directory+f'/app_nodes_{nodes}_t_cut_{t_cut}_p_{p:.02f}_p_s_{p_s:.02f}.jpg', dpi=1200)
        plt.savefig(save_directory+f'/app_nodes_{nodes}_t_cut_{t_cut}_p_{p:.02f}_p_s_{p_s:.02f}.pdf', dpi=1200)
        plt.savefig(save_directory+f'/app_nodes_{nodes}_t_cut_{t_cut}_p_{p:.02f}_p_s_{p_s:.02f}.svg', dpi=1200)

# This function is from chat gpt
def binary_array_to_decimal(binary_array):
    decimal_value = 0
    length = len(binary_array)
    for i in range(length):
        decimal_value += binary_array[i] * (2 ** (length - i - 1))
    return decimal_value

# this function is from chat gpt
def int_to_binary(num, length):
    binary_string = format(num, 'b')
    binary_array = [int(digit) for digit in binary_string]  

    m = len(binary_array)
    difference = length - m
    if difference > 0:
        prepend_array = [0] * difference
        result_array = prepend_array + binary_array
    else:
        # If no need to prepend, just use the original array
        result_array = binary_array
    return result_array

def get_training_check_points(nodes, t_cut, p, p_s, training_version):
        
    current_path = os.path.abspath(os.getcwd())
    save_path = os.path.join(current_path, '..', 'data/global_agent_swap_sum')
    sub_proj_rel_path = '/env_cc_a_alt_o_hist'
    all_model_folder = save_path+sub_proj_rel_path+f'/Training{training_version}'
    model_log_path = all_model_folder+f'/PPO_cc_nodes_{nodes}_t_cut_{t_cut}_p_{p:.02f}_p_s_{p_s:.02f}/'

    model_names = []
    if os.path.exists(model_log_path):
        model_names = [f for f in os.listdir(model_log_path) if f.endswith('.zip')]

    model_numbers = []
    for names in model_names:
        match = re.search(r'\d+', names)
        if match:
            number = int(match.group())  # Convert the extracted string to an integer
            model_numbers.append(number)

    return model_numbers

if __name__ == "__main__":

    nodes = 4
    t_cut_no_cc = 2
    
    prob_list = np.linspace(1, 0.7, 7)

    do_training, do_further_training, training_steps, train_new_model = 1, 0, int(1e7), 0
    do_simulation, simulation_eps, time_out_mult = 0, 0, 2

    training_version = 203 # the training version for generating fig 5 color plots

    trials = 25 # the number of independent episodes in the action plot

    for prob in prob_list:
        p, p_s = prob, prob

        t_cut_cc = t_cut_no_cc*Quantum_network(nodes, t_cut_no_cc, p, p_s).t_cut_cc_multiplier() # this assumes agent is at the center of the network
        print(f'nodes {nodes}, t_cut {t_cut_cc}, p_e {p}, p_s {p_s}')
        # do training if necessary 
        cc_train_and_sim = env_alt_hist_cc_train_sim(nodes, t_cut_cc, p, p_s, 
                                                        do_training, do_further_training, training_steps,
                                                        do_simulation, simulation_eps, time_out_mult,
                                                        new_training_version = train_new_model, training_version_=training_version,
                                                        callback=1, save_times=int(1e7/100))
        cc_train_and_sim.train_model()
        cc_train_and_sim.return_best_models_so_far_plot()
    
    # values of p, p_s should not affect t_cut_cc_multiplier()
    p, p_s = 1,1 
    t_cut_cc = t_cut_no_cc*Quantum_network(nodes, t_cut_no_cc, p, p_s).t_cut_cc_multiplier() # this assumes agent is at the center of the network
    action_plot(trials, nodes, t_cut_cc, prob_list, training_version)
