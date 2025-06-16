import os
import sys
from qnetcc.environments.MDPEnv import Environment

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tqdm
import math

from stable_baselines3 import PPO 
from stable_baselines3.common.utils import set_random_seed

from matplotlib import rc
import style
rc('font',**style.fontstyle)
rc('text', usetex=True)


# sys.path.append(os.path.abspath(os.getcwd()))
# sys.path.append(os.path.abspath(os.getcwd())+'/envs')
# for path in sys.path:
#     print(path)


def get_model(nodes, t_cut, p, p_s, training_steps, training_version=203):
    """Getting previously trained and saved RL model
    """

    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')
    model_path = data_path+'/global_agent_swap_sum/env_cc_a_alt_o_hist'+f'/Training{training_version}'+f'/PPO_cc_nodes_{nodes}_t_cut_{t_cut}_p_{p:.02f}_p_s_{p_s:.02f}/best_model_after_{training_steps}.zip'
    print(model_path)
    if os.path.exists(model_path):
        model = PPO.load(model_path)
    else:
        model = 'no model yet'
    return model

def individual_action_plot(trials, nodes, t_cut, p, p_s, training_steps):
    
    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')

    # do the simulation
    pos_center_agent = math.floor(nodes/2)
    env = Environment(pos_center_agent, nodes, t_cut, p, p_s)
    model = get_model(nodes, t_cut, p, p_s, training_steps)
    if model == 'no model yet':
        assert False, "No trained models for these params yet, so can't do simulation"
    model.set_env(env)

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

    action_hist = action_hist[:,:step_longest_ep]

    cmap = mcolors.ListedColormap(['#e6194b', '#3cb44b', '#ffe119', '#0082c8', 
                            '#f58231', '#911eb4', '#46f0f0', '#f032e6', 
                            '#d2f53c', '#fabebe', '#008080', '#e6beff', 
                            '#aa6e28', '#800000'])

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

    # Plotting the data
    fig, ax = plt.subplots()
    cax = ax.imshow(action_hist, cmap=cmap, norm=norm)
    ax.axis('off') 
    cbar = fig.colorbar(cax, ticks=np.arange(num_elements))
    cbar.ax.set_yticklabels(labels_arr)

    # Show the plot
    plt.savefig(data_path+f'/figures/fig6_{training_steps}.pdf', dpi=1200)
    plt.savefig(data_path+f'/figures/fig6_{training_steps}.svg', dpi=1200)

def do_action_plots(trials, nodes, t_cut, p, p_s, training_steps_arr):
    """makes several individual action plots. Each only is as long as its own longest episode.
    """
    seed = 0

    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')

    longest_eps_arr = [] # number of steps in the longest eps
    action_hist_arr = []

    for training_steps in training_steps_arr:
        # do the simulation
        pos_center_agent = math.floor(nodes/2)
        env = Environment(pos_center_agent, nodes, t_cut, p, p_s)
        model = get_model(nodes, t_cut, p, p_s, training_steps)
        if model == 'no model yet':
            assert False, f"No trained models for these params nodes_{nodes}_t_cut_{t_cut}_p_{p:.02f}_p_s_{p_s:.02f} at training step {training_steps}, so can't do simulation"
        model.set_env(env)
        set_random_seed(seed) # setting global stable baselines seed 

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
        longest_eps_arr.append(step_longest_ep)

    assert len(action_hist_arr) == len(training_steps_arr), "make sure that I have all the action histories"

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
        label = 'EG: '+''.join(map(str, action_bin))
        labels_arr.append(label)
    n_swap = nodes-2
    for j in range(2**n_swap):
        action_bin = int_to_binary(j, n_swap)
        label = 'Swap: '+''.join(map(str, action_bin))
        labels_arr.append(label)
    labels_arr.append('Episode over')
    assert len(labels_arr) == num_elements

    # Plotting the data
    for i, training_steps in enumerate(training_steps_arr):

        longest_eps = longest_eps_arr[i]
        action_hist = action_hist_arr[i][:,:longest_eps]

        sorted_indices = np.lexsort(np.fliplr(action_hist).T)
        sorted_action_hist = action_hist[sorted_indices][::-1]

        plt.rcParams.update({'font.size': 14})

        # color plots
        fig, ax = plt.subplots()
        cax = ax.imshow(action_hist, cmap=cmap, norm=norm)
        ax.axis('off') 
        cbar = fig.colorbar(cax, ticks=np.arange(num_elements))
        cbar.ax.set_yticklabels(labels_arr)
        
        save_directory = data_path+f'/figures/fig6'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        plt.savefig(save_directory+f'/fig6_{training_steps}.jpg', dpi=1200)
        plt.savefig(save_directory+f'/fig6_{training_steps}.pdf', dpi=1200)
        plt.savefig(save_directory+f'/fig6_{training_steps}.svg', dpi=1200)

def do_action_plots_equal_size(trials, nodes, t_cut, p, p_s, training_steps_arr, trim=0):
    """makes several individual action plots
    Makes sure all action plots have the same size, by taking the size of the largest one. 
    """
    seed = 0

    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')

    longest_eps = 0 # number of steps in the longest eps
    action_hist_arr = []

    for training_steps in training_steps_arr:
        # do the simulation
        pos_center_agent = math.floor(nodes/2)
        env = Environment(pos_center_agent, nodes, t_cut, p, p_s)
        model = get_model(nodes, t_cut, p, p_s, training_steps)
        if model == 'no model yet':
            assert False, f"No trained models for these params nodes_{nodes}_t_cut_{t_cut}_p_{p:.02f}_p_s_{p_s:.02f} at training step {training_steps}, so can't do simulation"
        model.set_env(env)
        set_random_seed(seed) # setting global stable baselines seed 

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

    assert len(action_hist_arr) == len(training_steps_arr), "make sure that I have all the action histories"

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
        label = 'EG: '+''.join(map(str, action_bin))
        labels_arr.append(label)
    n_swap = nodes-2
    for j in range(2**n_swap):
        action_bin = int_to_binary(j, n_swap)
        label = 'Swap: '+''.join(map(str, action_bin))
        labels_arr.append(label)
    labels_arr.append('Episode over')
    assert len(labels_arr) == num_elements

    longest_eps = int(longest_eps - trim*longest_eps)

    # Plotting the data
    for i, training_steps in enumerate(training_steps_arr):

        action_hist = action_hist_arr[i][:,:longest_eps]

        sorted_indices = np.lexsort(np.fliplr(action_hist).T)
        sorted_action_hist = action_hist[sorted_indices][::-1]

        plt.rcParams.update({'font.size': 14})

        # color plots
        fig, ax = plt.subplots()
        cax = ax.imshow(action_hist, cmap=cmap, norm=norm)
        ax.axis('off') 
        cbar = fig.colorbar(cax, ticks=np.arange(num_elements))
        cbar.ax.set_yticklabels(labels_arr)
        
        save_directory = data_path+f'/figures/fig6'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        plt.savefig(save_directory+f'/fig6_{training_steps}_filled.jpg', dpi=1200)
        plt.savefig(save_directory+f'/fig6_{training_steps}_filled.pdf', dpi=1200)
        plt.savefig(save_directory+f'/fig6_{training_steps}_filled.svg', dpi=1200)


def binary_array_to_decimal(binary_array):
    decimal_value = 0
    length = len(binary_array)
    for i in range(length):
        decimal_value += binary_array[i] * (2 ** (length - i - 1))
    return decimal_value

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


if __name__ == "__main__":
    # plot params
    nodes = 4
    t_cut = 12
    p = 1
    p_s = 1
    trials = 25
    # training_steps_arr = [int(a) for a in np.linspace(1.2*1e5, 1.5e5, 6)] + [int(5e5)]#[int(1e5), int(1.25*1e5) , int(5*1e5)]
    # training_steps_arr = [99957, 8399988, 8699992]; for trainig_version = 21 or 24? t_cut = 8 from cluster at 1e7 training steps, do_action_plots seed = 0
    training_steps_arr = [91590, 2131191] # for training_version = 203
    do_action_plots(trials, nodes, t_cut, p, p_s, training_steps_arr)
    do_action_plots_equal_size(trials, nodes, t_cut, p, p_s, training_steps_arr, trim=0.5)
    # for training_steps in training_steps_arr:
    #     individual_action_plot(trials, nodes, t_cut, p, p_s, training_steps)
