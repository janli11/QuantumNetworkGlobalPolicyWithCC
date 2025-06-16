import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import random
from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network
import math
import networkx as nx
import matplotlib.pyplot as plt
from qnetcc.environments.History import info_hist

class history_obs(gym.Env, info_hist):
    def __init__(self, pos_of_agent, nodes, t_cut):

        # some of the network parameters
        info_hist.__init__(self, pos_of_agent, nodes, t_cut)

        # for the observations
        self.agent_node = pos_of_agent # position of the agent
        self.max_dist_agent_to_end_node = max([abs(self.agent_node-0), abs(self.agent_node-self.nodes)])
        self.hist_length = self.t_cut
        self.length_info_list_per_qubit = 4 # swap action, swap result, ent_gen_action, ent_gen_result
        self.obs_shape = (self.hist_length, self.nodes, 2, self.length_info_list_per_qubit) # 2, because one for each qubit
        self.observation_space = spaces.Box(low=-2, high=t_cut, shape=self.obs_shape, dtype=int) 

    def reset_obs(self):
        return -2*np.ones(self.obs_shape) 
    
