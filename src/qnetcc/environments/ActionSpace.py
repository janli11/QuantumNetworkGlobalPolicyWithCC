import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import random
from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network
import math
import networkx as nx
import matplotlib.pyplot as plt

class actions(gym.Env):
    def __init__(self, nodes):

        # parameters depending on the actions and for the action space
        self.nodes = nodes
        # self.N_agents = self.nodes
        self.N_ent_gen_actions = self.nodes-1 # agents binary directly chooses which links get generated
        self.N_swap_actions = self.nodes-2 # agents binary directly chooses which nodes get swapped
        self.different_types_of_actions = 2 # swap and ent_gen
        self.actions_per_node = 1 # put swap and ent_gen in the same action space, separate it by the different time steps
        self.action_shape = self.N_ent_gen_actions
        self.action_space = spaces.MultiBinary(self.action_shape)

    def reset_actions(self):
        return -2*np.ones(self.action_shape) # technically this is outside the action space 
    