import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import random
from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network
import math
import networkx as nx
import matplotlib.pyplot as plt

class Env_root(gym.Env):
    metadata = {}

    def __init__(self, nodes, t_cut, p, p_s):
        '''
        For defining the environment and the allowed actions by the agent.
        how to update the enviroment according to each action comes in the step function.
        '''

        # parameters for the network 
        self.nodes = nodes
        self.non_end_nodes = self.nodes - 2
        self.t_cut = t_cut
        self.p = p
        self.p_s = p_s

    def reset(self, seed=None):
        """resetting/setting the initial starting point for the beginning of an episode. 
        """
        super().reset(seed=seed)
        self.quantum_network = Quantum_network(self.nodes, self.t_cut, self.p, self.p_s)


