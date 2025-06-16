import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import random
from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network
import math
import networkx as nx
import matplotlib.pyplot as plt


class info_hist(object):
    def __init__(self, pos_of_agent, nodes, t_cut, cc_effects = 1):
        self.nodes, self.t_cut = nodes, t_cut
        self.agent_node = pos_of_agent

        self.hist_length = self.t_cut
        self.length_info_list_per_qubit = 4 # swap action, swap resutl, ent_gen_action, ent_gen_result
        self.max_dist_agent_to_end_nodes = max([abs(self.agent_node-0), abs(self.nodes-1-self.agent_node)])

        self.cc_effects = cc_effects

    def roll_info_hist(self):
        """rolling the history by one row
        """
        roll_dim = math.prod(np.shape(self.info_hist)[1::]) # calculating the product of dim's all the axes except the first one
        self.info_hist = np.roll(self.info_hist,roll_dim)

    def unroll_info_hist(self):
        """rolling the history by one row
        """
        roll_dim = math.prod(np.shape(self.info_hist)[1::]) # calculating the product of dim's all the axes except the first one
        self.info_hist = np.roll(self.info_hist,-roll_dim)

    def get_swap_actions_from_info_hist(self):
        """getting the swap actions form the info hist with the corresponding delay
        These are then direclty used to update the quantum network

        Returns:
            _type_: _description_
        """
        swap_actions = []
        swap_actions.append(0) # swap action of form [0,...] to make the dimensions match with the ent_gen action arr
        for node_idx in range(self.nodes-2):
            non_end_node_idx = node_idx+1
            node_delay_time = abs(non_end_node_idx - self.agent_node)
            if self.cc_effects == 0:
                node_delay_time = 0
            swap_action = self.info_hist[node_delay_time, non_end_node_idx, 0, 0]
            swap_action = self.map_invalid_action_to_zero(swap_action)
            swap_actions.append(swap_action)
        return swap_actions
    
    def get_ent_gen_actions_from_info_hist(self):
        """getting the ent gen form the info hist with the correspond delay
        These are then direclty used to update the quantum network

        the link_idx is the idx of the left node in the elementary link. For link gen, alsways choose the 
        farthest away node from the agent for determining the delay

        Returns:
            _type_: _description_
        """
        ent_gen_actions = []
        for link_idx in range(self.nodes-1):
            link_delay_time = self.get_ent_gen_delay_of_link_(link_idx)
            ent_gen_action = self.info_hist[link_delay_time, link_idx, 1, 2]
            ent_gen_action = self.map_invalid_action_to_zero(ent_gen_action)
            ent_gen_actions.append(ent_gen_action)
        return ent_gen_actions
    
    def get_ent_gen_delay_of_link_(self,link_idx):
        """getting the CC delay of sending an ent gen action to farthest node corresponding to link_dx
        and the CC delay of getting an ent gen result from the farthest node corresponding to link_dx

        Args:
            link_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        if link_idx < self.agent_node: # link_idx is the same idx as the left node idx of the link
            link_delay_time = abs(self.agent_node-link_idx)
        elif link_idx >= self.agent_node:
            link_delay_time = abs(self.agent_node-(link_idx+1))
        assert link_delay_time >= 1
        if self.cc_effects == 0:
            link_delay_time = 0 
        return link_delay_time
    
    def map_invalid_action_to_zero(self,action):
        """if action not in [0,1], it's mapped to zero. This could e.g. happen if due to the 
        time delay, there is no valid action assigned yet in the info hist. 
          if action in [0,1], it remains unchanged

        Args:
            action (_type_): _description_

        Returns:
            _type_: _description_
        """
        if action not in [0,1]:
            assert action == -2
            action = 0
        return action 
    
    def get_obs_from_info_hist(self, node_idx, qubit_idx):
        """getting the observations from the action and result history
        Actions don't have delays because agent knows instantly which actions are performed
        Delays of the outcomes are twice the distance of the agent to the node

        Args:
            node_idx (_type_): _description_
            qubit_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        # let delay of swap fully depend on the time it takes from the furthest away to travel to the agent
        swap_out_come_delay = self.get_swap_result_delay() 
        swap_action_obs = self.info_hist[0,node_idx,qubit_idx,0]
        swap_result_obs = self.info_hist[swap_out_come_delay,node_idx,qubit_idx,1]
        # getting the ent gen obs and resutls obs
        ent_gen_out_come_delay = self.get_ent_gen_delay(node_idx,qubit_idx)
        ent_gen_obs = self.info_hist[0,node_idx,qubit_idx,2]
        ent_gen_result_obs = self.info_hist[ent_gen_out_come_delay,node_idx,qubit_idx,3]
        return [swap_action_obs, swap_result_obs, ent_gen_obs, ent_gen_result_obs]
    
    # swap outcome delay is incorrect, but not sure if this function is currently used
    # def get_results_from_info_hist(self):
    #     """getting the result with the corresponding delays from the info hist
    #     Here the swap result is given after 2*d(i,j) time steps and not d(i,j) + dist_agent_to_end_node 
    #     time steps. 
    #     """

    #     result_arr = -1*np.ones((self.nodes-1,2))
    #     for non_end_node_idx in range(self.nodes-2):
    #         swap_out_come_delay = abs(self.agent_node-non_end_node_idx)
    #         if self.cc_effects == 0:
    #             swap_out_come_delay = 0
    #         result_arr[non_end_node_idx,0] = self.info_hist[swap_out_come_delay,non_end_node_idx,0,1]
    #     for link_idx in range(self.nodes-1):
    #         ent_gen_out_come_delay = self.get_ent_gen_delay_of_link_(link_idx)
    #         result_arr[non_end_node_idx,1] = self.info_hist[ent_gen_out_come_delay,non_end_node_idx,0,3]    

    #     return result_arr

    def get_swap_result_delay(self):
        swap_out_come_delay = self.max_dist_agent_to_end_nodes-1 # -1 because we wait for a message from the farthest non-end node
        if self.cc_effects == 0:
            swap_out_come_delay = 0
        return swap_out_come_delay
    
    def get_swap_action_delay(self, node_idx):
        """_summary_

        Args:
            node_idx (_type_): the node that the action is sent to

        Returns:
            _type_: _description_
        """
        swap_action_delay = abs(self.agent_node-node_idx)
        if self.cc_effects == 0:
            swap_action_delay = 0
        return swap_action_delay
    
    def get_ent_gen_delay(self, node_idx, qubit_idx):
        """getting the delay of sending a ent gen action to qubit with qubit_idx at node with node_idx 
        or the delay of getting the result from the moment the ent gen has been performed at qubit with qubit_idx and node with node_idx 

        Args:
            node_idx (_type_): _description_
            qubit_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        if node_idx < self.agent_node: # node to the left of agent
            if qubit_idx == 0: # left qubit of node
                delay = abs(self.agent_node-(node_idx-1)) 
            elif qubit_idx == 1: # right qubit of node
                delay = abs(self.agent_node-node_idx) 
        elif node_idx > self.agent_node: # node to the right of agent
            if qubit_idx == 0: # left qubit of node
                delay = abs(node_idx - self.agent_node) 
            elif qubit_idx == 1: # right qubit of node
                delay = abs(node_idx + 1 - self.agent_node-node_idx)
        elif  node_idx == self.agent_node:
            delay = 1

        # double check if delay consistent with link version of getting delay
        if node_idx != 0 and qubit_idx != 0: 
            if node_idx != self.nodes-1 and qubit_idx != 1:
                link_idx = self.node_and_qubit_idx_to_link_idx(node_idx,qubit_idx)
                assert delay == self.get_ent_gen_delay_of_link_(link_idx)

        if self.cc_effects == 0:
            delay = 0

        return delay
    
    def update_info_hist_with_actions(self, action_time_step, action_arr):
        """This is the info hist with
        [swap action, swap result, ent gen action, ent gen result]
        for each qubit in each node at each timestep.

        It records the actions that the agent sends out the moment they are being sent out. 
        It gets swap and ent gen outcomes the moment they are available from the node.  
        """
        for node_idx in range(self.nodes):
            for qubit_idx in range(2):
                if action_time_step ==  'swap': 
                    swap_action = self.get_action_for_info_hist_from_action_arr_at(action_time_step, action_arr, node_idx, qubit_idx)
                    self.info_hist[0, node_idx, qubit_idx, 0] = swap_action
                elif action_time_step == 'ent_gen': 
                    ent_gen = self.get_action_for_info_hist_from_action_arr_at(action_time_step, action_arr, node_idx, qubit_idx)
                    self.info_hist[0, node_idx, qubit_idx, 2] = ent_gen

    def update_info_hist_with_results(self, action_time_step, results_arr):
        """This is the info hist with
        [swap action, swap result, ent gen action, ent gen result]
        for each qubit in each node at each timestep.

        It records the actions that the agent sends out the moment they are being sent out. 
        It gets swap and ent gen outcomes the moment they are available from the node.  
        """
        for node_idx in range(self.nodes):
            for qubit_idx in range(2):
                if action_time_step == 'swap': 
                    self.info_hist[0, node_idx, qubit_idx, 1] = results_arr[node_idx, qubit_idx]
                elif action_time_step == 'ent_gen': 
                    self.info_hist[0, node_idx, qubit_idx, 3] = results_arr[node_idx, qubit_idx]

    def get_action_for_info_hist_from_action_arr_at(self, action_time_step, action_arr, node_idx, qubit_idx):
        """getting the action corresponding to a specific node_idx and qubit_idx
          from the action array according to the gym action space

        Args:
            action_arr (_type_): _description_
            node_idx (_type_): _description_
            qubit_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        if action_time_step == 'swap':
            if node_idx>0 and node_idx<self.nodes-1:  
                action = action_arr[node_idx]
            else:
                action = 0
        elif action_time_step == 'ent_gen':
            link_idx = self.node_and_qubit_idx_to_link_idx(node_idx,qubit_idx)
            if 0<=link_idx and link_idx <=self.nodes-2: # if it is a valid link_idx
                action = action_arr[link_idx]
            else: 
                action = 0
        return action
    
    def node_and_qubit_idx_to_link_idx(self,node_idx,qubit_idx):
        """Translating the node idx and qubit idx to the link it belongs to

        Args:
            node_idx (_type_): _description_
            qubit_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        if node_idx == 0 and qubit_idx == 0:
            link_idx = -1 # assign invalid link idx
        elif node_idx == self.nodes-1 and qubit_idx == 1:
            link_idx = -1 # assign invalid link idx
        elif qubit_idx == 0:
            link_idx = node_idx-1
        elif qubit_idx == 1:
            link_idx = node_idx
        else:
            assert False
        return link_idx