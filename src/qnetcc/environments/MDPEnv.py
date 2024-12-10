import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import random
from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network
import math
import networkx as nx
import matplotlib.pyplot as plt

from qnetcc.environments.MDPRootClass import Env_root
from qnetcc.environments.ActionSpace import actions
from qnetcc.environments.ObsSpace import history_obs

from qnetcc.environments.History import info_hist

class Environment(Env_root, actions, history_obs):
    metadata = {}

    def __init__(self, pos_of_agent, nodes, t_cut, p, p_s, max_moves = 10**5, cc_effects=1):
        '''
        This is the environment of an MDP with alternating actions and the action-result history as
        the observation
        CC effects are included in this environment
        '''

        # parameters for the network 
        Env_root.__init__(self, nodes, t_cut, p, p_s) # constructor for the root environement

        # parameters depending on the actions and for the action space
        actions.__init__(self, nodes) # constructor for the action spaces

        # for the observations
        history_obs.__init__(self,pos_of_agent, nodes, t_cut) # constructor for the observations 

        # maximum allowed rounds in the MDP
        self.max_moves = max_moves

        # for having or not having CC effects
        self.cc_effects = cc_effects

        assert self.t_cut > self.nodes-2, "cut off time too small to reach end to end entanglement with swap sum model" # for an appropriately chosen cut_off time

    def reset(self, seed=None):
        """resetting/setting the initial starting point for the beginning of an episode. 
        """
        super().reset(seed=seed) # NOT SURE WHICH SEEDS THIS SETS
        self.quantum_network = Quantum_network(self.nodes,self.t_cut,self.p,self.p_s)
        # self.copy_of_quantum_network = self.quantum_network
        # self.copy_of_quantum_network.reset_network()
        # assert np.array_equal(self.quantum_network.get_link_config(), self.copy_of_quantum_network.get_link_config()) # MOVE THIS TO A UNITTEST

        # initializing the action time step to 'swap'
        assert self.quantum_network.swap_action_time_step()
        self.action_time_step = 'swap'
        self.mdp_time_step = 0
        self.consec_A_B_ent_time_steps = 0 # to keep tracks if end to end for 2*max_dist_agent_to_end_nodes time steps, so that we can end episode

        # resetting observations and observation related data
        self.observation = super().reset_obs()
        # self.most_recent_sent_action = -2*np.ones(self.action_shape) # REMOVE THIS IF IT DOESN'T APPEAR IN ANY OTHER MODULE
        self.delayed_actions = -2*np.ones((self.N_ent_gen_actions*self.actions_per_node))
        self.info_hist_length = 2*self.max_dist_agent_to_end_nodes+1+self.t_cut
        self.info_hist = -2*np.ones((self.info_hist_length,self.nodes,2,self.length_info_list_per_qubit)) # hist only needs to be 'the distance of agent to farthest away end nodes' long
        
        return self.observation, {}
    
    def step(self, action_arr):
        """the step function of the quantum network. 
        """

        # self.most_recent_sent_action = action_arr

        self.do_actions_and_get_obs(action_arr)
        self.update_mdp_time()

        reward, terminated = self.give_reward()

        return self.observation, reward, terminated, False, {}
    
    def give_reward(self):
        """reward and termination function
        Only terminate the episode if the end-to-end links have been held for long enough, such that the global agent has had enough time 
        to verify this, based on the results it gets back. 
        """

        # With verification 
        reward = 0

        result_verification_time = 2*self.max_dist_agent_to_end_nodes+1 # +1 because end-to-end is checked only after QN is already updated; 2* to go from round to time steps

        if self.quantum_network.A_B_entangled():
            self.consec_A_B_ent_time_steps += 1 
            if self.consec_A_B_ent_time_steps == result_verification_time:
                terminated = True
                reward += 0
            else:
                terminated = False
                reward += -1
                assert self.consec_A_B_ent_time_steps < result_verification_time
        elif self.quantum_network.micro_time_slot > self.max_moves:
            terminated = True
            reward += -1
            self.consec_A_B_ent_time_steps = 0
        else:
            terminated = False
            reward += -1
            self.consec_A_B_ent_time_steps = 0

        return reward, terminated

    def do_actions_and_get_obs(self, action_arr):
        """if we are in the swap time step, roll the info hist by one row
        - then put the actions that the agent selects directly into the first row of the info hist
        - Then get the actions that have arrived a the corresponding nodes (with the correct delay) and perform them on 
        the quantum network
        - put the results of the actions into the first row of the info hist
        - update the observations with the actions and the results with the corresponding delays

        The action time steps of the environment are not the same as the one from the quantum network. The ones of the environment are only updated and set to
        the ones of the quantum network once the info hist and the obs for the agent have also been updated
        """
        if self.action_time_step == 'swap':
            self.roll_info_hist()
        self.update_info_hist_with_actions(self.action_time_step, action_arr) 
        self.actions = self.get_actions() # how the actions to be passed onto the quantum network are gotten    
        self.quantum_network.local_actions_update_network(self.actions) # does actions and update link_config_hist
        results_arr = self.get_results_for_info_hist()
        self.update_info_hist_with_results(self.action_time_step, results_arr)
        self.update_obs()
        self.quantum_network.update_time_slots() # so action_time_steps are also only updated here

    def update_mdp_time(self):
        if self.action_time_step == 'ent_gen': # increase mdp time step after ent gen
            self.mdp_time_step += 1
        self.update_action_time_step()

    def update_action_time_step(self):
        """updating the action time step
        use this to update the action_time_step at the end of the 'do_actions_and_get_obs'-function
        """
        if self.action_time_step == 'ent_gen':
            self.action_time_step = 'swap'
        elif self.action_time_step == 'swap':
            self.action_time_step = 'ent_gen'
        
    def get_actions(self):
        """getting the actions to be applied on the quantum network (with delay if applicable) from the info hist 
        """
        if self.action_time_step == 'swap':
            actions = self.get_swap_actions_from_info_hist()
        elif self.action_time_step == 'ent_gen':
            actions = self.get_ent_gen_actions_from_info_hist()
        else: 
            assert False, "not a valid time step"

        return actions
                              
    def roll_obs(self):
        """rolling the observation by one row
        """
        roll_dim = math.prod(np.shape(self.observation)[1::]) # calculating the product of dim's all the axes except the first one
        self.observation = np.roll(self.observation,roll_dim)

    def update_obs(self):
        """the obs with of the form
        [swap action, swap result, ent gen action, ent gen result]
        for each qubit in each node 
        given to the agent with the corresponding delays.
        """
        # roll the obs because info hist has also been rolled
        if self.action_time_step == 'swap':
            self.roll_obs()
            assert np.shape(self.observation)[1::] == (self.nodes, 2, self.length_info_list_per_qubit)
            self.observation[0] = -2*np.ones((self.nodes, 2, self.length_info_list_per_qubit)) 
        for node_idx in range(self.nodes):
            for qubit_idx in range(2):
                self.observation[0,node_idx,qubit_idx] = self.get_obs_from_info_hist(node_idx,qubit_idx)

    def get_swap_result(self,node_idx):
        """getting the result of the action from the quantum network without any delay. 
        """
        if node_idx == 0:
            swap_result = -1
        elif node_idx == self.nodes-1:
            swap_result = -1
        elif 0<node_idx and node_idx<self.nodes-1:
            swap_result = self.quantum_network.swap_succes_list[node_idx-1] # swap succes list only contains the results for non end nodes

        if self.mdp_time_step < abs(self.agent_node-node_idx): # action cannot have arrived before this
            swap_result = -2 # because invalid actions are mapped to 0 and will give a result

        return swap_result

    def get_ent_gen_result(self,link_idx):
        """getting the ent gen result form the quantum network after the ent gen has 
        be performed without any delay. 
        """
        if link_idx == -1: # qubit in invalid segments, i.e. node, qubit = 0,0 or self.nodes-1,1 get mapped to link_idx -1
            ent_gen_result = -1
        else:
            ent_gen_result = self.quantum_network.ent_gen_succes_list[link_idx]

        if self.mdp_time_step < self.get_ent_gen_delay_of_link_(link_idx): # action cannot have arrived before this; get_ent_gen_delay_of_link_ also doesn't take into accoutn link_idx = -1
            ent_gen_result = -2 # because invalid actions are mapped to 0 and will give a result
        
        return ent_gen_result
    
    def get_results_for_info_hist(self):
        """putting the results to be directly given to the info hist in a results arr
        results_arr[node_idx,qubit_idx] corresponds to self.info_hist[0, node_idx, qubit_idx, 1] or self.info_hist[0, node_idx, qubit_idx, 3]
        depeding on whether it is a swap or ent-gen time step. 
        """
        results_arr = -3*np.ones((self.nodes,2)) # when returning results_arr at the end, there should be no -3 any more in the array
        for node_idx in range(self.nodes):
            for qubit_idx in range(2):
                if self.action_time_step == 'swap': 
                    swap_result = self.get_swap_result(node_idx)
                    results_arr[node_idx,qubit_idx] = swap_result
                elif self.action_time_step == 'ent_gen': 
                    link_idx = self.node_and_qubit_idx_to_link_idx(node_idx,qubit_idx)
                    ent_gen_result = self.get_ent_gen_result(link_idx)
                    results_arr[node_idx,qubit_idx] = ent_gen_result
        return results_arr

    # def swap_action_time_step(self):
    #     if self.action_time_step == 'swap':
    #         return True
    #     else:
    #         assert self.action_time_step == 'ent_gen', "not a valid action time step"
    #         return False
            
    # def ent_gen_time_step(self):
    #     if self.action_time_step == 'ent_gen':
    #         return True
    #     else:
    #         assert self.action_time_step == 'swap', "not a valid action time step"
    #         return False

    def close(self):
        return
    
    def render(self):
        print("just some place holder; need to fix")
        return




        