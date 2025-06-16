import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
one_level_up = os.path.dirname(current_directory)
for path in [one_level_up]:
    sys.path.append(os.path.abspath(path))

import unittest
from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network 
from qnetcc.environments.MDPEnv import Environment
from stable_baselines3 import PPO
import os

from qnetcc.environments.History import info_hist
import numpy as np
import random
import math
import tqdm
import copy

class test_qn(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.nodes, self.t_cut, self.p, self.p_s = 4, 16, 0.7, 0.7 # These have to correspond to a valid saved model 
        self.pos_center_agent = math.floor(self.nodes/2)
        self.mock_env = Environment(self.pos_center_agent, self.nodes, self.t_cut, self.p, self.p_s)
        self.mock_env.reset()

    def tearDown(self):
        pass

####################################################################################
# Some helper functions
####################################################################################

    def get_model(self, nodes, t_cut, p, p_s):
        """Getting previously trained and saved RL model
        """
        # This assumes data folder is placed in the same direcotry and the same level as the code folder. 
        abs_path = os.path.abspath(os.getcwd())
        model_path = os.path.join(abs_path, '..', f'/data/global_agent_swap_sum/env_cc_a_alt_o_hist/Training{0}/PPO_cc_nodes_{nodes}_t_cut_{t_cut}_p_{p}_p_s_{p_s}.zip')
        if os.path.exists(model_path):
            model = PPO.load(model_path)
        else:
            model = 'no model yet'
        return model
    
    def create_test_env(self):
        """randomly sample returns and environment after doing actions for self.t_cut*2+random.randint(0,1) steps"""
        env = Environment(self.pos_center_agent, self.nodes, self.t_cut, self.p, self.p_s)
        # this block is effectively the environment reset code
        env.reset()
        model = self.get_model(self.nodes, self.t_cut, self.p, self.p_s)
        if model == 'no model yet':
            model = PPO(policy = "MlpPolicy", env=env, verbose=0, ent_coef=0.001)
        model.set_env(env)
        obs, _ = env.reset()
        done = False
        self.action_time_step = 'swap'
        self.mdp_time_step = 0

        # Let the environment take some steps to generate some info hist
        steps = self.t_cut*2+random.randint(0,1) # so that the whole history has been filled and has had some extra time to evovle; +random.randint(0,1) so both rounds get seen
        counter = 0
        while counter < steps:
            action, _ = model.predict(obs)
            # this is the step function, but deconstructed for testing
            env.do_actions_and_get_obs(action)
            obs = env.observation
            reward, done = env.give_reward()
            # updating the mdp steps (only for testing?)
            env.update_mdp_time()
            self.update_mdp_time() # same function, but update separately from the one in the environment
            if done == True:
                counter = 0
            else:
                counter += 1
        return env, model
    
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

####################################################################################
# Start of test functions
####################################################################################

    def test_end_to_end_verification(self):
        """Tests if end-to-end entanglement has been held long enough; i.e. 2*max_dist_agent_to_end_nodes time steps
        """
        samples = 1
        for i in range(samples):
            env = Environment(self.pos_center_agent, self.nodes, self.t_cut, self.p, self.p_s)
            # this block is effectively the environment reset code
            env.reset()
            model = self.get_model(self.nodes, self.t_cut, self.p, self.p_s)
            if model == 'no model yet':
                model = PPO(policy = "MlpPolicy", env=env, verbose=0, ent_coef=0.001)
            model.set_env(env)
            obs, _ = env.reset()
            done = False
            self.action_time_step = 'swap'
            self.mdp_time_step = 0
            quantum_network_hist = [] # keep track of the last 2*max time steps quantum networks

            # Keep going until the terminal state has been reached
            while done == False:
                action, _ = model.predict(obs)
                # this is the step function, but deconstructed for testing
                env.do_actions_and_get_obs(action)
                obs = env.observation
                reward, done = env.give_reward()
                # updating the mdp steps (only for testing?)
                env.update_mdp_time()
                self.update_mdp_time() # same function, but update separately from the one in the environment
                # for keeping a list of the last 2*env.max_dist_agent_to_end_node quantum network. To see if there were indeed all end-to-end entangled. 
                quantum_network_hist.append(copy.deepcopy(env.quantum_network)) 
                quantum_network_hist = quantum_network_hist[-2*env.max_dist_agent_to_end_nodes:]

            if env.quantum_network.A_B_entangled(): # if end-to-end entanglement was reached 
                self.assertEqual(reward, 0)
                for i in range(len(quantum_network_hist)):
                    self.assertTrue(quantum_network_hist[i].A_B_entangled(), f"end to end entanglement is {env.quantum_network.A_B_entangled()}, {env.quantum_network.get_link_config()}, link config hist: {[quantum_network_hist[i].get_link_config() for i in range(len(quantum_network_hist))]}")

    def test_step(self):
        samples = 5
        for i in range(samples):
            env, model = self.create_test_env()
            while env.give_reward() == (0, True): # make sure that the state is not a terminal end-to-end state
                env, model = self.create_test_env()
            old_env = copy.deepcopy(env)
            random_actions = [random.randint(0,1) for i in range(self.nodes-1)]
            env.step(random_actions)

            # Make sure action time step switched
            self.assertNotEqual(env.action_time_step, old_env.action_time_step)

            # make sure that obs it just rolled by one index
            for i in range(np.shape(env.observation)[0]):
                if 0<i < np.shape(env.observation)[0]-1: # Everything except the first element in the observations
                    env.update_action_time_step() # flip flop the action time step; because it got updated after the end of the step function
                    if env.action_time_step == 'swap': # observations only rolled after the ent_gen time step
                        self.assertTrue(np.array_equal(env.observation[i+1], old_env.observation[i]), f'fails at index {i} at action round {env.action_time_step}: obs is {env.observation[i+1]}, but old obs is {old_env.observation[i]}')
                    elif env.action_time_step == 'ent_gen': # observations only rolled after the ent_gen time step
                        self.assertTrue(np.array_equal(env.observation[i], old_env.observation[i]), f'fails at index {i} at action round {env.action_time_step}: obs is {env.observation[i]}, but old obs is {old_env.observation[i]}')
                    env.update_action_time_step() # flip flop the action time step; because it got updated after the end of the step function

            # Make examples of inserting certain actions and results and see if they are being updated correctly. 
            
    def test_update_obs(self):
        samples = 20
        for samples in range(samples):
            env, model = self.create_test_env()
            swap_action_arr = [random.randint(0,1) for i in range(self.nodes-1)]
            swap_result_arr = [random.choice([-1,0,1]) for i in range(self.nodes-1)]
            ent_gen_action_arr = [random.randint(0,1) for i in range(self.nodes-1)]
            ent_gen_result_arr = [random.choice([-1,0,1])  for i in range(self.nodes-1)]

            # putting randomized actions and results into the info hist at places that should correspond to the most recent observation
            for node_idx in range(self.nodes):
                for qubit_idx in range(2):
                    swap_result_delay = env.max_dist_agent_to_end_nodes-1
                    link_idx = env.node_and_qubit_idx_to_link_idx(node_idx, qubit_idx)
                    ent_gen_delay = env.get_ent_gen_delay_of_link_(link_idx)
                    # for swap
                    if node_idx == 0 or node_idx == self.nodes-1:
                        env.info_hist[0, node_idx, qubit_idx, 0] = 0
                        env.info_hist[swap_result_delay, node_idx, qubit_idx, 1] = -1
                    else:
                        env.info_hist[0, node_idx, qubit_idx, 0] = swap_action_arr[node_idx]
                        env.info_hist[swap_result_delay, node_idx, qubit_idx, 1] = swap_result_arr[node_idx]
                    # for ent gen
                    if 0<=link_idx<=self.nodes-2:
                        env.info_hist[0, node_idx, qubit_idx, 2] = ent_gen_action_arr[link_idx]
                        env.info_hist[ent_gen_delay, node_idx, qubit_idx, 3] = ent_gen_result_arr[link_idx]
                    else:
                        self.assertEqual(link_idx, -1)
                        env.info_hist[0, node_idx, qubit_idx, 2] = 0
                        env.info_hist[ent_gen_delay, node_idx, qubit_idx, 3] = -1
            
            # testing if the correct elements from the info hist are used a for index 0 of history (i.e the new info that is added into the hist)
            # for swap round
            env.action_time_step = 'swap'
            env.update_obs()
            for node_idx in range(self.nodes):
                for qubit_idx in range(2):
                    swap_result_delay = env.max_dist_agent_to_end_nodes-1
                    if node_idx == 0 or node_idx == self.nodes-1: # for the non end nodes
                        self.assertEqual(env.observation[0, node_idx, qubit_idx, 0], 0)
                        self.assertEqual(env.observation[0, node_idx, qubit_idx, 1], -1)
                    else:
                        self.assertEqual(env.observation[0, node_idx, qubit_idx, 0], swap_action_arr[node_idx])
                        self.assertEqual(env.observation[0, node_idx, qubit_idx, 1], swap_result_arr[node_idx], f'node {node_idx}, qubit {qubit_idx}, swap result: {swap_result_arr[node_idx]}, swap result from info hist: {env.info_hist[swap_result_delay, node_idx, qubit_idx, 1]}')
            # for ent gen round 
            env.action_time_step = 'ent_gen'
            env.update_obs()
            for node_idx in range(self.nodes):
                for qubit_idx in range(2):
                    link_idx = env.node_and_qubit_idx_to_link_idx(node_idx, qubit_idx)
                    if 0<=link_idx<=self.nodes-2:
                        self.assertEqual(env.observation[0, node_idx, qubit_idx, 2], ent_gen_action_arr[link_idx])
                        self.assertEqual(env.observation[0, node_idx, qubit_idx, 3], ent_gen_result_arr[link_idx])
                    else: # for invalid link indices
                        self.assertEqual(link_idx, -1)
                        self.assertEqual(env.observation[0, node_idx, qubit_idx, 2], 0)
                        self.assertEqual(env.observation[0, node_idx, qubit_idx, 3], -1, f'node {node_idx}, qubit {qubit_idx}, swap result: {ent_gen_result_arr[link_idx]}, swap result from info hist: {env.info_hist[ent_gen_delay, node_idx, qubit_idx, 3]}')


if __name__ == '__main__':
    unittest.main()