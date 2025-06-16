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

class test_qn(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.nodes, self.t_cut, self.p, self.p_s = 4, 16, 0.4, 0.4 # These have to correspond to a valid saved model 
        self.pos_center_agent = math.floor(self.nodes/2)
        print(f"pos of agent: {self.pos_center_agent}")
        self.mock_env = Environment(self.pos_center_agent, self.nodes, self.t_cut, self.p, self.p_s)
        self.mock_env.reset()
        print('setUp')

    def tearDown(self):
        pass

####################################################################################
# Start of test functions
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

    def test_mdp_dynamics(self):

        N_eps = 10 # number of episodes to do in test
        for i in range(N_eps):
            # resetting the most recent actions and results
            self.N_most_recent_actions = -2*np.ones((self.mock_env.info_hist_length, 2, self.mock_env.N_ent_gen_actions)) # axis 1: 1st dim is for swap, 2nd dim is for ent gen actions
            self.N_most_recent_results = -2*np.ones((self.mock_env.info_hist_length, 2, self.mock_env.N_ent_gen_actions))

            # this block is effectively the environment reset code
            env = self.mock_env
            env.reset()
            model = self.get_model(self.nodes, self.t_cut, self.p, self.p_s)
            if model == 'no model yet':
                model = PPO(policy = "MlpPolicy", env=env, verbose=0, ent_coef=0.001)
            model.set_env(env)
            obs, _ = env.reset()
            done = False
            self.action_time_step = 'swap'
            self.mdp_time_step = 0
            self.most_recent_sent_action = -2*np.ones(env.action_shape)

            # the simulation on an episode until end-to-end 
            print(f"----------run {i}'----------")
            while not done:
                action, _ = model.predict(obs)
                # this is the step function, but deconstructed for testing
                self.most_recent_sent_action = action
                # print(f'---do action and get obs----')
                env.do_actions_and_get_obs(action)
                # print(f'---after do action and get obs----')
                # print(f'info hist swap action: {env.get_swap_actions_from_info_hist()}')
                # print(f'env action time step: {env.action_time_step}')
                executed_actions = env.get_actions()
                # print(f"executed action: {executed_actions}")
                reward, done = env.give_reward()
                # testing part of the step function
                self.update_N_most_recent_actions(env)
                self.update_N_most_recent_results(env)
                # print('-------------------------------------')
                # print(f"test MDP time slot: {self.mdp_time_step}")
                # print(f'test action time step: {self.action_time_step}')
                # print(f"env MDP time slot: {env.mdp_time_step}")
                # print(f'env action time step: {env.action_time_step}')
                # print('most recent actions')
                # print(f"{self.N_most_recent_actions}")
                # print("applied actions")
                # print(f"{env.get_actions()}")
                # print('most recent results')
                # print(f"{self.N_most_recent_results}")
                # print(f'info hist = {env.info_hist}')
                self._test_get_actions(env, executed_actions)
                self._test_obs_results(env)
                # updating the mdp steps (only for testing?)
                env.update_mdp_time()
                self.update_mdp_time() # same function, but update separately from the one in the environment
            env.close()
        pass

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

    def update_N_most_recent_actions(self, env):
        """to roll (if it is the swap time step) and update the list containing the actions of the last N time steps
        self.N_most_recent_actions is an array where the axis 0 corresponds to from how many time steps ago the actions were,
        axis 1 is for the swap (idx 0) or ent gen (idx 1) actions, and axis 2 is the actions at each node idx, from 0 to n-2
        """
        if self.action_time_step == 'swap':
            self.N_most_recent_actions = np.roll(self.N_most_recent_actions, 2*env.N_ent_gen_actions)
            self.N_most_recent_actions[0] = -2*np.ones((2, env.N_ent_gen_actions))
            self.N_most_recent_actions[0][0] = self.most_recent_sent_action
        elif self.action_time_step == 'ent_gen':
            self.N_most_recent_actions[0][1] = self.most_recent_sent_action

    def update_N_most_recent_results(self, env):
        if self.action_time_step == 'swap':
            self.N_most_recent_results = np.roll(self.N_most_recent_results, 2*env.N_ent_gen_actions)
            self.N_most_recent_results[0] = -2*np.ones((2, env.N_ent_gen_actions))
            self.N_most_recent_results[0][0][1::] = [env.get_swap_result(non_end_node_idx) for non_end_node_idx in range(1,self.nodes-1)]
        if self.action_time_step == 'ent_gen':
            self.N_most_recent_results[0][1] = [env.get_ent_gen_result(link_idx) for link_idx in range(self.nodes-1)]

    def _test_obs_results(self, env):
        """test to make sure that if there is a result for a qubit, an action, i.e. action[i]=1, has taken place 
        self.max_dist_agent_to_end_nodes + abs(node_idx-self.agent_node) time steps ago for swap 
        2*dist_to_farthest_node_of_el_link time steps ago for ent gen 
        observations for each qubit in each node are of the form [swap action, swap result, ent gen action, ent gen result]
        """
        for node_idx in range(self.nodes):
            for qubit_idx in range(2):
                if self.action_time_step == 'swap':
                    obs_result = env.get_obs_from_info_hist(node_idx,qubit_idx)[1]
                    swap_result_delay = env.max_dist_agent_to_end_nodes-1
                    swap_send_action_to_get_result_delay = swap_result_delay + abs(node_idx-env.agent_node)
                    # if self.mdp_time_step <= swap_send_action_to_get_result_delay:
                    #     obs_result = -2
                    if obs_result in [0,1]: # only if a result has been observed
                        if node_idx > 0 and node_idx < self.nodes-2:
                            # make sure there was an action the correct amount of time steps back
                            self.assertEqual(self.N_most_recent_actions[swap_send_action_to_get_result_delay][0][node_idx], 1, 
                                             f"result = {0}, action N steps back was {self.N_most_recent_actions[swap_send_action_to_get_result_delay][0][node_idx]}")
                            # make sure obs result matches the actual result from the correct time steps ago
                            actual_result = self.N_most_recent_results[swap_result_delay][0][node_idx] # the delay after an action is performed is the same for each node, i.e. wait for signal from farthest node
                            self.assertEqual(obs_result, actual_result,
                                              f"obs result = {obs_result}, actual result = {actual_result}")
                elif self.action_time_step == 'ent_gen':
                    obs_result = env.get_obs_from_info_hist(node_idx,qubit_idx)[3]
                    delay = env.get_ent_gen_delay(node_idx,qubit_idx)
                    # if self.mdp_time_step <= 2*delay:
                    #     obs_result = -2
                    if obs_result in [0,1]: #only if a result has been observed
                        if node_idx < self.nodes-1 or qubit_idx != 1:
                            if node_idx > 0 or qubit_idx != 0:
                                link_idx = env.node_and_qubit_idx_to_link_idx(node_idx,qubit_idx)
                                # make sure there was an action the correct amount of time steps back
                                self.assertEqual(self.N_most_recent_actions[2*delay][1][link_idx], 1, 
                                                 f"result = {obs_result}, action N steps back was {self.N_most_recent_actions[2*delay][1][link_idx]}."+f" node {node_idx} and qubit {qubit_idx}, agent at {self.pos_center_agent}, delay = {delay}")
                                # make sure obs result matches the actual result from the correct time steps ago
                                actual_result = self.N_most_recent_results[delay][1][link_idx]
                                self.assertEqual(obs_result, actual_result,
                                                  f"obs result = {obs_result}, actual result = {actual_result}")

    def _test_get_actions(self, env, executed_actions):
        """to make sure that the actions are done at the right time steps
        comparing it with the list of the N most recent actions and check that the action that 
        is performed on the quantum network is performed with the corresponding delay. 
        """
        if self.action_time_step == 'swap': 
            # actions = env.get_actions()[1::]
            actions = executed_actions[1::]
            for idx, action in enumerate(actions):
                non_end_node_idx = idx+1
                delay = abs(env.agent_node-non_end_node_idx)
                actual_action = self.N_most_recent_actions[delay][0][non_end_node_idx]
                if actual_action == -2:
                    self.assertEqual(action , 0,
                    f"applied action = {action}, but needs to be 0 at node {non_end_node_idx}")
                elif actual_action in [0,1]:
                    self.assertEqual(action, actual_action,
                                     f"applied action = {action}, actual action = {actual_action} at node {non_end_node_idx}")
                else:
                    assert False, "not a valid actual action"
        elif self.action_time_step == 'ent_gen':
            actions = env.get_actions()
            for link_idx, action in enumerate(actions):
                delay = env.get_ent_gen_delay_of_link_(link_idx)
                actual_action = self.N_most_recent_actions[delay][1][link_idx]
                if actual_action == -2:
                    self.assertEqual(action, 0, 
                                     f"applied action = {action}, but needs to be 0, at segment {link_idx}")
                elif actual_action in [0,1]:
                    self.assertEqual(action, actual_action,
                                      f"applied action = {action}, actual action {actual_action} at segment {link_idx}")
                else:
                    assert False, "not a valid actual action"

    def test_give_reward(self):
        reward, terminated = self.mock_env.give_reward()
        if terminated == True:
            assert self.mock_env.nodes == 4
            assert self.mock_env.consec_A_B_ent_time_steps == 4


if __name__ == '__main__':
    unittest.main()