import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
one_level_up = os.path.dirname(current_directory)
for path in [one_level_up]:
    sys.path.append(os.path.abspath(path))

import unittest
from qnetcc.environments.MDPEnv import Environment
from stable_baselines3 import PPO
import os

import numpy as np
import random

class test_qn(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.nodes, self.t_cut, self.p, self.p_s = 4, 16, 0.4, 0.4 # These have to correspond to a valid saved model 
        self.pos_center_agent = self.nodes//2 # floors after division
        self.env, self.model = self.create_test_env()

    def tearDown(self):
        pass

####################################################################################
# Some helper functions
####################################################################################
    def create_test_env(self):
        
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
        steps = self.t_cut*2 +random.randint(0,1)# so that the whole history has been filled and has had some extra time to evovle; +random.randint(0,1) so both rounds get seen
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
# Test functions
####################################################################################

    def test_roll_info_hist(self):
        old_info_hist = self.env.info_hist
        self.env.roll_info_hist()
        rolled_info_hist = self.env.info_hist

        for i in range(np.shape(old_info_hist)[0]):
            if i>0 and i < np.shape(old_info_hist)[0]-2:
                self.assertTrue(np.array_equal(old_info_hist[i], rolled_info_hist[i+1]))

    def test_get_swap_actions_from_info_hist(self):
        
        # The swap actions set manually in the examples are the ones to be applied directly on the quantum network

        # Example 1
        action_node_1 = 1
        action_node_2 = 0
        real_swap_actions = [0, action_node_1, action_node_2]
        self.env.info_hist[1, 1, :, 0] = action_node_1
        self.env.info_hist[0, 2, :, 0] = action_node_2

        swap_actions = self.env.get_swap_actions_from_info_hist()
        self.assertTrue(np.array_equal(swap_actions, real_swap_actions))

        # Example 2
        action_node_1 = 1
        action_node_2 = -2 # not a valid action
        corrected_action_node_2 = 0
        real_swap_actions = [0, action_node_1, corrected_action_node_2]
        self.env.info_hist[1, 1, :, 0] = action_node_1
        self.env.info_hist[0, 2, :, 0] = action_node_2

        swap_actions = self.env.get_swap_actions_from_info_hist()
        self.assertTrue(np.array_equal(swap_actions, real_swap_actions))

    def test_get_ent_gen_actions_from_info_hist(self):
        # The ent gen actions set manually in the examples are the ones to be applied directly on the quantum network

        # Example 1
        action_segment_1 = 1
        action_segment_2 = 0
        action_segment_3 = 1
        real_actions = [action_segment_1, action_segment_2, action_segment_3]
        self.env.info_hist[2, 0, 1, 2] = action_segment_1
        self.env.info_hist[2, 1, 0, 2] = action_segment_1
        self.env.info_hist[1, 1, 1, 2] = action_segment_2
        self.env.info_hist[1, 2, 0, 2] = action_segment_2
        self.env.info_hist[1, 2, 1, 2] = action_segment_3
        self.env.info_hist[1, 3, 0, 2] = action_segment_3

        ent_gen_actions = self.env.get_ent_gen_actions_from_info_hist()
        self.assertTrue(np.array_equal(ent_gen_actions, real_actions), f'real actions {real_actions} vs obtained actions form hist {ent_gen_actions}')

        # Example 2
        action_segment_1 = 1
        action_segment_2 = 1
        action_segment_3 = 0
        real_actions = [action_segment_1, action_segment_2, action_segment_3]
        self.env.info_hist[2, 0, 1, 2] = action_segment_1
        self.env.info_hist[2, 1, 0, 2] = action_segment_1
        self.env.info_hist[1, 1, 1, 2] = action_segment_2
        self.env.info_hist[1, 2, 0, 2] = action_segment_2
        self.env.info_hist[1, 2, 1, 2] = action_segment_3
        self.env.info_hist[1, 3, 0, 2] = action_segment_3

        ent_gen_actions = self.env.get_ent_gen_actions_from_info_hist()
        self.assertTrue(np.array_equal(ent_gen_actions, real_actions), f'real actions {real_actions} vs obtained actions form hist {ent_gen_actions}')

    def test_get_ent_gen_delay_of_link_(self):
        # making sure agent it located at the correct node
        self.assertEqual(self.pos_center_agent, 2)
        self.assertEqual(self.nodes, 4)

        self.assertEqual(self.env.get_ent_gen_delay_of_link_(link_idx=0), 2)
        self.assertEqual(self.env.get_ent_gen_delay_of_link_(link_idx=1), 1)
        self.assertEqual(self.env.get_ent_gen_delay_of_link_(link_idx=2), 1)

    def test_map_invalid_action_to_zero(self):
        self.assertEqual(self.env.map_invalid_action_to_zero(action=1), 1)
        self.assertEqual(self.env.map_invalid_action_to_zero(action=0), 0)
        self.assertEqual(self.env.map_invalid_action_to_zero(action=-2), 0)

    def test_get_obs_from_info_hist(self):

        # In these examples, the values for swap_action, swap_result, ent_gen_action, ent_gen_result that are set manually
        # is the new information that should been given in the current time step, i.e. part of the observation at the top of the history

        # Example 1
        node_idx, qubit_idx = 0, 1
        #so delay is:
        swap_delay = 1
        ent_gen_delay = 2
        # with observation of qubit being
        # swap part 
        swap_action = 1
        swap_result = -1
        self.env.info_hist[0, node_idx, qubit_idx, 0] = swap_action
        self.env.info_hist[swap_delay, node_idx, qubit_idx, 1] = swap_result
        # ent gen part
        ent_gen_action = 0
        ent_gen_result = 1
        self.env.info_hist[0, node_idx, qubit_idx, 2] = ent_gen_action
        self.env.info_hist[ent_gen_delay, node_idx, qubit_idx, 3] = ent_gen_result
        obs = [swap_action, swap_result, ent_gen_action, ent_gen_result]
        self.assertTrue(np.array_equal(self.env.get_obs_from_info_hist(node_idx=node_idx, qubit_idx=qubit_idx), obs), f'retrieved obs: {self.env.get_obs_from_info_hist(node_idx=node_idx, qubit_idx=qubit_idx)} should have been {obs}')

        # Example 2
        node_idx, qubit_idx = 1, 0
        #so delay is:
        swap_delay = 1
        ent_gen_delay = 2
        # with observation of qubit being
        # swap part 
        swap_action = 0
        swap_result = 0
        self.env.info_hist[0, node_idx, qubit_idx, 0] = swap_action
        self.env.info_hist[swap_delay, node_idx, qubit_idx, 1] = swap_result
        # ent gen part 
        ent_gen_action = 1
        ent_gen_result = -1
        self.env.info_hist[0, node_idx, qubit_idx, 2] = ent_gen_action
        self.env.info_hist[ent_gen_delay, node_idx, qubit_idx, 3] = ent_gen_result
        obs = [swap_action, swap_result, ent_gen_action, ent_gen_result]
        self.assertTrue(np.array_equal(self.env.get_obs_from_info_hist(node_idx=node_idx, qubit_idx=qubit_idx), obs), f'retrieved obs: {self.env.get_obs_from_info_hist(node_idx=node_idx, qubit_idx=qubit_idx)} should have been {obs}')

        # Example 3
        node_idx, qubit_idx = 2, 1
        #so delay is:
        swap_delay = 1
        ent_gen_delay = 1
        # with observation of qubit being
        # swap part 
        swap_action = 1
        swap_result = 0
        self.env.info_hist[0, node_idx, qubit_idx, 0] = swap_action
        self.env.info_hist[swap_delay, node_idx, qubit_idx, 1] = swap_result
        # ent gen part 
        ent_gen_action = 0
        ent_gen_result = -1
        self.env.info_hist[0, node_idx, qubit_idx, 2] = ent_gen_action
        self.env.info_hist[ent_gen_delay, node_idx, qubit_idx, 3] = ent_gen_result
        obs = [swap_action, swap_result, ent_gen_action, ent_gen_result]
        self.assertTrue(np.array_equal(self.env.get_obs_from_info_hist(node_idx=node_idx, qubit_idx=qubit_idx), obs), f'retrieved obs: {self.env.get_obs_from_info_hist(node_idx=node_idx, qubit_idx=qubit_idx)} should have been {obs}')

    def test_get_swap_action_delay(self):
        self.assertEqual(self.env.get_swap_action_delay(node_idx=1),1)
        self.assertEqual(self.env.get_swap_action_delay(node_idx=2),0)
    
    def test_get_swap_result_delay(self):
        self.assertEqual(self.env.get_swap_result_delay(), 1)
    
    def test_get_ent_gen_delay(self):
        self.assertEqual(self.env.get_ent_gen_delay(node_idx=0, qubit_idx=1), 2)
        self.assertEqual(self.env.get_ent_gen_delay(node_idx=1, qubit_idx=0), 2)
        self.assertEqual(self.env.get_ent_gen_delay(node_idx=2, qubit_idx=0), 1)
        self.assertEqual(self.env.get_ent_gen_delay(node_idx=3, qubit_idx=0), 1)

    def test_update_info_hist_with_actions(self):
        # for the swap round
        action_arr = [random.randint(0,1) for i in range(self.nodes-1)]
        self.env.update_info_hist_with_actions(action_time_step='swap', action_arr=action_arr)
        for non_end_nodes in range(1,self.nodes-1):
            self.assertTrue(np.array_equal(self.env.info_hist[0, non_end_nodes, :, 0], [action_arr[non_end_nodes] for i in range(2)]))
        # for the ent gen round
        action_arr = [random.randint(0,1) for i in range(self.nodes-1)]
        self.env.update_info_hist_with_actions(action_time_step='ent_gen', action_arr=action_arr)
        for segment_idx in range(self.nodes-1):
            self.assertEqual(self.env.info_hist[0, segment_idx, 1, 2], action_arr[segment_idx])
            self.assertEqual(self.env.info_hist[0, segment_idx+1, 0, 2], action_arr[segment_idx])

    def test_update_info_hist_with_results(self):
        # ADD RANDOMIZED RESULTS TEST
        self.assertEqual(self.nodes, 4)
        # for the swap round
        swap_results = np.array([[-1,-1], [-1,-1], [1,1], [-1,-1]])
        self.env.update_info_hist_with_results(action_time_step='swap', results_arr=swap_results)
        for node_idx in range(self.nodes):
            for qubit_idx in range(2):
                    self.assertEqual(self.env.info_hist[0, node_idx, qubit_idx, 1], swap_results[node_idx, qubit_idx])
        # for the ent gen round
        ent_gen_results = np.array([[1,1],[1,0],[0,1],[1,1]])
        self.env.update_info_hist_with_results(action_time_step='ent_gen', results_arr=ent_gen_results)
        for node_idx in range(self.nodes):
            for qubit_idx in range(2):
                    self.assertEqual(self.env.info_hist[0, node_idx, qubit_idx, 3], ent_gen_results[node_idx, qubit_idx])

    def test_get_action_for_info_hist_from_action_arr_at(self):

        # Example 1
        action_time_step = 'swap'  
        action_arr = [1,0,1]
        node_idx = 1
        qubit_idx = 0
        self.assertEqual(self.env.get_action_for_info_hist_from_action_arr_at(action_time_step, action_arr, node_idx, qubit_idx), 0)

        # Example 1
        action_time_step = 'swap'  
        action_arr = [1,1,1]
        node_idx = 2
        qubit_idx = 0
        self.assertEqual(self.env.get_action_for_info_hist_from_action_arr_at(action_time_step, action_arr, node_idx, qubit_idx), 1)

    def test_node_and_qubit_idx_to_link_idx(self):

        # Example 1
        node_idx, qubit_idx = 0, 0 
        self.assertEqual(self.env.node_and_qubit_idx_to_link_idx(node_idx,qubit_idx), -1)

        # Example 2
        node_idx, qubit_idx = self.nodes-1, 1 
        self.assertEqual(self.env.node_and_qubit_idx_to_link_idx(node_idx,qubit_idx), -1)

        # Example 3
        node_idx, qubit_idx = 0, 1 
        self.assertEqual(self.env.node_and_qubit_idx_to_link_idx(node_idx,qubit_idx), 0)

        # Example 4
        node_idx, qubit_idx = self.nodes-1, 0
        self.assertEqual(self.env.node_and_qubit_idx_to_link_idx(node_idx,qubit_idx), self.nodes-2)

        # Example 5
        node_idx, qubit_idx = 1, 1
        self.assertEqual(self.env.node_and_qubit_idx_to_link_idx(node_idx,qubit_idx), 1) 
        

if __name__ == '__main__':
    unittest.main()