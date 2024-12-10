import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
one_level_up = os.path.dirname(current_directory)
for path in [one_level_up]:
    sys.path.append(os.path.abspath(path))

import unittest
from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network 
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

        # network parameters
        self.nodes = 5
        self.t_cut = 3
        self.p_e = 0.8
        self.p_s = 0.5

        # getting random state resulting from swap-asap; don't allow for fully unlinked states
        rand_network = Quantum_network(self.nodes, self.t_cut, self.p_e, self.p_s)
        runs = 10
        run = 0
        while run < runs:
            actions = rand_network.instant_comm_swap_asap_actions()
            result_arr = rand_network.get_result_arr(actions)
            rand_network.do_actions(actions, result_arr) 
            rand_network.update_time_slots()
            if rand_network.A_B_entangled():
                run += 1
                rand_network = Quantum_network(self.nodes, self.t_cut, self.p_e, self.p_s)
            elif np.array_equal(rand_network.get_link_config(),-1*np.ones((self.nodes,2,2))) == True:
                continue 
            else:
                run += 1
        self.rand_state = rand_network.get_link_config()

        # fully unlinked state
        self.unlinked_state = -1*np.ones((self.nodes,2,2))

        # this end_to_end has only a link between the end nodes
        state_ = -1*np.ones((self.nodes,2,2))
        rand_age = random.randint(0, self.t_cut)
        state_[0,1,0], state_[0,1,1], state_[self.nodes-1,0,0], state_[self.nodes-1,0,1] = self.nodes-1, rand_age, 0, rand_age
        self.end_to_end_state = state_

        # specific fixed state for 5 nodes
        self.fixed_state = [[[-1,-1],[2,self.t_cut-1]],
                          [[-1,-1],[-1,-1]],
                          [[0,self.t_cut-1],[3,1]],
                          [[2,1],[-1,-1]],
                          [[-1,-1],[-1,-1]]]

    def tearDown(self):
        pass

    def test_valid_link_config(self):
        """test if the link config is a valid one"""
        for node_idx in range(self.nodes):
            for qubit_idx in range(2):
                linked_node = int(self.rand_state[node_idx][qubit_idx][0])
                link_age = int(self.rand_state[node_idx][qubit_idx][1])
                if linked_node != -1:
                    self.assertLessEqual(link_age, self.t_cut, "age cannot be higher than t_cut")
                    opposite_qubit_idx = (qubit_idx+1)%2
                    self.assertEqual(self.rand_state[linked_node][opposite_qubit_idx][0], node_idx, "make sure it links back")
                    self.assertEqual(self.rand_state[linked_node][opposite_qubit_idx][1], link_age, "make sure the ages is the same")

    def test_trajectory(self):
        """write code here to test a fixed trajectory 
        for 5 nodes, 0 1 2 3 4 , e.g. 
        t = 0: ent gen at link 1 2 and 4 succeed
        t = 1: swap node 1 succeeds
        t = 1: ent gen at link 3 succeeds
        t = 2: swap as node 2, 3 succeed
        have end to end 
        """
        self.swap_actions_arr = [[0,0,0,0],
                                 [0,1,0,0],
                                 [0,0,1,1]]
        self.swap_results = [[0,-1,-1,-1],
                             [0,1,-1,-1],
                             [0,-1,1,1]]
        self.ent_gen_actions_arr = [[1,1,1,1],
                                    [0,0,1,0]]
        self.ent_gen_results = [[1,1,0,1],
                                [-1,-1,1,-1]]
        
        network = Quantum_network(self.nodes, self.t_cut, self.p_e, self.p_s)
        step = 0
        while not network.A_B_entangled():
            if network.swap_action_time_step():
                actions = self.swap_actions_arr[step]
                result_arr = self.swap_results[step]
            elif network.ent_gen_time_step():
                actions = self.ent_gen_actions_arr[step]
                result_arr = self.ent_gen_results[step]
            network.do_actions(actions, result_arr) 
            if network.ent_gen_time_step():
                step += 1
            network.update_time_slots()
            self.assertLessEqual(step,2,"should have end to end link by now")


    def test_link_aging(self):
        """write test how link ages for a fixed trajectory
        for 5 nodes, 0 1 2 3 4 , e.g. 
        t = 0: ent gen at link 1 2 and 4 succeed
        t = 1: swap node 1 succeeds
        t = 1: ent gen at link 3 succeeds
        t = 2: swap as node 2, 3 succeed
        have end to end 
        """
        self.swap_actions_arr = [[0,0,0,0],
                                 [0,0,0,0],
                                 [0,0,0,0],
                                 [0,0,0,0],
                                 [0,0,0,0]]
        self.swap_results = [[0,-1,-1,-1],
                             [0,-1,-1,-1],
                             [0,-1,-1,-1],
                             [0,-1,-1,-1],
                             [0,-1,-1,-1]]
        self.ent_gen_actions_arr = [[1,1,1,1],
                                    [0,0,1,0],
                                    [0,0,0,0],
                                    [0,0,0,0],
                                    [0,0,0,0]]
        self.ent_gen_results = [[1,1,0,1],
                                [-1,-1,1,-1],
                                [-1,-1,-1,-1],
                                [-1,-1,-1,-1],
                                [-1,-1,-1,-1]]
        
        network = Quantum_network(self.nodes, self.t_cut, self.p_e, self.p_s)
        step = 0
        for i in range(2*len(self.swap_actions_arr)): # amount of time steps to loop
            if network.swap_action_time_step():
                print('swap step')
                actions = self.swap_actions_arr[step]
                result_arr = self.swap_results[step]
            elif network.ent_gen_time_step():
                print('EG step')
                actions = self.ent_gen_actions_arr[step]
                result_arr = self.ent_gen_results[step]
            network.do_actions(actions, result_arr) 
            if network.ent_gen_time_step():
                step += 1
            print('-------------------------------')
            print(f'time step = {network.time_slot}')
            print(f'{network.get_link_config()}')
            network.update_time_slots()
            # self.assertLessEqual(step,2,"should have end to end link by now")

if __name__ == '__main__':
    unittest.main()


