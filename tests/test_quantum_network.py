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
        # network parameters for the fixed network
        self.nodes = 8
        self.t_cut = 10
        self.p = 0.6
        self.p_s = 0.5
        self.fixed_network = Quantum_network(self.nodes, self.t_cut, self.p, self.p_s)
        # setting up a fixed config
        link_arr = [[0,4],[1,2],[4,5]]
        link_age = [2,5,3]
        self.fixed_network.create_network(link_arr, link_age)

        # randomly chosen network params
        self.rand_nodes = random.randint(3, 10)
        self.rand_t_cut = random.randint(1, 50)
        self.rand_p = random.random()
        self.rand_p_s = random.random()
        self.rand_network = Quantum_network(self.rand_nodes, self.rand_t_cut, self.rand_p, self.rand_p_s)

    def tearDown(self):
        pass

    def test_get_pos_agent_in_center(self):
        pos_agent = self.fixed_network.get_pos_agent_in_center()
        self.assertEqual(pos_agent, 4, "for 8 nodes, the center agent should be at node with idx 4")

    def test_reset_network(self):
        self.fixed_network.reset_network()
        empty_config = self.fixed_network.get_link_config()
        self.assertTrue(np.array_equal(empty_config, [[[-1,-1],[-1,-1]] for i in range(self.nodes)]))

    def test_create_network(self):
        target_link_config = [[[-1,-1],
                               [4,2]],
                                [[-1,-1],
                                 [2,5]],
                                [[1,5],
                                 [-1,-1]],
                                [[-1,-1],
                                 [-1,-1]],
                                [[0,2],
                                 [5,3]],
                                [[4,3],
                                 [-1,-1]],
                                [[-1,-1],
                                 [-1,-1]],
                                [[-1,-1],
                                 [-1,-1]]]
        
        # fixed_network is created using the create_network fucntion
        test_config = self.fixed_network.get_link_config()
        for i in range(len(target_link_config)):
            self.assertTrue(np.array_equal(test_config[i], np.array(target_link_config)[i]), i)

    def test_get_send_action_and_get_result_duration(self):
        wait_time = self.fixed_network.get_send_action_and_get_result_duration()
        self.assertEqual(wait_time, 8, "function should return 8 in this scenario")

    def test_get_global_to_regular_time_scale_multiplier(self):
        mult = self.fixed_network.get_global_to_regular_time_scale_multiplier()
        self.assertEqual(mult, 16)

    def test_entanglement_generation(self):
        """Ent gen only generates links for correct actions and positive results. 
        links are already discarded in do actions before ent gen is attempted. 
        """
        self.fixed_network.reset_network()
        for link_idx in range(self.nodes-1):
            left_agent_action, right_agent_action, result = 1, 1, 1
            self.fixed_network.entanglement_generation(link_idx, left_agent_action, right_agent_action, result)
            link_config = self.fixed_network.get_link_config()

            # checking the linked node and the ages of the qubits
            left_nodes_connected_node = link_config[link_idx][1][0]
            self.assertEqual(left_nodes_connected_node, link_idx+1)
            left_node_age = link_config[link_idx][1][1]
            self.assertEqual(left_node_age,0)
            right_nodes_connected_node = link_config[link_idx+1][0][0]
            self.assertEqual(right_nodes_connected_node,link_idx)
            right_node_age = link_config[link_idx+1][0][1]
            self.assertEqual(right_node_age,0)

    def test_local_swap(self):
        self.fixed_network.local_swap(node_i_idx=4, result=1)
        link_config = self.fixed_network.get_link_config()

        test_network = Quantum_network(self.nodes, self.t_cut, self.p, self.p_s)
        link_arr = [[0,5],[1,2]]
        link_age = [5,5]
        test_network.create_network(link_arr,link_age)
        test_link_config = test_network.get_link_config()

        self.assertTrue(np.array_equal(link_config, test_link_config), "link config after swap not correct")
            
    def test_discard(self):
        self.fixed_network.discard(node_idx=1,qubit_idx=1)
        link_config = self.fixed_network.get_link_config()

        test_network = Quantum_network(self.nodes, self.t_cut, self.p, self.p_s)
        link_arr = [[0,4],[4,5]]
        link_age = [2,3]
        test_network.create_network(link_arr,link_age)
        test_link_config = test_network.get_link_config()

        self.assertTrue(np.array_equal(link_config, test_link_config), "link config after discarding not correct")

    def test_remove_old_link(self):

        test_network_1 = Quantum_network(self.nodes, self.t_cut, self.p, self.p_s)
        link_arr = [[0,4],[1,2],[4,5]]
        link_age = [2,self.t_cut,3]
        test_network_1.create_network(link_arr,link_age)
        for node_idx in range(self.nodes):
            for qubit_idx in range(2):
                test_network_1.remove_old_link(node_idx, qubit_idx+1) # in this function qubit index should be 1 or 2
        link_config_1 = test_network_1.get_link_config()

        test_network_2 = Quantum_network(self.nodes, self.t_cut, self.p, self.p_s)
        link_arr = [[0,4],[4,5]]
        link_age = [2,3]
        test_network_2.create_network(link_arr,link_age)
        link_config_2 = test_network_2.get_link_config()

        self.assertTrue(np.array_equal(link_config_1, link_config_2, "didn't remove the old links correctly"))

    def test_add_age_to_qubit(self):
        for node_idx in range(self.nodes):
            for qubit_idx in [1,2]:
                old_age = self.fixed_network._node(node_idx).get_qubit(qubit_idx).get_connection_age()
                self.fixed_network.add_age_to_qubit(node_i_idx=node_idx, qubit_idx=qubit_idx)
                new_age = self.fixed_network._node(node_idx).get_qubit(qubit_idx).get_connection_age() 
                if old_age != -1:
                    self.assertEqual(old_age+1, new_age)

    def test_get_fidelity(self):
        nodes, t_cut, p, p_s = 8, 10, 0.6, 0.5
        EndToEndNetwork = Quantum_network(nodes, t_cut, p, p_s)
        for i in range(100): # testing muliple times
            link_arr = [[0,7],[1,2],[4,5]]
            end_age = random.randint(0, t_cut)
            link_age = [end_age,5,3]
            EndToEndNetwork.create_network(link_arr, link_age)
            EndToEndNetwork.get_link_config()
            EndToEndNetwork.get_fidelity()
            self.assertEqual(end_age, EndToEndNetwork.get_fidelity())

        nodes, t_cut, p, p_s = 8, 10, 0.6, 0.5
        EndToEndNetwork = Quantum_network(nodes, t_cut, p, p_s)
        for i in range(100): # testing muliple times
            link_arr = [[0,4],[1,2],[4,5]]
            link_age = [3,5,3]
            EndToEndNetwork.create_network(link_arr, link_age)
            EndToEndNetwork.get_link_config()
            EndToEndNetwork.get_fidelity()
            self.assertTrue(np.isnan(EndToEndNetwork.get_fidelity()))



    def test_get_result_arr(self):
        samples = 10
        for sample in range(samples):
            # For swap results 
            self.fixed_network.micro_time_slot = 0 
            self.assertTrue(self.fixed_network.swap_action_time_step(), "has to be set to swap action time step for this part of the test")
            actions = [random.randint(0,1) for i in range(self.nodes-1)]
            results = self.fixed_network.get_result_arr(actions)
            for i in range(len(actions)):
                # if not swap action attempted, always return -1 as a result, otherwise always 0 or 1
                if i == 0:
                    self.assertEqual(results[i],-1) # swap result first node always non attempted
                else:
                    if actions[i] == 0:
                        self.assertEqual(results[i],-1) # swap result first node always non attempted
                    else:
                        self.assertTrue(results[i] in [0,1])

            # For EG results 
            self.fixed_network.micro_time_slot = 1
            self.assertTrue(self.fixed_network.ent_gen_time_step(), "has to be set to EG action time step for this part of the test")
            actions = [random.randint(0,1) for i in range(self.nodes-1)]
            results = self.fixed_network.get_result_arr(actions)
            for i in range(len(actions)):
                # if action not attempted, always return -1 as a result, otherwise always 0 or 1
                if actions[i] == 0:
                    self.assertEqual(results[i],-1) # swap result first node always non attempted
                else:
                    self.assertTrue(results[i] in [0,1])

    def test_do_actions(self):

        # swap action part
        test_network_1 = Quantum_network(self.nodes, self.t_cut, self.p, self.p_s)
        link_arr = [[0,2],[2,3],[3,4]]
        link_age = [2,1,3]
        test_network_1.create_network(link_arr,link_age)
        test_network_1.micro_time_slot = 0 # should really use setter and getters for this
        actions_arr = [0,0,1,1,0,0,0] 
        result_arr = [0,0,1,1,0,0,0]
        test_network_1.do_actions(actions_arr, result_arr)
        link_config_1 = test_network_1.get_link_config()

        test_network_2 = Quantum_network(self.nodes, self.t_cut, self.p, self.p_s)
        link_arr = [[0,4]]
        link_age = [7] # after swap, we increas the age of the qubits by one. 
        test_network_2.create_network(link_arr,link_age)
        link_config_2 = test_network_2.get_link_config()

        self.assertTrue(np.array_equal(link_config_1, link_config_2, "didn't do parallel swaps correctly"))

        # Ent. gen. part
        test_network_1 = Quantum_network(self.nodes, self.t_cut, self.p, self.p_s)
        link_arr = [[0,2],[2,3],[3,4]]
        link_age = [2,1,3]
        test_network_1.create_network(link_arr,link_age)
        test_network_1.micro_time_slot = 1 # should really use setter and getters for this
        actions_arr = [1,0,0,0,0,1,1] 
        result_arr = [1,0,0,0,0,1,0] 
        test_network_1.do_actions(actions_arr, result_arr)
        link_config_1 = test_network_1.get_link_config()

        test_network_2 = Quantum_network(self.nodes, self.t_cut, self.p, self.p_s)
        link_arr = [[0,1],[2,3],[3,4],[5,6]]
        link_age = [0,1,3,0] # after swap, we increas the age of the qubits by one. 
        test_network_2.create_network(link_arr,link_age)
        link_config_2 = test_network_2.get_link_config()

        self.assertTrue(np.array_equal(link_config_1, link_config_2, "didn't do ent gen correctly"))

    def test_update_time_slots(self):
        #example 1
        # setting the time
        self.fixed_network.time_slot = 10
        self.fixed_network.micro_time_slot = 20
        self.fixed_network.time_slot_with_cc = 80
        
        # updating checking the time after updating it
        self.fixed_network.update_time_slots()
        self.assertEqual(self.fixed_network.time_slot, 10)
        self.assertEqual(self.fixed_network.micro_time_slot, 21)
        self.assertEqual(self.fixed_network.time_slot_with_cc, 80+8)

        #example 2
        # setting the time
        self.fixed_network.time_slot = 10
        self.fixed_network.micro_time_slot = 21
        self.fixed_network.time_slot_with_cc = 80
        
        # updating checking the time after updating it
        self.fixed_network.update_time_slots()
        self.assertEqual(self.fixed_network.time_slot, 11)
        self.assertEqual(self.fixed_network.micro_time_slot, 22)
        self.assertEqual(self.fixed_network.time_slot_with_cc, 80+8)

    def test_get_link_config(self):
        pass

    def test_is_swappable(self):
        test_network_1 = Quantum_network(self.nodes, self.t_cut, self.p, self.p_s)
        link_arr = [[0,2],[2,3],[3,4]]
        link_age = [2,1,3]
        test_network_1.create_network(link_arr,link_age)

        for node_idx in range(1,self.nodes):
            if node_idx in [2,3]:
                self.assertTrue(test_network_1.is_swappable(node_idx), "nodes 2 and 3 should be swapale")

    def test_get_swappable_node(self):
        test_network_1 = Quantum_network(self.nodes, self.t_cut, self.p, self.p_s)
        link_arr = [[0,2],[2,3],[3,4],[5,6]]
        link_age = [2,1,3,4]
        test_network_1.create_network(link_arr,link_age)

        swappable_nodes = test_network_1.get_swappable_nodes()
        self.assertTrue(np.array_equal(swappable_nodes,[2,3]))

    def test_get_nodes_and_qubits_with_links(self):
        test_network_1 = Quantum_network(self.nodes, self.t_cut, self.p, self.p_s)
        link_arr = [[0,2],[2,3],[3,4],[5,6]]
        link_age = [2,1,3,4]
        test_network_1.create_network(link_arr,link_age)

        nodes_and_qubits_with_links = test_network_1.get_nodes_and_qubits_with_links()
        self.assertTrue(np.array_equal(nodes_and_qubits_with_links,[[0,1],[2,0],[2,1],[3,0],[3,1],[4,0],[5,1],[6,0]]))

    def test_instant_comm_swap_asap_actions(self):
        test_network_1 = Quantum_network(self.nodes, self.t_cut, self.p, self.p_s)
        link_arr = [[0,2],[2,3],[3,4],[5,6]]
        link_age = [2,1,3,4]
        test_network_1.create_network(link_arr,link_age)

    def test_t_cut_cc_multiplier(self):
        # t_cut_cc_multiplier only depends on n, the size of the network
        self.assertEqual(Quantum_network(4, self.t_cut, self.p, self.p_s).t_cut_cc_multiplier(), 2+2+1+1)
        self.assertEqual(Quantum_network(5, self.t_cut, self.p, self.p_s).t_cut_cc_multiplier(), 2+2+1+1)
        self.assertEqual(Quantum_network(8, self.t_cut, self.p, self.p_s).t_cut_cc_multiplier(), 3+3+4+4)

    def test_A_B_entangled(self):
        pass


if __name__ == '__main__':
    unittest.main()


