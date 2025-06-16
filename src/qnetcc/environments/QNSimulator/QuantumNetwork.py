import numpy as np
import random
from qnetcc.environments.QNSimulator.node import Node
import math

###################
class Quantum_network(object):
    def __init__(self, nodes, t_cut, p_e, p_s):
        """Initializing handy attributes for our class
        """

        self.nodes = nodes # the number of nodes
        self.links = nodes-1
        self.t_cut = t_cut # this t_cut should be in the regular time step scale - because the do_actions function increases the link age by one in each regular time step
        self.p_e = p_e
        self.p_s = p_s

        # Constructing a dictionary the holds all the node objects
        self.config_dict = {}
        for i in range(nodes):
            self.config_dict[f'node_{i}'] = Node(i)

        # Initializing the link_config; start in fully unlinked state
        self.link_config = -1*np.ones((self.nodes,2,2))

        # Initializing the time slots
        self.time_slot_with_cc = 0
        self.time_slot_swap_asap_vanilla = 0
        self.time_slot = 0
        self.micro_time_slot = 0

        self.different_types_of_actions = 2 # for ent_gen, swap
        self.most_recent_action = None

    def _node(self, ix):
        """return the node object with the corresponding index 
        """
        if ix == -1:
            return None
        else:
            return self.config_dict[f'node_{ix}']

    def _is_swap_successful(self):
        """gives a 1 (success) or 1 (fail) with probability p_s
        """
        if random.random() < self.p_s:
            return 1
        else:
            assert random.random() <= 1
            return 0
    
    def _is_generation_successful(self):
        """gives a 1 (success) or 1 (fail) with probability p_e
        """
        if random.random() < self.p_e:
            return 1
        else:
            assert random.random() <= 1
            return 0
    
    def get_pos_agent_in_center(self):
        """return the index of the node in the center of the network
        If even number of nodes, to the right of the center. 
        """
        pos_of_agent  = math.floor(self.nodes/2)
        return pos_of_agent
    
    def reset_network(self):
        """reset all of the nodes to have no links in the network
        """
        for node_idx in range(self.nodes):
            for qubit_idx in range(2):
                self.discard(node_idx,qubit_idx)

    def create_network(self, link_arr, link_ages):
        """currently not enforcing yet that link arr and link ages are valid"""
        assert len(link_arr) == len(link_ages), "there must be the same number of ages as links"
        for idx, link in enumerate(link_arr):
            self._node(link[0]).set_qubit(2,link[1], link_ages[idx])
            self._node(link[1]).set_qubit(1,link[0], link_ages[idx])
    
    def get_send_action_and_get_result_duration(self):
        """Maximum wait time for sending action and getting result. The time is given in time steps, not rounds. 
        """
        pos_of_agent  = self.get_pos_agent_in_center()
        max_dist_agent_to_end_nodes = max([abs(pos_of_agent-0), abs(self.nodes-1-pos_of_agent)])
        send_action_and_get_result_duration = 2*max_dist_agent_to_end_nodes
        return send_action_and_get_result_duration

    def t_cut_cc_multiplier(self):
        """the multiplier to multiply the t_cut when going from instantaneous swap asap simulation to 
        the RL or pred swap asap. 

        For 4 nodes e.g., this should be 6. Because one instantaneous time step should be converted to 6 cc time steps. 4 for the 
        instantaneous EG round end 2 for the instantaneous swap round.  
        """
        multiplier = self.get_send_action_and_get_result_duration()+self.get_send_action_and_get_result_duration()-2
        return multiplier


    def get_global_to_regular_time_scale_multiplier(self):
        """
        value is 2*get_send_action_and_get_result_duration, but what is calculates the multiplier to go from one action round 
        to the full information wait time
        """
        send_action_and_get_result_duration = self.get_send_action_and_get_result_duration()
        global_time_step_factor = 2*send_action_and_get_result_duration # 2* for swap and ent gen
        return global_time_step_factor

    def entanglement_generation(self, link_idx, left_agent_action, right_agent_action, result):
        '''
        Attempt probabilistic entanglement generation at link_i

        Also keeps outputs whether ent_gen succeeded or failed
        '''
        
        # Entanglement generation part
        if left_agent_action == 1 and right_agent_action == 1: # if both agents decide to do entanglement generation
            qubit_left_node = self._node(link_idx).get_qubit(2)
            qubit_right_node = self._node(link_idx+1).get_qubit(1)
            if qubit_left_node.is_entangled() == False and qubit_right_node.is_entangled() == False:
                if result == 1:
                    self._node(link_idx).set_qubit(2,link_idx+1,0)
                    self._node(link_idx+1).set_qubit(1,link_idx,0)

    def local_swap(self, node_i_idx, result):
        """swap at node labelled by node_i_idx
        Args:
            node_i_idx (int): index of node at which swap is attempted
            result is either success/1 or fail/0
        """
        if int(node_i_idx) >= 0 and int(node_i_idx) <= int(self.nodes-1): # cannot do entanglement swaps in the first and last node
            new_left_node = None
            new_right_node = None
            # swap asap - if both qubits of the same node are linked, do the swap
            Node_i = self._node(node_i_idx)
            left_connected_node_idx = Node_i.get_qubit(1).get_connected_node()
            right_connected_node_idx = Node_i.get_qubit(2).get_connected_node()
            if Node_i.get_qubit(1).is_entangled() and Node_i.get_qubit(2).is_entangled():
                if result == 1:
                    new_link_age = Node_i.get_qubit(1).get_connection_age()+Node_i.get_qubit(2).get_connection_age()
                    self._node(left_connected_node_idx).set_qubit(2,right_connected_node_idx,new_link_age)
                    self._node(right_connected_node_idx).set_qubit(1,left_connected_node_idx,new_link_age)
                    # Unlink node that performed the swap
                    self._node(node_i_idx).reset(node_i_idx)
                    swap_success = True
                    new_left_node = left_connected_node_idx
                    new_right_node = right_connected_node_idx
                else:
                    # unlink every qubit involved if swap failed
                    self._node(left_connected_node_idx).reset_qubit(2)
                    self._node(right_connected_node_idx).reset_qubit(1)
                    self._node(node_i_idx).reset(node_i_idx)
            # if only left qubit of node is linked
            elif Node_i.get_qubit(1).is_entangled() == True and Node_i.get_qubit(2).is_entangled() == False:
                self._node(left_connected_node_idx).reset_qubit(2)
                self._node(node_i_idx).reset(node_i_idx)
            # if only right qubit of node is linked
            elif Node_i.get_qubit(1).is_entangled() == False and Node_i.get_qubit(2).is_entangled() == True:
                self._node(right_connected_node_idx).reset_qubit(1)
                self._node(node_i_idx).reset(node_i_idx)
            # if node is already in unlinked state
            else: # reset should do nothing
                self._node(node_i_idx).reset(node_i_idx)
        return new_left_node, new_right_node
            
    def discard(self,node_idx,qubit_idx):
        """throw out the two qubits at the ends of the link labelled by link_idx if action_at_link_idx = 1

        Args:
            node_idx (int): label for the link
            qubit_idx (int): 0 for left qubit and 1 for right qubit
        """

        if qubit_idx == 0: # agent throws away his left qubit
            link_left_node_idx = self._node(node_idx).get_qubit(qubit_idx+1).get_connected_node()
            link_right_node_idx = node_idx
            if link_left_node_idx != -1 and link_right_node_idx != -1:
                if self._node(link_left_node_idx).get_qubit(2).is_entangled() and self._node(link_right_node_idx).get_qubit(1).is_entangled():
                    assert self.config_dict[f'node_{link_left_node_idx}'].get_qubit(2).get_connected_node() == link_right_node_idx
                    assert self.config_dict[f'node_{link_right_node_idx}'].get_qubit(1).get_connected_node() == link_left_node_idx
                    self._node(link_left_node_idx).reset_qubit(2)
                    self._node(link_right_node_idx).reset_qubit(1)
            else:
                self._node(link_right_node_idx).reset_qubit(qubit_idx+1) # agent always resets its own qubit
        elif qubit_idx == 1: # agent throws away his right qubit
            link_left_node_idx = node_idx
            link_right_node_idx = self._node(node_idx).get_qubit(qubit_idx+1).get_connected_node()
            # if the qubit that was discarded was linked, both qubits get reset
            if link_left_node_idx != -1 and link_right_node_idx != -1:
                if self._node(link_left_node_idx).get_qubit(2).is_entangled() and self._node(link_right_node_idx).get_qubit(1).is_entangled():
                    assert self.config_dict[f'node_{link_left_node_idx}'].get_qubit(2).get_connected_node() == link_right_node_idx
                    assert self.config_dict[f'node_{link_right_node_idx}'].get_qubit(1).get_connected_node() == link_left_node_idx
                    self._node(link_left_node_idx).reset_qubit(2)
                    self._node(link_right_node_idx).reset_qubit(1)
            else:
                self._node(link_left_node_idx).reset_qubit(qubit_idx+1) # agent always resets its own qubit
        
    def remove_old_link(self,node_i_idx,qubit_idx):
        """Look at said qubit at said node, and if it has been linked for too long, discard it. 
        """
        if self._node(node_i_idx).get_qubit(qubit_idx).is_entangled():
            node_it_is_connected_to = self._node(node_i_idx).get_qubit(qubit_idx).get_connected_node()
            if self._node(node_i_idx).get_qubit(qubit_idx).get_connection_age() >= self.t_cut: # only allow it to be removed if it is older than t_cut
                qubit_idx_other_node = (qubit_idx)%2+1
                assert self._node(node_it_is_connected_to).get_qubit(qubit_idx_other_node).get_connection_age() >= self.t_cut # make sure other node is also too old
                self._node(node_i_idx).reset_qubit(qubit_idx)
                self._node(node_it_is_connected_to).reset_qubit(qubit_idx_other_node)
            else:
                assert self._node(node_i_idx).get_qubit(qubit_idx).get_connection_age() <= 2*self.t_cut, 'node age larger than t_cut'
        
    def add_age_to_qubit(self,node_i_idx,qubit_idx):
        """update the age of the qubit if it is linked
        """
        if self._node(node_i_idx).get_qubit(qubit_idx).is_entangled(): # qubit is etangled, update it's age
            self._node(node_i_idx).update_qubit_age(qubit_idx)

    def get_fidelity(self):
        "returns the fidelity of the end-to-end link"
        "only works if the end-to-end link is entangled"
        if self.A_B_entangled():
            fidelity = self.get_link_config()[0][1][1] 
        else:
            fidelity = np.nan
        assert fidelity <= self.t_cut, f"fidelity: {fidelity} cannot be larger than t_cut: {self.t_cut}"
        return fidelity

    def get_result_arr(self, action_mat):
        """ Get the results corresponding to the action_mat. 
        For each action that is performed (i.e. a 1), we will give a results 1 (succes) or 0 (fail).
        If no action has been performed (i.e. a 0), the result is always a -1. 
        """
        self.swap_actions = action_mat[1:] # one binary for each non end node, remove first element because there is one more elementary link than non-end node
        self.ent_gen_actions = action_mat # one binary for each potential elementary link

        result_arr = []
        if self.swap_action_time_step():
            for non_end_node_idx in range(self.nodes-2):
                if self.swap_actions[non_end_node_idx] == 1:
                    result_arr.append(self._is_swap_successful())
                else: 
                    result_arr.append(-1)
            result_arr.insert(0, -1) # used to be (0,0)
        elif self.ent_gen_time_step():
            for link_idx in range(self.nodes-1):
                if self.ent_gen_actions[link_idx] == 1:
                    result_arr.append(self._is_generation_successful())
                else:
                    result_arr.append(-1)
        result_arr = np.array(result_arr)
        return result_arr

    def do_actions(self, action_mat, result_arr):
        """perform the ent gen and swap action on the quantum network.

        swap/ent gen result is 0 (=fail) or 1 (=success) only is action=1
        is action=0, result always -1
        """

        # separating action_arr into individual action arr for ent_gen and swap

        self.swap_actions = action_mat[1:] # one binary for each non end node, remove first element because there is one more elementary link than non-end node
        self.ent_gen_actions = action_mat # one binary for each potential elementary link

        # keeping track which actions have been done and if they succeeded
        self.ent_gen_succes_list = [-1 for i in range(len(self.ent_gen_actions))] # whether entanglement has successfully been generated for each of the two qubits for each node
        self.swap_succes_list = [-1 for i in range(len(self.swap_actions))] # whether any swaps have been done, and if so, what the new connected nodes are

        assert len(self.ent_gen_actions) == self.nodes-1
        assert len(self.swap_actions) == self.nodes-2
        
        # entanglement swapping 
        if self.swap_action_time_step():
            self.most_recent_action = 'swap'
            for node_idx, action in enumerate(self.swap_actions): # looping over all Nodes except first and last one
                non_end_node_idx = node_idx+1
                self.swap_result = -1 # for no swap action, so no result 
                self.ent_gen_result = -1 # for not ent gen action, so no result 
                if action == 1: # do swap; only case about first binary
                    result = result_arr[non_end_node_idx]
                    new_left_node, new_right_node = self.local_swap(non_end_node_idx, result)
                    self.swap_result = result
                elif action == 0: # don't swap; only case about first binary
                    continue
                self.swap_succes_list[node_idx] = self.swap_result
            # if self.A_B_entangled() == False:
            # Wihtout if self.A_B_entangled() == False, will throw away link that is end-to-end but has reached cut-off age. 
            # Keep the link until it has reached the cutoff age
            for node_idx in range(self.nodes):
                for qubit_idx in [1,2]:
                    self.remove_old_link(node_idx,qubit_idx)
            # increasing the age of entangled qubits
            for node_idx in range(self.nodes):
                for qubit_idx in [1,2]:
                    self.add_age_to_qubit(node_idx,qubit_idx)
        # generate entanglement
        elif self.ent_gen_time_step():
            self.most_recent_action = 'ent_gen'
            for link_idx in range(self.links):
                self.ent_gen_succes_list[link_idx] = -1
                left_qubit_action = self.ent_gen_actions[link_idx]
                right_qubit_action= self.ent_gen_actions[link_idx]
                left_node_idx = link_idx
                right_node_idx = link_idx+1
                # discarding the link before attempting to generate entanglement, if agent want to attempt entanglement generation
                for qubit_idx in range(2):
                    if qubit_idx == 1 and left_qubit_action == 1: # discard left qubit of the link
                        self.discard(left_node_idx,qubit_idx)
                    elif qubit_idx == 0 and right_qubit_action == 1: # discard right qubit of the link
                        self.discard(right_node_idx,qubit_idx)
                # actual entanglement generation part
                ent_gen_succes = result_arr[link_idx]
                self.entanglement_generation(link_idx, left_qubit_action, right_qubit_action, ent_gen_succes)
                if ent_gen_succes == 1:
                    self.ent_gen_succes_list[link_idx] = 1
                elif ent_gen_succes == 0 and self.ent_gen_actions[link_idx] == 1: # if ent gen failed, both qubit discarded
                    self.ent_gen_succes_list[link_idx] = ent_gen_succes
                else:
                    assert self.ent_gen_succes_list[link_idx] == -1
                if self.ent_gen_actions[link_idx] == 0: 
                    assert self.ent_gen_succes_list[link_idx] == -1, f"ent gen result at link {link_idx} is {self.ent_gen_succes_list[link_idx]}"

    def update_time_slots(self):
        """Update the micro time slot each time an action can in pricinple be take, whether it is swap/ent gen/ etc.
        Update the regular time slot whenever all possible types of actions (swap/ent gen/etc.) have been looped through.
        Update the global time step as soon as enough time has passed such that each of the actions could have been send 
        to the farthest away node from the agent an for a message cotaining a result to have travelled back to the agent. 
        """
        # with cc effects, after each action, wait for one round trip of communication between agent and most distant node
        if self.micro_time_slot == 0: # don't increas in the first time step, because the first round of swap actions should be discarded in a real policy
            pass
        else:
            self.time_slot_with_cc += self.get_send_action_and_get_result_duration()
            if self.swap_action_time_step():
                self.time_slot_swap_asap_vanilla += self.get_send_action_and_get_result_duration()-2 # because for sending swap action, never needs to be send to farthest away node. Same for the result
            else:
                assert self.ent_gen_time_step()
                self.time_slot_swap_asap_vanilla += self.get_send_action_and_get_result_duration()


        # increase by 1 after each action/update
        self.micro_time_slot += 1

        # increase by 1 after each ent_gen - after ent gen is done, the new time step is the swap_action time step
        if self.swap_action_time_step():
            self.time_slot += 1


    def local_actions_update_network(self, action_mat):
        """ updates the quantum network (config dict) based on the action_mat
        Also updates the corresponding link_config
        """
        self.do_actions(action_mat, self.get_result_arr(action_mat))
        self.link_config = self.get_link_config()
        return self
    
    def not_global_update_step(self):
        if (self.time_slot%self.links != 0):
            return True
        else:
            return False
        
    def get_link_config(self):
        """getting the link_config from the current config_dict
        """
        link_config = -1*np.ones((self.nodes,2,2))
        for node_idx in range(self.nodes):
            for qubit_idx in [1,2]:
                qubit_connected_node = self._node(node_idx).get_qubit(qubit_idx).get_connected_node()
                if qubit_connected_node =='none': # translating 'none' to -1
                    qubit_connected_node = -1
                qubit_connection_age = self._node(node_idx).get_qubit(qubit_idx).get_connection_age()
                if qubit_connection_age == 'none': # translating 'none' to -1
                    qubit_connection_age = -1
                link_config[node_idx][qubit_idx-1] = [qubit_connected_node,qubit_connection_age]
        return link_config

    def is_swappable(self,node_idx):
        """This checks that node has two links. 

        Args:
            node_idx (int): index of the node 

        Returns:
            Boolean: True if node can do swap and False if it cannot
        """
        # assert node_idx > 0 and node_idx < self.nodes-1
        Node_i = self._node(node_idx)
        if Node_i.get_qubit(1).is_entangled() and Node_i.get_qubit(2).is_entangled():
            assert node_idx != 0 # first node can never be swapable
            assert node_idx != self.nodes-1 # last node can never be swapable
            swapable = True
        else:
            swapable = False
        return swapable
        
    def link_exists_and_correct(self,node_idx,qubit_idx):
        """For said qubit at said node, checks if link_i exists and if it's correctly linked, i.e if it is linked to another 
        qubit, that qubit also links back to the first qubit. 

        Args:
            link_i (int): the index of the link, this is the same as the left node of the link

        Returns:
            Bool: checks if it's actually 
        """
        
        connected_node_link = self._node(node_idx).get_qubit(qubit_idx+1).get_connected_node()
        if connected_node_link != -1:
            qubit_idx_at_other_side_link = (qubit_idx+1)%2
            original_node_link = self._node(connected_node_link).get_qubit(qubit_idx_at_other_side_link+1).get_connected_node()
            assert node_idx == original_node_link
            assert original_node_link != -1
            return True
        else:
            return False
        
    def get_swappable_nodes(self):
        """loops over all the nodes (including end nodes) and checks which ones are swappable. 

        Returns:
            list: list of indices of nodes with two links
        """
        swappable_nodes_list = []
        for node_idx in range(self.nodes): # will automatically filter out the nodes that are not swappable
            if self.is_swappable(node_idx) == True:
                swappable_nodes_list.append(node_idx)
        return swappable_nodes_list

    def get_nodes_and_qubits_with_links(self):
        """checks which qubits in which nodes are linked

        Returns:
            list: returns a list of [node_idx,qubit_idx] that are linked
        """
        nodes_and_qubits_with_links = []
        for node_idx in range(self.nodes):
            for qubit_idx in range(2):
                if self._node(node_idx).get_qubit(qubit_idx+1).is_entangled() == True:
                    assert self.link_exists_and_correct(node_idx,qubit_idx=qubit_idx) == True
                    nodes_and_qubits_with_links.append([node_idx,qubit_idx])
        return nodes_and_qubits_with_links

    def instant_comm_swap_asap_actions(self):
        """swap asap actions where actions can be performed every local time step
        """

        swappable_nodes_list = self.get_swappable_nodes()
        nodes_and_qubits_with_links = self.get_nodes_and_qubits_with_links()

        node_actions = []
        for node_idx in range(self.nodes):
            if self.swap_action_time_step(): # for swap action
                if node_idx in swappable_nodes_list: # only do swap if swappable in swap asap protocol
                    action = 1
                else:
                    action = 0
            elif self.ent_gen_time_step(): # for ent gen
                qubit_idx = 1 # check if link exists through checking if the left qubit in the potential link is linked
                if [node_idx,qubit_idx] in nodes_and_qubits_with_links: # don't attempt elementary link generation if the qubit is already linked
                    assert self.link_exists_and_correct(node_idx,qubit_idx) == True
                    action = 0
                else:
                    assert self.link_exists_and_correct(node_idx,qubit_idx) == False
                    other_node_potential_el_link = node_idx+1
                    if [other_node_potential_el_link, (qubit_idx+1)%2] in nodes_and_qubits_with_links: # if qubit in left node of potential el. link is not linked, make sure qubit in right node of potential el. link is also not part of another longer link
                        action = 0
                    else:
                        action = 1
            node_actions.append(action)
        node_actions.pop(-1) # len(node actions) equal to the number of links
        node_actions = np.array(node_actions)
        return node_actions

    def check_and_get_swap_time_step_with_waiting_for_info_from_ent_gen_result(self, pos_of_agent):
        """REMOVE IF delayed_comm_swap_asap_actions IS NO LONGER USED IN ANY OF THE MODULES"""
        """check if enough time has passed for an ent gen action to be send to the furthest away node and 
        for the result to travel back. 

        if just enough time has passed and the micro time step is also a swap action micro time step, return true,
        otherwise return false. 

        Args:
            pos_of_agent (int): the node idx of the node at which the agent is situated
        """
        send_action_and_get_result_duration = self.get_send_action_and_get_result_duration()
        if self.time_slot == 0: # for setting the most_recent_ent_gen_action_time_step if no ent gen done yet and we start with a swap. So that we can do a swap in time slot 0
            self.most_recent_ent_gen_action_time_step = -1*send_action_and_get_result_duration
        ent_gen_long_enough_ago = (self.time_slot - self.most_recent_ent_gen_action_time_step == send_action_and_get_result_duration)
        if ent_gen_long_enough_ago and self.swap_action_time_step():
            swap_time_step_with_waiting_for_info_from_ent_gen_result = True
            self.most_recent_swap_action_time_step = self.time_slot
            assert self.most_recent_swap_action_time_step%self.get_send_action_and_get_result_duration() == 0
        else:
            swap_time_step_with_waiting_for_info_from_ent_gen_result = False
        return swap_time_step_with_waiting_for_info_from_ent_gen_result
    
    def check_and_get_ent_gen_time_step_with_waiting_for_info_from_swap_result(self, pos_of_agent):

        send_action_and_get_result_duration = self.get_send_action_and_get_result_duration()
        swap_long_enough_ago = (self.time_slot - self.most_recent_swap_action_time_step == send_action_and_get_result_duration)
        if swap_long_enough_ago and self.ent_gen_time_step():
            ent_gen_time_step_with_waiting_for_info_from_swap_result = True
            self.most_recent_ent_gen_action_time_step = self.time_slot
            assert self.most_recent_ent_gen_action_time_step%self.get_send_action_and_get_result_duration() == 0
        else:
            ent_gen_time_step_with_waiting_for_info_from_swap_result = False
        return ent_gen_time_step_with_waiting_for_info_from_swap_result    

    def delayed_comm_swap_asap_actions(self,pos_of_agent):
        """swap asap actions where the agents have to wait untill they have received information from the farthest away node 
        before taking any actions.
        """

        swappable_nodes_list = self.get_swappable_nodes()
        nodes_and_qubits_with_links = self.get_nodes_and_qubits_with_links()

        node_actions = []
        for node_idx in range(self.nodes):
            if self.check_and_get_swap_time_step_with_waiting_for_info_from_ent_gen_result(pos_of_agent): # for swap action
                if node_idx in swappable_nodes_list: # only do swap if swappable in swap asap protocol
                    action = 1
                else:
                    action = 0
            elif self.check_and_get_ent_gen_time_step_with_waiting_for_info_from_swap_result(pos_of_agent): # for ent gen
                qubit_idx = 1 # check if link exists through checking if the left qubit in the potential link is linked
                if [node_idx,qubit_idx] in nodes_and_qubits_with_links: # don't attempt elementary link generation if the qubit is already linked
                    assert self.link_exists_and_correct(node_idx,qubit_idx) == True
                    action = 0
                else:
                    assert self.link_exists_and_correct(node_idx,qubit_idx) == False
                    other_node_potential_el_link = node_idx+1
                    if [other_node_potential_el_link, (qubit_idx+1)%2] in nodes_and_qubits_with_links: # if qubit in left node of potential el. link is not linked, make sure qubit in right node of potential el. link is also not part of another longer link
                        action = 0
                    else:
                        action = 1
            else:
                action = 0
            node_actions.append(action)
        node_actions.pop(-1) # len(node actions) equal to the number of links
        node_actions = np.array(node_actions)
        
        return node_actions

    def random_policy_actions(self):
        if self.swap_action_time_step():
            actions = self.random_policy_swap_actions()
        elif self.ent_gen_time_step():
            actions = self.random_policy_ent_gen_actions()
        return actions
    
    def random_policy_swap_actions(self, rand_swap_prob=0.5):
        swap_actions = []
        swap_actions.append(0) # to make length consistent with ent gen actions
        for non_end_node_idx in range(self.nodes-2):
            if random.random() < rand_swap_prob:
                swap_action = 1
            else:
                swap_action = 0
            swap_actions.append(swap_action)
        return swap_actions

    def random_policy_ent_gen_actions(self, rand_ent_gen_prob=0.5):
        ent_gen_actions = []
        for link_idx in range(self.nodes-1):
            if random.random() < rand_ent_gen_prob:
                ent_gen_action = 1
            else:
                ent_gen_action = 0
            ent_gen_actions.append(ent_gen_action)
        return ent_gen_actions
    
    def swap_action_time_step(self):
        return self.micro_time_slot%self.different_types_of_actions == 0
    
    def ent_gen_time_step(self):
        return self.micro_time_slot%self.different_types_of_actions == 1

    def A_B_entangled(self):
        # check if Alice's and Bob's nodes are entangled
        # If so, end the game
        link_config_ = self.get_link_config()
        if link_config_[0][1][0] == self.nodes-1 and link_config_[self.nodes-1][0][0] == 0:
            return True
        else:
            return False
    
    def give_reward(self):
        if self.A_B_entangled():
            return 0
        else: 
            return -1
        
    def get_length_longest_link(self):
        """finds the length of the longest link in the current link config/state of the quantum network"""
        link_config = self.get_link_config()
        length_dict = {}
        for node_idx in range(self.nodes):
            for qubit_idx in range(2):
                connected_node = int(link_config[node_idx][qubit_idx][0])
                if 0 <= connected_node <= self.nodes-1: # if it is actually connected, so not -1
                    length = abs(node_idx - connected_node)
                    length_dict[(node_idx, qubit_idx, connected_node, (qubit_idx+1)%2)] = length
        if len(length_dict) == 0:
            # only increase the time step if ent gen is attempted
            if self.swap_action_time_step():
                max_length = 0
            elif self.ent_gen_time_step():
                max_length = 1
        else:
            max_length_key = max(length_dict, key=length_dict.get)
            assert length_dict[max_length_key] == length_dict[(max_length_key[2],max_length_key[3],max_length_key[0],max_length_key[1])] # if (node_i,qubit_j) has the longest link, then the (node_u,(qubit_j+1)%2) that it is connected to must also have the longest link
            max_length = length_dict[max_length_key]
        return max_length
    





    
    
