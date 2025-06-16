###################
class Qubit(object):
    def __init__(self,connected_node,connection_age):
        if connected_node == 'none':
            connected_node = -1
        if connection_age == 'none':
            connection_age = -1
        self.connected_node = connected_node
        self.connection_age = connection_age

    # for connected qubit
    def get_connected_node(self):
        return self.connected_node # give the node that the qubit it is connected to is in
    
    def set_connected_node(self,connected_node):
        self.connected_node = connected_node
    # for connection age
    def get_connection_age(self):
        return self.connection_age 
    
    def set_connection_age(self,new_age):
        self.connection_age = new_age

    # check if qubit connected
    def is_entangled(self):
        if self.connected_node >= 0:
            assert self.connection_age >= 0
            entangled = True
        elif self.connection_age >= 0:
            assert self.connected_node >= 0
            entangled = True
        else:
            entangled = False
        return entangled


