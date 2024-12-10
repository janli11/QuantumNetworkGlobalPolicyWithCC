from qnetcc.environments.QNSimulator.qubit import Qubit 

class Node(object): 
    def __init__(self,label):
        self.label = label
        self.qubit_1 = Qubit(connected_node='none', connection_age='none')
        self.qubit_2 = Qubit(connected_node='none', connection_age='none')
    
    def set_label(self,new_label):
        self.label = new_label
    
    def get_label(self):
        return self.label

    def get_qubit(self,i):
        if i == 1:
            temp = self.qubit_1
        elif i == 2:
            temp = self.qubit_2
        else:
            assert i==1 or i==2
        return temp
    
    def set_qubit(self,i,connected_node,connection_age):
        if i == 1:
            self.qubit_1 = Qubit(connected_node, connection_age)
        elif i == 2:
            self.qubit_2 = Qubit(connected_node, connection_age)
        else:
            assert i==1 or i==2
        
    def is_swapable(self):
        if self.qubit_1.get_connection_age() >= 0 and self.qubit_2.get_connection_age() >= 0:
            return True
        else:
            assert self.qubit_1.get_connected_node() <= 0 or self.qubit_1.get_connected_node() == 'none'
            assert self.qubit_2.get_connected_node() <= 0 or self.qubit_2.get_connected_node() == 'none'
            return False
        

    def reset(self,label):
        self.label = label
        self.qubit_1 = Qubit(connected_node='none', connection_age='none')
        self.qubit_2 = Qubit(connected_node='none', connection_age='none')

    def reset_qubit(self,i):
        if i == 1:
            self.qubit_1 = Qubit(connected_node='none', connection_age='none')
        elif i == 2:
            self.qubit_2 = Qubit(connected_node='none', connection_age='none')
        else:
            assert i==1 or i==2

    def update_qubit_age(self,qubit_i):
        if qubit_i == 1:
            new_age = self.qubit_1.get_connection_age() + 1
            self.qubit_1.set_connection_age(new_age)
        elif qubit_i == 2:
            new_age = self.qubit_2.get_connection_age() + 1
            self.qubit_2.set_connection_age(new_age)
        else:
            assert qubit_i==1 or qubit_i==2
