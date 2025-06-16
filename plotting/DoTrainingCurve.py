from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network 
import matplotlib.pyplot as plt
from qnetcc.TrainSim.MDPTrainSim import training_and_policy_simulation as env_alt_hist_cc_train_sim

############################################################################
# The training and simulation part
############################################################################

# setting of the script 
# Running this script will produce plots of the delivery time vs the EG succes probability at fixed values for
# the number of nodes, swap succes probability and cut-off time. 

# training params 
do_training, do_further_training, training_steps, train_new_model = 1, 0, int(1e7), 0
do_simulation, simulation_eps = 0, int(1e3)
MC_runs = int(simulation_eps)
time_out_mult = 2 # how many seconds and episode on average is allowed to take, before the Monte Carlo simulation is aborted
# Callback
Callback=1 

# Quantum Network parameters
nodes_list = [4] 
t_cut_list = [2] # t_cut is this factor multiplied by the number of nodes, this is for taking the sum

# HPC trained model parameters
# p_list = np.linspace(1, 0.1, 10)
# p_s_list = [1, 0.75, 0.5]

# for ps=1 point where RL stops finding good policy
p_list = [0.3]
p_s_list = [1]

# for ps=0.5 point where RL stops finding good policy
p_list = [0.5]
p_s_list = [0.5]

# training_version_list = list(range(35,40))
training_version_list = list(range(20,40))
# training_version_list = [203]

if __name__ == "__main__":

    # RL alt hist WITH CC
    for nodes in nodes_list:
        for t_cut_no_cc in t_cut_list:
            for p in p_list:
                for p_s in p_s_list:
                    print(f'alt history cc')
                    t_cut_cc = t_cut_no_cc*Quantum_network(nodes, t_cut_no_cc, p, p_s).t_cut_cc_multiplier() # this assumes agent is at the center of the network
                    print(f'nodes {nodes}, t_cut {t_cut_cc}, p_e {p}, p_s {p_s}')
                    for training_version in training_version_list:
                        # make the best model so far plot
                        cc_train_and_sim = env_alt_hist_cc_train_sim(nodes, t_cut_cc, p, p_s, 
                                                                                do_training, do_further_training, training_steps,
                                                                                do_simulation, simulation_eps, time_out_mult,
                                                                                new_training_version = train_new_model, training_version_=training_version,
                                                                                callback=1)
                        cc_train_and_sim.return_best_models_so_far_plot()
                        plt.close('all') # always close the figure at the end