from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network 
from qnetcc.TrainSim.MDPTrainSim import training_and_policy_simulation as env_alt_hist_cc_train_sim

nodes, t_cut, p, p_s = 4, 2, 1, 1
training_version = 24

# training params 
do_training, do_further_training, training_steps, train_new_model = 1, 0, int(1e7), 0
trained_versions_start, trained_versions_stop = 20, 40 # cluster trained models
do_simulation, simulation_eps = 0, int(1e3)
MC_runs = int(simulation_eps)
time_out_mult = 2 # how many seconds and episode on average is allowed to take, before the Monte Carlo simulation is aborted

if __name__ == "__main__":
    # RL alt hist WITH CC
    t_cut_cc = t_cut*Quantum_network(nodes, t_cut, p, p_s).t_cut_cc_multiplier() # this assumes agent is at the center of the network
    cc_train_and_sim = env_alt_hist_cc_train_sim(nodes, t_cut_cc, p, p_s, 
                                                    do_training, do_further_training, training_steps,
                                                    do_simulation, simulation_eps, time_out_mult, 
                                                    new_training_version=train_new_model, training_version_=training_version,
                                                    callback=1, cluster=0, save_times=10)# save_times=int(training_steps/1000))
    sim_dat = cc_train_and_sim.do_training_and_simulation()