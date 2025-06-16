from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network
from qnetcc.TrainSim.MDPTrainSim import training_and_policy_simulation

# this is to simulate a few episode and going through the specific policy. 
if __name__ == "__main__":
    # network params
    nodes, t_cut_no_cc = 4, 2
    p, p_s = 1, 1

    # training params
    do_training = 1
    do_further_training = 0
    training_steps = int(3e6)

    # simulation params
    do_simulation = 1
    simulation_eps = int(2)

    t_cut_cc = t_cut_no_cc*Quantum_network(nodes, t_cut_no_cc, p, p_s).t_cut_cc_multiplier() # this assumes agent is at the center of the network
    t_cut = t_cut_cc
    train_and_sim = training_and_policy_simulation(nodes, t_cut, p, p_s, 
                                                    do_training, do_further_training, training_steps,
                                                    do_simulation, simulation_eps,
                                                    ent_reg= 0.0001,
                                                    agent_verbose=1, training_version_=24,
                                                    callback=1, save_times=100)
    train_and_sim.train_model()
    train_and_sim.simulate_policy_w_print()