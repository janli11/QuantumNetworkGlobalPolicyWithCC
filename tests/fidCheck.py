from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network
from qnetcc.TrainSim.MDPTrainSim import training_and_policy_simulation

# this is to simulate a few episode and going through the specific policy. 
if __name__ == "__main__":
    # network params
    nodes, t_cut_no_cc = 4, 2
    p, p_s = 0.8, 0.75

    # training params
    do_training = 1
    do_further_training = 0
    # training_steps = int(5e6)
    training_steps = int(5e4)

    ##########################################################
    # For simulating an episode 
    #########################################################

    # simulation params
    do_simulation = 1
    simulation_eps = int(2)

    t_cut_cc = t_cut_no_cc*Quantum_network(nodes, t_cut_no_cc, p, p_s).t_cut_cc_multiplier() # this assumes agent is at the center of the network
    t_cut = t_cut_cc
    train_and_sim = training_and_policy_simulation(nodes, t_cut, p, p_s, 
                                                    do_training, do_further_training, training_steps,
                                                    do_simulation, simulation_eps,
                                                    ent_reg= 0.0001,
                                                    agent_verbose=1, training_version_=20,
                                                    callback=1, save_times=100, indSim=1)
    train_and_sim.simulate_policy_w_print()

    ##############################################
    # looking at delivery times and fidelity for many episodes
    #############################################

    # simulation params
    do_simulation = 1
    simulation_eps = int(20)

    t_cut_cc = t_cut_no_cc*Quantum_network(nodes, t_cut_no_cc, p, p_s).t_cut_cc_multiplier() # this assumes agent is at the center of the network
    t_cut = t_cut_cc
    train_and_sim = training_and_policy_simulation(nodes, t_cut, p, p_s, 
                                                    do_training, do_further_training, training_steps,
                                                    do_simulation, simulation_eps,
                                                    ent_reg= 0.0001,
                                                    agent_verbose=1, training_version_=20,
                                                    callback=1, save_times=100, indSim=1)
    simulation_delivery_times, T_list, micro_T_list, fidelity_list = train_and_sim.simulate_policy_w_fidelity()
    T = T_list
    print(f'delivery times : {T}')
    fid = [float(x) for x in fidelity_list]
    print(f'fidelity : {fid}')

