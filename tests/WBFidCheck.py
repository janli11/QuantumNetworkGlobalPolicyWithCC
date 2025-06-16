# To check the end-to-end fidelity of the WB swap-asap

import numpy as np
from qnetcc.TrainSim.SwapAsapVanillaSim import swap_asap_simulation

# this is to simulate a few episode and going through the specific policy. 
if __name__ == "__main__":
    # network params
    nodes, t_cut = 4, 2
    p, p_s = 0.8, 0.75

    # training params
    do_training = 1
    do_further_training = 0
    # training_steps = int(5e6)
    training_steps = int(5e4)

    ##########################################################
    # For simulating an episode 
    #########################################################

    simulation_eps = int(1)
    seed = np.random.randint(0, int(1e6))
    swap_sim = swap_asap_simulation(nodes, t_cut, p, p_s, simulation_eps).simulate_policy_w_print(seed=seed)

    ##############################################
    # looking at delivery times and fidelity for many episodes
    #############################################

    # simulation params
    do_simulation = 1
    simulation_eps = int(20)

    np.random.seed(None)  
    T_list, micro_T_list, fidelity_list = swap_asap_simulation(nodes, t_cut, p, p_s, simulation_eps).simulate_policy_w_fidelity()
    
    T = T_list
    print(f'delivery times : {T}')
    fid = [float(x) for x in fidelity_list]
    print(f'fidelity : {fid}')