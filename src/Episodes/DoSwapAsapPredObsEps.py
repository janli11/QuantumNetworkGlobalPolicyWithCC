from qnetcc.TrainSim.SwapAsapPredObsSim import swap_asap_simulation

if __name__ == "__main__":
    nodes, t_cut, p, p_s, simulation_eps = 4, 8, 0.9, 0.9, int(1)
    seed = 0
    swap_sim = swap_asap_simulation(nodes, t_cut, p, p_s, simulation_eps).simulate_policy_w_print(seed=seed)