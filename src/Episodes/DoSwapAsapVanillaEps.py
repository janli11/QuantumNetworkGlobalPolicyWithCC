from qnetcc.TrainSim.SwapAsapVanillaSim import swap_asap_simulation

if __name__ == "__main__":
    nodes, t_cut, p, p_s, simulation_eps = 4, 16, 0.8, 0.8, int(1)
    seed = 8
    swap_sim = swap_asap_simulation(nodes, t_cut, p, p_s, simulation_eps).simulate_policy_w_print(seed=seed)