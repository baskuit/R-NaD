from game import Episodes, States

# assumes on-policy and roh bar, c bar = 1
def estimate (episodes : Episodes):

    roh_bar = 1
    c_bar = 1

    # estimate value and Q-function, assume pi = mu
    t_effective = episodes.turns.shape[0] - 1
    # _[id_] __{t_effective}
    v_1 = [0 for _ in range(t_effective+2)]
    v_2 = [0 for _ in range(t_effective+2)]
    V_1 = [0 for _ in range(t_effective+2)]
    V_2 = [0 for _ in range(t_effective+2)]
    r_1 = [0 for _ in range(t_effective+2)]
    r_2 = [0 for _ in range(t_effective+2)]
    zeta_1 = [1 for _ in range(t_effective+2)]
    zeta_2 = [1 for _ in range(t_effective+2)]
    for t in range(t_effective, -1, -1):
        # print(t)
        # print(episodes.observations[t])
        # print(episodes.values[t])
        if episodes.turns[t,0] == 1:
            v_1[t] = episodes.values[t]
            roh = zeta_1[t+1]
            c = zeta_1[t+1]
            delta_V_1 = roh * (episodes.rewards[t] + r_1[t+1] + V_1[t+1] - episodes.values[t])
            r_1[t] = 0
            zeta_1[t] = 1
            v_1[t] = episodes.values[t] + delta_V_1 + c *(v_1[t+1] - V_1[t+1])
            V_1[t+1] = episodes.values[t]

            v_2[t] = v_2[t+1]
            V_2[t] = V_2[t+1]
            r_2[t] = -episodes.rewards[t] + r_2[t+1] # remember rewards are p1's
            zeta_2[t] = zeta_2[t+1]
        else:
            v_2[t] = 0
            roh = zeta_2[t+1]
            c = zeta_2[t+1]
            delta_V_2 = roh * (-episodes.rewards[t] + r_1[t+1] + V_1[t+1] + episodes.values[t])
            r_1[t] = 0
            zeta_1[t] = 1
            v_1[t] = episodes.values[t] + delta_V_1 + c *(v_1[t+1] - V_1[t+1])
            V_1[t+1] = episodes.values[t]

            v_2[t] = v_2[t+1]
            V_2[t] = V_2[t+1]
            r_2[t] = episodes.rewards[t] + r_2[t+1]
            zeta_2[t] = zeta_2[t+1]