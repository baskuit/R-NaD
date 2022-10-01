from game import Episodes, States

import torch
import math

def transform_rewards (
    episodes : Episodes,
    net : torch.nn.Module,
    net_reg : torch.nn.Module, 
    net_reg_prev : torch.nn.Module, 
    eta=.2):
    # alpha is the interpolation function with 'n' implicit
    for t in range(episodes.t_eff):
        observations = episodes.observations[t]
        with torch.no_grad():
            pi = net.forward_policy(observations)
            pi_reg = net_reg.forward_policy(observations)
            pi_reg_prev = net_reg_prev.forward_policy(observations)
        # pi_reg_interp = alpha(pi_reg, pi_reg_prev)
        pi_reg_interp = pi_reg #temp

        for b in range(episodes.batch_size):
            turn = episodes.turns[t, b]
            a = episodes.actions[t, b]
            episodes.rewards[t, b] -= (-1)**turn * eta * (math.log(pi[b, a]) - math.log(pi_reg_interp[b, a]))





# assumes on-policy and roh  bar, c bar = 1
def estimate (episodes : Episodes, reward_transform, roh_bar = 1, c_bar = 1):

    # estimate value and Q-function, assume pi = mu
    t_effective = episodes.turns.shape[0] - 1
    # _[id_] __{t_effective}
    v_1 = [0 for _ in range(t_effective+2)]
    v_2 = [0 for _ in range(t_effective+2)]
    V_1 = [0 for _ in range(t_effective+2)]
    V_2 = [0 for _ in range(t_effective+2)]
    r_1 = [0 for _ in range(t_effective+2)]
    r_2 = [0 for _ in range(t_effective+2)]
    roh = min(roh_bar, 1)
    c = min(c_bar, 1)
    for t in range(t_effective, -1, -1):

        if episodes.turns[t,0] == 0:
            delta_V_1 = roh * (episodes.rewards[t] + r_1[t+1] + V_1[t+1] - episodes.values[t])
            r_1[t] = 0
            v_1[t] = episodes.values[t] + delta_V_1 + c *(v_1[t+1] - V_1[t+1])
            V_1[t] = episodes.values[t]

            v_2[t] = v_2[t+1]
            V_2[t] = V_2[t+1]
            r_2[t] = -episodes.rewards[t] + r_2[t+1]
        else:
            delta_V_2 = roh * (-episodes.rewards[t] + r_2[t+1] + V_2[t+1] - episodes.values[t])
            r_2[t] = 0
            v_2[t] = episodes.values[t] + delta_V_2 + c *(v_2[t+1] - V_2[t+1])
            V_2[t] = episodes.values[t]

            v_1[t] = v_1[t+1]
            V_1[t] = V_1[t+1]
            r_1[t] = episodes.rewards[t] + r_2[t+1]

    # v_1 = torch.tensor(v_1)
    # v_2 = torch.tensor(v_2)
    # V_1 = torch.tensor(V_1)
    # V_2 = torch.tensor(V_2)
    # r_1 = torch.tensor(r_1)
    # r_2 = torch.tensor(r_2)
    

    for t in range(t_effective+1):
        print('\n\nt: ', t)
        print('v')
        print(v_1[t])
        print(v_2[t])
        # print('V_')
        # print(V_1[t])
        # print(V_2[t])
        print('r')
        print(r_1[t])
        print(r_2[t])
