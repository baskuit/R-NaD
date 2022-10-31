from game import Episodes, States

import torch
import math
import net


def transform_rewards (
    episodes : Episodes,
    net : torch.nn.Module,
    net_reg : torch.nn.Module, 
    net_reg_ : torch.nn.Module,
    alpha=1,
    eta=.2) -> None:
    """
    Alters the player-one rewards tensor of an Episode given network and 
    two sequential regularization networks
    Two regularization policies gives us two transformed rewards, and the final reward is described
    as the interpolation between them.
    It happens to be that interpolation commutes with transormation, so we can just used the transormation given by the iterpolated policies.
    """
    for t in range(episodes.t_eff):
        observations = episodes.observations[t]
        with torch.no_grad():
            pi = net.forward_policy(observations)
            pi_reg = net_reg.forward_policy(observations)
            pi_reg_prev = net_reg_.forward_policy(observations)
        pi_interp = alpha * pi_reg + (1 - alpha) * pi_reg_prev
        log_diff = torch.log(pi) - torch.log(pi_interp)
        log_diff_action = log_diff[torch.arange(episodes.batch_size), episodes.actions[t]]
        sign = -torch.pow(-1, episodes.turns[t])
        episodes.rewards[t] += sign * eta * log_diff_action

def v_trace (
    episodes : Episodes,
    net : net.ConvNet,
    net_reg : net.ConvNet, 
    net_reg_ : net.ConvNet,
    alpha : float,
    eta : float,
    roh_bar=1,
    c_bar=1,

) -> None:

    """
    The estimates for the value and q-values are stored in the Episodes object

    ** Assumes the actor policy is the same as the estimating policy (net_target)**

    """
    table_shape = (episodes.t_eff + 2, episodes.batch_size)
    r_hat_1 = torch.zeros(table_shape, device=episodes.tree.device)
    r_hat_2 = torch.zeros(table_shape, device=episodes.tree.device)
    v_next_1 = torch.zeros(table_shape, device=episodes.tree.device)
    v_next_2 = torch.zeros(table_shape, device=episodes.tree.device)

    v_hat_1 = torch.zeros(table_shape, device=episodes.tree.device)
    v_hat_2 = torch.zeros(table_shape, device=episodes.tree.device)

    for t in range(episodes.t_eff, -1, -1):
        actions_one_hot = torch.zeros_like(episodes.policy[t])
        actions_one_hot[torch.arange(episodes.batch_size), episodes.actions[t]] = 1
        with torch.no_grad():
            policy = net.forward_policy(episodes.observations[t])
            policy_reg = net_reg.forward_policy(episodes.observations[t])
            policy_reg_ = net_reg_.forward_policy(episodes.observations[t])
        policy_reg_interp = alpha * policy_reg + (1 - alpha) * policy_reg_
        log_policy_ratio = torch.log(policy) - torch.log(policy_reg_interp)
        inverse_mu = 1 / episodes.policy[t]
        episodes.q_estimates[t] = episodes.values[t] - eta * log_policy_ratio

        if episodes.turns[t,0] == 0:
            delta_v_next_1 = roh_bar * (episodes.rewards[t] + r_hat_1[t+1] + v_next_1[t+1] - episodes.values[t])
            r_hat_1[t] = 0
            v_hat_1[t] = episodes.values[t] + delta_v_next_1 + c_bar *(v_hat_1[t+1] - v_next_1[t+1])
            v_next_1[t] = episodes.values[t]
            episodes.q_estimates[t] += \
                actions_one_hot * inverse_mu * (episodes.rewards[t] + eta * log_policy_ratio + r_hat_1[t+1] + v_hat_1[t+1] - episodes.values[t])

            v_hat_2[t] = v_hat_2[t+1]
            v_next_2[t] = v_next_2[t+1]
            r_hat_2[t] = -episodes.rewards[t] + r_hat_2[t+1]
        else:
            delta_v_next_2 = roh_bar * (-episodes.rewards[t] + r_hat_2[t+1] + v_next_2[t+1] - episodes.values[t])
            r_hat_2[t] = 0
            v_hat_2[t] = episodes.values[t] + delta_v_next_2 + c_bar * (v_hat_2[t+1] - v_next_2[t+1])
            v_next_2[t] = episodes.values[t]
            episodes.q_estimates[t] += \
                actions_one_hot * inverse_mu \
                     * (-episodes.rewards[t] + eta * log_policy_ratio + r_hat_2[t+1] + v_hat_2[t+1] \
                        - episodes.values[t])

            v_hat_1[t] = v_hat_1[t+1]
            v_next_1[t] = v_next_1[t+1]
            r_hat_1[t] = episodes.rewards[t] + r_hat_1[t+1]

    v_hat = torch.stack([v_hat_1, v_hat_2], dim=1)
    episodes.v_estimates = v_hat[torch.arange(episodes.t_eff+1), episodes.turns[:, 0]]
    
if __name__ == '__main__':
    import game 
    import net
    import logging
    logging.basicConfig(level=logging.DEBUG)

    tree = game.Tree()
    tree.load('recent')

    logging.debug('Tree parameters :')
    for key, value in tree.__dict__.items():
        if torch.torch.is_tensor(value):
            value = value.shape
        logging.debug('{}: {}'.format(key, value))

    net_ = net.ConvNet(size=tree.max_actions, channels=2**5)
    net_target = net.ConvNet(size=tree.max_actions, channels=2**5)
    net_reg = net.ConvNet(size=tree.max_actions, channels=2**5)
    net_reg_ = net.ConvNet(size=tree.max_actions, channels=2**5)

    episodes = Episodes(tree, batch_size=2)
    episodes.generate(net_)

    print('\ncompleted episode dict: \n')
    for key, value in episodes.__dict__.items():
        if torch.torch.is_tensor(value):
            if torch.numel(value) > 20:
                value = value.shape
        
        print(key, value)

    # transform_rewards(episodes, net_target, net_reg, net_reg_, .5, eta=.2)




    v_trace(episodes, net_target, net_reg, net_reg_, alpha=.5, eta=.2)
    print('\n\n\n')
    for key, value in episodes.__dict__.items():
        if torch.torch.is_tensor(value):
            if torch.numel(value) > 20:
                value = value.shape
        
        print(key, value)