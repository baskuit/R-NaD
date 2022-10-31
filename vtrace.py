import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import math

from game import Episodes, States
import net


def transform_rewards (
    episodes : Episodes,
    net : net.ConvNet,
    net_reg : net.ConvNet, 
    net_reg_ : net.ConvNet,
    alpha=1,
    eta=.2) -> None:
    """
    Alters the player-one rewards tensor of an Episode given network and 
    two sequential regularization networks
    Two regularization policies gives us two transformed rewards, and the final reward is described
    as the interpolation between them.
    It happens to be that interpolation commutes with transormation, so we can just used the transormation given by the iterpolated policies.
    """
    time_start = time.perf_counter()
    for t in range(episodes.t_eff + 1):
        observations = episodes.observations[t]
        with torch.no_grad():
            pi = net.forward_policy(observations)
            pi_reg = net_reg.forward_policy(observations)
            pi_reg_= net_reg_.forward_policy(observations)
        pi_interp = alpha * pi_reg + (1 - alpha) * pi_reg_
        log_diff = torch.log(pi) - torch.log(pi_interp)
        log_diff_action = log_diff[torch.arange(episodes.batch_size), episodes.actions[t]]
        sign = -torch.pow(-1, episodes.turns[t])
        episodes.rewards[t] += sign * eta * log_diff_action
    time_end = time.perf_counter()
    episodes.transformation_time = time_end - time_start

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
    time_start = time.perf_counter()

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

        episodes.q_estimates[t] = episodes.values[t].unsqueeze(dim=1).repeat(1, episodes.tree.max_actions) - eta * log_policy_ratio
        if episodes.turns[t,0] == 0:
            delta_v_next_1 = roh_bar * (episodes.rewards[t] + r_hat_1[t+1] + v_next_1[t+1] - episodes.values[t])
            r_hat_1[t] = 0
            v_hat_1[t] = episodes.values[t] + delta_v_next_1 + c_bar *(v_hat_1[t+1] - v_next_1[t+1])
            v_next_1[t] = episodes.values[t]
            episodes.q_estimates[t] += \
                actions_one_hot * inverse_mu \
                    * ((episodes.rewards[t] - episodes.values[t] + r_hat_1[t+1] + v_hat_1[t+1]).unsqueeze(dim=1).repeat(1, episodes.tree.max_actions) \
                         + eta * log_policy_ratio)

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
                    * ((-episodes.rewards[t] - episodes.values[t] + r_hat_2[t+1] + v_hat_2[t+1]).unsqueeze(dim=1).repeat(1, episodes.tree.max_actions) \
                         + eta * log_policy_ratio)
            v_hat_1[t] = v_hat_1[t+1]
            v_next_1[t] = v_next_1[t+1]
            r_hat_1[t] = episodes.rewards[t] + r_hat_1[t+1]

    v_hat = torch.stack([v_hat_1, v_hat_2], dim=1)
    episodes.v_estimates = v_hat[torch.arange(episodes.t_eff+1), episodes.turns[:, 0]]

    # Now we stack the is_non_terminal tensors [t, b] and use these as indices
    is_non_terminal = torch.flatten(episodes.indices != 0)
    for key, value in episodes.__dict__.items():
        if torch.torch.is_tensor(value):
            value = torch.flatten(value, start_dim=0, end_dim=1)
            episodes.__dict__[key] = value[is_non_terminal]

    time_end = time.perf_counter()
    episodes.estimation_time = time_end - time_start

def learn (
    episodes : Episodes,
    net : net.ConvNet,
    batch_size=None,
    clip_neurd : float = 10_000,
    beta :float = 2,
    ):

    if batch_size is None:
        batch_size = episodes.batch_size

    #clamp Q values
    episodes.q_estimates = torch.where(
        torch.isnan(episodes.q_estimates),
        0,
        episodes.q_estimates
    )
    episodes.q_estimates = torch.clamp(episodes.q_estimates, min=-clip_neurd, max=clip_neurd)


    """
    Take generated, reward transformed, and v-trace estimated episodes batch and compute gradient updates for net
    """
    assert(episodes.values is not None)
    assert(len(episodes.values.shape) == 1)

    total_batch_size = episodes.values.shape[0]

    num_batches = math.ceil(total_batch_size / batch_size)

    for batch in range(num_batches):
        slice = torch.arange(batch * batch_size,
            min(total_batch_size, (batch + 1) * batch_size))
        num_slice = slice.shape[0]

        observations_slice = episodes.observations[slice] 
        logits, policy, value, actions = net.forward(observations_slice)

        logits_q = logits * episodes.q_estimates[slice]

        can_decrease = logits_q > -beta
        can_increase = logits_q <  beta

        # TODO is it logits or logits times Q values that are clipped
        logits_q_negative = logits_q.detach().clone()
        logits_q_positive = logits_q.detach().clone()
        logits_q_negative[logits_q > 0] = 0
        logits_q_positive[logits_q < 0] = 0
        logits_q_clipped = (can_decrease * logits_q_negative + can_increase * logits_q_positive).detach()

        logits_q_clipped *= episodes.masks[slice]
        policy_loss = -torch.sum(logits_q_clipped)
        value_loss = torch.sum(torch.abs(episodes.v_estimates[slice] - value))
        loss = value_loss + policy_loss


if __name__ == '__main__':
    import game 
    import net
    import logging
    logging.basicConfig(level=logging.DEBUG)

    tree = game.Tree()
    tree.load('recent')
    tree.to(torch.device('cuda'))

    logging.debug('Tree parameters :')
    for key, value in tree.__dict__.items():
        if torch.torch.is_tensor(value):
            value = value.shape
        logging.debug('{}: {}'.format(key, value))

    net_ = net.ConvNet(size=tree.max_actions, channels=2**5, device=tree.device)
    net_target = net.ConvNet(size=tree.max_actions, channels=2**5, device=tree.device)
    net_reg = net.ConvNet(size=tree.max_actions, channels=2**5, device=tree.device)
    net_reg_ = net.ConvNet(size=tree.max_actions, channels=2**5, device=tree.device)

    episodes = Episodes(tree, batch_size=10**5)
    episodes.generate(net_)

    episodes.display()

    transform_rewards(episodes, net_target, net_reg, net_reg_, .5, eta=.2)
    v_trace(episodes, net_target, net_reg, net_reg_, alpha=.5, eta=.2)

    episodes.display()

    learn(episodes, net_, batch_size=1000)