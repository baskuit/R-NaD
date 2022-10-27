from game import Episodes, States

import torch
import math

def transform_rewards (
    episodes : Episodes,
    net : torch.nn.Module,
    net_reg : torch.nn.Module, 
    net_reg_prev : torch.nn.Module,
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
            pi_reg_prev = net_reg_prev.forward_policy(observations)
        pi_interp = alpha * pi_reg + (1 - alpha) * pi_reg_prev
        sign = -torch.pow(-1, episodes.turns[t])
        episodes.rewards[t] += sign * eta * (torch.log(pi) - torch.log(pi_interp))