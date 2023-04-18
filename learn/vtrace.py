# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code was modified from the original OpenSpiel distribution 
# to make it pytorch compatible, as it was written for Tensorflow

import torch
import functools
import numpy as np

from typing import Any, Callable, NamedTuple, Sequence, Tuple

def process_policy(
    policy: torch.Tensor, mask: torch.Tensor, n_disc, epsilon_threshold=0.03
):

    t_eff, batch_size, n_actions = policy.shape
    policy = torch.flatten(policy, 0, 1)
    mask = torch.flatten(mask, 0, 1)
    new_batch_range = torch.arange(t_eff * batch_size)

    # threshold
    mask = mask * (
        (policy >= epsilon_threshold)
        + (
            torch.max(policy, dim=-1, keepdim=True).values < epsilon_threshold
        )  # prevent degen case where all < eps)
    )
    policy = mask * policy / torch.sum(mask * policy, dim=-1, keepdim=True)

    # discretize
    blocks = torch.ceil(n_disc * policy).to(torch.int32)
    result = torch.zeros_like(policy)
    leftover = n_disc * torch.ones((policy.shape[0]), device=policy.device)
    order = torch.argsort(policy, descending=True)
    for i in range(n_actions):
        block = blocks[new_batch_range, order[:, i]]
        x = torch.minimum(leftover, block)
        leftover -= x
        result[new_batch_range, order[:, i]] += x
    result /= n_disc
    policy = policy.view(t_eff, batch_size, n_actions)
    mask = mask.view(t_eff, batch_size, n_actions)
    return result.view(t_eff, batch_size, n_actions)


class LoopVTraceCarry(NamedTuple):
    """The carry of the v-trace scan loop."""

    reward: torch.Tensor
    # The cumulated reward until the end of the episode. Uncorrected (v-trace).
    # Gamma discounted and includes eta_reg_entropy.
    reward_uncorrected: torch.Tensor
    next_value: torch.Tensor
    next_v_target: torch.Tensor
    importance_sampling: torch.Tensor


def _player_others(
    player_ids: torch.Tensor, valid: torch.Tensor, player: int
) -> torch.Tensor:
    """A vector of 1 for the current player and -1 for others.

    Args:
      player_ids: Tensor [...] containing player ids (0 <= player_id < N).
      valid: Tensor [...] containing whether these states are valid.
      player: The player id as int.

    Returns:
      player_other: is 1 for the current player and -1 for others [..., 1].
    """
    current_player_tensor = player_ids == player

    res = 2 * current_player_tensor - 1
    res = res * valid
    return torch.unsqueeze(res, dim=-1)


def _where(
    pred: torch.Tensor, true_data: torch.Tensor, false_data: torch.Tensor
) -> torch.Tensor:
    """Similar to jax.where but treats `pred` as a broadcastable prefix."""

    def _where_one(t, f):
        if (
            isinstance(t, LoopVTraceCarry)
            or isinstance(t, tuple)
            or isinstance(t, list)
        ):
            res = [_where_one(ts, fs) for ts, fs in zip(t, f)]
            if isinstance(t, LoopVTraceCarry):
                return LoopVTraceCarry(*res)
            else:
                return res
        else:
            # Expand the dimensions of pred if true_data and false_data are higher rank.
            p = torch.reshape(
                pred, pred.shape + (1,) * (len(t.shape) - len(pred.shape))
            ).to(torch.bool)
            return torch.where(p, t, f)

    output = [_where_one(td, fd) for td, fd in zip(true_data, false_data)]
    return output


def scan(f: Callable, init, xs, length=None, reverse: bool = False):
    if xs is None:
        xs = [torch.zeros((1,))] * length
    carry = init
    ys = []

    if reverse:
        xs = list(reversed(list(zip(*xs))))

    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)

    if reverse:
        ys = list(reversed(ys))

    if isinstance(ys[0], list):
        res = [torch.stack([n[i] for n in ys]) for i in range(len(ys[0]))]
    else:
        res = torch.stack(ys)

    return carry, res


def _has_played(
    valid: torch.Tensor, player_id: torch.Tensor, player: int
) -> torch.Tensor:
    """Compute a mask of states which have a next state in the sequence."""
    assert valid.shape == player_id.shape

    def _loop_has_played(carry, x):
        valid, player_id = x
        assert valid.shape == player_id.shape

        our_res = torch.ones_like(player_id)
        opp_res = carry
        reset_res = torch.zeros_like(carry)

        our_carry = carry
        opp_carry = carry
        reset_carry = torch.zeros_like(player_id)

        # pyformat: disable
        return _where(
            valid,
            _where(
                (player_id == player),
                (our_carry, our_res),
                (opp_carry, opp_res),
            ),
            (reset_carry, reset_res),
        )
        # pyformat: enable

    _, result = scan(
        f=_loop_has_played,
        init=torch.zeros_like(player_id[-1]),
        xs=(valid, player_id),
        reverse=True,
    )
    return result


def _policy_ratio(
    pi: torch.Tensor, mu: torch.Tensor, actions_oh: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Returns a ratio of policy pi/mu when selecting action a.

    By convention, this ratio is 1 on non valid states
    Args:
      pi: the policy of shape [..., A].
      mu: the sampling policy of shape [..., A].
      actions_oh: a one-hot encoding of the current actions of shape [..., A].
      valid: 0 if the state is not valid and else 1 of shape [...].

    Returns:
      pi/mu on valid states and 1 otherwise. The shape is the same
      as pi, mu or actions_oh but without the last dimension A.
    """
    assert pi.shape == mu.shape == actions_oh.shape
    # assert ((valid,), actions_oh.shape[:-1])

    def _select_action_prob(pi: torch.Tensor) -> torch.Tensor:
        return torch.sum(actions_oh * pi, dim=-1) * valid + (1 - valid)

    pi_actions_prob = _select_action_prob(pi)
    mu_actions_prob = _select_action_prob(mu)
    return pi_actions_prob / mu_actions_prob


def v_trace(
    v: torch.Tensor,
    valid: torch.Tensor,
    player_id: torch.Tensor,
    acting_policy: torch.Tensor,
    merged_policy: torch.Tensor,
    merged_log_policy: torch.Tensor,
    player_others: torch.Tensor,
    actions_oh: torch.Tensor,
    reward: torch.Tensor,
    player: int,
    # Scalars below.
    eta: float,
    lambda_: float,
    c: float,
    rho: float,
    gamma=1.0,
) -> Tuple[Any, Any, Any]:
    """Custom VTrace for trajectories with a mix of different player steps."""

    has_played = _has_played(valid, player_id, player)

    policy_ratio = _policy_ratio(merged_policy, acting_policy, actions_oh, valid)
    inv_mu = _policy_ratio(
        torch.ones_like(merged_policy), acting_policy, actions_oh, valid
    )

    eta_reg_entropy = (
        -eta
        * torch.sum(merged_policy * merged_log_policy, dim=-1)
        * torch.squeeze(player_others, dim=-1)
    )
    eta_log_policy = -eta * merged_log_policy * player_others

    init_state_v_trace = LoopVTraceCarry(
        reward=torch.zeros_like(reward[-1]),
        reward_uncorrected=torch.zeros_like(reward[-1]),
        next_value=torch.zeros_like(v[-1]),
        next_v_target=torch.zeros_like(v[-1]),
        importance_sampling=torch.ones_like(policy_ratio[-1]),
    )

    def _loop_v_trace(carry: LoopVTraceCarry, x) -> Tuple[LoopVTraceCarry, Any]:
        (
            cs,
            player_id,
            v,
            reward,
            eta_reg_entropy,
            valid,
            inv_mu,
            actions_oh,
            eta_log_policy,
        ) = x

        reward_uncorrected = reward + gamma * carry.reward_uncorrected + eta_reg_entropy
        discounted_reward = reward + gamma * carry.reward

        # V-target:
        our_v_target = (
            v
            + torch.unsqueeze(
                torch.clamp(cs * carry.importance_sampling, max=rho), dim=-1
            )
            * (
                torch.unsqueeze(reward_uncorrected, dim=-1)
                + gamma * carry.next_value
                - v
            )
            + lambda_
            * torch.unsqueeze(
                torch.clamp(cs * carry.importance_sampling, max=c), dim=-1
            )
            * gamma
            * (carry.next_v_target - carry.next_value)
        )

        opp_v_target = torch.zeros_like(our_v_target)
        reset_v_target = torch.zeros_like(our_v_target)

        # Learning output:
        our_learning_output = (
            v
            + eta_log_policy  # value
            + actions_oh  # regularisation
            * torch.unsqueeze(inv_mu, dim=-1)
            * (
                torch.unsqueeze(discounted_reward, dim=-1)
                + gamma
                * torch.unsqueeze(carry.importance_sampling, dim=-1)
                * carry.next_v_target
                - v
            )
        )

        opp_learning_output = torch.zeros_like(our_learning_output)
        reset_learning_output = torch.zeros_like(our_learning_output)

        # State carry:
        our_carry = LoopVTraceCarry(
            reward=torch.zeros_like(carry.reward),
            next_value=v,
            next_v_target=our_v_target,
            reward_uncorrected=torch.zeros_like(carry.reward_uncorrected),
            importance_sampling=torch.ones_like(carry.importance_sampling),
        )
        opp_carry = LoopVTraceCarry(
            reward=eta_reg_entropy + cs * discounted_reward,
            reward_uncorrected=reward_uncorrected,
            next_value=gamma * carry.next_value,
            next_v_target=gamma * carry.next_v_target,
            importance_sampling=cs * carry.importance_sampling,
        )
        reset_carry = init_state_v_trace

        # Invalid turn: init_state_v_trace and (zero target, learning_output)
        # pyformat: disable
        return _where(
            valid,
            _where(
                (player_id == player),
                (our_carry, (our_v_target, our_learning_output)),
                (opp_carry, (opp_v_target, opp_learning_output)),
            ),
            (reset_carry, (reset_v_target, reset_learning_output)),
        )
        # pyformat: enable

    _, (v_target, learning_output) = scan(
        f=_loop_v_trace,
        init=init_state_v_trace,
        xs=(
            policy_ratio,
            player_id,
            v,
            reward,
            eta_reg_entropy,
            valid,
            inv_mu,
            actions_oh,
            eta_log_policy,
        ),
        reverse=True,
    )

    return v_target, has_played, learning_output


def apply_force_with_threshold(
    decision_outputs: torch.Tensor,
    force: torch.Tensor,
    threshold: float,
    threshold_center: torch.Tensor,
) -> torch.Tensor:
    """Apply the force with below a given threshold."""
    can_decrease = decision_outputs - threshold_center > -threshold
    can_increase = decision_outputs - threshold_center < threshold
    force_negative = torch.clamp(force, max=0.0)
    force_positive = torch.clamp(force, min=0.0)
    clipped_force = can_decrease * force_negative + can_increase * force_positive
    return decision_outputs * clipped_force.detach()


def renormalize(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The `normalization` is the number of steps over which loss is computed."""
    loss = torch.sum(loss * mask)
    normalization = torch.sum(mask)
    return loss / (normalization + (normalization == 0.0))


def get_loss_v(
    v_list: Sequence[torch.Tensor],
    v_target_list: Sequence[torch.Tensor],
    mask_list: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Define the loss function for the critic."""
    loss_v_list = []
    for (v_n, v_target, mask) in zip(v_list, v_target_list, mask_list):
        assert v_n.shape[0] == v_target.shape[0]

        loss_v = torch.unsqueeze(mask, dim=-1) * (v_n - v_target.detach()) ** 2
        normalization = torch.sum(mask)
        loss_v = torch.sum(loss_v) / (normalization + (normalization == 0.0))

        loss_v_list.append(loss_v)

    return sum(loss_v_list)


def get_loss_nerd(
    logit_list: Sequence[torch.Tensor],
    policy_list: Sequence[torch.Tensor],
    q_vr_list: Sequence[torch.Tensor],
    valid: torch.Tensor,
    player_ids: Sequence[torch.Tensor],
    legal_actions: torch.Tensor,
    importance_sampling_correction: Sequence[torch.Tensor],
    clip: float = 100,
    threshold: float = 2,
) -> torch.Tensor:
    """Define the nerd loss."""
    assert isinstance(importance_sampling_correction, list)
    loss_pi_list = []
    for k, (logit_pi, pi, q_vr, is_c) in enumerate(
        zip(logit_list, policy_list, q_vr_list, importance_sampling_correction)
    ):
        assert logit_pi.shape[0] == q_vr.shape[0]
        # loss policy
        adv_pi = q_vr - torch.sum(pi * q_vr, dim=-1, keepdim=True)
        adv_pi = is_c * adv_pi  # importance sampling correction
        adv_pi = torch.clip(adv_pi, min=-clip, max=clip)
        adv_pi = adv_pi.detach()

        logits = logit_pi - torch.mean(logit_pi * legal_actions, dim=-1, keepdim=True)

        threshold_center = torch.zeros_like(logits)

        nerd_loss = torch.sum(
            legal_actions
            * apply_force_with_threshold(logits, adv_pi, threshold, threshold_center),
            axis=-1,
        )
        nerd_loss = -renormalize(nerd_loss, valid * (player_ids == k))
        loss_pi_list.append(nerd_loss)
    return sum(loss_pi_list)
