import torch
import torch.nn as nn
import torch.nn.functional as F

import environment.episode as episode

import time
import logging
import random

# import vtrace
# import metric

"""
As RNaD is an actor criti
"""

class MLP(nn.Module):
    def __init__(self, max_actions, width, device=torch.device("cpu:0"), dtype=torch.float):
        """
        Parallel value and policy networks

        max_actions:
            same parameter that belongs to Tree
        width:
            number of activations in the hidden layer of both value and policy nets
        """
        super().__init__()
        self.device = device
        self.value_fc0 = nn.Linear(2 * max_actions**2, width, device=device, dtype=dtype)
        self.value_fc1 = nn.Linear(width, 1, device=device, dtype=dtype)
        self.policy_fc0 = nn.Linear(2 * max_actions**2, width, device=device, dtype=dtype)
        self.policy_fc1 = nn.Linear(width, max_actions, device=device, dtype=dtype)
        self.max_actions = max_actions
        self.width = width

    def forward(self, input_batch):
        filter_row = input_batch[:, 1, :, 0].to(torch.bool)
        # legal actions one hot
        input_batch = input_batch.view(-1, 2 * self.max_actions**2)
        # flatten observation tensor
        value = self.value_fc1(torch.relu(self.value_fc0(input_batch)))
        logits = self.policy_fc1(torch.relu(self.policy_fc0(input_batch)))

        exp_logits = torch.where(filter_row, torch.exp(logits), 0)
        policy = torch.nn.functional.normalize(exp_logits, dim=-1, p=1)
        # softmax of logits over legal actions

        actions = torch.squeeze(torch.multinomial(policy, num_samples=1))
        # sample action for each observation in batch
        return logits, policy, value, actions

    def forward_policy(self, input_batch: torch.Tensor) -> torch.Tensor:
        """
        Does not use value head but does perform legal actions masking
        """
        filter_row = input_batch[:, 1, :, 0].to(torch.bool)
        input_batch = input_batch.reshape(-1, 2 * self.max_actions**2)
        logits = self.policy_fc1(torch.relu(self.policy_fc0(input_batch)))
        exp_logits = torch.where(filter_row, torch.exp(logits), 0)
        policy = torch.nn.functional.normalize(exp_logits, dim=-1, p=1)
        return policy

    def forward_batch(self, episodes: episode.Episodes):

        logit_list, log_policy_list, policy_list, value_list = [], [], [], []
        for t in range(0, episodes.t_eff + 1):
            observations = episodes.observations[t]
            filter_row = observations[:, 1, :, 0].to(torch.bool)
            observations = observations.view(-1, 2 * self.max_actions**2)
            value = self.value_fc1(torch.relu(self.value_fc0(observations)))
            observations = observations.view(-1, 2 * self.max_actions**2)
            logits = self.policy_fc1(torch.relu(self.policy_fc0(observations)))
            exp_logits = torch.where(filter_row, torch.exp(logits), 0)
            policy = torch.nn.functional.normalize(exp_logits, dim=-1, p=1)
            log_sum = torch.log(torch.sum(exp_logits, dim=-1, keepdim=True))
            log_policy = torch.where(filter_row, logits - log_sum, 0)
            logit_list.append(logits)
            log_policy_list.append(log_policy)
            policy_list.append(policy)
            value_list.append(value)
        return [
            torch.stack(_, dim=0)
            for _ in (logit_list, log_policy_list, policy_list, value_list)
        ]


class CrossConv(nn.Module):
    def __init__(
        self,
        max_actions,
        in_channels,
        out_channels,
        device=torch.device("cpu:0"),
        dtype=torch.float,
    ):
        """
        Convolutional network designed for matrix structure. (The filters are the union of a row and column)
        max_actions:
            same parameter that belongs to Tree
        in_channels:
        out_channels:
            self explanatory
        """
        super().__init__()
        self.max_actions = max_actions
        self.row_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 2 * max_actions - 1),
            device=device,
            dtype=dtype,
        )
        self.col_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(2 * max_actions - 1, 1),
            device=device,
            dtype=dtype,
        )

    def forward(self, input) -> torch.Tensor:
        x = F.pad(
            input,
            (
                self.max_actions - 1,
                self.max_actions - 1,
                0,
                0,
            ),
        )
        r = self.row_conv(x)
        y = F.pad(
            input,
            (
                0,
                0,
                self.max_actions - 1,
                self.max_actions - 1,
            ),
        )
        c = self.col_conv(y)
        return r + c


class ConvResBlock(nn.Module):
    def __init__(
        self,
        max_actions,
        channels,
        batch_norm=False,
        device=torch.device("cpu:0"),
        dtype=torch.float,
    ):
        super().__init__()
        """
        Residual block for CrossConv (same structure as normal resblock)
        """
        self.conv0 = CrossConv(max_actions, channels, channels, device=device, dtype=dtype)
        self.conv1 = CrossConv(max_actions, channels, channels, device=device, dtype=dtype)
        self.relu = torch.relu
        if batch_norm:
            self.batch_norm0 = nn.BatchNorm2d(channels, device=device, dtype=dtype)
            self.batch_norm1 = nn.BatchNorm2d(channels, device=device, dtype=dtype)
        else:
            self.batch_norm0 = nn.Identity()
            self.batch_norm1 = nn.Identity()

    def forward(self, input_batch) -> torch.Tensor:
        return input_batch + self.batch_norm1(
            self.relu(self.conv1(self.batch_norm0(self.relu(self.conv0(input_batch)))))
        )


class ConvNet(nn.Module):
    def __init__(
        self,
        max_actions,
        channels,
        depth=1,
        batch_norm=True,
        device=torch.device("cpu:0"),
        dtype=torch.float,
    ):
        super().__init__()
        """
        Two headed convolutional tower (i.e. AlphaZero)
        """
        self.device = device
        self.dtype = dtype
        self.max_actions = max_actions
        self.channels = channels
        self.pre = CrossConv(
            max_actions, in_channels=2, out_channels=channels, device=device, dtype=dtype
        )
        self.tower = nn.ParameterList(
            [
                ConvResBlock(
                    max_actions=max_actions,
                    channels=channels,
                    batch_norm=batch_norm,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(depth)
            ]
        )
        self.policy = nn.Linear(
            channels * (max_actions**2), max_actions, device=device, dtype=dtype
        )
        self.value = nn.Linear(channels * (max_actions**2), 1, device=device, dtype=dtype)

    def forward(self, input_batch):
        filter_row = input_batch[:, 1, :, 0]
        x = input_batch
        x = self.pre(x)
        for block in self.tower:
            x = block.forward(x)

        x = x.view(-1, self.channels * (self.max_actions**2))
        logits = self.policy(x)
        policy = F.softmax(logits, dim=1)
        policy *= filter_row
        policy = F.normalize(policy, dim=1, p=1)
        value = self.value(x)
        actions = torch.squeeze(torch.multinomial(policy, num_samples=1))
        return logits, policy, value, actions

    def forward_policy(self, input_batch) -> torch.Tensor:
        """
        Does not use value head but does perform legal actions masking
        """
        filter_row = input_batch[:, 1, :, 0]
        x = input_batch
        x = self.pre(x)
        for block in self.tower:
            x = block(x)

        x = x.view(-1, self.channels * (self.max_actions**2))
        logits = self.policy(x)
        policy = F.softmax(logits, dim=1)
        policy *= filter_row
        policy = F.normalize(policy, dim=1, p=1)
        return policy

    def forward_batch(self, episodes: episode.Episodes):

        logit_list, log_policy_list, policy_list, value_list = [], [], [], []
        for t in range(0, episodes.t_eff + 1):
            observations = episodes.observations[t]
            x = self.pre(observations)
            for block in self.tower:
                x = block(x)
            x = x.view(-1, self.channels * (self.max_actions**2))
            logits = self.policy(x)
            filter_row = observations[:, 1, :, 0].to(torch.bool)
            exp_logits = torch.where(filter_row, torch.exp(logits), 0)
            log_sum = torch.log(torch.sum(exp_logits, dim=-1, keepdim=True))
            log_policy = torch.where(filter_row, logits - log_sum, 0)
            policy = torch.nn.functional.normalize(exp_logits, dim=-1, p=1)
            value = self.value(x)
            logit_list.append(logits)
            log_policy_list.append(log_policy)
            policy_list.append(policy)
            value_list.append(value)
        return [
            torch.stack(_, dim=0)
            for _ in (logit_list, log_policy_list, policy_list, value_list)
        ]