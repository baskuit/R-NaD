import torch
import torch.nn as nn
import torch.nn.functional as F

import batch

import time
import logging
import random

# import vtrace
# import metric


class MLP(nn.Module):
    def __init__(self, size, width, device=torch.device("cpu:0"), dtype=torch.float):
        super().__init__()
        self.device = device
        self.value_fc0 = nn.Linear(2 * size**2, width, device=device, dtype=dtype)
        self.value_fc1 = nn.Linear(width, 1, device=device, dtype=dtype)
        self.policy_fc0 = nn.Linear(2 * size**2, width, device=device, dtype=dtype)
        self.policy_fc1 = nn.Linear(width, size, device=device, dtype=dtype)
        self.size = size
        self.width = width

    def forward(self, input_batch):
        filter_row = input_batch[:, 1, :, 0].to(torch.bool)
        input_batch = input_batch.view(-1, 2 * self.size**2)
        value = self.value_fc1(torch.relu(self.value_fc0(input_batch)))
        logits = self.policy_fc1(torch.relu(self.policy_fc0(input_batch)))
        exp_logits = torch.where(filter_row, torch.exp(logits), 0)
        policy = torch.nn.functional.normalize(exp_logits, dim=-1, p=1)
        actions = torch.squeeze(torch.multinomial(policy, num_samples=1))
        return logits, policy, value, actions

    def forward_policy(self, input_batch: torch.Tensor) -> torch.Tensor:
        """
        Does not use value head but does perform legal actions masking
        """
        filter_row = input_batch[:, 1, :, 0].to(torch.bool)
        input_batch = input_batch.reshape(-1, 2 * self.size**2)
        logits = self.policy_fc1(torch.relu(self.policy_fc0(input_batch)))
        exp_logits = torch.where(filter_row, torch.exp(logits), 0)
        policy = torch.nn.functional.normalize(exp_logits, dim=-1, p=1)
        return policy

    def forward_batch(self, episodes: batch.Episodes):

        logit_list, log_policy_list, policy_list, value_list = [], [], [], []
        for t in range(0, episodes.t_eff + 1):
            observations = episodes.observations[t]
            filter_row = observations[:, 1, :, 0].to(torch.bool)
            observations = observations.view(-1, 2 * self.size**2)
            value = self.value_fc1(torch.relu(self.value_fc0(observations)))
            observations = observations.view(-1, 2 * self.size**2)
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
        size,
        in_channels,
        out_channels,
        device=torch.device("cpu:0"),
        dtype=torch.float,
    ):
        super().__init__()
        self.size = size
        self.row_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 2 * size - 1),
            device=device,
            dtype=dtype,
        )
        self.col_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(2 * size - 1, 1),
            device=device,
            dtype=dtype,
        )

    def forward(self, input) -> torch.Tensor:
        x = F.pad(
            input,
            (
                self.size - 1,
                self.size - 1,
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
                self.size - 1,
                self.size - 1,
            ),
        )
        c = self.col_conv(y)
        return r + c


class ConvResBlock(nn.Module):
    def __init__(
        self,
        size,
        channels,
        batch_norm=False,
        device=torch.device("cpu:0"),
        dtype=torch.float,
    ):
        super().__init__()
        self.conv0 = CrossConv(size, channels, channels, device=device, dtype=dtype)
        self.conv1 = CrossConv(size, channels, channels, device=device, dtype=dtype)
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
        size,
        channels,
        depth=1,
        batch_norm=True,
        device=torch.device("cpu:0"),
        dtype=torch.float,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.size = size
        self.channels = channels
        self.pre = CrossConv(
            size, in_channels=2, out_channels=channels, device=device, dtype=dtype
        )
        self.tower = nn.ParameterList(
            [
                ConvResBlock(
                    size=size,
                    channels=channels,
                    batch_norm=batch_norm,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(depth)
            ]
        )
        self.policy = nn.Linear(
            channels * (size**2), size, device=device, dtype=dtype
        )
        self.value = nn.Linear(channels * (size**2), 1, device=device, dtype=dtype)

    def forward(self, input_batch):
        filter_row = input_batch[:, 1, :, 0]
        x = input_batch
        x = self.pre(x)
        for block in self.tower:
            x = block.forward(x)

        x = x.view(-1, self.channels * (self.size**2))
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

        x = x.view(-1, self.channels * (self.size**2))
        logits = self.policy(x)
        policy = F.softmax(logits, dim=1)
        policy *= filter_row
        policy = F.normalize(policy, dim=1, p=1)
        return policy

    def forward_batch(self, episodes: batch.Episodes):

        logit_list, log_policy_list, policy_list, value_list = [], [], [], []
        for t in range(0, episodes.t_eff + 1):
            observations = episodes.observations[t]
            x = self.pre(observations)
            for block in self.tower:
                x = block(x)
            x = x.view(-1, self.channels * (self.size**2))
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


if __name__ == "__main__":

    import game
    import metric

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def speed_test(net):
        start_generating = time.perf_counter()
        # Speed test
        batch_size = 10**4
        steps = 0
        for trial in range(100):
            episodes = batch.Episodes(tree, batch_size)
            episodes.generate(net)
            steps += episodes.t_eff * batch_size

        end_generating = time.perf_counter()
        speed = steps / (end_generating - start_generating)
        logging.debug("{} steps/sec".format(int(speed)))

    logging.basicConfig(level=logging.DEBUG)

    depth_bound_lambda = lambda tree: max(
        0, tree.depth_bound - (1 if random.random() < 0.5 else 2)
    )

    # tree = game.Tree(
    #     max_actions=2,
    #     depth_bound=2,
    #     max_transitions=2,
    #     # depth_bound_lambda=depth_bound_lambda
    # )

    # tree._generate()
    # print('tree generated, size: ', tree.value.shape)
    # tree.to(torch.device('cuda:0'))

    # batch_size = 2**1
    # net_channels=32
    # net_depth=1
    net = ConvNet(size=3, channels=2**5, depth=1, device=torch.device("cuda"))
    net_ = MLP(size=3, width=2**8, device=torch.device("cuda"))
    print(count_parameters(net_))

    # expl = metric.nash_conv(tree, net)
    # print(expl)
