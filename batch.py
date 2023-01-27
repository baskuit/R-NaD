import torch
import torch.nn as nn
import torch.nn.functional as F

from game import Tree
from collections import deque

import time
import random
import numpy


class States:
    def __init__(self, tree: Tree, batch_size):
        self.tree = tree
        self.batch_size = batch_size
        self.indices = torch.ones((batch_size,), dtype=torch.int32, device=tree.device)
        self.turns = torch.zeros(
            (batch_size,), dtype=torch.long, device=tree.device
        )  # indices along player dim
        self.actions_1 = None
        self.actions_2 = None
        self.terminal = False

    def observations(self) -> torch.Tensor:
        expected_value = torch.index_select(self.tree.expected_value, 0, self.indices)
        legal = torch.index_select(self.tree.legal, 0, self.indices)
        observations_1 = torch.cat([expected_value, legal], dim=1)
        observations_2 = torch.cat([-expected_value, legal], dim=1).swapaxes(2, 3)
        observations = torch.stack([observations_1, observations_2], dim=1)
        observations = observations[torch.arange(self.batch_size), self.turns]
        return observations

    def observations_noisy(
        self,
    ) -> torch.Tensor:
        """
        Instead of simply returning the expected value matrix from the pov of the player,
        we now transform the image so it is both high dimensional and noisy
        """

    def step(self, actions) -> torch.Tensor:
        if self.turns[0] == 0:
            self.actions_1 = actions
            self.turns = 1 - self.turns
            rewards_1 = torch.zeros((self.batch_size,), device=self.tree.device)
        else:
            self.actions_2 = actions
            self.turns = 1 - self.turns

            index = torch.index_select(self.tree.index, 0, self.indices)
            chance = torch.index_select(self.tree.chance, 0, self.indices)
            value = torch.index_select(self.tree.value, 0, self.indices)
            chance_entry = chance[
                torch.arange(self.batch_size), :, self.actions_1, self.actions_2
            ]
            index_entry = index[
                torch.arange(self.batch_size), :, self.actions_1, self.actions_2
            ]
            value_entry = value[
                torch.arange(self.batch_size), :, self.actions_1, self.actions_2
            ]
            actions_chance = torch.multinomial(chance_entry, num_samples=1).squeeze()
            self.indices = index_entry[torch.arange(self.batch_size), actions_chance]
            rewards_1 = value_entry[torch.arange(self.batch_size), actions_chance]
            rewards_1 *= self.indices == 0
            self.actions_1 = None
            self.actions_2 = None
        self.terminal = torch.all(self.indices == 0).item()
        return rewards_1


class Episodes:
    """
    This represents a sequence of States, with dim=0 being time
    """

    def __init__(self, tree: Tree, batch_size):
        self.tree = tree
        self.batch_size = batch_size
        self.states = States(tree, batch_size)
        self.generated = False
        self.generation_time = 0
        self.estimation_time = 0

        self.turns = None
        self.indices = None
        self.observations = None
        self.policy = None
        self.actions = None
        self.rewards = None
        self.values = None
        self.masks = None
        self.t_eff = -1

        self.q_estimates = None
        self.v_estimates = None

    @classmethod
    def collate(cls, lst: list["Episodes"]):

        t_eff = max(e.t_eff for e in lst)
        tree = lst[0].tree
        batch_size = sum(e.batch_size for e in lst)
        assert all(e.tree == tree for e in lst)
        assert all(e.generated for e in lst)
        """
        Pad each member of Episodes along time dimension. 0's for padding works for everything.
        valid is determined by indices==0, thus other padded tensors aren't used  
        """
        result = Episodes(tree, batch_size)
        for key, value in lst[0].__dict__.items():
            if torch.is_tensor(value):
                result.__dict__[key] = torch.cat(
                    [
                        F.pad(
                            e.__dict__[key].T,
                            (
                                0,
                                t_eff - e.t_eff,
                            ),
                        ).T
                        for e in lst
                    ],
                    dim=1,
                )
        result.generated = True
        result.t_eff = t_eff
        return result

    def generate(self, net: torch.nn.Module):
        """
        Play a batch of episodes with a given actor net
        """
        values_list = []
        indices_list = []  # this is essentially 'valid'
        turns_list = []
        observations_list = []
        policy_list = []
        actions_list = []
        rewards_list = []
        masks_list = []

        net.eval()
        time_start = time.perf_counter()

        while not self.states.terminal:

            indices_list.append(self.states.indices.clone())
            turns_list.append(self.states.turns.clone())

            observations = self.states.observations()

            with torch.no_grad():
                logits, policy, value, actions = net.forward(observations)
            # actions = torch.zeros_like(policy)
            rewards = self.states.step(actions)
            actions_oh = torch.zeros_like(policy)
            actions_oh[torch.arange(self.batch_size), actions] = 1
            values_list.append(value.squeeze().detach().clone())
            observations_list.append(observations)
            masks_list.append(observations[:, 1, :, 0])
            policy_list.append(policy)
            actions_list.append(actions_oh)
            rewards_list.append(rewards.clone())
            self.t_eff += 1

        time_end = time.perf_counter()
        self.generation_time = time_end - time_start

        # ends just before the all 0 indices tensor
        self.values = torch.stack(values_list, dim=0)
        self.indices = torch.stack(indices_list, dim=0)
        self.turns = torch.stack(turns_list, dim=0)
        self.observations = torch.stack(observations_list, dim=0)
        self.policy = torch.stack(policy_list, dim=0)
        self.actions = torch.stack(actions_list, dim=0)
        self.rewards = torch.stack(rewards_list, dim=0)

        self.masks = torch.stack(masks_list, dim=0)
        self.q_estimates = torch.zeros_like(self.policy)
        self.v_estimates = torch.zeros_like(self.rewards)
        self.generated = True
        net.train()

    def display(
        self,
    ):
        print("Episode params:")
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                if torch.numel(value) > 20:
                    value = value.shape
            print(key, value)

    def sample(self, batch_size):
        assert self.generated
        batch_size = min(batch_size, self.batch_size)
        selected = random.sample(range(self.batch_size), batch_size)
        selected = torch.tensor(selected, dtype=torch.long, device=self.tree.device)
        result = Episodes(self.tree, batch_size)
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                result.__dict__[key] = torch.index_select(
                    self.__dict__[key], dim=1, index=selected
                )
        result.generated = True
        result.t_eff = self.t_eff
        return result


class Buffer:
    def __init__(
        self,
        max_size,
    ) -> None:
        self.max_size = max_size  # num of episodes/batches
        self.episodes_buffer = deque(maxlen=max_size)

    def sample(self, batch_size):
        n = len(self.episodes_buffer)
        bucket_sizes = numpy.random.multinomial(batch_size, [1 / n] * n)
        assert sum(bucket_sizes) == batch_size
        return Episodes.collate(
            [
                self.episodes_buffer[_].sample(bucket_sizes[_])
                for _ in range(len(self.episodes_buffer))
            ]
        )

    def append(self, episodes: Episodes):
        self.episodes_buffer.append(episodes)
        while len(self.episodes_buffer) > self.max_size:
            self.episodes_buffer.popleft()

    def clear(
        self,
    ):
        self.episodes_buffer.clear()


if __name__ == "__main__":
    import game
    import net

    tree = game.Tree()
    tree.load("recent")
    net_ = net.MLP(size=tree.max_actions, width=1)
    buffer = Buffer(
        8,
    )

    for _ in range(10):
        episodes = Episodes(tree, 2)
        episodes.generate(net_)
        buffer.append(episodes)

    episodes_ = buffer.sample(10)
    episodes_.display()
