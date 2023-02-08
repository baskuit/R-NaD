import torch
import torch.nn as nn
import torch.nn.functional as F

from game import Tree
# from search import Search

from collections import deque
from typing import List

import time
import random
import numpy


class Search:

    """
    A tree-wide search can be simlutated by talking value inferences of the tree
    and applying a map that recursively replaces these values with the NE payoff
    of the matrix of previous value estimates.

    We can also test my imperfect information look-ahead idea. No explanation yet, sorry.
    """

    def __init__(self, tree: Tree, same_info_prob=.7):
        self.tree = tree
        self.size = tree.value.shape[0]
        self.policy = torch.zeros_like(tree.solution)
        self.value = torch.zeros((self.size, 2), device=tree.device, dtype=torch.float)
        same_info_prng = torch.rand_like(self.value)
        self.same_info = same_info_prng < same_info_prob

    def to(self, device):
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                self.__dict__[key] = value.to(device)


    def apply_net(self, net: torch.nn.Module, inference_batch_size=10**5):

        net.eval()
        for _ in range(self.size // inference_batch_size + 1):
            slice_range = torch.arange(
                _ * inference_batch_size,
                min((_ + 1) * inference_batch_size, self.size),
                device=self.tree.device,
            )
            value_slice = self.tree.expected_value[slice_range]
            legal_slice = self.tree.legal[slice_range]

            with torch.no_grad():
                inference_slice_row = torch.cat([value_slice, legal_slice], dim=1).contiguous()
                _, policy, value, _ = net.forward(inference_slice_row)
                self.policy[slice_range, : self.tree.max_actions] = policy
                self.value[slice_range, 0] = value[:, 0]
                inference_slice_col = torch.cat([-value_slice, legal_slice], dim=1).swapaxes(
                    2, 3
                ).contiguous()
                _, policy, value, _ = net.forward(inference_slice_col)
                self.policy[slice_range, self.tree.max_actions : ] = policy
                self.value[slice_range, 1] = value[:, 0]

    def step(self, idx_cur=1) -> torch.Tensor:


        matrix_1 = torch.zeros(
            (self.tree.max_transitions * self.tree.max_actions**2,),
            device=self.tree.device,
            dtype=torch.float,
        )
        matrix_2 = torch.zeros_like(matrix_1)

        value = torch.flatten(
            self.tree.value[idx_cur : idx_cur + 1]
        )
        index = torch.flatten(
            self.tree.index[idx_cur : idx_cur + 1]
        )
        chance = torch.flatten(
            self.tree.chance[idx_cur : idx_cur + 1]
            * self.tree.legal[idx_cur : idx_cur + 1]
        )
        # tensors at the root, shape (1, t, a, a)

        subtrees = (chance > 0).nonzero()
        for idx_local in subtrees:
            idx_local = idx_local[0]
            idx_next = index[idx_local]
            if idx_next == 0:
                v1 = value[idx_local]
                v2 = -value[idx_local]
            else:
                v1, v2 = self.step(idx_next)
                # v1 = self.value[idx_next, 0]
                # v2 = self.value[idx_next, 1]
                # Recursive call just before using update values
            transition_prob = chance[idx_local]
            matrix_1[idx_local] = v1 * transition_prob
            matrix_2[idx_local] = v2 * transition_prob

        matrix_1 = matrix_1.view(self.tree.max_transitions, self.tree.max_actions, self.tree.max_actions)
        matrix_2 = matrix_2.view(self.tree.max_transitions, self.tree.max_actions, self.tree.max_actions)
        matrix_1 = torch.sum(matrix_1, dim=0)
        matrix_2 = torch.sum(matrix_2, dim=0)
        # Add to get expected value estimates and reshape into matrix shape

        solution_1 = self.tree._solve( matrix_1, self.tree.max_actions)[0]
        solution_2 = self.tree._solve(-matrix_2, self.tree.max_actions)[0]
        # Solve NE in both players search trees

        pi_1, pi_2 = solution_1[: self.tree.max_actions].unsqueeze(dim=0), solution_1[
            self.tree.max_actions :
        ].unsqueeze(dim=1)
        self.value[idx_cur, 0] = torch.matmul(
            torch.matmul(pi_1, matrix_1), pi_2
        )
        self.policy[idx_cur, : self.tree.max_actions] = torch.flatten(pi_1)
        pi_1, pi_2 = solution_2[: self.tree.max_actions].unsqueeze(dim=0), solution_2[
            self.tree.max_actions :
        ].unsqueeze(dim=1)
        old_value = self.value[idx_cur].clone()
        self.value[idx_cur, 1] = torch.matmul(
            torch.matmul(pi_1, matrix_2), pi_2
        )
        self.policy[idx_cur, self.tree.max_actions : ] = torch.flatten(pi_2)
        return old_value
        # Get payoff that serves as new values


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
        we now transform the image so it is both high dimensional and noisy.

        This allows us to test the algorithm's robustness to the Credit Assignment problem,
        and the effictiveness of representation learning techniques

        Coming Soon.
        """
        return None

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
    def collate(cls, lst: List["Episodes"]):

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

    def generate(self, net: torch.nn.Module, search_depth=0,):
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

        search = Search(self.tree, same_info_prob=.7) # info probs does nothing currently
        search.apply_net(net)
        for depth in range(search_depth):
            search.step()

        while not self.states.terminal:

            indices_list.append(self.states.indices.clone())
            turns_list.append(self.states.turns.clone())

            observations = self.states.observations()

            # with torch.no_grad():
            #     logits, policy, value, actions = net.forward(observations)
            # actions = torch.zeros_like(policy)
            policy = search.policy
            policy = torch.index_select(policy, dim=0, index=self.states.indices)
            policy_1 = policy[:, : self.tree.max_actions]
            policy_2 = policy[:, self.tree.max_actions :]
            policy = torch.stack((policy_1, policy_2), dim=1)
            policy = policy[torch.arange(self.batch_size), self.states.turns]
            value = torch.index_select(search.value, dim=0, index=self.states.indices)
            value = value[torch.arange(self.batch_size), self.states.turns]
            # TODO investiage whether you should use the net values or solved values
            # my guess: the solved values should be just fine
            actions = torch.squeeze(torch.multinomial(policy, num_samples=1))

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
