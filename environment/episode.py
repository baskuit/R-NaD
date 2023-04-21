import torch
import torch.nn as nn
import torch.nn.functional as F

from environment.tree import Tree
from collections import deque

import time
import random
import numpy

# At least read the comment string in the main body to have an idea about how the game's structure and tensor members relate

"""
States represents a parallel collection of states that belong to a given Tree.
"""

class States:
    def __init__(self, tree: Tree, batch_size):
        self.tree = tree
        self.batch_size = batch_size
        self.indices = torch.ones((batch_size,), dtype=torch.int32, device=tree.device)
        self.player_to_move = torch.zeros(
            (batch_size,), dtype=torch.long, device=tree.device
        )  # indices along player dim
        self.row_actions = None
        self.col_actions = None
        self.terminal = False

        """
        batch_size:
            the number of parallel states
        indices:
            The indices along dim=0 of the tree's tensors that identify each state in the tree
        player_to_move:
            The Trees are in fact *matrix* trees. Conventionally we think of both players committing actions simultaneously.
            By implementation they actually alternate. A value of 0 means row player to move, otherwise column player.
            In practice the values of this tensor are all either 0 or 1. It should hardly make a difference in terms of performance.
        row_actions, 
        col_actions:
            The actions of the players as indices along their respective dimensions of the tree data tensors.
        terminal:
            A single state is terminal if its tree tensor index is 0. This bool simply denotes whether all states are terminal.
        """

    def observations(self) -> torch.Tensor:
        """
        Returns an observation at each batch index for the player to move.
        This observation consists of the expected value matrix together with legal actions mask.

        Thus the network is simply trying to learn to predict an unexploitable row player strategy,
        together with its associated NE payoff, for a zero sum matrix game.
        This would be straightforward if it were trained on the de factor solutions,
        and the loss function for the policy head were Cross Entropy Loss.
        However this will not work using only rollout trajectories and policy gradient.
        This is where R-NaD comes in.

        The expected value entries for non-legal pairs of joint actions are 0,
        but in theory those could be legal actions that happen to have an expected row payoff of 0.
        This would be unlikely, and in practice would could probably omit the legal actions mask without any detriment to learning. 
        """
        expected_value = torch.index_select(self.tree.expected_value_tensor, 0, self.indices)
        legal_actions = torch.index_select(self.tree.legal_tensor, 0, self.indices)
        row_observations = torch.cat([expected_value, legal_actions], dim=1)
        col_observations = torch.cat([-expected_value, legal_actions], dim=1).swapaxes(2, 3)
        observations = torch.stack([row_observations, col_observations], dim=1)
        observations = observations[torch.arange(self.batch_size), self.player_to_move]
        return observations

    def observations_noisy(
        self,
    ) -> torch.Tensor:
        """
        Instead of simply returning the expected value matrix from the pov of the player,
        we now transform the image so it is both high dimensional and noisy.

        This allows us to test the algorithm's robustness to the Credit Assignment problem,
        and the effectiveness of representation learning techniques

        Coming Soon! I mean it this time!
        """
        return None

    def step(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Input:
            actions: action indices for the row or column player to move
        Output:
            rewards: tensor of shape (self.batch_size,)

        The states do not transition and indices do not change until both row and column player actions have been committed.
        Once that happens, we select a chance player action for each state and transition accordingly.
        
        reward entries are 0 for any step that was not to a terminal state.
        """
        if self.player_to_move[0] == 0:
            # hmm, so we don't even allow for player_to_move tensor to have non-constant entries?
            # maybe I should just make it an int then..
            self.row_actions = actions
            self.player_to_move = 1 - self.player_to_move
            rewards = torch.zeros((self.batch_size,), device=self.tree.device)
        else:
            self.col_actions = actions
            self.player_to_move = 1 - self.player_to_move

            index = torch.index_select(self.tree.index_tensor, 0, self.indices)
            chance = torch.index_select(self.tree.chance_tensor, 0, self.indices)
            value = torch.index_select(self.tree.value_tensor, 0, self.indices)
            chance_strategy_profiles = chance[
                torch.arange(self.batch_size), :, self.row_actions, self.col_actions
            ]
            index_entry = index[
                torch.arange(self.batch_size), :, self.row_actions, self.col_actions
            ]
            value_entry = value[
                torch.arange(self.batch_size), :, self.row_actions, self.col_actions
            ]
            chance_actions = torch.multinomial(chance_strategy_profiles, num_samples=1).squeeze()
            self.indices = index_entry[torch.arange(self.batch_size), chance_actions]
            rewards = value_entry[torch.arange(self.batch_size), chance_actions]
            rewards *= self.indices == 0
            self.row_actions = None
            self.col_actions = None
        self.terminal = torch.all(self.indices == 0).item()
        return rewards

"""
Episodes represents a parallel batch of rollout trajectories, starting from the root of the Tree.
"""

class Episodes:

    def __init__(self, tree: Tree, batch_size):
        self.tree : Tree = tree
        self.batch_size : int = batch_size
        self.states : States = States(tree, batch_size)
        self.finished : bool = False
        self.generation_time : float = 0
        self.estimation_time : float = 0

        """
        batch_size:
            number of parallel trajectories to generate
        states:
            bathed states to be rolled out
        finished:
            Tree if all the states have been rollout out to completion
        generation_time:
            Time spent during rollout
        estimation_time:
            Time spent calculating v-trace estimates TODO
        """

        self.t_eff : int = -1
        self.turns : torch.Tensor = None
        self.indices : torch.Tensor = None
        self.observations : torch.Tensor = None
        self.policy : torch.Tensor = None
        self.actions : torch.Tensor = None
        self.rewards : torch.Tensor = None
        self.values : torch.Tensor = None
        self.masks : torch.Tensor = None

        """
        Raw trajectory information stored as tensors of shape (t_eff, ...)
        """

        self.q_estimates : torch.Tensor = None
        self.v_estimates : torch.Tensor = None

        """
        V-trace estimates
        """

    def generate(self, net: torch.nn.Module):

        """
        Play a batch of episodes with a given actor net
        """

        values_list = []
        indices_list = []  # this is essentially 'valid' in the v-trace code
        turns_list = []
        observations_list = []
        policy_list = []
        actions_list = []
        rewards_list = []
        masks_list = []
        # Stacked later

        net.eval()
        time_start = time.perf_counter()

        while not self.states.terminal:

            indices_list.append(self.states.indices.clone())
            turns_list.append(self.states.player_to_move.clone())

            observations = self.states.observations()
            with torch.no_grad():
                logits, policy, value, actions = net.forward(observations)

            observations_list.append(observations)
            rewards = self.states.step(actions)
            actions_oh = torch.zeros_like(policy)
            actions_oh[torch.arange(self.batch_size), actions] = 1
            values_list.append(value.squeeze().detach().clone())
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

        self.finished = True
        net.train()

    def __repr__(
        self,
    ):
        result = ""
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                if torch.numel(value) > 20:
                    value = value.shape
            result += f"{key}: {value}\n"
        return result

    def sample(self, batch_size):
        assert self.finished
        batch_size = min(batch_size, self.batch_size)
        selected = random.sample(range(self.batch_size), batch_size)
        selected = torch.tensor(selected, dtype=torch.long, device=self.tree.device)
        result = Episodes(self.tree, batch_size)
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                result.__dict__[key] = torch.index_select(
                    self.__dict__[key], dim=1, index=selected
                )
        result.finished = True
        result.t_eff = self.t_eff
        return result
    
    @classmethod
    def collate(cls, lst: list["Episodes"]):

        """
        Pad each member of Episodes along time dimension. 0's for padding works for everything.
        valid is determined by indices==0, thus other padded tensors aren't used  
        """

        t_eff = max(e.t_eff for e in lst)
        tree = lst[0].tree
        batch_size = sum(e.batch_size for e in lst)
        assert all(e.tree == tree for e in lst)
        assert all(e.finished for e in lst)

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
        result.finished = True
        result.t_eff = t_eff
        return result

class Buffer:

    """
    By necessity, all trajectories in an Episodes object are rolled out with the same actor net

    This class is a replay buffer of Episodes that were played with older actor nets.
    This facilitates testing of off-policy learning.

    Otherwise, the default protocol is on-policy: 
        1. Generate a batch of episodes with the learner net as the actor.
        2. Train the learner using the estimates as a single update.
        3. Repeat.

    TODO
    """

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