import torch
import torch.nn as nn
import torch.nn.functional as F

from game import Tree
from collections import deque

import time


class States () :

    def __init__ (self, tree : Tree, batch_size) :
        self.tree = tree
        self.batch_size = batch_size
        self.indices = torch.ones((batch_size,), dtype=torch.int32, device=tree.device)
        self.turns = torch.zeros((batch_size,), dtype=torch.long, device=tree.device) #indices along player dim
        self.actions_1 = None
        self.actions_2 = None
        self.terminal = False

    def observations (self) -> torch.Tensor:
        expected_value = torch.index_select(self.tree.expected_value, 0, self.indices)
        legal = torch.index_select(self.tree.legal, 0, self.indices)
        observations_1 = torch.cat([expected_value, legal], dim=1)
        observations_2 = torch.cat([-expected_value, legal], dim=1).swapaxes(2, 3)
        observations = torch.stack([observations_1, observations_2], dim=1)
        observations = observations[torch.arange(self.batch_size), self.turns]
        return observations

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
            chance_entry = chance[torch.arange(self.batch_size), :, self.actions_1, self.actions_2]
            index_entry = index[torch.arange(self.batch_size), :, self.actions_1, self.actions_2]
            value_entry = value[torch.arange(self.batch_size), :, self.actions_1, self.actions_2] 
            actions_chance = torch.multinomial(chance_entry, num_samples=1).squeeze()
            self.indices = index_entry[torch.arange(self.batch_size), actions_chance]
            rewards_1 = value_entry[torch.arange(self.batch_size), actions_chance]
            rewards_1 *= (self.indices == 0)
            self.actions_1 = None
            self.actions_2 = None
        self.terminal = torch.all(self.indices == 0).item()
        return rewards_1



class Episodes () :
    """
    This represents a sequence of States, with dim=0 being time
    """

    def __init__ (self, tree : Tree, batch_size) :
        self.tree = tree
        self.batch_size = batch_size
        self.states = States(tree, batch_size)
        self.generation_time = 0
        self.transformation_time = 0
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

    def generate (self, net : torch.nn.Module) :
        """
        Play a batch of episodes with a given actor net
        """
        values_list = []
        indices_list = []
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

        #ends just before the all 0 indices tensor
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

        net.train()
    
    def display (self,):
        print('Episode params:')
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                if torch.numel(value) > 20:
                    value = value.shape
            print(key, value)

class Buffer:

    def __init__(self, size) -> None:
        self.size = size #num of episodes/batches
        self.episodes = deque(maxlen=size)

