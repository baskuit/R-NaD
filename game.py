import torch
import torch.nn as nn
import torch.nn.functional as F

import pygambit
import numpy as np
import random
import os
import time
import logging

"""
This file contains:

    Tree
    The logic for building, solving, saving, and loading large stochastic matrix games

    States
    Represents a batch of states and observations of a tree game

    Episodes
    Saves the episode data for a batch of states played to completion

"""

class Tree () :

    def __init__ (self,
        is_root=True, # so terminal state is prepended during generation
        device=torch.device('cpu'),
        max_actions=3,
        max_transitions=1,
        row_actions=None,
        col_actions=None,
        depth_bound=1, # upper bound on longest path from root, and a strict bound when longest path =0
        transition_threshold=0,
        terminal_values=[-1, 1],
        row_actions_lambda=None,
        col_actions_lambda=None,
        depth_bound_lambda=None,
        desc=''):
    
        if row_actions_lambda is None:
            row_actions_lambda = lambda tree : tree.row_actions - 0
        if col_actions_lambda is None:
            col_actions_lambda = lambda tree : tree.col_actions - 0
        if depth_bound_lambda is None:
            depth_bound_lambda = lambda tree : tree.depth_bound - 1

        if row_actions is None: row_actions = max_actions
        if col_actions is None: col_actions = max_actions

        self.generated = False
        self.is_root=is_root
        self.device=device
        self.max_actions=max_actions
        self.max_transitions=max_transitions
        self.row_actions=row_actions
        self.col_actions=col_actions
        self.depth_bound=depth_bound
        self.transition_threshold=transition_threshold
        self.terminal_values=terminal_values
        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_trees')

        value_shape = (1, max_transitions, max_actions, max_actions)
        legal_shape = (1, 1, max_actions, max_actions)
        nash_shape = (1, max_actions * 2)
        self.value = torch.zeros(value_shape, device=device, dtype=torch.float)
        self.expected_value = torch.zeros(legal_shape, device=device, dtype=torch.float)
        self.legal = torch.zeros(legal_shape, device=device, dtype=torch.float)
        self.legal[0, 0, :self.row_actions, :self.col_actions] = 1.
        self.chance = self._transition_probs(max_actions, max_actions, max_transitions, transition_threshold)
        self.index = torch.zeros(value_shape, device=device, dtype=torch.long)
        self.payoff = torch.zeros((1, 1), device=device, dtype=torch.float)
        self.nash = torch.zeros(nash_shape, device=device, dtype=torch.float)
        self.desc=desc
        self.hash=torch.randint(-2**63, 2**63-1, size=(1,), device=device)
        self.size=0
        ####
        self.saved_keys = [key for key in self.__dict__.keys()]
        ####
        self.row_actions_lambda=row_actions_lambda
        self.col_actions_lambda=col_actions_lambda
        self.depth_bound_lambda=depth_bound_lambda

    def _child (self):
        child = Tree(
            is_root=False,
            device=self.device,
            max_actions=self.max_actions,
            max_transitions=self.max_transitions,
            row_actions=min(self.max_actions, max(1, self.row_actions_lambda(self))),
            col_actions=min(self.max_actions, max(1, self.col_actions_lambda(self))),
            depth_bound=max(0, self.depth_bound_lambda(self)),
            transition_threshold=self.transition_threshold,
            terminal_values=self.terminal_values,
            row_actions_lambda=self.row_actions_lambda,
            col_actions_lambda=self.col_actions_lambda,
            depth_bound_lambda=self.depth_bound_lambda,
        )
        return child

    def _transition_probs (self, rows, cols, n_trans, transition_threshold) :
        """
        Generates a random transition probabilty tensor. (It does not respect legality of moves, it is masked later)
        """
        probs = torch.from_numpy(np.random.dirichlet((1/n_trans,)*n_trans, (1,rows,cols))).to(self.device).to(torch.float)
        probs = probs - torch.where(probs < transition_threshold, probs, 0)
        probs = torch.nn.functional.normalize(probs, p=1, dim=3)
        probs = probs.movedim(3, 1)
        return probs

    def _assert_index_is_tree (self):
        """
        The index tensor describes a tree if and only if it is increasing
        (The non-zero values in the index[idx] are all > idx)
        and the non-zero values in index are one-to-one with some interval [1 + is_root, x]
        """
        indices = self.index[self.index != 0]
        indices = indices.tolist()
        indices.sort()
        all_possible_indices = list(range(1 + self.is_root, 2 + len(indices))) #
        assert(indices == all_possible_indices)
        for idx, index_slice in enumerate(self.index[1:]):
            greater_than_idx = index_slice > idx
            is_zero = index_slice == 0
            is_valid = torch.logical_or(greater_than_idx, is_zero)
            assert(torch.all(is_valid))


    def _solve (self, M : torch.Tensor, max_actions=2) :
        """
        M : tensor of shape (rows, cols)
        max_actions : dim=0 of output is padded to 2 * max_actions
        output : tensor of shape (2 * max_actions)
        """
        rows, cols = M.shape[:2]
        N = np.zeros((rows, cols), dtype=pygambit.Decimal)
        for _ in range(rows) :
            for __ in range(cols) :
                N[_][__] = pygambit.Decimal(M[_][__].item())

        g = pygambit.Game.from_arrays(N, -N)
        # TODO fix names here
        solutions = [[S[_] if _ < rows else 0 for _ in range(max_actions)] + [S[rows+_] if _ < cols else 0 for _ in range(max_actions)] for S in pygambit.nash.enummixed_solve(g, rational=False)]
        if not solutions:
            solutions = [[S[_] if _ < rows else 0 for _ in range(max_actions)] + [S[rows+_] if _ < cols else 0 for _ in range(max_actions)] for S in pygambit.nash.lcp_solve(g, rational=False)]
        purity = lambda solution : -int(1 in solution[:max_actions])  -int(1 in solution[max_actions:])
        solutions.sort(key=purity)
        return torch.tensor(solutions, dtype=torch.float).to(self.device)

    def _generate (self):
        child_list : list[Tree] = []
        lengths : list[int] = []

        for row in range(self.row_actions):
            for col in range(self.col_actions):
                for chance in range(self.max_transitions):

                    transition_prob = self.chance[0, chance, row, col]

                    if transition_prob > 0:
                        child = self._child()

                        if child.depth_bound > 0 and child.row_actions * child.col_actions > 0:
                            child._generate()
                            self.hash = torch.bitwise_xor(self.hash, child.hash)
                            child_list.append(child)
                            lengths.append(child.value.shape[0])
                            self.index[0, chance, row, col] = 1
                            child_payoff = child.payoff[:1]

                        else:
                            lengths.append(0)
                            child_payoff = torch.tensor(((random.choice(self.terminal_values),),)).to(self.device)
                        
                        self.value[0, chance, row, col] = child_payoff.item()
                self.expected_value[0, 0, row, col] = torch.sum(self.value[0, :, row, col] * self.chance[0, :, row, col])

        # Get NE payoff and strategies of parent expected value matrix
        matrix = self.expected_value[0, 0, :self.row_actions, :self.col_actions]
        solutions = self._solve(matrix, self.max_actions)
        if len(solutions) == 0:
            logging.error('matrix still not solved by _solve')
            logging.error(matrix)
        pi = solutions[0]
        
        pi_1, pi_2 = pi[:self.row_actions].unsqueeze(dim=0), pi[self.max_actions:][:self.col_actions].unsqueeze(dim=1)
        self.payoff = torch.matmul(torch.matmul(pi_1, matrix), pi_2)
        self.nash = pi.unsqueeze(dim=0)
                        
        # Update child index tensors
        for _, child in enumerate(child_list):
            mask = child.index.clone()
            mask[mask > 0] = 1.
            mask *= sum(lengths[:_]) + 1
            child.index += mask

        # Update root index tensor
        _ = 0
        sum_ = 0
        lengths.insert(0, 1)
        for row in range(self.row_actions):
            for col in range(self.col_actions):
                for chance in range(self.max_transitions):
                    if self.chance[0, chance, row, col] != 0:
                        sum_ += lengths[_]
                        self.index[0, chance, row, col] *= sum_
                        _ += 1
        
        child_list.insert(0, self)
        if self.is_root:
            terminal_state = Tree(
                is_root=False,
                device=self.device,
                max_actions=self.max_actions,
                max_transitions=self.max_transitions,
                depth_bound=0,
                row_actions=1,
                col_actions=1,
            )
            terminal_state.chance[0,:,0,0] = 0
            terminal_state.chance[0,0,0,0] = 1
            child_list.insert(0, terminal_state)

        self.value = torch.cat(tuple(child.value for child in child_list), dim=0)
        self.expected_value = torch.cat(tuple(child.expected_value for child in child_list), dim=0)
        self.legal = torch.cat(tuple(child.legal for child in child_list), dim=0)
        self.chance = torch.cat(tuple(child.chance for child in child_list), dim=0)
        self.chance *= self.legal # TODO nash_conv code cant currently assume this has been done 
        self.index = torch.cat(tuple(child.index for child in child_list), dim=0)
        self.payoff = torch.cat(tuple(child.payoff for child in child_list), dim=0)
        self.nash = torch.cat(tuple(child.nash for child in child_list), dim=0)

        if self.is_root:
            self.index += (self.index != 0)
            self.size = self.value.shape[0]
        self.generated = True

    def save (self, directory_name=None):
        if not self.generated:
            raise Exception('Attempting to save ungenerated tree')
        if not self.is_root:
            raise Exception('Attempting to save non-root tree')
        if not os.path.exists(self.directory):
            logging.info('saved_tree directory not found')
            logging.info('creating directory: {}'.format(self.directory))
            os.mkdir(self.directory)
            
        if directory_name is None:
            directory_name = str(int(time.time()))
        named_dir = os.path.join(self.directory, directory_name)
        if not os.path.exists(named_dir):
            logging.info('creating directory: {}'.format(named_dir))
            os.mkdir(named_dir)

        recent_dir = os.path.join(self.directory, 'recent')
        if not os.path.exists(recent_dir):
            logging.info('creating directory: {}'.format(recent_dir))
            os.mkdir(recent_dir)

        dict = {key:self.__dict__[key] for key in self.saved_keys}
        logging.info("saving trees to '{}' and 'recent'".format(named_dir))
        torch.save(dict, os.path.join(recent_dir, 'tree.tar'))
        torch.save(dict, os.path.join(named_dir, 'tree.tar'))

    def load (self, directory_name=None) :
        if directory_name is None:
            directory_name = 'recent'
        path = os.path.join(self.directory, directory_name, 'tree.tar')
        logging.info("loading tree from '{}'".format(directory_name))
        dict = torch.load(path)
        for key, value in dict.items():
            self.__dict__[key] = value
        logging.info("loaded tree has hash {}".format(self.hash.item()))

    def to (self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if torch.is_tensor(value): 
                self.__dict__[key] = value.to(device)

    def display (self,):
        print('Tree params:')
        for key, value in self.__dict__.items():
            if torch.torch.is_tensor(value):
                if torch.numel(value) > 20:
                    value = value.shape
            print(key, value)



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
    
    def generate_versus (self, net : torch.nn.Module, net_ : torch.nn.Module) :
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
        net_.eval()
        time_start = time.perf_counter()

        while not self.states.terminal:

            indices_list.append(self.states.indices.clone())
            turns_list.append(self.states.turns.clone())
        
            observations = self.states.observations()
            if self.states.turns[0] == 0:
                net_to_move = net
            else:
                net_to_move = net_
            with torch.no_grad():
                logits, policy, value, actions = net_to_move.forward(observations)
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


if __name__ == '__main__' :

    logging.basicConfig(level=logging.DEBUG)

    tree = Tree(
        max_actions=3,
        max_transitions=1,
        # transition_threshold=.45,
        # row_actions_lambda=lambda tree:tree.row_actions - 1 * (random.random() < .2),
        # col_actions_lambda=lambda tree:tree.row_actions - 1 * (random.random() < .2),
        row_actions_lambda=lambda tree:3,
        col_actions_lambda=lambda tree:3,
        row_actions=2,
        col_actions=2,
        # depth_bound_lambda=lambda tree:tree.depth_bound - 1 - 1 * (random.random() < .7),
        depth_bound=5,
        # desc='3x3 but 2x2 at root'
    )
    
    tree._generate()
    print(tree.hash)
    print(tree.size)
    tree.save('depth5')

    print(tree.expected_value[1])