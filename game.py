import torch
import pygambit
import numpy as np
import random
import os
import time
import logging

class Tree () :

    def __init__ (self,
        is_root=True, # so terminal state is prepended during generation
        device=torch.device('cpu'),
        max_actions=3,
        max_transitions=1,
        depth_bound=1, # upper bound on longest path from root, and a strict bound when longest path =0
        transition_threshold=0,
        terminal_values=[-1, 1],
        row_actions_lambda=None,
        col_actions_lambda=None,
        depth_bound_lambda=None,):
    
        if row_actions_lambda is None:
            row_actions_lambda = lambda tree : tree.row_actions - 0
        if col_actions_lambda is None:
            col_actions_lambda = lambda tree : tree.col_actions - 0
        if depth_bound_lambda is None:
            depth_bound_lambda = lambda tree : tree.depth_bound - 1

        self.is_root=is_root
        self.device=device
        self.max_actions=max_actions
        self.max_transitions=max_transitions
        self.row_actions=max_actions
        self.col_actions=max_actions
        self.depth_bound=depth_bound
        self.transition_threshold=transition_threshold
        self.terminal_values=terminal_values
        self.row_actions_lambda=row_actions_lambda
        self.col_actions_lambda=col_actions_lambda
        self.depth_bound_lambda=depth_bound_lambda
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
        self.saved_keys = ['value', 'expected_value', 'legal', 'chance', 'index', 'payoff', 'nash']

    def child (self):
        child = Tree(
            is_root=False,
            device=self.device,
            max_actions=self.max_actions,
            max_transitions=self.max_transitions,
            depth_bound=self.depth_bound_lambda(self),
            transition_threshold=self.transition_threshold,
            terminal_values=self.terminal_values,
            row_actions_lambda=self.row_actions_lambda,
            col_actions_lambda=self.col_actions_lambda,
            depth_bound_lambda=self.depth_bound_lambda,
        )
        child.row_actions = self.row_actions_lambda(self)
        child.col_actions = self.col_actions_lambda(self)
        return child

    def _transition_probs (self, rows, cols, n_trans, transition_threshold) :
        """
        Generates a random
        """
        probs = torch.from_numpy(np.random.dirichlet((1/n_trans,)*n_trans, (1,rows,cols))).to(self.device)
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
        solutions = [[S[_] if _ < rows else 0 for _ in range(max_actions)] + [S[rows+_] if _ < cols else 0 for _ in range(max_actions)] for S in pygambit.nash.enummixed_solve(g, rational=False)]
        if not solutions:
            solutions = [[S[_] if _ < rows else 0 for _ in range(max_actions)] + [S[rows+_] if _ < cols else 0 for _ in range(max_actions)] for S in pygambit.nash.lcp_solve(g, rational=False)]
        purity = lambda solution : -int(1 in solution[:max_actions])  -int(1 in solution[max_actions:])
        solutions.sort(key=purity)
        return torch.tensor(solutions, dtype=torch.float).to(self.device)

    def _generate (self):

        child_list : list[Tree] = []
        lengths : list[int] = [] # cant use comprehension on child_list since that doesn't include the required 0 entries for terminal states
        idx = 2
        for row in range(self.row_actions):
            for col in range(self.col_actions):

                for chance in range(self.max_transitions):

                    transition_prob = self.chance[0, chance, row, col]
                    # logging.log(level=0, msg=self.chance.shape)

                    if transition_prob > 0:

                        #create new tree
                        # TODO index messed up, maybe because copy?
                        child = self.child()

                        if child.depth_bound > 0:
                            child._generate()
                            child_list.append(child)
                            lengths.append(child.value.shape[0]-1)
                            self.index[0, chance, row, col] = idx
                            idx += 1
                            child_payoff = child.payoff[:1]

                        else:
                            #node is terminal
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

        # Update parent index tensor
        _ = 0 
        for row in range(self.row_actions):
            for col in range(self.col_actions):
                for chance in range(self.max_transitions):
                    if self.index[0, chance, row, col] != 0:
                        self.index[0, chance, row, col] += sum(lengths[:_])
                        _ += 1

        # Update child index tensors
        for _, child in enumerate(child_list):
            mask = child.index.clone()
            mask[mask > 0] = 1.
            mask *= sum(lengths[:_]) + _ + 1
            child.index += mask
        
        
        child_list.insert(0, self)

        if self.is_root:
            terminal_state = Tree(
                is_root=False,
                device=self.device,
                max_actions=self.max_actions,
                max_transitions=self.max_transitions,
                depth_bound=0,
            )
            terminal_state.row_actions = 0
            terminal_state.col_actions = 0
            terminal_state.chance[0,:,0,0] = 0
            terminal_state.chance[0,0,0,0] = 1
            child_list.insert(0, terminal_state)

        self.value = torch.cat(tuple(child.value for child in child_list), dim=0)

        self.expected_value = torch.cat(tuple(child.expected_value for child in child_list), dim=0)
        self.legal = torch.cat(tuple(child.legal for child in child_list), dim=0)
        self.chance = torch.cat(tuple(child.chance for child in child_list), dim=0)
        self.index = torch.cat(tuple(child.index for child in child_list), dim=0)
        self.payoff = torch.cat(tuple(child.payoff for child in child_list), dim=0)
        self.nash = torch.cat(tuple(child.nash for child in child_list), dim=0)

    def save (self):
        recent_dir = os.path.join(self.directory, 'recent')
        if not os.path.exists(recent_dir):
            os.mkdir(recent_dir)
        time_dir = os.path.join(self.directory, str(int(time.time())))
        if not os.path.exists(time_dir):
            os.mkdir(time_dir)
        dict = {key:self.__dict__[key] for key in self.saved_keys}
        torch.save(dict, os.path.join(recent_dir, 'tree.tar'))
        torch.save(dict, os.path.join(time_dir, 'tree.tar'))

    def load (self, path=None) :
        if path is None:
            path = os.path.join(self.directory, 'recent', 'tree.tar')
        dict = torch.load(path)
        for key, value in dict.items():
            self.__dict__[key] = value

    def to (self, device):
        self.device = device
        self.value = self.value.to(device)
        self.expected_value = self.expected_value.to(device)
        self.legal = self.legal.to(device)
        self.chance = self.chance.to(device)
        self.index = self.index.to(device)
        self.payoff = self.payoff.to(device)
        self.nash = self.nash.to(device)
    
    def initial (self, batch_size) :
        return States(self, batch_size)

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
        expected_value = torch.index_select(self.tree.self.expected_value, 0, self.indices)
        legal = torch.index_select(self.tree.self.legal, 0, self.indices)
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

            index = torch.index_select(self.tree.self.index, 0, self.indices)
            chance = torch.index_select(self.tree.self.chance, 0, self.indices)
            value = torch.index_select(self.tree.self.value, 0, self.indices)
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
        rewards = torch.stack((rewards_1, -rewards_1), dim=1)
        return rewards

# A batch of entire episode trajectories

class Episodes () :
    def __init__ (self, tree : Tree, batch_size) :
        self.tree = tree
        self.batch_size = batch_size
        self.states = tree.initial(batch_size)

        # below all have time dimentions at index=0

        self.turns = None
        self.observations = None
        self.policy = None
        self.actions = None
        self.rewards = None
        self.values = None
        self.t_eff = 0

    # Play out batch episodes and save trajectories

    def generate (self, net : torch.nn.Module) :

        values_list = []
        indices_list = []
        turns_list = []
        observations_list = []
        policy_list = []
        actions_list = []
        rewards_list = []

        net.eval()
        
        while not self.states.terminal:

            indices_list.append(self.states.indices.clone())
            turns_list.append(self.states.turns.clone())
        
            observations = self.states.observations()
            with torch.no_grad():
                logits, policy, value, actions = net.forward(observations)
            ###
            rewards = self.states.step(actions)
            ###
            values_list.append(value.squeeze().detach().clone())
            observations_list.append(observations.clone())
            policy_list.append(policy)
            actions_list.append(actions.clone())
            rewards_list.append(rewards.clone())
            self.t_eff += 1

        #ends just before the all 0 indices tensor
        self.values = torch.stack(values_list, dim=0)
        self.turns = torch.stack(turns_list, dim=0)
        self.observations = torch.stack(observations_list, dim=0)
        self.policy = torch.stack(policy_list, dim=0)
        self.actions = torch.stack(actions_list, dim=0)
        self.rewards = torch.stack(rewards_list, dim=0)

        net.train()

if __name__ == '__main__' :
    
    logging.basicConfig(level=logging.DEBUG)

    depth_bound_lambda = lambda tree : max(0, tree.depth_bound - (1 if random.random() < .5 else 2))

    tree = Tree(
        max_actions=2,
        depth_bound=9,
        max_transitions=2,
        depth_bound_lambda=depth_bound_lambda
        )
    # tree._generate()
    tree.load()
    # tree.save()
    print(tree.value.shape)
    # tree._assert_index_is_tree()