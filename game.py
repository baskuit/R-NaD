import torch
import pygambit
import numpy as np
import random
import os
import time

class Tree () :

    def __init__ (self,
        device=torch.device('cpu'),
        max_actions=3,
        max_transitions=1,
        depth_bound=1, # upper bound on longest path from root, and a strict bound when =0
        transition_threshold=0,
        terminal_values=[-1, 1],
        # lambda functions that generate matrix parameters of children
        row_actions_lambda=None,
        col_actions_lambda=None,
        depth_bound_lambda=None,):
    
        if row_actions_lambda is None:
            row_actions_lambda = lambda info : info.row_actions - 0
        if col_actions_lambda is None:
            col_actions_lambda = lambda info : info.col_actions - 0
        if depth_bound_lambda is None:
            depth_bound_lambda = lambda info : info.depth_bound - 1

        self.device=device
        self.max_actions=max_actions
        self.max_transitions=max_transitions
        self.depth_bound=depth_bound
        self.transition_threshold=transition_threshold
        self.terminal_values=terminal_values
        self.row_actions_lambda=row_actions_lambda
        self.col_actions_lambda=col_actions_lambda
        self.depth_bound_lambda=depth_bound_lambda
        self.data = None
        self.directory = os.path.dirname(os.path.realpath(__file__))

    def generate (self) :

        if not self.data is None:
            raise Exception('repeat generation of Tree')

        # Returns true if and only if data.indices describes a tree graph structure
        def assert_index_is_tree (data):
            indices = data.index[data.index != 0]
            indices = indices.tolist()
            indices.sort()
            all_possible_indices = list(range(2, 2 + len(indices)))
            assert(indices == all_possible_indices)
            for _, index_slice in enumerate(data.index[1:]):
                idx = _+1
                assert(~(0 < index_slice <= idx))

        def _transition_probs (rows, cols, n_trans, transition_threshold) :
            probs = torch.from_numpy(np.random.dirichlet((1/n_trans,)*n_trans, (1,rows,cols))).to(self.device)
            probs = probs - torch.where(probs < transition_threshold, probs, 0)
            probs = torch.nn.functional.normalize(probs, p=1, dim=3)
            probs = probs.movedim(3, 1)
            return probs
        class Info () :
            def __init__ (self, depth_bound=1, terminal_values=[-1, 1], row_actions=1, col_actions=1, transition_threshold=.1) :
                self.depth_bound = depth_bound
                self.terminal_values = terminal_values
                self.row_actions = row_actions
                self.col_actions = col_actions
                self.transition_threshold = transition_threshold
            def get_child (self, tree : Tree) :
                depth_bound_ = tree.depth_bound_lambda(self)
                row_actions_ = tree.row_actions_lambda(self)
                col_actions_ = tree.col_actions_lambda(self)
                return Info(depth_bound_, self.terminal_values, row_actions_, col_actions_, self.transition_threshold)
        class Data ():
            def __init__ (self, value=None, expected_value=None, legal=None, chance=None, index=None, payoff=None, nash=None) :
                self.value = value
                self.expected_value = expected_value
                self.legal = legal
                self.chance = chance
                self.index = index
                self.payoff = payoff
                self.nash = nash
            def clone(self):
                copy = Data(
                    self.value.clone(),
                    self.expected_value.clone(),
                    self.legal.clone(),
                    self.chance.clone(),
                    self.index.clone(),
                    self.payoff.clone(),
                    self.nash.clone())
                return copy

            def concat (self, DataList : list) :
                DataList.insert(0, self)
                self.value = torch.cat(tuple(data.value for data in DataList), dim=0).contiguous()
                self.expected_value = torch.cat(tuple(data.expected_value for data in DataList), dim=0).contiguous()
                self.legal = torch.cat(tuple(data.legal for data in DataList), dim=0).contiguous()
                self.chance = torch.cat(tuple(data.chance for data in DataList), dim=0).contiguous()
                self.index = torch.cat(tuple(data.index for data in DataList), dim=0).contiguous()
                self.payoff = torch.cat(tuple(data.payoff for data in DataList), dim=0).contiguous()
                self.nash = torch.cat(tuple(data.nash for data in DataList), dim=0).contiguous()
            def from_info (self, tree : Tree, info : Info) :
                value_shape = (1, tree.max_transitions, tree.max_actions, tree.max_actions)
                legal_shape = (1, 1, tree.max_actions, tree.max_actions)
                nash_shape = (1, tree.max_actions * 2)
                self.value = torch.zeros(value_shape, device=tree.device, dtype=torch.float)
                self.expected_value = torch.zeros(legal_shape, device=tree.device, dtype=torch.float)
                self.legal = torch.zeros(legal_shape, device=tree.device, dtype=torch.float)
                self.legal[0, 0, :info.row_actions, :info.col_actions] = 1.
                self.chance = _transition_probs(tree.max_actions, tree.max_actions, tree.max_transitions, info.transition_threshold)
                self.index = torch.zeros(value_shape, device=tree.device, dtype=torch.int32)
                self.payoff = torch.zeros((1, 1), device=tree.device, dtype=torch.float)
                self.nash = torch.zeros(nash_shape, device=tree.device, dtype=torch.float)
            def to (self, device):
                self.value = self.value.to(device)
                self.expected_value = self.expected_value.to(device)
                self.legal = self.legal.to(device)
                self.chance = self.chance.to(device)
                self.index = self.index.to(device)
                self.payoff = self.payoff.to(device)
                self.nash = self.nash.to(device)
            def save(self, dir):
                data_saved = {
                    'value':self.value,
                    'expected_value':self.expected_value,
                    'legal':self.legal,
                    'chance':self.chance,
                    'index':self.index,
                    'payoff':self.payoff,
                    'nash':self.nash,
                }
                torch.save(data_saved, dir)
            def load(self):
                data_saved = torch.load('./saved/data')

        def _generate (info : Info) -> Data:

            data = Data()
            data.from_info(self, info)
            child_data_list = []
            lengths = [] # can use comprehension on child_data_list since that doesn't include the required 0 entries for terminal states
            idx = 2
            for row in range(info.row_actions):
                for col in range(info.col_actions):

                    for chance in range(self.max_transitions):

                        transition_prob = data.chance[0, chance, row, col]

                        if transition_prob > 0:
                            child_info = info.get_child(self)

                            if child_info.depth_bound > 0:
                                child_data = _generate(child_info)
                                child_data_list.append(child_data)
                                lengths.append(child_data.value.shape[0]-1)
                                data.index[0, chance, row, col] = idx
                                idx += 1
                                child_payoff = child_data.payoff[:1]

                            else:
                                #node is terminal
                                lengths.append(0)
                                child_payoff = torch.tensor(((random.choice(self.terminal_values),),)).to(self.device)
                            
                            data.value[0, chance, row, col] = child_payoff.item()

                    
                    data.expected_value[0, 0, row, col] = torch.sum(data.value[0, :, row, col] * data.chance[0, :, row, col])

            # Get NE payoff and strategies of parent expected value matrix
            M = data.expected_value[0, 0, :info.row_actions, :info.col_actions]
            solutions = self.solve(M, self.max_actions)
            if len(solutions) == 0: 
                print(M)
                # TODO
            pi = solutions[0]
            pi_1, pi_2 = pi[:info.row_actions].unsqueeze(dim=0), pi[self.max_actions:][:info.col_actions].unsqueeze(dim=1)
            data.payoff = torch.matmul(torch.matmul(pi_1, M), pi_2)

            data.nash = pi.unsqueeze(dim=0)

            # Update parent index tensor
            _ = 0 
            for row in range(info.row_actions):
                for col in range(info.col_actions):
                    for chance in range(self.max_transitions):
                        if data.index[0, chance, row, col] != 0:
                            data.index[0, chance, row, col] += sum(lengths[:_])
                            _ += 1

            # Update child index tensors
            for _, child_data in enumerate(child_data_list):
                mask = child_data.index.clone()
                mask[mask > 0] = 1.
                mask *= sum(lengths[:_]) + _ + 1
                child_data.index += mask
            

            data.concat(child_data_list)

            return data.clone()

        root_info = Info(self.depth_bound, self.terminal_values, self.max_actions, self.max_actions, self.transition_threshold)
        root_data = _generate(root_info)

        # Add terminal/absorbing state at index=0
        terminal_info = Info(0, [0], 1, 1, 0)
        game_data = Data()
        game_data.from_info(self, terminal_info)
        game_data.concat([root_data])
        self.data = game_data

    # M : tensor of shape (rows, cols)
    # max_actions : dim=0 of output is padded to 2 * max_actions
    # output : tensor of shape (2 * max_actions)
    def solve (self, M, max_actions=2) :
        rows, cols = M.shape[:2]
        N = np.zeros((rows, cols), dtype=pygambit.Decimal)
        for _ in range(rows) :
            for __ in range(cols) :
                N[_][__] = pygambit.Decimal(M[_][__].item())

        g = pygambit.Game.from_arrays(N, -N)
        solutions = [[S[_] if _ < rows else 0 for _ in range(max_actions)] + [S[rows+_] if _ < cols else 0 for _ in range(max_actions)] for S in pygambit.nash.enummixed_solve(g, rational=False)]
        purity = lambda solution : -int(1 in solution[:max_actions])  -int(1 in solution[max_actions:])
        solutions.sort(key=purity)
        if not solutions:
            solutions = [[S[_] if _ < rows else 0 for _ in range(max_actions)] + [S[rows+_] if _ < cols else 0 for _ in range(max_actions)] for S in pygambit.nash.lcp_solve(g, rational=False)]
        return torch.tensor(solutions, dtype=torch.float).to(self.device)
    
    def save(self):
        recent_dir = os.path.join(self.directory, 'recent')
        if not os.path.exists(recent_dir):
            os.mkdir(recent_dir)
        self.data.save(recent_dir)



    def load (self) :
        pass

    def to (self, device) :
        self.device = device
        self.data.to(device)
    
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
        expected_value = torch.index_select(self.tree.data.expected_value, 0, self.indices)
        legal = torch.index_select(self.tree.data.legal, 0, self.indices)
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

            index = torch.index_select(self.tree.data.index, 0, self.indices)
            chance = torch.index_select(self.tree.data.chance, 0, self.indices)
            value = torch.index_select(self.tree.data.value, 0, self.indices)
            chance_entry = chance[torch.arange(self.batch_size), :, self.actions_1, self.actions_2]
            index_entry = index[torch.arange(self.batch_size), :, self.actions_1, self.actions_2]
            value_entry = value[torch.arange(self.batch_size), :, self.actions_1, self.actions_2] 
            actions_chance = torch.multinomial(chance_entry, 1).squeeze()
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
    
    cant_solve_test = torch.tensor([[-1.0000,  1.0000, -1.0000],
        [ 1.0000,  0.9991,  1.0000],
        [-1.0000, -0.5599, -0.1398]])

    tree = Tree()

    depth_bound_lambda = lambda info : max(0, info.depth_bound - (1 if random.random() < .5 else 2))

    tree = Tree()

    tree.generate()
    data = tree.data
    print(tree.directory)
    print(int(time.time()))
    # torch.save(data.chance, './saved/data')
    # data.save()
    # print('Printing game tree data')
    # print('Note: Index=0 is the terminal state and Index=1 is the initial state')

    # for _ in range(data.value.shape[0]):

    #     print('\nIndex: ', _)
    #     print('expected value', data.expected_value[_])
    #     print('index', data.index[_])
    #     print('chance', data.chance[_])
    #     print('payoff', data.payoff[_])
    #     print('strategy', data.nash[_])