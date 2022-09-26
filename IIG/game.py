import torch
import pygambit
import numpy as np
import random

# This is an implementation of a two-player matrix tree with stochastic transtions.
# The tree is represented as a tensor of states and transition indexes of shapes 
# (n_states, max_actions, max_actions, 2) and (n_states, max_actions, max_actions, max_transitions), resp.

# The states tensor has payoff and actions masking info at dim=3
# and dim=3 of the transitions tensor is a probability distribution on indexes of the states tensor

class TreeParameters () :

    def __init__ (self, 
        device=torch.device('cpu'),
        max_actions=3,
        max_transitions=1,
        depth_bound=1, # upper bound on longest path from root, with min path=0 => depth_bound=0
        transition_threshold=0,
        terminal_values=[-1, 1],
        row_actions_lambda=None, # these lambdas take a NodeInfo object and return the corresponding attribute of a child
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

class Tree () :

    def __init__ (self, params=TreeParameters()) :
        self.params = params
        self.data = None # TODO

    def generate (self) :

        def assert_index_is_tree (data):
            indices = data.index[data.index != 0]
            indices = indices.tolist()
            indices.sort()
            non_root_indices = list(range(2, 2 + len(indices)))
            assert(indices == non_root_indices)
            for _, index_slice in enumerate(data.index[1:]):
                assert(~(0 < index_slice <= _+1 ))
                # index tensor always points to higher indices
            # This guarantees tree structure
        def _transition_probs (rows, cols, n_trans, transition_threshold) :
            probs = torch.from_numpy(np.random.dirichlet((1/n_trans,)*n_trans, (1,rows,cols))).to(self.params.device)
            probs = probs - torch.where(probs < transition_threshold, probs, 0)
            probs = torch.nn.functional.normalize(probs, p=1, dim=3)
            return probs
        class Info () :
            def __init__ (self, depth_bound=1, terminal_values=[-1, 1], row_actions=1, col_actions=1, transition_threshold=.1) :
                self.depth_bound = depth_bound
                self.terminal_values = terminal_values
                self.row_actions = row_actions
                self.col_actions = col_actions
                self.transition_threshold = transition_threshold
            def get_child (self, tree : Tree) :
                depth_bound_ = tree.params.depth_bound_lambda(self)
                row_actions_ = tree.params.row_actions_lambda(self)
                col_actions_ = tree.params.col_actions_lambda(self)
                return Info(depth_bound_, self.terminal_values, row_actions_, col_actions_, self.transition_threshold)
        class Data:
            def __init__ (self, value=None, expected_value=None, legal=None, chance=None, index=None, payoff=None, nash=None) :
                self.value = value
                self.expected_value = expected_value
                self.legal = legal
                self.chance = chance
                self.index = index
                self.payoff = payoff
                self.nash = nash
            def concat (self, DataList : list) :
                DataList.insert(0, self)
                self.value = torch.cat(tuple(data.value for data in DataList), dim=0)
                self.expected_value = torch.cat(tuple(data.expected_value for data in DataList), dim=0)
                self.legal = torch.cat(tuple(data.legal for data in DataList), dim=0)
                self.chance = torch.cat(tuple(data.chance for data in DataList), dim=0)
                self.index = torch.cat(tuple(data.index for data in DataList), dim=0)
                self.payoff = torch.cat(tuple(data.payoff for data in DataList), dim=0)
                self.nash = torch.cat(tuple(data.nash for data in DataList), dim=0)
            def from_info (self, tree : Tree, info : Info) :
                self.value_shape = (1, tree.params.max_actions, tree.params.max_actions, tree.params.max_transitions)
                self.legal_shape = (1, tree.params.max_actions, tree.params.max_actions, 1)
                self.nash_shape = (1, tree.params.max_actions * 2)
                self.value = torch.zeros(self.value_shape, device=tree.params.device, dtype=torch.float)
                self.expected_value = torch.zeros(self.legal_shape, device=tree.params.device, dtype=torch.float)
                self.legal = torch.zeros(self.legal_shape, device=tree.params.device, dtype=torch.float)
                self.legal[0, :info.row_actions, :info.col_actions, 0] = 1.
                self.chance = _transition_probs(tree.params.max_actions, tree.params.max_actions, tree.params.max_transitions, info.transition_threshold)
                self.index = torch.zeros(self.value_shape, device=tree.params.device, dtype=torch.int32)
                self.payoff = torch.zeros((1, 1), device=tree.params.device, dtype=torch.float)
                self.nash = torch.zeros(self.nash_shape, device=tree.params.device, dtype=torch.float)

        def _generate (info : Info) -> Data:

            data = Data()
            data.from_info(self, info)
            child_data_list = []
            lengths = [] # can use comprehension on child_data_list since that doesn't include the required 0 entries for terminal states

            idx = 2
            for row in range(info.row_actions):
                for col in range(info.col_actions):

                    for chance in range(self.params.max_transitions):

                        transition_prob = data.chance[0, row, col, chance]

                        if transition_prob > 0:
                            child_info = info.get_child(self)

                            if child_info.depth_bound > 0:
                                child_data = _generate(child_info)
                                child_data_list.append(child_data)
                                lengths.append(child_data.value.shape[0]-1)
                                data.index[0, row, col, chance] = idx
                                idx += 1
                                child_payoff = child_data.payoff[:1]

                            else:
                                #node is terminal
                                lengths.append(0)
                                child_payoff = torch.tensor(((random.choice(self.params.terminal_values),),)).to(self.params.device)
                            
                            data.value[0, row, col, chance] = child_payoff.item()
                    data.expected_value[0, row, col, 0] = torch.sum(data.value[0, row, col, :] * data.chance[0, row, col, :])

            # Get NE payoff and strategies of parent expected value matrix
            M = data.expected_value[0, :info.row_actions, :info.col_actions, 0]
            solutions = self.solve(M, self.params.max_actions)
            pi = solutions[0]
            pi_1, pi_2 = pi[:info.row_actions].unsqueeze(dim=0), pi[self.params.max_actions:][:info.col_actions].unsqueeze(dim=1)
            data.payoff = torch.matmul(torch.matmul(pi_1, M), pi_2)
            data.nash = pi.unsqueeze(dim=0)

            # Update parent index tensor
            _ = 0 
            for row in range(info.row_actions):
                for col in range(info.col_actions):
                    for chance in range(self.params.max_transitions):
                        if data.index[0, row, col, chance] != 0:
                            data.index[0, row, col, chance] += sum(lengths[:_])
                            _ += 1

            # Update child index tensors
            for _, child_data in enumerate(child_data_list):
                mask = child_data.index.clone()
                mask[mask > 0] = 1.
                mask *= sum(lengths[:_]) + _ + 1
                child_data.index += mask
            

            data.concat(child_data_list)

            return data

        root_info = Info(self.params.depth_bound, self.params.terminal_values, self.params.max_actions, self.params.max_actions, self.params.transition_threshold)
        root_data = _generate(root_info)

        # Add terminal/absorbing state at index=0
        terminal_info = Info(0, [0], 1, 1, 0)
        game_data = Data()
        game_data.from_info(self, terminal_info)
        game_data.concat([root_data])

        return game_data

    # TODO Sort strats to prefer pure
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
        return torch.tensor(solutions, dtype=torch.float)
    
    def save (self) :
        pass

    def load (self) :
        pass

    # The following functions must be implemented for all compatible two-player IIG's


    def observations (self) :
        pass

    def initial (batch_size) :
        pass

    # Phi from DeepNash paper
    def turn (states) :
        pass

    def step () :
        pass


if __name__ == '__main__' :

    tree_params = TreeParameters(depth_bound=3, max_actions=2, transition_threshold=.4, max_transitions=1)
    tree = Tree(tree_params)

    data = tree.generate()


    print('Printing game tree data')
    print('Note: Index=0 is the terminal state and Index=1 is the initial state')

    for _ in range(data.value.shape[0]):

        print('\nIndex: ', _)
        print('expected value', data.expected_value[_])
        print('index', data.index[_])
        print('chance', data.chance[_])
        print('payoff', data.payoff[_])
        print('strategy', data.nash[_])