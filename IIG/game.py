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
        transition_threshold=1,
        terminal_values=[-1, 1],
        row_actions_lambda=None, # these lambdas take a NodeInfo object and return the corresponding attribute of a child
        col_actions_lambda=None,
        depth_bound_lamda=None,):
    
        if row_actions_lambda is None:
            row_actions_lambda = lambda info : info.row_actions_lambda - 1
        if col_actions_lambda is None:
            col_actions_lambda = lambda info : info.col_actions_lambda - 1
        if depth_bound_lamda is None:
            depth_bound_lamda = lambda info : info.depth_bound_lamda - 1

        self.device=device
        self.max_actions=max_actions
        self.max_transitions=max_transitions
        self.depth_bound=depth_bound
        self.transition_threshold=transition_threshold
        self.terminal_values=terminal_values
        self.row_actions_lambda=row_actions_lambda
        self.col_actions_lambda=col_actions_lambda
        self.depth_bound_lamda=depth_bound_lamda

class Tree () :

    def __init__ (self, params=TreeParameters()) :

        self.params = params

        self.value_tensor = None
        self.expected_value_tensor = None
        self.legal_tensor = None
        self.prob_tensor = None
        self.idx_tensor = None
        
        self.value_tensor_shape = (1, self.params.max_actions, self.params.max_actions, self.params.max_transitions)
        self.legal_tensor_shape = (1, self.params.max_actions, self.params.max_actions, 1)
        self.nash_tensor_shape = (1, self.params.max_actions * 2)

    # Generates transition probabilities for each matrix



    def generate (self) :

        # TODO test iff valid idx tensor.
        def test_idx_tensor (idx_tensor):
            l = idx_tensor[idx_tensor != 0]
            l = l.tolist()
            l.sort()
            k = list(range(2, 2 + len(l)))
            assert(l == k)

        def _transition_probs (rows, cols, n_trans, transition_threshold) :
            x = torch.from_numpy(np.random.dirichlet((1/n_trans,)*n_trans, (1,rows,cols))).to(self.params.device)
            x = x - torch.where(x < transition_threshold, x, 0)
            x = torch.nn.functional.normalize(x, p=1, dim=3)
            return x

        class Info () :
            def __init__ (self, depth=0, values=[-1, 1], row_actions=1, col_actions=1, transition_threshold=.1, solve=False) :
                self.depth = depth
                self.values = values
                self.row_actions = row_actions
                self.col_actions = col_actions
                self.transition_threshold = transition_threshold
                self.solve = solve

        def _generate (info : Info) :
            expected_value_tensor = torch.zeros(self.legal_tensor_shape, device=self.params.device, dtype=torch.float)
            value_tensor = torch.zeros(self.value_tensor_shape, device=self.params.device, dtype=torch.float)
            legal_tensor = torch.zeros(self.legal_tensor_shape, device=self.params.device, dtype=torch.float)
            legal_tensor[0, :info.row_actions, :info.col_actions, 0] = 1.
            prob_tensor = _transition_probs(self.params.max_actions, self.params.max_actions, self.params.max_transitions, info.transition_threshold)
            idx_tensor = torch.zeros(self.value_tensor_shape, device=self.params.device, dtype=torch.int32)

            child_value_tensors = []
            child_legal_tensors = []
            child_prob_tensors = []
            child_idx_tensors = []
            child_nash_tensors = []

            lengths = [] # as many entries as transitions, same as sub's

            idx = 2 # incremented herein

            for row in range(info.row_actions):
                for col in range(info.col_actions):

                    for chance in range(self.params.max_transitions):

                        transition_prob = prob_tensor[0, row, col, chance]

                        if transition_prob > 0:

                            # Get new depth etc for child
                            depth_ = self.params.depth_lambda(info)
                            row_actions_ = self.params.row_actions_lambda(info)
                            col_actions_ = self.params.col_actions_lambda(info)

                            if depth_ > 0:

                                idx_tensor[0, row, col, chance] = idx
                                idx += 1

                                info_ = Info(depth=depth_, values=info.values, row_actions=row_actions_, col_actions=col_actions_, transition_threshold=info.transition_threshold, solve=info.solve)
                                
                                v_, l_, p_, i_, payoff_ = _generate(info_)

                                lengths.append(v_.shape[0]-1)

                                child_value_tensors.append(v_) # wont change by batchin
                                child_legal_tensors.append(l_)
                                child_prob_tensors.append(p_)

                                child_idx_tensors.append(i_) # will change by batching

                            else:
                                #node is terminal
                                lengths.append(0)
                                payoff_ = random.choice(self.params.terminal_values)

                            value_tensor[0, row, col, chance] = payoff_
                            
                    expected_value_tensor[0, row, col, 0] = torch.sum(value_tensor[0, row, col, :] * prob_tensor[0, row, col, :])

            # Get payoff of parent value tensor
            nash_tensor = torch.zeros(self.nash_tensor_shape, device=self.params.device, dtype=torch.int)
            payoff = 0
            if info.solve:
                M = expected_value_tensor[0, :, :, 0]
                nash_tensor = self.solve(M)
                print(nash_tensor)
                pi = nash_tensor[0]
                pi_1, pi_2 = pi[:info.row_actions].unsqueeze(dim=0), pi[self.params.max_actions:][:info.col_actions].unsqueeze(dim=1)
                payoff = torch.matmul(torch.matmul(pi_1, M), pi_2).squeeze().item()
                
            # Make the changes to parent index tensors
            i = 0
            for row in range(info.n_actions):
                for col in range(info.n_actions):
                    for chance in range(self.params.max_transitions):
                        if idx_tensor[0, row, col, chance] != 0:
                            idx_tensor[0, row, col, chance] += sum(lengths[:i])
                            i += 1

            # Make the changes to child index tensors
            for _, t in enumerate(child_idx_tensors):
                mask = t.clone()
                mask[mask > 0] = 1.
                mask *= (sum(lengths[:_]) + _ + 1)
                t += mask
            
            # return

            child_value_tensors.insert(0, value_tensor)
            child_legal_tensors.insert(0, legal_tensor)
            child_prob_tensors.insert(0, prob_tensor)
            child_idx_tensors.insert(0, idx_tensor)

            v = torch.cat(child_value_tensors, dim=0)
            l = torch.cat(child_legal_tensors, dim=0)
            p = torch.cat(child_prob_tensors, dim=0)
            i = torch.cat(child_idx_tensors, dim=0)

            test_idx_tensor(i)

            return v, l, p, i, payoff

        # generate() body:

        info = Info(self.params.depth, self.params.terminal_values, self.params.max_actions, self.params.transition_threshold, self.params.solve)
        v, l, p, i, payoff = _generate(info)
        self.value_tensor = torch.cat((self.value_tensor, v), dim=0)

    



    def save (self) :

        pass

    def load (self) :
        self.generated = True
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


    def solve (self, M, max_actions=2) :

        rows, cols = M.shape[:2]
        N = np.zeros((rows, cols), dtype=pygambit.Decimal)
        for _ in range(rows) :
            for __ in range(cols) :
                N[_][__] = pygambit.Decimal(M[_][__].item())

        g = pygambit.Game.from_arrays(N, -N)
        solutions = [[S[_] if _ < rows else 0 for _ in range(max_actions)] + [S[rows+_] if _ < cols else 0 for _ in range(max_actions)] for S in pygambit.nash.enummixed_solve(g, rational=False)]
        return torch.tensor(solutions, dtype=torch.float)


if __name__ == '__main__' :


    tree = Tree()

    v, l, p, i, payoff = tree._generate(Info(depth=3, values=[-1, 1], n_actions=game_params.max_actions, transition_threshold=.3, solve=True))

    i_shape = list(i.shape)
    i_shape[0] = 1
    i = torch.cat((torch.zeros(i_shape, device=game_params.device), i), dim=0)
    # print(i.shape)
    # tree.test_idx_tensor(i)
    
    # for __, _ in enumerate(i):
    #     print('\n', __)
    #     print(_)  