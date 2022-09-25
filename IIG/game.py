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
            def __init__ (self, depth_bound=1, terminal_values=[-1, 1], row_actions=1, col_actions=1, transition_threshold=.1) :
                self.depth_bound = depth_bound
                self.terminal_values = terminal_values
                self.row_actions = row_actions
                self.col_actions = col_actions
                self.transition_threshold = transition_threshold

        def _generate (info : Info) :
            expected_value_tensor = torch.zeros(self.legal_tensor_shape, device=self.params.device, dtype=torch.float)
            value_tensor = torch.zeros(self.value_tensor_shape, device=self.params.device, dtype=torch.float)
            legal_tensor = torch.zeros(self.legal_tensor_shape, device=self.params.device, dtype=torch.float)
            legal_tensor[0, :info.row_actions, :info.col_actions, 0] = 1.
            prob_tensor = _transition_probs(self.params.max_actions, self.params.max_actions, self.params.max_transitions, info.transition_threshold)
            idx_tensor = torch.zeros(self.value_tensor_shape, device=self.params.device, dtype=torch.int32)
            payoff_tensor = torch.zeros((1, 1), device=self.params.device, dtype=torch.float)

            # need root payoffs and

            child_value_tensors = []
            child_legal_tensors = []
            child_prob_tensors = []
            child_idx_tensors = []
            child_payoff_tensors = []
            child_nash_tensors = []
            child_expected_tensors = []

            lengths = [] # as many entries as transitions, same as sub's

            idx = 2 # incremented herein

            for row in range(info.row_actions):
                for col in range(info.col_actions):

                    for chance in range(self.params.max_transitions):

                        transition_prob = prob_tensor[0, row, col, chance]

                        if transition_prob > 0:

                            # Get new depth etc for child
                            depth_bound_ = self.params.depth_bound_lambda(info)
                            row_actions_ = self.params.row_actions_lambda(info)
                            col_actions_ = self.params.col_actions_lambda(info)
                            if depth_bound_ > 0:

                                idx_tensor[0, row, col, chance] = idx
                                idx += 1

                                info_ = Info(
                                    depth_bound=depth_bound_, 
                                    terminal_values=info.terminal_values, 
                                    row_actions=row_actions_, 
                                    col_actions=col_actions_, 
                                    transition_threshold=info.transition_threshold)
    
                                v_, l_, p_, i_, n_, y_, e_ = _generate(info_)
                            
                                payoff_ = y_[:1]

                                lengths.append(v_.shape[0]-1)

                                child_value_tensors.append(v_)
                                child_legal_tensors.append(l_)
                                child_prob_tensors.append(p_)
                                child_idx_tensors.append(i_)
                                child_nash_tensors.append(n_)
                                child_payoff_tensors.append(y_)
                                child_expected_tensors.append(e_)

                            else:
                                #node is terminal
                                lengths.append(0)
                                payoff_ = torch.tensor(((random.choice(self.params.terminal_values),),)).to(self.params.device)
                            
                            value_tensor[0, row, col, chance] = payoff_.item()
                            
                    expected_value_tensor[0, row, col, 0] = torch.sum(value_tensor[0, row, col, :] * prob_tensor[0, row, col, :])

            # Get payoff of parent value tensor

            M = expected_value_tensor[0, :info.row_actions, :info.col_actions, 0]
            nash_tensor = self.solve(M, self.params.max_actions)
            pi = nash_tensor[0]
            pi_1, pi_2 = pi[:info.row_actions].unsqueeze(dim=0), pi[self.params.max_actions:][:info.col_actions].unsqueeze(dim=1)
            payoff_tensor = torch.matmul(torch.matmul(pi_1, M), pi_2)
            nash_tensor = nash_tensor[0].unsqueeze(dim=0)

            # Make the changes to parent index tensors
            _ = 0 
            for row in range(info.row_actions):
                for col in range(info.col_actions):
                    for chance in range(self.params.max_transitions):
                        if idx_tensor[0, row, col, chance] != 0:
                            idx_tensor[0, row, col, chance] += sum(lengths[:_])
                            _ += 1

            # Make the changes to child index tensors
            for _, child_idx_tensor in enumerate(child_idx_tensors):
                mask = child_idx_tensor.clone()
                mask[mask > 0] = 1.
                mask *= sum(lengths[:_]) + _ + 1
                child_idx_tensor += mask
            
            # return

            child_value_tensors.insert(0, value_tensor)
            child_legal_tensors.insert(0, legal_tensor)
            child_prob_tensors.insert(0, prob_tensor)
            child_idx_tensors.insert(0, idx_tensor)
            child_nash_tensors.insert(0, nash_tensor)
            child_payoff_tensors.insert(0, payoff_tensor)
            child_expected_tensors.insert(0, expected_value_tensor)

            v = torch.cat(child_value_tensors, dim=0)
            l = torch.cat(child_legal_tensors, dim=0)
            p = torch.cat(child_prob_tensors, dim=0)
            i = torch.cat(child_idx_tensors, dim=0)
            n = torch.cat(child_nash_tensors, dim=0)
            y = torch.cat(child_payoff_tensors, dim=0)
            e = torch.cat(child_expected_tensors, dim=0)

            return v, l, p, i, n, y, e

        # generate() body:

        root_info = Info(self.params.depth_bound, self.params.terminal_values, self.params.max_actions, self.params.max_actions, self.params.transition_threshold)
        v, l, p, i, n, y, e = _generate(root_info)

        expected_value_tensor = torch.zeros(self.legal_tensor_shape, device=self.params.device, dtype=torch.float)
        value_tensor = torch.zeros(self.value_tensor_shape, device=self.params.device, dtype=torch.float)
        legal_tensor = torch.zeros(self.legal_tensor_shape, device=self.params.device, dtype=torch.float)
        legal_tensor[0, 0, 0, 0] = 1.
        prob_tensor = torch.zeros(self.value_tensor_shape, device=self.params.device, dtype=torch.float)
        prob_tensor[0, 0, 0, 0] = 1.
        idx_tensor = torch.zeros(self.value_tensor_shape, device=self.params.device, dtype=torch.int32)

        test_idx_tensor(i)

        return v, l, p, i, n, y, e
    



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

    tree_params = TreeParameters(depth_bound=6, max_actions=2, transition_threshold=.4, max_transitions=2)
    tree = Tree(tree_params)

    v, l, p, i, n, y, e = tree.generate()


    for _ in range(v.shape[0]):

        print('\n', _ + 1)
        print('expected value', e[_])
        print('index', i[_])
        print('payoff', y[_])
        print('strategy', n[_])
        break