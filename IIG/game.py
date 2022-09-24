import torch
import pygambit
import numpy as np
import random

# This is an implementation of a two-player matrix tree with stochastic transtions.
# The tree is represented as a tensor of states and transition indexes of shapes 
# (n_states, max_actions, max_actions, 2) and (n_states, max_actions, max_actions, max_transitions), resp.

# The states tensor has payoff and actions masking info at dim=3
# and dim=3 of the transitions tensor is a probability distribution on indexes of the states tensor

# 

class Info () :

    def __init__ (self, depth=0, values=[-1, 1], n_actions=1, epsilon=.1, solve=False) :
 
        self.depth = depth
        self.values = values
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.solve = solve

class Tree () :

    def __init__ (self, params={}) :
    
        if 'device' not in params:
            params['device'] = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if 'max_actions' not in params:
            params['max_actions'] = 5
        if 'max_transitions' not in params:
            params['max_transitions'] = 5

        if 'n_actions_lambda' not in params:
            params['n_actions_lambda'] = lambda info : info.n_actions - 1
        if 'depth_lambda' not in params:
            params['depth_lambda'] = lambda info : info.depth  - 1
        if 'terminal_values' not in params:
            params['terminal_values'] = [1, -1]
        self.params = params

        # create absorbing state at index=0 to represent terminal states
        self.value_tensor =             torch.zeros((1, params['max_actions'], params['max_actions'], params['max_transitions']),   device=params['device'], dtype=torch.float)
        self.expected_value_tensor =    torch.zeros((1, params['max_actions'], params['max_actions'], 1),                           device=params['device'], dtype=torch.float)
        self.legal_tensor =             torch.zeros((1, params['max_actions'], params['max_actions'], 1),                           device=params['device'], dtype=torch.float)
        self.legal_tensor[0, 0, 0, 0] = 1.
        self.prob_tensor =              torch.zeros((1, params['max_actions'], params['max_actions'], params['max_transitions']),   device=params['device'], dtype=torch.float)
        self.prob_tensor[0, 0, 0, 0] = 1.
        self.idx_tensor =               torch.zeros((1, params['max_actions'], params['max_actions'], params['max_transitions']),   device=params['device'], dtype=torch.int)

    def _transition_probs (self, rows, cols, n_trans, epsilon) :
        x = torch.from_numpy(np.random.dirichlet((1/n_trans,)*n_trans, (1,rows,cols)))
        x = x - torch.where(x < epsilon, x, 0)
        x = torch.nn.functional.normalize(x, p=1, dim=3)
        return x

    def _generate (self, info) :

        expected_value_tensor = torch.zeros((1, self.params['max_actions'], self.params['max_actions'], 1), device=self.params['device'], dtype=torch.float)
        value_tensor = torch.zeros((1, self.params['max_actions'], self.params['max_actions'], self.params['max_transitions']), device=self.params['device'], dtype=torch.float)
        legal_tensor = torch.zeros((1, self.params['max_actions'], self.params['max_actions'], 1), device=self.params['device'], dtype=torch.float)
        legal_tensor[0, :info.n_actions, :info.n_actions, 0] = 1.
        prob_tensor = self._transition_probs(self.params['max_actions'], self.params['max_actions'], self.params['max_transitions'], info.epsilon)
        idx_tensor = torch.zeros((1, self.params['max_actions'], self.params['max_actions'], self.params['max_transitions']), device=self.params['device'], dtype=torch.int)

        sub_value_tensors = []
        sub_legal_tensors = []
        sub_prob_tensors = []
        sub_idx_tensors = []

        lengths = [] # as many entries as transitions, same as sub's
        lengths_root = [] # row x cols many entries, used to init idx_tensor

        idx = 1 # incremented herein

        for row in range(info.n_actions):
            for col in range(info.n_actions):
                
                length_entry = 0

                for chance in range(self.params['max_transitions']):

                    transition_prob = prob_tensor[0, row, col, chance]

                    if transition_prob > 0:


                        # get new depth etc for child
                        depth_ = self.params['depth_lambda'](info)
                        n_actions_ = self.params['n_actions_lambda'](info)

                        if depth_ > 0:

                            idx_tensor[0, row, col, chance] = idx
                            info_ = Info(depth=depth_, values=info.values, n_actions=n_actions_, epsilon=info.epsilon, solve=info.solve)
                            
                            v_, l_, p_, i_, payoff_ = self._generate(info_)

                            lengths.append(v_.shape[0])
                            length_entry += v_.shape[0]

                            sub_value_tensors.append(v_) # wont change by batchin
                            sub_legal_tensors.append(l_)
                            sub_prob_tensors.append(p_)

                            sub_idx_tensors.append(i_) # will change by batching

                            idx += 1

                        else:
                            #node is terminal
                            lengths.append(0)
                            payoff_ = random.choice(self.params['terminal_values'])

                        value_tensor[0, row, col, chance] = payoff_
                        
                expected_value_tensor[0, row, col, 0] = torch.sum(value_tensor[0, row, col, :] * prob_tensor[0, row, col, :])
                lengths_root.append(length_entry)

        payoff = 0
        if info.solve:
            pass
        #change payoff

            
        # Make the changes to index tensors

        for row in range(info.n_actions):
            for col in range(info.n_actions):

                idx_tensor[0, row, col]
                            


        sub_value_tensors.insert(0, value_tensor)
        sub_legal_tensors.insert(0, legal_tensor)
        sub_prob_tensors.insert(0, prob_tensor)
        sub_idx_tensors.insert(0, idx_tensor)

        v = torch.cat(sub_value_tensors, dim=0)
        l = torch.cat(sub_legal_tensors, dim=0)
        p = torch.cat(sub_prob_tensors, dim=0)
        i = torch.cat(sub_idx_tensors, dim=0)

        return v, l, p, i, payoff



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


    def solve (self, input) :

        M = input
        rows, cols = M.shape[:2]
        N = np.zeros((rows, cols), dtype=pygambit.Decimal)
        for _ in range(rows) :
            for __ in range(cols) :
                N[_][__] = pygambit.Decimal(M[_][__].item())

        g = pygambit.Game.from_arrays(N, -N)
        solutions = [[S[_] for _ in range(rows + cols)] for S in pygambit.nash.enummixed_solve(g, rational=False)]
        return torch.tensor(solutions, dtype=torch.float)


if __name__ == '__main__' :

    game_params = {
        'device':torch.device('cpu'), 
        'max_actions':3, 
        'max_transitions':2,
    }
    tree = Tree(game_params)

    v, l, p, i, payoff = tree._generate(Info(depth=1, values=[-1, 1], n_actions=3, epsilon=.4, solve=False))

    print(i.shape)
    
    for _ in v:
        print(_)