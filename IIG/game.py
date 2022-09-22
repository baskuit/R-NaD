import torch
import pygambit

# This is an implementation of a two-player matrix tree with stochastic transtions.
# The tree is represented as a tensor of states and transition indexes of shapes 
# (n_states, max_actions, max_actions, 2) and (n_states, max_actions, max_actions, max_transitions), resp.

# The states tensor has payoff and actions masking info at dim=3
# and dim=3 of the transitions tensor is a probability distribution on indexes of the states tensor

# 

class Info () :

    def __init__ (self, idx=1, depth=0, values=[-1, 1], rows=1, cols=0) :

        self.idx = idx  
        self.depth = depth
        self.values = values
        self.rows = rows

class Tree () :

    def __init__ (self, params) :
    
        self.states = None

        if 'device' not in params:
            params['device'] = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if 'max_actions' not in params:
            params['max_actions'] = 5
        if 'max_transitions' not in params:
            params['max_transitions'] = 5

        if 'n_actions_lambda' not in params:
            params['n_actions_lambda'] = lambda info : info.n_actions - 1 if info.depth > 0 else 1
        if 'depth_lambda' not in params:
            params['n_actions_lambda'] = lambda info : info.depth  - 1 if info.depth > 0 else 0
        if 'terminal_values' not in params:
            params['terminal_values'] = [1, -1]

        self.params = params

        self.states = torch.zeros((1, params['max_actions'], params['max_actions'], 2), device=params['device'], dtype=torch.float)
        self.states[0, 0, 0, 1] = 1.

        self.transition_probs = torch.zeros((1, params['max_actions'], params['max_actions'], params['max_transitions']), device=params['device'], dtype=torch.int)
        self.transition_indexes = torch.zeros((1, params['max_actions'], params['max_actions'], params['max_transitions']), device=params['device'], dtype=torch.int)
        self.transition_probs[0, 0, 0, 0] = 1.

        self.turn_ = 1

    # Randomly generate a game tree and solve it recursively
    def generate (self, info) :

        if info.depth == 0:

            pass

        else:

            pass

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





if __name__ == '__main__' :

    t = torch.rand((2**24, 5, 5, 4))