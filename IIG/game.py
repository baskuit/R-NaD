import torch
import pygambit

# This is an implementation of a two-player matrix tree with stochastic transtions.
# The tree is represented as a tensor of states and transition indexes of shapes 
# (n_states, max_actions, max_actions, 2) and (n_states, max_actions, max_actions, max_transitions), resp.

# The states tensor has payoff and actions masking info at dim=3
# and dim=3 of the transitions tensor is a probability distribution on indexes of the states tensor

# 

class Info () :

    def __init__ (self, idx=1, depth=0, values=[-1, 1], n_actions=1) :

        self.idx = idx  
        self.depth = depth
        self.values = values
        self.n_actions = n_actions

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

        # create absorbing state at index=0 to represent terminal states
        self.transition_probs = torch.zeros((1, params['max_actions'], params['max_actions'], params['max_transitions']), device=params['device'], dtype=torch.float)
        self.transition_indexes = torch.zeros((1, params['max_actions'], params['max_actions'], params['max_transitions']), device=params['device'], dtype=torch.int)
        self.transition_probs[0, 0, 0, 0] = 1.

        self.turn_ = 1

    # Randomly generate a game tree and solve it recursively
    def _generate (self, info) :

        # Depth is max possible distance from terminal state
        # so depth = 1

        # TODO Stochastics. 

        s = torch.zeros((1, self.param['max_actions'], self.param['max_actions'], 2), device=self.param['device'], dtype=torch.float)
        p = torch.zeros((1, self.param['max_actions'], self.param['max_actions'], self.param['max_trasitions']), device=self.param['device'], dtype=torch.float)
        i = torch.zeros((1, self.param['max_actions'], self.param['max_actions'], self.param['max_trasitions']), device=self.param['device'], dtype=torch.int)

        

        if info.depth > 0:

            idx_ = info.idx # incremented herein

            s[0, :info.n_actions, :info.n_actions, 1] = 1. # action mask

            for i in info.n_actions:
                for j in info.n_actions:
                    
                    # get new depth etc for child
                    depth_ = self.params['depth_lambda'](info.depth)
                    n_actions_ = self.params['n_actions_lambda'](info.n_actions)

                    if depth_ > 0:
                        idx_ += 1

                        info_ = Info(idx_, depth_, info.values, n_actions_)
                        
                        t_ = self._generate(info_)
                    else:
                        pass
                        #child is terminal, go to absorbing state

            return None



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