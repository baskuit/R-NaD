import torch
import pygambit
import numpy as np

import Metric

# Intended to compute, save, and load the collection of nxn zero sum matrices that satisfy certain conditions
class Game () :

    def __init__ (self, size, values=[-1, 0, 1]) :
        self.size = size
        self.values = values
        self.matrices = None
        self.strategies0 = None
        self.strategies1 = None
        self.expanded = False
    
    def generate (self, current=[], values=[-1, 0, 1]) :
        if len(current) == self.size**2 :
            yield torch.tensor(current, dtype=torch.float).view(-1, self.size, self.size)
        else:
            for _ in values :
                next = current.copy()
                next.append(_)
                for __ in self.generate(next, values) :
                    yield __

    def solve_filtered (self, unique=True, interior=True) :
        self.matrices = torch.cat(list(self.generate([], self.values)), dim=0)
    
        self.matrices, strategy_list = solve_batch_filtered(self.matrices, unique, interior)
        strategy_list = [s[0] for s in strategy_list]

        strategies = torch.stack(strategy_list, dim=0)
        strategies = split_strategies(strategies)
        
        self.strategies0 = strategies[0]
        self.strategies1 = strategies[1]

    def generate_input_batch (self, batch_size) :

        m = self.matrices.shape[0]
        idx = torch.randint(0, m-1, (batch_size,))
        return self.matrices[idx].detach(), self.strategies0[idx], self.strategies1[idx]

    # TODO if/when needed
    def save (self) :
        pass

    def load (self) :
        pass


    


#solve a single mxn matrix, return b x 2*n tensor
def solve (input) :

    M = input
    size = M.shape[0]
    N = np.zeros((size, size), dtype=pygambit.Decimal)
    for _ in range(size) :
        for __ in range(size) :
            N[_][__] = pygambit.Decimal(M[_][__].item())

    g = pygambit.Game.from_arrays(N, -N)
    solutions = [[S[_] for _ in range(2 * size)] for S in pygambit.nash.enummixed_solve(g, rational=False)]
    return torch.tensor(solutions, dtype=torch.float)

#split (b, 2 * n) into (2, b, n)
def split_strategies (solutions):
    b = solutions.shape[0]
    return solutions.view(b, 2, -1).swapaxes(0, 1)

# Returns filtered batch of length b' and *list* of strategies of shape (2*size, n)
def solve_batch_filtered (input_batch, unique=True, interior=True) :

    b = input_batch.shape[0]

    filter = [] # dim=0 has legnth b
    solutions_list = [] # has length <= b

    for _ in range(b) :
        solutions = solve(input_batch[_])
        if solutions.shape[0] > 1 and unique:
            filter.append(False)
            continue

        if interior :
            solutions = solutions[~torch.any(solutions == 0, dim=1)]

        if solutions.shape[0] == 0:
            filter.append(False)
            continue

        solutions_list.append(solutions)
        filter.append(True)

    return input_batch[filter], solutions_list





def normal_batch (size, batch_size, mean=0, std=1) :
# with torch.no_grad():
    return torch.normal(mean, std, size=(batch_size, size, size))

def discrete_batch (size, batch_size, values=(-1, 0, 1)) :
    values = torch.tensor(values, dtype=torch.float)
    return values[torch.randint(len(values), (batch_size, size, size))]  

def flip (input_batch) :
    return torch.swapaxes(-input_batch, 1, 2)

def flip_cat (input_batch) :
    return torch.cat((input_batch, flip(input_batch)), dim=0)

def normal_batch_flip_cat (size, batch_size, mean=0, std=1) :
    return flip_cat(normal_batch(size, batch_size, mean=0, std=1))

def first_half (input_batch) :
    return input_batch[ : input_batch.shape[0]//2]

def second_half (input_batch) :
    return input_batch[input_batch.shape[0]//2 : ]

def epsilon_threshold(input_batch, epsilon):
    pass



RPS = torch.tensor([[[0, -1, 1], [1, 0, -1], [-1, 1, 0]]], dtype=torch.float)



if __name__ == '__main__':

    x = Game(3)
    x.solve_filtered(unique=True, interior=True)
    input_batch, strategy0, strategy1  = x.generate_input_batch(2)
    print(input_batch)
    print(strategy0)
    print(strategy1)