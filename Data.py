import torch
import pygambit
import numpy as np

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


RPS = torch.tensor([[[0, -1, 1], [1, 0, -1], [-1, 1, 0]]], dtype=torch.float)

def solve (input_batch) :

    if len(input_batch.shape) == 3:
        return [solve(_) for _ in input_batch]

    # single n x n matrices now
    M = input_batch
    size = M.shape[0]
    N = np.zeros((size, size), dtype=pygambit.Decimal)
    for _ in range(size) :
        for __ in range(size) :
            N[_][__] = pygambit.Decimal(M[_][__].item())

    g = pygambit.Game.from_arrays(N, -N)
    solutions = pygambit.nash.enummixed_solve(g, rational=False)[0]

    solutions0 = [solutions[_] for _ in range(size)]
    solutions1 = [solutions[_] for _ in range(size, 2*size)]
    solutions = [solutions0, solutions1]
    return solutions

if __name__ == '__main__':
    print('rand int')
    print(discrete_batch(3, 1))

    input_batch = normal_batch(3, 2**6)

    strategies = solve(input_batch)
    policy_batch = torch.tensor(strategies).swapaxes(0, 1)

    import Metric

    expl = Metric.expl(input_batch, policy_batch[0], policy_batch[1])
    # print(expl)
        # print(s1)

    