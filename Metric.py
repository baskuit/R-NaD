import Game

import torch

def expl (input_batch, policy_batch0, policy_batch1) :

        policy_batch_size, size, _ = input_batch.shape

        M = input_batch
        A = policy_batch0.view(policy_batch_size, 1, size)
        B = policy_batch1.view(policy_batch_size, size, 1)
        X = torch.matmul(A, M)
        Y = torch.matmul(M, B)
        min_X = torch.min(X, dim=2)[0]
        max_Y = torch.max(Y, dim=1)[0]
        expl = max_Y - min_X
        return expl

def mean_expl(net, input_batch) :

        policy_batch_size, size, _ = input_batch.shape
        input_batch_flip_cat = Game.flip_cat(input_batch)
        logit_batch, policy_batch, value_batch = net.forward(input_batch_flip_cat)
        policy_batch0 = policy_batch[:policy_batch_size]
        policy_batch1 = policy_batch[policy_batch_size:]
        e = expl(input_batch, policy_batch0, policy_batch1)
        return torch.mean(e)
