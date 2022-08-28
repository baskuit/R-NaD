import torch

def expl (input_batch, policy_batch0, policy_batch1) :
        input_size, size, _ = input_batch.shape
        M = input_batch
        A = policy_batch0.view(input_size, 1, size)
        B = policy_batch1.view(input_size, size, 1)
        X = torch.matmul(A, M)
        Y = torch.matmul(M, B)
        min_X = torch.min(X, dim=2)[0]
        max_Y = torch.max(Y, dim=1)[0]
        expl = max_Y - min_X
        return expl

