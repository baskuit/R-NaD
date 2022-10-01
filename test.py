import torch
from torch import tensor

t = tensor([[[1, 2],
             [3, 4]],
            [[5, 6],
             [7, 8]]])

ix = tensor([1, 0])
ix0 = torch.arange(0, ix.shape.numel())

print(ix0)