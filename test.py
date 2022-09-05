import torch

batch = 2
n = 5
probs = torch.nn.functional.normalize(torch.rand((batch, n)), dim=1, p=1)

epsilon = .1
print(probs)
probs = probs - torch.where(probs < epsilon, probs, 0)
probs = torch.nn.functional.normalize(probs, dim=1, p=1)


print(probs)