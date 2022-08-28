import torch
import torch.nn as nn
import torch.nn.functional as F

import Data

def neurd_loss (net, input_batch) :

    batch_size, size = input_batch.shape[:2]
    policy_batch, value_batch = net.forward(input_batch)

    

class FCResBlock (nn.Module):

    def __init__ (self, m, n, batch_norm=True) :
        super().__init__()
        self.n = n
        self.fc0 = nn.Linear(m, n)
        self.fc1 = nn.Linear(n, m)
        if batch_norm:
            self.bn0 = nn.BatchNorm1d(n)
            self.bn1 = nn.BatchNorm1d(m)
        else:
            self.bn0 = nn.Identity()
            self.bn1 = nn.Identity()
    
    def forward (self, input_batch):

        return input_batch + self.bn1(torch.relu(self.fc1(self.bn0(torch.relu(self.fc0(input_batch))))))