from email import policy
from pickletools import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F

import Data

def step_neurd (net, optimizer, scheduler, input_batch) :

    batch_size, size = input_batch.shape[:2]
    # [H, H, .. H]
    # [H, H, .. H, -I, -I, ..-I]
    # flip cat spilts each matrix game into 2 states
    input_batch_flip_cat = Data.flip_cat(input_batch)
    logits_batch, policy_batch, value_batch = net.forward(input_batch_flip_cat)
    
    actions = torch.squeeze(torch.multinomial(policy_batch, num_samples=1), dim=-1)
    action_logits = logits_batch[torch.arange(2*batch_size), actions]
    action_probabilities = policy_batch[torch.arange(2*batch_size), actions]

    A = actions[:batch_size]
    B = actions[batch_size:]
    player_one_rewards = input_batch[torch.arange(batch_size), A][torch.arange(batch_size), B]
    # entries of matrix corresponding to pairs of selected actions

    rewards = torch.cat((player_one_rewards, -player_one_rewards), dim=0)
    advantages = rewards - value_batch.squeeze(dim=1)
    policy_loss = torch.sum(-action_logits * advantages.detach() / action_probabilities.detach())
    value_loss = torch.sum(advantages**2)
    loss = policy_loss + value_loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

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
        
class FCNet (nn.Module):

    def __init__ (self, size, width, depth=1, batch_norm=True) :
        super().__init__()
        self.size = size
        self.pre = nn.Linear(size**2, width)
        self.tower = nn.ParameterList([FCResBlock(width, width, batch_norm) for _ in range(depth)])
        self.policy = nn.Linear(width, size)
        self.value = nn.Linear(width, 1)

    def forward (self, input_batch) :
        x = input_batch.view(-1, self.size**2)
        x = self.pre(x)


        for block in self.tower:
            x = block(x)

        logits_batch = self.policy(x)
        policy_batch = F.softmax(logits_batch, dim=1)
        value_batch = self.value(x)
        return logits_batch, policy_batch, value_batch



if __name__ == '__main__' :
    batch_size = 2**1
    total_steps = 2**0
    net = FCNet(3, 27)
    input_batch = Data.normal_batch(3, batch_size)
    # net.forward(input_batch)
    optimizer = torch.optim.SGD(net.parameters(), lr=.01)
    def lr_lambda(epoch):
        return .001

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step_neurd(net, optimizer, scheduler, input_batch)
