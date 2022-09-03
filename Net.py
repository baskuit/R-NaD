import torch
import torch.nn as nn
import torch.nn.functional as F

import Data
import Metric

class FCResBlock (nn.Module):

    def __init__ (self, m, n, dropout=.5) :
        super().__init__()
        self.fc0 = nn.Linear(m, n)
        self.fc1 = nn.Linear(n, m)
        self.leaky = nn.LeakyReLU()
        if dropout > 0:
            self.dropout0 = nn.Dropout(p=dropout)
            self.dropout1 = nn.Dropout(p=dropout)
        else:
            self.dropout0 = nn.Identity()
            self.dropout1 = nn.Identity()
    
    def forward (self, input_batch):
        
        return input_batch + self.dropout1(self.leaky(self.fc1(self.dropout0(self.leaky(self.fc0(input_batch))))))
        
class FCNet (nn.Module):

    def __init__ (self, size, width, depth=1, dropout=.5) :
        super().__init__()
        self.size = size
        self.pre = nn.Linear(size**2, width)
        self.tower = nn.ParameterList([FCResBlock(width, width, dropout) for _ in range(depth)])
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

def step_neurd (net, optimizer, scheduler, input_batch, eta=0, net_fixed=None, grad_clip=1000) :

    batch_size, size = input_batch.shape[:2]
    # [H, H, .. H]
    # [H, H, .. H, -I, -I, ..-I]
    # flip cat spilts each matrix game into 2 states
    input_batch_flip_cat = Data.flip_cat(input_batch)
    logits_batch, policy_batch, value_batch = net.forward(input_batch_flip_cat)
    
    actions = torch.squeeze(torch.multinomial(policy_batch, num_samples=1), dim=-1)
    action_logits = logits_batch[torch.arange(2*batch_size), actions]
    pi = policy_batch[torch.arange(2*batch_size), actions]

    A = actions[:batch_size]
    B = actions[batch_size:]
    player_one_rewards = input_batch[torch.arange(batch_size), A][torch.arange(batch_size), B]
    # entries of matrix corresponding to pairs of selected actions

    if eta > 0:
        with torch.no_grad():
            logits_batch_, policy_batch_, value_batch_ = net_fixed.forward(input_batch_flip_cat)
        mu = policy_batch_[torch.arange(2*batch_size), actions]
        eta_log = eta * (torch.log(pi) - torch.log(mu))
        player_one_rewards -= Data.first_half(eta_log)
        player_one_rewards += Data.second_half(eta_log)

    rewards = torch.cat((player_one_rewards, -player_one_rewards), dim=0)

    advantages = rewards - value_batch.squeeze(dim=1)

    policy_loss = torch.mean(-action_logits * advantages.detach() / pi.detach())

    value_loss = torch.mean(advantages**2)

    entropy_loss = F.cross_entropy(policy_batch, policy_batch)

    loss = policy_loss + value_loss + entropy_loss
    loss.backward()
    # for _ in net.parameters():
    #     print(_.grad)
    nn.utils.clip_grad_norm_(net.parameters(), grad_clip) #TODO grad clip param
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

def step_cel (net, optimizer, scheduler, input_batch) :
    batch_size, size = input_batch.shape[:2]
    input_batch_flip_cat = Data.flip_cat(input_batch)
    logits_batch, policy_batch, value_batch = net.forward(input_batch_flip_cat)

    strategies = Data.solve(input_batch)
    target_batch = torch.tensor(strategies).swapaxes(0, 1)
    A = target_batch[0].view(batch_size, 1, size)
    B = target_batch[1].view(batch_size, size, 1)
    payoff_batch = torch.matmul(torch.matmul(A, input_batch), B)
    payoff_batch = Data.flip_cat(payoff_batch).view(2*batch_size, 1)
    target_batch = torch.cat((target_batch[0], target_batch[1]), dim=0)

    crossentropy_loss = F.cross_entropy(policy_batch, target_batch, reduction='mean')
    # value_loss = torch.mean((payoff_batch - value_batch)**2)

    loss = crossentropy_loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    # Need NE strats and payoffs

if __name__ == '__main__' :
    batch_size = 2**10
    total_steps = 2**10
    old_net = FCNet(3, 9, 1)
    net = FCNet(3, 9, 1)
    optimizer = torch.optim.SGD(net.parameters())
    def lr_lambda(epoch):
        return .01

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # for step in range(total_steps):
    #     input_batch = Data.normal_batch(3, batch_size)
    #     step_neurd(net, optimizer, scheduler, input_batch)

    step_neurd (net, optimizer, scheduler, Data.RPS, 0, old_net)