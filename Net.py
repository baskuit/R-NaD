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




    ### Conv 2D




class CrossConv (nn.Module) :
    def __init__ (self, size, in_channels, out_channels) :
        super().__init__()
        self.size = size
        self.row_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 2*size-1))
        self.col_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(2*size-1, 1))

    def forward (self, input):
        x = F.pad(input, (self.size-1, self.size-1, 0, 0))
        r = self.row_conv(x)
        y = F.pad(input, (0, 0, self.size-1, self.size-1))
        c = self.col_conv(y)
        return r + c

class ConvResBlock (nn.Module):

    def __init__ (self, size, channels, batch_norm=True) :
        super().__init__()
        self.conv0 = CrossConv(size, channels, channels)
        self.conv1 = CrossConv(size, channels, channels)
        self.leaky = nn.LeakyReLU()
        if batch_norm > 0:
            self.batch_norm0 = nn.BatchNorm2d(channels)
            self.batch_norm1 = nn.BatchNorm2d(channels)
        else:
            self.batch_norm0 = nn.Identity()
            self.batch_norm1 = nn.Identity()
    
    def forward (self, input_batch):
        return input_batch + self.batch_norm1(self.leaky(self.conv1(self.batch_norm0(self.leaky(self.conv0(input_batch))))))

class ConvNet (nn.Module):

    def __init__ (self, size, channels, depth=1, batch_norm=True) :
        super().__init__()
        self.size = size
        self.channels = channels
        self.pre = CrossConv(size, 1, channels)
        self.tower = nn.ParameterList([ConvResBlock(size, channels, batch_norm) for _ in range(depth)])
        self.policy = nn.Linear(channels * (size**2), size)
        self.value =  nn.Linear(channels * (size**2), 1)

    def forward (self, input_batch) :
        x = input_batch.unsqueeze(dim=1)
        x = self.pre(x)
        for block in self.tower:
            x = block(x)
        x = x.view(-1, self.channels*(self.size**2))
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

def step_cel (net, optimizer, scheduler, input_batch, strategies0, strategies1) :
    batch_size, size = input_batch.shape[:2]
    input_batch_flip_cat = Data.flip_cat(input_batch)
    logits_batch, policy_batch, value_batch = net.forward(input_batch_flip_cat)

    target_batch = torch.cat((strategies0, strategies1), dim=0)
    crossentropy_loss = F.cross_entropy(policy_batch, target_batch, reduction='mean')

    loss = crossentropy_loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

if __name__ == '__main__' :
    size = 5
    net = ConvNet(size=size, channels=3, depth=1, batch_norm=True)
    input_batch = Data.discrete_batch(size, 2**1)

    output_batch = net(input_batch)
    print(output_batch)
    exit()
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