import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import game

class CrossConv (nn.Module) :
 
    def __init__ (self, size, in_channels, out_channels) :
        super().__init__()
        self.size = size
        self.row_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 2*size-1))
        self.col_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(2*size-1, 1))

    def forward (self, input):
        x = F.pad(input, (self.size-1, self.size-1, 0, 0, 0, 0))
        r = self.row_conv(x)
        y = F.pad(input, (0, 0, self.size-1, self.size-1, 0, 0))
        c = self.col_conv(y)
        return r + c

class ConvResBlock (nn.Module):

    def __init__ (self, size, channels, batch_norm=True) :
        super().__init__()
        self.conv0 = CrossConv(size, channels, channels)
        self.conv1 = CrossConv(size, channels, channels)
        self.relu = torch.relu
        if batch_norm > 0:
            self.batch_norm0 = nn.BatchNorm2d(channels)
            self.batch_norm1 = nn.BatchNorm2d(channels)
        else:
            self.batch_norm0 = nn.Identity()
            self.batch_norm1 = nn.Identity()
    
    def forward (self, input_batch):
        return input_batch + self.batch_norm1(self.relu(self.conv1(self.batch_norm0(self.relu(self.conv0(input_batch))))))

class ConvNet (nn.Module):

    def __init__ (self, size, channels, depth=1, batch_norm=True) :
        super().__init__()
        self.size = size
        self.channels = channels
        self.pre = CrossConv(size, in_channels=2, out_channels=channels)
        self.tower = nn.ParameterList([ConvResBlock(size, channels, batch_norm) for _ in range(depth)])
        self.policy = nn.Linear(channels * (size**2), size)
        self.value =  nn.Linear(channels * (size**2), 1)

    def forward (self, input_batch) :
        x = input_batch
        x = self.pre(x)
        for block in self.tower:
            x = block(x)

        x = x.contiguous().view(-1, self.channels*(self.size**2))
        logits_batch = self.policy(x)
        policy_batch = F.softmax(logits_batch, dim=1)
        value_batch = self.value(x)
        return logits_batch, policy_batch, value_batch


if __name__ == '__main__' :

    tree_params = game.TreeParameters(depth_bound=4)
    tree = game.Tree(tree_params)
    tree.generate()

    net = ConvNet(size=tree_params.max_actions, channels=6, depth=3)

    states = tree.initial(11)
    observation = states.observation()
    # print(observation.shape)
    logits, policy, value = net.forward(observation)
    print(policy)