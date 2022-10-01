import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import game
import vtrace

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
        filter_row = input_batch[:, 1, :, 0]
        x = input_batch
        x = self.pre(x)
        for block in self.tower:
            x = block(x)

        x = x.contiguous().view(-1, self.channels*(self.size**2))
        logits = self.policy(x)
        policy = F.softmax(logits, dim=1)
        policy *= filter_row
        F.normalize(policy, dim=1)
        value = self.value(x)
        actions = torch.squeeze(torch.multinomial(policy, num_samples=1))
        return logits, policy, value, actions

    def forward_policy (self, input_batch) :
        filter_row = input_batch[:, 1, :, 0]
        x = input_batch
        x = self.pre(x)
        for block in self.tower:
            x = block(x)

        x = x.contiguous().view(-1, self.channels*(self.size**2))
        logits = self.policy(x)
        policy = F.softmax(logits, dim=1)
        policy *= filter_row #multiply policy by actions mask
        F.normalize(policy, dim=1)
        return policy



if __name__ == '__main__' :

    start = time.time()

    tree_params = game.TreeParameters(depth_bound=3, max_transitions=2, transition_threshold=.0)
    tree = game.Tree(tree_params)
    tree.generate()
    tree.to(torch.device('cuda:0'))
    print('tree chance shape', tree.data.chance.shape)

    done_generating = time.time()
    print('game generation time',(done_generating - start)/1)

    net = ConvNet(size=tree_params.max_actions, channels=6, depth=3).to(device=tree.params.device)
    net_ = ConvNet(size=tree_params.max_actions, channels=6, depth=3).to(device=tree.params.device)
    
    batch_size = 2**1
    episodes = game.Episodes(tree, batch_size)
    episodes.generate(net)

    vtrace.transform_rewards(episodes, net, net_, net_, .2)


    vtrace.estimate(episodes,reward_transform=None)

    done_stepping = time.time()
    print('episode generation time', (done_stepping - done_generating)/1)

