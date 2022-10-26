import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import game
import vtrace

class Inference ():

    def __init__ (self, logits_square=None, logits_move=None, value=None):
        self.logits_square = logits_square
        self.logits_move = logits_move
        self.value = value

    def flip (self):
        self.logits_square = torch.flip(self.logits_square, dims=(1,2,3))
        self.logits_move = torch.flip(self.logits_move, dims=(1,2,3))

class ConvResBlock (nn.Module):

    def __init__ (self, channels, batch_norm=True, device=torch.device('cpu'), dtype=torch.float) :
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=(3,3), device=device, padding=1, dtype=dtype)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(3,3), device=device, padding=1 ,dtype=dtype)
        self.relu = torch.relu
        if batch_norm:
            self.batch_norm0 = nn.BatchNorm2d(channels, device=device, dtype=dtype)
            self.batch_norm1 = nn.BatchNorm2d(channels, device=device, dtype=dtype)
        else:
            self.batch_norm0 = nn.Identity()
            self.batch_norm1 = nn.Identity()
    
    def forward (self, input_batch:torch.Tensor):
        return input_batch + self.batch_norm1(self.relu(self.conv1(self.batch_norm0(self.relu(self.conv0(input_batch))))))

class ConvNet (nn.Module):

    def __init__ (self, height, width, in_channels, channels, depth=1, batch_norm=True, device=torch.device('cpu'), dtype=torch.float) :
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.height = height
        self.width = width
        self.channels = channels
        self.pre = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=(3,3), padding=1, device=device, dtype=dtype)
        self.tower = nn.ParameterList([ConvResBlock(channels, batch_norm, device=device, dtype=dtype) for _ in range(depth)])
        self.policy_move = nn.Conv2d(channels, 16, kernel_size=(3,3), padding=1, device=device, dtype=dtype)
        self.policy_square = nn.Conv2d(channels, 1, kernel_size=(3,3), padding=1, device=device, dtype=dtype)
        self.value =  nn.Linear(channels * height * width, 1, device=device, dtype=dtype)

    def forward (self, input:torch.Tensor) -> Inference:
        x = self.pre(input)
        for block in self.tower:
            x = block(x)

        logits_move = self.policy_move(x)
        logits_square = self.policy_square(x)
        value = self.value(x.view(-1, self.channels * self.height * self.width))
        return Inference(logits_square, logits_move, value)

def get_size (net:torch.nn.Module):
    param_size = 0
    for param in net.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in net.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = param_size + buffer_size
    return size_all_mb / 2**20

if __name__ == '__main__' :

    start = time.time()

    tree = game.Tree(depth_bound=3, max_transitions=2, transition_threshold=.0)
    tree._generate()
    tree.to(torch.device('cuda:0'))
    print('tree chance shape', tree.chance.shape)

    done_generating = time.time()
    print('game generation time',(done_generating - start)/1)

    net = ConvNet(size=tree.max_actions, channels=6, depth=3).to(device=tree.device)
    net_ = ConvNet(size=tree.max_actions, channels=6, depth=3).to(device=tree.device)
    
    batch_size = 2**1
    episodes = game.Episodes(tree, batch_size)
    episodes.generate(net)

    vtrace.transform_rewards(episodes, net, net_, net_, .2)


    vtrace.estimate(episodes,reward_transform=None)

    done_stepping = time.time()
    print('episode generation time', (done_stepping - done_generating)/1)


