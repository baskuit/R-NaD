import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import logging
import random

# import vtrace
# import metric

class CrossConv (nn.Module) :
 
    def __init__ (self, size, in_channels, out_channels, device=torch.device('cpu:0'), dtype=torch.float) :
        super().__init__()
        self.size = size
        self.row_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 2*size-1), device=device, dtype=dtype)
        self.col_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(2*size-1, 1), device=device, dtype=dtype)

    def forward (self, input) -> torch.Tensor:
        x = F.pad(input, (self.size-1, self.size-1, 0, 0,))
        r = self.row_conv(x)
        y = F.pad(input, (0, 0, self.size-1, self.size-1,))
        c = self.col_conv(y)
        return r + c

class ConvResBlock (nn.Module):

    def __init__ (self, size, channels, batch_norm=True, device=torch.device('cpu:0'), dtype=torch.float) :
        super().__init__()
        self.conv0 = CrossConv(size, channels, channels, device=device, dtype=dtype)
        self.conv1 = CrossConv(size, channels, channels, device=device, dtype=dtype)
        self.relu = torch.relu
        if batch_norm > 0:
            self.batch_norm0 = nn.BatchNorm2d(channels, device=device, dtype=dtype)
            self.batch_norm1 = nn.BatchNorm2d(channels, device=device, dtype=dtype)
        else:
            self.batch_norm0 = nn.Identity()
            self.batch_norm1 = nn.Identity()
    
    def forward (self, input_batch) -> torch.Tensor:
        return input_batch + self.batch_norm1(self.relu(self.conv1(self.batch_norm0(self.relu(self.conv0(input_batch))))))

class ConvNet (nn.Module):

    def __init__ (self, size, channels, depth=1, batch_norm=True, device=torch.device('cpu:0'), dtype=torch.float) :
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.size = size
        self.channels = channels
        self.pre = CrossConv(size, in_channels=2, out_channels=channels, device=device, dtype=dtype)
        self.tower = nn.ParameterList([ConvResBlock(size=size, channels=channels, batch_norm=batch_norm, device=device, dtype=dtype) for _ in range(depth)])
        self.policy = nn.Linear(channels * (size**2), size, device=device, dtype=dtype)
        self.value =  nn.Linear(channels * (size**2), 1, device=device, dtype=dtype)

    def forward (self, input_batch):
        filter_row = input_batch[:, 1, :, 0]
        x = input_batch
        x = self.pre(x)
        for block in self.tower:
            x = block.forward(x)

        x = x.view(-1, self.channels*(self.size**2))
        logits = self.policy(x)
        policy = F.softmax(logits, dim=1)
        policy *= filter_row
        F.normalize(policy, dim=1)
        value = self.value(x)
        actions = torch.squeeze(torch.multinomial(policy, num_samples=1))
        return logits, policy, value, actions

    def forward_policy (self, input_batch) -> torch.Tensor:
        """
        Does not use value head but does perform legal actions masking
        """
        filter_row = input_batch[:, 1, :, 0]
        x = input_batch
        x = self.pre(x)
        for block in self.tower:
            x = block(x)

        x = x.view(-1, self.channels*(self.size**2))
        logits = self.policy(x)
        policy = F.softmax(logits, dim=1)
        policy *= filter_row
        F.normalize(policy, dim=1)
        return policy



if __name__ == '__main__' :


    import game
    import metric

    def speed_test (net):
        start_generating = time.perf_counter()
        # Speed test
        batch_size = 10**4
        steps = 0
        for trial in range(100):
            episodes = game.Episodes(tree, batch_size)
            episodes.generate(net)
            steps += episodes.t_eff * batch_size
        
        end_generating = time.perf_counter()
        speed = steps / (end_generating - start_generating)
        logging.debug('{} steps/sec'.format(int(speed)))


    
    logging.basicConfig(level=logging.DEBUG)

    depth_bound_lambda = lambda tree : max(0, tree.depth_bound - (1 if random.random() < .5 else 2))

    tree = game.Tree(
        max_actions=2,
        depth_bound=2,
        max_transitions=2,
        # depth_bound_lambda=depth_bound_lambda
    )

    tree._generate()
    print('tree generated, size: ', tree.value.shape)
    tree.to(torch.device('cuda:0'))

   
    batch_size = 2**1
    net_channels=32
    net_depth=1
    net =  ConvNet(size=tree.max_actions, channels=net_channels, depth=net_depth, device=tree.device)

    expl = metric.nash_conv(tree, net)
    print(expl)