import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np
import random
import time
import sys

class GridGameParams () :
    def __init__ (self, 
        device=torch.device('cpu'), 
        height=60,
        width=60,
        n_men=1,
        n_placable_walls=0,
        vision_weight=1/3, 
        vision_floor=.19,
        vision_seed_length=3,
        vision_padding=1#cant be 0 :(
        ) :

            
        self.device = device
        self.height = height
        self.width = width
        self.n_men = n_men
        self.n_placable_walls = n_placable_walls

        self.vision_seed_length = vision_seed_length
        self.vision_weight = vision_weight
        self.vision_floor = vision_floor
        self.vision_padding = vision_padding



class GridGame () :
    def __init__ (self, params = GridGameParams()):
        self.params = params
        k = params.vision_weight
        vision_filter_up = torch.tensor(
            ((0, 0, 0), (0, 0, 0), (k, k, k))
            , dtype=torch.float, device=self.params.device).unsqueeze(0).unsqueeze(0)
        vision_filter_right = torch.tensor(
            ((k, 0, 0), (k, 0, 0), (k, 0, 0))
            , dtype=torch.float, device=self.params.device).unsqueeze(0).unsqueeze(0)
        vision_filter_down = torch.tensor(
            ((k, k, k), (0, 0, 0), (0, 0, 0))
            , dtype=torch.float, device=self.params.device).unsqueeze(0).unsqueeze(0)
        vision_filter_left = torch.tensor(
            ((0, 0, k), (0, 0, k), (0, 0, k))
            , dtype=torch.float, device=self.params.device).unsqueeze(0).unsqueeze(0)

        filter = torch.cat([
            vision_filter_up,
            vision_filter_right,
            vision_filter_down,
            vision_filter_left], dim=0)
        self.vision_filter = filter.repeat_interleave(self.params.n_men, dim=0)
        

        vision_seed_vertical = torch.zeros((1, 1, 3, 3),device=self.params.device)
        vision_seed_vertical[0, 0, 1, :] = 1.

        vision_seed_horizonal = torch.zeros((1, 1, 3, 3), device=self.params.device)
        vision_seed_horizonal[0, 0, :, 1] = 1.

        vision_seed = torch.cat([vision_seed_vertical, vision_seed_horizonal, vision_seed_vertical, vision_seed_horizonal], dim=0)
        self.vision_seed = vision_seed.repeat_interleave(self.params.n_men, dim=0)
        
        



def clamp (x, a, b) :
    return max(a, min(x, b))

class States () :

    def __init__ (self, game : GridGame, batch_size=1, walls=None) :
        if walls is None:
            walls = torch.zeros((batch_size, 3, game.params.height, game.params.width), dtype=torch.float, device=game.params.device)

        self.game = game
        self.batch_size = batch_size
        self.walls = walls

        self.obstructions_mask_1 = 1 - torch.sum(self.walls[:, [0, 2], :, :], dim=1).unsqueeze(dim=1).repeat(1, self.game.params.n_men * 4, 1, 1)
        self.obstructions_mask_2 = 1 - torch.sum(self.walls[:, [0, 1], :, :], dim=1).unsqueeze(dim=1).repeat(1, self.game.params.n_men * 4, 1, 1)
        print('!', sys.getsizeof(self.obstructions_mask_1))

        men_shape = (batch_size, game.params.n_men, 2, game.params.height, game.params.width)
        self.men_1 = torch.zeros(men_shape, dtype=torch.float, device=self.game.params.device)
        self.men_2 = torch.zeros(men_shape, dtype=torch.float, device=self.game.params.device)

        # index in {0, 1, 2, 3} for north=0 and other directions, proceeding clockwise
        self.orientations_1 = torch.zeros((batch_size, game.params.n_men), dtype=torch.float, device=self.game.params.device)
        self.orientations_2 = torch.zeros((batch_size, game.params.n_men), dtype=torch.float, device=self.game.params.device)




    # input: B x * x ... x *
    # dim: 0 < scalar
    # index: B x M
    def batched_index_select(self, input, dim, index):
        views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)




    def vision (self, n_steps=10):
        idx_1 = (self.orientations_1 * self.game.params.n_men + torch.arange(self.game.params.n_men, device=self.game.params.device)).to(dtype=torch.long)
        men_1 = self.men_1[:, :, 0, :, :]
        # b , n_men , H , W

        # print('obstructions 1', self.obstructions_mask_1)

        men_all_view_1 = men_1.repeat((1, 4, 1, 1))
        # b , n_men x 4 , H , W

        men_all_view_1 = F.pad(men_all_view_1, (self.game.params.vision_padding+1,)*4)
        view = F.conv2d(men_all_view_1, self.game.vision_seed, stride=1, groups=self.game.params.n_men*4)

        view_sum = view.clone()
        for _ in range(20):
            view = F.pad(view, (1, 1, 1, 1,))
            view = F.conv2d(view, self.game.vision_filter, stride=1, groups=self.game.params.n_men*4)
            


            view[view > 1] = 1.
            floor = .07
            is_not_zero = (view != 0).clone()
            view[view < floor] = floor
            view *= is_not_zero

            # view *= self.obstructions_mask_1

            view_sum += view
        pad_ = self.game.params.vision_padding
        view_sum = view_sum[:, :, pad_:, pad_:]
        view_sum = view_sum[:, :, :-pad_, :-pad_]


        threshold = .2
        view_sum[view_sum < threshold] = 0 
        # view_sum += walls
        view_men_1 = self.batched_index_select(view_sum, dim=1, index=idx_1)
        return view_men_1 > 0


def plot_grid(data, rows, cols):
    # data = data[0, 0, :, :]
    fig, ax = plt.subplots()
    ax.imshow(data, cmap='hot')
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(0.5, rows, 1))
    ax.set_yticks(np.arange(0.5, cols, 1))
    plt.tick_params(axis='both', labelsize=0, length = 0)
    plt.show()
    # fig.set_size_inches((8.5, 11), forward=False)
    # plt.savefig(saveImageName + ".png", dpi=500)

if __name__ == '__main__' :

    game_params = GridGameParams(width=40,height=40,n_men=3, vision_padding=5,device=torch.device('cuda:0'))
    game = GridGame(game_params)
    states = States(game, batch_size=10**3)

    man = states.men_1[:, :, :, :]

    man[:, 0, 0, game.params.width//2, game.params.height*3//4] = 1.
    man[:, 0, 1, game.params.width//2 - 1, game.params.height*3//4] = 1.

    man[:, 1, 0, game.params.width//2, game.params.height//2] = 1.
    man[:, 1, 1, game.params.width//2 - 1, game.params.height//2] = 1.

    man[:, 2, 0, 0, 0] = 1.
    man[:, 2, 1, 0, 1] = 1.

    states.orientations_1[:, 0] = 3
    states.orientations_1[:, 1] = 3
    states.orientations_1[:, 2] = 2
    
    start = time.time()
    for _ in range(10**3):
        view_men_1 = states.vision()
        view_1 = torch.any(view_men_1, dim=1)


    done_generating = time.time()
    print('game generation time',(done_generating - start)/1)
    # print(view_1)

    plot_grid (view_1[0].cpu().to(torch.float), game.params.width, game.params.height)

