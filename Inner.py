import Net
import Data
import Metric

import torch
import random
import math
import matplotlib.pyplot as plt


class Inner () :

    def __init__ (self, params={}) :

        self.params = params
        self.checkpoints = {}

        # Game
        if 'size' not in params:
            params['size'] = 3
        if 'values' not in params:
            params['values'] = (-1, 0, 1)
        if 'memoization' not in params:
            params['memoization'] = Data.Game(params['size'], params['values'])
            params['memoization'].solve_filtered(unique=True, interior=True)

        # Net architecture
        if 'net_type' not in params: #only used if net is not passed
            params['net_type'] = 'FCNet'      
        if 'depth' not in params:
            params['depth'] = 2
        if 'width' not in params:
            params['width'] = 81
        if 'channels' not in params:
            params['channels'] = 9
        if 'dropout' not in params:
            params['dropout'] = .5
        if 'batch_norm' not in params:
            params['batch_norm'] = True
        if 'relu_leak' not in params:
            params['relu_leak'] = .05
        if 'net_fixed' not in params:
            params['net_fixed'] = None
        if 'net' not in params:
            if params['net_type'] == 'FCNet' :
                params['net'] = Net.FCNet(size=params['size'], width=params['width'], depth=params['depth'], dropout=params['dropout'])
            if params['net_type'] == 'ConvNet' :
                params['net'] = Net.ConvNet(size=params['size'], channels=params['channels'], depth=params['depth'], batch_norm=params['batch_norm'])

        # Learning
        if 'update' not in params:
            params['update'] = 'neurd'
        if 'lr' not in params:
            params['lr'] = .01
        if 'log_lr_decay' not in params:
            params['log_lr_decay'] = 0
        if 'eta' not in params:
            params['eta'] = 0 #important as non-zero will try to call forward on default net_fixed
        if 'log_eta_decay' not in params:
            params['log_eta_decay'] = 1
        if 'batch_size' not in params:
            params['batch_size'] = 2**6
        if 'total_steps' not in params:
            params['total_steps'] = 2**10
        if 'interval' not in params:
            params['interval'] = params['total_steps'] // 10
        if params['validation_batch'] is None:
            params['validation_batch'] = Data.discrete_batch(params['size'], params['validation_batch_size'])
        params['validation_batch_size'] = params['validation_batch'].shape[0]
        if 'optimizer' not in params:
            params['optimizer'] = torch.optim.SGD(params['net'].parameters(), lr=params['lr'])
        if 'scheduler' not in params:
            params['scheduler'] = torch.optim.lr_scheduler.LambdaLR(params['optimizer'], self.lr_lambda)

    def lr_lambda(self, step):
        return self.params['lr'] * math.exp(self.params['log_lr_decay'] * step / (self.params['total_steps'] + 1))

    def run (self) :

        for step in range(self.params['total_steps']):

            input_batch, strategies0, strategies1 = self.params['memoization'].generate_input_batch(self.params['batch_size'])

            if self.params['update'] == 'neurd' :
                Net.step_neurd(self.params['net'], self.params['optimizer'], self.params['scheduler'], input_batch, self.params['eta'], self.params['net_fixed'])
            elif self.params['update'] == 'cel':
                Net.step_cel(self.params['net'], self.params['optimizer'], self.params['scheduler'], input_batch, strategies0, strategies1)
            else:
                pass

            if step % self.params['interval'] == 0:
                self.params['net'].eval()
                checkpoint_data = {}

                _, validation_policy_batch, __ = self.params['net'](Data.flip_cat(self.params['validation_batch']))

                s0 = Data.first_half(validation_policy_batch)
                s1 = Data.second_half(validation_policy_batch)
                
                checkpoint_data['expl'] = torch.mean(Metric.expl(self.params['validation_batch'], s0, s1)).item()
                checkpoint_data['policy'] = validation_policy_batch

                self.checkpoints[step] = checkpoint_data
                self.params['net'].train()



def hyperparameter_generator (total_frames) :
    for _ in range(2**20):
        params = {
        'size' : 3,
        'lr' : .1,
        'log_lr_decay' : 0,
        'batch_size' : int(math.exp(2*random.random()+2)),
        'depth' : random.randint(1, 4),
        'width' : int(math.exp(2+3*random.random())),
        }
        params['total_steps'] = 2**16
        params['interval'] = params['total_steps'] // 100
        yield params

def random_search (params={}) :

    if 'tries' not in params:
        params['tries'] = 2**5
    total_frames = 2**20
    M = Data.Game(3)
    M.solve_filtered()
    validation_batch = M.matrices
    validation_batch_size = validation_batch.shape[0]    
    generator = hyperparameter_generator(total_frames)

    print('Total Frames: {}'.format(total_frames))
    print('Validation Batch Size: {}'.format(validation_batch_size))
    print('________________')

    data = []

    for _ in range(params['tries']):
        loop_params = generator.__next__()
        [print(_, __) for _, __ in loop_params.items()]

        loop_params['validation_batch'] = validation_batch
        loop_params['memoization'] = M

        try:
            loop = Inner(loop_params)
            loop.run()
            checkpoints = loop.checkpoints
        except Exception as e: 
            print(e)
            checkpoints = None
        
        if checkpoints is not None:
            keys = list(checkpoints.keys())
            keys.sort()

            expl = [checkpoints[_]['expl'] for _ in keys]
            print('min_expl: {}'.format(min(expl)))
            print()

if __name__ == '__main__' :

    random_search()

