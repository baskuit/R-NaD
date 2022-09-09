import Game
import Metric
import Net
import Inner

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Implementation of Regularized Nash Dynamics to approximate equilibrium for collections of zero-sum matrices.
# The update rule is single-action NeuRD, although PG and all-action updates may also be implemented
# The learning rate and eta parameters are on exponential decay schedule
class RNaD ():

    def __init__ (self, params={}):

        # Game
        if 'size' not in params:
            params['size'] = 3
        if 'values' not in params:
            params['values'] = (-1, 0, 1)
        if 'game' not in params:
            params['game'] = Game.Game(params['size'], params['values'])
            params['game'].solve_filtered(unique=True, interior=True)

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
        if 'relu_leak' not in params: #unused currently
            params['relu_leak'] = .05
        if 'net' not in params:
            if params['net_type'] == 'FCNet' :
                params['net'] = Net.FCNet(size=params['size'], width=params['width'], depth=params['depth'], dropout=params['dropout'])
                params['net_fixed'] = Net.FCNet(size=params['size'], width=params['width'], depth=params['depth'], dropout=params['dropout'])

            if params['net_type'] == 'ConvNet' :
                params['net'] = Net.ConvNet(size=params['size'], channels=params['channels'], depth=params['depth'], batch_norm=params['batch_norm'])
                params['net_fixed'] = Net.ConvNet(size=params['size'], channels=params['channels'], depth=params['depth'], batch_norm=params['batch_norm'])
            params['net_fixed'].load_state_dict(params['net'].state_dict())
        # if 'nex_fixed' not in params:
        #     params['net_fixed'] = copy.deepcopy(params['net'])

        # Learning
        if 'update' not in params:
            params['update'] = 'neurd'
        if 'logit_threshold' not in params:
            params['logit_threshold'] = 2
        if 'grad_clip' not in params:
            params['grad_clip'] = 1000
        if 'policy_batch_size' not in params:
            params['policy_batch_size'] = 2**6
        if 'validation_batch' not in params:
            params['validation_batch'] = params['game'].matrices
        params['validation_policy_batch_size'] = params['validation_batch'].shape[0]

        # RNaD
        if 'outer_steps' not in params:
            params['outer_steps'] = 2**7
        if 'inner_steps' not in params:
            params['inner_steps'] = 2**16
        if 'interval' not in params:
            params['interval'] = params['inner_steps'] // 3
        if 'eta_start' not in params:
            params['eta_start'] = 1.0
        if 'eta_end' not in params:
            params['eta_end'] = 0.1
        if 'lr_start' not in params:
            params['lr_start'] = .004
        if 'lr_end' not in params:
            params['lr_end'] = 0.0008
        if 'value_loss_weight' not in params:
            params['value_loss_weight'] = 1 
        if 'entropy_loss_weight' not in params:
            params['entropy_loss_weight'] = 1 

        self.outer_step_data = {}

        self.params = params
        self.terminated = False

    def run (self):
        if (self.terminated):
            raise Exception('run() called on terminated RNaD object')

        print("Begining RNaD run, main params:")
        print("outer_steps: {}, inner_steps: {}".format(params['outer_steps'], params['inner_steps']))
        print("eta_start: {}, eta_end: {}".format(params['eta_start'], params['eta_end']))
        print("lr_start: {}, lr_end: {}".format(params['lr_start'], params['lr_end']))
        print()

        log_eta_decay = (math.log(self.params['eta_end']) - math.log(self.params['eta_start'])) / self.params['outer_steps']
        log_lr_decay = (math.log(self.params['lr_end']) - math.log(self.params['lr_start'])) / self.params['outer_steps']

        for outer_step in range(self.params['outer_steps']):

            inner_loop_params = {
                'size':self.params['size'],
                'values':self.params['values'],
                'game':self.params['game'],
                'net':self.params['net'],
                'net_fixed':self.params['net_fixed'],
                'update':self.params['update'],
                'lr':math.exp(outer_step*log_lr_decay) * self.params['lr_start'],
                'value_loss_weight':self.params['value_loss_weight'],
                'entropy_loss_weight':self.params['entropy_loss_weight'],
                'log_lr_decay':log_lr_decay,
                'eta':math.exp(outer_step*log_eta_decay) *self.params['eta_start'],
                'log_eta_decay':log_eta_decay, # decay in the inner loop for smoothness
                'logit_threshold':self.params['logit_threshold'],
                'grad_clip':self.params['grad_clip'],
                'policy_batch_size':self.params['policy_batch_size'],
                'value_batch_size':self.params['value_batch_size'],
                'total_steps':self.params['inner_steps'],
                'validation_batch':self.params['validation_batch'],
                'interval':self.params['interval'],
            }

            print()
            print("OUTER STEP: {}".format(outer_step))
            print("eta: {}, lr: {}".format(inner_loop_params['eta'], inner_loop_params['lr']))

            inner_loop = Inner.Inner(inner_loop_params)
            inner_loop.run()
            checkpoints = inner_loop.checkpoints

            if self.params['net_type'] == 'FCNet' :
                self.params['net_fixed'] = Net.FCNet(size=self.params['size'], width=self.params['width'], depth=self.params['depth'], dropout=self.params['dropout'])
            if self.params['net_type'] == 'ConvNet' :
                self.params['net_fixed'] = Net.ConvNet(size=self.params['size'], channels=self.params['channels'], depth=self.params['depth'], batch_norm=self.params['batch_norm'])
            self.params['net_fixed'].load_state_dict(self.params['net'].state_dict())

        self.terminated = True


if __name__ == "__main__":

    params = {
        'net_type' : 'FCNet', 
        'depth' : 2,
        'width' : 128,
        'channels' : 9,
        'dropout' : .5,
        'batch_norm' : True,
        'update' : 'neurd_all_actions',
        'logit_threshold' : 2,
        'grad_clip':20,
        'policy_batch_size' : 2**8,
        'value_batch_size' : 2**5,
        'outer_steps' : 2**12,
        'inner_steps' : 2**16,
        'interval' : 2**16 // 2,
        'eta_start' : 5,
        'eta_end' : 5,
        'lr_start' : .001,
        'lr_end' : 0.0001,
        'value_loss_weight':2,
        'entropy_loss_weight':0,
    }

    x = RNaD(params)  
    x.run()


