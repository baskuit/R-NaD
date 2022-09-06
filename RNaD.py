import Data
import Metric
import Net
import Train

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

# Implementation of single action Regularized Nash Dynamics to approximate equilibrium for zero-sum normally-distributed matices.
# The update rule is single-action NeuRD, although PG and all-action updates may also be implemented
# The learning rate and eta parameters are on exponential decay schedule
class RNaD ():

    def __init__ (self, params={}):
        if 'size' not in params:
            params['size'] = 3
        if 'width' not in params:
            params['width'] = 128
        if 'depth' not in params:
            params['depth'] = 1
        if 'dropout' not in params:
            params['dropout'] = .5

        if 'outer_steps' not in params:
            params['outer_steps'] = 2**9
        if 'inner_steps' not in params:
            params['inner_steps'] = 2**16
        if 'eta_start' not in params:
            params['eta_start'] = 1.0
        if 'eta_end' not in params:
            params['eta_end'] = 0.001
        if 'lr_start' not in params:
            params['lr_start'] = .005
        if 'lr_end' not in params:
            params['lr_end'] = 0.0001
            # mentioned as good practice in section 7.1  

        if 'batch_size' not in params:
            params['batch_size'] = 2**7
        if 'validation_batch' not in params:
            params['validation_batch'] = Data.normal_batch(params['size'], 2**10)

        self.net_current = None
        self.net_past = None
        # net_past gives mu, net_current gives pi

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

        self.net_current = Net.FCNet(   self.params['size'], 
                                        self.params['width'],
                                        self.params['depth'], 
                                        self.params['dropout'])
        self.net_past = self.net_current #mu = pi on initialization

        for outer_step in range(self.params['outer_steps']):

            inner_loop_params = {
                'size':self.params['size'],
                'total_steps':self.params['inner_steps'],
                'batch_size':self.params['batch_size'],
                'validation_batch':self.params['validation_batch'],
                'net':self.net_current,
                'net_fixed':self.net_past,
                'eta':math.exp(outer_step*log_eta_decay) *self.params['eta_start'],
                'lr':math.exp(outer_step*log_lr_decay) * self.params['lr_start'],
                'log_eta_decay' : log_eta_decay, # decay in the inner loop for smoothness
                'log_lr_decay' : log_lr_decay
            }

            print("outer_step: {}".format(outer_step))
            print("eta: {}, lr: {}".format(inner_loop_params['eta'], inner_loop_params['lr']))

            try:
                checkpoints = self.run_inner_loop (inner_loop_params)
            except Exception as e: 
                print('Train() Error causing outer loop abort. Run terminated')
                print(e)
                self.terminated = True
                return

        self.terminated = True

    def run_inner_loop (self, inner_loop_params):
        result = Train.train_neurd(inner_loop_params)
        return result

if __name__ == "__main__":

    params = {
        'size' : 3,
        'width' : 81,
        'depth' : 1,
        'dropout' : .5,
        'outer_steps' : 2**6,
        'inner_steps' : 2**10,
        'eta_start' : 5,
        'eta_end' : 0.1,
        'lr_start' : .1,
        'lr_end' : 0.02,
        'batch_size' : 2**5,
    }

    x = RNaD(params)  
    x.run()


