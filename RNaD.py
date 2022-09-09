from email import policy
import Game
import Metric
import Net
import Inner

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

import matplotlib.pyplot

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
            params['validation_payoffs'] = params['game'].payoffs
        params['validation_batch_size'] = params['validation_batch'].shape[0]

        # RNaD
        if 'outer_steps' not in params:
            params['outer_steps'] = 2**7
        if 'inner_steps' not in params:
            params['inner_steps'] = 2**16
        if 'checkpoint_frequency' not in params:
            params['checkpoint_frequency'] = 100
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

        if 'verbose' not in params:
            params['verbose'] = True

        self.outer_step_data = []

        self.params = params
        self.terminated = False

    def run (self):
        if (self.terminated):
            raise Exception('run() called on terminated RNaD object')

        if self.params['verbose']:
            print("\n---- R-NaD ----")
            self.print_params()


        log_eta_decay = (math.log(self.params['eta_end']) - math.log(self.params['eta_start'])) / self.params['outer_steps']
        log_lr_decay = (math.log(self.params['lr_end']) - math.log(self.params['lr_start'])) / self.params['outer_steps']

        for outer_step in range(self.params['outer_steps']):

            if self.params['verbose']:
                print("\n    OUTER STEP: {}".format(outer_step))

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
                'validation_payoffs':self.params['validation_payoffs'],
                'interval':self.params['inner_steps']//self.params['checkpoint_frequency'],
                'verbose':self.params['verbose'],
                'tab':'        '
            }


            inner_loop = Inner.Inner(inner_loop_params)

            if self.params['verbose']:
                inner_loop.print_params()

            results = None
            try:
                inner_loop.run()
                results = inner_loop.results
            except Exception as e:
                print("INNER LOOP EXCEPTION")
                print(e)
                break

            self.outer_step_data.append(results)


            if self.params['verbose']:
                print('\n    min_expl: {}'.format(results['min_expl']))
                print('    value_loss: {}'.format(results['min_expl_asso_value_loss']))

            

            if self.params['net_type'] == 'FCNet' :
                self.params['net_fixed'] = Net.FCNet(size=self.params['size'], width=self.params['width'], depth=self.params['depth'], dropout=self.params['dropout'])
            if self.params['net_type'] == 'ConvNet' :
                self.params['net_fixed'] = Net.ConvNet(size=self.params['size'], channels=self.params['channels'], depth=self.params['depth'], batch_norm=self.params['batch_norm'])
            self.params['net_fixed'].load_state_dict(self.params['net'].state_dict())



        self.terminated = True

        if len(data) > 0:
            data = min(self.outer_step_data, key=lambda results: results['min_expl'])
            print(data['min_expl'], data['min_expl_asso_value_loss'])
        else:
            print('No outer steps, no data')

    def print_params (self) :
        for key, value in self.params.items() :
            if torch.is_tensor(value) :
                if torch.numel(value) < 27 :
                    print('{}:'.format(key))
                    print(value)
                else:
                    print('{}: {}'.format(key, value.shape))
            else:
                if key not in ('net', 'net_fixed', 'game'):
                    print('{}: {}'.format(key, value))


    def save_graph (self) :
        if not self.terminated:
            raise Exception('save_graph() called on non-terminated RNaD object')


def param_generator (total_steps=2**22) :


    for _ in range(2**20) :

        inner_steps = int(2**(10+9*random.random()))
        outer_steps = total_steps // inner_steps
        eta = 2**(-4+7*random.random())
        lr = 2**(-14+10*random.random())
        value_loss_weight = 2**(-3+6*random.random())
        entropy_loss_weight = 2**(-3+3*random.random())

        x = 7*random.random()
        y = x*random.random()
        policy_batch_size_exp = 3+x
        value_batch_size_exp =  3+y
        policy_batch_size = int(2**policy_batch_size_exp)
        value_batch_size = int(2**value_batch_size_exp)
        
        params = {
            'net_type' : 'FCNet', 
            'depth' : 2,
            'width' : 128,
            'channels' : 9,
            'dropout' : .5,
            'batch_norm' : True,
            'update' : 'neurd_all_actions',
            'logit_threshold' : 2+8*random.random(),
            'grad_clip':2**(5+5*random.random()),
            'policy_batch_size' : policy_batch_size,
            'value_batch_size' : value_batch_size,
            'outer_steps' : outer_steps,
            'inner_steps' : inner_steps,
            'checkpoint_frequency' : 100,
            'eta_start' : eta,
            'eta_end' : eta,
            'lr_start' : lr,
            'lr_end' : lr/10,
            'value_loss_weight':value_loss_weight,
            'entropy_loss_weight':entropy_loss_weight,

            'verbose':False
        }

        yield params









if __name__ == "__main__":

    total_steps = 2**22

    gen = param_generator(total_steps)

    game = Game.Game(3, [-1, 0, 1])
    game.solve_filtered(unique=True, interior=False)

    for params in gen:
        params['game'] = game
        x = RNaD(params)
        print()
        x.print_params()
        x.run()