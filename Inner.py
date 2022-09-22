import Net
import Game
import Metric

import torch
import random
import math
import matplotlib.pyplot as plt


class Inner () :

    def __init__ (self, params={}) :

        self.params = params
        self.results = {}

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
        if 'value_loss_weight' not in params:
            params['value_loss_weight'] = 1 
        if 'entropy_loss_weight' not in params:
            params['entropy_loss_weight'] = 1
        if 'log_lr_decay' not in params:
            params['log_lr_decay'] = 0
        if 'eta' not in params:
            params['eta'] = 0 #important as non-zero will try to call forward on default net_fixed
        if 'log_eta_decay' not in params:
            params['log_eta_decay'] = 0
        if 'logit_threshold' not in params:
            params['logit_threshold'] = 2
        if 'grad_clip' not in params:
            params['grad_clip'] = 1000
        if 'policy_batch_size' not in params:
            params['policy_batch_size'] = 2**8
        if 'value_batch_size' not in params:
            params['value_batch_size'] = 2**5
        if 'total_steps' not in params:
            params['total_steps'] = 2**10
        if 'interval' not in params:
            params['interval'] = params['total_steps'] // 10
        if 'validation_batch_size' not in params:
            params['validation_batch_size'] = None
        if 'validation_batch' not in params:
            if params['validation_batch_size'] is None:
                params['validation_batch'] = params['game'].matrices
                params['validation_payoffs'] = params['game'].payoffs
            else:
                pass # TODO
        if 'optimizer' not in params:
            params['optimizer'] = torch.optim.Adam(params['net'].parameters(), lr=params['lr'])
        if 'scheduler' not in params:
            params['scheduler'] = torch.optim.lr_scheduler.LambdaLR(params['optimizer'], self.lr_lambda)

        if 'verbose' not in params:
            params['verbose'] = True
        if 'tab' not in params:
            params['tab'] = ''

    def lr_lambda(self, step):
        return self.params['lr'] * math.exp(self.params['log_lr_decay'] * step / (self.params['total_steps'] + 1))

    def run (self) :

        checkpoints = {}

        for step in range(self.params['total_steps']):

            input_batch, strategies0, strategies1, payoffs = self.params['game'].generate_input_batch(self.params['policy_batch_size'])

            if self.params['update'] == 'neurd' :
                Net.step_neurd(
                    net=self.params['net'], 
                    optimizer=self.params['optimizer'], 
                    scheduler=self.params['scheduler'], 
                    input_batch=input_batch, 
                    eta=self.params['eta'], 
                    net_fixed=self.params['net_fixed'],
                    logit_threshold=self.params['logit_threshold'],
                    grad_clip=self.params['grad_clip'],
                    value_loss_weight=self.params['value_loss_weight'],
                    entropy_loss_weight=self.params['entropy_loss_weight'],
                )

            elif self.params['update'] == 'neurd_all_actions' :
                Net.step_neurd_all_actions(
                    net=self.params['net'], 
                    optimizer=self.params['optimizer'], 
                    scheduler=self.params['scheduler'], 
                    input_batch=input_batch, 
                    eta=self.params['eta'], 
                    net_fixed=self.params['net_fixed'],
                    logit_threshold=self.params['logit_threshold'],
                    grad_clip=self.params['grad_clip'],
                    value_loss_weight=self.params['value_loss_weight'],
                    entropy_loss_weight=self.params['entropy_loss_weight'],
                    value_batch_size=self.params['value_batch_size'],
                )

            elif self.params['update'] == 'cel':
                Net.step_cel(
                    net=self.params['net'], 
                    optimizer=self.params['optimizer'], 
                    scheduler=self.params['scheduler'], 
                    input_batch=input_batch,
                    strategies0=strategies0, 
                    strategies1=strategies1
                )

            if step % self.params['interval'] == 0 or step + 1 == self.params['total_steps']:

                self.params['net'].eval()
                checkpoint_data = {}

                validation_logits_batch, validation_policy_batch, validation_value_batch = self.params['net'](Game.flip_cat(self.params['validation_batch']))

                s0 = Game.first_half(validation_policy_batch)
                s1 = Game.second_half(validation_policy_batch)
                
                checkpoint_data['expl'] = torch.mean(Metric.expl(self.params['validation_batch'], s0, s1)).item()
                validation_payoffs_flip_cat = torch.cat((self.params['validation_payoffs'], -self.params['validation_payoffs']), dim=0)
                checkpoint_data['value_loss'] = torch.mean((validation_value_batch - validation_payoffs_flip_cat)**2).item()
                # checkpoint_data['policy'] = validation_policy_batch.clone().detach()

                checkpoints[step] = checkpoint_data
                self.params['net'].train()
                
                if self.params['verbose']:
                    print()
                    print(self.params['tab']+"checkpoint: {}".format(step))
                    print(self.params['tab']+"max abs logit: {}".format(torch.max(torch.abs(validation_logits_batch)).item()))
                    print(self.params['tab']+'expl: {}'.format(checkpoint_data['expl']))
                    print(self.params['tab']+'value loss: {}'.format(checkpoint_data['value_loss']))

        # inner loop finished

        checkpoint_keys = list(checkpoints.keys())
        # checkpoint_keys.sort()
        
        min_expl_key = min(checkpoint_keys, key=lambda key:checkpoints[key]['expl'])
        self.results['min_expl'] = checkpoints[min_expl_key]['expl']
        self.results['min_expl_asso_value_loss'] = checkpoints[min_expl_key]['value_loss']
        self.results['checkpoints'] = checkpoints



    def print_params (self) :
        for key, value in self.params.items() :
            if isinstance(value, int) or isinstance(value, str) :
                print(self.params['tab']+'{}: {}'.format(key, value))
            if torch.is_tensor(value) :
                if torch.numel(value) < 27 :
                    print(self.params['tab']+'{}:'.format(key))
                    print(self.params['tab']+value)
                else:
                    print(self.params['tab']+'{}: {}'.format(key, value.shape))



# def hyperparameter_generator (total_frames) :
#     for _ in range(2**20):
#         params = {
#         'size' : 3,
#         # 'lr' : .001,
#         # 'log_lr_decay' : 0,
#         # 'policy_batch_size' : int(math.exp(2*random.random()+2)),
#         # 'depth' : random.randint(1, 4),
#         # 'width' : int(math.exp(2+3*random.random())),
#         }
#         params['total_steps'] = 2**16
#         params['interval'] = params['total_steps'] // 100
#         yield params

# def random_search (params={}) :

#     if 'tries' not in params:
#         params['tries'] = 2**3
#     total_frames = 2**16
#     M = Data.Game(3)
#     M.solve_filtered()

#     net = Net.FCNet(size=3, width=81, depth=2, dropout=.5)

#     validation_batch = M.matrices
#     validation_policy_batch_size = validation_batch.shape[0]    
#     generator = hyperparameter_generator(total_frames)

#     print('Total Frames: {}'.format(total_frames))
#     print('Validation Batch Size: {}'.format(validation_policy_batch_size))
#     print('________________')

#     data = []

#     for _ in range(params['tries']):
#         loop_params = {}
#         # [print(_, __) for _, __ in loop_params.items()]
#         loop_params['total_steps'] = 2**16
#         loop_params['interval'] = loop_params['total_steps'] // 10
        
#         loop_params['validation_batch'] = validation_batch
#         loop_params['memoization'] = M

#         loop_params['net'] = net
#         loop_params['update'] = 'neurd_all_actions'
#         loop_params['net_fixed'] = net
#         loop_params['eta'] = math.exp(-_/10)
#         loop_params['lr'] = .001
#         loop_params['logit_threshold'] = 1000

#         loop = Inner(loop_params)
#         loop.run()
#         checkpoints = loop.checkpoints

#         # try:
#         #     loop = Inner(loop_params)
#         #     loop.run()
#         #     checkpoints = loop.checkpoints

#         # except Exception as e: 
#         #     print(e)
#         #     checkpoints = None
        
#         if checkpoints is not None:
#             keys = list(checkpoints.keys())
#             keys.sort()

#             expl = [checkpoints[_]['expl'] for _ in keys]
#             print('min_expl: {}'.format(min(expl)))
#             print()

if __name__ == '__main__' :

    # random_search()
    pass



