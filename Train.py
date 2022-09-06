import Net
import Data
import Metric

import torch
import random
import math
import matplotlib.pyplot as plt



# inner loop of RNaD
def train (params={}) :

    if 'size' not in params:
        params['size'] = 3
    if 'values' not in params:
        params['values'] = (-1, 0, 1)
    if 'memoization' not in params:
        params['memoization'] = Data.Memoized(params['size'], params['values'])
        params['memoization'].solve_filtered(unique=True, interior=True)

    if 'depth' not in params:
        params['depth'] = 2
    if 'width' not in params:
        params['width'] = 9
    if 'dropout' not in params:
        params['dropout'] = .5
    if 'update' not in params: #unused
        params['update'] = 'neurd'


    if 'lr' not in params:
        params['lr'] = .01
    if 'log_lr_decay' not in params:
        params['log_lr_decay'] = 0

    if 'eta' not in params:
        params['eta'] = 0 #important as non-zero will try to call forward on default net_fixed
    if 'log_eta_decay' not in params:
        params['log_eta_decay'] = 1

    if 'net' not in params:
        params['net'] = Net.FCNet(size=params['size'], width=params['width'], depth=params['depth'], dropout=params['dropout'])
    net = params['net']
    if 'net_fixed' not in params:
        params['net_fixed'] = None

    if 'batch_size' not in params:
        params['batch_size'] = 2**6
    if 'total_steps' not in params:
        params['total_steps'] = 2**10
    if 'interval' not in params:
        params['interval'] = params['total_steps'] // 10
    if params['validation_batch'] is None:
        params['validation_batch'] = Data.discrete_batch(params['size'], params['validation_batch_size'])
    validation_batch =  params['validation_batch']
    params['validation_batch_size'] = validation_batch.shape[0]
    if 'example' not in params:
        params['example'] = Data.normal_batch(params['size'], 1)

    optimizer = torch.optim.SGD(net.parameters(), lr=params['lr'])
    def lr_lambda(step):
        return params['lr'] * math.exp(params['log_lr_decay'] * step / (params['total_steps'] + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    checkpoints = {}

    for step in range(params['total_steps']):

        input_batch, strategies0, strategies1 = params['memoization'].generate_input_batch(params['batch_size'])
        Net.step_neurd(net, optimizer, scheduler, input_batch, params['eta'], params['net_fixed'])

        if step % params['interval'] == 0:
            net.eval()
            checkpoint_data = {}

            _, validation_policy_batch, __ = net(Data.flip_cat(validation_batch))

            s0 = Data.first_half(validation_policy_batch)
            s1 = Data.second_half(validation_policy_batch)
            
            checkpoint_data['expl'] = torch.mean(Metric.expl(validation_batch, s0, s1)).item()
            checkpoint_data['policy'] = validation_policy_batch

            checkpoints[step] = checkpoint_data
            net.train()
    
    return checkpoints

def train_cel (params={}) :

    if 'size' not in params:
        params['size'] = 3
    if 'values' not in params:
        params['values'] = (-1, 0, 1)
    if 'memoization' not in params:
        params['memoization'] = Data.Memoized(params['size'], params['values'])
        params['memoization'].solve_filtered(unique=True, interior=True)

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

    if 'lr' not in params:
        params['lr'] = .01
    if 'log_lr_decay' not in params:
        params['log_lr_decay'] = 1

    if 'eta' not in params:
        params['eta'] = 0 #important as non-zero will try to call forward on default net_fixed
    if 'log_eta_decay' not in params:
        params['log_eta_decay'] = 1

    if 'net' not in params:
        params['net'] = Net.FCNet(size=params['size'], width=params['width'], depth=params['depth'], dropout=params['dropout'])
    net = params['net']
    if 'net_fixed' not in params:
        params['net_fixed'] = None

    if 'batch_size' not in params:
        params['batch_size'] = 2**6
    if 'total_steps' not in params:
        params['total_steps'] = 2**10
    if 'interval' not in params:
        params['interval'] = params['total_steps'] // 10
    if params['validation_batch'] is None:
        params['validation_batch'] = Data.discrete_batch(params['size'], params['validation_batch_size'])
    validation_batch =  params['validation_batch']
    params['validation_batch_size'] = validation_batch.shape[0]

    optimizer = torch.optim.SGD(net.parameters(), lr=params['lr'])
    def lr_lambda(step):
        return params['lr'] * math.exp(params['log_lr_decay'] * step / (params['total_steps'] + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    checkpoints = {}

    for step in range(params['total_steps']):

        input_batch, strategies0, strategies1 = params['memoization'].generate_input_batch(params['batch_size'])
        Net.step_cel(net, optimizer, scheduler, input_batch, strategies0, strategies1)

        if step % params['interval'] == 0:
            net.eval()
            checkpoint_data = {}

            _, validation_policy_batch, __ = net(Data.flip_cat(validation_batch))

            s0 = Data.first_half(validation_policy_batch)
            s1 = Data.second_half(validation_policy_batch)
            
            checkpoint_data['expl'] = torch.mean(Metric.expl(validation_batch, s0, s1)).item()
            checkpoint_data['policy'] = validation_policy_batch

            checkpoints[step] = checkpoint_data
            net.train()
    
    return checkpoints


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
    M = Data.Memoized(3)
    M.solve_filtered()
    validation_batch = M.matrices
    validation_batch_size = validation_batch.shape[0]    
    generator = hyperparameter_generator(total_frames)

    print('Total Frames: {}'.format(total_frames))
    print('Validation Batch Size: {}'.format(validation_batch_size))
    print('________________')

    data = []

    for _ in range(params['tries']):
        train_params = generator.__next__()
        [print(_, __) for _, __ in train_params.items()]
        train_params['validation_batch'] = validation_batch
        train_params['memoization'] = M

        try:
            checkpoints = train_neurd(train_params)
        except:
            print('error error error')
            checkpoints = None
        
        if checkpoints is not None:
            keys = list(checkpoints.keys())
            keys.sort()

            expl = [checkpoints[_]['expl'] for _ in keys]
            print('min_expl: {}'.format(min(expl)))
            print()

if __name__ == '__main__' :

    random_search()

