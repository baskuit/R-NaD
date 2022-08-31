import math
from pickletools import optimize
import Net
import Data
import Metric

import torch
import random
# import ray.tune

def train (params : dict) :
    result = {}

    if 'size' not in params:
        params['size'] = 3
    if 'depth' not in params:
        params['depth'] = 2
    if 'width' not in params:
        params['width'] = 9
    if 'lr' not in params:
        params['lr'] = .01
    if 'update' not in params:
        params['update'] = 'neurd'

    if 'batch_size' not in params:
        params['batch_size'] = 2**6
    if 'total_steps' not in params:
        params['total_steps'] = 2**10
    if 'interval' not in params:
        params['interval'] = 1
    if 'validation_batch' not in params:
        params['validation_batch'] = None
    if 'validation_batch_size' not in params:
        params['validation_batch_size'] = 2**10

    net = Net.FCNet(params['size'], params['width'], batch_norm=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=params['lr'])
    def lr_lambda(epoch):
        return params['lr']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if params['validation_batch'] is None:
        params['validation_batch_size'] = Data.normal_batch(params['size'], params['validation_batch_size'])
    validation_batch =  params['validation_batch']

    checkpoints = {}

    for step in range(params['total_steps']):
        input_batch = Data.normal_batch(params['size'], params['batch_size'])
        Net.step_neurd(net, optimizer, scheduler, input_batch)

        if step%params['interval'] == 0:
            logits, policy, value = net.forward(Data.flip_cat(validation_batch))
            data = {}
            data['policy'] = policy.detach()
            checkpoints[step] = data
            mean_policy = sum([_['policy'] for _ in checkpoints.values()]) / len(checkpoints)
            data['mean_policy'] = mean_policy
            expl = torch.mean(Metric.expl(validation_batch, mean_policy[:params['validation_batch_size']], mean_policy[params['validation_batch_size']:]))
            data['expl'] = expl
    result['net'] = net
    result['checkpoints'] = checkpoints
    result['score'] = checkpoints[max(checkpoints.keys())]['expl'].item()
    result['final_policy'] = checkpoints[max(checkpoints.keys())]['policy']
    result['validation_batch'] = validation_batch

    return result

def hyperparameter_generator (total_frames) :
    while True:
        params = {
        'size' : 3,
        'lr' : math.exp(-6*random.random()-2),
        'batch_size' : int(math.exp(2*random.random()+2)),
        'depth' : random.randint(1, 4),
        'width' : int(math.exp(2+3*random.random())),
        }
        params['total_steps'] = total_frames // params['batch_size']
        params['interval'] = params['total_steps'] // 100
        yield params


if __name__ == '__main__' :
    total_frames = 2**18
    validation_batch_size = 2**10
    validation_batch = Data.normal_batch(3, validation_batch_size)
    generator = hyperparameter_generator(total_frames)

    print('Total Frames: {}'.format(total_frames))
    print('Validator Batch Size: {}'.format(validation_batch_size))

    data = []

    for _ in range(2**6):
        params = generator.__next__()
        params['validation_batch'] = validation_batch


        result = None
        try:
            result = train(params)
        except Exception as e: 
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(e)
        
        if not result is None:
            del params['validation_batch']
            data.append((params, result['score']))

    data.sort(key=lambda _:_[1])
    for _ in data[:10]:
        print(_[0])
        print(_[1])
        print()

