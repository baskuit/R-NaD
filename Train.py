from pickletools import optimize
import Net
import Data
import Metric

import torch
# import ray.tune

def train (params : dict) :
    if 'size' not in params:
        params['size'] = 3
    if 'depth' not in params:
        params['depth'] = 1
    if 'width' not in params:
        params['width'] = 27
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
    if 'validation_batch_size' not in params:
        params['validation_batch_size'] = 2**10

    net = Net.FCNet(params['size'], params['width'], batch_norm=False)
    optimizer = torch.optim.SGD(net.parameters(), lr=params['lr'])
    def lr_lambda(epoch):
        return params['lr']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    validation_batch = Data.normal_batch(params['size'], params['validation_batch_size'])

    policy_checkpoints = {}

    for step in range(params['total_steps']):
        input_batch = Data.normal_batch(params['size'], params['batch_size'])
        Net.step_neurd(net, optimizer, scheduler, input_batch)

        if step%params['interval'] == 0:
            logits, policy, value = net.forward(Data.flip_cat(validation_batch))
            policy_checkpoints[step] = policy

    total_policy = sum(policy_checkpoints.values())
    s = torch.sum(total_policy) / 2

    mean_policy = total_policy / s
    print(total_policy)
    print(s)
    print(mean_policy)
    expl = Metric.expl(validation_batch, mean_policy[:params['validation_batch_size']], mean_policy[params['validation_batch_size']:])
    print(expl)





if __name__ == '__main__' :
    train(
        {'validation_batch_size':1,
        'interval':2**9,
        'total_steps':2**10
        })