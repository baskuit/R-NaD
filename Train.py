import Net
import Data
import Metric

import torch
import random
import math

# inner loop of RNaD
def train (params={}) :
    result = {}

    if 'size' not in params:
        params['size'] = 3
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
    if 'eta' not in params:
        params['eta'] = 0 #important as non-zero will try to call forward on default net_fixed
    if 'log_eta_decay' not in params:
        params['log_eta_decay'] = 1
    if 'log_lr_decay' not in params:
        params['log_lr_decay'] = 1

    if 'batch_size' not in params:
        params['batch_size'] = 2**6
    if 'total_steps' not in params:
        params['total_steps'] = 2**10
    if 'interval' not in params:
        params['interval'] = params['total_steps'] // 10
    if 'validation_batch_size' not in params:
        params['validation_batch_size'] = int(2**10)
    if params['validation_batch'] is None:
        params['validation_batch'] = Data.normal_batch(params['size'], params['validation_batch_size'])
    validation_batch =  params['validation_batch']

    if 'net' not in params:
        params['net'] = Net.FCNet(params['size'], params['width'], dropout=params['dropout'])
    net = params['net']
    if 'net_fixed' not in params:
        params['net_fixed'] = None

    optimizer = torch.optim.SGD(net.parameters(), lr=params['lr'])
    def lr_lambda(step):
        return params['lr'] * math.exp(params['log_lr_decay'] * step / (params['total_steps'] + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    checkpoints = {}

    for step in range(params['total_steps']):
        input_batch = Data.normal_batch(params['size'], params['batch_size'])
        eta = params['eta'] * math.exp(params['log_eta_decay'] * step / (params['total_steps'] + 1))
        Net.step_neurd(net, optimizer, scheduler, input_batch, eta, params['net_fixed'])

        if step % params['interval'] == 0:
            logits, policy, value = net.forward(Data.flip_cat(validation_batch))
            expl = torch.mean(Metric.expl(validation_batch, policy[:params['validation_batch_size']], policy[params['validation_batch_size']:]))
            data = {'policy': policy.clone().detach()}
            checkpoints[step] = data
            
            mean_policy = sum([_['policy'] for _ in checkpoints.values()]) / len(checkpoints)
            mean_expl = torch.mean(Metric.expl(validation_batch, mean_policy[:params['validation_batch_size']], mean_policy[params['validation_batch_size']:]))
            
            data['mean_policy'] = mean_policy
            data['mean_expl'] = mean_expl
            data['expl'] = expl
            data['lr'] = scheduler.get_last_lr()

    result['net_dict'] = net.state_dict().copy()
    result['checkpoints'] = checkpoints
    last_data = checkpoints[max(checkpoints.keys())]

    result['policy'] = last_data['policy']
    result['mean_policy'] = last_data['mean_policy']
    result['expl'] = last_data['expl'].item()
    result['mean_expl'] = last_data['mean_expl'].item()
    
    result['validation_batch'] = validation_batch # in case not passed?

    return result

def hyperparameter_generator (total_frames) :
    while True:
        params = {
        'size' : 3,
        'lr' : math.exp(-6*random.random()-2),
        'log_lr_decay' : -math.log(10),
        'batch_size' : int(math.exp(2*random.random()+2)),
        'depth' : random.randint(1, 4),
        'width' : int(math.exp(2+3*random.random())),
        }
        params['total_steps'] = total_frames // params['batch_size']
        params['interval'] = params['total_steps'] // 100
        yield params

def train_cel (params={}) :
    result = {}

    if 'size' not in params:
        params['size'] = 3
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
    if 'eta' not in params:
        params['eta'] = 0 #important as non-zero will try to call forward on default net_fixed
    if 'log_eta_decay' not in params:
        params['log_eta_decay'] = 1
    if 'log_lr_decay' not in params:
        params['log_lr_decay'] = 1

    if 'batch_size' not in params:
        params['batch_size'] = 2**6
    if 'total_steps' not in params:
        params['total_steps'] = 2**10
    if 'interval' not in params:
        params['interval'] = params['total_steps'] // 10
    if params['validation_batch'] is None:
        params['validation_batch'] = Data.discrete_batch(params['size'], params['validation_batch_size'])
    validation_batch =  params['validation_batch']
    if 'example' not in params:
        params['example'] = Data.normal_batch(params['size'], 1)
    params['validation_batch_size'] = params['validation_batch'].shape[0]

    if 'net' not in params:
        params['net'] = Net.FCNet(params['size'], params['width'], dropout=params['dropout'])
    net = params['net']
    if 'net_fixed' not in params:
        params['net_fixed'] = None

    optimizer = torch.optim.SGD(net.parameters(), lr=params['lr'])
    def lr_lambda(step):
        return params['lr'] * math.exp(params['log_lr_decay'] * step / (params['total_steps'] + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    checkpoints = {}

    for step in range(params['total_steps']):
        input_batch = Data.discrete_batch(params['size'], params['batch_size'])
        Net.step_cel(net, optimizer, scheduler, input_batch)
        if step % params['interval'] == 0:
            logits, policy, value = net.forward(Data.flip_cat(validation_batch))
            expl = torch.mean(Metric.expl(validation_batch, policy[:params['validation_batch_size']], policy[params['validation_batch_size']:]))

            data = {'policy': policy.clone().detach()}
            checkpoints[step] = data

            
            mean_policy = sum([_['policy'] for _ in checkpoints.values()]) / len(checkpoints)
            mean_expl = torch.mean(Metric.expl(validation_batch, mean_policy[:params['validation_batch_size']], mean_policy[params['validation_batch_size']:]))
            
            data['mean_policy'] = mean_policy
            data['mean_expl'] = mean_expl
            data['expl'] = expl
            data['lr'] = scheduler.get_last_lr()

    # result['net_dict'] = net.state_dict().copy()
    # result['checkpoints'] = checkpoints
    last_data = checkpoints[max(checkpoints.keys())]

    result['policy'] = last_data['policy']
    result['expl'] = last_data['expl'].item()
    result['mean_expl'] = last_data['mean_expl'].item()
    result['min_expl'] = min([_['expl'].item() for _ in checkpoints.values()])
    [print(_) for __, _ in result.items()]
    result['validation_batch'] = validation_batch # in case not passed?

    # example = params['example']
    # print('expl')
    # print(result['expl'])
    # print('min_expl')
    # print(result['min_expl'])
    # print(example)
    # _, policy_batch, __ = net.forward(Data.flip_cat(example))
    # print(Data.solve(example))
    # print(policy_batch)


    return result

if __name__ == '__main__' :
    total_frames = 2**18
    validation_batch_size = 2**8
    validation_batch = Data.discrete_batch(3, validation_batch_size)
    generator = hyperparameter_generator(total_frames)

    print('Total Frames: {}'.format(total_frames))
    print('Validator Batch Size: {}'.format(validation_batch_size))

    data = []

    for _ in range(2**5):
        params = generator.__next__()
        params['validation_batch'] = validation_batch


        result = None
        try:
            result = train_cel(params)
        except Exception as e: 
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(e)
        