import torch
import os
import matplotlib.pyplot as pyplot

"""
Rudimentary script for looking at the saved logs from a RNaD run
"""
d = 'test_fixed_5'
e = 'test_broken_5'

def sort_dict (dict) -> list[tuple]:

    x = [(key, value) for key, value in dict.items()]
    x = sorted(x, key=lambda _:_[0])
    y = [_[0] for _ in x]
    z = [_[1] for _ in x]

    return y, z

def load_stuff(directory_name):
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_runs', directory_name, 'params')
    params_dict = torch.load(dir)
    # for _, __ in params_dict.items():
    #     print(_, __)
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_runs', directory_name, 'logs')  
    logs_dict = torch.load(dir)
    return params_dict, logs_dict

def multiple_on_one_graph(directory_name_list, colors=['r', 'b', 'g']):
    for idx ,directory_name in enumerate(directory_name_list):
        params_dict, logs_dict = load_stuff(directory_name)
        nash_conv_target = logs_dict['nash_conv_target']
        print(nash_conv_target)
        for _, __ in nash_conv_target.items():
            print(_, __)
        y, z = sort_dict(nash_conv_target)
        pyplot.plot(y, z, color=colors[idx])
    pyplot.ylim(0, 2)
    pyplot.show()


multiple_on_one_graph([d, e])