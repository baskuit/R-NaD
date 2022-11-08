import torch
import os
import matplotlib.pyplot as pyplot

"""
Rudimentary script for looking at the saved logs from a RNaD run
"""

directory_name = 'depth5-1'
dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_runs', directory_name, 'params')
params_dict = torch.load(dir)
for _, __ in params_dict.items():
    print(_, __)
dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_runs', directory_name, 'logs')  
logs_dict = torch.load(dir)
nash_conv = logs_dict['nash_conv']
nash_conv_target = logs_dict['nash_conv_target']
loss_value = logs_dict['loss_value']
loss_neurd = logs_dict['loss_neurd']
fig, ax = pyplot.subplots(4)
ax[0].plot(list(loss_value.keys()), list(loss_value.values()))
ax[1].plot(list(loss_neurd.keys()), list(loss_neurd.values()))
ax[2].plot(list(nash_conv.keys()), list(nash_conv.values()))
ax[3].plot(list(nash_conv_target.keys()), list(nash_conv_target.values()))
ax[0].set_ylim(0, 2)
ax[0].set_title('value loss')
ax[1].set_ylim(-2, 2)
ax[1].set_title('neurd loss')
ax[2].set_ylim(0, 2)
ax[2].set_title('NashConv')
ax[3].set_ylim(0, 2)
ax[3].set_title('NashConv (target)')
pyplot.show()