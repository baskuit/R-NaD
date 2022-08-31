import Data
import Metric
import Net
import Train

import torch
import torch.nn as nn
import torch.nn.functional as F

class RNaD ():

    def __init__ (self, params):
        if 'outer_steps' not in params:
            params['outer_steps'] = 2**8
        if 'inner_steps' not in params:
            params['inner_steps'] = 2**8
        if 'eta' not in params:
            params['eta'] = 1
        if 'eta_decay' not in params:
            params['eta_decay'] = .99
            # mentioned as good practice in section 7.1 

        # if 'r' not in params:
        #     params['r'] = lambda rewards, pi, mu : rewards + torch.log(pi) - torch.log(mu)
        #     # r^{i}_{pi}(h, a) = r^{i}(h, a) + \eta (1 - 2\delta_{h,i}) \log{frac{\pi(a)}{\mu{a}}}
        #     # r(rewards, pi, mu):
        #     # gets policy info from self

        self.nets_current = None
        self.nets_past = None
        # net_past gives mu, net_current gives pi

        self.outer_step_data = {}

        self.params = params

    def run (self):

        for outer_step in range(self.params['outer_steps']): 
            self.run_outer_step ()

    def run_outer_step (self):
        pass





if __name__ == "__main__":
    
    