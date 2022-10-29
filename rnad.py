import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import os
import time

import game
import net

class RNaD () :

    def __init__ (self,
        # R-NaD parameters, see paper
        tree_id : str,
        eta=.2,
        delta_m_0 = (100, 165, 200, 250),
        delta_m_1 = (10**4, 10**5, 35000, 0),
        lr=5*10**-5,
        beta=2,
        grad_clip=10**4,
        b1_adam=0,
        b2_adam=.999,
        epsilon_adam=10**-8,
        gamma_averaging=.001,
        roh_bar=1,
        c_bar=1,
        batch_size=768,
        epsilon_threshold=.03,
        n_discrete=32,
        # checkpoint. dont pass m,n since we either start from scratch or used saved
        directory_name=None,
        checkpoint_mod=10**3,

    ):
        self.tree_id = tree_id,

        self.eta = eta
        self.delta_m_0 = delta_m_0
        self.delta_m_1 = delta_m_1
        self.lr = lr
        self.beta = beta
        self.grad_clip = grad_clip
        self.b1_adam = b1_adam
        self.b2_adam = b2_adam
        self.epsilon_adam = epsilon_adam
        self.gamma_averaging = gamma_averaging
        self.roh_bar = roh_bar
        self.c_bar = c_bar
        self.batch_size = batch_size #one Episodes per step TODO
        self.epsilon_threshold = epsilon_threshold
        self.n_discrete = n_discrete

        saved_runs_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_runs')    
        if not os.path.exists(saved_runs_dir):
            os.mkdir(saved_runs_dir)
        if directory_name is None:
            # poverty name
            directory_name = str(int(time.perf_counter()))
        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_runs', directory_name)       
        # Create folder on intialiazation of RNaD
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        self.checkpoint_mod = checkpoint_mod

        self.saved_keys = [key for key in self.__dict__.keys()]

        self.tree = game.Tree()
        self.tree.load(tree_id)
        logging.debug("Tree {} loaded, index tensor has shape {}".format(tree_id, self.tree.index.shape))
        self.tree.to(torch.device('cuda:0'))

        self.m = 0
        self.n = 0

        self.net_params = {'size':self.tree.max_actions,'channels':2**5,'depth':2,'device':torch.device('cuda')}
        self.net = None
        self.net_target = None
        self.net_reg = None
        self.net_reg_ = None

        # self.optimizer = torch.optim.Adam()
        # state of optimizer persists throughout run, must be saved

    def _new_net (self) -> nn.Module:
        # on cuda by default since large trees will have to be stored on cpu so may as well used shared memory with workers if it comes to that
        return net.ConvNet(**self.net_params)
         
    def initialize (self):

        # look through dir to find most recent saved networks, otherwise initialize:

        folder_names_as_int = [int(os.path.relpath(f.path, self.directory)) for f in os.scandir(self.directory) if f.is_dir()]
        
        if not folder_names_as_int:
            
            params_dict = {key : self.__dict__[key] for key in self.saved_keys}
            torch.save(params_dict,  os.path.join(self.directory, 'params'))

            os.mkdir(os.path.join(self.directory, '0'))
            self.m = 0
            self.n = 0
            self.net = self._new_net()
            self.net_target = self.net
            self.net_reg = self.net
            self.net_reg_ = self.net
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=[self.b1_adam, self.b2_adam], eps=self.epsilon_adam)
            saved_dict = {
                'net_params':self.net_params,
                'net':self.net.state_dict(),
                'net_target':self.net.state_dict(),
                'net_reg':self.net.state_dict(),
                'net_reg_':self.net.state_dict(),
                'optimizer':self.optimizer.state_dict(),

            }
            torch.save(saved_dict, os.path.join(self.directory, '0', '0'))

        else:

            params_dict = torch.load(os.path.join(self.directory, 'params'))
            assert(self.tree_id == params_dict['tree_id'])
            for key, value in params_dict.items():
                self.__dict__[key] = value

            self.m = max(folder_names_as_int)
            logging.debug('initializing net, found largest m: {}'.format(self.m))
            last_outer_step_dir = os.path.join(self.directory, str(self.m))
            checkpoints = [int(os.path.relpath(f.path, last_outer_step_dir)) for f in os.scandir(last_outer_step_dir) if not f.is_dir()]
            self.n = max(checkpoints)
            saved_dict = torch.load(os.path.join(self.directory, str(self.m), str(self.n)))
            self.net_params = saved_dict['net_params']
            self.net = self._new_net()
            self.net.load_state_dict(saved_dict['net'])
            self.net_target = self._new_net()
            self.net_target.load_state_dict(saved_dict['net_target'])
            self.net_reg = self._new_net()
            self.net_reg.load_state_dict(saved_dict['net_reg'])
            self.net_reg_ = self._new_net()
            self.net_reg_.load_state_dict(saved_dict['net_reg_'])
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=[self.b1_adam, self.b2_adam], eps=self.epsilon_adam)
            self.optimizer.load_state_dict(saved_dict['optimizer'])

    def _get_delta_m (self) :

        idx = min([_ for _, bound in enumerate(self.delta_m_0) if bound > self.m])
            
    def resume (self) :
        pass


    def get_last_saved (self):
        pass

    def run (self):

        net_checkpoint_data = self.get_last_saved()


if __name__ == '__main__' :

    

    test_run = RNaD(tree_id='1667036398', directory_name='test')
    test_run.initialize()

    test_run.resume()