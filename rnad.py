import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import os
import time

import game
import net
import vtrace
import metric

class RNaD () :

    def __init__ (self,
        # R-NaD parameters, see paper
        tree_id : str,
        eta=.2,
        delta_m_0 = (100, 165, 200, 250),
        delta_m_1 = (100, 1_000, 3_000, 0),
        lr=5*10**-5,
        beta=2,
        grad_clip=10**4,
        b1_adam=0,
        b2_adam=.999,
        epsilon_adam=10**-8,
        gamma=.001, #averaging for target net
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
        self.gamma = gamma
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

        self.alpha_lambda = lambda n, delta_m: 1 if n > delta_m / 2 else n * 2 / delta_m

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
            self.net_target = self._new_net()
            self.net_target.load_state_dict(self.net.state_dict())
            self.net_reg = self._new_net()
            self.net_reg.load_state_dict(self.net.state_dict())
            self.net_reg_ = self._new_net()
            self.net_reg_.load_state_dict(self.net.state_dict())
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

    def _get_delta_m (self) -> tuple[bool, int]:
        bounding_indices = [_ for _, bound in enumerate(self.delta_m_0) if bound > self.m]
        if not bounding_indices:
            return False, 0

        idx = min(bounding_indices)
        return True, self.delta_m_1[idx]

            
    def resume (self) :
        print('resume start')
        
        may_resume, delta_m = self._get_delta_m()

        while may_resume:
            print('may resume, delta m')
            print(may_resume, delta_m)
            print('m', self.m, 'n', self.n)
            alpha = self.alpha_lambda(self.n, delta_m)
            print('alpha', alpha)

            while self.n < delta_m:
                if self.n % self.checkpoint_mod == 0 and self.n > 0:
                    saved_dict = {
                        'net_params':self.net_params,
                        'net':self.net.state_dict(),
                        'net_target':self.net.state_dict(),
                        'net_reg':self.net.state_dict(),
                        'net_reg_':self.net.state_dict(),
                        'optimizer':self.optimizer.state_dict(),
                    }
                    torch.save(saved_dict, os.path.join(self.directory, str(self.m), str(self.n)))

                self.n += 1

                episodes = game.Episodes(self.tree, self.batch_size)
                episodes.generate(self.net_target)
                vtrace.transform_rewards(
                    episodes, 
                    self.net_target, 
                    self.net_reg,
                    self.net_reg_,
                    alpha=alpha,
                    eta=self.eta)
                vtrace.v_trace(
                    episodes,
                    self.net_target,
                    self.net_reg,
                    self.net_reg_,
                    alpha=alpha,
                    eta=self.eta,
                )
                vtrace.accumulate_gradients(
                    episodes,
                    self.net,
                    batch_size=1000,
                    clip_neurd=self.grad_clip,
                    beta=self.beta,
                )
                # this accumulates gradients in net

                self.optimizer.step()
                self.optimizer.zero_grad()

                params1 = self.net.state_dict()
                params2 = self.net_target.state_dict()
                for name1, param1 in params1.items():
                    params2[name1].data.copy_(
                        self.gamma * param1.data + (1-self.gamma)*params2[name1].data)
                self.net_target.load_state_dict(params2)
                print('n: ', self.n)
            # outerloop resume

            self.n = 0
            self.m += 1
            self.net_reg_ = self.net_reg
            self.net_reg = self.net_target

            # NashConv
            print('starting NashConv calculation')
            # self.tree.to(torch.device('cpu'))
            expl = metric.nash_conv(self.tree, self.net_reg, inference_batch_size=1000)
            # self.tree.to(torch.device('cuda'))
            print('NashConv:', expl)
            print('root strats', self.tree.nash[1])
            print(self.tree.expected_value[1])
            print()

            may_resume, delta_m = self._get_delta_m()

    def run (self):

        self.initialize()
        self.resume()


if __name__ == '__main__' :

    
    # make new tree
    test_run = RNaD(tree_id='1667264620', directory_name='first',)
    test_run.run()