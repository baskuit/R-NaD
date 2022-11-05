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

import matplotlib.pyplot as pyplot

from typing import Any, List, Sequence, Union

class RNaD () :

    def __init__ (self,
        # R-NaD parameters, see paper
        tree_id : str,
        eta=.2,
        delta_m_0 = (100, 165, 200,),
        delta_m_1 = (10_000, 100_000, 35_000,),
        lr=5*10**-5,
        beta=2,
        neurd_clip=10**3,
        grad_clip=10**3,
        b1_adam=0,
        b2_adam=.999,
        epsilon_adam=10**-8,
        gamma=.001,
        roh_bar=1,
        c_bar=1,
        batch_size=3*2**8,
        epsilon_threshold=.03,
        n_discrete=32,
        device=torch.device('cuda'),
        directory_name=None,

    ):
        self.tree_id = tree_id,

        self.eta = eta
        self.delta_m_0 = delta_m_0
        self.delta_m_1 = delta_m_1
        self.lr = lr
        self.beta = beta
        self.neurd_clip = neurd_clip
        self.grad_clip = grad_clip
        self.b1_adam = b1_adam
        self.b2_adam = b2_adam
        self.epsilon_adam = epsilon_adam
        self.gamma = gamma
        self.roh_bar = roh_bar
        self.c_bar = c_bar
        self.batch_size = batch_size
        self.epsilon_threshold = epsilon_threshold
        self.n_discrete = n_discrete

        saved_runs_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_runs')    
        if not os.path.exists(saved_runs_dir):
            os.mkdir(saved_runs_dir)
        if directory_name is None:
            directory_name = str(int(time.perf_counter()))
        self.directory_name = directory_name
        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_runs', directory_name)       
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

        #### #### #### ####
        self.saved_keys = [key for key in self.__dict__.keys()]
        #### #### #### ####

        self.device = device

        self.alpha_lambda = lambda n, delta_m: 1 if n > delta_m / 2 else n * 2 / delta_m

        self.tree = game.Tree()
        self.tree.load(tree_id)
        logging.debug("Tree {} loaded, index tensor has shape {}".format(tree_id, self.tree.index.shape))
        self.tree.to(self.device)

        self.m = 0
        self.n = 0

        self.net_params = {'size':self.tree.max_actions,'channels':2**7,'depth':2,'device':self.device}
        self.net: net.ConvNet = None
        self.net_target: net.ConvNet = None
        self.net_reg: net.ConvNet = None
        self.net_reg_: net.ConvNet = None

    def _learn(
        self,
        episodes: game.Episodes,
        alpha:float,
    ):

            player_id = episodes.turns
            action_oh = episodes.actions
            policy = episodes.policy
            rewards = torch.stack([episodes.rewards, -episodes.rewards], dim=0)
            valid = (episodes.indices != 0).to(torch.float)
            T = valid.shape[0]

            logit, log_pi, pi, v = self.net.forward_batch(episodes)
            v_target_list, has_played_list, v_trace_policy_target_list = [], [], []

            with torch.no_grad():
                _, _, _, v_target = self.net_target.forward_batch(episodes)
                _, log_pi_reg, _, _ = self.net_reg.forward_batch(episodes)
                _, log_pi_reg_, _, _ = self.net_reg_.forward_batch(episodes)


                log_policy_reg = log_pi - (alpha * log_pi_reg + (1 - alpha) * log_pi_reg_)

                for player in range(2):
                    reward = rewards[player, :, :]
                    v_target, has_played, policy_target_ = vtrace.v_trace(
                        v_target,
                        valid,
                        player_id,
                        policy,
                        pi,
                        log_policy_reg,
                        vtrace._player_others(player_id, valid, player),
                        action_oh,
                        reward,
                        player,
                        lambda_=1.0,
                        c=1.0,
                        rho=torch.inf,
                        eta=0.2,
                    )
                    v_target_list.append(v_target)
                    has_played_list.append(has_played)
                    v_trace_policy_target_list.append(policy_target_)

            loss_v = vtrace.get_loss_v([v] * 2, v_target_list, has_played_list)

            is_vector = torch.unsqueeze(torch.ones_like(valid), dim=-1)
            importance_sampling_correction = [is_vector] * 2

            loss_nerd = vtrace.get_loss_nerd(
                [logit] * 2,
                [pi] * 2,
                v_trace_policy_target_list,
                valid,
                player_id,
                episodes.masks,
                importance_sampling_correction,
                clip=self.neurd_clip,
                threshold=2.0,
            )

            loss = loss_v + loss_nerd
            loss.backward()

            nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)

            avg_traj_len = valid.sum(0).mean(-1)

            to_log = {
                "loss_v": loss_v.item(),
                "loss_nerd": loss_nerd.item(),
                "loss_total": loss.item(),
                "traj_len": avg_traj_len.item(),
            }

            # tqdm_repr = {k: round(v, 4) for k, v in to_log.items()}
            # logging.info(f"Training step: {n} {repr(tqdm_repr)}")

    def _new_net (self) -> nn.Module:
        return net.ConvNet(**self.net_params)

    def _load_checkpoint (self, m, n):
        saved_dict = torch.load(os.path.join(self.directory, str(m), str(n)))
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

    def _save_checkpoint (self, nash_conv_data=None):
        saved_dict = {
            'net_params':self.net_params,
            'net':self.net.state_dict(),
            'net_target':self.net.state_dict(),
            'net_reg':self.net.state_dict(),
            'net_reg_':self.net.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'nash_conv_data':nash_conv_data,
        }
        if not os.path.exists(os.path.join(self.directory, str(self.m))):
            os.mkdir(os.path.join(self.directory, str(self.m)))
        torch.save(saved_dict, os.path.join(self.directory, str(self.m), str(self.n)))
         
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
            self._save_checkpoint()

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

            self._load_checkpoint(self.m, self.n)

    def _get_delta_m (self) -> tuple[bool, int]:
        bounding_indices = [_ for _, bound in enumerate(self.delta_m_0) if bound > self.m]
        if not bounding_indices:
            return False, 0

        idx = min(bounding_indices)
        return True, self.delta_m_1[idx]

            
    def resume (
        self,
        checkpoint_mod=1000,
        expl_mod=1,
    ) :

        may_resume, delta_m = self._get_delta_m()

        while may_resume:

            logging.info('m:{}, n:{}'.format(self.m, self.n))
            while self.n < delta_m:
                alpha = self.alpha_lambda(self.n, delta_m)

                if self.n % checkpoint_mod == 0:
                    nash_conv_data = None
                    if self.m % expl_mod == 0 and self.n == 0:
                        nash_conv_data = metric.nash_conv(self.tree, self.net)
                        mean_nash_conv = metric.mean_nash_conv_by_depth(nash_conv_data)
                        logging.info('mean nash_conv by depth:')
                        for depth, nash_conv in mean_nash_conv.items():
                            logging.info('depth:{}, nash_conv:{}'.format(depth, nash_conv))
                    self._save_checkpoint(nash_conv_data=nash_conv_data)

                episodes = game.Episodes(self.tree, self.batch_size)
                episodes.generate(self.net_target)
                self._learn(episodes, alpha)
                self.optimizer.step()
                self.optimizer.zero_grad()

                params1 = self.net.state_dict()
                params2 = self.net_target.state_dict()
                for name1, param1 in params1.items():
                    params2[name1].data.copy_(
                        self.gamma * param1.data + (1-self.gamma)*params2[name1].data
                    )
                self.net_target.load_state_dict(params2)

                self.n += 1

            self.n = 0
            self.m += 1
            self.net_reg_ = self.net_reg
            self.net_reg = self.net_target
            
            may_resume, delta_m = self._get_delta_m()

    def run (
        self,
        checkpoint_mod=1000,
        expl_mod=1,
    ):
        self.initialize()
        self.resume(checkpoint_mod=checkpoint_mod, expl_mod=expl_mod)
        # try:
        #     self.resume(expls, expl_mod=expl_mod)
        # except KeyboardInterrupt:
        #     fig, ax = pyplot.subplots()
        #     ax.plot([_[0] for _ in expls], [_[1] for _ in expls])
        #     pyplot.ylim([0, 2])
        #     fig.savefig("test.png")
        #     pyplot.show()


if __name__ == '__main__' :
    """"
    These are the default main hyperparameters of the R-NaD algorithm.
    In my earlier tests, I had success with small (~16) batches sizes
    and higher (.01) learning rates
    """

    logging.basicConfig(level=logging.DEBUG)

    trial = RNaD(
        device=torch.device('cpu'),
        eta=.2,
        # schedule for number of steps before updating regularizer policies
        # e.g. if m < 100 then delta_m = 10_000 etc
        delta_m_0 = (400, 1000, 2000,),
        delta_m_1 = (100, 200, 1000,),
        lr=5*10**-5,
        batch_size=2**4,
        beta=2, # logit clip
        neurd_clip=10**4, # Q value clip
        grad_clip=10**4, # gradient clip

        # These probably aren't as important
        b1_adam=0,
        b2_adam=.999,
        epsilon_adam=10**-8, # Adam optim params
        gamma=.001, #averaging for target net
        roh_bar=1,
        c_bar=1,
        # directory_name='expl save test',
        # tree_id='2x2rootmixed',
        tree_id='recent',
        )

    trial.run(
        checkpoint_mod=10**3,
        expl_mod=1
    ) 
    # expl mod is after how many steps to calculate NashConv. Set to 1 for large delta_m