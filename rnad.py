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

from typing import Any, List, Sequence, Union

class RNaD () :

    def __init__ (self,
        # R-NaD parameters, see paper
        tree_id : str,
        eta=.2,
        delta_m_0 = (100, 165, 200, 250),
        delta_m_1 = (1000, 2_000, 5_000, 0),
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

        self.net_params = {'size':self.tree.max_actions,'channels':2**7,'depth':1,'device':torch.device('cuda')}
        self.net:net.ConvNet = None
        self.net_target:net.ConvNet = None
        self.net_reg:net.ConvNet = None
        self.net_reg_:net.ConvNet = None

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
            self.save()

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
        may_resume, delta_m = self._get_delta_m()

        while may_resume:
            print('may resume, delta m')
            print(may_resume, delta_m)
            print('m', self.m, 'n', self.n)
            
            while self.n < delta_m:
                alpha = self.alpha_lambda(self.n, delta_m)

                if self.n % self.checkpoint_mod == 0:
                    self.save()

                self.n += 1

                episodes = game.Episodes(self.tree, self.batch_size)
                episodes.generate(self.net_target)
                self._learn(episodes, alpha)
                # this accumulates gradients in net

                self.optimizer.step()
                self.optimizer.zero_grad()

                params1 = self.net.state_dict()
                params2 = self.net_target.state_dict()
                for name1, param1 in params1.items():
                    params2[name1].data.copy_(
                        self.gamma * param1.data + (1-self.gamma)*params2[name1].data
                    )
                self.net_target.load_state_dict(params2)
            # outerloop resume

            self.n = 0
            self.m += 1
            self.net_reg_ = self.net_reg
            self.net_reg = self.net_target

            # NashConv
            print('starting NashConv calculation')
            expl = metric.nash_conv(self.tree, self.net_reg, inference_batch_size=1000)
            print('NashConv:', expl)
            print(self.tree.expected_value[1])
            print('root strats', self.tree.nash[1])
            print('payoff', self.tree.payoff[1])
            # value_slice = self.tree.expected_value[self.tree.index[self.tree.index != 0]]
            # torch.index_select(self.tree.expected_value, dim=0, )
            
            may_resume, delta_m = self._get_delta_m()

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

            loss_v = self.get_loss_v([v] * 2, v_target_list, has_played_list)

            is_vector = torch.unsqueeze(torch.ones_like(valid), dim=-1)
            importance_sampling_correction = [is_vector] * 2

            loss_nerd = self.get_loss_nerd(
                [logit] * 2,
                [pi] * 2,
                v_trace_policy_target_list,
                valid,
                player_id,
                episodes.masks,
                importance_sampling_correction,
                clip=10_000,
                threshold=2.0,
            )

            loss = loss_v + loss_nerd
            loss.backward()

            nn.utils.clip_grad_norm_(self.net.parameters(), 10_000)

            avg_traj_len = valid.sum(0).mean(-1)

            to_log = {
                "loss_v": loss_v.item(),
                "loss_nerd": loss_nerd.item(),
                "loss_total": loss.item(),
                "traj_len": avg_traj_len.item(),
            }

            # tqdm_repr = {k: round(v, 4) for k, v in to_log.items()}
            # logging.info(f"Training step: {n} {repr(tqdm_repr)}")

    def get_loss_v(
        self,
        v_list: Sequence[torch.Tensor],
        v_target_list: Sequence[torch.Tensor],
        mask_list: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Define the loss function for the critic."""
        loss_v_list = []
        for (v_n, v_target, mask) in zip(v_list, v_target_list, mask_list):
            assert v_n.shape[0] == v_target.shape[0]

            loss_v = torch.unsqueeze(mask, dim=-1) * (v_n - v_target.detach()) ** 2
            normalization = torch.sum(mask)
            loss_v = torch.sum(loss_v) / (normalization + (normalization == 0.0))

            loss_v_list.append(loss_v)

        return sum(loss_v_list)


    def get_loss_nerd(
        self,
        logit_list: Sequence[torch.Tensor],
        policy_list: Sequence[torch.Tensor],
        q_vr_list: Sequence[torch.Tensor],
        valid: torch.Tensor,
        player_ids: Sequence[torch.Tensor],
        legal_actions: torch.Tensor,
        importance_sampling_correction: Sequence[torch.Tensor],
        clip: float = 100,
        threshold: float = 2,
    ) -> torch.Tensor:
        """Define the nerd loss."""
        assert isinstance(importance_sampling_correction, list)
        loss_pi_list = []
        for k, (logit_pi, pi, q_vr, is_c) in enumerate(
            zip(logit_list, policy_list, q_vr_list, importance_sampling_correction)
        ):
            assert logit_pi.shape[0] == q_vr.shape[0]
            # loss policy
            adv_pi = q_vr - torch.sum(pi * q_vr, dim=-1, keepdim=True)
            adv_pi = is_c * adv_pi  # importance sampling correction
            adv_pi = torch.clip(adv_pi, min=-clip, max=clip)
            adv_pi = adv_pi.detach()

            logits = logit_pi - torch.mean(logit_pi * legal_actions, dim=-1, keepdim=True)

            threshold_center = torch.zeros_like(logits)

            nerd_loss = torch.sum(
                legal_actions * vtrace.apply_force_with_threshold(logits, adv_pi, threshold, threshold_center), axis=-1
            )
            nerd_loss = -vtrace.renormalize(nerd_loss, valid * (player_ids == k))
            loss_pi_list.append(nerd_loss)
        return sum(loss_pi_list)


    def run (self):

        self.initialize()
        self.resume()

    def save (self):
        saved_dict = {
            'net_params':self.net_params,
            'net':self.net.state_dict(),
            'net_target':self.net.state_dict(),
            'net_reg':self.net.state_dict(),
            'net_reg_':self.net.state_dict(),
            'optimizer':self.optimizer.state_dict(),
        }
        if not os.path.exists(os.path.join(self.directory, str(self.m))):
            os.mkdir(os.path.join(self.directory, str(self.m)))
        torch.save(saved_dict, os.path.join(self.directory, str(self.m), str(self.n)))

if __name__ == '__main__' :
    # make new tree
    test_run = RNaD(
        tree_id='recent', #3x3 
        directory_name=str(int(time.time())),
        eta=.2,
        delta_m_0 = (100, 165, 200, 250),
        delta_m_1 = (1000, 1000, 2000, 0), 
        batch_size=700,
        lr=.00005,
        beta=10,)
    test_run.run()