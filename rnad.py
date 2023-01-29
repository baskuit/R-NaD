import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import os
import time

import game
import batch
import net
import vtrace
import metric

import wandb  # much better than usin pyplot >_>

from typing import Dict


class RNaD:
    def __init__(
        self,
        tree: game.Tree,
        # R-NaD parameters, see paper
        eta=0.2,
        bounds=[
            100,
            165,
            200,
        ],
        delta_m=[
            10_000,
            100_000,
            35_000,
        ],
        buffer_size=1,  # This simulates no buffer
        buffer_mod=1,  # How many steps to grab new batch
        lr=5 * 10**-5,
        logit_clip=2,
        neurd_clip=10**3,
        grad_clip=10**3,
        b1_adam=0,
        b2_adam=0.999,
        epsilon_adam=10**-8,
        gamma_averaging=0.001,
        roh_bar=1,
        c_bar=1,
        batch_size=3 * 2**8,
        epsilon_threshold=0.03,
        n_discrete=32,
        device=torch.device("cuda"),
        directory_name=None,
        net_params=None,
        vtrace_gamma=1,
        wandb=False,
        same_init_net=False,
        value_weight=1,
        neurd_weight=1,
    ):
        self.neurd_weight = neurd_weight
        self.value_weight = value_weight
        self.tree = tree
        self.tree_hash = 0

        self.eta = eta
        self.bounds = bounds
        self.delta_m = delta_m
        self.buffer_size = buffer_size
        self.buffer_mod = buffer_mod
        self.lr = lr
        self.beta = logit_clip
        self.neurd_clip = neurd_clip
        self.grad_clip = grad_clip
        self.b1_adam = b1_adam
        self.b2_adam = b2_adam
        self.epsilon_adam = epsilon_adam
        self.gamma_averaging = gamma_averaging
        self.roh_bar = roh_bar
        self.c_bar = c_bar
        self.batch_size = batch_size
        self.epsilon_threshold = epsilon_threshold
        self.n_discrete = n_discrete
        self.vtrace_gamma = vtrace_gamma
        self.wandb = wandb

        if directory_name is None:
            directory_name = str(int(time.perf_counter()))
        self.directory_name = directory_name

        if net_params is None:
            net_params = {
                "type": "ConvNet",
                "size": self.tree.max_actions,
                "depth": 2,
                "channels": 2**5,
                "batch_norm": False,
                "device": self.device,
            }
        self.net_params = net_params

        #### #### #### ####
        self.saved_keys = [key for key in self.__dict__.keys() if key != "tree"]
        #### #### #### ####
        self.device = device
        saved_runs_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "saved_runs"
        )
        if not os.path.exists(saved_runs_dir):
            os.mkdir(saved_runs_dir)
        self.directory = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "saved_runs", directory_name
        )
        self.same_init_net = same_init_net

        self.m = 0
        self.n = 0
        self.total_steps = 0  # saved in checkpoint
        self.net: net.ConvNet = None
        self.net_target: net.ConvNet = None
        self.net_reg: net.ConvNet = None
        self.net_reg_: net.ConvNet = None

        self.alpha_lambda = lambda n, delta_m: 1 if n > delta_m / 2 else n * 2 / delta_m

    def _new_net(
        self,
    ) -> nn.Module:
        if self.net_params["type"] == "ConvNet":
            t = net.ConvNet
        if self.net_params["type"] == "MLP":
            t = net.MLP
        net_params = {_: __ for _, __ in self.net_params.items() if _ != "type"}
        net_params["device"] = self.device
        new_net = t(**net_params)
        new_net.eval()
        return new_net

    def _initialize(self):

        logging.info("Initializing R-NaD run: {}".format(self.directory_name))

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

        updates = [
            int(os.path.relpath(f.path, self.directory))
            for f in os.scandir(self.directory)
            if f.is_dir()
        ]
        if not updates:
            if hasattr(self.tree, "hash"):
                self.tree_hash = self.tree.hash
            params_dict = {key: self.__dict__[key] for key in self.saved_keys}
            torch.save(params_dict, os.path.join(self.directory, "params"))

            os.mkdir(os.path.join(self.directory, "0"))
            self.net = self._new_net()
            self.net.train()
            if self.same_init_net:
                net_dir = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "saved_runs",
                    self.same_init_net,
                    "0",
                    "0",
                )
                checkpoint = torch.load(net_dir)
                self.net.load_state_dict(checkpoint["net"])
                logging.info("Loading init net from {}".format(self.same_init_net))
            self.net_target = self._new_net()
            self.net_target.load_state_dict(self.net.state_dict())
            self.net_reg = self._new_net()
            self.net_reg.load_state_dict(self.net.state_dict())
            self.net_reg_ = self._new_net()
            self.net_reg_.load_state_dict(self.net.state_dict())
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=self.lr,
                betas=[self.b1_adam, self.b2_adam],
                eps=self.epsilon_adam,
            )
            self.m = 0
            self.n = 0
            self._save_checkpoint()

        else:

            params_dict = torch.load(os.path.join(self.directory, "params"))
            for key, value in params_dict.items():
                if key == "directory_name":
                    params_dict[key] = self.directory_name
                    continue
                if key == "device":
                    continue
                if torch.is_tensor(value):
                    params_dict[key] = params_dict[key].to(self.device)
                if key == "tree_hash":
                    assert params_dict["tree_hash"] == self.tree.hash
                self.__dict__[key] = value
            torch.save(params_dict, os.path.join(self.directory, "params"))

            self.m = max(updates)
            last_update = os.path.join(self.directory, str(self.m))
            checkpoints = [
                int(os.path.relpath(f.path, last_update))
                for f in os.scandir(last_update)
                if not f.is_dir()
            ]
            self.n = max(checkpoints)
            self._load_checkpoint(self.m, self.n)

        if self.wandb:
            wandb.init(
                resume=bool(updates),
                project="RNaD",
                config={key: self.__dict__[key] for key in self.saved_keys},
            )

    def _load_checkpoint(self, m, n):
        saved_dict = torch.load(os.path.join(self.directory, str(m), str(n)))
        self.total_steps = saved_dict["total_steps"]
        self.net_params = saved_dict["net_params"]
        self.net = self._new_net()
        self.net.load_state_dict(saved_dict["net"])
        self.net_target = self._new_net()
        self.net_target.load_state_dict(saved_dict["net_target"])
        self.net_reg = self._new_net()
        self.net_reg.load_state_dict(saved_dict["net_reg"])
        self.net_reg_ = self._new_net()
        self.net_reg_.load_state_dict(saved_dict["net_reg_"])
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.lr,
            betas=[self.b1_adam, self.b2_adam],
            eps=self.epsilon_adam,
        )
        self.optimizer.load_state_dict(saved_dict["optimizer"])

    def _save_checkpoint(self):
        saved_dict = {
            "total_steps": self.total_steps,
            "net_params": self.net_params,
            "net": self.net.state_dict(),
            "net_target": self.net_target.state_dict(),
            "net_reg": self.net_reg.state_dict(),
            "net_reg_": self.net_reg_.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if not os.path.exists(os.path.join(self.directory, str(self.m))):
            os.mkdir(os.path.join(self.directory, str(self.m)))
        torch.save(saved_dict, os.path.join(self.directory, str(self.m), str(self.n)))

    def _load_saved_net(self, net: torch.nn.Module, m, key="net_reg"):
        """
        Modify net in place to a checkpoint net from m=m, n=0
        """
        # m is the step you are loading it from
        # use m-1 to get net_reg_...
        if m is None:
            net.load_state_dict(self.net.state_dict())
            return
        m = max(m, 0)
        saved_dict = torch.load(os.path.join(self.directory, str(m), "0"))
        net.load_state_dict(saved_dict[key])

    def _get_update_info(self) -> tuple[bool, int]:
        bounding_indices = [_ for _, bound in enumerate(self.bounds) if bound > self.m]
        if not bounding_indices:
            return False, 0

        idx = min(bounding_indices)
        return True, self.delta_m[idx]

    def _nash_conv(
        self,
    ):
        logging.info(
            "NashConv at m: {}, n: {}, step {}".format(self.m, self.n, self.total_steps)
        )
        # logging.info('\nnet')
        # nash_conv_data = metric.nash_conv(self.tree, self.net)
        # mean_nash_conv = metric.mean_nash_conv_by_depth(nash_conv_data)
        # for depth, nash_conv in mean_nash_conv.items():
        #     logging.info('depth:{}, nash_conv:{}'.format(depth, nash_conv))
        logging.info("\nnet target")
        nash_conv_data_target = metric.nash_conv(self.tree, self.net_target)
        mean_nash_conv_target = metric.mean_nash_conv_by_depth(nash_conv_data_target)
        for depth, nash_conv in mean_nash_conv_target.items():
            logging.info("depth:{}, nash_conv:{}".format(depth, nash_conv))
        # self.nash_conv[self.total_steps] = (nash_conv_data.max_1 - nash_conv_data.min_2)[1].item()
        return (nash_conv_data_target.max_1 - nash_conv_data_target.min_2)[1].item()

    def _learn(
        self,
        episodes: batch.Episodes,
        alpha: float,
        log: dict = None,
    ):

        player_id = episodes.turns
        action_oh = episodes.actions
        policy = episodes.policy
        rewards = torch.stack([episodes.rewards, -episodes.rewards], dim=0)
        valid = (episodes.indices != 0).to(torch.float)
        T = valid.shape[0]

        logit, log_pi, pi, v = self.net.forward_batch(episodes)
        pi_processed = pi
        # pi_processed = vtrace.process_policy(pi, episodes.masks, self.n_discrete, self.epsilon_threshold) # TODO this function seems to filter masks too? ofc duh
        v_target_list, has_played_list, v_trace_policy_target_list = [], [], []

        with torch.no_grad():
            _, _, _, v_target = self.net_target.forward_batch(episodes)
            _, log_pi_reg, _, _ = self.net_reg.forward_batch(episodes)
            _, log_pi_reg_, _, _ = self.net_reg_.forward_batch(episodes)

            log_policy_reg = log_pi - (alpha * log_pi_reg + (1 - alpha) * log_pi_reg_)

            for player in range(2):
                reward = rewards[player, :, :]
                v_target_, has_played, policy_target_ = vtrace.v_trace(
                    v_target,
                    valid,
                    player_id,
                    policy,
                    pi_processed,
                    log_policy_reg,
                    vtrace._player_others(player_id, valid, player),
                    action_oh,
                    reward,
                    player,
                    lambda_=1.0,
                    c=self.c_bar,
                    rho=self.roh_bar,
                    eta=self.eta,
                    gamma=self.vtrace_gamma,
                )

                v_target_list.append(v_target_)
                has_played_list.append(has_played)
                v_trace_policy_target_list.append(policy_target_)
        loss_v = vtrace.get_loss_v([v] * 2, v_target_list, has_played_list)

        is_vector = torch.unsqueeze(torch.ones_like(valid), dim=-1)
        importance_sampling_correction = [is_vector] * 2

        loss_nerd = vtrace.get_loss_nerd(
            [logit] * 2,
            [pi_processed] * 2,
            v_trace_policy_target_list,
            valid,
            player_id,
            episodes.masks,
            importance_sampling_correction,
            clip=self.neurd_clip,
            threshold=self.beta,
        )

        loss = self.value_weight * loss_v + self.neurd_weight * loss_nerd
        loss.backward()

        total_norm = 0
        for p in self.net.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)

        avg_traj_len = valid.sum(0).mean(-1)
        logit_mean = logit.mean().item()
        logit_max_from_mean = torch.max(torch.abs(logit - logit_mean)).item()

        to_log = {
            "loss_v": loss_v.item(),
            "loss_nerd": loss_nerd.item(),
            "traj_len": avg_traj_len.item(),
            "gradient_norm": total_norm,
            "logit_mean": logit_mean,
            "logit_max": logit_max_from_mean,
        }
        if log is not None:
            log.update(to_log)

    def _resume(
        self,
        max_updates=10**6,
        checkpoint_mod=1000,
        expl_mod=1,
        log_mod=20,
    ) -> None:

        buffer = batch.Buffer(self.buffer_size)

        for _ in range(max_updates):
            may_resume, delta_m = self._get_update_info()
            log = {}
            if not may_resume:
                return

            logging.info("m: {}, delta_m: {}".format(self.m, delta_m))

            buffer.max_size = self.buffer_size

            if self.m % expl_mod == 0 and self.n == 0 and self.m != 0:
                nash_conv = self._nash_conv()
                if self.wandb:
                    wandb.log({"nash_conv": nash_conv}, step=self.total_steps)

            while self.n < delta_m:

                alpha = self.alpha_lambda(self.n, delta_m)

                if self.n % checkpoint_mod == 0:
                    self._save_checkpoint()

                if self.total_steps % self.buffer_mod == 0:
                    episodes = batch.Episodes(self.tree, self.batch_size)
                    episodes.generate(self.net)
                    buffer.append(episodes)

                episodes_sample = buffer.sample(self.batch_size)

                self._learn(episodes_sample, alpha, log=log)

                self.optimizer.step()
                self.optimizer.zero_grad()
                params1: Dict['str', torch.Tensor] = self.net.state_dict()
                params2: Dict['str', torch.Tensor] = self.net_target.state_dict()
                for name1, param1 in params1.items():
                    params2[name1].data.copy_(
                        self.gamma_averaging * param1.data
                        + (1 - self.gamma_averaging) * params2[name1].data
                    )
                self.net_target.load_state_dict(params2)

                if self.n % log_mod == 0 and self.wandb:
                    wandb.log(log, step=self.total_steps)

                self.n += 1
                self.total_steps += 1

            self.n = 0
            self.m += 1
            self.net_reg_.load_state_dict(self.net_reg.state_dict())
            self.net_reg.load_state_dict(self.net_target.state_dict())

    def run(self, max_updates=10**6, checkpoint_mod=1000, expl_mod=1, log_mod=20):
        self._initialize()
        self._resume(
            max_updates=max_updates,
            checkpoint_mod=checkpoint_mod,
            expl_mod=expl_mod,
            log_mod=log_mod,
        )
        if self.wandb:
            wandb.finish()


if __name__ == "__main__":

    pass
