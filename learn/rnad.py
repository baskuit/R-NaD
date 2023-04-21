import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import os
import time
from typing import Dict

import environment.tree as tree
import environment.episode as episode
import nn.net as net
import learn.vtrace as vtrace
import util.metric as metric

import wandb

class RNaD:

    """
    Represents a run of the Regularized Nash Dynamics algorithm on a given game Tree and fixed set of hyperparameters.
    Calling "run" on an RNaD object will automaticlly create a folder with its 'directory_name' in the saved_runs directory.
    There a 'params' object will be saved together with the initial network weights ('checkpoint') at n=0, m=0.
    
    During its run, the RNaD object will periodically save checkpoint info at update n, step m in 'm' in the directory 'saved_runs/directory_name/n'.
    If the run is ended prematurely, the next time 'run' is called the object will look in 'saved_runs/directory_name' to find the latest checkpoint and automatically resume from there.
    Its associated wandb run will also be resumed.

    We use the paper's convention of refering to an update of the learner nets parameters as a 'step' and an update of the regularization nets as an 'update'.

    The default parameters are copied from the paper for clarity. They are not recommended for small scale tests. Instead, refer to those in main.py.
    """

    def __init__(
        self,
        tree: tree.Tree,
        device=torch.device("cuda"),
        directory_name=None,

        batch_size=3 * 2**8,
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
        epsilon_threshold=0.03,
        n_discrete=32,
        # parameters from https://arxiv.org/abs/2206.15378

        n_batches_per_buffer=1,  # This simulates no buffer
        buffer_mod=1,  # How many steps to grab new batch
        net_params=None,
        vtrace_gamma=1,
        value_loss_weight=1,
        neurd_loss_weight=1,
        wandb=False,
        use_same_init_net_as=False,
    ):

        """
        tree:
            The tree game the object will be tied to. On resume, a check is performed that the trees hash corresponds to the one saved when the RNaD object was first initialized.
        eta:
            Coefficient for reward transformation. eta=0 means no regularization, which is equivalent to vanilla policy gradient.
        bounds:
        delta_m:
            Schedule for regularization. The correspond to 'n' and 'm' in the paper, respectively.
        lr:
            Learning rate. It is currently fixed thoughout the run.
        logit_clip:
            The logit output of the policy net is bounded by [-logit_clip, logit_clip], for numerical stability. If a step has logs enabled the the mean logit ouput and the max diffenence from that mean are recorded.
        neurd_clip:
            xxx
        grad_clip:
            Normal gradient clip applied each step.
        b1_adam:
        b2_adam:
        epsilon_adam
            Parameters for the Adam optimizer.
        gamma_averaging:
            Weight for the exponential averaging of the target network.
        epsilon_threashold:
        n_discrete:
            Parameters for post-processing of the learner policy. Not used during data generation.buffer
        n_batches_per_buffer
            Number of 'Episodes' objects to store in the replay buffer. Default value of one means a fresh batch of episodes is used every step, so the algorithm is on-policy.
        buffer_mod:
            Number of steps before a new Episodes object is added to the buffer.
        vtrace_loss_weight:
        neurd_loss_weight:
            Weights for the loss of value and policy, respectively
        wandb:
            Whether to use wandb for data visualization.
        use_same_init_net_as:
            If a directory_name is passed, the RNaD object will use the same initial network weights as that run. 
        """

        self.tree = tree
        self.tree_hash = 0
        self.device = device

        self.eta = eta
        self.bounds = bounds
        self.delta_m = delta_m
        self.n_batches_per_buffer = n_batches_per_buffer
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
        self.neurd_weight = neurd_loss_weight
        self.value_weight = value_loss_weight
        self.wandb = wandb

        if directory_name is None:
            directory_name = str(int(time.perf_counter()))
        self.directory_name = directory_name

        if net_params is None:
            net_params = {
                "type": "MLP",
                "max_actions": self.tree.max_actions,
                "width": 2**8,
            }
        self.net_params = net_params

        self.saved_keys = [key for key in self.__dict__.keys() if key != "tree"]
        # only the above members are saved in and reloaded from the 'params' object

        saved_runs_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "saved_runs"
        )
        if not os.path.exists(saved_runs_dir):
            os.mkdir(saved_runs_dir)
        self.directory = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "saved_runs", directory_name
        )
        self.use_same_init_net_as = use_same_init_net_as

        self.m = 0
        self.n = 0
        self.total_steps = 0  # saved in checkpoint
        self.net: net.ConvNet = None
        self.net_target: net.ConvNet = None
        self.net_reg: net.ConvNet = None
        self.net_reg_: net.ConvNet = None

    def __new_net(
        self,
    ) -> nn.Module:
        """
        Initialize a network on self.device with self.net_params
        """
        if self.net_params["type"] == "ConvNet":
            t = net.ConvNet
        if self.net_params["type"] == "MLP":
            t = net.MLP
        net_params = {_: __ for _, __ in self.net_params.items() if _ != "type"}
        net_params["device"] = self.device
        new_net = t(**net_params)
        new_net.eval()
        return new_net

    def __initialize(self):

        """
        Creates a new directory and saves initial network weights in the first update
        """

        logging.info("Initializing R-NaD run: {}".format(self.directory_name))

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

        saved_updates = [
            int(os.path.relpath(f.path, self.directory))
            for f in os.scandir(self.directory)
            if f.is_dir()
        ]
        if not saved_updates:
            self.tree_hash = self.tree.hash
            params_dict = {key: self.__dict__[key] for key in self.saved_keys}
            torch.save(params_dict, os.path.join(self.directory, "params"))

            os.mkdir(os.path.join(self.directory, "0"))
            self.net = self.__new_net()            
            if self.use_same_init_net_as:
                net_dir = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "..",
                    "saved_runs",
                    self.use_same_init_net_as,
                    "0",
                    "0",
                )
                checkpoint = torch.load(net_dir)
                self.net.load_state_dict(checkpoint["net"])
                logging.info("Loading init net from {}".format(self.use_same_init_net_as))
            self.net.train()
            self.net_target = self.__new_net()
            self.net_target.load_state_dict(self.net.state_dict())
            self.net_reg = self.__new_net()
            self.net_reg.load_state_dict(self.net.state_dict())
            self.net_reg_ = self.__new_net()
            self.net_reg_.load_state_dict(self.net.state_dict())
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=self.lr,
                betas=[self.b1_adam, self.b2_adam],
                eps=self.epsilon_adam,
            )
            self.m = 0
            self.n = 0
            self.__save_checkpoint()
            # this initial checkpoint marks the object has having been initialized; 'saved_updates' will be non-empty now

        else:

            params_dict = torch.load(os.path.join(self.directory, "params"))
            for key, value in params_dict.items():
                if key == "directory_name":
                    params_dict[key] = self.directory_name
                    # renaming the directory will update the saved value for 'directory_name' 
                    continue
                if key == "device":
                    # resuming a run with a new device will move the nets to that device
                    continue
                if torch.is_tensor(value):
                    params_dict[key] = params_dict[key].to(self.device)
                if key == "tree_hash":
                    # resuming will fail if the keys dont match
                    assert params_dict["tree_hash"] == self.tree.hash
                self.__dict__[key] = value
            torch.save(params_dict, os.path.join(self.directory, "params"))
            # overwrite the saved params with the above changes

            self.m = max(saved_updates)
            last_update = os.path.join(self.directory, str(self.m))
            checkpoints = [
                int(os.path.relpath(f.path, last_update))
                for f in os.scandir(last_update)
                if not f.is_dir()
            ]
            self.n = max(checkpoints)
            self.__load_checkpoint(self.m, self.n)
            # use the latest checkpoint

        if self.wandb:
            wandb.init(
                resume=bool(saved_updates),
                project="RNaD",
                config={key: self.__dict__[key] for key in self.saved_keys},
            )
            wandb.run.name = self.directory_name

    def __load_checkpoint(self, m, n):

        """
        Updates the net weights, optimizer state, and certain stat members from those saved in the checkpoint
        """

        saved_dict = torch.load(os.path.join(self.directory, str(m), str(n)))
        self.total_steps = saved_dict["total_steps"]
        self.net_params = saved_dict["net_params"]
        self.net = self.__new_net()
        self.net.load_state_dict(saved_dict["net"])
        self.net_target = self.__new_net()
        self.net_target.load_state_dict(saved_dict["net_target"])
        self.net_reg = self.__new_net()
        self.net_reg.load_state_dict(saved_dict["net_reg"])
        self.net_reg_ = self.__new_net()
        self.net_reg_.load_state_dict(saved_dict["net_reg_"])
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.lr,
            betas=[self.b1_adam, self.b2_adam],
            eps=self.epsilon_adam,
        )
        self.optimizer.load_state_dict(saved_dict["optimizer"])

    def __save_checkpoint(self):
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

    def __get_update_info(self) -> tuple[bool, int]:

        """
        The bool value is whether the run is finished, and second is the new delta_m value which determines when how many steps until the next update.
        """

        bounding_indices = [_ for _, bound in enumerate(self.bounds) if bound > self.m]
        if not bounding_indices:
            return False, 0

        idx = min(bounding_indices)
        return True, self.delta_m[idx]

    def __nashconv(self,) -> float:
        
        """
        Returns the NashConv at the root and logs the depth stratified NashConv values.
        Note: NashConv at the largest depth is for the root node/the whole game tree
        This is the metric of primary interest. The target net is used instead of the actor,
        although that can be added if wanted. In my experience, the target is the one that converges. 
        """

        logging.info(
            "NashConv at m: {}, n: {}, step {}".format(self.m, self.n, self.total_steps)
        )
        nashconv_data = metric.NashConvData(self.tree)
        nashconv_data.get_nashconv_from_net(self.tree, self.net_target)
        mean_nashconv: Dict[int, float] = nashconv_data.mean_nashconv_by_depth()
        for depth, nashconv in mean_nashconv.items():
            logging.info("depth:{}, nash_conv:{}".format(depth, nashconv))
        return (nashconv_data.row_best[1] + nashconv_data.col_best[1]).item()

    def __learn(
        self,
        episodes: episode.Episodes,
        alpha: float,
        log: dict = None,
    ):
        
        """
        Computes gradients for learner net from sampled batch of trajectories.
        This is where the reward transformation/regularization is performed.
        """

        player_id = episodes.turns
        action_oh = episodes.actions
        policy = episodes.policy
        rewards = torch.stack([episodes.rewards, -episodes.rewards], dim=0)
        valid = (episodes.indices != 0).to(torch.float)
        masks = episodes.masks
        T = valid.shape[0]

        logit, log_pi, pi, v = self.net.forward_batch(episodes)
        pi_processed = vtrace.process_policy(pi, episodes.masks, self.n_discrete, self.epsilon_threshold)
        v_target_list, has_played_list, v_trace_policy_target_list = [], [], []

        with torch.no_grad():
            _, _, pi_target, v_target = self.net_target.forward_batch(episodes)
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
            masks,
            importance_sampling_correction,
            clip=self.neurd_clip,
            threshold=self.beta,
        )

        loss = self.value_weight * loss_v + self.neurd_weight * loss_nerd
        loss.backward()

        if log is not None:
            total_norm = 0
            for p in self.net.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5

            avg_traj_len = valid.sum(0).mean(-1)
            logit_mean = logit.mean().item()
            logit_max_from_mean = torch.max(torch.abs(logit - logit_mean)).item()

            uniform_policy = torch.nn.functional.normalize(masks, p=1, dim=-1)
            entropy = metric.kld(pi, uniform_policy, valid, legal_actions=masks)
            entropy_target = metric.kld(pi_target, uniform_policy, valid, legal_actions=masks)
            actor_learner_kld = metric.kld(pi, policy, valid, legal_actions=masks)

            to_log = {
                "loss_v": loss_v.item(),
                "loss_nerd": loss_nerd.item(),
                "traj_len": avg_traj_len.item(),
                "gradient_norm": total_norm,
                "logit_mean": logit_mean,
                "logit_max": logit_max_from_mean,
                "entropy": entropy,
                "entropy_target": entropy_target,
                "actor_learner_kld": actor_learner_kld,
            }
            log.update(to_log)

        nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)

    def __resume(
        self,
        max_updates=10**6,
        checkpoint_mod=1000,
        expl_mod=1,
        log_mod=20,
    ) -> None:
        
        """
        Resumes training loop. Terminates when schedule is completed or when max_updates is reached.

        max_updates:
            The max number of updates. Allows partial runs without having to use a short schedule.
        checkpoint_mod:
            Saves a checkpoint after this many steps, always starting at m=0.
        expl_mod:
            Compute NashConv of the target net after this many steps, starting at m=0.
        log_mod:
            Compute logs during learning after this many steps, starting at m=0. 
        """
        
        buffer = episode.Buffer(self.n_batches_per_buffer)
        for _ in range(max_updates):
            may_resume, delta_m = self.__get_update_info()
            
            if not may_resume:
                return

            logging.info("m: {}, delta_m: {}".format(self.m, delta_m))

            buffer.max_size = self.n_batches_per_buffer

            if self.m % expl_mod == 0 and self.n == 0 and self.m != 0:
                nashconv = self.__nashconv()
                if self.wandb:
                    wandb.log({"nashconv": nashconv}, step=self.total_steps)

            while self.n < delta_m:

                alpha = 1 if self.n > delta_m / 2 else self.n * 2 / delta_m

                if self.n % checkpoint_mod == 0:
                    self.__save_checkpoint()

                if self.total_steps % self.buffer_mod == 0:
                    episodes = episode.Episodes(self.tree, self.batch_size)
                    episodes.generate(self.net)
                    buffer.append(episodes)

                episodes_sample = buffer.sample(self.batch_size)

                log = {} if (self.n % log_mod == 0 and self.wandb) else None
                self.__learn(episodes_sample, alpha, log=log)
                if log:
                    wandb.log(log, step=self.total_steps)

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

                self.n += 1
                self.total_steps += 1

            self.n = 0
            self.m += 1
            self.net_reg_.load_state_dict(self.net_reg.state_dict())
            self.net_reg.load_state_dict(self.net_target.state_dict())

    def run(self, max_updates=10**6, checkpoint_mod=1000, expl_mod=1, log_mod=20):

        """
        Either starts or resumes a run.
        """

        self.__initialize()
        self.__resume(
            max_updates=max_updates,
            checkpoint_mod=checkpoint_mod,
            expl_mod=expl_mod,
            log_mod=log_mod,
        )
        if self.wandb:
            wandb.finish()