import torch
import logging
from time import time
from random import random

from environment.tree import Tree
from learn.rnad import RNaD

if __name__ == "__main__":

    """
    The following is a minimal application of the project.
    We first generate a small tree, so that we can show convergence
    on even on CPU in a reasonable amount of time. Then we save the tree onto disc,
    so that if we want to do another RNaD run on the same tree, we can simply
    call "tree.load()" instead.

    The project uses wandb for data visualization. It can be disabled.

    How to interpret results:
    The objective is to minimize the exploitablity ("NashConv") of our networks policy.
    this is defined as {p1's best response expected reward vs p2's policy} - {p2's best response expected reward vs p1's policy}.
    Some facts about NashConv:
    The lowest possible NashConv is 0, aka the policy is a Nash Eqilibrium.
    The highest possible NashConv is 2, when rewards are bounded [-1, 1].
    In general, the NashConv converges to the max value rapidly as the depth of the game increases
    """

    logging.basicConfig(level=logging.DEBUG)

    tree = Tree(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_actions=3,
        max_transitions=2,
        transition_threshold=0.3,
        depth_bound=4,
        depth_bound_lambda=lambda tree: tree.depth_bound - 1 - 2 * (random() < 0.5),
        desc="3x3 stochastic tree, with depth up to 4",
    )

    tree.generate()
    tree.assert_index_is_tree()
    tree.save("small_tree")
    # tree.load('small_tree')
    # Swap comments in the above two lines if you have already generated the tree. Otherwise it will simply be overwritten.

    etas_to_test = [0, 0.2, 0.5, 1]
    timestamp = str(int(time()))

    for idx, eta in enumerate(etas_to_test):

        same_init_net = None if idx == 0 else f"{timestamp}-eta={etas_to_test[0]}"
        # We want to use same initial net to compare. Usually I have a pre-saved run sitting around.

        trial = RNaD(
            use_same_init_net_as=same_init_net,
            tree=tree,
            directory_name=f"{timestamp}-eta={eta}",
            device=tree.device,
            wandb=False,
            eta=eta,
            bounds=[
                64,
            ],
            delta_m=[
                100,
            ],
            lr=1 * 10**-3,
            gamma_averaging=0.01,
            batch_size=2**9,
            logit_clip=2,
            # net_params= {'type':'ConvNet','size':tree.max_actions,'channels':2**4,'depth':2,'batch_norm':True,'device':tree.device},
            net_params={"type": "MLP", "max_actions": tree.max_actions, "width": 2**8},
        )

        trial.run(
            # every _ steps:
            log_mod=10,  # send wandb log
            expl_mod=1,  # calc expl
            checkpoint_mod=1,  # save nets
        )
