import torch
import torch.nn as nn
import torch.nn.functional as F

import game
import net

class Search:

    """
    A tree-wide search can be simlutated by talking value inferences of the tree
    and applying a map that recursively replaces these values with the NE payoff
    of the matrix of previous value estimates.

    We can also test my imperfect information look-ahead idea. No explanation yet, sorry.
    """

    def __init__(self, tree: game.Tree, same_info_prob=.7):
        self.tree = tree
        self.size = tree.value.shape[0]
        self.policy = torch.zeros_like(tree.solution)
        self.value = torch.zeros((self.size, 2), device=tree.device, dtype=torch.float)
        same_info_prng = torch.rand_like(self.value)
        self.same_info = same_info_prng < same_info_prob

    def to(self, device):
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                self.__dict__[key] = value.to(device)


    def apply_net(self, net: net.ConvNet, inference_batch_size=10**5):

        net.eval()
        for _ in range(self.size // inference_batch_size + 1):
            slice_range = torch.arange(
                _ * inference_batch_size,
                min((_ + 1) * inference_batch_size, self.size),
                device=self.tree.device,
            )
            value_slice = self.tree.expected_value[slice_range]
            legal_slice = self.tree.legal[slice_range]

            with torch.no_grad():
                inference_slice_row = torch.cat([value_slice, legal_slice], dim=1)
                _, policy, _, value = net.forward(inference_slice_row)
                self.policy[slice_range, : self.tree.max_actions] = policy
                self.value[slice_range, : 1] = value
                inference_slice_col = torch.cat([-value_slice, legal_slice], dim=1).swapaxes(
                    2, 3
                )
                _, policy, _, value = net.forward(inference_slice_col)
                self.policy[slice_range, self.tree.max_actions : ] = policy
                self.value[slice_range, 1 : ] = value

    def step(self, idx_cur=0):
        matrix_1 = torch.zeros(
            (self.tree.max_transitions * self.tree.max_actions**2,),
            device=self.tree.device,
            dtype=torch.float,
        )
        matrix_2 = torch.zeros_like(matrix_1)

        index = torch.flatten(
            self.tree.index[idx_cur : idx_cur + 1]
        )
        chance = torch.flatten(
            self.tree.chance[idx_cur : idx_cur + 1]
            * self.tree.legal[idx_cur : idx_cur + 1]
        )
        # tensors at the root, shape (1, t, a, a)

        subtrees = (chance > 0).nonzero()
        depths = [0]
        for idx_flat in subtrees:

            transition_prob = chance[idx_flat]
            root_index_ = index[idx_flat[0]]
            matrix_1[idx_flat] = self.value[root_index_, 0] * transition_prob
            matrix_2[idx_flat] = self.value[root_index_, 1] * transition_prob

        matrix_1 = matrix_1.view(self.tree.max_transitions, self.tree.max_actions, self.tree.max_actions)
        matrix_2 = matrix_2.view(self.tree.max_transitions, self.tree.max_actions, self.tree.max_actions)


        matrix_1 = torch.sum(matrix_1, dim=0)
        matrix_2 = torch.sum(matrix_2, dim=0)

        pi_1 = self.tree._solve( matrix_1)[:self.tree.max_actions].unsqueeze(dim=1)
        pi_2 = self.tree._solve(-matrix_2)[self.tree.max_actions:].unsqueeze(dim=1)

        row_actions_mask = self.tree.legal[idx_cur, 0, :, 0].to(torch.bool)
        col_actions_mask = self.tree.legal[idx_cur, 0, 0, :].to(torch.bool)
        prod_1 = torch.flatten(torch.matmul(matrix_1, pi_2))[row_actions_mask]
        prod_2 = torch.flatten(torch.matmul(pi_1, matrix_2))[col_actions_mask]