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
                inference_slice_row = torch.cat([value_slice, legal_slice], dim=1).contiguous()
                _, policy, value, _ = net.forward(inference_slice_row)
                self.policy[slice_range, : self.tree.max_actions] = policy
                self.value[slice_range, 0] = value[:, 0]
                inference_slice_col = torch.cat([-value_slice, legal_slice], dim=1).swapaxes(
                    2, 3
                ).contiguous()
                _, policy, value, _ = net.forward(inference_slice_col)
                self.policy[slice_range, self.tree.max_actions : ] = policy
                self.value[slice_range, 1] = value[:, 0]

    def step(self, idx_cur=1):
        matrix_1 = torch.zeros(
            (self.tree.max_transitions * self.tree.max_actions**2,),
            device=self.tree.device,
            dtype=torch.float,
        )
        matrix_2 = torch.zeros_like(matrix_1)

        value = torch.flatten(
            self.tree.value[idx_cur : idx_cur + 1]
        )
        index = torch.flatten(
            self.tree.index[idx_cur : idx_cur + 1]
        )
        chance = torch.flatten(
            self.tree.chance[idx_cur : idx_cur + 1]
            * self.tree.legal[idx_cur : idx_cur + 1]
        )
        # tensors at the root, shape (1, t, a, a)

        subtrees = (chance > 0).nonzero()
        for idx_flat in subtrees:
            idx_next = index[idx_flat[0]]
            if idx_next == 0:
                v1 = value[idx_next]
                v2 = -value[idx_next]
            else:
                self.step(idx_next)
                v1 = self.value[idx_next, 0]
                v2 = self.value[idx_next, 1]
                # Recursive call just before using update values
            transition_prob = chance[idx_flat]
            matrix_1[idx_flat] = v1 * transition_prob
            matrix_2[idx_flat] = v2 * transition_prob

        matrix_1 = matrix_1.view(self.tree.max_transitions, self.tree.max_actions, self.tree.max_actions)
        matrix_2 = matrix_2.view(self.tree.max_transitions, self.tree.max_actions, self.tree.max_actions)
        matrix_1 = torch.sum(matrix_1, dim=0)
        matrix_2 = torch.sum(matrix_2, dim=0)
        # Add to get expected value estimates and reshape into matrix shape

        solution_1 = self.tree._solve( matrix_1)[0]
        solution_2 = self.tree._solve(-matrix_2)[0]
        # Solve NE in both players search trees

        pi_1, pi_2 = solution_1[: self.tree.max_actions].unsqueeze(dim=0), solution_1[
            self.tree.max_actions :
        ].unsqueeze(dim=1)
        self.value[idx_cur, 0] = torch.matmul(
            torch.matmul(pi_1, matrix_1), pi_2
        )
        pi_1, pi_2 = solution_2[: self.tree.max_actions].unsqueeze(dim=0), solution_2[
            self.tree.max_actions :
        ].unsqueeze(dim=1)
        self.value[idx_cur, 1] = -torch.matmul(
            torch.matmul(pi_1, matrix_2), pi_2
        )
        # Get payoff that serves as new values


if __name__ == '__main__':
    tree = game.Tree(
        max_actions=2,
        depth_bound=2,
    )
    tree.generate()
    net = net.MLP(size=tree.max_actions, width=128)

    search = Search(tree)
    search.apply_net(net)
    for steps in range(3):
        for idx, _ in enumerate(search.value):
            print(idx, _)
        print()
        search.step()