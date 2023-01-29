import torch
import torch.nn as nn
import torch.nn.functional as F

import game
import net

"""
Efficiently calculate the exploitabilty of a networks policy on a game tree
root_index is the matrix in tree.expected_value for which to compute expoloitability
inference_batch_size is for splitting very large games into chunks that fit into GPU memory

By default we store game-wide tensors on cpu
"""


class NashConvData:
    def __init__(self, tree: game.Tree):
        self.size = tree.value.shape[0]
        self.policy = torch.zeros(
            (self.size, 2 * tree.max_actions), device=tree.device, dtype=torch.float
        )
        self.max_1 = torch.zeros((self.size,), device=tree.device, dtype=torch.float)
        self.min_2 = torch.zeros((self.size,), device=tree.device, dtype=torch.float)
        self.depth = torch.zeros((self.size,), device=tree.device, dtype=torch.int)

    def to(self, device):
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                self.__dict__[key] = value.to(device)


def nash_conv(tree: game.Tree, net: net.ConvNet, inference_batch_size=10**5):

    data = NashConvData(tree)

    net.eval()
    for _ in range(data.size // inference_batch_size + 1):
        slice_range = torch.arange(
            _ * inference_batch_size,
            min((_ + 1) * inference_batch_size, data.size),
            device=tree.device,
        )
        value_slice = tree.expected_value[slice_range]
        legal_slice = tree.legal[slice_range]

        with torch.no_grad():
            inference_slice = torch.cat([value_slice, legal_slice], dim=1)
            data.policy[slice_range, : tree.max_actions] = net.forward_policy(
                inference_slice
            )
            inference_slice = torch.cat([-value_slice, legal_slice], dim=1).swapaxes(
                2, 3
            )
            data.policy[slice_range, tree.max_actions :] = net.forward_policy(
                inference_slice
            )
    tree.to(torch.device("cpu"))
    data.to(torch.device("cpu"))
    max_min(tree, data)
    tree.to(net.device)
    data.to(net.device)

    net.train()
    return data


def max_min(tree: game.Tree, data: NashConvData, root_index=1, depth=0):

    matrix_1 = torch.zeros(
        (tree.max_transitions * tree.max_actions**2,),
        device=tree.device,
        dtype=torch.float,
    )
    matrix_2 = torch.zeros(
        (tree.max_transitions * tree.max_actions**2,),
        device=tree.device,
        dtype=torch.float,
    )

    value_root = torch.flatten(
        tree.value[root_index : root_index + 1]
    )  # (1, n_trans, max_actions, max_actions)
    index_root = torch.flatten(
        tree.index[root_index : root_index + 1]
    )  # (1, n_trans, max_actions, max_actions)
    chance_root = torch.flatten(
        tree.chance[root_index : root_index + 1]
        * tree.legal[root_index : root_index + 1]
    )

    places = (chance_root > 0).nonzero()
    depths = [0]
    for idx_flat in places:

        transition_prob = chance_root[idx_flat]
        root_index_ = index_root[idx_flat[0]]
        if root_index_ == 0:
            v = value_root[idx_flat]
            max_1_, min_2_ = v, v
        else:
            max_min(tree, data, root_index_, depth=depth + 1)
            max_1_ = data.max_1[root_index_]
            min_2_ = data.min_2[root_index_]
            depths.append(data.depth[root_index_])
        matrix_1[idx_flat] = max_1_ * transition_prob
        matrix_2[idx_flat] = min_2_ * transition_prob

    matrix_1 = matrix_1.view(tree.max_transitions, tree.max_actions, tree.max_actions)
    matrix_2 = matrix_2.view(tree.max_transitions, tree.max_actions, tree.max_actions)
    matrix_1 = torch.sum(matrix_1, dim=0)
    matrix_2 = torch.sum(matrix_2, dim=0)

    pi = data.policy[root_index]
    pi_1, pi_2 = pi[: tree.max_actions].unsqueeze(dim=0), pi[
        tree.max_actions :
    ].unsqueeze(dim=1)
    row_actions_mask = tree.legal[root_index, 0, :, 0].to(torch.bool)
    col_actions_mask = tree.legal[root_index, 0, 0, :].to(torch.bool)
    prod_1 = torch.flatten(torch.matmul(matrix_1, pi_2))[row_actions_mask]
    prod_2 = torch.flatten(torch.matmul(pi_1, matrix_2))[col_actions_mask]
    data.max_1[root_index] = torch.max(prod_1)
    data.min_2[root_index] = torch.min(prod_2)

    depth = 1 + max(depths)
    data.depth[root_index] = depth


def mean_nash_conv_by_depth(data: NashConvData):

    max_depth = int(data.depth[1].item())
    nash_conv = data.max_1 - data.min_2
    means = {}
    for depth in range(1, max_depth + 1):
        idx = data.depth == depth
        expls = nash_conv[idx]
        means[depth] = torch.mean(expls).item()
    return means


def kld(
    p: torch.Tensor,
    q: torch.Tensor,
    valid: torch.Tensor,
    legal_actions: torch.Tensor,
    valid_count: int = None,
):
    if valid_count is None:
        valid_count = valid.sum().item()
    return (
        torch.where(
            (valid.unsqueeze(-1) * legal_actions).to(torch.bool), p * (torch.log(p) - torch.log(q)), 0
        )
        .sum()
        .item()
        / valid_count
    )


if __name__ == "__main__":
    import game
    import net

    tree = game.Tree(
        depth_bound=4,
        max_actions=3,
    )
    # tree.load('1667264620')
    tree.generate()
    tree.assert_index_is_tree()
    net_ = net.ConvNet(tree.max_actions, 1, 1)

    # pi = tree.nash
    # data = NashConvData(tree)
    # data.policy = pi
    # print(tree.chance.dtype)
    # max_min(tree, data)
    # print(data.max_1[1] - data.min_2[1])

    data = nash_conv(tree, net_)
    means = mean_nash_conv_by_depth(data)
    for _, __ in means.items():
        print(_, __)
