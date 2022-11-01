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

def nash_conv (tree : game.Tree, net : net.ConvNet, inference_batch_size=10**5) :

    net.eval()

    tree_size = tree.value.shape[0]
    policy = torch.zeros((tree_size, 2 * tree.max_actions), device=torch.device('cpu'))

    for _ in range(tree_size // inference_batch_size + 1):
        slice_range = torch.arange(_ * inference_batch_size, min((_+1) * inference_batch_size, tree_size), device=tree.device)
        value_slice = tree.expected_value[slice_range]
        legal_slice = tree.legal[slice_range]

        inference_slice = torch.cat([value_slice, legal_slice], dim=1)
        with torch.no_grad():
            policy[slice_range, :tree.max_actions] = net.forward_policy(inference_slice).to(torch.device('cpu'))
            inference_slice = torch.cat([-value_slice, legal_slice], dim=1).swapaxes(2, 3)
            policy[slice_range, tree.max_actions:] = net.forward_policy(inference_slice).to(torch.device('cpu'))

    net.train()
    print(policy[1])
    max_1, min_2 = max_min(tree, policy)

    exploitability = max_1 - min_2
    return exploitability

def max_min (tree : game.Tree, policy : torch.Tensor, root_index=1, depth=0):

    matrix_1 = torch.zeros((tree.max_transitions * tree.max_actions**2,))
    matrix_2 = torch.zeros((tree.max_transitions * tree.max_actions**2,))

    value_root = torch.flatten(tree.value[root_index:root_index+1]) #(1, n_trans, max_actions, max_actions)
    index_root = torch.flatten(tree.index[root_index:root_index+1]) #(1, n_trans, max_actions, max_actions)
    chance_root = torch.flatten(tree.chance[root_index:root_index+1] * tree.legal[root_index:root_index+1])

    places = (chance_root > 0).nonzero()
    for idx_flat in places:

        transition_prob = chance_root[idx_flat]
        root_index_ = index_root[idx_flat[0]]
        if root_index_ == 0:
            v = value_root[idx_flat]
            max_1_, min_2_ = v, v
        else:
            max_1_, min_2_ = max_min (tree, policy, root_index_, depth=depth+1)

        matrix_1[idx_flat] = (max_1_ * transition_prob).to(torch.float)
        matrix_2[idx_flat] = (min_2_ * transition_prob).to(torch.float)

    matrix_1 = matrix_1.view(tree.max_transitions, tree.max_actions, tree.max_actions)
    matrix_2 = matrix_2.view(tree.max_transitions, tree.max_actions, tree.max_actions)
    matrix_1 = torch.sum(matrix_1, dim=0)
    matrix_2 = torch.sum(matrix_2, dim=0)

    pi = policy[root_index]

    pi_1, pi_2 = pi[:tree.max_actions].unsqueeze(dim=0), pi[tree.max_actions:].unsqueeze(dim=1)

    row_actions_mask = tree.legal[root_index, 0, :, 0].to(torch.bool)
    col_actions_mask = tree.legal[root_index, 0, 0, :].to(torch.bool)
    prod_1 = torch.flatten(torch.matmul(matrix_1, pi_2))[row_actions_mask]
    prod_2 = torch.flatten(torch.matmul(pi_1, matrix_2))[col_actions_mask]
    max_1 = torch.max(prod_1).item()
    min_2 = torch.min(prod_2).item()

    return max_1, min_2


    

if __name__ == '__main__' :
    import game
    import net
    tree = game.Tree()
    tree.load('1667264620')
    tree._assert_index_is_tree()
    # net = net.ConvNet(tree.max_actions, 1, 1)
    # expl = nash_conv(tree, net, 1000)
    # print(expl)
    pi_1,pi_2 = tree.legal[:, 0, :, 0], tree.legal[:, 0, 0, :]
    pi_1 = torch.nn.functional.normalize(pi_1, dim=1, p=1)
    pi_2 = torch.nn.functional.normalize(pi_2, dim=1, p=1)
    pi = torch.cat([pi_1, pi_2], dim=1)
    print(pi[:10])
    print(max_min(tree, pi, 1))