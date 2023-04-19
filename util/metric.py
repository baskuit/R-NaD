import torch
import torch.nn as nn
import torch.nn.functional as F

import environment.tree as tree
import nn.net as net

from typing import Dict, List

class NashConvData:

    """
    Efficiently calculate the exploitabilty of a networks policy on a game tree
    root_index is the matrix in tree.expected_value for which to compute expoloitability
    inference_batch_size is for splitting very large games into chunks that fit into GPU memory

    By default we store game-wide tensors on cpu
    """

    def __init__(self, tree: tree.Tree):
        self.size = tree.value_tensor.shape[0]
        self.joint_policy = torch.zeros(
            (self.size, 2 * tree.max_actions), device=tree.device, dtype=torch.float
        )
        self.row_best = torch.zeros((self.size,), device=tree.device, dtype=torch.float)
        self.col_best = torch.zeros((self.size,), device=tree.device, dtype=torch.float)
        self.reach_probability = torch.zeros((self.size,), device=tree.device, dtype=torch.float)
        self.depth = torch.zeros((self.size,), device=tree.device, dtype=torch.int)

        """
        joint_policy:
            The entire tree is inferenced and the policy for both players is stored here
        row_best:
            Assuming the column player is playing the network's strategy, this value is the expected payoff it can achieve if it plays a maximally exploiting strategy from this state onwards
        col_best:
            Likewise.
        reach_probablity:
            The probabilty of reaching a given state using the joint policy.
        depth:
            The longest distance to a terminal node from this state.
            While the NashConv for the game is simply that of the root state, we can also calculate the NashConv of every state since we store values for the entire tree.
            Then we can calculate the average NashConv for states of any given depth.
        """

    def to(self, device):
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                self.__dict__[key] = value.to(device)


    def get_nashconv_from_net(
        self, 
        tree: tree.Tree, 
        net: net.ConvNet, 
        inference_batch_size: int=10**5
    ) -> None:

        net.eval()

        for _ in range(self.size // inference_batch_size + 1):
            slice_range = torch.arange(
                _ * inference_batch_size,
                min((_ + 1) * inference_batch_size, self.size),
                device=tree.device,
            )
            value_slice = tree.expected_value_tensor[slice_range]
            legal_slice = tree.legal_tensor[slice_range]

            with torch.no_grad():

                observation = torch.cat([value_slice, legal_slice], dim=1)
                self.joint_policy[slice_range, : tree.max_actions] = net.forward_policy(
                    observation
                )
                # row player
                observation = torch.cat([-value_slice, legal_slice], dim=1).swapaxes(
                    2, 3
                )
                self.joint_policy[slice_range, tree.max_actions :] = net.forward_policy(
                    observation
                )
                # column player

        tree.to(torch.device("cpu"))
        self.to(torch.device("cpu"))
        self.get_nashconv(tree, self.joint_policy)
        tree.to(net.device)
        self.to(net.device)

        net.train()


    def get_nashconv(
        self, 
        tree: tree.Tree, 
        joint_policy: torch.Tensor,
        state_index: int=1,
        reach_probablity: float=1,
        depth: int=0
    ) -> None:
        
        """
        Calculate the NashConv of a joint policy with shape=(size, max_actions * 2)
        """

        row_best_case_matrix = torch.zeros(
            (tree.max_transitions * tree.max_actions**2,),
            device=tree.device,
            dtype=torch.float,
        )
        col_best_case_matrix = torch.zeros(
            (tree.max_transitions * tree.max_actions**2,),
            device=tree.device,
            dtype=torch.float,
        )

        pi = joint_policy[state_index]
        pi_row = pi[: tree.max_actions].unsqueeze(dim=0)
        pi_col = pi[tree.max_actions :].unsqueeze(dim=1)

        value_root = torch.flatten(
            tree.value_tensor[state_index : state_index + 1]
        )  # (1, n_trans, max_actions, max_actions)
        index_root = torch.flatten(
            tree.index_tensor[state_index : state_index + 1]
        )  # (1, n_trans, max_actions, max_actions)
        chance_root = torch.flatten(
            tree.chance_tensor[state_index : state_index + 1]
        )
        joint_policy_matrix_flat = torch.flatten(
            torch.matmul(pi_col, pi_row)
        ).repeat(tree.chance_tensor.shape[1])

        children = (chance_root > 0).nonzero()
        depths: List[int] = []
        for idx_flat in children:
            # idx_flat is not the tree index, but rather the index of flattened tensor

            transition_prob = chance_root[idx_flat]
            child_index = index_root[idx_flat].item()
            if child_index == 0:

                v = value_root[idx_flat]
                row_b, col_b = v, -v

            else:

                self.get_nashconv(
                    tree, 
                    self.joint_policy, 
                    state_index=child_index, 
                    reach_probablity=reach_probablity * joint_policy_matrix_flat[idx_flat] * transition_prob, 
                    depth=depth + 1,
                )
                row_b = self.row_best[child_index]
                col_b = self.col_best[child_index]
                depths.append(self.depth[child_index])

            row_best_case_matrix[idx_flat] = row_b * transition_prob
            col_best_case_matrix[idx_flat] = col_b * transition_prob

        row_best_case_matrix = row_best_case_matrix.view(tree.max_transitions, tree.max_actions, tree.max_actions)
        col_best_case_matrix = col_best_case_matrix.view(tree.max_transitions, tree.max_actions, tree.max_actions)
        row_best_case_matrix = torch.sum(row_best_case_matrix, dim=0)
        col_best_case_matrix = torch.sum(col_best_case_matrix, dim=0)

        row_actions_mask = tree.legal_tensor[state_index, 0, :, 0].to(torch.bool)
        col_actions_mask = tree.legal_tensor[state_index, 0, 0, :].to(torch.bool)
        row_responses = torch.flatten(torch.matmul(row_best_case_matrix, pi_col))[row_actions_mask]
        col_responses = torch.flatten(torch.matmul(pi_row, col_best_case_matrix))[col_actions_mask]

        self.row_best[state_index] = torch.max(row_responses)
        self.col_best[state_index] = torch.max(col_responses)
        self.reach_probability[state_index] = reach_probablity
        self.depth[state_index] = 1 + max(depths, default=0)


    def mean_nashconv_by_depth(self,) -> Dict[int, float]:

        max_depth = int(self.depth[1].item())
        nashconv = self.row_best + self.col_best
        # This exploits the fact that the game is zero-sum
        # Otherwise the calculcation is a tiny bit more involved

        means: Dict[int ,float] = {}
        for depth in range(1, max_depth + 1):
            idx = self.depth == depth
            expls = nashconv[idx]
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
            (valid.unsqueeze(-1) * legal_actions).to(torch.bool),
            p * (torch.log(p) - torch.log(q)),
            0,
        )
        .sum()
        .item()
        / valid_count
    )