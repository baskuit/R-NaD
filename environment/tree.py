import torch
import torch.nn as nn
import torch.nn.functional as F

import pygambit
import numpy as np
import random
import os
import time
import logging

from typing import List, Dict

"""
A constructed game tree is described by several tensors.

For all tensors:
- the index 0 < s < tree_size at dim=0 corresponds to a state in the abstract game tree
- the index 0 < t < max_trasitions is the chance players action
- the indices 0 < r, c < max_actions are the row and column players actions respectively


index:
shape=(s, t, r, c)
    This tensor describes the topology of the tree. Each matrix node on that tree corresponds to an idx for dim=0
    The 3 dimensional tensor at any idx tells us the location of the next state to transition to after the actions of
    the row player, column player, and chance player, respectively.

    The state at idx=0 is an absorbing state. It represents all the terminal states/leaf nodes of the abstract tree simultaneously.
    This greatly simplifies the logic for batched playouts. A single playout that has concluded can still be handled just like the ongoing playouts,
    it is simply stuck at idx=0, since all the entries for the 3D index tensor at idx=0 are also 0.
    The state at idx=1 is the root state. Thus the non-zero entries of the index tensor are 1-to-1 with [2, max_idx].

    Not all entries of this tensor correspond to valid state transitions. For a coordinate of (idx, row_action, col_action, chance_action),
    only those where row_action < the number of legal row actions for that state (ditto fro column player) and the entry of the "chance" tensor is > 0
    are valid. 

value:
shape=(s, t, r, c)
    This tensor has the same shape as the index tensor. At an entry that corresponds to a transtition to a terminal state,
    the value is the reward for the row player upon reaching that state. The Trees are zero sum games, so the column players reward is simply the negative.

    For any other index, the value at an entry of this tensor is the (row player's) 
    nash equilibrium payoff of the state that the corresponding entry of the index tensor points to.

chance:
shape=(s, t, r, c)
    The values are the strategy of the chance player, or the likelihood of transitioning to another state
    Thus the sum of the matrix over dim=1 is always 1, at least for valid row and column player actions.

expected_value:
shape=(s, 1, r, c)
    This tensor is simply the weighted sum of the the value tensor with respect to the chance players strategy.

legal:
shape=(s, 1, r, c)
    A simple bit mask for the legal actions of the row and column players.

solution:
shape (s, 2 * max_actions)
    Computing value for non terminal states means recursively solving the tree as we generate it.
    [s, :max_actions] is the row players strategy, [s, max_actions:] is the column players.
"""


class Tree:
    def __init__(
        self,
        is_root=True,
        device=torch.device("cpu"),
        max_actions=3,
        max_transitions=1,
        row_actions=None,
        col_actions=None,
        depth_bound=1,
        row_actions_lambda=None,
        col_actions_lambda=None,
        depth_bound_lambda=None,
        transition_threshold=0,
        terminal_values=(-1, 1),
        desc="",
    ):
        """
        is_root:
            The tree is constructed recursively. First the information at the root state is determined.
            Then each subsequent state is treated as its own tree, and the resulting tensors of all the child trees are joined
            True if the Tree is the actual root, False otherwise. This is needed so we only join the absorbing index at the last step.
        max_actions:
            The maximum number of actions for both row and column player
        max_transitions:
            The maximum number of actions for the chance player
        row_actions,
        col_actions:
            The number of legal actions for the players at the root state of a sub-tree.
        depth_bound:
            The maximum depth a sub tree is allowed to grow
        row_actions_lambda,
        col_actions_lambda,
        depth_bound_lambda:
            These functions determine the previous 3 parameters based for a child subtree using the corresponding
            parameters for the parent subtree.
            The default functions keep row_actions and col_actions the same, and decrement the depth_bound by 1.
            This produces a very 'regular' tree topology.
        transition_threshold:
            When a new chance player strategy is created for each pair of row and column player actions,
            any chance action which has probability less than this is zeroed. The resulting distribution is renormalized.
        terminal_values:
            When the value at a terminal state is assigned, this collection is sampled uniformly.
            This permits values like (1, -1, -1) to create a tree that favors that column player, for example.
        desc:
            Since large trees can take a while to generate, we include functionality for saving them.
            This string is intended to be a verbose descriptor for convenience.
        """

        self.is_root = is_root
        self.device = device
        self.max_actions = max_actions
        self.max_transitions = max_transitions
        self.row_actions = row_actions if row_actions is not None else max_actions
        self.col_actions = col_actions if col_actions is not None else max_actions
        self.depth_bound = depth_bound
        self.transition_threshold = transition_threshold
        self.terminal_values = terminal_values

        value_shape = (1, max_transitions, max_actions, max_actions)
        legal_shape = (1, 1, max_actions, max_actions)
        nash_shape = (1, max_actions * 2)

        self.index_tensor = torch.zeros(value_shape, device=device, dtype=torch.long)
        self.value_tensor = torch.zeros(value_shape, device=device, dtype=torch.float)
        self.expected_value_tensor = torch.zeros(legal_shape, device=device, dtype=torch.float)
        self.legal_tensor = torch.zeros(legal_shape, device=device, dtype=torch.float)
        self.legal_tensor[0, 0, : self.row_actions, : self.col_actions] = 1.0
        self.chance_tensor = self._transition_probs(
            max_actions, max_actions, max_transitions, transition_threshold
        )
        self.chance_tensor *= self.legal_tensor
        self.root_value_tensor = torch.zeros((1, 1), device=device, dtype=torch.float)
        # NE payoff for the subtree
        self.solution_tensor = torch.zeros(nash_shape, device=device, dtype=torch.float)
        self.desc = desc
        self.hash = 0
        # Unique identifier for a tree, so as not to mix up trees with nets that were trained on them.

        self.saved_keys = list(self.__dict__.keys())
        # Only the keys above this are saved and reloaded.

        self.row_actions_lambda = (
            row_actions_lambda
            if row_actions_lambda is not None
            else lambda tree: tree.row_actions
        )
        self.col_actions_lambda = (
            col_actions_lambda
            if col_actions_lambda is not None
            else lambda tree: tree.col_actions
        )
        self.depth_bound_lambda = (
            depth_bound_lambda
            if depth_bound_lambda is not None
            else lambda tree: tree.depth_bound - 1
        )

    def _init_child(self):
        child = Tree(
            is_root=False,
            device=self.device,
            max_actions=self.max_actions,
            max_transitions=self.max_transitions,
            row_actions=min(self.max_actions, max(1, self.row_actions_lambda(self))),
            col_actions=min(self.max_actions, max(1, self.col_actions_lambda(self))),
            # 1 <= row_actions, col_actions <= max_actions
            depth_bound=max(0, self.depth_bound_lambda(self)),
            transition_threshold=self.transition_threshold,
            terminal_values=self.terminal_values,
            row_actions_lambda=self.row_actions_lambda,
            col_actions_lambda=self.col_actions_lambda,
            depth_bound_lambda=self.depth_bound_lambda,
        )
        return child

    def _transition_probs(self, rows, cols, n_trans, transition_threshold):
        """
        Generates a random transition probabilty tensor.
        It does not respect legality of row and column player actions yet, it is masked later.
        """
        chance = (
            torch.from_numpy(
                np.random.dirichlet((1 / n_trans,) * n_trans, (1, rows, cols))
            )
            .to(self.device)
            .to(torch.float)
        )
        chance = chance - torch.where(chance < transition_threshold, chance, 0)
        chance = torch.nn.functional.normalize(chance, p=1, dim=3)
        chance = chance.movedim(3, 1)
        return chance

    def _solve(self, M: torch.Tensor, max_actions=2):
        """
        M : tensor of shape (rows, cols)
        output : Joint NE strategy tensor of shape (2 * max_actions)
        """
        rows, cols = M.shape[:2]
        pygambit_matrix = np.zeros((rows, cols), dtype=pygambit.Decimal)
        for _ in range(rows):
            for __ in range(cols):
                pygambit_matrix[_][__] = pygambit.Decimal(M[_][__].item())
        # iirc pygambit is kinda finicky. There may be a nicer way to do this.

        g = pygambit.Game.from_arrays(pygambit_matrix, -pygambit_matrix)

        solutions = [
            [S[_] if _ < rows else 0 for _ in range(max_actions)]
            + [S[rows + _] if _ < cols else 0 for _ in range(max_actions)]
            for S in pygambit.nash.enummixed_solve(g, rational=False)
        ]
        if not solutions:
            # pygambit failed to solve for some reason
            solutions = [
                [S[_] if _ < rows else 0 for _ in range(max_actions)]
                + [S[rows + _] if _ < cols else 0 for _ in range(max_actions)]
                for S in pygambit.nash.lcp_solve(g, rational=False)
            ]
        # I don't believe this has failed to solve a matrix, but I need to write tests TODO.

        purity_score = lambda solution: -int(1 in solution[:max_actions]) - int(
            1 in solution[max_actions:]
        )
        solutions.sort(key=purity_score)
        # We favor mixed over pure strategies.
        # Besides R-NaD we can also train a network using the de facto NE strategies and payoffs as labels.
        # This is useful for setting a benchmark to compare R-NaD to
        return torch.tensor(solutions, dtype=torch.float).to(self.device)

    def generate(self):
        """
        Generate subtrees and later join their tensor data to this tree's.
        """
        children: List[Tree] = []
        subtree_sizes: List[int] = []
        # used for index tensor adjustment later

        """
        Iterate through pair of row and column player actions. 
        For each pair, generate a strategy profile for the chance player.

        For each non-zero entry of that profile, we have a new subtree.

        If the resulting depth_bound for the subtree = 0, then that subtree corresponds to a terminal state.
        Since terminal states do not get indices in the tensors, those are handled differently.
        """
        for row in range(self.row_actions):
            for col in range(self.col_actions):
                for chance in range(self.max_transitions):
                    transition_prob = self.chance_tensor[0, chance, row, col]

                    if transition_prob > 0:
                        child = self._init_child()

                        if (
                            child.depth_bound > 0
                            and child.row_actions * child.col_actions > 0
                        ):
                            child.generate()
                            children.append(child)
                            subtree_sizes.append(child.value_tensor.shape[0])
                            self.index_tensor[0, chance, row, col] = 1
                            child_payoff = child.root_value_tensor[:1]

                        else:
                            subtree_sizes.append(0)
                            child_payoff = torch.tensor(
                                ((random.choice(self.terminal_values),),)
                            ).to(self.device)

                        self.value_tensor[0, chance, row, col] = child_payoff.item()
                        # Set the entry for the parents value tensor

                self.expected_value_tensor[0, 0, row, col] = torch.sum(
                    self.value_tensor[0, :, row, col] * self.chance_tensor[0, :, row, col]
                )
                # Calculate expected value for the parents tensor for the joint player actions.

        game_matrix = self.expected_value_tensor[0, 0, : self.row_actions, : self.col_actions]
        game_solutions = self._solve(game_matrix, self.max_actions)
        if len(game_solutions) == 0:
            raise Exception(
                f"Game matrix not solved by pygambit: {game_matrix.tolist()}"
            )
        # Solve expected value matrix to get self.root_value and self.solution

        joint_strategy = game_solutions[0]
        p1_strategy = joint_strategy[: self.row_actions].unsqueeze(dim=0)
        p2_strategy = joint_strategy[self.max_actions :][: self.col_actions].unsqueeze(
            dim=1
        )
        self.root_value_tensor = torch.matmul(
            torch.matmul(p1_strategy, game_matrix), p2_strategy
        )
        self.solution_tensor = joint_strategy.unsqueeze(dim=0)

        """
        Below we adjust the index tensor entries for the child subtrees and the parent (at the index).
 
        If we allowed terminal states to have an index in the tree then I'm sure this code would be simpler.
        That however would increase the size of the tensors (at dim=0) significantly.

        I am not interested in explaining this code, sorry.
        """
        total_length = 1
        for child in children:
            index_adjustment = child.index_tensor.clone()
            index_adjustment[index_adjustment > 0] = 1.0
            index_adjustment *= total_length
            child.index_tensor += index_adjustment
            total_length += child.value_tensor.shape[0]
        # update index tensors for subtrees

        transition_idx = 0
        total_sum = 0
        subtree_sizes.insert(0, 1)
        for row in range(self.row_actions):
            for col in range(self.col_actions):
                for chance in range(self.max_transitions):
                    if self.chance_tensor[0, chance, row, col] != 0:
                        total_sum += subtree_sizes[transition_idx]
                        self.index_tensor[0, chance, row, col] *= total_sum
                        transition_idx += 1
        # update index tensor for parent's root

        """
        Concatenate the tensor data of the parent and child subtrees. 
        If the tree is_root, then we also append the absorbing state and adjust accordingly.
        """
        prefix = [self]
        if self.is_root:
            absorbing_state = Tree(
                is_root=False,
                device=self.device,
                max_actions=self.max_actions,
                max_transitions=self.max_transitions,
                depth_bound=0,
                row_actions=1,
                col_actions=1,
            )
            absorbing_state.chance_tensor[0, :, 0, 0] = 0
            absorbing_state.chance_tensor[0, 0, 0, 0] = 1
            prefix.insert(0, absorbing_state)

        for key in [
            "index_tensor",
            "value_tensor",
            "expected_value_tensor",
            "chance_tensor",
            "legal_tensor",
            "root_value_tensor",
            "solution_tensor",
        ]:
            self.__dict__[key] = torch.cat(
                tuple(tree.__dict__[key] for tree in prefix + children), dim=0
            )

        if self.is_root:
            self.index_tensor += self.index_tensor != 0
            self.hash = torch.randint(-(2**63), 2**63 - 1, size=(1,)).item()

    def assert_index_is_tree(self):
        """
        The index tensor describes a tree if and only if it is increasing,
        i.e. the non-zero values in the index[idx] are all > idx,
        and the non-zero values in index are one-to-one with some interval [1 + is_root, x]
        """
        indices = self.index_tensor[self.index_tensor != 0]
        indices = indices.tolist()
        indices.sort()
        all_possible_indices = list(range(1 + self.is_root, 2 + len(indices)))  #
        assert indices == all_possible_indices
        for idx, index_slice in enumerate(self.index_tensor[1:]):
            greater_than_idx = index_slice > idx
            is_zero = index_slice == 0
            is_valid = torch.logical_or(greater_than_idx, is_zero)
            assert torch.all(is_valid)

    def save(self, directory_name=None):
        """
        Save tree data to 
            ./saved_trees/directory_name/tree.tar
        as well as 
            ./saved_trees/recent/tree.tar

        If no directory_name is given then we simply use the UTC time
        """
        if not self.is_root:
            raise Exception("Attempting to save non-root tree")

        directory = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "saved_trees"
        )
        if not os.path.exists(directory):
            os.mkdir(directory)

        if directory_name is None:
            directory_name = str(int(time.time()))
        path = os.path.join(directory, directory_name)
        if not os.path.exists(path):
            os.mkdir(path)
        recent_path = os.path.join(directory, "recent")
        if not os.path.exists(recent_path):
            os.mkdir(recent_path)

        saved_dict = {key: self.__dict__[key] for key in self.saved_keys}
        torch.save(saved_dict, os.path.join(recent_path, "tree.tar"))
        torch.save(saved_dict, os.path.join(path, "tree.tar"))
        logging.info("saving trees to '{}' and 'recent'".format(path))

    def load(self, directory_name="recent"):
        """
        Overwrites this tree's data with data from
            ./directory_name/tree.tar
        """
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "saved_trees",
            directory_name,
            "tree.tar",
        )
        logging.info("loading tree from '{}'".format(directory_name))
        dict: Dict = torch.load(path)
        for key, value in dict.items():
            self.__dict__[key] = value
        logging.info("loaded tree has hash {}".format(self.hash))

    def to(self, device):
        """
        Move all member tensors to device, and update the 'device' member.
        """
        self.device = device
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                self.__dict__[key] = value.to(device)