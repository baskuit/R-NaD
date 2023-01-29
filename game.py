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
        transition_threshold=0,
        terminal_values=(-1, 1),
        row_actions_lambda=None,
        col_actions_lambda=None,
        depth_bound_lambda=None,
        desc="",
    ):

        self.is_root = is_root
        self.device = device
        self.max_actions = max_actions
        self.max_transitions = max_transitions
        self.row_actions = row_actions if row_actions is not None else max_actions
        self.col_actions = col_actions if col_actions is not None else max_actions
        self.depth_bound = depth_bound
        self.transition_threshold = transition_threshold
        self.terminal_values = terminal_values
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

        value_shape = (1, max_transitions, max_actions, max_actions)
        legal_shape = (1, 1, max_actions, max_actions)
        nash_shape = (1, max_actions * 2)

        self.index = torch.zeros(value_shape, device=device, dtype=torch.long)
        self.value = torch.zeros(value_shape, device=device, dtype=torch.float)
        self.expected_value = torch.zeros(legal_shape, device=device, dtype=torch.float)
        self.legal = torch.zeros(legal_shape, device=device, dtype=torch.float)
        self.legal[0, 0, : self.row_actions, : self.col_actions] = 1.0
        self.chance = self._transition_probs(
            max_actions, max_actions, max_transitions, transition_threshold
        )
        self.chance *= self.legal
        self.root_value = torch.zeros((1, 1), device=device, dtype=torch.float)
        self.solution = torch.zeros(nash_shape, device=device, dtype=torch.float)
        self.desc = desc
        self.hash = 0

    def _init_child(self):
        child = Tree(
            is_root=False,
            device=self.device,
            max_actions=self.max_actions,
            max_transitions=self.max_transitions,
            row_actions=min(self.max_actions, max(1, self.row_actions_lambda(self))),
            col_actions=min(self.max_actions, max(1, self.col_actions_lambda(self))),
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
        Generates a random transition probabilty tensor. (It does not respect legality of moves, it is masked later)
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
        max_actions : dim=0 of output is padded to 2 * max_actions
        output : tensor of shape (2 * max_actions)
        """
        rows, cols = M.shape[:2]
        pygambit_matrix = np.zeros((rows, cols), dtype=pygambit.Decimal)
        for _ in range(rows):
            for __ in range(cols):
                pygambit_matrix[_][__] = pygambit.Decimal(M[_][__].item())
        # iirc pygambit is kinda finicky

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
        # this seems to be pretty safe

        purity_score = lambda solution: -int(1 in solution[:max_actions]) - int(
            1 in solution[max_actions:]
        )
        solutions.sort(key=purity_score)
        # we choose to return interior strategies if possible
        return torch.tensor(solutions, dtype=torch.float).to(self.device)

    def generate(self):
        children: List[Tree] = []
        subtree_sizes: List[int] = []
        """
        If we allowed Trees to have size 0, representing a terminal node, we could simplify this code.
        That would effectively increase tree depth by 1.
        So instead, "subtree" encompasses terminal nodes while "child" does not
        """

        for row in range(self.row_actions):
            for col in range(self.col_actions):
                for chance in range(self.max_transitions):
                    transition_prob = self.chance[0, chance, row, col]

                    if transition_prob > 0:
                        child = self._init_child()

                        if (
                            child.depth_bound > 0
                            and child.row_actions * child.col_actions > 0
                        ):
                            child.generate()
                            children.append(child)
                            subtree_sizes.append(child.value.shape[0])
                            self.index[0, chance, row, col] = 1
                            child_payoff = child.root_value[:1]

                        else:
                            subtree_sizes.append(0)
                            child_payoff = torch.tensor(
                                ((random.choice(self.terminal_values),),)
                            ).to(self.device)

                        self.value[0, chance, row, col] = child_payoff.item()

                self.expected_value[0, 0, row, col] = torch.sum(
                    self.value[0, :, row, col] * self.chance[0, :, row, col]
                )

        # Solve expected value matrix to get root_value and solution
        game_matrix = self.expected_value[0, 0, : self.row_actions, : self.col_actions]
        game_solutions = self._solve(game_matrix, self.max_actions)
        if len(game_solutions) == 0:
            raise Exception(
                f"Game matrix not solved by pygambit: {game_matrix.tolist()}"
            )

        joint_strategy = game_solutions[0]
        p1_strategy = joint_strategy[: self.row_actions].unsqueeze(dim=0)
        p2_strategy = joint_strategy[self.max_actions :][: self.col_actions].unsqueeze(
            dim=1
        )
        self.root_value = torch.matmul(
            torch.matmul(p1_strategy, game_matrix), p2_strategy
        )
        self.solution = joint_strategy.unsqueeze(dim=0)

        # Update index of children since they will be concatenated
        total_length = 1
        for child in children:
            index_adjustment = child.index.clone()
            index_adjustment[index_adjustment > 0] = 1.0
            index_adjustment *= total_length
            child.index += index_adjustment
            total_length += child.value.shape[0]

        # Update root index to point to children
        transition_idx = 0
        total_sum = 0
        subtree_sizes.insert(0, 1)
        for row in range(self.row_actions):
            for col in range(self.col_actions):
                for chance in range(self.max_transitions):
                    if self.chance[0, chance, row, col] != 0:
                        total_sum += subtree_sizes[transition_idx]
                        self.index[0, chance, row, col] *= total_sum
                        transition_idx += 1

        # Concatenate tensors
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
            absorbing_state.chance[0, :, 0, 0] = 0
            absorbing_state.chance[0, 0, 0, 0] = 1
            prefix.insert(0, absorbing_state)

        for key in [
            "index",
            "value",
            "expected_value",
            "chance",
            "legal",
            "root_value",
            "solution",
        ]:
            self.__dict__[key] = torch.cat(
                tuple(tree.__dict__[key] for tree in prefix + children), dim=0
            )

        if self.is_root:
            self.index += self.index != 0
            self.hash = torch.randint(-(2**63), 2**63 - 1, size=(1,)).item()

    def assert_index_is_tree(self):
        """
        The index tensor describes a tree if and only if it is increasing,
        i.e. the non-zero values in the index[idx] are all > idx,
        and the non-zero values in index are one-to-one with some interval [1 + is_root, x]
        """
        indices = self.index[self.index != 0]
        indices = indices.tolist()
        indices.sort()
        all_possible_indices = list(range(1 + self.is_root, 2 + len(indices)))  #
        assert indices == all_possible_indices
        for idx, index_slice in enumerate(self.index[1:]):
            greater_than_idx = index_slice > idx
            is_zero = index_slice == 0
            is_valid = torch.logical_or(greater_than_idx, is_zero)
            assert torch.all(is_valid)

    def save(self, directory_name=None):
        if not self.is_root:
            raise Exception("Attempting to save non-root tree")

        directory = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "saved_trees"
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

        torch.save(self.__dict__, os.path.join(recent_path, "tree.tar"))
        torch.save(self.__dict__, os.path.join(path, "tree.tar"))
        logging.info("saving trees to '{}' and 'recent'".format(path))

    def load(self, directory_name="recent"):
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "saved_trees", "tree.tar"
        )
        logging.info("loading tree from '{}'".format(directory_name))
        dict: Dict = torch.load(path)
        for key, value in dict.items():
            self.__dict__[key] = value
        logging.info("loaded tree has hash {}".format(self.hash.item()))

    def to(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                self.__dict__[key] = value.to(device)


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    tree = Tree(
        max_actions=2,
        max_transitions=2,
        transition_threshold=0.4,
        # row_actions_lambda=lambda tree:tree.row_actions - 1 * (random.random() < .2),
        # col_actions_lambda=lambda tree:tree.row_actions - 1 * (random.random() < .2),
        # row_actions_lambda=lambda tree:3,
        # col_actions_lambda=lambda tree:3,
        # row_actions=2,
        # col_actions=2,
        depth_bound_lambda=lambda tree: tree.depth_bound
        - 1
        - 2 * (random.random() < 0.5),
        # depth_bound_lambda=lambda tree:tree.depth_bound - 2,
        depth_bound=5,
        # desc='3x3 but 2x2 at root'
    )

    tree.generate()
    for _ in tree.index:
        print(_)
    tree.assert_index_is_tree()
