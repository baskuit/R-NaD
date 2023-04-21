import torch

from ..environment.tree import Tree
from ..util.metric import NashConvData

# import environment.tree
# import util.metric

"""
Test to verify that the NashConv of a tree solved strategy is zero.
"""

def test_nashconv (max_actions: int, max_transitions: int, depth: int,):

    tree = Tree(
        max_actions=max_actions,
        max_transitions=max_transitions,
        depth_bound=depth,
    )
    tree.generate()

    data = NashConvData(tree)
    data.get_nashconv(tree, tree.solution_tensor)

    nashconv = (data.row_best[1] + data.col_best[1]).item()
    total_reach_probablity = torch.sum(data.reach_probability).item()

    assert(nashconv == 0)
    assert(total_reach_probablity == 2)


if __name__ == '__main__':

    for depth in range(2, 7):
        nashconv = test_nashconv(depth, 1, 3)
        print(nashconv)
