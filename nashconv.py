from environment.tree import Tree
from util.metric import NashConvData

# import environment.tree
# import util.metric

"""
Test to verify that the NashConv of a tree solved strategy is zero.
"""

def test (max_actions: int, max_transitions: int, depth: int,):

    tree = Tree(
        max_actions=max_actions,
        max_transitions=max_transitions,
        depth_bound=depth,
    )
    tree.generate()

    data = NashConvData(tree)
    data.get_nashconv(tree, tree.solution)

    return (data.row_best[1] + data.col_best[1]).item()


if __name__ == '__main__':


    for depth in range(2, 7):
        nashconv = test(depth, 1, 3)
        print(nashconv)
