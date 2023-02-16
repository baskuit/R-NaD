import torch

from game import Tree

class CRFData ():

    def __init__(self, tree: Tree) -> None:
        self.tree = tree
        self.v = torch.zeros((tree.value.shape[0], 2))
        self.values = torch.zeros((tree.value.shape[0], 2, tree.max_actions))

        self.sigma = torch.zeros_like(self.values)
        self.regret = torch.zeros_like(self.values)

    def CRF (self, 
        idx_cur=1,
        i=1,
        t=1,
        pi_1=1,
        pi_2=1,
    ):

        self.v[idx_cur, i-1] = 0
        self.values[idx_cur, i-1, ...] = 0

        for a in range(self.tree.max_actions): # TODO change to legal actions

            





if __name__ == '__main__':

    tree = Tree(
        max_actions=2,
        depth_bound=3,
    )

