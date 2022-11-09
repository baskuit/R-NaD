import math
import random
import numpy

class Pool :

    """
    Refs is all the recent nets worth considering. Prune occasionaly

    Generates matchups to feed to the episodes.generate_versus() function
    the results of which update this pool

    we know the net is learning if  new nets are consistently dominanting the answer() distribution

    """

    def __init__ (self, refs:list, eta=.1):
        self.refs = refs
        self.logits = [0 for _ in refs]
        self.scores = [0 for _ in refs]
        self.selections = [0 for _ in refs]
        self.eta = eta
        self.size = len(refs)

    def _cap (self,):
        max_ = max(self.logits)
        self.logits = [_ - max_ for _ in self.logits]

    def filter (self, mask):
        self.refs = [_ for idx, _ in enumerate(self.refs) if mask[idx]]
        self.logits = [_ for idx, _ in enumerate(self.logits) if mask[idx]]
        self.scores = [_ for idx, _ in enumerate(self.scores) if mask[idx]]
        self.selections = [_ for idx, _ in enumerate(self.selections) if mask[idx]]
        self.size = sum(mask)

    def introduce (self, x):
        if x in self.refs:
            return
        # mean = sum(self.logits) / len(self.logits)
        self.refs.append(x)
        self.logits = [0 for _ in self.refs]
        self.scores = [0 for _ in self.refs]
        self.selections = [0 for _ in self.refs]
        self.size += 1
        # we clear reset estimate for NE when we introduce a new action
        # be we keep logits, so its reasonable to assume that the new NE is close

    def answer (self,):
        sum_ = max(1, sum(self.selections))
        empirical = [_ / sum_ for _ in self.selections]
        fixed = [(_ - self.eta / len(empirical)) / (1 - self.eta) for _ in empirical]
        return fixed
        

    def sample (self,) -> tuple:
        """
        We don't need to change the sample probs since the update for matching indices is zero in expectation
        """
        self._cap()
        exp_logits = [math.exp(_) for _ in self.logits]
        sum_ = sum(exp_logits)
        policy = [(1 - self.eta) * _/sum_  + self.eta / len(exp_logits) for _ in exp_logits]
        
        i, j = 0, 0
        if len(self.refs) > 1:
            i, j = numpy.random.choice(len(policy), 2, True, policy)
            self.selections[i] += 1
            self.selections[j] += 1

        p, q = policy[i], policy[j]
        return i, j, p, q

    def softmax (self,) -> tuple:
        self._cap()
        exp_logits = [math.exp(_) for _ in self.logits]
        sum_ = sum(exp_logits)
        policy = [_/sum_ for _ in exp_logits]
        return policy

    def update (self, i, j, p, q, u):
        self.logits[i] += u / p * self.eta
        self.logits[j] -= u / q * self.eta
        self.scores[i] += u
        self.scores[j] -= u

    def display (self,):
        print('Pool:')
        p = self.softmax()
        for _, logit, w, n, p in zip(self.refs, self.logits, self.scores, self.selections, p):
            u = w / n if n else -1
            print("{0: <4}: {1: <4}, {2: <4}, {3: <4}, {4}".format(str(_), round(p, 3), round(u, 3), n, round(logit, 3)))


if __name__ == '__main__':

    x = list(range(3))
    pool = Pool(x)
    for _ in range(10):
        i, j, p, q = pool.sample()
        u = 1 - 2 * random.random()
        pool.update(i, j, p, q, u)
    print(pool.logits)
    print(pool.softmax())
    print(pool.answer())
    pool.introduce(3)
    print(pool.logits)
    print(pool.softmax())
    import torch
    torch.pickle