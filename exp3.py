import math
import random
class Pool :
    def __init__ (self, refs:list):
        self.refs = refs
        self.logits = [0 for _ in refs]
        self.scores = [0 for _ in refs]
        self.selections = [0 for _ in refs]
        self.n = 0

    def _cap (self,):
        max_ = max(self.logits)
        self.logits = [_ - max_ for _ in self.logits]

    def introduce (self, x):
        self.selections = [0 for _ in self.refs]
        # we clear reset estimate for NE when we introduce a new action
        # be we keep logits, so its reasonable to assume that the new NE is close
        self.refs.append(x)
        mean = sum(self.logits) / len(self.logits)
        self.logits = [_ - mean for _ in self.logits]
        
    def sample (self,) -> tuple:
        self._cap()
        exp_logits = [math.exp(_) for _ in self.logits]
        sum_ = sum(exp_logits)
        policy = [_/sum_ for _ in exp_logits]
        random.sample(policy)

