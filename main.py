import game
import rnad

import logging
import threading

class MeteredRNaD (rnad.RNaD):

    """
    You are allowed to change max samples. Functions like min_expl will use max_samples to limit their view of logs
    and the run function will be limited by it in the wrapper. So you can prolong tests by increasing m_s
    """

    def __init__ (self, max_samples, **args):
        super().__init__(**args)
        self.max_samples = max_samples
        self.used_samples = 0
        """
        batch_size * steps <= max_samples
        """

    def metered_run(self, max_updates=1, **args):
        print(args)
        args['max_updates'] = 1
        for _ in range(max_updates):
            if self.used_samples >= self.max_samples:
                break
            self.run(**args)
            

class RNaD_Pool ():

    def __init__ (self, pool=None):
            
        if pool is None:
            pool = []
        self.pool = pool

        self.lock = threading.Lock()

    def run (params=None):
        if params is None:
            params = {}

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    tree = game.Tree()
    tree.load()

    test = MeteredRNaD(max_samples=10**7, 
        tree=tree,
        # directory_name='test_fixed_{}'.format(int(time.time())), 
        directory_name='test_fixed_5', 
        device=tree.device,
        eta=.2,
        # delta_m_0 = [16, 32, 64, 128, 256],
        # delta_m_1 = [16, 32, 64, 128, 256],
        # buffer_size= [1, 1, 1, 1, 1],
        # buffer_mod=  [1, 1, 1, 1, 1],
        delta_m_0 = [128,],
        delta_m_1 = [100,],
        buffer_size= [1,],
        buffer_mod=  [1,],
        lr=1*10**-4,
        batch_size=2**12,
        beta=2, # logit clip
        neurd_clip=10**4, # Q value clip
        grad_clip=10**4, # gradient clip
        net_params= {'type':'ConvNet','size':tree.max_actions,'channels':2**4,'depth':2,'batch_norm':False,'device':tree.device},

        b1_adam=0,
        b2_adam=.999,
        epsilon_adam=10**-8,
        gamma_averaging=.001,
        roh_bar=1,
        c_bar=1,
        vtrace_gamma=1,
        same_init_net='test_fixed_5',
        fixed=True,
    )
    test.metered_run(max_updates=1)
        
