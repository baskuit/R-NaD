import os
import torch

import game

# A container for the RNaD algorithm presented 

class RNaD () :

    def __init__ (self,
        # R-NaD parameters, see paper
        eta=.2,
        delta_m=lambda m : 10**4 if m <= 100 else (10**5 if m <= 165 else 35*10**3),
        lr=5*10**-5,
        beta=2,
        grad_clip=10**4,
        b1_adam=0,
        b2_adam=.999,
        epsilon_adam=10**-8,
        gamma_averaging=.001,
        roh_bar=1,
        c_bar=1,
        batch_size=768,
        epsiolon_threshold=.03,
        n_discrete=32,
        # more parameters
        new_net=None,


        #c heckpoint
        m=0,
        n=0,
        directory=None
    ):
        self.eta = eta
        self.delta_m = delta_m
        self.lr = lr
        self.beta = beta
        self.grad_clip = grad_clip
        self.b1_adam = b1_adam
        self.b2_adam = b2_adam
        self.epsilon_adam = epsilon_adam
        self.gamma_averaging = gamma_averaging
        self.roh_bar = roh_bar
        self.c_bar = c_bar
        self.batch_size = batch_size
        self.epsiolon_threshold = epsiolon_threshold
        self.n_discrete = n_discrete
        self.m = m
        self.n = n
        self.directory = directory
        
    def initialize_net (self):

        # look through dir to find most recent saved networks, otherwise initialize:
        self.m = 0
        self.n = 0
        net_theta_n = None
        net_theta_n_target = None

        pass

    def get_last_saved (self):
        pass

    def run (self):

        net_checkpoint_data = self.get_last_saved()

