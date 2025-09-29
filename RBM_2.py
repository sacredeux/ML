import numpy as np
from itertools import product
from time import time
import matplotlib.pyplot as plt
import random
import os 



class RestrictedBoltzmannMachine:
    def __init__(self, learning_rate: float = 0.01, k: int = 10, M:int = 2):
        self.learning_rate = learning_rate
        self.k = k
        self.M = M

        # Hardcocded parameters from the solution manual 
        self.nu_max = 10_000
        self.p0 = 4
        self.RNG = np.random.default_rng()

    def contrastive_divergence(self, patterns):


        n, self.N = patterns.shape
        self.weights = np.random.normal(loc=0.0, scale=0.1, size=(self.M, self.N))
        self.t_h = np.zeros(self.M)
        self.t_v = np.zeros(self.N)

        # Epochs:
        for _ in range(self.nu_max):
            
            dw = np.zeros_like(self.weights, dtype=np.float32)
            dt_h = np.zeros_like(self.t_h, dtype=np.float32)
            dt_v = np.zeros_like(self.t_v, dtype=np.float32)
            
            sample = self.RNG.choice(patterns, size=self.p0, axis=0, replace=False)
            # Sample patterns: 
            for x in sample:
                # Initialize: 
                b_h0 = (self.weights @ x) - self.t_h          # (M,)
                prob_bh = (1.0 + np.tanh(b_h0)) / 2.0                        # sigmoid(2x) = (1 + tanh(x)) / 2    
                h = np.where(np.random.rand(self.M) < prob_bh, 1.0, -1.0)
                v = x.copy()

                # Gibbs sampling: 
                for t in range(self.k):
                    b_v = (h @ self.weights) - self.t_v         # (N,)
                    prob_bv = (1.0 + np.tanh(b_v)) / 2.0
                    v = np.where(np.random.rand(self.N) < prob_bv, 1.0, -1.0)

                    b_h = (self.weights @ v) - self.t_h
                    prob_bh = (1.0 + np.tanh(b_h)) / 2.0
                    h = np.where(np.random.rand(self.M) < prob_bh, 1.0, -1.0)

                tanh_b_h0 = np.tanh(b_h0)
                tanh_b_h  = np.tanh(b_h)

                dw += (np.outer(tanh_b_h0, x) - np.outer(tanh_b_h, v))
                dt_v -= (x - v)                 # accumulate positive - negative
                dt_h -= (tanh_b_h0 - tanh_b_h)

            # average over mini-batch and scale by eta
            dw = (self.learning_rate / self.p0) * dw
            dt_v = (self.learning_rate / self.p0) * dt_v
            dt_h = (self.learning_rate / self.p0) * dt_h

            self.weights += dw
            self.t_v += dt_v
            self.t_h += dt_h


    def discover_dynamics(self):
        pass


    def plot_diagnostics(self):
        pass

    def plot_results(self):
        pass


if __name__ == "__main__":
    print('Running RBM')

    patterns = np.array([[-1, -1, -1],
                         [ 1, -1,  1],
                         [-1,  1,  1],
                         [ 1,  1, -1]])
    train_rbm = False
    train_rbm = True

    rbm = RestrictedBoltzmannMachine(M = 2, learning_rate= 0.01, k = 10)

    rbm.contrastive_divergence(patterns=patterns)

