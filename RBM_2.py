import numpy as np
from itertools import product
from time import time
import matplotlib.pyplot as plt
import random
import os 



class RestrictedBoltzmannMachine:
    def __init__(self, patterns: np.ndarray, learning_rate: float = 0.01, k: int = 10, M:int = 2):
        self.learning_rate = learning_rate
        self.k = k
        self.M = M
        self.patterns = patterns
        _, self.N = self.patterns.shape
        # Hardcocded parameters from the solution manual 
        self.nu_max = 10_000
        self.p0 = 4
        self.RNG = np.random.default_rng()

    def sample_hidden(self, v):
        b_h = (self.weights @ v) - self.t_h
        prob_h = (1.0 + np.tanh(b_h)) / 2.0
        return np.where(np.random.rand(self.M) < prob_h, 1.0, -1.0)

    def sample_visible(self, h):
        b_v = (h @ self.weights) - self.t_v
        prob_v = (1.0 + np.tanh(b_v)) / 2.0
        return np.where(np.random.rand(self.N) < prob_v, 1.0, -1.0)


    def contrastive_divergence(self):


        self.weights = np.random.normal(loc=0.0, scale=0.1, size=(self.M, self.N))
        self.t_h = np.zeros(self.M)
        self.t_v = np.zeros(self.N)

        # Epochs:
        for _ in range(self.nu_max):
            dw = np.zeros_like(self.weights, dtype=np.float32)
            dt_h = np.zeros_like(self.t_h, dtype=np.float32)
            dt_v = np.zeros_like(self.t_v, dtype=np.float32)
            
            sample = self.RNG.choice(self.patterns, size=self.p0, axis=0, replace=False)
            # Sample self.patterns: 
            for x in sample:
                # Initialize: 
                b_h0 = (self.weights @ x) - self.t_h          # (M,)
                h = self.sample_hidden(x)
                v = x.copy()

                # Gibbs sampling: 
                for t in range(self.k):
                    v = self.sample_visible(h)
                    h = self.sample_hidden(v)

                tanh_b_h0 = np.tanh(b_h0)
                tanh_b_h  = np.tanh((self.weights @ v) - self.t_h)

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
        P_b = np.zeros(4)
        v = np.random.choice([-1, 1], size=self.N)
        burn_in = 100_000
        count = 1_000_000
        # Gibbs sampling: 
        for t in range(burn_in):
            h = self.sample_hidden(v)
            v = self.sample_visible(h)

        for t in range(count):
            h = self.sample_hidden(v)
            v = self.sample_visible(h)
            s = np.equal(self.patterns, v).all(1)  
            P_b[s] += 1

        P_b /= count
        dkl = -1/4*np.sum(np.log(P_b*4))
        print(dkl)


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

    rbm = RestrictedBoltzmannMachine(patterns=patterns, M = 4, learning_rate= 0.1, k = 20)

    rbm.contrastive_divergence()
    rbm.discover_dynamics()

