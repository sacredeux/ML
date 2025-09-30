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
        self.vmax = 25_000
        self.p0 = 4
        self.RNG = np.random.default_rng(137)
        
        # self.weights = np.random.normal(loc=0.0, scale=0.1, size=(self.M, self.N))
        # self.t_h = np.zeros(self.M, dtype=np.float64)
        # self.t_v = np.zeros(self.N, dtype=np.float64)
        self.weight_changes = []
        # self.errors = []

    def _logistic2(self, local_field):
        return (1.0 + np.tanh(local_field)) / 2.0

    def contrastive_divergence(self):
        N = 3
        M = 1
        patterns = np.array([[-1, -1, -1],
                            [ 1, -1,  1],
                            [-1,  1,  1],
                            [ 1,  1, -1]])

        # From here: https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
        weights = np.random.normal(loc=0.0, scale=0.1, size=(M, N))
        t_h = np.zeros(M)
        t_v = np.zeros(N)

        eta = 0.01 
        vmax = 10_000       # start smaller for testing
        k = 20          # CD-k
        p0 = 4            # mini-batch size (<=4)
        for _ in range(vmax):
            mu_indices = np.random.choice(4, size=p0, replace=False)

            dw = np.zeros_like(weights, dtype=np.float32)
            dt_h = np.zeros_like(t_h, dtype=np.float32)
            dt_v = np.zeros_like(t_v, dtype=np.float32)

            for idx_mu in mu_indices:
                v0 = patterns[idx_mu].astype(np.float32)
                b_h0 = weights @ v0 - t_h          # (M,)

                # P(h=+1|v0)
                prob_bh = (1.0 + np.tanh(b_h0)) / 2.0
                h = np.where(np.random.rand(M) < prob_bh, 1.0, -1.0)

                # k-step Gibbs from v0
                v = v0.copy()
                for _t in range(k):
                    b_v = h @ weights - t_v         # (N,)
                    prob_bv = (1.0 + np.tanh(b_v)) / 2.0
                    v = np.where(np.random.rand(N) < prob_bv, 1.0, -1.0)

                    b_h = weights @ v - t_h
                    prob_bh = (1.0 + np.tanh(b_h)) / 2.0
                    h = np.where(np.random.rand(M) < prob_bh, 1.0, -1.0)

                tanh_b_h0 = np.tanh(b_h0)
                tanh_b_h  = np.tanh(b_h)

                dw += (np.outer(tanh_b_h0, v0) - np.outer(tanh_b_h, v))
                dt_v -= (v0 - v)                 # accumulate positive - negative
                dt_h -= (tanh_b_h0 - tanh_b_h)

            # average over mini-batch and scale by eta
            dw = (eta / p0) * dw
            dt_v = (eta / p0) * dt_v
            dt_h = (eta / p0) * dt_h

            weights += dw
            t_v += dt_v
            t_h += dt_h

            self.weight_changes += [np.linalg.norm(dw)]
        self.weights = weights
        self.t_h = t_h
        self.t_v = t_v


    def discover_dynamics_2(self, burn_in=50_000, n_samples=5_000_000):
        """
        Run Gibbs sampling on the RBM and estimate the distribution over patterns.
        
        Args:
            weights : (M, N) array, hidden-to-visible weights
            t_v     : (N,) array, visible biases
            t_h     : (M,) array, hidden biases
            burn_in : number of Gibbs steps to discard before collecting samples
            n_samples : number of Gibbs samples to collect
        
        Returns:
            Q    : empirical probabilities of the 4 XOR patterns
            Dkl  : KL divergence vs. true distribution (uniform over 4 patterns)
        """
        M, N = self.weights.shape
        patterns = np.array([[-1, -1, -1],
                            [ 1, -1,  1],
                            [-1,  1,  1],
                            [ 1,  1, -1]])
        
        # init state
        v = np.random.choice([-1, 1], size=N)
        counts = np.zeros(len(patterns))
        
        def sample_hidden(v):
            b_h = self.weights @ v - self.t_h
            prob_h = (1 + np.tanh(b_h)) / 2
            return np.where(np.random.rand(M) < prob_h, 1, -1)

        def sample_visible(h):
            b_v = h @ self.weights - self.t_v
            prob_v = (1 + np.tanh(b_v)) / 2
            return np.where(np.random.rand(N) < prob_v, 1, -1)

        # burn-in phase
        for _ in range(burn_in):
            h = sample_hidden(v)
            v = sample_visible(h)

        # sampling phase
        for _ in range(n_samples):
            h = sample_hidden(v)
            v = sample_visible(h)

            # check if v matches one of the 4 XOR patterns
            s = np.equal(patterns, v).all(1) 
            # if ~np.any(s):
            #     print(f"No pattern found. v = {v}") 
            counts[s] += 1

        
        Q = counts / n_samples
        Dkl = -np.sum(0.25 * np.log((Q + 1e-12) / 0.25))  # KL(true||model)

        print(f"D_KL = {Dkl:.3f}")
        
        # Plot distribution
        fig, ax = plt.subplots()
        ax.bar(range(len(Q)), Q, tick_label=[str(p) for p in patterns])
        ax.set_ylabel("Estimated probability")
        ax.set_title("RBM learned distribution")
        plt.show()

        return Q, Dkl

    def discover_dynamics(self):
        P_b = np.zeros(len(self.patterns))
        v = np.random.choice([-1, 1], size=self.N)
        burn_in = 1_000_000
        count = 1_000_000
        # Gibbs sampling: 
        for t in range(burn_in):
            b_h = np.dot(self.weights, v) - self.t_h
            p_h = self._logistic2(b_h)
            h = np.where(np.random.rand(self.M) < p_h, 1.0, -1.0)

            b_v = np.dot(h, self.weights) - self.t_v         # (N,)
            p_v = self._logistic2(b_v)
            v = np.where(np.random.rand(self.N) < p_v, 1.0, -1.0)

        for t in range(count):
            b_h = np.dot(self.weights, v) - self.t_h
            p_h = self._logistic2(b_h)
            h = np.where(np.random.rand(self.M) < p_h, 1.0, -1.0)
            
            b_v = np.dot(h, self.weights) - self.t_v         # (N,)
            p_v = self._logistic2(b_v)
            v = np.where(np.random.rand(self.N) < p_v, 1.0, -1.0)

            s = np.equal(self.patterns, v).all(1) 
            # if np.any(s):
            P_b[s] += 1

        # TODO: Lägg till possible patterns här. ... Gör histogram och se wtf is going on. 
        P_b /= 1e6
        epsilon = 1e-10
        dkl = np.sum(0.25 * np.log(0.25 / (P_b+epsilon)))
        print(f"THE DKL IS {dkl}")

    def plot_convergence(self):
        fig1, ax1 = plt.subplots(figsize=(5,5))
        ax1.plot(self.weight_changes)
        ax1.set_yscale('log')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Weight Change (L2 Norm)")
        ax1.set_title("RBM Convergence Behavior")
        ax1.grid(True)
        fig1.tight_layout()
        fig1.savefig("rbm_convergence.png")
        print("Convergence plot saved as 'rbm_convergence.png'.")

        # fig2, ax2 = plt.subplots(figsize=(5,5))
        # ax2.plot(self.errors)
        # ax2.set_xlabel("Epoch")
        # ax2.set_ylabel("Difference between x and v")
        # ax2.set_title("RBM Convergence Behavior")
        # ax2.grid(True)
        # fig2.tight_layout()
        # fig2.savefig("rbm_error.png")
        # print("Error plot saved as 'rbm_error.png'.")

if __name__ == "__main__":
    time_start = time()
    print('Running RBM')

    patterns = np.array([[-1, -1, -1],
                         [ 1, -1,  1],
                         [-1,  1,  1],
                         [ 1,  1, -1]])

    rbm = RestrictedBoltzmannMachine(patterns=patterns, M = 1, learning_rate= 0.1, k = 10)

    rbm.contrastive_divergence()
    rbm.plot_convergence()
    rbm.discover_dynamics_2()
    time_end = time()
    print(f"Time to run: {(time_end-time_start):.2f}")
