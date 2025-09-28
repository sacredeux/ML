import numpy as np
from itertools import product
from time import time
import matplotlib.pyplot as plt
import random
import os 


#TODO: Plot 4.40
#TODO: Ex. 4.7
#TODO: Ex 4.12
#TODO: Calculate DK-N
#TODO: M = 1,2,4,8
#TODO: 

def short_plot():
    patterns = np.array([[-1, -1, -1],
                        [1, -1, 1],
                        [-1, 1, 1],
                        [1, 1, -1]])

    fig, ax = plt.subplots(figsize=(5,5))

    ax.imshow(patterns, cmap='gray_r')
    fig2, ax2 = plt.subplots(figsize=(5,5))

    N = 3
    DKn = np.zeros(4)
    Ms = np.array([2**i for i in range(4)]) 

    for i, M in enumerate(Ms):
        d = int(np.log2(M + 1))
        if M < (2**(N-1) - 1):
            DKn[i] = np.log(2)*(N - d - (M+1)/2**d)
        else:
            DKn[i] = 0

    ax2.plot(Ms, DKn,'-.', label=r'Upper bound $D_{KL}$')

    ax.set_xticks([])
    ax.set_yticks([])

    ax2.set_xlabel('M')
    ax2.set_ylabel(r'$D_{KL}$')
    ax2.set_ylim([-.1,1])

    ax2.legend()
    ax2.grid(axis='y')

    plt.show()


def train():
    N = 3
    M = 4
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

    # save (create dir if needed)
    dir_path = os.path.join("OpenTA","Homework2", 'rbm_weights')
    os.makedirs(dir_path, exist_ok=True)
    np.savetxt(os.path.join(dir_path,'w.csv'), weights, delimiter=',')
    np.savetxt(os.path.join(dir_path,'t_h.csv'), t_h, delimiter=',')
    np.savetxt(os.path.join(dir_path,'t_v.csv'), t_v, delimiter=',')
    return weights, t_v, t_h



def discover_dynamics(weights, t_v, t_h, burn_in=50_000, n_samples=5_000_000):
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
    M, N = weights.shape
    patterns = np.array([[-1, -1, -1],
                         [ 1, -1,  1],
                         [-1,  1,  1],
                         [ 1,  1, -1]])
    
    # init state
    v = np.random.choice([-1, 1], size=N)
    counts = np.zeros(len(patterns))
    
    def sample_hidden(v):
        b_h = weights @ v - t_h
        prob_h = (1 + np.tanh(b_h)) / 2
        return np.where(np.random.rand(M) < prob_h, 1, -1)

    def sample_visible(h):
        b_v = h @ weights - t_v
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
        idx = np.where((patterns == v).all(axis=1))[0]
        if len(idx) > 0:
            counts[idx[0]] += 1
    
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

if __name__ == "__main__":
    print('letsa go')
    # short_plot()
    train_rbm = False
    train_rbm = True
    if train_rbm:
        weights, t_v, t_h = train()
    else:
        dir_path = os.path.join("OpenTA","Homework2", 'rbm_weights')
        weights = np.loadtxt(os.path.join(dir_path,'w.csv'), delimiter=',')
        t_h = np.loadtxt(os.path.join(dir_path,'t_h.csv'), delimiter=',')
        t_v = np.loadtxt(os.path.join(dir_path,'t_v.csv'), delimiter=',')
    discover_dynamics(weights, t_v, t_h)
