
import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
# from tkinter import *
# from PIL import Image
# from PIL import ImageTk as itk
# import time


# Decide on the parameters
N = 100
n_atoms = N**2
J = 1       # Neighbor interaction [k_BT/unit spin] 
s1 = +1
s2 = +1
ds = [3,5,7,10]
# ds = [3]


t = 0 
H = 0
T = 2.3

n_subsamples = n_atoms//10
t_end = 500_000
# t_end = 75_00
t_start_avg = 50_000

k_iterations = np.arange(t_end) / 1e3

results = []
for s2 in [+1, -1]:
    for d in ds:
        e_tots = np.zeros(t_end)
        mean_spin = np.zeros(t_end)
        # Initialize the lattice
        S = np.sign(np.random.rand(N, N) - 0.5)
        # Plate rows: 
        N1 = N//2 - d
        N2 = N//2 + d
        # Insert plates
        S[N1,:] = +s1
        S[N2,:] = +s2


        for t in range(t_end): 
            i = np.random.choice([i for i in range(N) if i not in [N1, N2]], n_subsamples)      # The rows of the elements that will be iterated
            j = np.random.randint(0,N,n_subsamples)      # The columns of the elements that will be iterated

            E = (-J * ( S[i-1,j] + S[(i+1)%N,j] + S[i,j-1] + S[i,(j+1)%N] ) - H)/T     # The energy of a plus spin
            S[i,j] =  (( np.random.rand(n_subsamples) < np.exp(-E)/(np.exp(-E)+np.exp(E)) ) -0.5) * 2   # Monte Carlo algorithm 

            
            # Define neighbor spins:
            neighbor_up = np.roll(S, 1, axis=0)
            neighbor_down = np.roll(S, -1, axis=0)
            neighbor_left = np.roll(S, 1, axis=1)
            neighbor_right = np.roll(S, -1, axis=1)

            # We will use their sum in calculating the energy
            S_neighbors = (neighbor_up + neighbor_down + neighbor_left + neighbor_right)

            # Total energy:
            e_tot = -J/2/N**2*np.sum(S * S_neighbors)
            e_tots[t] = e_tot
            # mean_spin[t] = np.mean(S)

        fig, ax = plt.subplots(figsize=(4,4))
        ax.plot(k_iterations, e_tots, label='Simulation')

        e_tot_average = np.mean(e_tots[t_start_avg:])
        e_tot_std = np.std(e_tots[t_start_avg:])
        
        results.append({'Average energy': e_tot_average,
                        'std energy': e_tot_std,
                        's2': s2,
                        'd': d})


        ax.hlines(y=e_tot_average,xmin=k_iterations[t_start_avg], xmax=k_iterations[-1], colors='black', linestyles='dashed', label=r'$\bar{e}_{\mathrm{tot}}$'+fr"$={e_tot_average:.2f}$")
        # moving_average = np.convolve(e_tots, np.ones(10000), 'valid') / 10000
        # ax.plot(k_iterations[:len(moving_average)], moving_average)

        ax.set_title(fr"$d = {d}$, $s_2 = {'+1' if s2 > 0 else '-1'}$")
        ax.set_xlabel(fr"Steps $(10^3)$ ")
        ax.set_ylabel(f"Total energy")
        ax.grid(alpha=0.5)
        ax.set_ylim(top=-1.0, bottom=-1.6)
        ax.legend()
        fig.tight_layout()
        fname = f"etot_plusplus_d={d}_s2={'+1' if s2 > 0 else '-1'}.pdf"
        fig.savefig(os.path.join("hw1", "q2_pics", fname))

        fig2, ax2 = plt.subplots(figsize=(4,4))
        ax2.imshow(S, cmap='viridis')
        ax2.axhline(N2, color='firebrick', alpha=0.1)
        ax2.axhline(N1, color='firebrick', alpha=0.1)
        ax2.set_title(fr"$d = {d}$, $s_2 = {'+1' if s2 > 0 else '-1'}$")
        ax2.set_xticks([])
        ax2.set_yticks([N1, N2])
        fig2.tight_layout()
        fname2 = f"sigma_endstate_d={d}_s2={'+1' if s2 > 0 else '-1'}.pdf"
        fig2.savefig(os.path.join("hw1", "q2_pics", fname2))

df = pd.DataFrame(results)
df.to_csv(os.path.join("hw1", "q2_pics", "q2_results.csv"))

plt.show()
