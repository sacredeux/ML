import math
import numpy as np 
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
# Initialize the neighbour list.
def list_neighbours(x, y, N_particles, cutoff_radius):
    '''Prepare a neigbours list for each particle.'''
    neighbours = []
    neighbour_number = []
    for j in range(N_particles):
        distances = np.sqrt((x - x[j]) ** 2 + (y - y[j]) ** 2)
        neighbor_indices = np.where(distances <= cutoff_radius)
        neighbours.append(neighbor_indices)
        neighbour_number.append(len(neighbor_indices))
    return neighbours, neighbour_number


def total_force_cutoff(x, y, N_particles, sigma, epsilon, neighbours):
    '''
    Calculate the total force on each particle due to the interaction with a 
    neighbours list with the particles interacting through a Lennard-Jones 
    potential.
    '''
    Fx = np.zeros(N_particles)
    Fy = np.zeros(N_particles)
    for i in range(N_particles):
        for j in list(neighbours[i][0]):
            if i != j:
                r2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
                r = np.sqrt(r2)
                ka2 = sigma ** 2 / r2
                
                # Force on i due to j.
                F = 24 * epsilon / r * (2 * ka2 ** 6 - ka2 ** 3)  # Modulus.
                
                Fx[i] += F * (x[i] - x[j]) / r
                Fy[i] += F * (y[i] - y[j]) / r
    return Fx, Fy



def lennard_jones(m = 1, sigma = 1, L = 10, eps = 1, v0 = 1, N_particles = 100, dt = 0.001,T_tot=10, tracked_particle=45):
    """
    Parameters for the Lennard-Jones gas.
    m - Mass (units of m0).
    sigma - Size (units of sigma0).
    eps - Energy (unit of epsilon0).
    v0 - Initial speed (units of v0 = sqrt((2 * epsilon0) / m0)).
    L - Box size (units of sigma0).

    Parameters for the simulation:
    N_particles = 100  # Number of particles.
    dt - Time step (units of t0 = sigma * sqrt(m0 /(2 * epsilon0))).
    """

    L = L*sigma
    x_min, x_max, y_min, y_max = -L/2, L/2, -L/2, L/2

    cutoff_radius = 5 * sigma  # Cutoff_radius for neighbours list.

    # Generate initial positions on a grid and orientations at random.
    x0, y0 = np.meshgrid(
        np.linspace(- L / 2, L / 2, int(np.sqrt(N_particles))),
        np.linspace(- L / 2, L / 2, int(np.sqrt(N_particles))),
    )
    x0 = x0.flatten()[:N_particles]
    y0 = y0.flatten()[:N_particles]
    phi0 = (2 * np.random.rand(N_particles) - 1) * np.pi

    neighbours, neighbour_number = list_neighbours(x0, y0, N_particles, cutoff_radius)

    # Initialize the variables for the leapfrog algorithm.
    # Current time step.
    x = x0
    y = y0
    x_half = np.zeros(N_particles)
    y_half = np.zeros(N_particles)
    v = v0
    phi = phi0
    vx = v0 * np.cos(phi0)
    vy = v0 * np.sin(phi0)

    # Next time step.
    nx = np.zeros(N_particles)
    ny = np.zeros(N_particles)
    nv = np.zeros(N_particles)
    nphi = np.zeros(N_particles)
    nvx = np.zeros(N_particles)
    nvy = np.zeros(N_particles)

    n_steps = int(T_tot/dt) + 1

    tracked_trajectory = np.zeros(shape=(n_steps, 2))
    tracked_trajectory[0,:] = nx[tracked_particle],ny[tracked_particle]

    for step in range(n_steps):
        x_half = x + 0.5 * vx * dt      
        y_half = y + 0.5 * vy * dt      

        fx, fy = \
            total_force_cutoff(x_half, y_half, N_particles, sigma, eps, neighbours)
        
        nvx = vx + fx / m * dt
        nvy = vy + fy / m * dt
            
        nx = x_half + 0.5 * nvx * dt
        ny = y_half + 0.5 * nvy * dt       
        
        # Reflecting boundary conditions.
        for j in range(N_particles):
            if nx[j] < x_min:
                nx[j] = x_min + (x_min - nx[j])
                nvx[j] = - nvx[j]

            if nx[j] > x_max:
                nx[j] = x_max - (nx[j] - x_max)
                nvx[j] = - nvx[j]

            if ny[j] < y_min:
                ny[j] = y_min + (y_min - ny[j])
                nvy[j] = - nvy[j]
                
            if ny[j] > y_max:
                ny[j] = y_max - (ny[j] - y_max)
                nvy[j] = - nvy[j]
        
        nv = np.sqrt(nvx ** 2 + nvy ** 2)
        for i in range(N_particles):
            nphi[i] = math.atan2(nvy[i], nvx[i])
        
        # Update neighbour list.
        if step % 10 == 0:
            neighbours, neighbour_number = \
                list_neighbours(nx, ny, N_particles, cutoff_radius)

        # Update variables for next iteration.
        x = nx
        y = ny
        vx = nvx
        vy = nvy
        v = nv
        phi = nphi

        tracked_trajectory[step,:] = nx[tracked_particle],ny[tracked_particle]


    fig1, ax1 = plt.subplots(figsize=(4,4))
    ax1.set_xlim(-L/2, L/2)
    ax1.set_ylim(-L/2, L/2)
    ax1.set_aspect('equal')  # ensures x and y are same scale
    ax1.set_title(fr"t = {step*dt}")
    ax1.set_xlabel(fr"$x $ $[\sigma]$")
    ax1.set_ylabel(fr"$y $ $[\sigma]$")

    # Draw circles
    for i, (x_c, y_c) in enumerate(zip(nx, ny)):
        circle = Circle((x_c, y_c), radius=sigma/2, edgecolor='k', facecolor='royalblue' if i!=tracked_particle else 'firebrick', lw=.1)
        ax1.add_patch(circle)

    assert np.all(np.abs(nx) <= L/2) and np.all(np.abs(ny) <= L/2)

    # Let's plot trajectory:
    fig2, ax2 = plt.subplots(figsize=(4,4))
    ax2.plot(tracked_trajectory[:,0], tracked_trajectory[:, 1], color='k')
    ax2.grid(alpha=0.25)
    # ax2.scatter(nx, ny, c='royalblue', s=1)
    ax2.set_xlim(-L/2, L/2)
    ax2.set_ylim(-L/2, L/2)
    ax2.set_title(fr"Trajectory for particle {tracked_particle}")
    ax2.set_xlabel(fr"$x $ $[\sigma]$")
    ax2.set_ylabel(fr"$y $ $[\sigma]$")
    
    # Finally, let's calculate the mean square displacement (MSD) and then plot it:
    print(tracked_trajectory)
    msd =  np.zeros(n_steps)
    for n in range(1, n_steps):
        dx = tracked_trajectory[n:, 0] - tracked_trajectory[:-n, 0]
        dy = tracked_trajectory[n:, 1] - tracked_trajectory[:-n, 1]
        msd[n] = np.mean(dx**2 + dy**2)
    print(msd)

    # Let's plot trajectory:
    fig3, ax3 = plt.subplots(figsize=(4,4))
    ax3.loglog([dt*n for n in range(n_steps)],msd, scalex="log", scaley="log")
    # ax3.set_xlim(-L/2, L/2)
    # ax3.set_ylim(-L/2, L/2)
    ax3.set_title(r"$\mathrm{MSD}, $"+ rf"$L = {L}$")
    ax3.set_xlabel(fr"$n\Delta t$")
    ax3.set_ylabel(r"$\mathrm{MSD}$ $[\sigma^2]$")
    ax3.grid(alpha=0.25)


    fig4, ax4 = plt.subplots(figsize=(4,4))
    ax4.set_xlim(-L/2, L/2)
    ax4.set_ylim(-L/2, L/2)
    ax4.set_aspect('equal')  # ensures x and y are same scale
    ax4.set_title(fr"t = 0")
    ax4.set_xlabel(fr"$x $ $[\sigma]$")
    ax4.set_ylabel(fr"$y $ $[\sigma]$")

    # Draw circles
    for i, (x_c, y_c) in enumerate(zip(x0, y0)):
        circle = Circle((x_c, y_c), radius=sigma/2, edgecolor='k', facecolor='royalblue' if i!=tracked_particle else 'firebrick', lw=.1)
        ax4.add_patch(circle)

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()

    fig1.savefig(os.path.join('HW1', 'part1','pics', f'hw1q1_finalstep_L={L}.pdf'))
    fig2.savefig(os.path.join('HW1', 'part1','pics', f'hw1q1_trajectory_L={L}.pdf'))
    fig3.savefig(os.path.join('HW1', 'part1','pics', f'hw1q1_MSD_L={L}.pdf'))
    fig4.savefig(os.path.join('HW1', 'part1','pics', f'hw1q1_firststep_L={L}.pdf'))




    return


def run():

    for L in [10, 16]:
        lennard_jones(T_tot=15, L=L)
    return


if __name__ == "__main__":
    run()
    plt.show()