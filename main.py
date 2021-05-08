# Quasi-1D Nozzle Flow Simulation with Time-Marching Method
# As featured in Appendix B of Modern Compressible Flow with Historical Perspective, 4th Edition by John D. Anderson
# Author: Supakorn Suttiruang

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.rcParams.update({'font.size': 20})

# Global variables
gamma = 1.4  # Specific heat ratio
C = 0.5  # Courant number

# Grid generation
dx = 0.2  # Grid size
x_start = 0.0  # Starting x-position
x_stop = 10.0  # Stopping x-position
x = np.arange(x_start, x_stop + dx, dx)
n = len(x)  # Number of grid points

# Nozzle area distribution
# A = 1 + 2.2 * ((x - 1.5) ** 2) # Anderson Appendix B
A = np.zeros(n)
half = int(np.round(n/2))
A[:half] = 1.75 - 0.75*np.cos((0.2*x[:half]-1.0)*np.pi)
A[half:] = 1.25 - 0.25*np.cos((0.2*x[half:] - 1.0) * np.pi)

# Flow field variables
# Note that these are dimensionless
rho = 1 - 0.03146 * x  # Density normalized by reservoir density, rho0
# T = 1 - 0.2314 * x  # Temperature normalized by reservoir temperature, T0
T = 1 - 0.02314 * x  # Temperature normalized by reservoir temperature, T0
V = (0.1 + 0.1 * x) * np.sqrt(T)  # Velocity normalized by stagnation acoustic speed, a0

# Time history data
# Each variable is indexed by time-step
history = dict(rho=[], V=[], T=[], p=[], M=[])

max_t = 600  # Maximum number of time-steps
for t in np.arange(max_t+1):
    # Prediction step
    ddt_rho = np.zeros(n)
    ddt_V = np.zeros(n)
    ddt_T = np.zeros(n)

    for i in np.arange(1, n-1):
        ddt_rho[i] = -rho[i] * ((V[i+1] - V[i]) / dx) - rho[i] * V[i] * (np.log(A[i+1]) - np.log(A[i])) / dx \
                     - V[i] * (rho[i+1] - rho[i]) / dx
        ddt_V[i] = -V[i] * (V[i+1] - V[i]) / dx - (1 / gamma) * (
                ((T[i+1] - T[i]) / dx) + (T[i] / rho[i]) * ((rho[i+1] - rho[i]) / dx))
        ddt_T[i] = (-V[i] * (T[i+1] - T[i]) / dx) - ((gamma - 1) * T[i] * (
                ((V[i+1] - V[i]) / dx) + (V[i] * (np.log(A[i+1]) - np.log(A[i])) / dx)))

    # Time-step size determination
    dt_n = C * dx / (np.sqrt(T) + V)
    dt = np.min(dt_n)

    # Predicted values at next time-step
    rho_bar = rho + (ddt_rho * dt)
    V_bar = V + (ddt_V * dt)
    T_bar = T + (ddt_T * dt)

    # Correction step
    ddt_rho_bar = np.zeros(n)
    ddt_V_bar = np.zeros(n)
    ddt_T_bar = np.zeros(n)

    ddt_rho_avg = np.zeros(n)
    ddt_V_avg = np.zeros(n)
    ddt_T_avg = np.zeros(n)

    for i in np.arange(1, n-1):
        ddt_rho_bar[i] = -rho_bar[i] * ((V_bar[i] - V_bar[i-1]) / dx) - rho_bar[i] * V_bar[i] \
                         * (np.log(A[i]) - np.log(A[i-1])) / dx - V_bar[i] * (rho_bar[i] - rho_bar[i-1]) / dx
        ddt_V_bar[i] = (-V_bar[i] * (V_bar[i] - V_bar[i-1]) / dx) - (1 / gamma) \
                       * (((T_bar[i] - T_bar[i-1]) / dx) + (T_bar[i] / rho_bar[i]) * ((rho_bar[i] - rho_bar[i-1]) / dx))
        ddt_T_bar[i] = (-V_bar[i] * (T_bar[i] - T_bar[i-1]) / dx) - ((gamma - 1) * T_bar[i]
                            * (((V_bar[i] - V_bar[i-1]) / dx) + (V_bar[i] * (np.log(A[i]) - np.log(A[i-1])) / dx)))

        ddt_rho_avg[i] = 0.5 * (ddt_rho[i] + ddt_rho_bar[i])
        ddt_V_avg[i] = 0.5 * (ddt_V[i] + ddt_V_bar[i])
        ddt_T_avg[i] = 0.5 * (ddt_T[i] + ddt_T_bar[i])

        # Update flow field variables
        rho[i] += ddt_rho_avg[i] * dt
        V[i] += ddt_V_avg[i] * dt
        T[i] += ddt_T_avg[i] * dt

    # Compute pressure from equation of state
    p = rho * T

    # Compute boundary points
    # Subsonic inflow
    V[0] = (2 * V[1]) - V[2]
    # Supersonic outflow
    rho[n-1] = (2 * rho[n-2]) - rho[n-3]
    V[n-1] = (2 * V[n-2]) - V[n-3]
    T[n-1] = (2 * T[n-2]) - T[n-3]

    # Compute Mach number
    M = V / np.sqrt(T)

    # Data collection
    history['rho'].append(rho.copy())
    history['V'].append(V.copy())
    history['T'].append(T.copy())
    history['p'].append(p.copy())
    history['M'].append(M.copy())

# Plot steady-state results
PLOT_FLAG = False
if PLOT_FLAG:
    viz_var = [
        ('p', r'Static pressure, $p/p_0$'),
        ('rho', r'Density, $\rho/\rho_0$'),
        ('T', r'Static temperature, $T/T_0$'),
        ('M', r'Mach number, $M$'),
    ]
    fig, axs = plt.subplots(len(viz_var) + 1, figsize=(16, 20))
    # Plot area distribution
    axs[0].set_xlim(x_start, x_stop)
    axs[0].set_ylim(-np.ceil(np.max(A)), np.ceil(np.max(A)))
    axs[0].set_ylabel(r'$y$')
    axs[0].plot(x, A, color='k', linewidth=2)
    axs[0].plot(x, -A, color='k', linewidth=2)
    axs[0].plot(x, np.zeros(len(x)), '-.', color='r', linewidth=0.8)  # Center line
    axs[0].plot(x, np.zeros(len(x)), 'o', color='r', linewidth=3)  # Grid
    lines = []
    for idx, (var, label) in enumerate(viz_var):
        axs[idx + 1].grid(alpha=0.3)
        axs[idx + 1].set(xlim=(x_start, x_stop))
        axs[idx + 1].set_ylabel(label)
        if idx + 1 == len(viz_var):
            axs[idx + 1].set_xlabel(r'$x$')
        axs[idx + 1].plot(x, history[var][-1], '--', color='b', linewidth=2, marker='o')
    fig.show()

# NASA's CDV data from J. W. Slater (2001)
M_data = np.genfromtxt('data/M.csv', delimiter=',')
M_x, M_y = M_data[:, 0], M_data[:, 1]
p_data = np.genfromtxt('data/p.csv', delimiter=',')
p_x, p_y = p_data[:, 0], p_data[:, 1]
slater_data = dict()
slater_data['M'] = (M_x, M_y)
slater_data['p'] = (p_x, p_y)

# Animation
ANIM_FLAG = True
if ANIM_FLAG:
    # Variables to visualize
    viz_var = [
        ('p', r'Static pressure, $p/p_0$'),
        ('rho', r'Density, $\rho/\rho_0$'),
        ('T', r'Static temperature, $T/T_0$'),
        ('M', r'Mach number, $M$'),
    ]
    fig, axs = plt.subplots(len(viz_var) + 1, figsize=(16, 20))
    # Plot area distribution
    axs[0].set_xlim(x_start, x_stop)
    axs[0].set_ylim(-np.ceil(np.max(A)), np.ceil(np.max(A)))
    axs[0].set_ylabel(r'$y$')
    axs[0].plot(x, A, color='k', linewidth=2)
    axs[0].plot(x, -A, color='k', linewidth=2)
    axs[0].plot(x, np.zeros(len(x)), '-.', color='r', linewidth=0.8)  # Center line
    axs[0].plot(x, np.zeros(len(x)), 'o', color='r', linewidth=3)  # Grid
    # Plots with historical data
    for idx, var in zip([1, 4], ['p', 'M']):
        axs[idx].plot(slater_data[var][0], slater_data[var][1], '--', color='r', label="J. W. Slater (2001)")
        axs[idx].legend()
    lines = []  # All variable plots
    for idx, (var, label) in enumerate(viz_var):
        axs[idx + 1].grid(alpha=0.3)
        axs[idx + 1].set(xlim=(x_start, x_stop))
        axs[idx + 1].set_ylabel(label)
        if idx + 1 == len(viz_var):
            axs[idx + 1].set_xlabel(r'$x$')
        lines.append(axs[idx + 1].plot(x, history[var][0], '--', color='b', linewidth=2, marker='o')[0])


    def animate(t):
        # Update data at each time-step
        for line, var in zip(lines, viz_var):
            line.set_ydata(history[var[0]][t])
        # Update y-limit and offset by a bit
        for idx, var in enumerate(viz_var):
            if var[0] not in ['M', 'p']:
                ylim_min = np.min(history[var[0]][t])
                ylim_max = np.max(history[var[0]][t])
                offset = 0.05 * (ylim_max - ylim_min)
                axs[idx + 1].set(ylim=(ylim_min - offset, ylim_max + offset))
            else:
                ylim_min = np.min((np.min(slater_data[var[0]][1]), np.min(history[var[0]][t])))
                ylim_max = np.max((np.max(slater_data[var[0]][1]), np.max(history[var[0]][t])))
                offset = 0.05 * (ylim_max - ylim_min)
                axs[idx + 1].set(ylim=(ylim_min - offset, ylim_max + offset))

        # Show iteration number
        axs[0].set_title(r'$\Delta t$ = {}'.format(t))

    # Animate frames
    anim = FuncAnimation(
        fig, animate, interval=30, frames=max_t)
    plt.draw()
    plt.show()
    anim.save('animation.mp4')
