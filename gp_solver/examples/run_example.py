"""
run_example.py 
================================================

PART A — Stationary (black) soliton   v = 0
PART B — Moving (grey)  soliton   v = 0.5 * c_s

For each experiment the steps are:
    1. Build the system     (choose physics parameters)
    2. Plot initial state   (the analytical seed, before any evolution)
    3. Imaginary time       (relax to the GP ground state)
    4. Plot ground state    (plot ground state)
    5. Real time            (evolution)
    6. Plot final state     (plot final stage)
    7. Space-time map       (full trajectory as space time map)


Expected behavior
--------------
Part A (v=0, black soliton)
    Initial    : density dip to ~0 at x=0, flat at n=1 elsewhere, pi phase kink
    After imag : profile tightens to exact GP solution. Phase kink = exactly pi (does not really work)
    After real : soliton does not move, so density plot is identical to ground state (does not really work)
    Space-time : dark vertical stripe at x=0

Part B (v=0.5, dark soliton)
    Initial    : shallower dip, phase kink < pi
    No imaginary time
    After real : dip has moved with v ≈ 0.50 = c_s/2
    Space-time : diagonal dark stripe with slope = 1/v_s = 2

Units
-----
All dimensionless: length in xi (healing length), time in xi/c_s, c_s = 1, hbar = M = 1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib.pyplot as plt

from gp_solver.core.grid         import Grid
from gp_solver.core.propagator   import SplitStepPropagator
from gp_solver.systems.scalar_gp import ScalarGPSystem, ScalarGPParams
from gp_solver.utils.plotting    import plot_density_phase, plot_density_spacetime


# save figure helper

def save(fig, name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"    -> saved {name}")


# Settings

N  = 1024       # grid points — power of 2 for fast FFT
L  = 120.0      # box length in units of xi (healing lengths)
g  = 1.0        # interaction strength (dimensionless)
DT = 5e-3       # real-time timestep


# PART A — Stationary (black) soliton   v = 0
###############################################


# A1. Build the system

# soliton_velocity=0 for black soliton psi = sqrt(n_0) * tanh( x/sqrt(2xi) ).

grid_A   = Grid(N=N, L=L)
system_A = ScalarGPSystem(grid_A, ScalarGPParams(
    g                = g,
    soliton_velocity = 0.0,    # stationary
    soliton_position = 0.0,    # centre of the box
    n_solitons       = 1,
))
print(f"\nSystem built.  State: {system_A.state}")

# A2. Plot the analytical seed

# Seed with analytical solution and let imaginary time evolution refine
fig, _ = plot_density_phase(
    system_A.state,
    title="Initial seed (black soliton)",
    show_analytical=True,
    g=g,
)
plt.show()
#save(fig, "A1_initial_seed.png")

# A3. Imaginary-time evolution

# t -> -itau
#  GPE becomes a diffusion equation that filters out all excited states. After enough steps only lowest energy state should survive. (ground state search)
# The norm is renormalised after every step, since diffusion equation is not norm conserving.

# Stability condition: dtau < dx^2/2 = (120/1024)^2/2 ≈ 0.007. This simulation dtau=5e-3

prop_A = SplitStepPropagator(system_A)
prop_A.run_imaginary_time(
    steps        = 2000,
    dtau         = 5e-3,
    conv_tol     = 1e-9,      # stop when energy change < 1e-9 per step (old)
    record_every = 200,
    verbose      = True,
)

# Quick checks
psi   = system_A.state.psi_complex(0)
phase = np.unwrap(np.angle(psi))
kink  = phase[-1] - phase[0]
print(f"\n  Phase kink Delta theta = {kink/np.pi:.4f} pi   (exact black soliton = 1.0 pi)")
print(f"  Norm          = {system_A.state.norm():.6f}   (should equal L = {L:.1f})")

fig, _ = plot_density_phase(
    system_A.state,
    title="Ground state (after imaginary time) (black soliton)",
    show_analytical=True,
    g=g,
)
plt.show()
#save(fig, "A2_ground_state.png")

# A4 Real-time evolution
# Evolve in real time. The black soliton (v=0) is a stationary solution and should not move.

densities_A, times_A = [], []

def record_A(state):
    densities_A.append(state.density(0).copy())
    times_A.append(state.t)
    return {}

prop_A.add_diagnostic(record_A)
prop_A.run_real_time(t_end=30.0, dt=DT, record_every=40, verbose=True)

# Norm conservation check
norms = [s["norm"] for s in system_A.state.history if "norm" in s]
drift = (max(norms) - min(norms)) / norms[0] * 100
print(f"\n  Norm conservation drift = {drift:.5f}%   (< 0.01% is good)")

fig, _ = plot_density_phase(
    system_A.state,
    title=f"After real time  t = {system_A.state.t:.0f} (black soliton)",
)
plt.show()
#save(fig, "A3_after_real_time.png")

fig, _ = plot_density_spacetime(
    x         = grid_A.x,
    times     = np.array(times_A),
    densities = np.array(densities_A),
    title     = "Space-time density (black soliton)",
)
plt.show()
#save(fig, "A4_spacetime.png")


# PART B — Moving (dark) soliton   v = 0.5 * c_s
################################################


# B1. Build the system
#######################

# soliton_velocity = 0.5 gives a dark soliton
#
# Core density  n_core = n_0 * (v/c_s)^2 = 0.25   (not zero so grey soliton)
# Phase kink    Delta tetha < pi

grid_B   = Grid(N=N, L=L)
system_B = ScalarGPSystem(grid_B, ScalarGPParams(
    g                = g,
    soliton_velocity = 0.5,     # half the speed of sound
    soliton_position = 0.0,    # start right of centre
    n_solitons       = 1,
))
print(f"\nSystem built.  State: {system_B.state}")

# B2. Plot the analytical seed
# The grey soliton (v=0.5). The density dip is shallower, phase kink less.

fig, _ = plot_density_phase(
    system_B.state,
    title="Initial seed  (grey soliton v=0.5)",
    show_analytical=True,
    g=g,
)
plt.show()
#save(fig, "B1_initial_seed.png")

# B3. NO imaginary time, real time evolution
# Imaginary time destroys grey soliton, so just evolve

prop_B = SplitStepPropagator(system_B)

# B4. Real time evolution
# The soliton moves to the right at v = 0.5

densities_B = []
times_B     = []
sol_x_B     = []
x_tracker   = 20.0   # rolling window tracker — follows the soliton

#record density, times, and soliton position
def record_B(state):
    global x_tracker
    n = state.density(0)
    x = state.grid.x
    # Find the density minimum within a window around the last known position.
    window = 8.0
    mask   = np.abs(x - x_tracker) < window
    if mask.sum() > 0:
        x_s = float(x[mask][np.argmin(n[mask])])
    else:
        x_s = float(x[np.argmin(n)])
    x_tracker = x_s      # update rolling tracker
    densities_B.append(n.copy())
    times_B.append(state.t)
    sol_x_B.append(x_s)
    return {"soliton_x": x_s}

prop_B.add_diagnostic(record_B)
prop_B.run_real_time(t_end=20.0, dt=DT, record_every=40, verbose=True)

# Measure velocity from the recorded trajectory
t_arr = np.array(times_B)
x_arr = np.array(sol_x_B)
mask  = t_arr < 20.0           # first 20 time units, before sound-wave reflections
slope, intercept = np.polyfit(t_arr[mask], x_arr[mask], 1)
print(f"\n  Measured soliton velocity = {slope:.4f}")
print(f"  Expected velocity         = {0.5:.4f}")

fig, _ = plot_density_phase(
    system_B.state,
    title=f"After real time  t = {system_B.state.t:.0f}  (grey soliton)", # moved right
)
plt.show()
#save(fig, "B2_after_real_time.png")

fig, _ = plot_density_spacetime(
    x         = grid_B.x,
    times     = np.array(times_B),
    densities = np.array(densities_B),
    title     = "Space-time density (grey soliton)",
)
plt.show()
#save(fig, "B3_spacetime.png")