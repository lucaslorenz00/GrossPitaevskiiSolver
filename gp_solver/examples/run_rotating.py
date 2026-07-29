import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..")) # For running from inside the folder

import numpy as np
import matplotlib.pyplot as plt

from gp_solver.core.grid         import Grid
from gp_solver.core.propagator   import SplitStepPropagator
from gp_solver.systems.scalar_gp import ScalarGPSystem, ScalarGPParams
from gp_solver.utils.plotting    import plot_density_phase, plot_density_spacetime

N  = 1024       # grid points
L  = 200.0      # box length in units of ξ (healing lengths)
g  = 1       # interaction strength (dimensionless)
DT = 5e-3       # real-time timestep


# Building the system with a rotation velocity
grid_A   = Grid(N=N, L=L)
system_A = ScalarGPSystem(grid_A, ScalarGPParams(
    g                = g,
    soliton_velocity = 0.5,    # stationary
    soliton_position = 0.0,    # centre of the box
    n_solitons       = 1,
    V_ext            = None,     # no external potential
))

print(f"\nSystem built.  State: {system_A.state}")

print("\n[A] Plotting analytical seed (before imaginary time)...")
fig, _ = plot_density_phase(
    system_A.state,
    title="A — Initial seed  (analytical tanh, t=0)",
    show_analytical=True,
    g=g)
plt.show()

print("\n[A] Running imaginary-time evolution (finding ground state)...")
prop_A = SplitStepPropagator(system_A)
prop_A.run_imaginary_time(
    steps        = 2000,
    dtau         = 5e-3,
    conv_tol     = 1e-9,      # Dont use anymore
    record_every = 200,
    verbose      = True,
)

# Quick checks
psi   = system_A.state.psi_complex(0)
phase = np.unwrap(np.angle(psi))
kink  = phase[-1] - phase[0]
print(r"\n  Phase kink $\Delta\theta$ = {kink/np.pi:.4f} $\pi$   (exact black soliton = 1.000 π)")
print(f"  Norm          = {system_A.state.norm():.6f}   (should equal L = {L:.1f})")

print("\n[A] Plotting ground state (after imaginary time)...")
fig, _ = plot_density_phase(
    system_A.state,
    title="A — Ground state  (after imaginary time)",
    show_analytical=False,
    g=g,
)
plt.show()

print("\n[A] Running real-time evolution  (t = 0 -> 30)...")

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

print("\n[A] Plotting final state  (t=30)...")
fig, _ = plot_density_phase(
    system_A.state,
    title=f"A — After real time  t = {system_A.state.t:.0f}  (soliton must stay at x=0)",
)
plt.show()

print("\n[A] Plotting space-time density map...")
fig, _ = plot_density_spacetime(
    x         = grid_A.x,
    times     = np.array(times_A),
    densities = np.array(densities_A),
    title     = "A — Space-time density  (vertical stripe = stationary soliton)",
)
plt.show()