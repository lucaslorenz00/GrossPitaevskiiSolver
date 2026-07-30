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
L  = 200.0      # box length in units of xi (healing lengths)
g  = 1          # interaction strength (dimensionless)
DT = 1e-2       # real-time timestep

# Building the system with a rotation velocity
grid_A   = Grid(N=N, L=L)
system_A = ScalarGPSystem(grid_A, ScalarGPParams(
    g                = g,
    soliton_velocity = 0.0, # only for seeding important
    soliton_position = 0.0,    # centre of the box
    n_solitons       = 1,
    V_ext            = None,     # no external potential
    noise_amplitude  = 0,   # noise
    V0              = 0.0,   # external potential amplitude for harmonic trap
    omega = np.pi/L*0,   # rotation frequency (for rotating frame)
    winding = 0,   # winding number for ring soliton (phase jump across box)
))

print(f"\nSystem ready.\nState: {system_A.state}")

fig, _ = plot_density_phase(
    system_A.state,
    title="Black soliton",
    show_analytical=True,
    g=g)
plt.show()

prop_A = SplitStepPropagator(system_A)
prop_A.run_imaginary_time(
    steps        = 5000,
    dtau         = 5e-3,
    conv_tol     = 1e-9,      # Dont use at the moment
    record_every = 200,
    verbose      = False,
)

# Quick checks
psi   = system_A.state.psi_complex(0)
phase = np.unwrap(np.angle(psi))
kink  = phase[-1] - phase[0]
print(r"\n  Phase kink $\Delta\theta$ =" + f"{kink/np.pi:.4f}" + r" $\pi$   (exact black soliton = 1.000 pi)")
print(f"  Norm          = {system_A.state.norm():.6f}   (should equal L = {L:.1f})")

fig, _ = plot_density_phase(
    system_A.state,
    title="Ground state (after imaginary time)",
    show_analytical=False,
    g=g,
)
plt.show()

print("\nRunning real-time evolution  (t = 0 -> 30)")

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
    title=f"After real time  t = {system_A.state.t:.0f}",
)
plt.show()

fig, _ = plot_density_spacetime(
    x         = grid_A.x,
    times     = np.array(times_A),
    densities = np.array(densities_A),
    title     = "Space-time density map",
)
plt.show()
