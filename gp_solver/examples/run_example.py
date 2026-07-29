"""
run_example.py — First steps with the GP solver
================================================

This script walks through the complete workflow in the simplest possible way.
It runs two back-to-back experiments:

    PART A — Stationary (black) soliton   v = 0
    PART B — Moving     (dark)  soliton   v = 0.5 * c_s

For each experiment the steps are:
    1. Build the system     (choose physics parameters)
    2. Plot initial state   (the analytical seed, before any evolution)
    3. Imaginary time       (relax to the self-consistent GP ground state)
    4. Plot ground state    (see what the GP equation settled on)
    5. Real time            (watch it evolve dynamically)
    6. Plot final state     (see what happened)
    7. Space-time map       (full trajectory as a single colour image)

Run from the folder that CONTAINS gp_solver/:
    python run_example.py

Nine PNG files are saved next to this script.

What to expect
--------------
Part A (v=0, black soliton)
  Initial    : density dip to ~0 at x=0, flat at n=1 elsewhere, π phase step.
  After imag : profile tightens to exact GP solution. Phase kink = exactly π.
  After real : soliton sits still. Density plot is identical to ground state.
  Space-time : dark vertical stripe at x=0. No motion.

Part B (v=0.5, dark soliton)
  Initial    : shallower dip (n_core = v² = 0.25), phase kink < π.
               Placed at x=+20, will travel rightward.
  No imaginary time (see note below).
  After real : dip has moved. Measured velocity ≈ 0.50 = c_s/2.
  Space-time : diagonal dark stripe. Slope = 1/v_s = 2.

NOTE — why no imaginary time for the moving soliton
----------------------------------------------------
A moving soliton has a complex wavefunction: ψ = √n₀(iv/cs + cos·tanh).
The imaginary part iv/cs IS the velocity — it encodes the superfluid current.
Imaginary time replaces t → -iτ and applies real damping e^{-H·dτ}, which
kills all imaginary parts and drives ψ to be real. Running imaginary time on
a moving soliton seed destroys the velocity and gives back the v=0 black
soliton. So for a moving soliton: use the analytical seed directly, skip
imaginary time, and start real-time evolution immediately.

Units
-----
All dimensionless: length in ξ (healing length), time in ξ/c_s, c_s = 1.
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


# ─────────────────────────────────────────────────────────────────────────────
# Small helper: save figure and print path
# ─────────────────────────────────────────────────────────────────────────────

def save(fig, name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"    → saved {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared grid and time settings
# ─────────────────────────────────────────────────────────────────────────────

N  = 1024       # grid points — power of 2 for fast FFT
L  = 120.0      # box length in units of ξ (healing lengths)
g  = 1.0        # interaction strength (dimensionless)
DT = 5e-3       # real-time timestep


# ═════════════════════════════════════════════════════════════════════════════
# PART A — Stationary (black) soliton   v = 0
# ═════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("  PART A: Stationary black soliton   v = 0")
print("=" * 60)

# ── A1. Build the system ──────────────────────────────────────────────────────
# ScalarGPParams describes what condensate you want.
# soliton_velocity=0 gives the black soliton ψ = √n₀ tanh(x/√2ξ).

grid_A   = Grid(N=N, L=L)
system_A = ScalarGPSystem(grid_A, ScalarGPParams(
    g                = g,
    soliton_velocity = 0.0,    # stationary
    soliton_position = 0.0,    # centre of the box
    n_solitons       = 1,
))
print(f"\nSystem built.  State: {system_A.state}")

# ── A2. Plot the analytical seed ─────────────────────────────────────────────
# This is the starting point — the exact analytical formula, not yet the
# self-consistent GP solution. The profile is already very close to correct
# because the analytical tanh IS the exact GP solution for a uniform background,
# but imaginary time will still tighten any small finite-box errors.

print("\n[A] Plotting analytical seed (before imaginary time)...")
fig, _ = plot_density_phase(
    system_A.state,
    title="A — Initial seed  (analytical tanh, t=0)",
    show_analytical=True,
    g=g,
)
save(fig, "A1_initial_seed.png")

# ── A3. Imaginary-time evolution ──────────────────────────────────────────────
# Replace t → -iτ. The GP equation becomes a diffusion equation that filters
# out all excited-state components. After enough steps only the lowest-energy
# state (the ground state) survives. The norm is renormalised after every step.
#
# Stability condition: dtau < dx²/2 = (120/1024)²/2 ≈ 0.007
# We use dtau=5e-3 which is safely below this limit.

print("\n[A] Running imaginary-time evolution (finding ground state)...")
prop_A = SplitStepPropagator(system_A)
prop_A.run_imaginary_time(
    steps        = 2000,
    dtau         = 5e-3,
    conv_tol     = 1e-9,      # stop when energy change < 1e-9 per step
    record_every = 200,
    verbose      = True,
)

# Quick checks
psi   = system_A.state.psi_complex(0)
phase = np.unwrap(np.angle(psi))
kink  = phase[-1] - phase[0]
print(f"\n  Phase kink Δθ = {kink/np.pi:.4f} π   (exact black soliton = 1.000 π)")
print(f"  Norm          = {system_A.state.norm():.6f}   (should equal L = {L:.1f})")

print("\n[A] Plotting ground state (after imaginary time)...")
fig, _ = plot_density_phase(
    system_A.state,
    title="A — Ground state  (after imaginary time)",
    show_analytical=True,
    g=g,
)
save(fig, "A2_ground_state.png")

# ── A4. Real-time evolution ────────────────────────────────────────────────────
# Evolve with the full GP equation. The black soliton (v=0) is a stationary
# solution — it should not move at all. Norm must stay constant to < 0.01%.

print("\n[A] Running real-time evolution  (t = 0 → 30)...")

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
save(fig, "A3_after_real_time.png")

print("\n[A] Plotting space-time density map...")
fig, _ = plot_density_spacetime(
    x         = grid_A.x,
    times     = np.array(times_A),
    densities = np.array(densities_A),
    title     = "A — Space-time density  (vertical stripe = stationary soliton)",
)
save(fig, "A4_spacetime.png")


# ═════════════════════════════════════════════════════════════════════════════
# PART B — Moving (dark) soliton   v = 0.5 * c_s
# ═════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("  PART B: Moving dark soliton   v = 0.5 · c_s")
print("=" * 60)

# ── B1. Build the system ──────────────────────────────────────────────────────
# soliton_velocity = 0.5 gives a dark soliton at half the speed of sound.
# Placed at x=+20 so we can watch it travel rightward for ~60 time units
# before it wraps around the periodic box.
#
# Physical properties at v = 0.5:
#   Core density  n_core = n₀ · (v/c_s)² = 0.25   (not zero — grey soliton)sel
#   Phase kink    Δθ = 2 arccos(v/c_s) ≈ 2.09 rad ≈ 0.67 π  (less than π)
#   Width         ξ_eff = ξ/√(1-v²) ≈ 1.15 ξ       (slightly wider)

grid_B   = Grid(N=N, L=L)
system_B = ScalarGPSystem(grid_B, ScalarGPParams(
    g                = g,
    soliton_velocity = 0.5,     # half the speed of sound
    soliton_position = 20.0,    # start right of centre
    n_solitons       = 1,
))
print(f"\nSystem built.  State: {system_B.state}")
print(f"  Expected core density : {0.5**2:.2f}   (= v²)")
print(f"  Expected phase kink   : {2*np.arccos(0.5)/np.pi:.4f} π")

# ── B2. Plot the analytical seed ─────────────────────────────────────────────
# The grey (dark) soliton at v=0.5. The density dip is shallower and the
# phase step is smaller than the black soliton.

print("\n[B] Plotting analytical seed (before any evolution)...")
fig, _ = plot_density_phase(
    system_B.state,
    title="B — Initial seed  (dark soliton v=0.5, at x=+20)",
    show_analytical=True,
    g=g,
)
save(fig, "B1_initial_seed.png")

# ── B3. NO imaginary time — go straight to real time ─────────────────────────
# See the note at the top of this file. Imaginary time kills the velocity.
# The analytical dark soliton IS the exact solution of the GP equation,
# so we do not need imaginary time here.

print("\n[B] Skipping imaginary time (would destroy the velocity).")
print("    Using the analytical seed directly as the initial condition.")

prop_B = SplitStepPropagator(system_B)

# ── B4. Real-time evolution ────────────────────────────────────────────────────
# The soliton moves to the right at v = 0.5.
# With L=120 and start at x=20, it has ~100 units to travel before wrapping.
# At v=0.5 the soliton moves ~10 units in t=20, giving clean linear motion.

print("\n[B] Running real-time evolution  (t = 0 → 80)...")

densities_B = []
times_B     = []
sol_x_B     = []
x_tracker   = 20.0   # rolling window tracker — follows the soliton

def record_B(state):
    global x_tracker
    n = state.density(0)
    x = state.grid.x
    # Find the density minimum within a window around the last known position.
    # This avoids being confused by sound waves emitted into the bulk.
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
print(f"  Error                     = {abs(slope-0.5)/0.5*100:.2f}%")

print(f"\n[B] Plotting final state  (t={system_B.state.t:.0f})...")
fig, _ = plot_density_phase(
    system_B.state,
    title=f"B — After real time  t = {system_B.state.t:.0f}  (soliton moved right)",
)
save(fig, "B2_after_real_time.png")

print("\n[B] Plotting space-time density map...")
fig, _ = plot_density_spacetime(
    x         = grid_B.x,
    times     = np.array(times_B),
    densities = np.array(densities_B),
    title     = "B — Space-time density  (diagonal stripe = moving soliton, slope=1/v)",
)
save(fig, "B3_spacetime.png")

# ── B5. Overlay the measured trajectory on the space-time map ─────────────────
print("\n[B] Space-time map with tracked trajectory overlaid...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.pcolormesh(
    grid_B.x, np.array(times_B), np.array(densities_B),
    cmap="inferno", vmin=0, vmax=1.2, shading="auto", rasterized=True,
)
# Overlay trajectory
ax.plot(x_arr, t_arr, color="cyan", lw=1.5, ls="--", label="Tracked position")
# Overlay linear fit
x_fit = slope * t_arr[mask] + intercept
ax.plot(x_fit, t_arr[mask], color="lime", lw=1.5, ls="-",
        label=f"Linear fit  v = {slope:.3f}")
ax.set_xlabel(r"Position $x/\xi$", fontsize=11)
ax.set_ylabel(r"Time $t/\tau_s$", fontsize=11)
ax.set_title("B — Space-time with trajectory  (v = 0.5, slope = 2)", fontsize=12)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, alpha=0.2)
fig.tight_layout()
save(fig, "B4_spacetime_trajectory.png")


# ═════════════════════════════════════════════════════════════════════════════
# Summary side-by-side comparison
# ═════════════════════════════════════════════════════════════════════════════

print("\n[Summary] Side-by-side density comparison...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Black soliton (v=0) vs Dark soliton (v=0.5)", fontsize=13)

for ax, state, title, color in [
    (axes[0], system_A.state, "Stationary  v=0  (after t=30)", "#4C72B0"),
    (axes[1], system_B.state, "Moving  v=0.5  (after t=80)",   "#DD8452"),
]:
    x = state.grid.x
    n = state.density(0)
    ax.plot(x, n, color=color, lw=1.8)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6, label="Background n₀=1")
    ax.set_xlabel(r"Position $x/\xi$", fontsize=11)
    ax.set_ylabel(r"Density $|\psi|^2$", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 1.4)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

# Annotate the physics
axes[0].annotate("n=0\n(complete dip)", xy=(0, 0), xytext=(8, 0.15),
                 arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
axes[1].annotate(f"n={0.5**2:.2f}\n(partial dip)", xy=(0, 0.25),
                 xytext=(35, 0.25), fontsize=9, color=color)


fig.tight_layout()
save(fig, "summary_comparison.png")


# ═════════════════════════════════════════════════════════════════════════════
# Print final summary
# ═════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("  Results summary")
print("=" * 60)
print(f"  Part A — black soliton (v=0):")
print(f"    Phase kink        = {kink/np.pi:.4f} π   (expect 1.0000 π)")
print(f"    Norm drift        = {drift:.5f}%")
print(f"  Part B — dark soliton (v=0.5):")
print(f"    Measured velocity = {slope:.4f}   (expect 0.5000)")
print(f"    Velocity error    = {abs(slope-0.5)/0.5*100:.2f}%")
print()
print("  Files saved:")
for f in ["A1_initial_seed.png", "A2_ground_state.png",
          "A3_after_real_time.png", "A4_spacetime.png",
          "B1_initial_seed.png", "B2_after_real_time.png",
          "B3_spacetime.png", "B4_spacetime_trajectory.png",
          "summary_comparison.png"]:
    print(f"    {f}")
print()
print("  Close figure windows to exit.")
plt.show()