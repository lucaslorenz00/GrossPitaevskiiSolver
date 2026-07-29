"""
utils/plotting.py

Plot helpers to produce the plots

Available plots
----------------
plot_density_phase(state)
    density |psi|^2 and unwrapped phase theta (for scalar GP)
plot_history(state, keys)
    time series of recodred state.history

"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# from gp_solver.core import state
from gp_solver.core.state       import GPState
from gp_solver.analysis.diagnostics import unwrap_phase, find_soliton_position, find_soliton_positions

# Style
_CMAP_DENSITY = "inferno"
# _CMAP_PHASE   = "twilight"
_COLORS       = ["#275FBA", "#DD8452", "#55A868","#C44E52"]  # palette (intersctive)

def _style_ax(ax, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    """
    Apply common styling to a matplotlib axis
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)

def plot_density_phase(
        state: GPState, comp: int = 0, title: str = None, 
        show_analytical: bool= False, g: float = 1.0
        ) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot the density |psi|^2 and unwrapped phase theta of a GPState (scalar or spinor)

    Parameters
    ----------
    state: GPState
        The wavefunction state to plot
    comp: int
        Component index (for spinor)
    title: str
        Optional title for the plot
    show_analytical: bool
        Whether to show analytical solution
    g: float
        Nonlinear interaction strength (for analytical solution)

    Returns
    -------
    fig, axes: (Figure, ndarrray of Axes)
        figure and axes objects
    """
    # Not for components > 1, but can be extended later, only included half
    x = state.grid.x
    n = state.density(comp)
    phase = unwrap_phase(state, comp)
    
    # Create figure and axes
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True, sharex=True)
    fig.suptitle(title or f"{state.lable} | t= {state.t:.3f}", fontsize=13)

    # density
    ax = axs[0]
    ax.plot(x, n, color=_COLORS[comp % len(_COLORS)], lw=1.5, label=f"$|\psi(x)|^2$")

    if show_analytical:
        n0 = float(np.max(n))
        xi = 1.0 / np.sqrt(2.0 * g * n0 + 1e-30)
        # Find soliton positions for overlay
        x_sol = find_soliton_positions(state, comp)
        for xs in x_sol:
            n_ana = n0 * np.tanh((x - xs) / (np.sqrt(2.0) * xi)) ** 2
            ax.plot(x, n_ana, "--", color=_COLORS[3], lw=1.2,
                    label=r"$n_0 \tanh^2$")
        if len(x_sol):
            ax.legend(fontsize=9)

    _style_ax(ax, xlabel="x", ylabel=r"$Density |\psi|^2$", title=f"Density (Component {comp})")
    ax.set_ylim(bottom=0.0)

    # Plot phase
    ax = axs[1]
    ax.plot(x, phase/np.pi, color=_COLORS[comp % len(_COLORS)+1], lw=1.5)
    #ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    #ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    _style_ax(ax, xlabel=r"Position $x/\xi$", ylabel=r"Phase $\theta(x)/\pi$")

    fig.tight_layout()
    return fig, axs

# No method yet to produce trajecories
def plot_soliton_trajectory(
        times: np.ndarray,
        trajectories: List[np.ndarray],
        title: str = "Soliton Trajectories"
        ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Space time plot of soliton trajectories x_s(t).

    Expected:
    Black soliton: v = 0 (flat horizontal line)
    Dark soliton v != 0 (diagonal line, slope 1/v_s)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, traj in enumerate(trajectories):
        mask = ~np.isnan(traj)
        ax.plot(times[mask], traj[mask],
                color=_COLORS[i % len(_COLORS)],
                lw=1.8, label=f"Soliton {i+1}")

    ax.legend(fontsize=10)
    _style_ax(ax,
              xlabel=r"Time $t/\tau_s$",
              ylabel=r"Position $x/\xi$",
              title=title)
    fig.tight_layout()
    return fig, ax

def plot_history(
        state: GPState, keys: List[str] = None, title: str = "Simulation history"
        ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the recorded observables in state.history.

    Parameters
    ----------
    keys: list of str
        List of keys to plot (must be present in state.history)
    
    Expected behavior
    --------
    'norm' should stay constant during real time(drift > 1e-3 indicates too large dt)
    'mu' converges monotonically during imaginary time
    'E_total' conserved during real time
    """
    if not state.history:
        print("[plotting] History is empty.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No history recorded", ha="center", va="center", transform=ax.transAxes)
        return fig, np.array([ax])

    times = [s.get("t", i) for i, s in enumerate(state.history)]

    # If no keys provided, plot all found in state.hitsory
    if keys is None:
        keys = sorted({
            k for s in state.history
            for k, v in s.items()
            if isinstance(v, (int, float)) and k != "t"
        })

    if not keys:
        print("[plotting] No scalar keys found in history.")
        return plt.subplots()

    n_panels = len(keys)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(9, 2.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = np.array([axes]) # make array in case only one axes

    fig.suptitle(title, fontsize=13)

    for ax, key in zip(axes, keys):
        vals = [s.get(key, np.nan) for s in state.history]
        ax.plot(times, vals, color=_COLORS[0], lw=1.5)
        _style_ax(ax, ylabel=key)

    axes[-1].set_xlabel(r"Time $t$", fontsize=11)
    fig.tight_layout()
    return fig, axes

# Plot density vs position and time as a 2D colour map
def plot_density_spacetime(
        x: np.ndarray,
        times: np.ndarray,
        densities: np.ndarray,
        title: str = "Density Space-Time",
        vmax: float = None,
        comp_label: str = r"$|\psi(x,t)|^2$",
        ) -> Tuple[plt.Figure, plt.Axes]:
    """
    2-D colour map of density vs position and time.

    Parameters
    ----------
    x: (N,) array
        spatial grid
    times: (T,) array
        snapshot times
    densities: (T, N) array
        density at each time and position
    vmax : float
        colour scale maximum (default: global max)

    Expected behavior
    ---------------
    Scalar soliton:
    a dark stripe at x=0 that stays fixed or drifts diagonally
    Spinor quench:
    initially uniform, then density modulations grow and form a domain lattice after the quench time
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    vmax = vmax or float(np.nanmax(densities))
    im = ax.pcolormesh( # 2D colormesh function
        x, times, densities,
        cmap=_CMAP_DENSITY,
        vmin=0, vmax=vmax,
        shading="auto",
        rasterized=True,
    )
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(comp_label, fontsize=11)

    _style_ax(ax, xlabel=r"Position $x/\xi$", ylabel=r"Time $t/\tau_s$", title=title)
    fig.tight_layout()
    return fig, ax