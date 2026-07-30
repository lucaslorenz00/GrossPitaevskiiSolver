"""
analysis/diagnostics.py

Phyiscal observables and diagnostics for the GPE simulation.

Operations are done on GPState objects, return float or np.ndarray

Obersavbles:
------------#
energy_scalar(state,g): (float, float, float)
    (E-kin, E_int, E_pot)
momentum(state): float
    total momentum p_tot
current_density(state): (N,) np.ndarray
    probability current density j(x) = Im(psi* d/dx psi)
bogoliubov_spectrum(state, g, k_max): 
    (k, omega) # Not implemented yet
sound_velocity(state, g): float
    c_s = sqrt(g*n0)
healing_length(state, g): float
    xi = 1/sqrt(2*g*n0)
spin_length(state): (N,) np.ndarray
    |<F>| = sqrt(<F_x>^2 + <F_y>^2 + <F_z>^2) or |F(x)|. needed for scalar?
mean spin length: float
    <|<F>|> = 1/N int |<F>| dx

unwrap_phase(state): (N,) np.ndarray
    Unwrap the phase of a wavefunction to remove discontinuities.
phase_kink_magnitude(state): float
    Total phase accumulated accross the box
find_soliton_positions(state, comp, min_depth, n_expected): (N,) np.ndarray
    Find the positions of a soliton with a fit
find_soliton_position(state, comp): float
    Find the position of a single soliton as the minimum of the density
"""

from __future__ import annotations
import numpy as np
from scipy.fft import fft, ifft
#from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from typing import Tuple, List

from gp_solver.core.state    import GPState

# Scalar obersvables
#####################

def energy_scalar(state: GPState, g: float) -> Tuple[float, float, float]:
    """
    Compute the kinetic, interaction and potential energy of a scalar GPE state.

    Parameters
    ----------
    state: GPState
        The wavefunction state
    g: float
        Nonlinear interaction strength

    Returns
    -------
    E_kin: float
        Kinetic energy
    E_int: float
        Interaction energy
    E_pot: float
        Potential energy (if V_ext is defined)
    """

    dx = state.grid.dx
    k2 = state.grid.k2
    N = state.grid.N
    psi_c = state.psi_complex()  # Complex wavefunction


    # Kinetic energy in k-space
    # Parseval int dpsi^2 dx = sum |k|^2 |psi_k|^2 / N
    psi_k = fft(psi_c)
    E_kin = 0.5 * np.sum(k2 * np.abs(psi_k)**2) * dx / N

    # Interaction energy in real space 1/2 g int |psi|^4 dx
    n = np.abs(psi_c)**2
    E_int = 0.5 * g * np.sum(n**2) * dx

    # Potential energy (if V_ext is defined)
    if hasattr(state, 'V_ext') and state.V_ext is not None:
        E_pot = float(np.sum(state.V_ext * n) * dx)
    else:
        E_pot = 0.0

    return E_kin, E_int, E_pot

def total_energy_scalar(
    state: GPState,
    g:     float,
    V_ext: np.ndarray = None,
) -> float:
    """Sum of all energy contributions for scalar GP."""
    return sum(energy_scalar(state, g))


# Soliton specific diagnostics
##############################

# phase unwrapping

def unwrap_phase(state: GPState, comp: int = 0) -> np.ndarray:
    """
    Unwrap the phase of a wavefunction to remove discontinuities.

    Soliton has a single Delta theta = pi kink visible after unwrapping.

    Returns
    -------
    unwrapped_phase: (N,) np.ndarray (float64)
        The unwrapped phase array
    """

    phase = np.ascontiguousarray(state.phase(comp), dtype=np.float64)
    return np.unwrap(phase) # does it work?

# Only for later analysis
def phase_kink_magnitude(state: GPState, comp: int = 0) -> float:
    """
    Total phase accumulated accross the box
    Delta theta = theta(x_max) - theta(x_min) = pi n, n integer
    """

    ph = unwrap_phase(state, comp)
    return float(ph[-1] - ph[0])

# Soliton position and velocity

def find_soliton_positions(state: GPState, comp: int = 0, min_depth: float = 0.1, n_expected: int = None) -> np.ndarray:
    """
    Find the position of a soliton as the minimum of the density

    scipy peak-finder on |psi|^2 
    Then sub-grid precision with parabolic fit on 3-point neighborhood
    
    Parameters
    ----------
    state: GPState
        The wavefunction state
    comp: int
        Component
    min_depth: float
        Minimum fractional density dip |1-n_min/n_0| (filters noise)
    n_expected: int
        If set, always return exactly n_expected minima

    Returns
    -------
    x_soliton: (N,) float 64 np.ndarray
        Position of the soliton in real space sorted ascending
    """

    n = state.density(comp)  # Density profile
    x = state.grid.x
    n0 = float(np.max(n))  # Background density

    # Find minima in density profile
    peaks, properties = find_peaks(-n, height=-n0*(1-min_depth))
    if len(peaks) == 0:
        return np.array([])  # No solitons found

    # Sub-grid refinement: fit parabola through 3 neighbouring points
    x_soliton = []
    depth = []
    for idx in peaks:
        if idx == 0 or idx == len(n) - 1:
            x_soliton.append(x[idx])  # Edge case, no refinement
            depth.append(float(n[idx]))
            continue
        # Parabolic interpolation
        y0, y1, y2 = n[idx - 1], n[idx], n[idx + 1]
        dx = x[1] - x[0]
        delta = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2 + 1e-30)
        x_soliton.append(float(x[idx] + delta * dx))
        depth.append(float(y1)) # think about better extrapolation

    if n_expected is not None and len(x_soliton) > n_expected:
        # Sort by depth and take the n_expected deepest minima
        sorted_indices = np.argsort(depth)[:n_expected]
        x_soliton = x_soliton[sorted_indices]

    return np.sort(x_soliton)

def find_soliton_position(state: GPState, comp: int = 0) -> float:
    """
    Find the position of a single soliton as the minimum of the density

    Parameters
    ----------
    state: GPState
        The wavefunction state
    comp: int
        Component
    min_depth: float
        Minimum fractional density dip |1-n_min/n_0| (filters noise)

    Returns
    -------
    x_soliton: float
        Position of the soliton in real space
    """

    n = state.density(comp)  # Density profile

    idx_min = np.argmin(n)  # Index of minimum density
    x_soliton = state.grid.x[idx_min]  # Corresponding position
    
    return x_soliton

def superfluid_fraction(state, comp=0):
    n, L, dx = state.density(comp), state.grid.L, state.grid.dx
    n_bar     = np.sum(n) * dx / L
    inv_n_bar = np.sum(1.0 / n) * dx / L
    return float(1.0 / (n_bar * inv_n_bar))