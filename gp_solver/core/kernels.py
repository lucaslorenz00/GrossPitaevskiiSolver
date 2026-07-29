# Probably obsolete, but keep for now

"""
core/kernels.py

Numerically used inner loops for the GPE propagator:
-----------------------------------------------------
apply_nonlinear_phase_scalar(psi_r, psi_i, n, dt, g)
    e^{-i g |psi|² dt} applied in-place on the real/imag arrays.
    Used during half-step in Strang splitting.
    
apply_nonlinear_phase_spinor(psi_r, psi_i, n_comp, n_pts, dt, c0, c1, p_lin, q_quad)
    Full F=1 spinor nonlinear phase + spin-mixing step, in-place.
    # Not implemented yet

apply_nonlinear_damp_scalar(psi_r, psi_i, n, dtau, g)
    e^{- g |psi|² dtau} applied in-place on the real/imag arrays.
    Used during imaginary time evolution.

compute_density(psi_r, psi_i, n_pts, density_out)
    |psi|² written to density_out array, in-place.

energy_density_scalar(psi_r, psi_i, dpsi_r, dpsi_i, n_pts, g, out)
    Local energy density e(x) = 1/2|d/dx psi|² + 1/2 g|psi|⁴  (k-space)
"""

import numpy as np

# moved into state
def nonlinear_phase_scalar(psi_r, psi_i, dt, g):
    density  = psi_r**2 + psi_i**2
    phase    = -g * density * dt
    c, s     = np.cos(phase), np.sin(phase)
    new_r    = psi_r*c - psi_i*s
    psi_i[:] = psi_r*s + psi_i*c
    psi_r[:] = new_r

# moved into state
def nonlinear_damp_scalar(psi_r, psi_i, dtau, g):
    density  = psi_r**2 + psi_i**2
    factor   = np.exp(-g * density * dtau)
    psi_r   *= factor
    psi_i   *= factor

# moved into state
def renormalize(psi_r, psi_i, dx, N0):
    norm     = np.sum(psi_r**2 + psi_i**2) * dx
    factor   = np.sqrt(N0 / norm)
    psi_r   *= factor
    psi_i   *= factor
