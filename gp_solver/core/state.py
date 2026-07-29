"""
core/state.py

GPState class for representing a wavefunction on a grid,
stored as a C-contigous float64 array.
Real and Imaginary parts are stored separetely.

For Scalar:
psi_r[0,..,N-1], psi_i[0,..,N-1]

For Spinor (F=1):
psi_r[0,..,3N-1] = [psi_{+1}_r, psi_{0}_r, psi_{-1}_r]
psi_i[0,..,3N-1] = [psi_{+1}_i, psi_{0}_i, psi_{-1}_i]

The t tracks physical time. 
History optionally stores (t,observable) for plotting.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

from gp_solver.core.grid import Grid

@dataclass
class GPState:
    """
    Wavefunction state for the GPE

    Parameters
    ----------
    grid: Grid
        Spatial grid
    n_comp: int
        Number of spinor components:  1 = scalar, 3 = (F=1)-spinor
    psi_r: (n_comp*N,) float64 array
        Real part of the wavefunction (C-contiguous)
    psi_i: (n_comp*N,) float64 array
        Imaginary part of the wavefunction (C-contiguous)
    t: float
        Current physical time
    label: str
        label (for plot titles or file names)
    history: list of dict(str, Any) (optional)
        History of the state (t, observable) (for plotting through propagator)
    """

    # Default values
    grid:   Grid
    n_comp: int = 1
    psi_r:  np.ndarray = field(default=None, repr=False)
    psi_i:  np.ndarray = field(default=None, repr=False)
    t:      float = 0.0
    label:  str = "GPState"
    history: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    # Initialization
    def __post_init__(self):
        total = self.n_comp * self.grid.N
        # Initialize psi arrays if not provided
        if self.psi_r is None:
            self.psi_r = np.zeros(total, dtype=np.float64)
        if self.psi_i is None:
            self.psi_i = np.zeros(total, dtype=np.float64)

        # Enforce C-contiguity and float64 dtype
        self.psi_r = np.ascontiguousarray(self.psi_r, dtype=np.float64)
        self.psi_i = np.ascontiguousarray(self.psi_i, dtype=np.float64)

        # check for correct length
        expected = self.n_comp * self.grid.N
        if len(self.psi_r) != expected or len(self.psi_i) != expected:
            raise ValueError(
                f"psi arrays must have length n_comp*N = {expected}, "
                f"got {len(self.psi_r)} and {len(self.psi_i)}"
            )

    # Complex number helpers
    ########################

    def psi_complex(self, comp: int = 0) -> np.ndarray:
        """
        Return a copy as a complex128 of the component 'comp'.

        Parameters
        ----------
        comp: int
            Component index (0=m+1, 1=m0,2 = m-1 for F=1 spinor), 0 = scalar
        """

        N = self.grid.N
        r = self.psi_r[comp*N:(comp+1)*N]
        i = self.psi_i[comp*N:(comp+1)*N]
        return r + 1j * i

    def set_component(self, comp: int, psi: np.ndarray) -> None:
        """
        Write a complex128 array into the component 'comp'.

        Parameters
        ----------
        comp: int
            Component index (0=m+1, 1=m0,2 = m-1 for F=1 spinor), 0 = scalar
        psi: (N,) complex128 array or float64 array
            New wavefunction
        """

        N = self.grid.N
        psi = np.asarray(psi)
        if np.iscomplexobj(psi):
            self.psi_r[comp * N: (comp + 1) * N] = psi.real.astype(np.float64)
            self.psi_i[comp * N: (comp + 1) * N] = psi.imag.astype(np.float64)
        else:
            self.psi_r[comp * N: (comp + 1) * N] = psi.astype(np.float64)
            self.psi_i[comp * N: (comp + 1) * N] = 0.0

    def psi_all_complex(self) -> np.ndarray:
        """
        Return all components as a (n_comp, N) complex128 array.

        Returns
        -------
        (n_comp, N) complex128 array
        """

        N = self.grid.N
        out = np.empty((self.n_comp, N), dtype=np.complex128)
        # Component for component
        for m in range(self.n_comp):
            out[m] = self.psi_complex(m)
        return out

    # Computations
    ################

    def nonlinear_phase_scalar(self, dt, g):
        """
        apply_nonlinear_phase_scalar(psi_r, psi_i, n, dt, g)
        e^{-i g |psi|² dt} applied in-place on the real/imag arrays.
        Used during half-step in Strang splitting.
        """
        density  = self.density()
        phase    = -g * density * dt
        c, s     = np.cos(phase), np.sin(phase)
        new_r    = self.psi_r*c - self.psi_i*s
        self.psi_i[:] = self.psi_r*s + self.psi_i*c
        self.psi_r[:] = new_r


    def nonlinear_damp_scalar(self, dtau, g):
        """
        e^{- g |psi|² dtau} applied in-place on the real/imag arrays.
        Used during imaginary time evolution.
        """
        density  = self.density()
        factor   = np.exp(-g * density * dtau)
        self.psi_r   *= factor
        self.psi_i   *= factor

    # Density and normalization helpers
    ####################################

    def density(self, comp: int = 0) -> np.ndarray:
        """
        Return the density of component

        Parameters
        ----------
        comp: int
            Component index (0=m+1, 1=m0,2 = m-1 for F=1 spinor), 0 = scalar

        Returns
        -------
        (N,) float64 array
        """

        N = self.grid.N
        r = self.psi_r[comp*N:(comp+1)*N]
        i = self.psi_i[comp*N:(comp+1)*N]
        return r*r + i*i

    def total_density(self) -> np.ndarray:
        """
        Total density of all components

        Returns
        -------
        (N,) float64 array2 
        """
        n = np.zeros(self.grid.N, dtype=np.float64)
        for m in range(self.n_comp):
            n += self.density(m)
        return n

    def phase(self, comp: int = 0) -> np.ndarray:
        """
        Return the phase of component in (-pi,pi]
        Unwrapped version in analysis.soliton.unwrapped_phase()

        Parameters
        ----------
        comp: int
            Component index (0=m+1, 1=m0,2 = m-1 for F=1 spinor), 0 = scalar

        Returns
        -------
        (N,) float64 array
        """
        # Computational expensive, maybe C KERNEL

        N = self.grid.N
        r = self.psi_r[comp*N:(comp+1)*N]
        i = self.psi_i[comp*N:(comp+1)*N]
        return np.arctan2(i, r)

    # Norm
    def norm(self, comp: int = None) -> float:
        """
        Compute the norm of the wavefunction.

        If comp = None, compute the total norm of all components.
        Else only of that specific component.
        """

        dx = self.grid.dx
        if comp is None:
            return float(np.sum(self.total_density()) * dx)
        else:
            return float(np.sum(self.total_density(comp)) * dx)

    # Renormalizes the wavefunction to a given norm N0
    def renormalize(self, N0, comp: int = None):
        norm     = self.norm(comp)
        factor   = np.sqrt(N0 / norm)
        if comp is None:
            self.psi_r *= factor
            self.psi_i *= factor
        else:
            self.psi_r[comp*self.grid.N:(comp+1)*self.grid.N] *= factor
            self.psi_i[comp*self.grid.N:(comp+1)*self.grid.N] *= factor

    # Copy and snapshots
    ####################

    def copy(self) -> GPState:
        """
        Return a copy of the GPState object.
        """
        return GPState(
            grid=self.grid,
            n_comp=self.n_comp,
            psi_r=self.psi_r.copy(),
            psi_i=self.psi_i.copy(),
            t=self.t,
            label=self.label + '_copy',
        )

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a snapshot of key observable.
        Used by propagator to fill state.history.

        Returns
        -------
        dict
            Dictionary containing evolution of observable(s).
        """
        snap = {
            "t":    self.t,
            "norm": self.norm(),
        }
        # For spinor per component
        if self.n_comp > 1:
            snap["norms"] = [self.norm(m) for m in range(self.n_comp)]
        return snap

    # Representation
    ####################
    def __repr__(self) -> str:
        return (
            f"GPState('{self.label}', n_comp={self.n_comp}, "
            f"N={self.grid.N}, t={self.t:.4f}, norm={self.norm():.6f})"
        )