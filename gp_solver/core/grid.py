"""
core/grid.py

Spatial and momentum grid on which all calculations are run

Convention: 
dimensionaless units
    length scale: xi = hbar / sqrt(2*m*g*n_0)
    speed of sound: c_s = sqrt(g*n_0/m)
    time scale: tau = xi /c_s
    energy scale: g*n_0

Then GPE is
    i d/dt psi = -1/2 d^2/dx^2 psi + |psi|^2 psi

coordinates:
x[i] = -L/2 + i*dx,     i=0,...,N-1
k[j] = 2pi*j_shifted / L (FFT shifted wavenumber)
k^2 = k*k (kinetic energy in k-space)
)
    
"""

from __future__ import annotations
import numpy as np
from scipy.fft import fftfreq, fft, ifft, fft2, ifft2
from dataclasses import dataclass, field

@dataclass # Initiates base functions for data storing classes
class Grid:
    """
    1D or 2D grid with FFT compatible k-vectors:

    Parameter:
    ----------
    N: int
        Number of grid points (2^x for FFT performance)
    L: float
        Physical box length (dimensionless)

    Derived attributes:
    -------------------
    dx: float
        grid spacing
    x: (N,) array
        real space coordinates
    k: (N,) array
        wavenumbers
    k2: (N,) array
        k^2 kinetic energy factor
    """

    # Default values
    N: int = 1024
    L: float = 100

    # Derived attributes (populated in __post_init__)
    dx:  float       = field(init=False, repr=False)
    x:   np.ndarray  = field(init=False, repr=False)
    k:   np.ndarray  = field(init=False, repr=False)
    k2:  np.ndarray  = field(init=False, repr=False)

    #########################################################

    def _build_grid(self):
        N, L = self.N, self.L #local copies
        self.dx  = L / N

        # x runs from -L/2 to L/2 : dx  (periodic)
        self.x  = np.linspace(-L / 2.0, L / 2.0 - self.dx, N)
        # fftfreq returns frequencies, multiply by 2pi for k-vec
        self.k  = 2.0 * np.pi * fftfreq(N, d=self.dx)
        self.k2 = self.k ** 2

    # Initialization
    def __post_init__(self):
        if self.N & (self.N - 1) != 0: #binary and
            import warnings
            warnings.warn(
                f"N={self.N} is not a power of 2. FFT will be slower.",
                stacklevel=3,
            )
        self._build_grid()


    # FFT wrappers (no direct imports for callers needed)
    #####################################################
    
    def fft(self, x: np.ndarray) -> np.ndarray:
        # Forward FFT (using scipy.fft)
        return fft(x)

    def ifft(self, k: np.ndarray) -> np.ndarray:
        # Inverse FFT (using scipy.ifft)
        return ifft(k)

    # Utilities
    ###########

    def kinetic_phase_factor_notrotating(self, dt: float, hbar_over_2m: float = 0.5) -> np.ndarray:
        """
        Pre compute the kinetic propagator e^[ -i (h/2m) k^2 dt] 
        For dimensionless (h=m=1), h/2m = 0.5

        Parameter:
        ----------
        dt: float
            Timestep (negative -> imaginary)
        hbar_over_2m: float
            = 0.5 in dimensionaless units, could be changed

        Returns
        -------
        (N,) complex128 array
        """

        phase = -1j * hbar_over_2m * self.k2 * dt
        return np.exp(phase)

    def kinetic_phase_factor_rotating(self, dt: float, omega: float, hbar_over_2m: float = 0.5) -> np.ndarray:
        """
        Pre compute the kinetic propagator e^[ -i ((h/2m) k^2) -omega k) dt] 
        For dimensionless (h=m=1), h/2m = 0.5

        Parameter:
        ----------
        dt: float
            Timestep (negative -> imaginary)
        omega: float
            rotation frequency
        hbar_over_2m: float
            = 0.5 in dimensionaless units, could be changed

        Returns
        -------
        (N,) complex128 array
        """

        phase = -1j * (hbar_over_2m * self.k2  - omega * self.k)* dt
        return np.exp(phase)

    def kinetic_damp_factor(self, dtau: float, hbar_over_2m: float = 0.5) -> np.ndarray:
        """Imaginary time damping e^[-(h/2m) k^2 dtau]

        Parameter
        ---------
        dtau: float
            damping factor
       hbar_over_2m: float
                   = 0.5 in dimensionaless units, could be changed
        
        Returns
        -------
        (N,) float64 array
        """

        return np.exp(-hbar_over_2m * self.k2 * dtau)

    def kinetic_damp_factor_rotating(self, dtau: float, omega: float, hbar_over_2m: float = 0.5) -> np.ndarray:
        """Imaginary time damping e^[-(h/2m) k^2 dtau]

        Parameter
        ---------
        dtau: float
            damping factor
        omega: float
            rotation frequency
        hbar_over_2m: float
            = 0.5 in dimensionaless units, could be changed

        Returns
        -------
        (N,) float64 array
        """

        return np.exp(-hbar_over_2m * (self.k2 - 2j * omega * self.k) * dtau)

    # For representing object when printing
    # Only N, L, dx, kmax important
    def __repr__(self) -> str:
        return (
            f"Grid(N={self.N}, L={self.L:.2f}, dx={self.dx:.4f}, "
            f"kmax={self.k.max():.3f})"
        )