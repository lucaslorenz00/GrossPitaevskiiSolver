"""
system/scalar_gp.py
Scalar GPE system class, derived from BaseGPSystem.

Describing dark and black solitons

Implements 1D scalar GPE: i d_t psi = [-1/2 d_xx + V_ext+ g |psi|^2] psi
(h=m=1, lengths in units of xi = 1/sqrt{2gn_0})

stationary solution: psi = tanh(x/sqrt{2*xi})
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
# from typing import Optional

from gp_solver.core.grid       import Grid
from gp_solver.core.state      import GPState
from gp_solver.systems.base_system import BaseGPSystem

@dataclass
class ScalarGPParams:
    """
    Physical parameters for the scalar GPE system.

    Parameters
    ----------
    g: float
        Nonlinear interaction strength (dimensionless)
        g > 0 repulsive (supports dark solitons)
        g < 0 attractive
    N0: float
        Total number of particles (norm of the wavefunction)
    V0: float
        Amplitude of the external potential (dimensionless)
        V = 1/2 V0 x^2
        v0 = 0.0 for free space, uniformbackground, periodic BEC
    soliton_position: float
        Initial position of the soliton (dimensionless) (0.0 center)
    soliton_velocity: float
        Initial velocity of the soliton (dimensionless)
        (v/c_s in [0,1) )
    n_solitons: int
        Number of solitons in the initial state (placed symmetrically)
    noise_amplitude: float
        Optional noise added to the initial state
        (e.g. for Bogoliubov mode)
    """

    g: float = 1.0
    N0: float = None # computed from L if None
    V0: float = 0.0
    soliton_position: float = 0.0
    soliton_velocity: float = 0.0 # If later more than one soliton
    n_solitons: int = 1
    noise_amplitude: float = 0.0
    V_ext: np.ndarray = None # To save external potential, but maybe unused
    #mu: float = 1.0

class ScalarGPSystem(BaseGPSystem):
    """
    Scalar GPE system class, derived from BaseGPSystem.

    Ussage:
    -----------
    system = ScalarGPSystem.black_soliton_default()
    prop = SplitStepPropagator(system)
    prop.run_imaginary_time_evolution(steps) (e.g. steps = 2000)
    prop.run_real_time_evolution(t_end, dt) (e.g. t_end = 30.0, dt = 5e-3)
    
    Parameters
    -----------
    grid: Grid
        Spatial grid
    params: ScalarGPParams
        Physical parameters for the scalar GPE system
    """

    def __init__(self, grid: Grid, params: ScalarGPParams):
        self.params = params
        super().__init__(grid)
        self.g = params.g
        if self.params.V0 != 0.0:
            self.V_ext = 0.5 * params.V0 * self.grid.x**2
        else:
            self.V_ext = None

    # BaseGPSystem interface
    ##########################

    def _default_state(self) -> GPState:
        """
        Build the initial state for the scalar GPE system.

        This is where the physics is defined

        Steps:
        1. Start with analytical dark soliton solution (tanh)
        2. Add optional noise
        3. Imaginary time evolution to relax to true ground state
        """
        grid = self.grid
        params = self.params
        N = grid.N
        x = grid.x
        L = grid.L

        # Bulk density n0 = 1 in dimensionless units
        n0 = 1.0
        # Default norm = n0 * L
        N0 = params.N0 if params.N0 is not None else n0 * L

        psi = self._build_single_soliton(x, params.soliton_position, params.soliton_velocity, n0, params.g)

        # Noise
        if params.noise_amplitude > 0.0:
            rng  = np.random.default_rng()
            noise = params.noise_amplitude * (
                rng.standard_normal(N) + 1j * rng.standard_normal(N)
            )
            psi = psi + noise

        # Normalize to target norm N0
        psi = psi * np.sqrt(N0 / np.sum(np.abs(psi)**2) / grid.dx)

        state = GPState(
            grid=grid,
            n_comp=1,
            label="ScalarGPState")
        state.set_component(0,psi)
        return state

    def nonlinear_halfstep(self, dt: float, imag_time: bool= False) -> None:
        """
        For applying the scalar nonlinear half-step e^{-i g |psi|^2 dt})
        
        Parameters
        -----------
        dt: float
            Time step (propagator already passes dt/2)
        imag_time: bool
            If True, use imaginary time evolution (damping)
        """

        state = self.state
        g = self.params.g
        if imag_time:
            # Imaginary time evolution (damping)
            state.nonlinear_damp_scalar(dt, g)
        else:
            # Real time evolution (phase)
            state.nonlinear_phase_scalar(dt, g)

    # Printing method
    ##################

    def print_summary(self) -> None:
        """
        Print a summary of the system's parameters and state.
        """
        p = self.params
        super().print_summary()
        print(f"ScalarGPSystem\n"
        f"  g = {p.g}\n"
        f"  n_solitons = {p.n_solitons}\n"
        f"  v/c_s = {p.soliton_velocity:.3f}\n"
        f"  soliton_pos = {p.soliton_position:.2f}\n"
        f"  trap V0 = {p.V0}\n"
        f"  grid N = {self.grid.N},  L = {self.grid.L}"
        f" N0 = {p.N0}")
        
    # Methods for setup
    ###################

    @classmethod
    def black_soliton_default(cls, N:int = 1024, L: float = 80.0) -> "ScalarGPSystem":
        """
        Build a default scalar GPE system with a black soliton in the center and g=1.
        Parameters
        ----------
        N: int
            Number of grid points (power of 2 for fast FFT)
        L: float
            box length (in units of xi)

        Returns
        -------
        ScalarGPSystem
            A scalar GPE system with a black soliton in the center.
        """
        grid = Grid(N=N, L=L)
        params = ScalarGPParams(
            g=1.0,
            soliton_position=0.0,
            soliton_velocity=0.0,
            n_solitons=1,
        )
        return cls(grid, params)

    @staticmethod
    def _build_single_soliton(x: np.ndarray, x0: float, v: float, n0: float, g: float = 1.0) -> np.ndarray:
        """
        Analytic dark soliton wavefunction.

        psi(x) = sqrt(n0) * [iv/c_s + sqrt(1-v^2/c_s^2) tanh((x-x0)/sqrt(2*xi))]
        (alternative representation:
        psi_v(x) = sqrt(n0) * [iv/c_s + cos(theta) tanh((x-x0)/xi_v)])
        with xi_v = xi / cos(theta), cos(theta) = sqrt(1-v^2/c_s^2), c_s = sqrt(g*n0/m) = 1 in dimensionless units, xi = 1/sqrt(2*g*n0)

        Parameters
        ----------
        x: np.ndarray
            Spatial grid positions
        x0: float
            Soliton position
        v: float
            Soliton velocity (in units of c_s, v/c_s in [0,1))
        n0: float
            Bulk density
        g: float
            Nonlinear interaction strength (dimensionless)

        Returns
        -------
        (N,) complex128 np.ndarray
            Complex wavefunction of the single dark soliton.
        """
        cs = np.sqrt(g * n0)  # speed of sound
        beta = v/cs
        if np.abs(beta) >= 1.0:
            raise ValueError(f"Soliton velocity v/c_s must be in [0,1), got {beta}")

        cos_th = np.sqrt(1.0 - beta**2)
        xi_v = 1.0 / (np.sqrt(2 * g * n0) * cos_th)  # Lorentz contracted width

        tanh = cos_th * np.tanh((x - x0) / xi_v)

        psi = np.sqrt(n0) * (1j * beta + tanh).astype(np.complex128)
        return psi
