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
    soliton_velocity: float = 0.0
    n_solitons: int = 1
    noise_amplitude: float = 0.0
    V_ext: np.ndarray = None # To save external potential, but maybe unused
    #mu: float = 1.0
    omega: float = 0.0 # rotation frequency (for rotating frame)
    winding: int = 0 # winding number for ring soliton (phase jump across box)

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
        if params.V_ext == None and params.V0 != 0.0:
            params.V_ext = 0.5 * params.V0 * grid.x**2

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

        n0 = 1.0 # Bulk density (dimensionless) for the initial state
        N0 = params.N0 if params.N0 is not None else n0 * L # default norm = n0 * L

        # psi = self._build_single_soliton(x, params.soliton_position, params.soliton_velocity, n0, params.g)
        # psi = self._build_gaussian(x, 0, 100)
        # psi = self._build_jacobian(x, n0)
        psi = self._build_ring_soliton(x, L, g=params.g, n0=n0, v=params.soliton_velocity, x0=params.soliton_position, winding=params.winding)
        # psi = self._build_uniform(x)  # uniform background
        # psi = self._build_ring_dark_soliton(x, L, g=params.g, n0=n0, v=params.soliton_velocity, x0=params.soliton_position, winding=params.winding)
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
    def _build_uniform(x: np.ndarray) -> np.ndarray:
        """
        Build a uniform background wavefunction with optional phase gradient.


        Returns
        -------
        (N,) complex128 np.ndarray
            Complex wavefunction of the uniform background.
        """
        return np.ones_like(x, dtype=np.complex128)/np.sqrt(x.size)

    @staticmethod
    def _build_gaussian(x: np.ndarray, x0: float, sigma: float) -> np.ndarray:
        """
        Build a Gaussian wavefunction.

        Parameters
        ----------
        x: np.ndarray
            Spatial grid positions
        x0: float
            Center position of the Gaussian
        sigma: float
            Width of the Gaussian

        Returns
        -------
        (N,) complex128 np.ndarray
            Complex wavefunction of the Gaussian.
        """
        return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)).astype(np.complex128)

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

    # A try to get along with the boundary conditions
    def _build_jacobian(self,x : np.ndarray, n0: float) -> np.ndarray:
        from scipy.special import ellipj as ej
        from scipy.special import ellipk as ek

        m = 0.99
        K = ek(m)
        lam = self.grid.L / (4*K)

        sn, cn, dn, ph = ej(x / lam, m)

        psi = np.sqrt(n0) * np.sqrt(m) * sn
        return psi

    # probably same as _build_ring_soliton, maybe slightly different tanh argument
    def _build_ring_soliton(self, x, L, g=1, n0=1, v=0.0, x0=0, winding=0)-> np.ndarray:

        if g <= 0:
            raise ValueError("g must be positive for dark solitons")
        else:
            c = np.sqrt(g*n0)

        v_rel = v/c
        if np.abs(v_rel) >= 1.0:
            raise ValueError(f"Soliton velocity v/c_s must be in [0,1), got {v_rel}")

        beta = np.sqrt(1.0-v_rel**2)

        # grey soliton
        psi = (1j*v/c + beta*np.tanh(beta*(x-x0)))

        delta_phi = 2*np.arccos(v/c) # phase kink

        k = (2*np.pi*winding - delta_phi)/L # periodic BC background flow compensation (phase gradient)

        # impose winding
        psi *= np.exp(1j*k*x)

        return np.sqrt(n0)*psi

    # probably same as _build_ring_soliton, maybe slightly different tanh argument
    def _build_ring_dark_soliton(
        self, x: np.ndarray, L: float, g: float = 1.0, n0: float = 1.0,
        v: float = 0.0, x0: float = 0.0, winding: int = 0) -> np.ndarray:

        cs = np.sqrt(g * n0)
        v_rel = v / cs
        if np.abs(v_rel) >= 1.0:
            raise ValueError(f"Soliton velocity v/c_s must be in (-1, 1), got {v_rel}")

        cos_th = np.sqrt(1.0 - v_rel**2)
        tanh_arg = cos_th * (x - x0) * np.sqrt(g * n0)

        psi_sol = np.sqrt(n0) * (1j * v_rel + cos_th * np.tanh(tanh_arg))

        delta_phi = 2.0 * np.arccos(v_rel) # phase kink

        k_bg = (2.0 * np.pi * winding - delta_phi) / L # periodic BC background flow compensation (phase gradient)

        return psi_sol * np.exp(1j * k_bg * (x - x0))

    def get_energy(self) -> float:
        """
        Compute the total energy of the system.

        Returns
        -------
        float
            Total energy of the system.
        """
        state = self.state
        psi = state.psi_complex(0)
        dx = self.grid.dx

        # Kinetic energy term: (1/2) |d_x psi|^2
        dpsi_dx = np.gradient(psi, dx)
        kinetic_energy = 0.5 * np.sum(np.abs(dpsi_dx)**2) * dx

        # Interaction energy term: (g/2) |psi|^4
        interaction_energy = 0.5 * self.g * np.sum(np.abs(psi)**4) * dx

        # External potential energy term: V_ext |psi|^2
        if self.V_ext is not None:
            potential_energy = np.sum(self.V_ext * np.abs(psi)**2) * dx
        else:
            potential_energy = 0.0
    
        return kinetic_energy + interaction_energy + potential_energy
