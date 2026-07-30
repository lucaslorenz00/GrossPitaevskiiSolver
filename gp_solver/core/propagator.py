"""
core/propagator.py


"""

from __future__ import annotations
import numpy as np
from typing import Callable, Optional, List
import time as _time_module
import matplotlib.pyplot as plt # only for analysis
from gp_solver.utils.plotting import plot_density_phase # only for anaylsis right now
import gp_solver.analysis.diagnostics as diagnostics

#from gp_solver.core.grid      import Grid
from gp_solver.core.state     import GPState

class SplitStepPropagator:
    """
    Split-step propagator for the GPE

    Propagator does not know the system and calls back into the object
    and does nonlinear (interaction) half steps, so same propagator can
    handle both scalar and spinor GPEs.

    Parameters
    ----------
    system: Base GP system (Any object)
        Spinor or scalar
        Must implement methods in systems/base_systems.py
        Must provide:
            .state: GPState
            .nonlinear_half_step(dt, imag) -> None
            .external_potential_phase(dt) -> None (optional)
            .mu: float (chemical potential, Used for damping)

    gamma: float
        Damping parameter (dimensionless) (0.0 for deterministic)
    grid: Grid
        Spatial grid
    temp: float
        Temperature (dimensionless) (only for gamma > 0)
    rng: np.random.Generator
        Random number generator (for Langevin noise)
    kin_damp: (N,) array float64
        Precomputed kinetic damping factor for imaginary time evolution
    kin_phase: (N,) array complex128
        Precomputed kinetic phase factor for real time evolution
    """

    def __init__(
        self,
        system,
        gamma:       float                    = 0.0,
        temperature: float                    = 0.0,
        rng:         np.random.Generator      = None,
    ):
        self.system      = system
        self.state       = system.state
        self.grid        = system.state.grid
        self.gamma       = gamma
        self.temperature = temperature
        self.rng         = rng or np.random.default_rng() # default if None given
        self.omega = system.params.omega # rotation frequency
        print(self.omega)

        # Precompute external potential phase once (time-independent traps)
        self._V_ext = getattr(system, "V_ext", None)   # (N,) array or None
        # self.soliton_velocity = system.params.soliton_velocity if hasattr(system, "params") else 0.0

        # Precomputing kinetic factors now in run_imaginary_time() and run_real_time() with correct dt, dtau
        # self.kin_damp = self.grid.kinetic_damp_factor(1.0).real # float64
        # self.kin_phase = self.grid.kinetic_phase_factor_notrotating(1.0) 

        # Diagnostics callback: called every record_every steps during time steps
        self._diag_callbacks: List[Callable[[GPState], dict]] = []


    # Public methods
    #######################

    def add_diagnostic(self, fn: Callable[[GPState], dict]) -> None:
        """
        Add a diagnostic callback, called every snapshot step.

        fn must accept a GPState and return a dict of observables,
        results are stored in state.history.
        """
        self._diag_callbacks.append(fn)

    def run_imaginary_time(
        # Default values
        self,
        steps:          int   = 3000,
        dtau:           float = 1e-2,
        conv_tol:       float = 1e-10,
        record_every:   int   = 100,
        verbose:        bool  = True,
    ) -> GPState:
        """
         Imaginary time evolution: tau = -it to find the ground state.
        Might find saddle point.

        The wavefunction is renormalized to N_0 after every step to prevent collapse.

        Parameters
        ----------
        steps: int
            maximum number of steps
        dtau: float
            imaginary timestep (stability: dtau < dx^2/2)
        conv_tol: float
            relative convergence threshold on mu
        record_every: int
            how often to record snapshot and check convergence
        verbose: bool
            print detailed progress

        Returns
        -------
        GPState: final (ground) state
        """
        state = self.state
        grid  = self.grid
        N0    = state.norm()      # target norm (conserved)
        energy = []
        print(f"Using rotating frame kinetic damping factor with omega={self.omega}") # check if omega is correct
        print(f"using v = {self.system.params.soliton_velocity}")
        if self.omega != 0.0:
            self.kin_damp = self.grid.kinetic_damp_factor_rotating(dtau, self.omega)
            # print(self.grid.kinetic_damp_factor(dtau))
            # print(f"Difference in kin damp: {(np.abs(self.kin_damp - grid.kinetic_damp_factor(dtau)).max())/np.abs(self.kin_damp).max()}")
        else:
            self.kin_damp = grid.kinetic_damp_factor(dtau)
        # plt.plot(self.kin_damp.real, label="Real part") # Just to check the kinetic damping factor

        if verbose:
            print(f"[Imaginary time] N0={N0:.4f}, dtau={dtau:.2e}, "
                  f"max_steps={steps}")

        # Precompute k-space kinetic damping factor (real, time-independent)
        # Precomputation moved to __init__ to avoid downpassing
        # kin_damp = grid.kinetic_damp_factor(dtau).real    # (N,) float64

        # mu_prev   = None
        t_start   = _time_module.time()

        for step in range(1, steps + 1):
            energy.append(diagnostics.total_energy_scalar(state, self.system.g))
            self._step_imaginary(dtau, N0)

            step_to_plot = [100, 1000, 2000]
            if step in step_to_plot: 
                fig, ax = plot_density_phase(
                    state,
                    title=f"Imaginary time step {step}  (t={state.t:.3f})",
                    show_analytical=False,
                    g=self.system.params.g,
                )
                plt.show()

            if step % record_every == 0:
                snap = state.snapshot()
                # Estimate chemical potential
                # mu_est = self._estimate_mu_imtime(dtau) # Change mu estimates, and mu estimates not needed
                # snap["mu"] = mu_est
                # state.history.append(snap)

                # Just for verbose output
                if verbose:
                    elapsed = _time_module.time() - t_start
                    print(f"  step {step:6d}/{steps}  |  "
                          f"norm={snap['norm']:.6f}  |  "
                          #f"μ≈{mu_est:.6f}  |  "
                          f"t={elapsed:.1f}s")
                """
                # Taken out for now
                if mu_prev is not None:
                    rel_change = abs(mu_est - mu_prev) / (abs(mu_prev) + 1e-30)
                    if rel_change < conv_tol:
                        if verbose:
                            print(f"[Imaginary time] Converged at step {step} "
                                  f"(Δμ/μ = {rel_change:.2e})")
                        break

                mu_prev = mu_est
                """

        #self.system.mu = mu_prev or 1.0
        plt.plot(np.arange(len(energy)), energy)
        plt.xlabel("Step")
        plt.ylabel("Energy")
        plt.title("Imaginary Time Evolution")
        plt.show()
        return state

    def run_real_time(
        self,
        t_end: float,
        dt: float = 1e-3,
        record_every: int   = 50,
        verbose: bool  = True,
    ) -> GPState:
        """
        Real time evolution from state.t to t_end using Strang splitting.

        Parameters
        ----------
        t_end: float
            target end time
        dt: float
            timestep (should satisfy Courant: dt < dx/c_s)
        record_every: int
            how often to record snapshot and check convergence
        verbose: bool
            print detailed progress

        Returns
        -------
        GPState: final state
        """

        state = self.state
        grid  = self.grid

        n_steps = int(np.ceil((t_end - state.t) / dt))
        if n_steps <= 0:
            raise ValueError(f"t_end={t_end} must be > state.t={state.t}")

        if verbose:
            print(f"[Real time] t: {state.t:.4f} → {t_end:.4f}, "
                  f"dt={dt:.2e}, n_steps={n_steps}")
            print(f"  Langevin γ={self.gamma:.3g}, "
                  f"kT={self.temperature:.3g}")

        # Precompute kinetic phase factor (complex, for Strang splitting)
        # Precomputation moved to __init__ to avoid downpassing
        if self.omega != 0.0:
            self.kin_phase = grid.kinetic_phase_factor_rotating(dt, self.omega)   # (N,) complex128
        else:
            self.kin_phase = grid.kinetic_phase_factor_notrotating(dt)   # (N,) complex128

        t_start = _time_module.time()

        for step in range(1, n_steps + 1):
            self._step_real(dt)
            state.t += dt

            if self.gamma > 0.0:
                self._apply_langevin(dt)

            if step % record_every == 0:
                snap = state.snapshot()
                for fn in self._diag_callbacks:
                    snap.update(fn(state)) # Add diagnostics to snapshot
                state.history.append(snap)

                if verbose:
                    elapsed = _time_module.time() - t_start
                    print(f"  step {step:7d}/{n_steps}  |  "
                          f"t={state.t:.4f}  |  "
                          f"norm={snap['norm']:.6f}  |  "
                          f"wall={elapsed:.1f}s")

        return state

    # Internal step methods
    #######################

    def _step_imaginary(
        self,
        dtau: float,
        N0: float,
    ) -> None:
        """
        One imaginary-time Strang step:
            1. Nonlinear  half-step (damping) and potential
            2. Full kinetic step in k-space
            3. Nonlinear  half-step (damping) and potential
            4. Renormalize to N0
        """
        
        sys = self.system
        # state = self.state
        # Nonlinear half-step
        sys.nonlinear_halfstep(dtau / 2.0, imag_time=True)
        # External potential half-step
        if self._V_ext is not None:
            self._apply_V_ext_imag(dtau / 2.0)
        # Full kinetic step in k-space
        self._apply_kinetic_imag()
        # External potential half-step
        if self._V_ext is not None:
            self._apply_V_ext_imag(dtau / 2.0)
        # Nonlinear half-step
        sys.nonlinear_halfstep(dtau / 2.0, imag_time=True)
        # Renormalize
        sys.state.renormalize(N0)

    # ------------------------------------------------------------------
    def _step_real(
        self,
        dt:        float,
    ) -> None:
        """
        One real-time Strang step:
            1. Nonlinear half-step (phase rotation) and potential
            2. Full kinetic step in k-space
            3. Nonlinear half-step (phase rotation) and potential
        """
        # Nonlinear half-step
        self.system.nonlinear_halfstep(dt / 2.0, imag_time=False)
        # External potential half-step
        if self._V_ext is not None:
            self._apply_V_ext_real(dt / 2.0)
        # Full kinetic step in k-space
        self._apply_kinetic_real()
        # External potential half-step
        if self._V_ext is not None:
            self._apply_V_ext_real(dt / 2.0)
        # Nonlinear half-step
        self.system.nonlinear_halfstep(dt / 2.0, imag_time=False)

    def _apply_kinetic_real(self) -> None:
        """
        Apply e^[-i T_k dt} in k-space
        For n_comp components, each is transformed independently
        """
        state = self.state
        N     = self.grid.N
        grid  = self.grid

        for m in range(state.n_comp):
            psi_c  = state.psi_complex(m)          # (N,) complex128
            psi_k  = grid.fft(psi_c)               # forward FFT
            psi_k *= self.kin_phase                     # multiply by phase factor
            psi_c  = grid.ifft(psi_k)              # inverse FFT
            state.psi_r[m * N: (m + 1) * N] = np.ascontiguousarray(psi_c.real)
            state.psi_i[m * N: (m + 1) * N] = np.ascontiguousarray(psi_c.imag)

    def _apply_kinetic_imag(self) -> None:
        """
        Apply e^[-T_k dtau] in k-space (imaginary time)
        """
        state = self.state
        N     = self.grid.N
        grid  = self.grid

        for m in range(state.n_comp):
            psi_c  = state.psi_complex(m)
            psi_k  = grid.fft(psi_c)
            psi_k *= self.kin_damp                      # real damping
            psi_c  = grid.ifft(psi_k)
            state.psi_r[m * N: (m + 1) * N] = np.ascontiguousarray(psi_c.real)
            state.psi_i[m * N: (m + 1) * N] = np.ascontiguousarray(psi_c.imag)

    def _apply_V_ext_real(self, dt_half: float) -> None:
        """
        Apply external potential phase  e^[-i V_ext dt/2]
        V_ext is a (N,) real array
        """
        V     = self._V_ext
        phase = np.exp(-1j * V * dt_half) # dt/2 is already inputed
        state = self.state
        N     = self.grid.N

        for m in range(state.n_comp):
            psi_c = state.psi_complex(m) * phase # phase is complex
            state.psi_r[m * N: (m + 1) * N] = np.ascontiguousarray(psi_c.real)
            state.psi_i[m * N: (m + 1) * N] = np.ascontiguousarray(psi_c.imag)

    def _apply_V_ext_imag(self, dtau_half: float) -> None:
        """
        Apply external potential damping e^[-V_ext dtau/2]
        V_ext is a (N,) real array
        """
        V      = self._V_ext
        factor = np.exp(-V * dtau_half)
        state  = self.state
        N      = self.grid.N

        for m in range(state.n_comp):
            sl = slice(m * N, (m + 1) * N) # slice for component m
            state.psi_r[sl] *= factor
            state.psi_i[sl] *= factor

    # Implement later if of interest
    def _apply_langevin(self, dt: float) -> None:
        """
        Stochastic GP (PGPE) noise + damping step after the deterministic step

        Noise amplitude: sqrt(2 gamma k_B T dt)
        Damping: projects each momentum mode toward equilibrium.
        Only applied to the low-k sector (|k| < k_cut = k_max/2) to
        implement the projector P in the PGPE.
        """
        gamma = self.gamma
        kT    = self.temperature
        dt_   = dt
        amp   = np.sqrt(2.0 * gamma * kT * dt_)

        state = self.state
        N     = self.grid.N

        for m in range(state.n_comp):
            sl    = slice(m * N, (m + 1) * N)
            pr    = state.psi_r[sl]
            pi    = state.psi_i[sl]
            # placeholder.add_langevin_noise(pr, pi, amp, self.rng) # change from KERNELS to somehwere else

    # Highly likely that deprecated
    def _estimate_mu_imtime_old(self, dtau: float) -> float:
        """
        Estimate chemical potential mu from the energy per particle E/N.
        This is only a rough estimate, exact mu should come from the energy functional.
        """
        # wrong formula used, probably deprecated
        state = self.state
        N   = self.grid.N
        dx  = self.grid.dx
        psi = state.psi_complex(0)

        # Kinetic energy via k-space
        psi_k  = self.grid.fft(psi)
        kin    = 0.5 * np.sum(self.grid.k2 * np.abs(psi_k) ** 2) / N * dx

        # Interaction energy
        n = np.abs(psi) ** 2
        g = getattr(self.system, "g", 1.0)
        inter = 0.5 * g * np.sum(n ** 2) * dx

        total_norm = state.norm()
        if total_norm < 1e-30:
            return 0.0
        return float((kin + inter) / total_norm)

    def _estimate_mu_imtime(self, dtau: float) -> float:
            """
            Estimate chemical potential mu from the energy per particle E/N.
            (E_kin+E_pot+2*E_int)/N
            This is only a rough estimate, exact mu should come from the energy functional.
            """
            # Needs validation which is correvt approach
            state = self.state
            N   = self.grid.N
            dx  = self.grid.dx
            psi = state.psi_complex(0)
    
            # Kinetic energy via k-space
            psi_k  = self.grid.fft(psi)
            kin    = 0.5 * np.sum(self.grid.k2 * np.abs(psi_k) ** 2) / N * dx
    
            # Interaction energy
            n = np.abs(psi) ** 2
            g = getattr(self.system, "g", 1.0)
            inter = 0.5 * g * np.sum(n ** 2) * dx

            # External potential energy
            V = getattr(self.system, "V_ext", np.zeros(N))
            pot = np.sum(V * n) * dx
    
            total_norm = state.norm()
            if total_norm < 1e-30:
                return 0.0
            return float((kin + pot + 2 * inter) / total_norm)
    
    # Implement in case mu estimate beomes important
    def _estimate_mu_imtime_decay(self) -> float:
        """
        Estimate chemical potential mu from the local norm decay rate during imaginary time. 
        Uses the relation  mu ≈ -d(ln N)/dτ / (2 dtau).
        This is only a rough estimate, exact mu should come from the energy functional.
        """

        #Needs implementation of local norm decay rate tracking during imaginary time evolution.

        state = self.state
        N   = self.grid.N
        dx  = self.grid.dx
        psi = state.psi_complex(0)

        return 0.0  # Placeholder, actual implementation needed, if actual important