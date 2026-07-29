"""
systems/base_system.py
Abstract base class for a GPE system (scalar or spinor)



Minimal interface
------------------
.state
._default_stat()
.nonlinear_halfstep()
.mu

Optional
-----------------
.V_ext
.g
"""

from __future__ import annotations
from abc import ABC, abstractmethod # abstract base class
from typing import Optional
import numpy as np

from gp_solver.core.grid  import Grid
from gp_solver.core.state import GPState

class BaseGPSystem(ABC):
    """
    Abstract base class for a GPE system (scalar or spinor)
    """

    def __init__(self, grid: Grid):
        self.grid  = grid
        self.state = self._default_state()
        self.V_ext: Optional[np.ndarray] = None
        self.mu: float = 1.0
        self.g = 1.0

    
    @abstractmethod
    def _default_state(self) -> GPState:
        """
        Return a default initial state for the system (wavefunction + grid)
        """
        pass

    @abstractmethod
    def nonlinear_halfstep(self, dt: float, imag_time: bool = False) -> None:
        """
        Apply the nonlinear half-step of the GPE propagator.
        """
        pass

    def print_summary(self) -> None:
        """
        Print a summary of the system's parameters and state.
        """
        print("="*60)
        print(f"Grid: {self.grid}")
        print(f"State: {self.state}")
        print(f"External potential: {self.V_ext}")
        print(f"Chemical potential: {self.mu}")
        print(f"Interaction strength: {self.g}")