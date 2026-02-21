# bank/separations/mass_transfer/interface.py
"""Interface concentration calculations"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import math

from bank.core.validation import (
    check_positive, check_in_closed_01, ChemEngError, InputError
)
from bank.core.numerical import bisection
from bank.separations.equilibrium import EquilibriumModel


@dataclass
class InterfaceSpec:
    """
    Specification for interface concentration calculations.
    
    Determines the interface compositions (x_i, y_i) given:
    - Bulk compositions (x_b, y_b)
    - Mass transfer coefficients (kG, kL)
    - Equilibrium relationship
    """
    # Bulk concentrations
    x_b: float  # Liquid bulk
    y_b: float  # Gas bulk
    
    # Mass transfer coefficients
    kG: float   # Gas film coefficient
    kL: float   # Liquid film coefficient
    
    # Equilibrium
    eq: EquilibriumModel
    
    # Optional: flux if known
    N: Optional[float] = None
    
    # Numerical
    tol: float = 1e-12
    maxiter: int = 400


class InterfaceConcentration:
    """
    Solver for interface concentrations.
    
    Key equation: kG(y_b - y_i) = kL(x_i - x_b)
    with y_i = f(x_i) from equilibrium.
    """
    
    def __init__(self, spec: InterfaceSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        """Validate inputs"""
        check_in_closed_01("x_b", self.spec.x_b)
        check_in_closed_01("y_b", self.spec.y_b)
        check_positive("kG", self.spec.kG)
        check_positive("kL", self.spec.kL)
    
    def _residual(self, x_i: float) -> float:
        """Residual for interface calculation"""
        # Equilibrium at interface
        y_i = self.spec.eq.y_of_x(x_i)
        
        # Flux equality
        flux_gas = self.spec.kG * (self.spec.y_b - y_i)
        flux_liquid = self.spec.kL * (x_i - self.spec.x_b)
        
        return flux_gas - flux_liquid
    
    def _residual_with_flux(self, x_i: float) -> float:
        """Residual when flux is specified"""
        y_i = self.spec.eq.y_of_x(x_i)
        flux_gas = self.spec.kG * (self.spec.y_b - y_i)
        return flux_gas - self.spec.N
    
    def solve(self) -> Dict[str, Any]:
        """Solve for interface concentrations"""
        
        # Find x_i
        x_low = 0.0
        x_high = 1.0
        
        if self.spec.N is None:
            # Solve flux equality
            residual_func = self._residual
        else:
            # Solve for given flux
            residual_func = self._residual_with_flux
        
        try:
            x_i = bisection(
                residual_func, x_low, x_high,
                tol=self.spec.tol, maxiter=self.spec.maxiter
            )
        except:
            x_i = bisection(
                residual_func, x_low, x_high,
                expand_bracket=True, tol=self.spec.tol
            )
        
        y_i = self.spec.eq.y_of_x(x_i)
        
        # Calculate flux
        if self.spec.N is None:
            N = self.spec.kG * (self.spec.y_b - y_i)
        else:
            N = self.spec.N
        
        # Calculate driving forces
        return {
            "interface": {
                "x_interface": x_i,
                "y_interface": y_i,
            },
            "flux": N,
            "driving_forces": {
                "gas_phase": self.spec.y_b - y_i,
                "liquid_phase": x_i - self.spec.x_b,
            },
            "resistances": {
                "gas_resistance": 1.0 / self.spec.kG,
                "liquid_resistance": 1.0 / self.spec.kL,
                "total_resistance": (1.0/self.spec.kG + 1.0/self.spec.kL),
            }
        }
    
    def calculate_enhancement_factor(self, reaction_case: str = "none") -> float:
        """
        Calculate enhancement factor for chemical absorption.
        
        Args:
            reaction_case: "none", "instantaneous", "slow", "fast"
        
        Returns:
            Enhancement factor E (multiplies kL)
        """
        if reaction_case == "none":
            return 1.0
        elif reaction_case == "instantaneous":
            # Simplified - would need diffusivities
            return 10.0  # Placeholder
        else:
            raise NotImplementedError(f"Enhancement for {reaction_case} not implemented")