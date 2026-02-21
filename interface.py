"""Interface concentration calculations - Eqs. 22.1-31 to 22.1-38"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from bank.core.validation import check_positive, check_in_closed_01
from bank.core.numerical import bisection
from bank.separations.equilibrium import EquilibriumModel


@dataclass
class InterfaceSpec:
    """Specification for interface concentration calculations"""
    x_b: float
    y_b: float
    kG: float
    kL: float
    eq: EquilibriumModel
    N: Optional[float] = None
    tol: float = 1e-12
    maxiter: int = 400


class InterfaceConcentration:
    """
    Solver for interface concentrations - Eqs. 22.1-31 to 22.1-38
    
    Key equation: kG(y_b - y_i) = kL(x_i - x_b) with y_i = f(x_i)
    """
    
    def __init__(self, spec: InterfaceSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        check_in_closed_01("x_b", self.spec.x_b)
        check_in_closed_01("y_b", self.spec.y_b)
        check_positive("kG", self.spec.kG)
        check_positive("kL", self.spec.kL)
    
    def _residual(self, x_i: float) -> float:
        """Residual for interface calculation - Eq. 22.1-31 equality"""
        y_i = self.spec.eq.y_of_x(x_i)
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
        
        if self.spec.N is None:
            residual_func = self._residual
        else:
            residual_func = self._residual_with_flux
        
        x_i = bisection(residual_func, 0.0, 1.0, tol=self.spec.tol, expand_bracket=True)
        y_i = self.spec.eq.y_of_x(x_i)
        
        if self.spec.N is None:
            N = self.spec.kG * (self.spec.y_b - y_i)
        else:
            N = self.spec.N
        
        return {
            "interface": {"x_i": x_i, "y_i": y_i},
            "flux": N,
            "driving_forces": {"gas": self.spec.y_b - y_i, "liquid": x_i - self.spec.x_b},
        }