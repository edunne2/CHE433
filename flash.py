"""Flash distillation calculations - Eqs. 26.3-5 to 26.3-6"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from bank.core.validation import check_positive, check_in_closed_01
from bank.core.numerical import bisection
from bank.separations.distillation.relative_volatility import RelativeVolatility


@dataclass
class FlashSpec:
    """Specification for flash distillation"""
    F: float           # Feed flow rate
    xF: float          # Feed composition
    V: float           # Vapor flow rate
    alpha: float       # Relative volatility
    P: float = 101.325 # Pressure (kPa)
    tol: float = 1e-12
    maxiter: int = 400


class FlashDistillation:
    """
    Flash distillation calculations.
    
    Eqs. 26.3-5 and 26.3-6:
    F xF = V y + L x                     (26.3-5)
    F xF = V y + (F - V) x               (26.3-6)
    """
    
    def __init__(self, spec: FlashSpec):
        self.spec = spec
        self.rv = RelativeVolatility(spec.alpha)
        self._validate()
    
    def _validate(self):
        check_positive("F", self.spec.F)
        check_in_closed_01("xF", self.spec.xF)
        check_positive("V", self.spec.V)
        if self.spec.V >= self.spec.F:
            raise ValueError("V must be less than F")
    
    def _flash_equation(self, x: float) -> float:
        """Flash equation - derived from Eq. 26.3-6"""
        y = self.rv.y_from_x(x)
        L = self.spec.F - self.spec.V
        return self.spec.F * self.spec.xF - self.spec.V * y - L * x
    
    def solve(self) -> Dict[str, Any]:
        """Solve flash distillation"""
        
        # Check endpoints
        f0 = self._flash_equation(0.0)
        f1 = self._flash_equation(1.0)
        
        if abs(f0) <= self.spec.tol:
            x = 0.0
        elif abs(f1) <= self.spec.tol:
            x = 1.0
        else:
            if f0 * f1 > 0:
                raise ValueError("No solution in [0,1] - check feed conditions")
            x = bisection(self._flash_equation, 0.0, 1.0, 
                         tol=self.spec.tol, maxiter=self.spec.maxiter)
        
        y = self.rv.y_from_x(x)
        L = self.spec.F - self.spec.V
        
        return {
            "inputs": {"F": self.spec.F, "xF": self.spec.xF, "V": self.spec.V},
            "outputs": {"x": x, "y": y, "L": L},
            "verification": {"balance_error": self._flash_equation(x)},
        }       