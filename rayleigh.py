"""Rayleigh (differential) distillation - Eqs. 26.3-7 to 26.3-11"""

from dataclasses import dataclass
from typing import Dict, Any, Callable, List
import math
import numpy as np

from bank.core.validation import check_positive, check_in_closed_01
from bank.core.numerical import integrate_trapezoid


@dataclass
class RayleighSpec:
    """Specification for Rayleigh distillation"""
    L1: float           # Initial liquid moles
    x1: float           # Initial composition
    L2: float = None    # Final liquid moles (or provide x2)
    x2: float = None    # Final composition (or provide L2)
    eq_func: Callable = None  # Equilibrium function y(x)
    alpha: float = None  # Relative volatility (if constant)
    tol: float = 1e-10
    n_int: int = 1000


class RayleighDistillation:
    """
    Rayleigh (differential) distillation.
    
    Eqs. 26.3-7 to 26.3-11:
    dL/L = dx/(y-x)                                      (26.3-9)
    ln(L1/L2) = ∫(x1 to x2) dx/(y-x)                     (26.3-10)
    L1 x1 = L2 x2 + (L1 - L2) yav                         (26.3-11)
    """
    
    def __init__(self, spec: RayleighSpec):
        self.spec = spec
        self._validate()
        
        if spec.alpha is not None:
            from bank.separations.distillation import RelativeVolatility
            self.rv = RelativeVolatility(spec.alpha)
            self.eq_func = self.rv.y_from_x
        elif spec.eq_func is not None:
            self.eq_func = spec.eq_func
        else:
            raise ValueError("Must provide either alpha or eq_func")
    
    def _validate(self):
        check_positive("L1", self.spec.L1)
        check_in_closed_01("x1", self.spec.x1)
        
        if self.spec.L2 is not None and self.spec.x2 is not None:
            raise ValueError("Provide either L2 or x2, not both")
        
        if self.spec.L2 is not None:
            check_positive("L2", self.spec.L2)
            if self.spec.L2 >= self.spec.L1:
                raise ValueError("L2 must be less than L1")
            self.solve_for = "x2"
            self.target_ln = math.log(self.spec.L1 / self.spec.L2)
        elif self.spec.x2 is not None:
            check_in_closed_01("x2", self.spec.x2)
            if self.spec.x2 >= self.spec.x1:
                raise ValueError("x2 must be less than x1")
            self.solve_for = "L2"
        else:
            raise ValueError("Must provide either L2 or x2")
    
    def _integrand(self, x: float) -> float:
        """1/(y - x) with protection against division by zero"""
        y = self.eq_func(x)
        denom = y - x
        if abs(denom) < 1e-12:
            # At the limit, return a large number
            # This handles the case where y ≈ x at the endpoints
            return 1e12
        return 1.0 / denom
    
    def _integral(self, x2: float) -> float:
        """∫ dx/(y-x) from x2 to x1"""
        if x2 >= self.spec.x1:
            return 0.0
        
        # Add small epsilon to avoid exact endpoint where y=x
        x2_adj = max(x2, 1e-10)
        x1_adj = self.spec.x1 - 1e-10
        
        if x2_adj >= x1_adj:
            return 0.0
        
        return integrate_trapezoid(
            self._integrand, x2_adj, x1_adj, n=self.spec.n_int
        )
    
    def _residual_x2(self, x2: float) -> float:
        """Residual for solving x2 given L2"""
        return self._integral(x2) - self.target_ln
    
    def solve(self) -> Dict[str, Any]:
        """Solve Rayleigh distillation"""
        
        if self.solve_for == "x2":
            from bank.core.numerical import bisection
            
            # Find bounds for x2
            x_low = 0.0
            x_high = self.spec.x1 - 1e-12
            
            # Check if there's a solution
            f_low = self._residual_x2(x_low)
            f_high = self._residual_x2(x_high)
            
            if f_low * f_high > 0:
                # Try expanding bounds
                if f_low < 0:
                    # Need lower x2
                    x_low = 0.0
                else:
                    # Need higher x2
                    x_low = self.spec.x1 * 0.5
            
            x2 = bisection(self._residual_x2, x_low, x_high, 
                          tol=self.spec.tol, expand_bracket=True)
            
            L2 = self.spec.L2
            distilled = self.spec.L1 - L2
            
        else:  # solve_for == "L2"
            integral_value = self._integral(self.spec.x2)
            L2 = self.spec.L1 * math.exp(-integral_value)
            distilled = self.spec.L1 - L2
            x2 = self.spec.x2
        
        # Calculate average distillate composition (Eq. 26.3-11)
        if distilled > 0:
            y_avg = (self.spec.L1 * self.spec.x1 - L2 * x2) / distilled
        else:
            y_avg = self.spec.x1
        
        return {
            "inputs": {"L1": self.spec.L1, "x1": self.spec.x1},
            "outputs": {"L2": L2, "x2": x2, "distilled": distilled, "y_avg": y_avg},
            "integral": integral_value if self.solve_for == "L2" else self._integral(x2),
        }