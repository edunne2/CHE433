# bank/separations/equilibrium/linear_slope.py
"""Linear equilibrium model: y = m*x + b"""

from typing import Optional
from bank.core.validation import check_in_closed_01, check_positive
from .base import EquilibriumModel


class LinearEquilibrium(EquilibriumModel):
    """
    Linear equilibrium relationship: y = m*x + b
    Useful for dilute systems or simplified analysis.
    """
    
    def __init__(self, m: float, b: float = 0.0):
        """
        Args:
            m: Slope of equilibrium line
            b: Intercept (usually 0 for dilute systems)
        """
        self.m = check_positive("m", m)
        self.b = b
        
        # Check physical constraints
        if b < 0:
            raise ValueError(f"Intercept b must be >= 0, got {b}")
        
        # At x=1, y should be <= 1
        if m + b > 1.0:
            raise ValueError(f"At x=1, y={m+b} > 1 - not physical")
    
    def y_of_x(self, x: float) -> float:
        check_in_closed_01("x", x)
        y = self.m * x + self.b
        return max(0.0, min(1.0, y))
    
    def x_of_y(self, y: float) -> float:
        check_in_closed_01("y", y)
        if abs(self.m) < 1e-12:
            return 0.0
        x = (y - self.b) / self.m
        return max(0.0, min(1.0, x))
    
    def K_value(self, x: float) -> float:
        """K = y/x = m + b/x"""
        if x < 1e-12:
            return float('inf')
        return self.m + self.b / x


__all__ = ['LinearEquilibrium']   