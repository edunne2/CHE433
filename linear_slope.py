"""Linear equilibrium model: y = m*x + b"""

from bank.core.validation import check_in_closed_01, check_positive
from .base import EquilibriumModel


class LinearEquilibrium(EquilibriumModel):
    """Linear equilibrium relationship: y = m*x + b"""
    
    def __init__(self, m: float, b: float = 0.0):
        self.m = check_positive("m", m)
        self.b = b
        
        if b < 0:
            raise ValueError(f"Intercept b must be >= 0, got {b}")
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