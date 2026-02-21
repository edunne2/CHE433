# bank/separations/equilibrium/constant_alpha.py
"""Constant relative volatility model"""
from typing import Optional
from bank.core.validation import check_in_closed_01, check_positive
from .base import BinaryEquilibrium

class BinaryConstantAlpha(BinaryEquilibrium):
    """Binary system with constant relative volatility"""
    
    def __init__(self, alpha: float):
        self.alpha = check_positive("alpha", alpha)
    
    def y_of_x(self, x: float) -> float:
        check_in_closed_01("x", x)
        return (self.alpha * x) / (1.0 + (self.alpha - 1.0) * x)
    
    def x_of_y(self, y: float) -> float:
        check_in_closed_01("y", y)
        denom = self.alpha - (self.alpha - 1.0) * y
        if abs(denom) < 1e-15:
            return 1.0
        x = y / denom
        return max(0.0, min(1.0, x))
    
    def relative_volatility(self, x: float) -> float:
        return self.alpha
    
# bank/separations/equilibrium/constant_alpha.py
__all__ = ['BinaryConstantAlpha']    