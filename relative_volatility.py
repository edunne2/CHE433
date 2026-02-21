"""Relative volatility calculations - Eqs. 26.3-1 to 26.3-4"""

import math
from typing import Union

from bank.core.validation import check_positive, check_in_closed_01


class RelativeVolatility:
    """
    Relative volatility calculations for binary mixtures.
    
    Eqs. 26.3-1 to 26.3-4:
    αAB = (yA/xA) / (yB/xB) = (yA/xA) / ((1-yA)/(1-xA))    (26.3-1)
    yA = (PA xA)/P, yB = (PB xB)/P                          (26.3-2)
    αAB = PA / PB                                            (26.3-3)
    yA = (α xA) / [1 + (α - 1) xA]                           (26.3-4)
    """
    
    def __init__(self, alpha: float = None):
        """
        Args:
            alpha: Relative volatility (if constant)
        """
        self.alpha = check_positive("alpha", alpha) if alpha else None
    
    @staticmethod
    def from_compositions(yA: float, xA: float) -> float:
        """Calculate α from equilibrium compositions - Eq. 26.3-1"""
        check_in_closed_01("yA", yA)
        check_in_closed_01("xA", xA)
        
        if xA <= 0 or xA >= 1:
            return float('inf')
        
        return (yA / xA) / ((1 - yA) / (1 - xA))
    
    @staticmethod
    def from_vapor_pressures(PA: float, PB: float) -> float:
        """Calculate α from pure component vapor pressures - Eq. 26.3-3"""
        check_positive("PA", PA)
        check_positive("PB", PB)
        return PA / PB
    
    def y_from_x(self, x: float, alpha: float = None) -> float:
        """Calculate equilibrium y from x - Eq. 26.3-4"""
        check_in_closed_01("x", x)
        a = alpha if alpha is not None else self.alpha
        if a is None:
            raise ValueError("Relative volatility not provided")
        
        if abs(a - 1.0) < 1e-12:
            return x
        
        return (a * x) / (1 + (a - 1) * x)
    
    def x_from_y(self, y: float, alpha: float = None) -> float:
        """Calculate equilibrium x from y - Eq. 26.3-4 rearranged"""
        check_in_closed_01("y", y)
        a = alpha if alpha is not None else self.alpha
        if a is None:
            raise ValueError("Relative volatility not provided")
        
        if abs(a - 1.0) < 1e-12:
            return y
        
        return y / (a - (a - 1) * y)
    
    def geometric_mean(self, alpha1: float, alpha2: float) -> float:
        """Calculate geometric mean relative volatility - used in Fenske eq."""
        check_positive("alpha1", alpha1)
        check_positive("alpha2", alpha2)
        return math.sqrt(alpha1 * alpha2)