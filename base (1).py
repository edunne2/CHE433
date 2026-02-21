"""Base classes for equilibrium models"""

from abc import ABC, abstractmethod
from typing import Optional

from bank.core.validation import check_in_closed_01


class EquilibriumModel(ABC):
    """Base class for all equilibrium models - implements Eq. 22.1-30"""
    
    @abstractmethod
    def y_of_x(self, x: float) -> float:
        """Vapor composition at given liquid composition"""
        pass
    
    @abstractmethod
    def x_of_y(self, y: float) -> float:
        """Liquid composition at given vapor composition"""
        pass
    
    def K_value(self, x: float) -> float:
        """K = y/x at given composition"""
        check_in_closed_01("x", x)
        y = self.y_of_x(x)
        if abs(x) < 1e-15:
            return float('inf')
        return y / x
    
    def relative_volatility(self, x: float) -> float:
        """Relative volatility α = (y/x)/((1-y)/(1-x))"""
        check_in_closed_01("x", x)
        y = self.y_of_x(x)
        
        if abs(x) < 1e-15 or abs(1 - x) < 1e-15:
            return float('inf')
        
        return (y / x) / ((1 - y) / (1 - x))


class BinaryEquilibrium(EquilibriumModel):
    """Base class for binary systems"""
    pass