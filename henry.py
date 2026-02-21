"""Henry's law equilibrium models - Eqs. 22.1-2 and 22.1-3"""

from typing import Optional
import math

from bank.core.validation import check_positive, check_in_closed_01
from .base import EquilibriumModel


class HenryLaw(EquilibriumModel):
    """
    Henry's law: pA = H xA (Eq. 22.1-2)
    or yA = H' xA where H' = H/P (Eq. 22.1-3)
    """
    
    def __init__(self, H: float, P_total: Optional[float] = None):
        """
        Args:
            H: Henry's constant (pressure units)
            P_total: Total system pressure (required for y = H' x form)
        """
        self.H = check_positive("H", H)
        self.P_total = P_total
        if P_total is not None:
            self.H_prime = self.H / check_positive("P_total", P_total)
    
    def y_of_x(self, x: float) -> float:
        """Calculate y from x using Eq. 22.1-3: y = (H/P) x"""
        check_in_closed_01("x", x)
        if self.P_total is None:
            raise ValueError("P_total must be set to use y = H' x form")
        return self.H_prime * x
    
    def x_of_y(self, y: float) -> float:
        """Calculate x from y using Eq. 22.1-3 inverted"""
        check_in_closed_01("y", y)
        if self.P_total is None:
            raise ValueError("P_total must be set to use y = H' x form")
        return y / self.H_prime
    
    def pA_of_x(self, x: float) -> float:
        """Calculate partial pressure from x using Eq. 22.1-2: pA = H x"""
        check_in_closed_01("x", x)
        return self.H * x
    
    def x_of_pA(self, pA: float) -> float:
        """Calculate x from partial pressure using Eq. 22.1-2 inverted"""
        check_positive("pA", pA)
        return pA / self.H
    
    def set_pressure(self, P_total: float):
        """Update total pressure"""
        self.P_total = check_positive("P_total", P_total)
        self.H_prime = self.H / self.P_total 