# bank/separations/equilibrium/henry.py
"""Henry's law equilibrium models"""
from typing import Optional
import math

from bank.core.validation import check_positive, check_in_closed_01
from .base import EquilibriumModel


class HenryLaw(EquilibriumModel):
    """
    Henry's law for dilute systems: y_i * P = H_i * x_i
    Assumes component follows Henry's law and carrier gas is insoluble.
    """
    
    def __init__(self, H: float, P_total: float):
        """
        Args:
            H: Henry's constant (pressure units)
            P_total: Total system pressure
        """
        self.H = check_positive("H", H)
        self.P_total = check_positive("P_total", P_total)
        self.K = self.H / self.P_total
    
    def y_of_x(self, x: float) -> float:
        check_in_closed_01("x", x)
        return self.K * x
    
    def x_of_y(self, y: float) -> float:
        check_in_closed_01("y", y)
        return y / self.K
    
    def K_value(self, x: float) -> float:
        return self.K
    
    def set_pressure(self, P_total: float):
        """Update total pressure"""
        self.P_total = check_positive("P_total", P_total)
        self.K = self.H / self.P_total
    
    def set_temperature(self, T: float, H_coeffs: Optional[tuple] = None):
        """
        Update Henry's constant for temperature (if coefficients provided).
        
        Args:
            T: Temperature
            H_coeffs: Optional (A, B) for ln(H) = A - B/T
        """
        if H_coeffs is not None:
            A, B = H_coeffs
            self.H = math.exp(A - B / T)
            self.K = self.H / self.P_total


class HenryBinary(EquilibriumModel):
    """
    Binary system where one component follows Henry's law,
    the other follows Raoult's law (common in absorption).
    """
    
    def __init__(self, H_solute: float, Psat_solvent: float, P_total: float):
        """
        Args:
            H_solute: Henry's constant for solute
            Psat_solvent: Vapor pressure of solvent
            P_total: Total pressure
        """
        self.H_solute = check_positive("H_solute", H_solute)
        self.Psat_solvent = check_positive("Psat_solvent", Psat_solvent)
        self.P_total = check_positive("P_total", P_total)
        
        self.K_solute = self.H_solute / self.P_total
        self.K_solvent = self.Psat_solvent / self.P_total
    
    def y_of_x(self, x: float) -> float:
        """
        For binary, x is solute mole fraction.
        y_solute = K_solute * x
        y_solvent = K_solvent * (1-x)
        """
        check_in_closed_01("x", x)
        y_solute = self.K_solute * x
        y_solvent = self.K_solvent * (1 - x)
        return y_solute / (y_solute + y_solvent)  # Normalize
    
    def x_of_y(self, y: float) -> float:
        check_in_closed_01("y", y)
        # From material balance and equilibrium
        x = y / (self.K_solute + (self.K_solvent - self.K_solute) * y)
        return max(0.0, min(1.0, x))
    
# bank/separations/equilibrium/henry.py
__all__ = ['HenryLaw', 'HenryBinary']    