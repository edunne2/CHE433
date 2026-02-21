"""Raoult's Law for vapor-liquid equilibrium - Eqs. 26.1-1 to 26.1-4"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math

from bank.core.validation import check_positive, check_in_closed_01
from bank.separations.equilibrium import BinaryEquilibrium


class RaoultLaw(BinaryEquilibrium):
    """
    Raoult's Law for ideal binary mixtures.
    
    Eqs. 26.1-1 to 26.1-4:
    pA = PA xA                      (26.1-1)
    pA + pB = P                     (26.1-2)
    PA xA + PB (1 - xA) = P         (26.1-3)
    yA = pA / P = (PA xA) / P       (26.1-4)
    """
    
    def __init__(self, PA_func, PB_func, P_total: float):
        """
        Args:
            PA_func: Function PA(T) returning vapor pressure of component A
            PB_func: Function PB(T) returning vapor pressure of component B
            P_total: Total pressure (same units as vapor pressures)
        """
        self.PA_func = PA_func
        self.PB_func = PB_func
        self.P_total = check_positive("P_total", P_total)
        self.T = None  # Temperature to be set later
    
    def set_temperature(self, T: float):
        """Set temperature and update vapor pressures"""
        self.T = T
        self.PA = self.PA_func(T)
        self.PB = self.PB_func(T)
    
    def y_of_x(self, x: float) -> float:
        """Calculate y from x at current temperature - Eq. 26.1-4"""
        check_in_closed_01("x", x)
        if self.T is None:
            raise ValueError("Temperature must be set first")
        return (self.PA * x) / self.P_total
    
    def x_of_y(self, y: float) -> float:
        """Calculate x from y at current temperature"""
        check_in_closed_01("y", y)
        if self.T is None:
            raise ValueError("Temperature must be set first")
        return (y * self.P_total) / self.PA
    
    def bubble_point(self, x: float) -> Tuple[float, float]:
        """
        Calculate bubble point temperature and vapor composition.
        Solves Eq. 26.1-3 for T: PA(T) xA + PB(T) (1-xA) = P
        """
        check_in_closed_01("x", x)
        
        from bank.core.numerical import bisection
        
        def f(T):
            PA = self.PA_func(T)
            PB = self.PB_func(T)
            return PA * x + PB * (1 - x) - self.P_total
        
        T_bubble = bisection(f, 0, 200, expand_bracket=True)
        self.set_temperature(T_bubble)
        y = self.y_of_x(x)
        
        return T_bubble, y
    
    def dew_point(self, y: float) -> Tuple[float, float]:
        """
        Calculate dew point temperature and liquid composition.
        Solves 1 = Σ (yi / Ki) where Ki = PAi/P
        """
        check_in_closed_01("y", y)
        
        from bank.core.numerical import bisection
        
        def f(T):
            PA = self.PA_func(T)
            PB = self.PB_func(T)
            return y * self.P_total / PA + (1 - y) * self.P_total / PB - 1.0
        
        T_dew = bisection(f, 0, 200, expand_bracket=True)
        self.set_temperature(T_dew)
        x = self.x_of_y(y)
        
        return T_dew, x


class RaoultAntoine(RaoultLaw):
    """
    Raoult's Law with Antoine equation for vapor pressures.
    """
    
    def __init__(
        self,
        A_light: float, B_light: float, C_light: float,
        A_heavy: float, B_heavy: float, C_heavy: float,
        P_total: float
    ):
        """
        Antoine constants for log10(P) = A - B/(T + C)
        """
        self.A_light = A_light
        self.B_light = B_light
        self.C_light = C_light
        self.A_heavy = A_heavy
        self.B_heavy = B_heavy
        self.C_heavy = C_heavy
        
        def PA_func(T):
            return 10.0 ** (A_light - B_light / (T + C_light))
        
        def PB_func(T):
            return 10.0 ** (A_heavy - B_heavy / (T + C_heavy))
        
        super().__init__(PA_func, PB_func, P_total)