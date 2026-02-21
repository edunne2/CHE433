"""Raoult's law equilibrium models"""

from typing import Optional
import math

from bank.core.validation import check_positive, check_in_closed_01
from bank.core.numerical import bisection
from .base import BinaryEquilibrium


class RaoultLaw(BinaryEquilibrium):
    """
    Raoult's law for binary systems at fixed temperature.
    yA = (Psat_A/P) xA
    """
    
    def __init__(self, Psat_light: float, Psat_heavy: float, P_total: float):
        self.Psat_light = check_positive("Psat_light", Psat_light)
        self.Psat_heavy = check_positive("Psat_heavy", Psat_heavy)
        self.P_total = check_positive("P_total", P_total)
        
        self.K_light = self.Psat_light / self.P_total
        self.K_heavy = self.Psat_heavy / self.P_total
        self.alpha = self.K_light / self.K_heavy
    
    def y_of_x(self, x: float) -> float:
        check_in_closed_01("x", x)
        return self.K_light * x
    
    def x_of_y(self, y: float) -> float:
        check_in_closed_01("y", y)
        return y / self.K_light


class RaoultAntoine(BinaryEquilibrium):
    """
    Raoult's law with Antoine equation for vapor pressures.
    Can solve for bubble/dew points.
    """
    
    def __init__(
        self,
        A_light: float, B_light: float, C_light: float,
        A_heavy: float, B_heavy: float, C_heavy: float,
        P_total: Optional[float] = None,
        T: Optional[float] = None,
        tol: float = 1e-10,
        maxiter: int = 200
    ):
        self.A_light = A_light
        self.B_light = B_light
        self.C_light = C_light
        self.A_heavy = A_heavy
        self.B_heavy = B_heavy
        self.C_heavy = C_heavy
        self.tol = tol
        self.maxiter = maxiter
        
        if P_total is not None and T is not None:
            self.P_total = check_positive("P_total", P_total)
            self.T = T
            self._update_vapor_pressures()
        elif P_total is not None:
            self.P_total = check_positive("P_total", P_total)
            self.T = None
        elif T is not None:
            self.T = T
            self.P_total = None
            self._update_vapor_pressures()
        else:
            raise ValueError("Must provide either P_total or T")
    
    def _Psat(self, T: float, A: float, B: float, C: float) -> float:
        """Antoine equation: log10(P) = A - B/(T + C)"""
        return 10.0 ** (A - B / (T + C))
    
    def _update_vapor_pressures(self):
        if self.T is None:
            return
        self.Psat_light = self._Psat(self.T, self.A_light, self.B_light, self.C_light)
        self.Psat_heavy = self._Psat(self.T, self.A_heavy, self.B_heavy, self.C_heavy)
        if self.P_total is not None:
            self.K_light = self.Psat_light / self.P_total
            self.K_heavy = self.Psat_heavy / self.P_total
            self.alpha = self.K_light / self.K_heavy
    
    def set_temperature(self, T: float):
        self.T = T
        self._update_vapor_pressures()
    
    def set_pressure(self, P_total: float):
        self.P_total = check_positive("P_total", P_total)
        if self.T is not None:
            self._update_vapor_pressures()
    
    def y_of_x(self, x: float) -> float:
        check_in_closed_01("x", x)
        if self.T is None or self.P_total is None:
            raise ValueError("Both T and P_total must be set")
        return self.K_light * x
    
    def x_of_y(self, y: float) -> float:
        check_in_closed_01("y", y)
        if self.T is None or self.P_total is None:
            raise ValueError("Both T and P_total must be set")
        return y / self.K_light
    
    def bubble_point_T(self, x: float, P_total: Optional[float] = None) -> float:
        """Solve for bubble point T using Eq. 22.1-4/5 principles"""
        if P_total is not None:
            self.set_pressure(P_total)
        if self.P_total is None:
            raise ValueError("Pressure must be set")
        
        check_in_closed_01("x", x)
        
        def f(T: float) -> float:
            Psat_l = self._Psat(T, self.A_light, self.B_light, self.C_light)
            Psat_h = self._Psat(T, self.A_heavy, self.B_heavy, self.C_heavy)
            return (x * Psat_l + (1 - x) * Psat_h) - self.P_total
        
        return bisection(f, 0.0, 200.0, tol=self.tol, maxiter=self.maxiter)
    
    def dew_point_T(self, y: float, P_total: Optional[float] = None) -> float:
        """Solve for dew point T"""
        if P_total is not None:
            self.set_pressure(P_total)
        if self.P_total is None:
            raise ValueError("Pressure must be set")
        
        check_in_closed_01("y", y)
        
        def f(T: float) -> float:
            Psat_l = self._Psat(T, self.A_light, self.B_light, self.C_light)
            Psat_h = self._Psat(T, self.A_heavy, self.B_heavy, self.C_heavy)
            return 1.0 / (y / Psat_l + (1 - y) / Psat_h) - self.P_total
        
        return bisection(f, 0.0, 200.0, tol=self.tol, maxiter=self.maxiter)