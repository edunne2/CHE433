# bank/separations/equilibrium/raoult.py
"""Raoult's law equilibrium models"""
from typing import Optional, Callable, List
import math

from bank.core.validation import check_positive, check_in_closed_01
from .base import EquilibriumModel, BinaryEquilibrium


class RaoultLaw(BinaryEquilibrium):
    """
    Raoult's law: y_i * P = x_i * P_sat_i(T)
    For binary systems at fixed temperature.
    """
    
    def __init__(self, Psat_light: float, Psat_heavy: float, P_total: float):
        """
        Args:
            Psat_light: Vapor pressure of light component
            Psat_heavy: Vapor pressure of heavy component
            P_total: Total system pressure
        """
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
    
    def K_value(self, x: float) -> float:
        return self.K_light  # Constant for light component
    
    def relative_volatility(self, x: float) -> float:
        return self.alpha


class RaoultAntoine(BinaryEquilibrium):
    """
    Raoult's law with Antoine equation for vapor pressures.
    Can solve for bubble/dew points at given T or P.
    """
    
    def __init__(
        self,
        A_light: float, B_light: float, C_light: float,
        A_heavy: float, B_heavy: float, C_heavy: float,
        P_total: Optional[float] = None,
        T: Optional[float] = None,
        units: str = "mmHg_C",  # or "kPa_C", "bar_K", etc.
        tol: float = 1e-10,
        maxiter: int = 200
    ):
        """
        Antoine constants for log10(P) = A - B/(T + C)
        
        Args:
            A_light, B_light, C_light: Antoine constants for light component
            A_heavy, B_heavy, C_heavy: Antoine constants for heavy component
            P_total: Total pressure (if T is to be found)
            T: Temperature (if P_total is to be found)
            units: Units system for constants
            tol: Tolerance for numerical methods
            maxiter: Maximum iterations for numerical methods
        """
        from bank.core.validation import check_positive
        
        self.A_light = A_light
        self.B_light = B_light
        self.C_light = C_light
        self.A_heavy = A_heavy
        self.B_heavy = B_heavy
        self.C_heavy = C_heavy
        self.units = units
        self.tol = tol
        self.maxiter = maxiter
        
        # Validate and set initial conditions
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
        """Calculate vapor pressure using Antoine equation"""
        return 10.0 ** (A - B / (T + C))
    
    def _update_vapor_pressures(self):
        """Update vapor pressures at current temperature"""
        if self.T is None:
            return  # Can't update without temperature
        
        self.Psat_light = self._Psat(self.T, self.A_light, self.B_light, self.C_light)
        self.Psat_heavy = self._Psat(self.T, self.A_heavy, self.B_heavy, self.C_heavy)
        
        if self.P_total is not None:
            self.K_light = self.Psat_light / self.P_total
            self.K_heavy = self.Psat_heavy / self.P_total
            self.alpha = self.K_light / self.K_heavy
    
    def set_temperature(self, T: float):
        """Set temperature and update vapor pressures"""
        self.T = T
        self._update_vapor_pressures()
    
    def set_pressure(self, P_total: float):
        """Set total pressure"""
        from bank.core.validation import check_positive
        self.P_total = check_positive("P_total", P_total)
        if self.T is not None:
            self._update_vapor_pressures()
    
    def y_of_x(self, x: float) -> float:
        """Calculate vapor composition at given liquid composition"""
        from bank.core.validation import check_in_closed_01
        check_in_closed_01("x", x)
        
        if self.T is None or self.P_total is None:
            raise ValueError("Both T and P_total must be set to calculate y(x)")
        
        return self.K_light * x
    
    def x_of_y(self, y: float) -> float:
        """Calculate liquid composition at given vapor composition"""
        from bank.core.validation import check_in_closed_01
        check_in_closed_01("y", y)
        
        if self.T is None or self.P_total is None:
            raise ValueError("Both T and P_total must be set to calculate x(y)")
        
        return y / self.K_light
    
    def bubble_point_T(self, x: float, P_total: Optional[float] = None, 
                       T_guess_low: float = 0.0, T_guess_high: float = 200.0) -> float:
        """
        Calculate bubble point temperature at given composition and pressure.
        
        Args:
            x: Liquid composition
            P_total: Total pressure (if not already set)
            T_guess_low: Lower bound for temperature search (°C)
            T_guess_high: Upper bound for temperature search (°C)
        
        Returns:
            Bubble point temperature (°C)
        """
        from bank.core.validation import check_in_closed_01
        from bank.core.numerical import bisection
        
        # Handle pressure
        if P_total is not None:
            self.set_pressure(P_total)
        if self.P_total is None:
            raise ValueError("Pressure must be set either in constructor or method call")
        
        check_in_closed_01("x", x)
        
        def f(T: float) -> float:
            Psat_l = self._Psat(T, self.A_light, self.B_light, self.C_light)
            Psat_h = self._Psat(T, self.A_heavy, self.B_heavy, self.C_heavy)
            return (x * Psat_l + (1 - x) * Psat_h) - self.P_total
        
        # Validate temperature bounds
        f_low = f(T_guess_low)
        f_high = f(T_guess_high)
        
        # If no sign change, try to find reasonable bounds
        if f_low * f_high > 0:
            # Try expanding the range
            if f_low < 0:  # Need higher temperature
                T_guess_high *= 2
            else:  # Need lower temperature
                T_guess_low = -50.0  # Allow below freezing
        
        return bisection(f, T_guess_low, T_guess_high, 
                        tol=self.tol, maxiter=self.maxiter, expand_bracket=True)
    
    def dew_point_T(self, y: float, P_total: Optional[float] = None,
                    T_guess_low: float = 0.0, T_guess_high: float = 200.0) -> float:
        """
        Calculate dew point temperature at given composition and pressure.
        
        Args:
            y: Vapor composition
            P_total: Total pressure (if not already set)
            T_guess_low: Lower bound for temperature search (°C)
            T_guess_high: Upper bound for temperature search (°C)
        
        Returns:
            Dew point temperature (°C)
        """
        from bank.core.validation import check_in_closed_01
        from bank.core.numerical import bisection
        
        # Handle pressure
        if P_total is not None:
            self.set_pressure(P_total)
        if self.P_total is None:
            raise ValueError("Pressure must be set either in constructor or method call")
        
        check_in_closed_01("y", y)
        
        def f(T: float) -> float:
            Psat_l = self._Psat(T, self.A_light, self.B_light, self.C_light)
            Psat_h = self._Psat(T, self.A_heavy, self.B_heavy, self.C_heavy)
            return 1.0 / (y / Psat_l + (1 - y) / Psat_h) - self.P_total
        
        return bisection(f, T_guess_low, T_guess_high,
                        tol=self.tol, maxiter=self.maxiter, expand_bracket=True)
    
    def bubble_point_P(self, x: float, T: float,
                       P_guess_low: float = 0.1, P_guess_high: float = 1000.0) -> float:
        """
        Calculate bubble point pressure at given composition and temperature.
        
        Args:
            x: Liquid composition
            T: Temperature (°C)
            P_guess_low: Lower bound for pressure search (kPa)
            P_guess_high: Upper bound for pressure search (kPa)
        
        Returns:
            Bubble point pressure (kPa)
        """
        from bank.core.validation import check_in_closed_01
        from bank.core.numerical import bisection
        
        check_in_closed_01("x", x)
        self.set_temperature(T)
        
        def f(P: float) -> float:
            return (x * self.Psat_light + (1 - x) * self.Psat_heavy) - P
        
        return bisection(f, P_guess_low, P_guess_high,
                        tol=self.tol, maxiter=self.maxiter)
    
    def dew_point_P(self, y: float, T: float,
                    P_guess_low: float = 0.1, P_guess_high: float = 1000.0) -> float:
        """
        Calculate dew point pressure at given composition and temperature.
        
        Args:
            y: Vapor composition
            T: Temperature (°C)
            P_guess_low: Lower bound for pressure search (kPa)
            P_guess_high: Upper bound for pressure search (kPa)
        
        Returns:
            Dew point pressure (kPa)
        """
        from bank.core.validation import check_in_closed_01
        from bank.core.numerical import bisection
        
        check_in_closed_01("y", y)
        self.set_temperature(T)
        
        def f(P: float) -> float:
            return 1.0 / (y / self.Psat_light + (1 - y) / self.Psat_heavy) - P
        
        return bisection(f, P_guess_low, P_guess_high,
                        tol=self.tol, maxiter=self.maxiter)


__all__ = ['RaoultLaw', 'RaoultAntoine']  