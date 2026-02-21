"""Enthalpy-concentration method for distillation - Eqs. 26.7-1 to 26.7-17"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Callable
import math
import numpy as np

from bank.core.validation import check_positive, check_in_closed_01, ChemEngError
from bank.core.numerical import bisection, linear_interpolate


class EnthalpyConcentrationDiagram:
    """
    Enthalpy-concentration diagram calculations.
    
    Eqs. 26.7-1 to 26.7-5:
    h = xA cpA (T - T0) + (1-xA) cpB (T - T0) + ΔHsoln          (26.7-1)
    H = yA [λA + cpvA (T - T0)] + (1-yA) [λB + cpvB (T - T0)]   (26.7-2)
    λA = cpA (TbA - T0) + λAb - cpvA (TbA - T0)                 (26.7-3)
    λB = cpB (TbB - T0) + λBb - cpvB (TbB - T0)                 (26.7-4)
    """
    
    def __init__(
        self,
        T0: float,
        cpA: float, cpB: float,        # Liquid heat capacities
        cpvA: float, cpvB: float,      # Vapor heat capacities
        lambda_Ab: float, lambda_Bb: float,  # Latent heats at boiling points
        TbA: float, TbB: float,         # Boiling points
        delta_H_soln: float = 0.0,
        # Optional temperature-composition functions
        T_x_func: Optional[Callable] = None,  # Function T(x) for bubble points
        T_y_func: Optional[Callable] = None   # Function T(y) for dew points
    ):
        self.T0 = T0
        self.cpA = cpA
        self.cpB = cpB
        self.cpvA = cpvA
        self.cpvB = cpvB
        self.lambda_Ab = lambda_Ab
        self.lambda_Bb = lambda_Bb
        self.TbA = TbA
        self.TbB = TbB
        self.delta_H_soln = delta_H_soln
        self.T_x_func = T_x_func
        self.T_y_func = T_y_func
        
        # Calculate latent heats at reference temperature - Eqs. 26.7-3, 26.7-4
        self.lambda_A = (
            cpA * (TbA - T0) + lambda_Ab - cpvA * (TbA - T0)
        )
        self.lambda_B = (
            cpB * (TbB - T0) + lambda_Bb - cpvB * (TbB - T0)
        )
    
    def liquid_enthalpy(self, x: float, T: float) -> float:
        """Saturated liquid enthalpy - Eq. 26.7-1"""
        check_in_closed_01("x", x)
        return (
            x * self.cpA * (T - self.T0) +
            (1 - x) * self.cpB * (T - self.T0) +
            self.delta_H_soln
        )
    
    def vapor_enthalpy(self, y: float, T: float) -> float:
        """Saturated vapor enthalpy - Eq. 26.7-2"""
        check_in_closed_01("y", y)
        term_A = y * (self.lambda_A + self.cpvA * (T - self.T0))
        term_B = (1 - y) * (self.lambda_B + self.cpvB * (T - self.T0))
        return term_A + term_B
    
    def temperature_from_x(self, x: float) -> float:
        """Get bubble point temperature from liquid composition"""
        if self.T_x_func is not None:
            return self.T_x_func(x)
        # Default linear interpolation between pure component boiling points
        return self.TbA * x + self.TbB * (1 - x)
    
    def temperature_from_y(self, y: float) -> float:
        """Get dew point temperature from vapor composition"""
        if self.T_y_func is not None:
            return self.T_y_func(y)
        # Default linear interpolation between pure component boiling points
        return self.TbA * y + self.TbB * (1 - y)


class EnrichingSection:
    """
    Enriching section calculations using enthalpy-concentration method.
    
    Eqs. 26.7-6 to 26.7-12:
    Vn+1 = Ln + D                                            (26.7-6)
    Vn+1 yn+1 = Ln xn + D xD                                  (26.7-7)
    yn+1 = (Ln / Vn+1) xn + (D xD / Vn+1)                     (26.7-8)
    Vn+1 Hn+1 = Ln hn + D hD + qc                             (26.7-9)
    qc = V1 H1 - Ln hD - D hD                                 (26.7-10)
    Vn+1 Hn+1 = Ln hn + V1 H1 - Ln hD                         (26.7-11)
    Vn+1 Hn+1 = (Vn+1 - D) hn + V1 H1 - Ln hD                 (26.7-12)
    
    Note: D in these equations is the distillate flow rate (mol/h)
    """
    
    def __init__(
        self,
        D: float,           # Distillate flow rate (mol/h)
        xD: float,          # Distillate composition
        R: float,           # Reflux ratio
        hD: float,          # Enthalpy of distillate
        H1: float,          # Enthalpy of vapor from top tray
        enthalpy_diagram: EnthalpyConcentrationDiagram
    ):
        """
        Initialize enriching section.
        
        Args:
            D: Distillate flow rate (mol/h)
            xD: Distillate composition (mole fraction)
            R: Reflux ratio
            hD: Enthalpy of distillate liquid (kJ/mol)
            H1: Enthalpy of vapor from top tray (kJ/mol)
            enthalpy_diagram: Enthalpy-concentration diagram object
        """
        self.D = check_positive("D", D)
        self.xD = check_in_closed_01("xD", xD)
        self.R = check_positive("R", R)
        self.hD = hD
        self.H1 = H1
        self.enthalpy = enthalpy_diagram
        
        # Top flows - from Eq. 26.7-6
        self.L0 = R * D
        self.V1 = self.L0 + D
    
    def operating_line_point(self, xn: float, tol: float = 1e-6, max_iter: int = 10) -> Dict[str, float]:
        """
        Calculate operating line point for given xn.
        
        This implements the iterative procedure described in the text:
        1. Assume Vn+1 = V1, Ln = L0
        2. Calculate approximate yn+1 from Eq. 26.7-8
        3. Get Hn+1 and hn from diagram
        4. Solve Eq. 26.7-12 for Vn+1
        5. Get Ln from Eq. 26.7-6
        6. Recalculate yn+1 from Eq. 26.7-8
        7. Iterate if needed
        
        Args:
            xn: Liquid composition on tray n
            tol: Convergence tolerance
            max_iter: Maximum iterations
            
        Returns:
            Dictionary with xn, yn+1, Vn+1, Ln, hn, Hn+1
        """
        check_in_closed_01("xn", xn)
        
        # Step 1: Initial assumption
        Vn1 = self.V1
        Ln = self.L0
        
        for iteration in range(max_iter):
            # Step 2: Approximate yn+1 from Eq. 26.7-8
            yn1_approx = (Ln / Vn1) * xn + (self.D * self.xD) / Vn1
            yn1_approx = max(0.0, min(1.0, yn1_approx))
            
            # Step 3: Get enthalpies using temperature functions
            T_liq = self.enthalpy.temperature_from_x(xn)
            T_vap = self.enthalpy.temperature_from_y(yn1_approx)
            
            hn = self.enthalpy.liquid_enthalpy(xn, T_liq)
            Hn1 = self.enthalpy.vapor_enthalpy(yn1_approx, T_vap)
            
            # Step 4: Solve Eq. 26.7-12 for Vn1
            # Vn1 Hn1 = (Vn1 - D) hn + V1 H1 - Ln hD
            numerator = self.V1 * self.H1 - Ln * self.hD - self.D * hn
            denominator = Hn1 - hn
            
            if abs(denominator) < 1e-12:
                Vn1_new = self.V1
            else:
                Vn1_new = numerator / denominator
            
            if Vn1_new <= self.D:
                Vn1_new = self.D + 1e-6
            
            # Step 5: Get Ln from Eq. 26.7-6
            Ln_new = Vn1_new - self.D
            if Ln_new < 0:
                Ln_new = 0
            
            # Step 6: Recalculate yn+1 from Eq. 26.7-8
            yn1_new = (Ln_new / Vn1_new) * xn + (self.D * self.xD) / Vn1_new
            yn1_new = max(0.0, min(1.0, yn1_new))
            
            # Check convergence
            if abs(yn1_new - yn1_approx) < tol:
                break
            
            Vn1 = Vn1_new
            Ln = Ln_new
        
        return {
            "xn": xn,
            "yn1": yn1_new,
            "Vn1": Vn1_new,
            "Ln": Ln_new,
            "hn": hn,
            "Hn1": Hn1,
        }
    
    def generate_operating_line(self, x_points: List[float]) -> List[Dict[str, float]]:
        """Generate multiple points on the enriching operating line"""
        results = []
        for x in sorted(x_points):
            results.append(self.operating_line_point(x))
        return results


class StrippingSection:
    """
    Stripping section calculations using enthalpy-concentration method.
    
    Eqs. 26.7-13 to 26.7-17:
    Lm = W + Vm+1                                            (26.7-13)
    Lm xm = W xW + Vm+1 ym+1                                  (26.7-14)
    ym+1 = (Lm / Vm+1) xm - (W xW / Vm+1)                     (26.7-15)
    Vm+1 Hm+1 = (Vm+1 + W) hm + qR - W hW                     (26.7-16)
    qR = D hD + W hW + qc - F hF                              (26.7-17)
    
    Note: W in these equations is the bottoms flow rate (mol/h)
    """
    
    def __init__(
        self,
        W: float,           # Bottoms flow rate (mol/h)
        xW: float,          # Bottoms composition
        hW: float,          # Enthalpy of bottoms
        qR: float,          # Reboiler duty
        enthalpy_diagram: EnthalpyConcentrationDiagram,
        # Optional parameters for energy balance verification
        F: Optional[float] = None,
        hF: Optional[float] = None,
        qc: Optional[float] = None,
        D: Optional[float] = None,
        hD: Optional[float] = None
    ):
        """
        Initialize stripping section.
        
        Args:
            W: Bottoms flow rate (mol/h)
            xW: Bottoms composition (mole fraction)
            hW: Enthalpy of bottoms liquid (kJ/mol)
            qR: Reboiler duty (kJ/h)
            enthalpy_diagram: Enthalpy-concentration diagram object
            F: Feed rate (optional, for energy balance)
            hF: Feed enthalpy (optional, for energy balance)
            qc: Condenser duty (optional, for energy balance)
            D: Distillate rate (optional, for energy balance)
            hD: Distillate enthalpy (optional, for energy balance)
        """
        self.W = check_positive("W", W)
        self.xW = check_in_closed_01("xW", xW)
        self.hW = hW
        self.qR = qR
        self.enthalpy = enthalpy_diagram
        self.F = F
        self.hF = hF
        self.qc = qc
        self.D = D
        self.hD = hD
    
    def operating_line_point(self, ym1: float, tol: float = 1e-6, max_iter: int = 10) -> Dict[str, float]:
        """
        Calculate operating line point for given ym+1.
        
        Iterative procedure:
        1. Assume Lm and Vm+1 from equimolar overflow
        2. Calculate approximate xm from Eq. 26.7-15
        3. Get hm and Hm+1 from diagram
        4. Solve Eq. 26.7-16 for Vm+1
        5. Get Lm from Eq. 26.7-13
        6. Recalculate xm from Eq. 26.7-15
        7. Iterate if needed
        
        Args:
            ym1: Vapor composition entering from below (ym+1)
            tol: Convergence tolerance
            max_iter: Maximum iterations
            
        Returns:
            Dictionary with ym1, xm, Vm+1, Lm, hm, Hm+1
        """
        check_in_closed_01("ym1", ym1)
        
        # Initial assumptions
        Vm1 = self.W * 1.5  # Rough estimate
        Lm = self.W + Vm1
        
        for iteration in range(max_iter):
            # Step 2: Approximate xm from Eq. 26.7-15
            xm_approx = ((ym1 * Vm1) + (self.W * self.xW)) / Lm
            xm_approx = max(0.0, min(1.0, xm_approx))
            
            # Step 3: Get enthalpies using temperature functions
            T_liq = self.enthalpy.temperature_from_x(xm_approx)
            T_vap = self.enthalpy.temperature_from_y(ym1)
            
            hm = self.enthalpy.liquid_enthalpy(xm_approx, T_liq)
            Hm1 = self.enthalpy.vapor_enthalpy(ym1, T_vap)
            
            # Step 4: Solve Eq. 26.7-16 for Vm1
            # Vm1 Hm1 = (Vm1 + W) hm + qR - W hW
            numerator = self.W * hm + self.qR - self.W * self.hW
            denominator = Hm1 - hm
            
            if abs(denominator) < 1e-12:
                Vm1_new = self.W
            else:
                Vm1_new = numerator / denominator
            
            if Vm1_new <= 0:
                Vm1_new = 0.1
            
            # Step 5: Get Lm from Eq. 26.7-13
            Lm_new = self.W + Vm1_new
            
            # Step 6: Recalculate xm from Eq. 26.7-15
            xm_new = ((ym1 * Vm1_new) + (self.W * self.xW)) / Lm_new
            xm_new = max(0.0, min(1.0, xm_new))
            
            # Check convergence
            if abs(xm_new - xm_approx) < tol:
                break
            
            Vm1 = Vm1_new
            Lm = Lm_new
        
        return {
            "ym1": ym1,
            "xm": xm_new,
            "Vm1": Vm1_new,
            "Lm": Lm_new,
            "hm": hm,
            "Hm1": Hm1,
        }
    
    def generate_operating_line(self, y_points: List[float]) -> List[Dict[str, float]]:
        """Generate multiple points on the stripping operating line"""
        results = []
        for y in sorted(y_points, reverse=True):
            results.append(self.operating_line_point(y))
        return results
    
    def verify_energy_balance(self) -> float:
        """
        Verify overall energy balance using Eq. 26.7-17
        qR = D hD + W hW + qc - F hF
        
        Returns:
            Difference between calculated and specified qR
        """
        if None in [self.D, self.hD, self.qc, self.F, self.hF]:
            raise ChemEngError("Need D, hD, qc, F, and hF for energy balance verification")
        
        qR_calc = self.D * self.hD + self.W * self.hW + self.qc - self.F * self.hF
        return qR_calc - self.qR