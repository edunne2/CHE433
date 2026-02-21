"""Tray efficiency calculations - Eqs. 26.5-1 to 26.5-4"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import math

from bank.core.validation import check_positive, check_in_closed_01, ChemEngError


def overall_efficiency(N_ideal: float, N_actual: float) -> float:
    """
    Overall tray efficiency - Eq. 26.5-1
    
    EO = (number of ideal trays) / (number of actual trays)
    """
    check_positive("N_ideal", N_ideal)
    check_positive("N_actual", N_actual)
    return N_ideal / N_actual


def murphree_efficiency(yn: float, yn1: float, yn_star: float) -> float:
    """
    Murphree tray efficiency - Eq. 26.5-2
    
    EM = (yn - yn+1) / (yn* - yn+1)
    """
    check_in_closed_01("yn", yn)
    check_in_closed_01("yn1", yn1)
    check_in_closed_01("yn_star", yn_star)
    
    denominator = yn_star - yn1
    if abs(denominator) < 1e-12:
        return 1.0
    
    return (yn - yn1) / denominator


def point_efficiency(yn_prime: float, yn1_prime: float, yn_star: float) -> float:
    """
    Point or local efficiency - Eq. 26.5-3
    
    EMP = (yn' - yn+1') / (yn* - yn+1')
    """
    check_in_closed_01("yn_prime", yn_prime)
    check_in_closed_01("yn1_prime", yn1_prime)
    check_in_closed_01("yn_star", yn_star)
    
    denominator = yn_star - yn1_prime
    if abs(denominator) < 1e-12:
        return 1.0
    
    return (yn_prime - yn1_prime) / denominator


def efficiency_relationship(EM: float, m: float, V: float, L: float) -> float:
    """
    Relationship between Murphree and overall efficiency - Eq. 26.5-4
    
    EO = log[1 + EM (mV/L - 1)] / log(mV/L)
    """
    check_in_closed_01("EM", EM)
    check_positive("m", m)
    check_positive("V", V)
    check_positive("L", L)
    
    lambda_factor = m * V / L
    
    if abs(lambda_factor - 1.0) < 1e-12:
        return EM
    
    numerator = math.log(1.0 + EM * (lambda_factor - 1.0))
    denominator = math.log(lambda_factor)
    
    return numerator / denominator


class TrayEfficiency:
    """
    Comprehensive tray efficiency calculations.
    
    Eqs. 26.5-1 to 26.5-4:
    - Overall efficiency EO
    - Murphree efficiency EM
    - Point efficiency EMP
    - Relationship between efficiencies
    """
    
    def __init__(
        self,
        N_theoretical: float = None,
        N_actual: float = None,
        EM: float = None,
        m: float = None,
        V: float = None,
        L: float = None
    ):
        self.N_theoretical = N_theoretical
        self.N_actual = N_actual
        self.EM = EM
        self.m = m
        self.V = V
        self.L = L
    
    def calculate_EO_from_N(self) -> float:
        """Calculate overall efficiency from tray counts"""
        if self.N_theoretical is None or self.N_actual is None:
            raise ChemEngError("Need N_theoretical and N_actual")
        return overall_efficiency(self.N_theoretical, self.N_actual)
    
    def calculate_actual_trays(self, EO: float = None) -> float:
        """Calculate actual trays from overall efficiency"""
        if self.N_theoretical is None:
            raise ChemEngError("Need N_theoretical")
        eo = EO if EO is not None else self.calculate_EO_from_N()
        return self.N_theoretical / eo
    
    def calculate_EO_from_EM(self) -> float:
        """Calculate overall efficiency from Murphree efficiency"""
        if None in [self.EM, self.m, self.V, self.L]:
            raise ChemEngError("Need EM, m, V, and L")
        return efficiency_relationship(self.EM, self.m, self.V, self.L)
    
    def graphical_efficiency_tray(
        self,
        x_prev: float,
        y_prev: float,
        y_eq: float,
        y_op_at_x: float
    ) -> Tuple[float, float]:
        """
        Apply Murphree efficiency to a single tray in graphical construction.
        
        For a given tray, the actual vapor leaving (y_actual) is:
        y_actual = y_prev + EM * (y_eq - y_prev)
        """
        if self.EM is None:
            raise ChemEngError("Need EM for graphical construction")
        
        y_actual = y_prev + self.EM * (y_eq - y_prev)
        
        # Find corresponding x on operating line
        # This would require the operating line equation
        return y_actual
    
    def step_off_with_efficiency(
        self,
        x_start: float,
        y_start: float,
        eq_func,
        op_func,
        x_target: float,
        max_stages: int = 100
    ) -> Dict[str, Any]:
        """
        Step off stages using Murphree efficiency.
        
        This implements the graphical method shown in Fig. 26.5-2.
        """
        if self.EM is None:
            raise ChemEngError("Need EM for efficient stepping")
        
        points = [(x_start, y_start)]
        x_current = x_start
        y_current = y_start
        stage_count = 0
        
        for n in range(max_stages):
            # Equilibrium vapor at current liquid
            y_eq = eq_func(x_current)
            
            # Actual vapor leaving (with efficiency)
            y_actual = y_current + self.EM * (y_eq - y_current)
            
            # Find x on operating line at this y
            # This requires solving x from operating line
            # Simplified - in practice would need operating line equation
            
            points.append((x_current, y_actual))
            stage_count += 1
            
            if x_current <= x_target + 1e-12:
                break
        
        return {
            "points": points,
            "stages": stage_count,
        }