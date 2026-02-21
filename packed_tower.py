"""Packed tower design - Eqs. 22.5-1 to 22.5-55, 22.7-1"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
import math

from bank.core.validation import (
    check_positive, check_in_closed_01, ChemEngError, InputError
)
from bank.core.numerical import integrate_trapezoid, log_mean
from bank.separations.equilibrium import EquilibriumModel


@dataclass
class PackedTowerSpec:
    """
    Specification for packed tower design.
    Uses Eqs. 22.5-1 to 22.5-3 for operating line.
    """
    G: float
    L: float
    y_in: float
    x_in: float
    eq: EquilibriumModel
    y_out: Optional[float] = None
    x_out: Optional[float] = None
    H: Optional[float] = None
    HOG: Optional[float] = None
    HETP: Optional[float] = None
    NOG: Optional[float] = None
    kya: Optional[float] = None
    kxa: Optional[float] = None
    Kya: Optional[float] = None
    S: float = 1.0
    tol: float = 1e-12
    maxiter: int = 400


class PackedTowerSolver:
    """Solver for packed tower design - Eqs. 22.5-1 to 22.5-55"""
    
    def __init__(self, spec: PackedTowerSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        check_positive("G", self.spec.G)
        check_positive("L", self.spec.L)
        check_in_closed_01("y_in", self.spec.y_in)
        check_in_closed_01("x_in", self.spec.x_in)
        
        n_outlets = sum([
            self.spec.y_out is not None,
            self.spec.x_out is not None
        ])
        
        if n_outlets == 0:
            raise InputError("Must specify at least one outlet composition")
        
        if n_outlets == 1:
            self._calculate_missing_outlet()
    
    def _calculate_missing_outlet(self):
        """Calculate missing outlet from mass balance - Eq. 22.5-1"""
        if self.spec.y_out is not None and self.spec.x_out is None:
            solute_removed = self.spec.G * (self.spec.y_in - self.spec.y_out)
            self.x_out_calc = self.spec.x_in + solute_removed / self.spec.L
            self.y_out_calc = self.spec.y_out
        elif self.spec.x_out is not None and self.spec.y_out is None:
            solute_removed = self.spec.L * (self.spec.x_out - self.spec.x_in)
            self.y_out_calc = self.spec.y_in - solute_removed / self.spec.G
            self.x_out_calc = self.spec.x_out
        else:
            self.x_out_calc = self.spec.x_out
            self.y_out_calc = self.spec.y_out
    
    def _driving_force_gas(self, y: float) -> float:
        """Calculate gas phase driving force (y - yi)"""
        x = self.spec.x_in + (self.spec.L / self.spec.G) * (self.spec.y_in - y)
        yi = self._estimate_yi(x, y)  # Need interface calc
        return y - yi
    
    def _driving_force_overall_gas(self, y: float) -> float:
        """Calculate overall gas phase driving force (y - y*)"""
        x = self.spec.x_in + (self.spec.L / self.spec.G) * (self.spec.y_in - y)
        y_star = self.spec.eq.y_of_x(x)
        return y - y_star
    
    def _estimate_yi(self, x: float, y: float) -> float:
        """Estimate interface concentration - simplified"""
        # For dilute systems, yi ≈ y*
        return self.spec.eq.y_of_x(x)
    
    def calculate_NOG(self) -> float:
        """
        Calculate number of transfer units - Eq. 22.5-43
        NOG = ∫ dy/(y - y*)
        """
        y_start = self.spec.y_in
        y_end = self.y_out_calc
        
        if abs(y_start - y_end) < 1e-12:
            return 0.0
        
        def integrand(y: float):
            return 1.0 / self._driving_force_overall_gas(y)
        
        return integrate_trapezoid(integrand, y_end, y_start, n=1000)
    
    def calculate_HOG(self) -> float:
        """
        Calculate height of a transfer unit - Eq. 22.5-38
        HOG = V/(K'ya S)
        """
        if self.spec.HOG is not None:
            return self.spec.HOG
        
        if self.spec.Kya is not None:
            V_avg = (self.spec.G + (self.spec.G * (1 - self.y_out_calc))) / 2
            return V_avg / (self.spec.Kya * self.spec.S)
        
        # Default empirical correlation
        return 0.5
    
    def calculate_HETP(self) -> float:
        """
        Calculate HETP - Eq. 22.5-52
        HETP = HOG * ln(1/A) / ((1-A)/A)
        """
        if self.spec.HETP is not None:
            return self.spec.HETP
        
        HOG = self.calculate_HOG()
        x_avg = (self.spec.x_in + self.x_out_calc) / 2
        m = self.spec.eq.K_value(x_avg)
        A = self.spec.L / (m * self.spec.G)
        
        if abs(A - 1.0) < 1e-12:
            return HOG
        
        return HOG * math.log(1/A) / ((1 - A) / A)
    
    def solve(self) -> Dict[str, Any]:
        """Main solving method - implements Eqs. 22.5-14 to 22.5-17"""
        
        NOG = self.calculate_NOG()
        HOG = self.calculate_HOG()
        H = NOG * HOG
        HETP = self.calculate_HETP()
        
        N_theoretical = H / HETP if HETP > 0 else None
        
        solute_in = self.spec.L * self.spec.x_in + self.spec.G * self.spec.y_in
        solute_out = self.spec.L * self.x_out_calc + self.spec.G * self.y_out_calc
        
        return {
            "inputs": {
                "y_in": self.spec.y_in, "y_out": self.y_out_calc,
                "x_in": self.spec.x_in, "x_out": self.x_out_calc,
                "L": self.spec.L, "G": self.spec.G,
            },
            "transfer_units": {
                "NOG": NOG, "HOG": HOG, "H": H, "HETP": HETP,
            },
            "stages": {
                "N_theoretical": N_theoretical,
                "N_actual": math.ceil(N_theoretical) if N_theoretical else None,
            },
            "verification": {
                "mass_balance_error": solute_in - solute_out,
            }
        }