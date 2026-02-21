"""Countercurrent multistage towers - Eqs. 22.1-10 to 22.1-29, 22.4-1 to 22.4-2"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import math

from bank.core.validation import (
    check_positive, check_in_closed_01, ChemEngError, InputError
)
from bank.core.numerical import bisection
from bank.separations.equilibrium import EquilibriumModel


@dataclass
class CountercurrentSpec:
    """
    Specification for countercurrent multistage tower.
    Uses equations 22.1-10 to 22.1-16.
    """
    L: float
    G: float
    x_in: float
    y_in: float
    eq: EquilibriumModel
    x_out: Optional[float] = None
    y_out: Optional[float] = None
    N_stages: Optional[int] = None
    mode: str = "absorption"
    tol: float = 1e-12
    maxiter: int = 400


class CountercurrentAbsorber:
    """Solver for countercurrent multistage towers - Eqs. 22.1-10 to 22.1-29"""
    
    def __init__(self, spec: CountercurrentSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        check_positive("L", self.spec.L)
        check_positive("G", self.spec.G)
        check_in_closed_01("x_in", self.spec.x_in)
        check_in_closed_01("y_in", self.spec.y_in)
        
        if self.spec.mode not in ["absorption", "stripping"]:
            raise InputError(f"Mode must be 'absorption' or 'stripping'")
        
        n_specified = sum([
            self.spec.x_out is not None,
            self.spec.y_out is not None,
            self.spec.N_stages is not None
        ])
        
        if n_specified == 0:
            raise InputError("Must specify at least one of: x_out, y_out, N_stages")
    
    def _absorption_factor(self, x_avg: Optional[float] = None) -> float:
        """Calculate absorption factor A = L/(mV) (Eq. 22.1-23)"""
        if x_avg is None:
            x_avg = (self.spec.x_in + (self.spec.x_out or self.spec.x_in)) / 2
        m = self.spec.eq.K_value(x_avg)
        return self.spec.L / (m * self.spec.G)
    
    def _kremser_absorption(self, N: int, A: float) -> float:
        """Kremser equation for absorption - Eq. 22.1-27"""
        if abs(A - 1.0) < 1e-12:
            return N / (N + 1)
        return (A ** (N + 1) - A) / (A ** (N + 1) - 1)
    
    def _kremser_stripping(self, N: int, S: float) -> float:
        """Kremser equation for stripping - Eq. 22.1-24"""
        if abs(S - 1.0) < 1e-12:
            return N / (N + 1)
        return (S ** (N + 1) - S) / (S ** (N + 1) - 1)
    
    def solve_with_N(self, N: int) -> Dict[str, Any]:
        """Solve for outlet compositions given N stages using Kremser equations"""
        if N < 1:
            raise ValueError("Number of stages must be >= 1")
        
        A = self._absorption_factor()
        
        if self.spec.mode == "absorption":
            fraction = self._kremser_absorption(N, A)
            y_out = self.spec.y_in * (1 - fraction)
            solute_transferred = self.spec.G * (self.spec.y_in - y_out)
            x_out = self.spec.x_in + solute_transferred / self.spec.L
        else:
            S = 1.0 / A
            fraction = self._kremser_stripping(N, S)
            x_out = self.spec.x_in * (1 - fraction)
            solute_transferred = self.spec.L * (self.spec.x_in - x_out)
            y_out = self.spec.y_in + solute_transferred / self.spec.G
        
        return {
            "outlets": {"x_out": x_out, "y_out": y_out},
            "stages": N,
            "absorption_factor": A,
            "fraction_transferred": fraction,
        }
    
    def solve_for_N(self, target_x_out: Optional[float] = None, 
                    target_y_out: Optional[float] = None) -> int:
        """Solve for N required to achieve target outlet using Kremser equations"""
        if target_x_out is None and target_y_out is None:
            raise InputError("Must provide target_x_out or target_y_out")
        
        def residual(N_float: float):
            N = int(round(N_float))
            if N < 1:
                return float('inf')
            result = self.solve_with_N(N)
            if target_x_out is not None:
                return result["outlets"]["x_out"] - target_x_out
            else:
                return result["outlets"]["y_out"] - target_y_out
        
        N = 1
        max_N = 100
        last_error = float('inf')
        
        while N <= max_N:
            error = residual(N)
            if abs(error) <= self.spec.tol:
                return N
            if error * last_error < 0:
                return self._binary_search_N(N-1, N, target_x_out, target_y_out)
            last_error = error
            N += 1
        
        raise ChemEngError(f"Could not find N <= {max_N} for target")
    
    def _binary_search_N(self, N1: int, N2: int, target_x: Optional[float], target_y: Optional[float]) -> int:
        while N2 - N1 > 1:
            N_mid = (N1 + N2) // 2
            result = self.solve_with_N(N_mid)
            if target_x is not None:
                value = result["outlets"]["x_out"]
            else:
                value = result["outlets"]["y_out"]
            
            if value > (target_x or target_y):
                N2 = N_mid
            else:
                N1 = N_mid
        
        r1 = self.solve_with_N(N1)
        r2 = self.solve_with_N(N2)
        v1 = r1["outlets"]["x_out"] if target_x is not None else r1["outlets"]["y_out"]
        v2 = r2["outlets"]["x_out"] if target_x is not None else r2["outlets"]["y_out"]
        target = target_x or target_y
        return N1 if abs(v1 - target) < abs(v2 - target) else N2
    
    def solve(self) -> Dict[str, Any]:
        """Main solving method"""
        if self.spec.N_stages is not None:
            result = self.solve_with_N(self.spec.N_stages)
            result["method"] = "given_N"
        elif self.spec.x_out is not None:
            N = self.solve_for_N(target_x_out=self.spec.x_out)
            result = self.solve_with_N(N)
            result["method"] = "solve_for_N_from_x_out"
        elif self.spec.y_out is not None:
            N = self.solve_for_N(target_y_out=self.spec.y_out)
            result = self.solve_with_N(N)
            result["method"] = "solve_for_N_from_y_out"
        else:
            raise ChemEngError("Insufficient specifications")
        
        if self.spec.mode == "absorption":
            removal = (self.spec.y_in - result["outlets"]["y_out"]) / self.spec.y_in
        else:
            removal = (self.spec.x_in - result["outlets"]["x_out"]) / self.spec.x_in
        
        result["performance"] = {"removal_efficiency": removal}
        return result


def kremser_N_absorption(
    y_in: float,
    y_out: float,
    x_in: float,
    m: float,
    A: float
) -> float:
    """
    Kremser equation for number of stages in absorption - Eq. 22.1-28
    
    N = ln[ ((y_in - m*x_in)/(y_out - m*x_in)) (1 - 1/A) + 1/A ] / ln A
    """
    if A <= 0 or abs(A - 1.0) < 1e-10:
        return float('inf')
    
    numerator = ((y_in - m * x_in) / (y_out - m * x_in)) * (1 - 1/A) + 1/A
    
    if numerator <= 0:
        return float('inf')
    
    return math.log(numerator) / math.log(A)


def kremser_N_stripping(
    x_in: float,
    x_out: float,
    y_in: float,
    m: float,
    A: float
) -> float:
    """
    Kremser equation for number of stages in stripping - Eq. 22.1-25
    
    N = ln[ ((x_in - y_in/m)/(x_out - y_in/m)) (1-A) + A ] / ln(1/A)
    """
    if A <= 0 or abs(A - 1.0) < 1e-10:
        return float('inf')
    
    numerator = ((x_in - y_in/m) / (x_out - y_in/m)) * (1 - A) + A
    
    if numerator <= 0:
        return float('inf')
    
    return math.log(numerator) / math.log(1/A)