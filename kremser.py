# bank/separations/absorption/kremser.py
"""Kremser equations for absorbers and strippers"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

from bank.core.validation import (
    check_positive, check_in_closed_01, ChemEngError, InputError
)


@dataclass
class KremserSpec:
    """
    Specification for Kremser equation calculations.
    
    For absorption:
        A = L/(mG) is absorption factor
        fraction_absorbed = (A^(N+1) - A)/(A^(N+1) - 1)
    
    For stripping:
        S = mG/L is stripping factor
        fraction_stripped = (S^(N+1) - S)/(S^(N+1) - 1)
    """
    # Choose one mode
    A: Optional[float] = None  # Absorption factor
    S: Optional[float] = None  # Stripping factor
    
    N: Optional[int] = None     # Number of stages
    fraction: Optional[float] = None  # Fraction absorbed/stripped
    
    # Numerical
    tol: float = 1e-12
    maxiter: int = 400


class KremserSolver:
    """Solver using Kremser equations"""
    
    def __init__(self, spec: KremserSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        """Validate inputs"""
        if self.spec.A is not None and self.spec.S is not None:
            raise InputError("Provide either A or S, not both")
        
        if self.spec.A is not None:
            check_positive("A", self.spec.A)
            self.mode = "absorption"
            self.factor = self.spec.A
        elif self.spec.S is not None:
            check_positive("S", self.spec.S)
            self.mode = "stripping"
            self.factor = self.spec.S
        else:
            raise InputError("Must provide either A or S")
        
        # Count specified parameters
        n_specified = sum([
            self.spec.N is not None,
            self.spec.fraction is not None
        ])
        
        if n_specified != 1:
            raise InputError("Must specify exactly one of: N, fraction")
    
    def _kremser_absorption(self, N: int, A: float) -> float:
        """Fraction absorbed for given N and A"""
        if abs(A - 1.0) < 1e-12:
            return N / (N + 1)
        
        return (A ** (N + 1) - A) / (A ** (N + 1) - 1)
    
    def _kremser_stripping(self, N: int, S: float) -> float:
        """Fraction stripped for given N and S"""
        if abs(S - 1.0) < 1e-12:
            return N / (N + 1)
        
        return (S ** (N + 1) - S) / (S ** (N + 1) - 1)
    
    def _fraction(self, N: int) -> float:
        """Calculate fraction for given N"""
        if self.mode == "absorption":
            return self._kremser_absorption(N, self.factor)
        else:
            return self._kremser_stripping(N, self.factor)
    
    def solve_for_N(self, fraction: float) -> int:
        """Solve for N given fraction"""
        check_in_closed_01("fraction", fraction)
        
        if fraction <= 0 or fraction >= 1:
            raise ChemEngError(f"Fraction must be between 0 and 1, got {fraction}")
        
        # Find N by solving
        N = 1
        max_N = 100
        
        while N <= max_N:
            f_calc = self._fraction(N)
            if f_calc >= fraction - self.spec.tol:
                return N
            N += 1
        
        raise ChemEngError(f"Fraction {fraction} requires > {max_N} stages")
    
    def solve_for_fraction(self, N: int) -> float:
        """Solve for fraction given N"""
        if N < 1:
            raise ValueError("N must be >= 1")
        
        return self._fraction(N)
    
    def solve(self) -> Dict[str, Any]:
        """Main solving method"""
        
        if self.spec.N is not None:
            # Solve for fraction
            fraction = self.solve_for_fraction(self.spec.N)
            result = {
                "method": "given_N",
                "N": self.spec.N,
                "fraction": fraction,
            }
        else:
            # Solve for N
            N = self.solve_for_N(self.spec.fraction)
            result = {
                "method": "given_fraction",
                "N": N,
                "fraction": self.spec.fraction,
            }
        
        # Add factor info
        if self.mode == "absorption":
            result["absorption_factor"] = self.factor
            result["stripping_factor"] = 1.0 / self.factor
        else:
            result["stripping_factor"] = self.factor
            result["absorption_factor"] = 1.0 / self.factor
        
        return result


# Convenience functions
def kremser_absorption(A: float, N: int) -> float:
    """Quick Kremser absorption calculation"""
    spec = KremserSpec(A=A, N=N)
    solver = KremserSolver(spec)
    return solver.solve()["fraction"]


def kremser_stripping(S: float, N: int) -> float:
    """Quick Kremser stripping calculation"""
    spec = KremserSpec(S=S, N=N)
    solver = KremserSolver(spec)
    return solver.solve()["fraction"]


def stages_for_absorption(A: float, fraction: float) -> int:
    """Number of stages needed for given absorption factor and fraction"""
    spec = KremserSpec(A=A, fraction=fraction)
    solver = KremserSolver(spec)
    return solver.solve()["N"]


def stages_for_stripping(S: float, fraction: float) -> int:
    """Number of stages needed for given stripping factor and fraction"""
    spec = KremserSpec(S=S, fraction=fraction)
    solver = KremserSolver(spec)
    return solver.solve()["N"]