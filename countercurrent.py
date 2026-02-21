# bank/separations/absorption/countercurrent.py
"""Countercurrent multistage absorption/stripping towers"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
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
    
    For absorption:
    - Liquid enters at top (stage 1), gas at bottom (stage N)
    - Solute transfers from gas to liquid
    
    For stripping:
    - Gas enters at bottom, liquid at top
    - Solute transfers from liquid to gas
    """
    # Stream flows (assumed constant, dilute system)
    L: float          # Liquid flow rate
    G: float          # Gas flow rate
    
    # Inlet compositions
    x_in: float       # Liquid inlet (top)
    y_in: float       # Gas inlet (bottom)
    
    # Equilibrium
    eq: EquilibriumModel
    
    # Specification (provide one)
    x_out: Optional[float] = None  # Liquid outlet (bottom)
    y_out: Optional[float] = None  # Gas outlet (top)
    N_stages: Optional[int] = None # Number of stages
    
    # Operating mode
    mode: str = "absorption"  # "absorption" or "stripping"
    
    # Numerical
    tol: float = 1e-12
    maxiter: int = 400


class CountercurrentAbsorber:
    """
    Solver for countercurrent multistage towers.
    
    Can solve for:
    - Outlet compositions given number of stages
    - Number of stages required for given separation
    """
    
    def __init__(self, spec: CountercurrentSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        """Validate inputs"""
        check_positive("L", self.spec.L)
        check_positive("G", self.spec.G)
        check_in_closed_01("x_in", self.spec.x_in)
        check_in_closed_01("y_in", self.spec.y_in)
        
        if self.spec.mode not in ["absorption", "stripping"]:
            raise InputError(f"Mode must be 'absorption' or 'stripping'")
        
        # Check we have enough info
        n_specified = sum([
            self.spec.x_out is not None,
            self.spec.y_out is not None,
            self.spec.N_stages is not None
        ])
        
        if n_specified == 0:
            raise InputError("Must specify at least one of: x_out, y_out, N_stages")
        
        self.A = self._absorption_factor()
    
    def _absorption_factor(self) -> float:
        """Calculate absorption factor A = L/(mG)"""
        # Approximate m from equilibrium at average conditions
        x_avg = (self.spec.x_in + (self.spec.x_out or self.spec.x_in)) / 2
        m = self.spec.eq.K_value(x_avg)
        return self.spec.L / (m * self.spec.G)
    
    def _kremser_absorption(self, N: int, A: float) -> float:
        """
        Kremser equation for absorption factor method.
        Returns fraction absorbed.
        """
        if abs(A - 1.0) < 1e-12:
            return N / (N + 1)
        
        return (A ** (N + 1) - A) / (A ** (N + 1) - 1)
    
    def _kremser_stripping(self, N: int, S: float) -> float:
        """
        Kremser equation for stripping factor method.
        S = 1/A is stripping factor.
        Returns fraction stripped.
        """
        S = 1.0 / self._absorption_factor()
        if abs(S - 1.0) < 1e-12:
            return N / (N + 1)
        
        return (S ** (N + 1) - S) / (S ** (N + 1) - 1)
    
    def solve_with_N(self, N: int) -> Dict[str, Any]:
        """
        Solve for outlet compositions given number of stages.
        
        Args:
            N: Number of theoretical stages
        
        Returns:
            Dictionary with outlet compositions
        """
        if N < 1:
            raise ValueError("Number of stages must be >= 1")
        
        A = self._absorption_factor()
        
        if self.spec.mode == "absorption":
            fraction = self._kremser_absorption(N, A)
            
            # For absorption: y_out is top, x_out is bottom
            # Fraction of solute in inlet gas that is absorbed
            y_out = self.spec.y_in * (1 - fraction)
            
            # Mass balance for liquid
            solute_absorbed = self.spec.G * (self.spec.y_in - y_out)
            x_out = self.spec.x_in + solute_absorbed / self.spec.L
            
        else:  # stripping
            S = 1.0 / A
            fraction = self._kremser_stripping(N, S)
            
            # For stripping: x_out is bottom, y_out is top
            # Fraction of solute in inlet liquid that is stripped
            x_out = self.spec.x_in * (1 - fraction)
            
            # Mass balance for gas
            solute_stripped = self.spec.L * (self.spec.x_in - x_out)
            y_out = self.spec.y_in + solute_stripped / self.spec.G
        
        # Verify mass balance
        solute_in = self.spec.L * self.spec.x_in + self.spec.G * self.spec.y_in
        solute_out = self.spec.L * x_out + self.spec.G * y_out
        
        return {
            "outlets": {
                "x_out": x_out,
                "y_out": y_out,
            },
            "stages": N,
            "absorption_factor": A,
            "fraction_transferred": fraction,
            "verification": {
                "mass_balance_error": solute_in - solute_out,
            }
        }
    
    def solve_for_N(self, target_x_out: Optional[float] = None, 
                    target_y_out: Optional[float] = None) -> int:
        """
        Solve for number of stages required to achieve target outlet.
        
        Args:
            target_x_out: Desired liquid outlet composition
            target_y_out: Desired gas outlet composition
        
        Returns:
            Required number of theoretical stages
        """
        if target_x_out is None and target_y_out is None:
            raise InputError("Must provide target_x_out or target_y_out")
        
        if target_x_out is not None:
            check_in_closed_01("target_x_out", target_x_out)
            target = target_x_out
            is_liquid = True
        else:
            check_in_closed_01("target_y_out", target_y_out)
            target = target_y_out
            is_liquid = False
        
        def residual(N_float: float):
            N = int(round(N_float))
            if N < 1:
                return float('inf')
            
            result = self.solve_with_N(N)
            
            if is_liquid:
                return result["outlets"]["x_out"] - target
            else:
                return result["outlets"]["y_out"] - target
        
        # Find N by trial
        N = 1
        max_N = 100
        last_error = float('inf')
        
        while N <= max_N:
            error = residual(N)
            if abs(error) <= self.spec.tol:
                return N
            
            # Check if we've passed the target
            if error * last_error < 0:
                # Binary search between N-1 and N
                return self._binary_search_N(N-1, N, target, is_liquid)
            
            last_error = error
            N += 1
        
        raise ChemEngError(f"Could not find N <= {max_N} for target")
    
    def _binary_search_N(self, N1: int, N2: int, target: float, is_liquid: bool) -> int:
        """Binary search for N between N1 and N2"""
        while N2 - N1 > 1:
            N_mid = (N1 + N2) // 2
            result = self.solve_with_N(N_mid)
            
            if is_liquid:
                value = result["outlets"]["x_out"]
            else:
                value = result["outlets"]["y_out"]
            
            if value > target:
                N2 = N_mid
            else:
                N1 = N_mid
        
        # Return closer one
        r1 = self.solve_with_N(N1)
        r2 = self.solve_with_N(N2)
        
        v1 = r1["outlets"]["x_out"] if is_liquid else r1["outlets"]["y_out"]
        v2 = r2["outlets"]["x_out"] if is_liquid else r2["outlets"]["y_out"]
        
        return N1 if abs(v1 - target) < abs(v2 - target) else N2
    
    def solve(self) -> Dict[str, Any]:
        """Main solving method"""
        
        if self.spec.N_stages is not None:
            # Solve for outlets given N
            result = self.solve_with_N(self.spec.N_stages)
            result["method"] = "given_N"
            
        elif self.spec.x_out is not None:
            # Solve for N given x_out
            N = self.solve_for_N(target_x_out=self.spec.x_out)
            result = self.solve_with_N(N)
            result["method"] = "solve_for_N_from_x_out"
            
        elif self.spec.y_out is not None:
            # Solve for N given y_out
            N = self.solve_for_N(target_y_out=self.spec.y_out)
            result = self.solve_with_N(N)
            result["method"] = "solve_for_N_from_y_out"
            
        else:
            raise ChemEngError("Insufficient specifications")
        
        # Calculate performance metrics
        if self.spec.mode == "absorption":
            removal = (self.spec.y_in - result["outlets"]["y_out"]) / self.spec.y_in
            result["performance"] = {
                "removal_efficiency": removal,
                "solute_recovered_in_liquid": result["outlets"]["x_out"] - self.spec.x_in,
            }
        else:  # stripping
            removal = (self.spec.x_in - result["outlets"]["x_out"]) / self.spec.x_in
            result["performance"] = {
                "stripping_efficiency": removal,
                "solute_recovered_in_gas": result["outlets"]["y_out"] - self.spec.y_in,
            }
        
        return result


def stage_to_stage_absorption(
    L: float,
    G: float,
    x_in: float,
    y_in: float,
    eq: EquilibriumModel,
    N: int,
    method: str = "kremser",
) -> List[Dict[str, float]]:
    """
    Perform stage-to-stage calculation for absorber.
    
    Args:
        L: Liquid flow rate
        G: Gas flow rate
        x_in: Liquid inlet composition (top)
        y_in: Gas inlet composition (bottom)
        eq: Equilibrium model
        N: Number of stages
        method: "kremser" for analytical, "stagewise" for numerical
    
    Returns:
        List of stage compositions
    """
    if method == "kremser":
        spec = CountercurrentSpec(
            L=L, G=G, x_in=x_in, y_in=y_in,
            eq=eq, N_stages=N, mode="absorption"
        )
        solver = CountercurrentAbsorber(spec)
        result = solver.solve_with_N(N)
        
        # Generate stage profile (simplified)
        stages = []
        A = solver._absorption_factor()
        
        for i in range(1, N + 1):
            stages.append({
                "stage": i,
                "x": x_in + (result["outlets"]["x_out"] - x_in) * (i / N),
                "y": y_in + (result["outlets"]["y_out"] - y_in) * ((N - i + 1) / N),
            })
        
        return stages
    
    else:  # stagewise (iterative)
        # Will implement full stage-by-stage if needed
        raise NotImplementedError("Stagewise calculation coming soon")