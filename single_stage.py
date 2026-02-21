"""Single-stage equilibrium contact - Eqs. 22.1-4 to 22.1-9"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from bank.core.validation import (
    check_positive, check_in_closed_01, ChemEngError, InputError
)
from bank.core.numerical import bisection
from bank.separations.equilibrium import EquilibriumModel


@dataclass
class SingleStageSpec:
    """Specification for single-stage equilibrium contact - Eqs. 22.1-4 to 22.1-6"""
    L_in: float
    G_in: float
    x_in: float
    y_in: float
    eq: EquilibriumModel
    x_out: Optional[float] = None
    y_out: Optional[float] = None
    mode: str = "absorption"
    tol: float = 1e-12
    maxiter: int = 400
    
    def __post_init__(self):
        check_positive("L_in", self.L_in)
        check_positive("G_in", self.G_in)
        check_in_closed_01("x_in", self.x_in)
        check_in_closed_01("y_in", self.y_in)
        
        if self.mode not in ["absorption", "stripping"]:
            raise InputError(f"Mode must be 'absorption' or 'stripping'")


class SingleStageAbsorber:
    """Solver for single-stage equilibrium contact - Eqs. 22.1-4 to 22.1-9"""
    
    def __init__(self, spec: SingleStageSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        n_specified = sum([
            self.spec.x_out is not None,
            self.spec.y_out is not None
        ])
        
        if n_specified == 2:
            self._verify_mass_balance()
    
    def _verify_mass_balance(self):
        """Verify mass balance using Eq. 22.1-7"""
        if self.spec.x_out is None or self.spec.y_out is None:
            return
        
        L = self.spec.L_in
        G = self.spec.G_in
        x_in = self.spec.x_in
        y_in = self.spec.y_in
        x_out = self.spec.x_out
        y_out = self.spec.y_out
        
        L_prime = L * (1 - x_in)
        V_prime = G * (1 - y_in)
        
        lhs = L_prime * (x_in/(1-x_in)) + V_prime * (y_in/(1-y_in))
        rhs = L_prime * (x_out/(1-x_out)) + V_prime * (y_out/(1-y_out))
        
        if abs(lhs - rhs) > 1e-6 * max(lhs, rhs, 1.0):
            raise ChemEngError(f"Mass balance failed: {lhs} vs {rhs}")
    
    def _equilibrium_relation(self, x: float) -> float:
        """Equilibrium: y = f(x) (Eq. 22.1-8)"""
        return self.spec.eq.y_of_x(x)
    
    def _operating_line(self, x: float) -> float:
        """
        Operating line from mass balance (Eq. 22.1-7 rearranged)
        y = y_in + (L/G)(x_in - x) for constant flows
        """
        L = self.spec.L_in
        G = self.spec.G_in
        return self.spec.y_in + (L / G) * (self.spec.x_in - x)
    
    def _stage_residual(self, x: float) -> float:
        """Residual: y_eq(x) - y_op(x) = 0 at equilibrium"""
        y_eq = self._equilibrium_relation(x)
        y_op = self._operating_line(x)
        return y_eq - y_op
    
    def solve_outlets(self) -> Dict[str, Any]:
        """Solve for outlet compositions using Eq. 22.1-7"""
        
        if self.spec.mode == "absorption":
            x_low = self.spec.x_in
            x_high = 1.0
        else:
            x_low = 0.0
            x_high = self.spec.x_in
        
        try:
            x_out = bisection(
                self._stage_residual, x_low, x_high,
                tol=self.spec.tol, maxiter=self.spec.maxiter,
                expand_bracket=True
            )
        except Exception as e:
            x_out = bisection(
                self._stage_residual, 0.0, 1.0,
                tol=self.spec.tol, maxiter=self.spec.maxiter,
                expand_bracket=True, expand_factor=1.5
            )
        
        y_out = self._equilibrium_relation(x_out)
        
        return {
            "outlets": {"x_out": x_out, "y_out": y_out},
            "flows": {"L_out": self.spec.L_in, "G_out": self.spec.G_in},
        }
    
    def solve_LG_ratio(self, target_x_out: float) -> float:
        """Solve for L/G ratio required to achieve target liquid outlet"""
        check_in_closed_01("target_x_out", target_x_out)
        
        original_L = self.spec.L_in
        original_G = self.spec.G_in
        
        def residual(LG_ratio: float):
            self.spec.L_in = LG_ratio * original_G
            result = self._stage_residual(target_x_out)
            self.spec.L_in = original_L
            return result
        
        LG_ratio = bisection(residual, 0.01, 100.0, tol=self.spec.tol, expand_bracket=True)
        return LG_ratio
    
    def solve(self) -> Dict[str, Any]:
        """Main solving method"""
        if self.spec.x_out is not None and self.spec.y_out is not None:
            self._verify_mass_balance()
            result = {
                "outlets": {"x_out": self.spec.x_out, "y_out": self.spec.y_out},
                "flows": {"L_out": self.spec.L_in, "G_out": self.spec.G_in},
            }
        else:
            result = self.solve_outlets()
        
        x_out = result["outlets"]["x_out"]
        y_out = result["outlets"]["y_out"]
        
        if self.spec.mode == "absorption":
            recovery = (self.spec.y_in - y_out) / self.spec.y_in if self.spec.y_in > 0 else 0
        else:
            recovery = (self.spec.x_in - x_out) / self.spec.x_in if self.spec.x_in > 0 else 0
        
        return {
            "specification": {
                "L_in": self.spec.L_in, "G_in": self.spec.G_in,
                "x_in": self.spec.x_in, "y_in": self.spec.y_in,
                "mode": self.spec.mode,
            },
            "results": result,
            "performance": {"recovery": recovery},
        }