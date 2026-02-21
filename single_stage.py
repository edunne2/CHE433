# bank/separations/absorption/single_stage.py
"""Single-stage equilibrium contact for absorption/stripping"""
from dataclasses import dataclass
from typing import Dict, Any, Optional

from bank.core.validation import (
    check_positive, check_in_closed_01, ChemEngError, InputError
)
from bank.separations.equilibrium import EquilibriumModel, HenryLaw


@dataclass
class SingleStageSpec:
    """
    Specification for single-stage equilibrium contact.
    
    For absorption: Liquid absorbs solute from gas
    For stripping: Gas strips solute from liquid
    """
    # Stream flows
    L_in: float          # Liquid inlet flow rate
    G_in: float          # Gas inlet flow rate
    
    # Compositions (solute mole fraction)
    x_in: float          # Liquid inlet composition
    y_in: float          # Gas inlet composition
    
    # Equilibrium
    eq: EquilibriumModel # Equilibrium model (usually Henry's law)
    
    # Optional: if outlet specified
    x_out: Optional[float] = None  # Liquid outlet composition
    y_out: Optional[float] = None  # Gas outlet composition
    
    # Operating mode
    mode: str = "absorption"  # "absorption" or "stripping"
    
    tol: float = 1e-12
    maxiter: int = 400


class SingleStageAbsorber:
    """
    Solver for single-stage equilibrium contact.
    
    Can solve for:
    - Outlet compositions given flows
    - Flow ratio required for given separation
    - Number of stages for given separation
    """
    
    def __init__(self, spec: SingleStageSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        """Validate inputs"""
        check_positive("L_in", self.spec.L_in)
        check_positive("G_in", self.spec.G_in)
        check_in_closed_01("x_in", self.spec.x_in)
        check_in_closed_01("y_in", self.spec.y_in)
        
        if self.spec.mode not in ["absorption", "stripping"]:
            raise InputError(f"Mode must be 'absorption' or 'stripping', got {self.spec.mode}")
        
        # Check if we have enough info
        n_specified = sum([
            self.spec.x_out is not None,
            self.spec.y_out is not None
        ])
        
        if n_specified == 0:
            # Will solve for both outlets
            pass
        elif n_specified == 1:
            # Will solve for the other outlet
            pass
        else:
            # Both specified - verify they're consistent
            self._verify_mass_balance()
    
    def _verify_mass_balance(self):
        """Verify specified outlets satisfy mass balance"""
        if self.spec.x_out is None or self.spec.y_out is None:
            return
        
        L_in = self.spec.L_in
        G_in = self.spec.G_in
        x_in = self.spec.x_in
        y_in = self.spec.y_in
        x_out = self.spec.x_out
        y_out = self.spec.y_out
        
        # Solute mass balance: L_in*x_in + G_in*y_in = L_out*x_out + G_out*y_out
        # Assuming constant liquid and gas rates (dilute)
        L_out = L_in
        G_out = G_in
        
        lhs = L_in * x_in + G_in * y_in
        rhs = L_out * x_out + G_out * y_out
        
        if abs(lhs - rhs) > 1e-6 * max(lhs, rhs):
            raise ChemEngError(
                f"Specified outlets don't satisfy mass balance:\n"
                f"L_in*x_in + G_in*y_in = {lhs:.4f}\n"
                f"L_out*x_out + G_out*y_out = {rhs:.4f}"
            )
    
    def _equilibrium_relation(self, x: float) -> float:
        """Equilibrium: y = f(x)"""
        return self.spec.eq.y_of_x(x)
    
    def _operating_line(self, x: float) -> float:
        """
        Operating line from mass balance:
        L(x_in - x) = G(y - y_in)
        y = y_in + (L/G)(x_in - x)
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
        """Solve for outlet compositions"""
        from bank.core.numerical import bisection
        
        # Find x that satisfies equilibrium
        x_low = 0.0
        x_high = self.spec.x_in if self.spec.mode == "absorption" else 1.0
        
        try:
            x_out = bisection(
                self._stage_residual, x_low, x_high,
                tol=self.spec.tol, maxiter=self.spec.maxiter
            )
        except:
            # Try expanding bracket
            x_out = bisection(
                self._stage_residual, x_low, x_high,
                expand_bracket=True, tol=self.spec.tol, maxiter=self.spec.maxiter
            )
        
        y_out = self._equilibrium_relation(x_out)
        
        # Verify with operating line
        y_op = self._operating_line(x_out)
        
        return {
            "outlets": {
                "x_out": x_out,
                "y_out": y_out,
            },
            "flows": {
                "L_out": self.spec.L_in,
                "G_out": self.spec.G_in,
            },
            "verification": {
                "equilibrium_minus_operating": y_out - y_op,
                "solute_balance": self._check_balance(x_out, y_out),
            }
        }
    
    def _check_balance(self, x_out: float, y_out: float) -> float:
        """Check solute mass balance"""
        L = self.spec.L_in
        G = self.spec.G_in
        solute_in = L * self.spec.x_in + G * self.spec.y_in
        solute_out = L * x_out + G * y_out
        return solute_in - solute_out
    
    def solve_LG_ratio(self, target_x_out: float) -> float:
        """
        Solve for L/G ratio required to achieve target liquid outlet.
        
        Args:
            target_x_out: Desired liquid outlet composition
        
        Returns:
            Required L/G ratio
        """
        check_in_closed_01("target_x_out", target_x_out)
        
        def residual(LG_ratio: float):
            # Operating line with this L/G
            y_op = self.spec.y_in + LG_ratio * (self.spec.x_in - target_x_out)
            y_eq = self._equilibrium_relation(target_x_out)
            return y_eq - y_op
        
        from bank.core.numerical import bisection
        
        # Find L/G ratio
        LG_min = 0.0
        LG_max = 10.0  # Upper bound
        
        return bisection(residual, LG_min, LG_max, tol=self.spec.tol)
    
    def solve(self) -> Dict[str, Any]:
        """Main solving method"""
        result = self.solve_outlets()
        
        # Calculate recovery and other metrics
        solute_in = self.spec.L_in * self.spec.x_in + self.spec.G_in * self.spec.y_in
        solute_out_liq = self.spec.L_in * result["outlets"]["x_out"]
        solute_out_gas = self.spec.G_in * result["outlets"]["y_out"]
        
        if self.spec.mode == "absorption":
            # Solute removed from gas
            solute_removed = self.spec.G_in * (self.spec.y_in - result["outlets"]["y_out"])
            recovery = solute_removed / (self.spec.G_in * self.spec.y_in) if self.spec.y_in > 0 else 0
        else:  # stripping
            # Solute removed from liquid
            solute_removed = self.spec.L_in * (self.spec.x_in - result["outlets"]["x_out"])
            recovery = solute_removed / (self.spec.L_in * self.spec.x_in) if self.spec.x_in > 0 else 0
        
        return {
            "specification": self.spec.to_dict(),
            "results": result,
            "performance": {
                "solute_recovery": recovery,
                "solute_removed": solute_removed,
                "LG_ratio": self.spec.L_in / self.spec.G_in,
            }
        }