# bank/separations/distillation/flash.py
"""Flash and differential distillation calculations"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable
import math

from bank.core.validation import (
    check_positive, check_in_closed_01, check_in_open_01,
    ChemEngError, InputError
)
from bank.core.numerical import bisection, integrate_trapezoid
from bank.separations.equilibrium import EquilibriumModel, TabulatedEquilibrium


# ============================================================================
# Single-Stage Flash
# ============================================================================

@dataclass
class FlashSpec:
    """Specification for single-stage equilibrium flash"""
    F: float                    # Feed flow rate
    z: float                    # Feed composition
    V: float                    # Vapor flow rate
    eq: EquilibriumModel        # Equilibrium model
    
    # Optional: if V not given, provide f
    f: Optional[float] = None   # Vapor fraction
    
    tol: float = 1e-12
    maxiter: int = 400


class FlashSolver:
    """Solver for single-stage equilibrium flash"""
    
    def __init__(self, spec: FlashSpec):
        self.spec = spec
        self._validate()
        
    def _validate(self):
        """Validate inputs"""
        check_positive("F", self.spec.F)
        check_in_closed_01("z", self.spec.z)
        
        if self.spec.V is not None and self.spec.f is not None:
            raise InputError("Provide either V or f, not both")
        
        if self.spec.V is not None:
            check_positive("V", self.spec.V)
            if self.spec.V >= self.spec.F:
                raise ChemEngError(f"Require V < F, got V={self.spec.V}, F={self.spec.F}")
            self.f = self.spec.V / self.spec.F
        elif self.spec.f is not None:
            check_in_open_01("f", self.spec.f)
            self.f = self.spec.f
            self.V = self.f * self.spec.F
        else:
            raise InputError("Must provide either V or f")
        
        self.L = self.spec.F - self.V
    
    def _flash_equation(self, x: float) -> float:
        """Flash equation: (1-f)*x + f*y(x) - z = 0"""
        return (1.0 - self.f) * x + self.f * self.spec.eq.y_of_x(x) - self.spec.z
    
    def solve(self) -> Dict[str, Any]:
        """Solve flash equation for x"""
        
        # Check endpoints
        f0 = self._flash_equation(0.0)
        f1 = self._flash_equation(1.0)
        
        if abs(f0) <= self.spec.tol:
            x = 0.0
        elif abs(f1) <= self.spec.tol:
            x = 1.0
        else:
            if f0 * f1 > 0:
                # Try to find if there's a solution by checking intermediate points
                x_test = 0.5
                if self._flash_equation(x_test) * f0 < 0:
                    x = bisection(
                        self._flash_equation, 0.0, x_test,
                        tol=self.spec.tol, maxiter=self.spec.maxiter
                    )
                elif self._flash_equation(x_test) * f1 < 0:
                    x = bisection(
                        self._flash_equation, x_test, 1.0,
                        tol=self.spec.tol, maxiter=self.spec.maxiter
                    )
                else:
                    raise ChemEngError("No solution found in [0,1] - check feed conditions")
            else:
                x = bisection(
                    self._flash_equation, 0.0, 1.0,
                    tol=self.spec.tol, maxiter=self.spec.maxiter
                )
        
        y = self.spec.eq.y_of_x(x)
        
        return {
            "inputs": {
                "F": self.spec.F,
                "z": self.spec.z,
                "V": self.V,
                "L": self.L,
                "f": self.f,
            },
            "outputs": {
                "x": x,
                "y": y,
            },
            "verification": {
                "material_balance": abs(self.spec.F * self.spec.z - self.V * y - self.L * x),
                "flash_residual": self._flash_equation(x),
            }
        }


# ============================================================================
# Rayleigh (Differential) Distillation
# ============================================================================

@dataclass
class RayleighSpec:
    """Specification for Rayleigh differential distillation"""
    F0: float                   # Initial liquid moles
    x0: float                   # Initial composition
    W: float                    # Final liquid moles
    eq: EquilibriumModel        # Equilibrium model
    
    # Optional: if W not given, provide xW
    xW: Optional[float] = None  # Final composition
    
    n_int: int = 6000           # Integration points
    tol: float = 1e-10
    maxiter: int = 400


class RayleighSolver:
    """Solver for Rayleigh differential distillation"""
    
    def __init__(self, spec: RayleighSpec):
        self.spec = spec
        self._validate()
        
    def _validate(self):
        """Validate inputs"""
        check_positive("F0", self.spec.F0)
        check_in_closed_01("x0", self.spec.x0)
        
        if self.spec.W is not None and self.spec.xW is not None:
            raise InputError("Provide either W or xW, not both")
        
        if self.spec.W is not None:
            check_positive("W", self.spec.W)
            if self.spec.W >= self.spec.F0:
                raise ChemEngError(f"Require W < F0, got W={self.spec.W}, F0={self.spec.F0}")
            self.D = self.spec.F0 - self.spec.W
            self.target_ln = math.log(self.spec.F0 / self.spec.W)
            self.solve_for = "xW"
        elif self.spec.xW is not None:
            check_in_closed_01("xW", self.spec.xW)
            if self.spec.xW >= self.spec.x0:
                raise ChemEngError(f"Require xW < x0, got xW={self.spec.xW}, x0={self.spec.x0}")
            self.solve_for = "W"
        else:
            raise InputError("Must provide either W or xW")
    
    def _integrand(self, x: float) -> float:
        """1/(y(x) - x)"""
        y = self.spec.eq.y_of_x(x)
        denom = y - x
        if denom <= 1e-14:
            raise ChemEngError(f"Encountered y(x) - x <= 0 at x={x}")
        return 1.0 / denom
    
    def _rayleigh_integral(self, xW: float) -> float:
        """Compute ∫ dx/(y-x) from xW to x0"""
        if xW >= self.spec.x0:
            return 0.0
        
        return integrate_trapezoid(
            self._integrand, xW, self.spec.x0,
            n=self.spec.n_int
        )
    
    def _residual_W(self, W: float) -> float:
        """Residual for solving W given xW"""
        if self.spec.xW is None:
            raise ChemEngError("xW must be specified when solving for W")
        target = math.log(self.spec.F0 / W)
        return self._rayleigh_integral(self.spec.xW) - target
    
    def _residual_W(self, W: float) -> float:
        """Residual for solving W given xW"""
        target = math.log(self.spec.F0 / W)
        return self._rayleigh_integral(self.spec.xW) - target
    
    def solve(self) -> Dict[str, Any]:
        """Solve Rayleigh distillation problem"""
        
        if self.solve_for == "xW":
            # Solve for xW given W
            a = 1e-12
            b = self.spec.x0 - 1e-12
            
            # Try to bracket
            fa = self._residual_xW(a)
            fb = self._residual_xW(b)
            
            if fa * fb > 0:
                # Adjust lower bound
                a2 = max(1e-6, 0.05 * self.spec.x0)
                fa2 = self._residual_xW(a2)
                if fa2 * fb <= 0:
                    a = a2
                else:
                    raise ChemEngError("Could not bracket xW solution")
            
            xW = bisection(
                self._residual_xW, a, b,
                tol=self.spec.tol, maxiter=self.spec.maxiter
            )
            
            # Calculate average distillate composition
            y_avg = (self.spec.F0 * self.spec.x0 - self.spec.W * xW) / self.D
            
            return {
                "inputs": {
                    "F0": self.spec.F0,
                    "x0": self.spec.x0,
                    "W": self.spec.W,
                    "D": self.D,
                },
                "outputs": {
                    "xW": xW,
                    "y_avg_distillate": y_avg,
                },
                "rayleigh_integral": self._rayleigh_integral(xW),
                "target_ln": self.target_ln,
            }
            
        else:  # solve_for == "W"
            # Solve for W given xW
            W_min = 1e-12
            W_max = self.spec.F0 - 1e-12
            
            # Check endpoints
            f_min = self._residual_W(W_min)
            f_max = self._residual_W(W_max)
            
            if f_min * f_max > 0:
                raise ChemEngError("Could not bracket W solution")
            
            W = bisection(
                self._residual_W, W_min, W_max,
                tol=self.spec.tol, maxiter=self.spec.maxiter
            )
            
            D = self.spec.F0 - W
            y_avg = (self.spec.F0 * self.spec.x0 - W * self.spec.xW) / D
            
            return {
                "inputs": {
                    "F0": self.spec.F0,
                    "x0": self.spec.x0,
                    "xW": self.spec.xW,
                },
                "outputs": {
                    "W": W,
                    "D": D,
                    "y_avg_distillate": y_avg,
                },
                "rayleigh_integral": self._rayleigh_integral(self.spec.xW),
                "target_ln": math.log(self.spec.F0 / W),
            }
# Backward compatibility functions
def solve_flash(spec):
    """Backward compatibility function for single-stage flash"""
    solver = FlashSolver(spec)
    return solver.solve()

def solve_rayleigh(spec):
    """Backward compatibility function for Rayleigh distillation"""
    solver = RayleighSolver(spec)
    return solver.solve()        