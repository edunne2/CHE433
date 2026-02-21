# bank/separations/distillation/mccabe_thiele.py
"""McCabe-Thiele method for binary distillation"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import math

from bank.core.validation import (
    check_positive, check_in_closed_01, check_in_open_01,
    ChemEngError, InputError
)
from bank.core.numerical import bisection
from bank.separations.equilibrium import EquilibriumModel, BinaryConstantAlpha


@dataclass
class McCabeThieleSpec:
    """Specification for McCabe-Thiele binary distillation"""
    eq_model: EquilibriumModel
    x_D: float
    x_B: float
    x_F: float
    R: Optional[float] = None
    q: float = 1.0
    rect_slope: Optional[float] = None
    rect_intercept: Optional[float] = None
    strip_slope: Optional[float] = None
    strip_intercept: Optional[float] = None
    max_stages: int = 400
    tol: float = 1e-12
    
    def __post_init__(self):
        check_in_open_01("x_D", self.x_D)
        check_in_open_01("x_B", self.x_B)
        check_in_open_01("x_F", self.x_F)
        
        if self.x_B >= self.x_D:
            raise ChemEngError(f"Require x_B < x_D, got {self.x_B} >= {self.x_D}")
        
        if self.R is None and (self.rect_slope is None or self.rect_intercept is None):
            raise InputError("Must provide either R or both rect_slope and rect_intercept")
        
        if self.R is not None:
            check_positive("R", self.R)


class McCabeThieleSolver:
    """Solver for McCabe-Thiele binary distillation"""
    
    def __init__(self, spec: McCabeThieleSpec):
        self.spec = spec
        self._solve_operating_lines()
    
    def _solve_operating_lines(self):
        """Determine rectifying and stripping operating lines"""
        spec = self.spec
        
        # Rectifying line
        if spec.rect_slope is not None and spec.rect_intercept is not None:
            self.mR = spec.rect_slope
            self.bR = spec.rect_intercept
        else:
            self.mR = spec.R / (spec.R + 1.0)
            self.bR = spec.x_D / (spec.R + 1.0)
        
        # q-line
        if abs(spec.q - 1.0) < 1e-12:
            self.mq = float('inf')
            self.bq = spec.x_F
        else:
            self.mq = spec.q / (spec.q - 1.0)
            self.bq = -spec.x_F / (spec.q - 1.0)
        
        # Intersection
        self._find_intersection()
        
        # Stripping line
        if spec.strip_slope is not None and spec.strip_intercept is not None:
            self.mS = spec.strip_slope
            self.bS = spec.strip_intercept
        else:
            if abs(self.x_int - spec.x_B) < 1e-12:
                raise ChemEngError("Cannot form stripping line: x_int equals x_B")
            self.mS = (self.y_int - spec.x_B) / (self.x_int - spec.x_B)
            self.bS = self.y_int - self.mS * self.x_int
    
    def _find_intersection(self):
        if math.isfinite(self.mR) and math.isfinite(self.mq):
            # Both lines have finite slopes
            if abs(self.mR - self.mq) < 1e-12:
                raise ChemEngError("Rectifying line and q-line are parallel")
            self.x_int = (self.bq - self.bR) / (self.mR - self.mq)
            self.y_int = self.mR * self.x_int + self.bR
        elif not math.isfinite(self.mq):
            # q-line is vertical at x = x_F
            self.x_int = self.bq  # This is x_F
            self.y_int = self.mR * self.x_int + self.bR  # CORRECT: uses rectifying line
        else:
            # rectifying line is vertical (unlikely)
            self.x_int = self.bR
            self.y_int = self.mq * self.x_int + self.bq
        
        self.x_int = max(0.0, min(1.0, self.x_int))
        self.y_int = max(0.0, min(1.0, self.y_int))
    
    def solve_stages(self) -> Dict[str, Any]:
        """Perform stage-by-stage calculation"""
        spec = self.spec
        points = [(spec.x_D, spec.x_D)]
        y_current = spec.x_D
        x_history = []
        stage_count = 0
        
        for n in range(1, spec.max_stages + 1):
            x_eq = spec.eq_model.x_of_y(y_current)
            points.append((x_eq, y_current))
            x_history.append(x_eq)
            
            if x_eq >= self.x_int:
                y_next = self.mR * x_eq + self.bR
            else:
                y_next = self.mS * x_eq + self.bS
            
            y_next = max(0.0, min(1.0, y_next))
            points.append((x_eq, y_next))
            
            if x_eq <= spec.x_B + 1e-12:
                if len(x_history) < 2:
                    frac = 1.0
                else:
                    x_prev, x_now = x_history[-2], x_history[-1]
                    if abs(x_prev - x_now) < 1e-12:
                        frac = 1.0
                    else:
                        frac = (x_prev - spec.x_B) / (x_prev - x_now)
                        frac = max(0.0, min(1.0, frac))
                
                stage_count = (n - 1) + frac
                break
            
            y_current = y_next
            stage_count = n
        
        # Find feed tray
        feed_tray = self._find_feed_tray(x_history)
        
        return {
            "N_theoretical": stage_count,
            "N_ceiling": math.ceil(stage_count - 1e-12),
            "feed_tray_from_top": feed_tray,
            "points": points,
            "x_history": x_history,
        }
    
    def _find_feed_tray(self, x_history: List[float]) -> int:
        """Determine feed tray location"""
        for i, x in enumerate(x_history, start=1):
            if x <= self.x_int + 1e-12:
                return i
        return len(x_history) + 1
    
    def solve_minimum_reflux(self) -> float:
        """Calculate minimum reflux ratio for binary systems"""
        spec = self.spec
        
        if math.isfinite(self.mq):
            def f(x):
                y_q = self.mq * x + self.bq
                y_eq = spec.eq_model.y_of_x(x)
                return y_q - y_eq
            
            x_int_eq = bisection(f, spec.x_B, spec.x_D, tol=spec.tol)
            y_int_eq = spec.eq_model.y_of_x(x_int_eq)
        else:
            x_int_eq = spec.x_F
            y_int_eq = spec.eq_model.y_of_x(x_int_eq)
        
        Rmin = (spec.x_D - y_int_eq) / (y_int_eq - x_int_eq)
        return max(0.0, Rmin)
    
    def solve(self) -> Dict[str, Any]:
        """Complete McCabe-Thiele design"""
        stages = self.solve_stages()
        
        rmin_info = {}
        if self.spec.R is not None:
            try:
                rmin = self.solve_minimum_reflux()
                rmin_info = {
                    "Rmin": rmin,
                    "R_over_Rmin": self.spec.R / rmin if rmin > 0 else None
                }
            except:
                rmin_info = {"Rmin": None, "R_over_Rmin": None}
        
        return {
            "operating_lines": {
                "rectifying": {"slope": self.mR, "intercept": self.bR},
                "q_line": {
                    "slope": self.mq if math.isfinite(self.mq) else "vertical",
                    "intercept": self.bq
                },
                "stripping": {"slope": self.mS, "intercept": self.bS},
            },
            "intersection": {"x": self.x_int, "y": self.y_int},
            "stages": stages,
            "minimum_reflux": rmin_info,
            "specification": self.spec.to_dict(),
        }


def mccabe_thiele_binary(
    alpha: float,
    x_D: float,
    x_B: float,
    x_F: float,
    R: float,
    q: float = 1.0,
) -> Dict[str, Any]:
    """Quick McCabe-Thiele calculation for constant alpha"""
    eq = BinaryConstantAlpha(alpha)
    spec = McCabeThieleSpec(
        eq_model=eq,
        x_D=x_D,
        x_B=x_B,
        x_F=x_F,
        R=R,
        q=q
    )
    solver = McCabeThieleSolver(spec)
    return solver.solve()

# At the bottom of mccabe_thiele.py
__all__ = [
    'McCabeThieleSolver',
    'McCabeThieleSpec',
    'mccabe_thiele_binary',
]