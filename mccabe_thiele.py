"""McCabe-Thiele method for binary distillation - Eqs. 26.4-1 to 26.4-37"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Callable
import math
import numpy as np

from bank.core.validation import (
    check_positive, check_in_closed_01, ChemEngError, InputError
)
from bank.core.numerical import bisection, linear_interpolate
from bank.separations.distillation.relative_volatility import RelativeVolatility


def enriching_operating_line(R: float, xD: float) -> Tuple[float, float]:
    """
    Enriching section operating line - Eq. 26.4-8
    
    yn+1 = [R/(R+1)] xn + xD/(R+1)
    
    Returns:
        slope, intercept
    """
    check_positive("R", R)
    check_in_closed_01("xD", xD)
    
    slope = R / (R + 1.0)
    intercept = xD / (R + 1.0)
    return slope, intercept


def stripping_operating_line(Lm: float, Vm: float, xW: float, W: float) -> Tuple[float, float]:
    """
    Stripping section operating line - Eq. 26.4-11
    
    ym+1 = (Lm/Vm+1) xm - (W xW / Vm+1)
    
    Returns:
        slope, intercept
    """
    check_positive("Lm", Lm)
    check_positive("Vm", Vm)
    check_in_closed_01("xW", xW)
    check_positive("W", W)
    
    slope = Lm / Vm
    intercept = - (W * xW) / Vm
    return slope, intercept


def q_line(q: float, xF: float) -> Tuple[float, float]:
    """
    q-line equation - Eq. 26.4-19
    
    y = [q/(q-1)] x - xF/(q-1)
    
    Returns:
        slope, intercept (or vertical line indicator)
    """
    check_in_closed_01("xF", xF)
    
    if abs(q - 1.0) < 1e-12:
        # Vertical line at x = xF
        return float('inf'), xF
    
    slope = q / (q - 1.0)
    intercept = -xF / (q - 1.0)
    return slope, intercept


def intersect_lines(
    m1: float, b1: float, m2: float, b2: float
) -> Tuple[float, float]:
    """Find intersection of two lines"""
    if not math.isfinite(m1) and not math.isfinite(m2):
        raise ValueError("Both lines vertical")
    if not math.isfinite(m1):
        x = b1
        y = m2 * x + b2
        return x, y
    if not math.isfinite(m2):
        x = b2
        y = m1 * x + b1
        return x, y
    if abs(m1 - m2) < 1e-12:
        raise ValueError("Lines are parallel")
    
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y


class FenskeEquation:
    """
    Fenske equation for minimum stages at total reflux - Eq. 26.4-23
    
    Nm = log {[xD/(1-xD)] [(1-xW)/xW]} / log αav
    """
    
    @staticmethod
    def calculate(xD: float, xW: float, alpha_avg: float) -> float:
        check_in_closed_01("xD", xD)
        check_in_closed_01("xW", xW)
        check_positive("alpha_avg", alpha_avg)
        
        if xD <= 0 or xD >= 1 or xW <= 0 or xW >= 1:
            return float('inf')
        
        # Check for alpha = 1 (no separation possible)
        if abs(alpha_avg - 1.0) < 1e-10:
            return float('inf')
        
        numerator = math.log((xD / (1 - xD)) * ((1 - xW) / xW))
        denominator = math.log(alpha_avg)
        
        return numerator / denominator


class MinimumReflux:
    """
    Minimum reflux ratio calculations - Eq. 26.4-24
    
    Rm/(Rm+1) = (xD - y')/(xD - x')
    """
    
    @staticmethod
    def calculate(xD: float, x_prime: float, y_prime: float) -> float:
        check_in_closed_01("xD", xD)
        check_in_closed_01("x_prime", x_prime)
        check_in_closed_01("y_prime", y_prime)
        
        if abs(xD - x_prime) < 1e-12:
            return float('inf')
        
        slope = (xD - y_prime) / (xD - x_prime)
        if slope >= 1.0:
            return float('inf')
        
        return slope / (1 - slope)


@dataclass
class McCabeThieleSpec:
    """Specification for McCabe-Thiele distillation"""
    F: float           # Feed flow rate
    xF: float          # Feed composition
    xD: float          # Distillate composition
    xW: float          # Bottoms composition
    R: float           # Reflux ratio
    q: float           # Feed quality
    alpha: float       # Relative volatility (constant)
    tray_efficiency: Optional[float] = None
    max_stages: int = 100
    tol: float = 1e-12


class McCabeThieleSolver:
    """
    McCabe-Thiele method for binary distillation.
    
    Implements Eqs. 26.4-1 to 26.4-24.
    """
    
    def __init__(self, spec: McCabeThieleSpec):
        self.spec = spec
        self.rv = RelativeVolatility(spec.alpha)
        self._validate()
        self._calculate_flows()
        self._calculate_operating_lines()
    
    def _validate(self):
        check_positive("F", self.spec.F)
        check_in_closed_01("xF", self.spec.xF)
        check_in_closed_01("xD", self.spec.xD)
        check_in_closed_01("xW", self.spec.xW)
        check_positive("R", self.spec.R)
        
        if self.spec.xD <= self.spec.xW:
            raise ChemEngError("xD must be greater than xW")
    
    def _calculate_flows(self):
        """Calculate D and W from mass balances - Eqs. 26.4-3, 26.4-4"""
        denom = self.spec.xD - self.spec.xW
        if abs(denom) < 1e-12:
            raise ChemEngError("xD and xW too close")
        
        self.D = self.spec.F * (self.spec.xF - self.spec.xW) / denom
        self.W = self.spec.F - self.D
    
    def _calculate_operating_lines(self):
        """Calculate enriching, stripping, and q-lines"""
        # Enriching line - Eq. 26.4-8
        self.mR, self.bR = enriching_operating_line(self.spec.R, self.spec.xD)
        
        # q-line - Eq. 26.4-19
        self.mq, self.bq = q_line(self.spec.q, self.spec.xF)
        
        # Find intersection of enriching and q-lines
        self.x_int, self.y_int = intersect_lines(self.mR, self.bR, self.mq, self.bq)
        
        # Stripping line through (xW, xW) and intersection point
        if abs(self.x_int - self.spec.xW) < 1e-12:
            raise ChemEngError("Cannot form stripping line")
        
        self.mS = (self.y_int - self.spec.xW) / (self.x_int - self.spec.xW)
        self.bS = self.y_int - self.mS * self.x_int
    
    def equilibrium_y(self, x: float) -> float:
        """Equilibrium from relative volatility - Eq. 26.3-4"""
        return self.rv.y_from_x(x)
    
    def equilibrium_x(self, y: float) -> float:
        """Equilibrium from relative volatility - Eq. 26.3-4 rearranged"""
        return self.rv.x_from_y(y)
    
    def solve_stages(self) -> Dict[str, Any]:
        """Step off stages to find number of theoretical trays"""
        points = [(self.spec.xD, self.spec.xD)]
        x_history = []
        
        y_current = self.spec.xD
        stage = 1
        
        while stage <= self.spec.max_stages:
            # Find x in equilibrium with current y
            x_eq = self.equilibrium_x(y_current)
            points.append((x_eq, y_current))
            x_history.append(x_eq)
            
            # Use appropriate operating line
            if x_eq >= self.x_int:
                y_next = self.mR * x_eq + self.bR
            else:
                y_next = self.mS * x_eq + self.bS
            
            points.append((x_eq, y_next))
            
            # Check if we've reached bottoms
            if x_eq <= self.spec.xW + 1e-12:
                # Fractional stage
                if len(x_history) >= 2:
                    x_prev = x_history[-2]
                    x_curr = x_history[-1]
                    if abs(x_curr - x_prev) > 1e-12:
                        frac = (x_prev - self.spec.xW) / (x_prev - x_curr)
                        frac = max(0.0, min(1.0, frac))
                    else:
                        frac = 1.0
                else:
                    frac = 1.0
                
                N = (stage - 1) + frac
                break
            
            y_current = y_next
            stage += 1
        else:
            raise ChemEngError("Exceeded max stages")
        
        # Determine feed tray
        feed_tray = self._find_feed_tray(x_history)
        
        return {
            "N_theoretical": N,
            "N_ceiling": math.ceil(N - 1e-12),
            "points": points,
            "x_history": x_history,
            "feed_tray_from_top": feed_tray,
        }
    
    def _find_feed_tray(self, x_history: List[float]) -> int:
        """Determine feed tray location"""
        for i, x in enumerate(x_history, start=1):
            if x <= self.x_int + 1e-12:
                return i
        return len(x_history) + 1
    
    def minimum_reflux(self) -> float:
        """Calculate minimum reflux ratio - Eq. 26.4-24"""
        # Find intersection of q-line with equilibrium curve
        if math.isfinite(self.mq):
            def f(x):
                y_q = self.mq * x + self.bq
                y_eq = self.equilibrium_y(x)
                return y_q - y_eq
            
            x_prime = bisection(f, self.spec.xW, self.spec.xD, 
                               tol=self.spec.tol, expand_bracket=True)
            y_prime = self.equilibrium_y(x_prime)
        else:
            x_prime = self.spec.xF
            y_prime = self.equilibrium_y(x_prime)
        
        return MinimumReflux.calculate(self.spec.xD, x_prime, y_prime)
    
    def minimum_stages_total_reflux(self) -> float:
        """
        Minimum stages at total reflux - Eq. 26.4-23
        
        For constant relative volatility systems, use the alpha from the spec.
        This avoids division by zero issues.
        """
        alpha_avg = self.spec.alpha
        
        # Ensure alpha is not 1.0
        if abs(alpha_avg - 1.0) < 1e-10:
            # If alpha is 1.0, separation is impossible
            return float('inf')
        
        return FenskeEquation.calculate(self.spec.xD, self.spec.xW, alpha_avg)
    
    def solve(self) -> Dict[str, Any]:
        """Complete McCabe-Thiele analysis"""
        stages = self.solve_stages()
        
        return {
            "inputs": {
                "F": self.spec.F, "xF": self.spec.xF,
                "xD": self.spec.xD, "xW": self.spec.xW,
                "R": self.spec.R, "q": self.spec.q,
            },
            "flows": {"D": self.D, "W": self.W},
            "operating_lines": {
                "enriching": {"slope": self.mR, "intercept": self.bR},
                "stripping": {"slope": self.mS, "intercept": self.bS},
                "q_line": {"slope": self.mq, "intercept": self.bq},
            },
            "intersection": {"x": self.x_int, "y": self.y_int},
            "stages": stages,
            "minimum_reflux": self.minimum_reflux(),
            "minimum_stages_total_reflux": self.minimum_stages_total_reflux(),
        }