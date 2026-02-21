"""Shortcut distillation methods - Kremser equations (22.1-24 to 22.1-29)"""

from dataclasses import dataclass
from typing import Dict, Any, Sequence, List, Optional
import math

from bank.core.validation import (
    check_positive, check_in_closed_01, normalize_composition,
    check_composition_match, ChemEngError, InputError
)
from bank.core.numerical import bisection


def kremser_N_absorption(
    y_in: float,
    y_out: float,
    x_in: float,
    m: float,
    A: float
) -> float:
    """
    Kremser equation for absorption - Eq. 22.1-28
    
    N = ln[ ((y_in - m*x_in)/(y_out - m*x_in)) (1 - 1/A) + 1/A ] / ln A
    """
    check_in_closed_01("y_in", y_in)
    check_in_closed_01("y_out", y_out)
    check_in_closed_01("x_in", x_in)
    check_positive("m", m)
    check_positive("A", A)
    
    if abs(A - 1.0) < 1e-10:
        return (y_in - y_out) / (y_out - m * x_in)
    
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
    Kremser equation for stripping - Eq. 22.1-25
    
    N = ln[ ((x_in - y_in/m)/(x_out - y_in/m)) (1 - A) + A ] / ln(1/A)
    """
    check_in_closed_01("x_in", x_in)
    check_in_closed_01("x_out", x_out)
    check_in_closed_01("y_in", y_in)
    check_positive("m", m)
    check_positive("A", A)
    
    if abs(A - 1.0) < 1e-10:
        return (x_in - x_out) / (x_out - y_in/m)
    
    numerator = ((x_in - y_in/m) / (x_out - y_in/m)) * (1 - A) + A
    
    if numerator <= 0:
        return float('inf')
    
    return math.log(numerator) / math.log(1/A)


@dataclass
class ShortcutSpec:
    """Specification for shortcut distillation design"""
    zF: Sequence[float]
    xD: Sequence[float]
    xB: Sequence[float]
    LK_index: int
    HK_index: int
    q: float
    R: float
    alpha_rel_HK: Optional[Sequence[float]] = None
    K_top: Optional[Sequence[float]] = None
    K_bottom: Optional[Sequence[float]] = None
    theta_lo: float = 0.0
    theta_hi: float = 10.0
    tol: float = 1e-12
    maxiter: int = 400


class ShortcutSolver:
    """Solver for shortcut distillation design using Kremser equations"""
    
    def __init__(self, spec: ShortcutSpec):
        self.spec = spec
        self._validate_and_normalize()
        self._calculate_alpha()
    
    def _validate_and_normalize(self):
        self.zF = normalize_composition(self.spec.zF, "zF")
        self.xD = normalize_composition(self.spec.xD, "xD")
        self.xB = normalize_composition(self.spec.xB, "xB")
        
        n = len(self.zF)
        check_composition_match({"zF": self.zF, "xD": self.xD, "xB": self.xB})
        
        if not (0 <= self.spec.LK_index < n):
            raise InputError(f"LK_index {self.spec.LK_index} out of range")
        if not (0 <= self.spec.HK_index < n):
            raise InputError(f"HK_index {self.spec.HK_index} out of range")
        if self.spec.LK_index == self.spec.HK_index:
            raise InputError("LK and HK must be different")
        
        check_positive("q", self.spec.q)
        check_positive("R", self.spec.R)
    
    def _calculate_alpha(self):
        n = len(self.zF)
        
        if self.spec.alpha_rel_HK is not None:
            alpha = [float(a) for a in self.spec.alpha_rel_HK]
            if len(alpha) != n:
                raise InputError("alpha_rel_HK length mismatch")
            
            alpha_HK = alpha[self.spec.HK_index]
            if abs(alpha_HK - 1.0) > 1e-6:
                alpha = [a / alpha_HK for a in alpha]
            
            self.alpha = alpha
            
        elif self.spec.K_top is not None and self.spec.K_bottom is not None:
            Kt = [float(v) for v in self.spec.K_top]
            Kb = [float(v) for v in self.spec.K_bottom]
            
            if len(Kt) != n or len(Kb) != n:
                raise InputError("K_top and K_bottom length mismatch")
            
            alpha_top = [Kt[i] / Kt[self.spec.HK_index] for i in range(n)]
            alpha_bot = [Kb[i] / Kb[self.spec.HK_index] for i in range(n)]
            self.alpha = [math.sqrt(at * ab) for at, ab in zip(alpha_top, alpha_bot)]
        else:
            raise InputError("Must provide either alpha_rel_HK or both K_top and K_bottom")
        
        self.alpha_LK = self.alpha[self.spec.LK_index]
        self.alpha_HK = 1.0
    
    def solve(self) -> Dict[str, Any]:
        """Perform shortcut design calculations using Kremser equations"""
        
        xD_LK = self.xD[self.spec.LK_index]
        xB_LK = self.xB[self.spec.LK_index]
        xD_HK = self.xD[self.spec.HK_index]
        xB_HK = self.xB[self.spec.HK_index]
        alpha_avg_LK_HK = self.alpha_LK / self.alpha_HK
        
        # Use Kremser equations for absorption/stripping as appropriate
        if self.spec.q >= 1:  # Absorption case
            N = kremser_N_absorption(
                y_in=xD_LK, y_out=xB_LK, x_in=0, m=alpha_avg_LK_HK, A=self.spec.R
            )
        else:  # Stripping case
            N = kremser_N_stripping(
                x_in=xB_LK, x_out=xD_LK, y_in=0, m=alpha_avg_LK_HK, A=self.spec.R
            )
        
        return {
            "fenske": {"Nmin": N},
            "alpha_LK_HK": alpha_avg_LK_HK,
            "summary": {"N_total_theoretical": N},
        } 