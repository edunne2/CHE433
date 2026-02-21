# bank/separations/distillation/shortcut.py
"""Shortcut distillation methods: Fenske, Underwood, Gilliland, Kirkbride"""
from dataclasses import dataclass
from typing import Dict, Any, Sequence, List, Optional, Tuple
import math

from bank.core.validation import (
    check_positive, check_in_closed_01, normalize_composition,
    check_composition_match, ChemEngError, InputError
)
from bank.core.numerical import bisection
from bank.separations.equilibrium import EquilibriumModel


def fenske_nmin(
    xD_LK: float,
    xB_LK: float,
    xD_HK: float,
    xB_HK: float,
    alpha_avg_LK_HK: float,
) -> float:
    """
    Fenske equation for minimum stages at total reflux.
    
    Nmin = ln[(xD_LK/xB_LK)*(xB_HK/xD_HK)] / ln(alpha_avg)
    """
    check_in_closed_01("xD_LK", xD_LK)
    check_in_closed_01("xB_LK", xB_LK)
    check_in_closed_01("xD_HK", xD_HK)
    check_in_closed_01("xB_HK", xB_HK)
    check_positive("alpha_avg_LK_HK", alpha_avg_LK_HK)
    
    arg = (xD_LK / xB_LK) * (xB_HK / xD_HK)
    if arg <= 0:
        raise ChemEngError(f"Fenske argument must be positive, got {arg}")
    
    return math.log(arg) / math.log(alpha_avg_LK_HK)


def underwood_theta(
    alpha: Sequence[float],
    zF: Sequence[float],
    q: float,
    theta_lo: float = 0.0,
    theta_hi: float = 10.0,
    maxiter: int = 200,
    tol: float = 1e-12,
) -> float:
    """
    Underwood equation root: 1 - q = sum(alpha_i * zF_i / (alpha_i - theta))
    
    Returns theta between alpha_HK and alpha_LK.
    """
    zF_norm = normalize_composition(zF, "zF")
    alpha_list = [float(a) for a in alpha]
    
    if len(alpha_list) != len(zF_norm):
        raise InputError("alpha and zF must have same length")
    
    for a in alpha_list:
        check_positive("alpha", a)
    
    def f(theta: float) -> float:
        s = 0.0
        for a, z in zip(alpha_list, zF_norm):
            denom = a - theta
            if abs(denom) < 1e-18:
                return float('inf')
            s += a * z / denom
        return (1.0 - q) - s
    
    # Find bracket between alpha_HK and alpha_LK
    # This assumes alpha are sorted and HK < LK
    alpha_sorted = sorted(alpha_list)
    
    # Try provided bracket first
    fa, fb = f(theta_lo), f(theta_hi)
    if fa * fb <= 0:
        return bisection(f, theta_lo, theta_hi, tol=tol, maxiter=maxiter)
    
    # Try between each adjacent alpha pair
    for i in range(len(alpha_sorted) - 1):
        a1, a2 = alpha_sorted[i], alpha_sorted[i+1]
        if f(a1 + 1e-6) * f(a2 - 1e-6) <= 0:
            return bisection(f, a1 + 1e-6, a2 - 1e-6, tol=tol, maxiter=maxiter)
    
    raise ChemEngError("Could not find theta root in Underwood equation")


def underwood_rmin(
    alpha: Sequence[float],
    xD: Sequence[float],
    theta: float,
) -> float:
    """
    Underwood equation for minimum reflux:
    Rmin = sum(alpha_i * xD_i / (alpha_i - theta)) - 1
    """
    xD_norm = normalize_composition(xD, "xD")
    alpha_list = [float(a) for a in alpha]
    
    if len(alpha_list) != len(xD_norm):
        raise InputError("alpha and xD must have same length")
    
    s = 0.0
    for a, x in zip(alpha_list, xD_norm):
        denom = a - theta
        if abs(denom) < 1e-18:
            return float('inf')
        s += a * x / denom
    
    Rmin = s - 1.0
    return max(0.0, Rmin)


def gilliland_eduljee(
    Nmin: float,
    R: float,
    Rmin: float,
) -> Dict[str, float]:
    """
    Gilliland correlation (Eduljee explicit form):
    X = (R - Rmin)/(R + 1)
    Y = 1 - exp[((1 + 54.4X)/(11 + 117.2X))*(X - 1)]
    N = (Nmin + Y)/(1 - Y)
    """
    check_positive("Nmin", Nmin)
    check_positive("R", R)
    check_positive("Rmin", Rmin)
    
    if R <= Rmin:
        raise ChemEngError(f"Require R > Rmin for Gilliland, got R={R}, Rmin={Rmin}")
    
    X = (R - Rmin) / (R + 1.0)
    X = max(0.0, min(1.0, X))
    
    Y = 1.0 - math.exp(((1.0 + 54.4 * X) / (11.0 + 117.2 * X)) * (X - 1.0))
    Y = max(0.0, min(0.999999, Y))
    
    N = (Nmin + Y) / (1.0 - Y)
    
    return {"X": X, "Y": Y, "N": N}


def kirkbride_feed_location(
    xF_LK: float,
    xF_HK: float,
    xD_LK: float,
    xB_HK: float,
    D: float,
    B: float,
) -> float:
    """
    Kirkbride equation for feed tray location:
    log10(Nb/Na) = 0.206 log10[(xF_LK/xF_HK)*(B/D)*(xB_HK/xD_LK)]
    """
    check_in_closed_01("xF_LK", xF_LK)
    check_in_closed_01("xF_HK", xF_HK)
    check_in_closed_01("xD_LK", xD_LK)
    check_in_closed_01("xB_HK", xB_HK)
    check_positive("D", D)
    check_positive("B", B)
    
    arg = (xF_LK / xF_HK) * (B / D) * (xB_HK / xD_LK)
    if arg <= 0:
        raise ChemEngError(f"Kirkbride argument must be positive, got {arg}")
    
    return 10.0 ** (0.206 * math.log10(arg))


@dataclass
class ShortcutSpec:
    """Specification for shortcut distillation design"""
    zF: Sequence[float]      # Feed composition
    xD: Sequence[float]      # Distillate composition
    xB: Sequence[float]      # Bottoms composition
    LK_index: int            # Light key index
    HK_index: int            # Heavy key index
    q: float                 # Feed quality
    R: float                 # Operating reflux ratio
    
    # Either provide relative volatilities or K-values
    alpha_rel_HK: Optional[Sequence[float]] = None
    K_top: Optional[Sequence[float]] = None
    K_bottom: Optional[Sequence[float]] = None
    
    # Optional for Kirkbride
    D: Optional[float] = None
    B: Optional[float] = None
    
    # Numerical parameters
    theta_lo: float = 0.0
    theta_hi: float = 10.0
    tol: float = 1e-12
    maxiter: int = 400


class ShortcutSolver:
    """Solver for shortcut distillation design"""
    
    def __init__(self, spec: ShortcutSpec):
        self.spec = spec
        self._validate_and_normalize()
        self._calculate_alpha()
        
    def _validate_and_normalize(self):
        """Validate and normalize compositions"""
        self.zF = normalize_composition(self.spec.zF, "zF")
        self.xD = normalize_composition(self.spec.xD, "xD")
        self.xB = normalize_composition(self.spec.xB, "xB")
        
        n = len(self.zF)
        check_composition_match({"zF": self.zF, "xD": self.xD, "xB": self.xB})
        
        if not (0 <= self.spec.LK_index < n):
            raise InputError(f"LK_index {self.spec.LK_index} out of range [0, {n-1}]")
        if not (0 <= self.spec.HK_index < n):
            raise InputError(f"HK_index {self.spec.HK_index} out of range [0, {n-1}]")
        if self.spec.LK_index == self.spec.HK_index:
            raise InputError("LK and HK must be different")
        
        check_positive("q", self.spec.q)  # q can be >1 for subcooled feed
        check_positive("R", self.spec.R)
    
    def _calculate_alpha(self):
        """Calculate average relative volatilities"""
        n = len(self.zF)
        
        if self.spec.alpha_rel_HK is not None:
            alpha = [float(a) for a in self.spec.alpha_rel_HK]
            if len(alpha) != n:
                raise InputError("alpha_rel_HK length mismatch")
            
            # Normalize so HK = 1
            alpha_HK = alpha[self.spec.HK_index]
            if abs(alpha_HK - 1.0) > 1e-6:
                alpha = [a / alpha_HK for a in alpha]
            
            self.alpha = alpha
            
        elif self.spec.K_top is not None and self.spec.K_bottom is not None:
            Kt = [float(v) for v in self.spec.K_top]
            Kb = [float(v) for v in self.spec.K_bottom]
            
            if len(Kt) != n or len(Kb) != n:
                raise InputError("K_top and K_bottom length mismatch")
            
            # Calculate alphas relative to HK
            alpha_top = [Kt[i] / Kt[self.spec.HK_index] for i in range(n)]
            alpha_bot = [Kb[i] / Kb[self.spec.HK_index] for i in range(n)]
            
            # Geometric average
            self.alpha = [math.sqrt(at * ab) for at, ab in zip(alpha_top, alpha_bot)]
        else:
            raise InputError("Must provide either alpha_rel_HK or both K_top and K_bottom")
        
        self.alpha_LK = self.alpha[self.spec.LK_index]
    
    def solve(self) -> Dict[str, Any]:
        """Perform shortcut design calculations"""
        
        # Fenske
        xD_LK = self.xD[self.spec.LK_index]
        xB_LK = self.xB[self.spec.LK_index]
        xD_HK = self.xD[self.spec.HK_index]
        xB_HK = self.xB[self.spec.HK_index]
        alpha_avg_LK_HK = self.alpha_LK / self.alpha_HK
        
        Nmin = fenske_nmin(xD_LK, xB_LK, xD_HK, xB_HK, alpha_avg_LK_HK)
        
        # Underwood
        theta = underwood_theta(
            self.alpha, self.zF, self.spec.q,
            self.spec.theta_lo, self.spec.theta_hi,
            self.spec.maxiter, self.spec.tol
        )
        
        Rmin = underwood_rmin(self.alpha, self.xD, theta)
        
        # Gilliland
        gill = gilliland_eduljee(Nmin, self.spec.R, Rmin)
        
        # Kirkbride
        D = 1.0 if self.spec.D is None else float(self.spec.D)
        B = 1.0 if self.spec.B is None else float(self.spec.B)
        
        Nb_Na = kirkbride_feed_location(
            xF_LK=self.zF[self.spec.LK_index],
            xF_HK=self.zF[self.spec.HK_index],
            xD_LK=xD_LK,
            xB_HK=xB_HK,
            D=D,
            B=B,
        )
        
        # Calculate tray distribution
        N_total = gill["N"]
        N_rect = N_total / (1.0 + Nb_Na)
        N_strip = N_total - N_rect
        
        if N_rect < 0 or N_strip < 0:
            raise ChemEngError("Invalid tray distribution - check Kirkbride result")
        
        return {
            "fenske": {
                "Nmin": Nmin,
                "alpha_LK_HK": alpha_avg_LK_HK,
            },
            "underwood": {
                "theta": theta,
                "Rmin": Rmin,
            },
            "gilliland": gill,
            "kirkbride": {
                "Nb_over_Na": Nb_Na,
                "N_rectifying": N_rect,
                "N_stripping": N_strip,
                "feed_tray_from_top": round(N_rect) + 1,
            },
            "summary": {
                "N_total_theoretical": N_total,
                "R_operating": self.spec.R,
                "R_over_Rmin": self.spec.R / Rmin if Rmin > 0 else None,
            }
        }
# At the bottom of shortcut.py
def shortcut_design(spec):
    """Backward compatibility function for shortcut design"""
    solver = ShortcutSolver(spec)
    return solver.solve()    