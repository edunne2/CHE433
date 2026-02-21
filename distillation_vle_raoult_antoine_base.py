# bank/distillation_vle_raoult_antoine_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Sequence, List, Optional, Callable, Tuple

import math


# -----------------------------
# Core validation / utilities
# -----------------------------
def _vp(name: str, v: float) -> None:
    if v <= 0:
        raise ValueError(f"{name} must be > 0.")


def _vf01(name: str, v: float) -> None:
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"{name} must be in [0,1].")


def _vf01_open(name: str, v: float) -> None:
    if not (0.0 <= v < 1.0):
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1.")


def normalize(z: Sequence[float], name: str = "composition") -> List[float]:
    if len(z) < 2:
        raise ValueError(f"{name} must have length >= 2.")
    s = float(sum(z))
    if s <= 0:
        raise ValueError(f"{name} sum must be > 0.")
    out = [float(v) / s for v in z]
    for i, v in enumerate(out):
        if v < -1e-15:
            raise ValueError(f"{name}[{i}] must be >= 0.")
    return out


def logmean(a: float, b: float) -> float:
    _vp("logmean(a)", a)
    _vp("logmean(b)", b)
    if abs(a - b) < 1e-14:
        return a
    return (a - b) / math.log(a / b)


# -----------------------------
# Antoine (optional helper)
# -----------------------------
@dataclass(frozen=True)
class AntoineSpec:
    """
    Antoine equation helper (if your summary sheet uses it):
      log10(Psat) = A - B/(T + C)

    Units must be consistent with the constants you supply.
    """
    A: float
    B: float
    C: float
    base10: bool = True  # True => log10 form (typical). False => ln form.


def Psat_antoine(T: float, spec: AntoineSpec) -> float:
    if spec.base10:
        return 10.0 ** (spec.A - spec.B / (T + spec.C))
    return math.exp(spec.A - spec.B / (T + spec.C))


# -----------------------------
# Raoult / K-values / alpha
# -----------------------------
def K_from_Psat(Psat: Sequence[float], P_total: float) -> List[float]:
    _vp("P_total", P_total)
    if len(Psat) < 2:
        raise ValueError("Psat must have length >= 2.")
    Ks: List[float] = []
    for i, p in enumerate(Psat):
        _vp(f"Psat[{i}]", float(p))
        Ks.append(float(p) / P_total)
    return Ks


def y_from_xK(x: Sequence[float], K: Sequence[float]) -> List[float]:
    xN = normalize(x, "x")
    if len(K) != len(xN):
        raise ValueError("K length must match x.")
    denom = 0.0
    tmp: List[float] = []
    for i, (xi, Ki) in enumerate(zip(xN, K)):
        _vp(f"K[{i}]", float(Ki))
        tmp.append(xi * float(Ki))
        denom += tmp[-1]
    _vp("sum(Kx)", denom)
    return [v / denom for v in tmp]


def x_from_yK(y: Sequence[float], K: Sequence[float]) -> List[float]:
    yN = normalize(y, "y")
    if len(K) != len(yN):
        raise ValueError("K length must match y.")
    denom = 0.0
    tmp: List[float] = []
    for i, (yi, Ki) in enumerate(zip(yN, K)):
        _vp(f"K[{i}]", float(Ki))
        tmp.append(yi / float(Ki))
        denom += tmp[-1]
    _vp("sum(y/K)", denom)
    return [v / denom for v in tmp]


def relative_volatility(K: Sequence[float], i: int, j: int) -> float:
    if i == j:
        raise ValueError("i and j must be different.")
    if i < 0 or j < 0 or i >= len(K) or j >= len(K):
        raise ValueError("i,j out of range.")
    _vp(f"K[{i}]", float(K[i]))
    _vp(f"K[{j}]", float(K[j]))
    return float(K[i]) / float(K[j])


def alpha_avg_geometric(alpha_top: float, alpha_bottom: float) -> float:
    # summary sheet: alpha_avg = sqrt(alpha_D * alpha_W)
    _vp("alpha_top", alpha_top)
    _vp("alpha_bottom", alpha_bottom)
    return math.sqrt(alpha_top * alpha_bottom)


# -----------------------------
# Bubble / Dew at known K
# -----------------------------
def bubble_sum_xK(x: Sequence[float], K: Sequence[float]) -> float:
    xN = normalize(x, "x")
    if len(K) != len(xN):
        raise ValueError("K length must match x.")
    return sum(xi * float(Ki) for xi, Ki in zip(xN, K))


def dew_sum_y_over_K(y: Sequence[float], K: Sequence[float]) -> float:
    yN = normalize(y, "y")
    if len(K) != len(yN):
        raise ValueError("K length must match y.")
    return sum(yi / float(Ki) for yi, Ki in zip(yN, K))


def bubble_residual(x: Sequence[float], K: Sequence[float]) -> float:
    return bubble_sum_xK(x, K) - 1.0


def dew_residual(y: Sequence[float], K: Sequence[float]) -> float:
    return dew_sum_y_over_K(y, K) - 1.0


def solve_bubble_or_dew_T(
    z: Sequence[float],
    K_of_T: Callable[[float], Sequence[float]],
    T_lo: float,
    T_hi: float,
    mode: str,
    maxiter: int = 200,
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """
    Summary-sheet bubble/dew:
      bubble: sum x_i K_i(T) = 1
      dew:    sum y_i / K_i(T) = 1
    """
    zN = normalize(z, "z")
    _vp("T_lo", T_lo)
    _vp("T_hi", T_hi)
    if T_hi <= T_lo:
        raise ValueError("Require T_hi > T_lo.")
    if mode not in ("bubble", "dew"):
        raise ValueError("mode must be 'bubble' or 'dew'.")

    def resid(T: float) -> float:
        K = [float(v) for v in K_of_T(T)]
        if len(K) != len(zN):
            raise ValueError("K(T) length mismatch.")
        if mode == "bubble":
            return bubble_sum_xK(zN, K) - 1.0
        return dew_sum_y_over_K(zN, K) - 1.0

    a, b = T_lo, T_hi
    fa, fb = resid(a), resid(b)
    if fa == 0.0:
        T = a
    elif fb == 0.0:
        T = b
    else:
        if fa * fb > 0:
            raise ValueError("Temperature bracket does not straddle root; adjust T_lo/T_hi.")
        lo, hi = a, b
        flo, fhi = fa, fb
        for _ in range(maxiter):
            mid = 0.5 * (lo + hi)
            fm = resid(mid)
            if abs(fm) <= tol:
                T = mid
                break
            if flo * fm <= 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        else:
            T = 0.5 * (lo + hi)

    K_T = [float(v) for v in K_of_T(T)]
    return {"mode": mode, "T": T, "K(T)": K_T, "residual": resid(T)}


# -----------------------------
# Binary constant-alpha VLE
# -----------------------------
def y_eq_from_x_constant_alpha(x: float, alpha: float) -> float:
    # y = (alpha x) / (1 + (alpha-1)x)
    _vf01_open("x", x)
    _vp("alpha", alpha)
    return (alpha * x) / (1.0 + (alpha - 1.0) * x)


def x_eq_from_y_constant_alpha(y: float, alpha: float) -> float:
    # x = y / (alpha - (alpha-1)y)
    _vf01_open("y", y)
    _vp("alpha", alpha)
    denom = alpha - (alpha - 1.0) * y
    _vp("denom", denom)
    return y / denom