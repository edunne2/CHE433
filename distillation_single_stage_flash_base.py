# bank/distillation_single_stage_flash_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Sequence, List, Tuple

import math


def _vp(name: str, v: float) -> None:
    if v <= 0:
        raise ValueError(f"{name} must be > 0.")


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


# ---------------------------------------
# Bubble / dew checks at given (z,K)
# ---------------------------------------
def bubble_sum_zK(z: Sequence[float], K: Sequence[float]) -> float:
    zN = normalize(z, "z")
    if len(K) != len(zN):
        raise ValueError("K length must match z.")
    return sum(zi * float(Ki) for zi, Ki in zip(zN, K))


def dew_sum_z_over_K(z: Sequence[float], K: Sequence[float]) -> float:
    zN = normalize(z, "z")
    if len(K) != len(zN):
        raise ValueError("K length must match z.")
    return sum(zi / float(Ki) for zi, Ki in zip(zN, K))


# ---------------------------------------
# Rachford-Rice
# ---------------------------------------
def rachford_rice_residual(f: float, zF: List[float], K: List[float]) -> float:
    # sum z_i (K_i - 1) / (1 + f (K_i - 1)) = 0
    s = 0.0
    for zi, Ki in zip(zF, K):
        denom = 1.0 + f * (Ki - 1.0)
        s += zi * (Ki - 1.0) / denom
    return s


def bracket_rr(zF: List[float], K: List[float]) -> Tuple[float, float]:
    a, b = 0.0, 1.0
    fa = rachford_rice_residual(a, zF, K)
    fb = rachford_rice_residual(b, zF, K)
    if fa == 0.0:
        return a, a
    if fb == 0.0:
        return b, b
    if fa * fb > 0:
        return (a, a) if abs(fa) <= abs(fb) else (b, b)
    return a, b


def bisection(fn, a: float, b: float, maxiter: int = 200, tol: float = 1e-12) -> float:
    fa = fn(a)
    fb = fn(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        raise ValueError("Bisection requires sign change.")
    lo, hi = a, b
    flo, fhi = fa, fb
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fm = fn(mid)
        if abs(fm) <= tol:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class FlashSpec:
    """
    Summary sheet flash:
      f = V/F
      x_i = z_i / (1 + f (K_i - 1))
      y_i = K_i x_i
      RR: sum z_i (K_i - 1) / (1 + f (K_i - 1)) = 0
    """
    F_total: float
    zF: Sequence[float]
    K: Sequence[float]
    maxiter: int = 200
    tol: float = 1e-12


def solve_flash(spec: FlashSpec) -> Dict[str, Any]:
    _vp("F_total", spec.F_total)
    zF = normalize(spec.zF, "zF")
    K = [float(v) for v in spec.K]
    if len(K) != len(zF):
        raise ValueError("K length must match zF.")
    for i, Ki in enumerate(K):
        _vp(f"K[{i}]", Ki)

    checks = {
        "bubble_sum_zK": bubble_sum_zK(zF, K),
        "dew_sum_z_over_K": dew_sum_z_over_K(zF, K),
    }

    a, b = bracket_rr(zF, K)
    if a == b:
        f = a
        method = "best_endpoint_or_single_phase"
    else:
        f = bisection(lambda ff: rachford_rice_residual(ff, zF, K), a, b, maxiter=spec.maxiter, tol=spec.tol)
        method = "bisection"

    x: List[float] = []
    y: List[float] = []
    for zi, Ki in zip(zF, K):
        denom = 1.0 + f * (Ki - 1.0)
        xi = zi / denom
        yi = Ki * xi
        x.append(xi)
        y.append(yi)

    sx, sy = sum(x), sum(y)
    if sx <= 0 or sy <= 0:
        raise ValueError("Computed phase compositions invalid.")
    x = [v / sx for v in x]
    y = [v / sy for v in y]

    V = f * spec.F_total
    L = (1.0 - f) * spec.F_total

    return {
        "method": method,
        "inputs": {"F_total": spec.F_total, "zF": zF, "K": K},
        "checks": checks,
        "split": {"f_V_over_F": f, "V": V, "L": L},
        "phases": {"x": x, "y": y},
    }