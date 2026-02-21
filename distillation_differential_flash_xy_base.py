# bank/distillation_differential_flash_xy_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Sequence, List, Tuple

import math


# -----------------------------
# validation / utilities
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


def _sort_pairs_by_first(a: Sequence[float], b: Sequence[float]) -> Tuple[List[float], List[float]]:
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Sequences must have same length >= 2.")
    pairs = sorted((float(ai), float(bi)) for ai, bi in zip(a, b))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _interp_piecewise_linear(x: float, xs: List[float], ys: List[float]) -> float:
    # clamp at endpoints
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]

    # binary search for bracket
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid

    x0, x1 = xs[lo], xs[lo + 1]
    y0, y1 = ys[lo], ys[lo + 1]
    if abs(x1 - x0) < 1e-18:
        return 0.5 * (y0 + y1)
    w = (x - x0) / (x1 - x0)
    return y0 + w * (y1 - y0)


def _bisection(fn, a: float, b: float, tol: float = 1e-12, maxiter: int = 400) -> float:
    fa = fn(a)
    fb = fn(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError("Bisection requires sign change.")
    lo, hi = a, b
    flo, fhi = fa, fb
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fm = fn(mid)
        if abs(fm) <= tol:
            return mid
        if flo * fm <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


# -----------------------------
# equilibrium model (tabulated x-y)
# -----------------------------
@dataclass(frozen=True)
class XYEquilibrium:
    """
    Binary equilibrium y(x) for the light component, defined by tabulated (x,y).
    - Uses piecewise-linear interpolation in x for y(x)
    - Provides inverse x(y) by building a y-sorted table and interpolating.
    Assumes y(x) is monotone increasing (typical for binary VLE at fixed P).
    """
    x: Sequence[float]
    y: Sequence[float]

    def __post_init__(self) -> None:
        xs, ys = _sort_pairs_by_first(self.x, self.y)
        for i, (xi, yi) in enumerate(zip(xs, ys)):
            _vf01(f"x[{i}]", xi)
            _vf01(f"y[{i}]", yi)
        object.__setattr__(self, "_xs", xs)
        object.__setattr__(self, "_ys", ys)

        # inverse table: sort by y
        ys2, xs2 = _sort_pairs_by_first(ys, xs)
        object.__setattr__(self, "_ys2", ys2)
        object.__setattr__(self, "_xs2", xs2)

    def y_of_x(self, x: float) -> float:
        _vf01("x", x)
        return float(_interp_piecewise_linear(float(x), self._xs, self._ys))

    def x_of_y(self, y: float) -> float:
        _vf01("y", y)
        return float(_interp_piecewise_linear(float(y), self._ys2, self._xs2))


# -----------------------------
# Differential (Rayleigh) distillation
# -----------------------------
@dataclass(frozen=True)
class RayleighDifferentialSpec:
    """
    Rayleigh equation for differential (batch) distillation:

      ln(F0/W) = ∫_{xW}^{x0} dx / (y(x) - x)

    Inputs:
      F0 : initial liquid moles
      x0 : initial liquid light-component mole fraction
      W  : final remaining liquid moles
      eq : XYEquilibrium giving y(x)
    """
    F0: float
    x0: float
    W: float
    eq: XYEquilibrium

    n_int: int = 6000
    tol: float = 1e-10
    maxiter: int = 400


def _rayleigh_integral(eq: XYEquilibrium, xW: float, x0: float, n: int) -> float:
    _vf01("xW", xW)
    _vf01("x0", x0)
    if not (xW < x0):
        raise ValueError("Require xW < x0.")

    # composite trapezoid on uniform grid
    dx = (x0 - xW) / (n - 1)
    s = 0.0
    for i in range(n):
        x = xW + i * dx
        y = eq.y_of_x(x)
        denom = y - x
        if denom <= 1e-14:
            raise ValueError("Encountered y(x) - x <= 0; Rayleigh integral invalid on this interval.")
        f = 1.0 / denom
        w = 0.5 if (i == 0 or i == n - 1) else 1.0
        s += w * f
    return s * dx


def solve_rayleigh(spec: RayleighDifferentialSpec) -> Dict[str, Any]:
    _vp("F0", spec.F0)
    _vp("W", spec.W)
    _vf01("x0", spec.x0)
    if spec.W >= spec.F0:
        raise ValueError("Require W < F0.")

    D = spec.F0 - spec.W
    _vp("D", D)

    target = math.log(spec.F0 / spec.W)

    def g(xW: float) -> float:
        return _rayleigh_integral(spec.eq, xW, spec.x0, spec.n_int) - target

    a = 1e-12
    b = spec.x0 - 1e-12
    ga = g(a)
    gb = g(b)
    if ga * gb > 0.0:
        # try moving lower bound upward (avoid extreme endpoint behavior)
        a2 = max(1e-6, 0.05 * spec.x0)
        ga2 = g(a2)
        if ga2 * gb > 0.0:
            raise ValueError("Could not bracket xW for Rayleigh. Check W target and equilibrium curve.")
        a = a2

    xW = _bisection(g, a, b, tol=spec.tol, maxiter=spec.maxiter)

    # average distillate composition from component balance:
    # F0*x0 = D*ybar + W*xW
    ybar = (spec.F0 * spec.x0 - spec.W * xW) / D
    ybar = max(0.0, min(1.0, ybar))

    return {
        "inputs": {"F0": spec.F0, "x0": spec.x0, "W": spec.W, "D": D},
        "outputs": {"xW": xW, "ybar_distillate": ybar},
        "rayleigh": {"ln_F0_over_W": target},
    }


# -----------------------------
# Flash (equilibrium) distillation with tabulated y(x)
# -----------------------------
@dataclass(frozen=True)
class FlashXYSpec:
    """
    Single-stage equilibrium flash using tabulated y(x).

    With vapor fraction f = V/F:
      z = (1-f)*x + f*y(x)

    Inputs:
      F : feed moles
      z : feed light-component mole fraction
      V : vapor moles removed (or distilled)
      eq: XYEquilibrium
    """
    F: float
    z: float
    V: float
    eq: XYEquilibrium

    tol: float = 1e-12
    maxiter: int = 400


def solve_flash_xy(spec: FlashXYSpec) -> Dict[str, Any]:
    _vp("F", spec.F)
    _vp("V", spec.V)
    _vf01("z", spec.z)
    if spec.V >= spec.F:
        raise ValueError("Require V < F.")

    f = spec.V / spec.F

    def h(x: float) -> float:
        _vf01("x", x)
        return (1.0 - f) * x + f * spec.eq.y_of_x(x) - spec.z

    ha = h(0.0)
    hb = h(1.0)
    if ha == 0.0:
        x = 0.0
    elif hb == 0.0:
        x = 1.0
    else:
        if ha * hb > 0.0:
            raise ValueError("Could not bracket flash x in [0,1]. Check inputs.")
        x = _bisection(h, 0.0, 1.0, tol=spec.tol, maxiter=spec.maxiter)

    y = spec.eq.y_of_x(x)
    L = spec.F - spec.V

    return {"inputs": {"F": spec.F, "z": spec.z, "V": spec.V, "f": f, "L": L}, "outputs": {"x": x, "y": y}}