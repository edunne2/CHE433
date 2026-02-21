# bank/packed_absorption_height_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Sequence, Tuple, Optional, List

import math
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


@dataclass(frozen=True)
class PackedTowerExample2252Spec:
    """
    Packed-tower height methodology matching Example 22.5-2 (screenshots).

    Basis:
      - Dilute absorption, use mole fractions x, y.
      - Use solute-free flowrates:
          V' = inert-gas molar flow (constant)
          L' = solvent molar flow (constant)
        If only total gas is given, you may approximate V' = V_total*(1-y_in).

    Required endpoints:
      Gas: y1 (inlet, bottom/rich), y2 (outlet, top/lean)
      Liquid: x2 (inlet, top/lean, often 0), x1 (outlet, bottom/rich) computed by overall balance.

    Overall balance to compute x1 (matches the example’s equation form):
        L' * x2/(1-x2) + V' * y1/(1-y1) = L' * x1/(1-x1) + V' * y2/(1-y2)

    Equilibrium:
      Provided as tabulated (x_eq, y_eq). Use PCHIP interpolation y* = f(x).

    Interface compositions at an endpoint P(x, y):
      Iterative graphical slope method:
        First trial (dilute): slope ≈ - [k'xa(1-x)] / [k'ya(1-y)]
        Improved slope uses log-mean factors:
          (1-y)_M = [(1-yi)-(1-y)] / ln[(1-yi)/(1-y)]
          (1-x)_M = [(1-x)-(1-xi)] / ln[(1-x)/(1-xi)]
        Updated slope:
          slope = - [k'xa(1-x)_M] / [k'ya(1-y)_M]

      Find intersection with equilibrium curve:
        line: y_line(x) = y + slope (x - x_bulk)
        intersection gives (x_i, y_i = f(x_i))

    Height calculations (all match the example’s structure):

      Define:
        S = cross-sectional area = πD^2/4
        V1 = V'/(1-y1), V2 = V'/(1-y2),  V_av = (V1+V2)/2
        L1 ≈ L2 ≈ L' (dilute),           L_av = L'

      (a) Using k'ya:
        (y - yi)_M = log-mean of (y1-yi1) and (y2-yi2)
        (V_av/S)(y1 - y2) = (k'ya z) (y - yi)_M
        => z = (V_av/S)(y1-y2) / [k'ya (y-yi)_M]

      (b) Using k'xa:
        (x_i - x)_M = log-mean of (xi1-x1) and (xi2-x2)
        (L_av/S)(x1 - x2) = (k'xa z) (x_i - x)_M
        => z = (L_av/S)(x1-x2) / [k'xa (x_i-x)_M]

      (c) Using K'ya:
        Compute overall K'ya at the rich end (y1,x1) using:
          (1-y)_RM = log-mean of (1-y1*) and (1-y1)
          (1-y)_M  = log-mean of (1-yi1) and (1-y1)
          (1-x)_M  = log-mean of (1-x1)  and (1-xi1)
          m_local  = dy*/dx at x1 (from equilibrium curve derivative)

          Define “bracketed” coefficients:
            ky_eff = k'ya / (1-y)_M
            kx_eff = k'xa / (1-x)_M
            Ky     = K'ya / (1-y)_RM

          Resistance sum (example form):
            1/Ky = 1/ky_eff + m_local/kx_eff
          => Ky = 1 / (1/ky_eff + m_local/kx_eff)
          => K'ya = Ky (1-y)_RM

        Driving force:
          y* at ends: y1* = f(x1), y2* = f(x2)
          (y - y*)_M = log-mean of (y1-y1*) and (y2-y2*)
        Height:
          (V_av/S)(y1 - y2) = (K'ya z) (y - y*)_M
          => z = (V_av/S)(y1-y2) / [K'ya (y-y*)_M]

    Units:
      - V', L', V_av in mol/time
      - k'ya, k'xa, K'ya in mol/(time·m^3·molfrac)
      - S in m^2
      - z in m
    """
    V_prime: float
    L_prime: float
    y1: float
    y2: float
    x2: float
    diameter_m: float
    x_eq: Sequence[float]
    y_eq: Sequence[float]
    k_ya: float
    k_xa: float
    n_scan: int = 6001
    it_max: int = 50
    tol_slope: float = 1e-6


def _vf(name: str, v: float) -> None:
    if not (0.0 <= v < 1.0):
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1.")


def _vp(name: str, v: float) -> None:
    if v <= 0:
        raise ValueError(f"{name} must be > 0.")


def _logmean(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        raise ValueError("log-mean inputs must be > 0.")
    if abs(a - b) < 1e-14:
        return a
    return (a - b) / math.log(a / b)


def logmean_one_minus(v1: float, v2: float) -> float:
    _vf("v1", v1)
    _vf("v2", v2)
    return _logmean(1.0 - v1, 1.0 - v2)


def logmean_delta(a1: float, a2: float) -> float:
    # log-mean of positive differences
    if a1 <= 0 or a2 <= 0:
        raise ValueError("logmean_delta requires both arguments > 0.")
    return _logmean(a1, a2)


def build_eq(x_eq: Sequence[float], y_eq: Sequence[float]) -> PchipInterpolator:
    x = np.asarray(x_eq, dtype=float)
    y = np.asarray(y_eq, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Equilibrium table must have same length >= 2.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x_eq must be strictly increasing.")
    if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
        raise ValueError("Equilibrium values must be in [0,1].")
    return PchipInterpolator(x, y, extrapolate=False)


def Y_from_y(y: float) -> float:
    _vf("y", y)
    return y / (1.0 - y)


def X_from_x(x: float) -> float:
    _vf("x", x)
    return x / (1.0 - x)


def x_from_X(X: float) -> float:
    if X < 0:
        raise ValueError("X must be >= 0.")
    return X / (1.0 + X)


def compute_x1_from_overall_balance(Vp: float, Lp: float, y1: float, y2: float, x2: float) -> float:
    _vp("V_prime", Vp)
    _vp("L_prime", Lp)
    _vf("y1", y1)
    _vf("y2", y2)
    _vf("x2", x2)

    left = Lp * X_from_x(x2) + Vp * Y_from_y(y1)
    right_known = Vp * Y_from_y(y2)
    X1 = (left - right_known) / Lp
    if X1 < 0:
        raise ValueError("Computed X1 < 0; check inputs.")
    x1 = x_from_X(X1)
    if not (0.0 <= x1 < 1.0):
        raise ValueError("Computed x1 out of [0,1).")
    return x1


def _line_eq_residual(eq: PchipInterpolator, x_bulk: float, y_bulk: float, slope: float, x: float) -> float:
    return float(eq(x)) - (y_bulk + slope * (x - x_bulk))


def find_intersection_with_equilibrium(
    eq: PchipInterpolator,
    x_bulk: float,
    y_bulk: float,
    slope: float,
    x_min: float,
    x_max: float,
    n_scan: int,
) -> float:
    xs = np.linspace(x_min, x_max, n_scan)
    fs = np.array([_line_eq_residual(eq, x_bulk, y_bulk, slope, float(xx)) for xx in xs])

    idx = np.where(np.sign(fs[:-1]) * np.sign(fs[1:]) <= 0)[0]
    if len(idx) == 0:
        raise ValueError("No intersection found between construction line and equilibrium within table range.")
    i = int(idx[0])

    a = float(xs[i])
    b = float(xs[i + 1])

    def f(x: float) -> float:
        return _line_eq_residual(eq, x_bulk, y_bulk, slope, x)

    return float(brentq(f, a, b, xtol=1e-14, maxiter=200))


def interface_at_point_example2252(
    eq: PchipInterpolator,
    k_ya: float,
    k_xa: float,
    x_bulk: float,
    y_bulk: float,
    x_min: float,
    x_max: float,
    n_scan: int = 6001,
    it_max: int = 50,
    tol_slope: float = 1e-6,
) -> Dict[str, Any]:
    """
    Returns interface (x_i, y_i) at bulk point (x_bulk, y_bulk) using the exact
    iterative slope method shown in the example.
    """
    _vp("k_ya", k_ya)
    _vp("k_xa", k_xa)
    _vf("x_bulk", x_bulk)
    _vf("y_bulk", y_bulk)

    # Trial 1 (dilute): slope ≈ -kxa(1-x)/kya(1-y)
    slope = - (k_xa * (1.0 - x_bulk)) / (k_ya * (1.0 - y_bulk))

    hist: List[Dict[str, float]] = []

    for it in range(1, it_max + 1):
        x_i = find_intersection_with_equilibrium(eq, x_bulk, y_bulk, slope, x_min, x_max, n_scan)
        y_i = float(eq(x_i))

        one_minus_y_M = logmean_one_minus(y_i, y_bulk)
        one_minus_x_M = logmean_one_minus(x_bulk, x_i)

        slope_new = - (k_xa * one_minus_x_M) / (k_ya * one_minus_y_M)

        hist.append(
            {
                "iter": float(it),
                "slope": float(slope),
                "x_i": float(x_i),
                "y_i": float(y_i),
                "one_minus_y_M": float(one_minus_y_M),
                "one_minus_x_M": float(one_minus_x_M),
                "slope_new": float(slope_new),
            }
        )

        if abs(slope_new - slope) <= tol_slope * max(1.0, abs(slope)):
            slope = slope_new
            break

        slope = slope_new

    return {
        "x_i": hist[-1]["x_i"],
        "y_i": hist[-1]["y_i"],
        "slope_final": slope,
        "one_minus_y_M": hist[-1]["one_minus_y_M"],
        "one_minus_x_M": hist[-1]["one_minus_x_M"],
        "history": hist,
    }


def packed_height_example2252_all(spec: PackedTowerExample2252Spec) -> Dict[str, Any]:
    _vp("V_prime", spec.V_prime)
    _vp("L_prime", spec.L_prime)
    _vp("diameter_m", spec.diameter_m)
    _vp("k_ya", spec.k_ya)
    _vp("k_xa", spec.k_xa)
    _vf("y1", spec.y1)
    _vf("y2", spec.y2)
    _vf("x2", spec.x2)

    eq = build_eq(spec.x_eq, spec.y_eq)
    x_min, x_max = float(min(spec.x_eq)), float(max(spec.x_eq))

    # 1) overall balance for x1
    x1 = compute_x1_from_overall_balance(spec.V_prime, spec.L_prime, spec.y1, spec.y2, spec.x2)

    # 2) interface at top (rich end): (x1,y1)
    top = interface_at_point_example2252(
        eq=eq,
        k_ya=spec.k_ya,
        k_xa=spec.k_xa,
        x_bulk=x1,
        y_bulk=spec.y1,
        x_min=x_min,
        x_max=x_max,
        n_scan=spec.n_scan,
        it_max=spec.it_max,
        tol_slope=spec.tol_slope,
    )

    # 3) interface at bottom (lean end): (x2,y2)
    bot = interface_at_point_example2252(
        eq=eq,
        k_ya=spec.k_ya,
        k_xa=spec.k_xa,
        x_bulk=spec.x2,
        y_bulk=spec.y2,
        x_min=x_min,
        x_max=x_max,
        n_scan=spec.n_scan,
        it_max=spec.it_max,
        tol_slope=spec.tol_slope,
    )

    xi1, yi1 = float(top["x_i"]), float(top["y_i"])
    xi2, yi2 = float(bot["x_i"]), float(bot["y_i"])

    # geometry and average flows
    S = math.pi * (spec.diameter_m ** 2) / 4.0
    V1 = spec.V_prime / (1.0 - spec.y1)
    V2 = spec.V_prime / (1.0 - spec.y2)
    Vav = 0.5 * (V1 + V2)
    Lav = spec.L_prime  # dilute approximation in example

    # (a) height using k'ya
    dy1 = spec.y1 - yi1
    dy2 = spec.y2 - yi2
    dyM = logmean_delta(dy1, dy2)
    z_kya = (Vav / S) * (spec.y1 - spec.y2) / (spec.k_ya * dyM)

    # (b) height using k'xa
    dx1 = xi1 - x1
    dx2 = xi2 - spec.x2
    dxM = logmean_delta(dx1, dx2)
    z_kxa = (Lav / S) * (x1 - spec.x2) / (spec.k_xa * dxM)

    # (c) overall K'ya at rich end, then height
    y1_star = float(eq(x1))
    y2_star = float(eq(spec.x2))

    one_minus_y_RM = logmean_one_minus(y1_star, spec.y1)
    one_minus_y_M_top = logmean_one_minus(yi1, spec.y1)
    one_minus_x_M_top = logmean_one_minus(x1, xi1)

    ky_eff = spec.k_ya / one_minus_y_M_top
    kx_eff = spec.k_xa / one_minus_x_M_top

    m_local = float(eq.derivative()(x1))

    Ky_bracket = 1.0 / (1.0 / ky_eff + m_local / kx_eff)  # Ky = K'ya/(1-y)_RM
    Kya_prime = Ky_bracket * one_minus_y_RM

    dy_star_1 = spec.y1 - y1_star
    dy_star_2 = spec.y2 - y2_star
    dy_star_M = logmean_delta(dy_star_1, dy_star_2)

    z_Kya = (Vav / S) * (spec.y1 - spec.y2) / (Kya_prime * dy_star_M)

    return {
        "bulk": {
            "x1": x1,
            "x2": spec.x2,
            "y1": spec.y1,
            "y2": spec.y2,
        },
        "interface": {
            "top": {"x_i1": xi1, "y_i1": yi1, "slope": float(top["slope_final"])},
            "bottom": {"x_i2": xi2, "y_i2": yi2, "slope": float(bot["slope_final"])},
        },
        "equilibrium_star": {
            "y1_star": y1_star,
            "y2_star": y2_star,
            "m_local_at_x1": m_local,
        },
        "averages": {
            "S_m2": S,
            "V1": V1,
            "V2": V2,
            "Vav": Vav,
            "Lav": Lav,
        },
        "driving_forces": {
            "dyM_y_minus_yi": dyM,
            "dxM_xi_minus_x": dxM,
            "dyM_y_minus_ystar": dy_star_M,
        },
        "overall_Kya": {
            "one_minus_y_RM": one_minus_y_RM,
            "one_minus_y_M_top": one_minus_y_M_top,
            "one_minus_x_M_top": one_minus_x_M_top,
            "ky_eff": ky_eff,
            "kx_eff": kx_eff,
            "Ky_bracket": Ky_bracket,
            "Kya_prime": Kya_prime,
        },
        "heights": {
            "z_using_kya": z_kya,
            "z_using_kxa": z_kxa,
            "z_using_Kya": z_Kya,
        },
        "iteration_history": {
            "top": top["history"],
            "bottom": bot["history"],
        },
    }