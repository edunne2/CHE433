# bank/two_film_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Sequence, List

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class WettedWallTwoFilmTableSpec:
    """
    Wetted-wall two-film model with tabulated equilibrium y = f(x),
    for A diffusing through stagnant/nondiffusing B in gas and nondiffusing liquid.

    Bulk point:
        P(x_AL, y_AG)

    Interfacial flux relations (Eq. 22.1-37 form):
        N_A = [k'y/(1-yA)iM] (y_AG - y_Ai) = [k'x/(1-xA)iM] (x_Ai - x_AL)

    Log-mean factors (nondiffusing components):
        (1-yA)iM = [(1-y_Ai)-(1-y_AG)] / ln[(1-y_Ai)/(1-y_AG)]
        (1-xA)iM = [(1-x_AL)-(1-x_Ai)] / ln[(1-x_AL)/(1-x_Ai)]

    Interfacial equilibrium:
        y_Ai = f(x_Ai)

    Iterative graphical construction:
        slope PM = (y_AG - y_Ai)/(x_AL - x_Ai)
                = - { [k'x/(1-x)iM] / [k'y/(1-y)iM] }

    OVERALL COEFFICIENTS (Eq. 22.1-51 to 22.1-56 form):
      Define equilibrium end points:
        y_A*  = f(x_AL)              (gas in equilibrium with bulk liquid)
        x_A*  such that f(x_A*)=y_AG (liquid in equilibrium with bulk gas)

      Chord slopes:
        m'  = (y_Ai - y_A*)/(x_Ai - x_AL)     (Eq. 22.1-42)
        m'' = (y_AG - y_Ai)/(x_A* - x_Ai)     (Eq. 22.1-46)

      Overall driving forces:
        N_A = [K'y/(1-yA)BM] (y_AG - y_A*) = [K'x/(1-xA)BM] (x_A* - x_AL)  (Eq. 22.1-51)

      Overall resistances (stagnant B case):
        1/[K'y/(1-yA)BM] = 1/[k'y/(1-yA)iM] + m'/[k'x/(1-xA)iM]           (Eq. 22.1-53)
        1/[K'x/(1-xA)BM] = 1/[m'' k'y/(1-yA)iM] + 1/[k'x/(1-xA)iM]        (Eq. 22.1-55)

      Log-mean BM factors:
        (1-yA)BM = [(1-y_A*)-(1-y_AG)] / ln[(1-y_A*)/(1-y_AG)]            (Eq. 22.1-54)
        (1-xA)BM = [(1-x_AL)-(1-x_A*)] / ln[(1-x_AL)/(1-x_A*)]            (Eq. 22.1-56)

    Outputs include:
      - interface (x_Ai, y_Ai), flux N_A
      - overall coefficients: K'y, K'x and bracketed Ky=K'y/(1-y)BM, Kx=K'x/(1-x)BM
      - percent resistance in gas film based on Eq. 22.1-53 split
    """
    x_table: Sequence[float]
    y_table: Sequence[float]
    k_y: float                 # k'y (kgmol/(s*m^2*molfrac))
    k_x: float                 # k'x (kgmol/(s*m^2*molfrac))
    y_AG: float                # bulk gas mole fraction
    x_AL: float                # bulk liquid mole fraction
    max_iter: int = 50
    tol_slope: float = 1e-8


def _arr(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    if a.ndim != 1:
        raise ValueError("Table inputs must be 1D.")
    return a


def _validate_table(x: np.ndarray, y: np.ndarray) -> None:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x_table and y_table must have the same length >= 2.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Equilibrium table has non-finite values.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x_table must be strictly increasing.")
    if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
        raise ValueError("x_table and y_table must be in [0,1].")


def _validate_inputs(kx: float, ky: float, xAL: float, yAG: float) -> None:
    if kx <= 0 or ky <= 0:
        raise ValueError("k_x and k_y must be > 0.")
    if not (0.0 <= xAL <= 1.0):
        raise ValueError("x_AL must be in [0,1].")
    if not (0.0 <= yAG <= 1.0):
        raise ValueError("y_AG must be in [0,1].")


def _logmean(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        raise ValueError("Log-mean inputs must be > 0.")
    if abs(a - b) < 1e-14:
        return a
    return (a - b) / np.log(a / b)


def _logmean_one_minus(v1: float, v2: float) -> float:
    # log-mean of (1 - v) between v1 and v2
    if not (0.0 <= v1 < 1.0) or not (0.0 <= v2 < 1.0):
        raise ValueError("Values must satisfy 0 <= v < 1 for log-mean of (1-v).")
    return _logmean(1.0 - v1, 1.0 - v2)


def _find_intersection_x(eq: PchipInterpolator, xAL: float, yAG: float, slope: float, x_min: float, x_max: float) -> float:
    # Solve eq(x) = yAG + slope*(x - xAL)
    def f(x: float) -> float:
        return float(eq(x)) - (yAG + slope * (x - xAL))

    xs = np.linspace(x_min, x_max, 4001)
    fs = np.array([f(xx) for xx in xs])
    idx = np.where(np.sign(fs[:-1]) * np.sign(fs[1:]) <= 0)[0]
    if len(idx) == 0:
        raise ValueError("No intersection found between PM line and equilibrium curve in table range.")
    i = int(idx[0])
    return float(brentq(f, float(xs[i]), float(xs[i + 1]), xtol=1e-14, maxiter=200))


def _invert_y_to_x(eq: PchipInterpolator, y_target: float, x_min: float, x_max: float) -> float:
    def f(x: float) -> float:
        return float(eq(x)) - y_target

    xs = np.linspace(x_min, x_max, 4001)
    fs = np.array([f(xx) for xx in xs])
    idx = np.where(np.sign(fs[:-1]) * np.sign(fs[1:]) <= 0)[0]
    if len(idx) == 0:
        raise ValueError("Could not invert y->x within table range.")
    i = int(idx[0])
    return float(brentq(f, float(xs[i]), float(xs[i + 1]), xtol=1e-14, maxiter=200))


def solve_wetted_wall_two_film_from_table(spec: WettedWallTwoFilmTableSpec) -> Dict[str, Any]:
    x = _arr(spec.x_table)
    y = _arr(spec.y_table)
    _validate_table(x, y)
    _validate_inputs(spec.k_x, spec.k_y, spec.x_AL, spec.y_AG)

    eq = PchipInterpolator(x, y, extrapolate=False)
    x_min, x_max = float(x[0]), float(x[-1])

    if not (x_min <= spec.x_AL <= x_max):
        raise ValueError("x_AL is outside x_table range; extend equilibrium data.")

    # Equilibrium endpoints for overall driving forces
    y_star = float(eq(spec.x_AL))                  # y_A* in equilibrium with bulk liquid
    x_star = _invert_y_to_x(eq, spec.y_AG, x_min, x_max)  # x_A* in equilibrium with bulk gas

    # Iteration 1: dilute => (1-y)iM ~ 1, (1-x)iM ~ 1
    one_minus_y_iM = 1.0
    one_minus_x_iM = 1.0
    slope = - (spec.k_x / one_minus_x_iM) / (spec.k_y / one_minus_y_iM)

    hist: List[Dict[str, float]] = []

    for it in range(1, spec.max_iter + 1):
        x_i = _find_intersection_x(eq, spec.x_AL, spec.y_AG, slope, x_min, x_max)
        y_i = float(eq(x_i))

        one_minus_y_new = _logmean_one_minus(y_i, spec.y_AG)
        one_minus_x_new = _logmean_one_minus(spec.x_AL, x_i)

        slope_new = - (spec.k_x / one_minus_x_new) / (spec.k_y / one_minus_y_new)

        hist.append(
            {
                "iter": float(it),
                "slope": float(slope),
                "x_Ai": float(x_i),
                "y_Ai": float(y_i),
                "one_minus_y_iM": float(one_minus_y_new),
                "one_minus_x_iM": float(one_minus_x_new),
                "slope_new": float(slope_new),
            }
        )

        if abs(slope_new - slope) <= spec.tol_slope * max(1.0, abs(slope)):
            slope = slope_new
            one_minus_y_iM = one_minus_y_new
            one_minus_x_iM = one_minus_x_new
            break

        slope = slope_new
        one_minus_y_iM = one_minus_y_new
        one_minus_x_iM = one_minus_x_new
    else:
        raise RuntimeError("Interface iteration did not converge.")

    x_Ai = hist[-1]["x_Ai"]
    y_Ai = hist[-1]["y_Ai"]

    # Effective film coefficients for stagnant/nondiffusing components
    ky_eff = spec.k_y / one_minus_y_iM
    kx_eff = spec.k_x / one_minus_x_iM

    # Flux (Eq. 22.1-37 form)
    N_A_g = ky_eff * (spec.y_AG - y_Ai)
    N_A_l = kx_eff * (x_Ai - spec.x_AL)
    N_A = 0.5 * (N_A_g + N_A_l)

    # Chord slopes for overall-resistance equations
    m_prime = (y_Ai - y_star) / (x_Ai - spec.x_AL)
    m_dprime = (spec.y_AG - y_Ai) / (x_star - x_Ai)

    # BM log-mean factors for overall driving force endpoints
    one_minus_y_BM = _logmean_one_minus(y_star, spec.y_AG)
    one_minus_x_BM = _logmean_one_minus(spec.x_AL, x_star)

    # Overall bracketed coefficients (Eq. 22.1-53, 22.1-55)
    Ky_overall = 1.0 / (1.0 / ky_eff + m_prime / kx_eff)                 # = K'y/(1-y)BM
    Kx_overall = 1.0 / (1.0 / kx_eff + 1.0 / (m_dprime * ky_eff))        # = K'x/(1-x)BM

    # Convert to primed overall coefficients (Eq. 22.1-52)
    Ky_prime = Ky_overall * one_minus_y_BM
    Kx_prime = Kx_overall * one_minus_x_BM

    # Consistent flux from overall driving forces (Eq. 22.1-51)
    N_A_from_overall_y = Ky_overall * (spec.y_AG - y_star)
    N_A_from_overall_x = Kx_overall * (x_star - spec.x_AL)

    # Resistance split (based on Eq. 22.1-53 total resistance)
    R_gas = 1.0 / ky_eff
    R_liq_equiv = m_prime / kx_eff
    pct_R_gas = (R_gas / (R_gas + R_liq_equiv)) * 100.0

    return {
        "iteration": {
            "history": hist,
            "slope_final": slope,
            "one_minus_y_iM": one_minus_y_iM,
            "one_minus_x_iM": one_minus_x_iM,
        },
        "equilibrium_endpoints": {
            "y_A_star_at_xAL": y_star,
            "x_A_star_at_yAG": x_star,
            "one_minus_y_BM": one_minus_y_BM,
            "one_minus_x_BM": one_minus_x_BM,
        },
        "interface": {
            "x_Ai": x_Ai,
            "y_Ai": y_Ai,
            "m_prime": m_prime,
            "m_dprime": m_dprime,
        },
        "film_effective": {
            "ky_eff": ky_eff,
            "kx_eff": kx_eff,
        },
        "overall": {
            # bracketed coefficients used directly in Eq. 22.1-51
            "Ky_overall": Ky_overall,     # = K'y/(1-y)BM
            "Kx_overall": Kx_overall,     # = K'x/(1-x)BM
            # primed coefficients (Eq. 22.1-52)
            "Ky_prime": Ky_prime,
            "Kx_prime": Kx_prime,
        },
        "flux": {
            "N_A": N_A,
            "N_A_from_gas_film": N_A_g,
            "N_A_from_liq_film": N_A_l,
            "N_A_from_overall_y": N_A_from_overall_y,
            "N_A_from_overall_x": N_A_from_overall_x,
        },
        "resistance": {
            "percent_R_gas": pct_R_gas,
            "percent_R_liq": 100.0 - pct_R_gas,
        },
    }