# bank/tray_absorption_lmin_kremser_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import math


@dataclass(frozen=True)
class TrayAbsorberLinearEqSpec:
    """
    Countercurrent tray absorber, dilute, linear equilibrium:
        y = m x

    This module implements the full "tray absorber with L = factor*Lmin" workflow:

    Core definitions (mole fractions):
      y : solute in gas
      x : solute in liquid

    Solute-free ratios (for exact solute accounting in dilute derivations):
      Y = y/(1-y)
      X = x/(1-x)

    Solute-free flowrates (assumed constant):
      V' = V_in_total * (1 - y_in)   (inert gas flow)
      L' = L_in_total * (1 - x_in)   (solvent flow; for pure solvent, L' = L_in_total)

    Recovery target:
      Y_out = (1 - recovery)*Y_in    -> converts back to y_out

    Minimum liquid rate (pinch at top, lean outlet gas and rich outlet liquid at equilibrium):
      x1* = y_in / m
      Convert to solute-free:
        X1* = x1*/(1-x1*)
        X2  = x_in/(1-x_in)
        Y1  = y_in/(1-y_in)
        Y2  = y_out/(1-y_out)
      Then:
        L'min = V' (Y1 - Y2) / (X1* - X2)

    Actual design liquid:
      L' = factor * L'min

    Operating line (dilute straight-line form in x-y):
      y = y_out + (L'/V') (x - x_in)

    Graphical trays (step-off in x-y):
      Start at (x_in, y_out):
        horizontal to equilibrium: x = y/m
        vertical to operating line: y_new = y_out + (L'/V')(x - x_in)
      Count trays until y reaches y_in; last tray fractional.

    Analytical trays (Kremser-style closed form for linear equilibrium, includes nonzero x_in):
      Absorption factor:
        A = L'/(m V')
      Let:
        num = y_in  - m x_in
        den = y_out - m x_in
      For A != 1:
        N = ln( (num/den)*(1 - 1/A) + 1/A ) / ln(A)
      For A -> 1:
        (den/num) = 1/(N+1)  ->  N = (num/den) - 1

    Inputs:
      - V_in_total : inlet gas total
      - y_in       : inlet gas mole fraction
      - m          : equilibrium slope
      - x_in       : inlet liquid mole fraction (pure solvent => 0)
      - recovery   : fraction of inlet solute removed (0..1)
      - L_factor_over_min : multiplier on L'min (e.g., 1.3)
    """
    V_in_total: float
    y_in: float
    m: float
    x_in: float = 0.0
    recovery: float = 0.0
    L_factor_over_min: float = 1.0
    max_trays_graphical: int = 200


def _vf(name: str, v: float) -> None:
    if not (0.0 <= v < 1.0):
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1.")


def _vp(name: str, v: float) -> None:
    if v <= 0:
        raise ValueError(f"{name} must be > 0.")


def X_from_x(x: float) -> float:
    _vf("x", x)
    return x / (1.0 - x)


def Y_from_y(y: float) -> float:
    _vf("y", y)
    return y / (1.0 - y)


def x_from_X(X: float) -> float:
    if X < 0:
        raise ValueError("X must be >= 0.")
    return X / (1.0 + X)


def y_from_Y(Y: float) -> float:
    if Y < 0:
        raise ValueError("Y must be >= 0.")
    return Y / (1.0 + Y)


def compute_V_prime(V_in_total: float, y_in: float) -> float:
    _vp("V_in_total", V_in_total)
    _vf("y_in", y_in)
    Vp = V_in_total * (1.0 - y_in)
    _vp("V_prime", Vp)
    return Vp


def compute_L_prime_from_total(L_total: float, x_in: float) -> float:
    _vp("L_total", L_total)
    _vf("x_in", x_in)
    Lp = L_total * (1.0 - x_in)
    _vp("L_prime", Lp)
    return Lp


def compute_L_total_from_prime(L_prime: float, x_in: float) -> float:
    _vp("L_prime", L_prime)
    _vf("x_in", x_in)
    return L_prime / (1.0 - x_in)


def y_out_from_recovery(y_in: float, recovery: float) -> float:
    _vf("y_in", y_in)
    if not (0.0 <= recovery <= 1.0):
        raise ValueError("recovery must satisfy 0 <= recovery <= 1.")
    Y1 = Y_from_y(y_in)
    Y2 = (1.0 - recovery) * Y1
    return y_from_Y(Y2)


def compute_Lmin_prime(V_prime: float, y_in: float, y_out: float, m: float, x_in: float) -> Dict[str, float]:
    _vp("V_prime", V_prime)
    _vf("y_in", y_in)
    _vf("y_out", y_out)
    _vp("m", m)
    _vf("x_in", x_in)

    x1_star = y_in / m
    if x1_star >= 1.0:
        raise ValueError("x1* = y_in/m >= 1; invalid.")
    if x1_star <= x_in:
        raise ValueError("x1* <= x_in; pinch/minimum-L calculation invalid.")

    X1_star = X_from_x(x1_star)
    X2 = X_from_x(x_in)
    Y1 = Y_from_y(y_in)
    Y2 = Y_from_y(y_out)

    denom = (X1_star - X2)
    if denom <= 0:
        raise ValueError("X1* - X2 must be > 0.")

    Lmin_prime = V_prime * (Y1 - Y2) / denom
    _vp("Lmin_prime", Lmin_prime)

    return {
        "x1_star": x1_star,
        "X1_star": X1_star,
        "X2": X2,
        "Y1": Y1,
        "Y2": Y2,
        "Lmin_prime": Lmin_prime,
    }


def absorption_factor(L_prime: float, m: float, V_prime: float) -> float:
    _vp("L_prime", L_prime)
    _vp("m", m)
    _vp("V_prime", V_prime)
    return L_prime / (m * V_prime)


def trays_analytical_kremser(y_in: float, y_out: float, x_in: float, m: float, L_prime: float, V_prime: float) -> float:
    _vf("y_in", y_in)
    _vf("y_out", y_out)
    _vf("x_in", x_in)
    _vp("m", m)
    _vp("L_prime", L_prime)
    _vp("V_prime", V_prime)

    A = absorption_factor(L_prime, m, V_prime)

    num = (y_in - m * x_in)
    den = (y_out - m * x_in)
    if num <= 0 or den <= 0:
        raise ValueError("Require y - m x_in > 0 for analytical form.")

    if abs(A - 1.0) < 1e-12:
        # limit A -> 1: den/num = 1/(N+1)
        return (num / den) - 1.0

    inside = (num / den) * (1.0 - 1.0 / A) + 1.0 / A
    if inside <= 0:
        raise ValueError("Log argument <= 0 in analytical trays equation.")
    return math.log(inside) / math.log(A)


def trays_graphical_stepoff_xy(
    y_in: float,
    y_out: float,
    x_in: float,
    m: float,
    L_over_V: float,
    max_trays: int = 200,
) -> Dict[str, Any]:
    _vf("y_in", y_in)
    _vf("y_out", y_out)
    _vf("x_in", x_in)
    _vp("m", m)
    _vp("L_over_V", L_over_V)
    _vp("max_trays", float(max_trays))

    pts: List[Tuple[float, float]] = []
    trays = 0

    # start at top: (x_in, y_out)
    y_curr = y_out
    pts.append((x_in, y_out))

    if y_curr >= y_in:
        return {"N_float": 0.0, "N_ceiling": 0, "points": pts}

    for n in range(1, max_trays + 1):
        # horizontal to equilibrium
        x_eq = y_curr / m
        pts.append((x_eq, y_curr))

        # vertical to operating line
        y_next = y_out + L_over_V * (x_eq - x_in)
        pts.append((x_eq, y_next))

        trays += 1

        if y_next >= y_in:
            frac = 1.0 if y_next == y_curr else (y_in - y_curr) / (y_next - y_curr)
            frac = max(0.0, min(1.0, frac))
            N_float = (trays - 1) + frac
            return {"N_float": N_float, "N_ceiling": int(math.ceil(N_float - 1e-12)), "points": pts}

        y_curr = y_next

    raise RuntimeError("Exceeded max_trays without reaching y_in; check feasibility or increase max_trays.")


def design_tray_absorber_linear(spec: TrayAbsorberLinearEqSpec) -> Dict[str, Any]:
    _vp("V_in_total", spec.V_in_total)
    _vf("y_in", spec.y_in)
    _vp("m", spec.m)
    _vf("x_in", spec.x_in)
    if not (0.0 <= spec.recovery <= 1.0):
        raise ValueError("recovery must satisfy 0 <= recovery <= 1.")
    _vp("L_factor_over_min", spec.L_factor_over_min)

    y_out = y_out_from_recovery(spec.y_in, spec.recovery)

    Vp = compute_V_prime(spec.V_in_total, spec.y_in)
    pinch = compute_Lmin_prime(Vp, spec.y_in, y_out, spec.m, spec.x_in)

    Lmin_prime = pinch["Lmin_prime"]
    Lp = spec.L_factor_over_min * Lmin_prime
    LV = Lp / Vp
    A = absorption_factor(Lp, spec.m, Vp)

    N_analytical = trays_analytical_kremser(spec.y_in, y_out, spec.x_in, spec.m, Lp, Vp)
    N_graph = trays_graphical_stepoff_xy(spec.y_in, y_out, spec.x_in, spec.m, LV, max_trays=spec.max_trays_graphical)

    # also provide operating line endpoints in x-y for plotting
    x1_star = pinch["x1_star"]  # pinch equilibrium at top
    op_line = {
        "slope_Lprime_over_Vprime": LV,
        "point_top": (spec.x_in, y_out),
        "point_bottom_at_x1_star": (x1_star, y_out + LV * (x1_star - spec.x_in)),
    }

    return {
        "inlet": {"V_in_total": spec.V_in_total, "y_in": spec.y_in, "x_in": spec.x_in},
        "target": {"recovery": spec.recovery, "y_out": y_out},
        "equilibrium": {"m": spec.m},
        "solute_free": {
            "V_prime": Vp,
            "Lmin_prime": Lmin_prime,
            "L_prime": Lp,
            "L_over_V": LV,
            "A": A,
        },
        "pinch": pinch,
        "operating_line_xy": op_line,
        "trays": {
            "graphical_N_float": N_graph["N_float"],
            "graphical_N_ceiling": N_graph["N_ceiling"],
            "graphical_step_points_xy": N_graph["points"],
            "analytical_N_float": N_analytical,
            "analytical_N_ceiling": int(math.ceil(N_analytical - 1e-12)),
        },
    }