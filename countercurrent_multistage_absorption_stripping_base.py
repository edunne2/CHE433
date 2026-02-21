# bank/countercurrent_multistage_absorption_stripping_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import math


# ============================================================
# --------------------- Utilities -----------------------------
# ============================================================

def _vp(name: str, v: float) -> None:
    if v is None or v <= 0:
        raise ValueError(f"{name} must be > 0.")


def _vf(name: str, v: float) -> None:
    if v is None or not (0.0 <= v < 1.0):
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1.")


def Y_from_y(y: float) -> float:
    _vf("y", y)
    return y / (1.0 - y)


def X_from_x(x: float) -> float:
    _vf("x", x)
    return x / (1.0 - x)


def y_from_Y(Y: float) -> float:
    if Y < 0:
        raise ValueError("Y must be >= 0.")
    return Y / (1.0 + Y)


def x_from_X(X: float) -> float:
    if X < 0:
        raise ValueError("X must be >= 0.")
    return X / (1.0 + X)


# ============================================================
# -------- Exact Ratio-Space Equilibrium (y = m x) ----------
# ============================================================

def Yeq_from_X_linear(m: float, X: float) -> float:
    """
    Exact equilibrium in solute-free ratio coordinates.

        y = m x
        Y* = m X / [1 + X(1 - m)]
    """
    _vp("m", m)
    if X < 0:
        raise ValueError("X must be >= 0.")

    denom = 1.0 + X * (1.0 - m)
    if denom <= 0:
        return float("inf")
    return (m * X) / denom


def Xeq_from_Y_linear(m: float, Y: float) -> float:
    """
    Invert exact equilibrium in ratio coordinates.
    """
    _vp("m", m)
    if Y < 0:
        raise ValueError("Y must be >= 0.")

    y = y_from_Y(Y)
    x = y / m
    if x >= 1.0:
        return float("inf")
    return X_from_x(x)


# ============================================================
# -------------------- Main Spec ------------------------------
# ============================================================

@dataclass(frozen=True)
class CountercurrentMultistageSpec:
    """
    General countercurrent absorption/stripping (theoretical stages).

    Supports:
      • Absorption and stripping
      • x–y analytical (Kremser) solution
      • Exact ratio-space Y–X stepping
      • Fraction absorbed / stripped specification
      • Solute-free flow basis

    Linear equilibrium: y = m x
    """

    m: float
    V_in_total: float
    y_in: float
    L_in_total: float
    x_in: float = 0.0

    y_out: Optional[float] = None
    x_out: Optional[float] = None
    fraction_removed_from_gas: Optional[float] = None

    mode: str = "absorption"  # "absorption" or "stripping"


# ============================================================
# ---------------- Solute-Free Flows -------------------------
# ============================================================

def compute_solutefree_flows(
    V_total: float,
    y_in: float,
    L_total: float,
    x_in: float,
) -> Tuple[float, float]:

    _vp("V_total", V_total)
    _vp("L_total", L_total)
    _vf("y_in", y_in)
    _vf("x_in", x_in)

    Vp = V_total * (1.0 - y_in)
    Lp = L_total * (1.0 - x_in)

    _vp("V_prime", Vp)
    _vp("L_prime", Lp)

    return Vp, Lp


# ============================================================
# ---------------- Target Handling ---------------------------
# ============================================================

def infer_y_out(
    y_in: float,
    y_out: Optional[float],
    fraction_removed: Optional[float],
) -> float:

    _vf("y_in", y_in)

    if y_out is None and fraction_removed is None:
        raise ValueError("Must provide y_out or fraction_removed_from_gas.")

    if y_out is not None:
        _vf("y_out", y_out)
        return y_out

    if not (0.0 <= fraction_removed <= 1.0):
        raise ValueError("fraction_removed must satisfy 0 <= value <= 1.")

    Y_in = Y_from_y(y_in)
    Y_out = (1.0 - fraction_removed) * Y_in
    return y_from_Y(Y_out)


# ============================================================
# ----------- Analytical Kremser (General Form) --------------
# ============================================================

def kremser_N_general(
    y_in: float,
    y_out: float,
    x_in: float,
    m: float,
    Lp: float,
    Vp: float,
) -> float:

    _vp("m", m)
    _vp("Lp", Lp)
    _vp("Vp", Vp)
    _vf("y_in", y_in)
    _vf("y_out", y_out)
    _vf("x_in", x_in)

    A = Lp / (m * Vp)

    num = y_in - m * x_in
    den = y_out - m * x_in

    if num <= 0 or den <= 0:
        raise ValueError("Invalid driving force for analytical form.")

    if abs(A - 1.0) < 1e-12:
        return (num / den) - 1.0

    inside = (num / den) * (1 - 1 / A) + 1 / A
    return math.log(inside) / math.log(A)


# ============================================================
# -------- Ratio-Space Step-Off (Exact Y–X Method) ----------
# ============================================================

def stage_count_ratio_space(
    spec: CountercurrentMultistageSpec,
    max_stages: int = 200,
) -> Dict[str, Any]:

    _vp("m", spec.m)
    _vp("max_stages", float(max_stages))

    Vp, Lp = compute_solutefree_flows(
        spec.V_in_total,
        spec.y_in,
        spec.L_in_total,
        spec.x_in,
    )

    LV = Lp / Vp

    y_out = infer_y_out(
        spec.y_in,
        spec.y_out,
        spec.fraction_removed_from_gas,
    )

    Y_in = Y_from_y(spec.y_in)
    Y1 = Y_from_y(y_out)
    X0 = X_from_x(spec.x_in)

    X_points: List[float] = [X0]
    Y_points: List[float] = [Y1]

    X_prev = X0
    Y_n = Y1
    stage = 0

    while stage < max_stages:

        stage += 1

        X_n = Xeq_from_Y_linear(spec.m, Y_n)
        Y_np1 = Y_n + LV * (X_n - X_prev)

        X_points.append(X_n)
        Y_points.append(Y_np1)

        if Y_np1 >= Y_in:

            Y_before = Y_points[-2]
            Y_after = Y_np1

            if Y_after == Y_before:
                frac = 1.0
            else:
                frac = (Y_in - Y_before) / (Y_after - Y_before)
                frac = max(0.0, min(1.0, frac))

            N_float = (stage - 1) + frac

            return {
                "N_theoretical": N_float,
                "N_ceiling": int(math.ceil(N_float - 1e-12)),
                "A": Lp / (spec.m * Vp),
                "V_prime": Vp,
                "L_prime": Lp,
                "X_points": X_points,
                "Y_points": Y_points,
            }

        X_prev = X_n
        Y_n = Y_np1

    raise RuntimeError("Exceeded max_stages; check feasibility.")


# ============================================================
# -------------------- Master Solver -------------------------
# ============================================================

def solve_countercurrent_multistage(
    spec: CountercurrentMultistageSpec,
    method: str = "analytical",
) -> Dict[str, Any]:

    Vp, Lp = compute_solutefree_flows(
        spec.V_in_total,
        spec.y_in,
        spec.L_in_total,
        spec.x_in,
    )

    y_out = infer_y_out(
        spec.y_in,
        spec.y_out,
        spec.fraction_removed_from_gas,
    )

    A = Lp / (spec.m * Vp)

    results: Dict[str, Any] = {
        "V_prime": Vp,
        "L_prime": Lp,
        "absorption_factor_A": A,
        "y_in": spec.y_in,
        "y_out": y_out,
        "x_in": spec.x_in,
        "m": spec.m,
    }

    if method == "analytical":
        N = kremser_N_general(
            spec.y_in,
            y_out,
            spec.x_in,
            spec.m,
            Lp,
            Vp,
        )
        results["N_theoretical"] = N
        results["N_ceiling"] = int(math.ceil(N - 1e-12))

    elif method == "ratio_stepoff":
        ratio_results = stage_count_ratio_space(spec)
        results.update(ratio_results)

    else:
        raise ValueError("method must be 'analytical' or 'ratio_stepoff'.")

    return results
# ============================================================
# Tray absorber workflow: Lmin, L = factor*Lmin, x–y step-off
# (added to countercurrent_multistage_absorption_stripping_base.py)
# ============================================================

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import math


@dataclass(frozen=True)
class TrayAbsorberLinearEqSpec:
    """
    Countercurrent tray absorber, dilute, linear equilibrium y = m x.

    Adds the standard "L = factor * Lmin" design workflow:
      - Compute y_out from recovery (defined on solute-free gas ratio Y)
      - Compute V' from inlet gas, L'min from pinch at top
      - Choose L' = factor * L'min
      - Compute A = L'/(m V')
      - Compute N (analytical) and N (graphical x–y step-off), plus step points

    Inputs:
      V_in_total : inlet gas total flow
      y_in       : inlet gas solute mole fraction
      m          : equilibrium slope
      x_in       : inlet liquid solute mole fraction (often 0)
      recovery   : fraction of inlet solute removed from gas (0..1), defined on Y
      L_factor_over_min : multiplier on L'min
    """
    V_in_total: float
    y_in: float
    m: float
    L_in_total: float
    x_in: float = 0.0
    recovery: float = 0.0
    L_factor_over_min: float = 1.0
    max_trays_graphical: int = 200


def y_out_from_recovery_Ybasis(y_in: float, recovery: float) -> float:
    _vf("y_in", y_in)
    if not (0.0 <= recovery <= 1.0):
        raise ValueError("recovery must satisfy 0 <= recovery <= 1.")
    Y_in = Y_from_y(y_in)
    Y_out = (1.0 - recovery) * Y_in
    return y_from_Y(Y_out)


def compute_Lmin_prime_pinch_top(
    V_prime: float,
    y_in: float,
    y_out: float,
    m: float,
    x_in: float,
) -> Dict[str, float]:
    """
    Pinch at top: lean outlet gas (y_out) and rich outlet liquid at equilibrium with inlet gas (y_in).
      x1* = y_in / m

    Uses solute-free ratios:
      L'min = V' (Y1 - Y2) / (X1* - X2)
    """
    _vp("V_prime", V_prime)
    _vf("y_in", y_in)
    _vf("y_out", y_out)
    _vp("m", m)
    _vf("x_in", x_in)

    x1_star = y_in / m
    if not (0.0 <= x1_star < 1.0):
        raise ValueError("x1* = y_in/m must satisfy 0 <= x1* < 1.")
    if x1_star <= x_in:
        raise ValueError("x1* <= x_in; pinch/minimum-L invalid for these inputs.")

    X1_star = X_from_x(x1_star)
    X2 = X_from_x(x_in)
    Y1 = Y_from_y(y_in)
    Y2 = Y_from_y(y_out)

    denom = X1_star - X2
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


def trays_graphical_stepoff_xy(
    y_in: float,
    y_out: float,
    x_in: float,
    m: float,
    Lprime_over_Vprime: float,
    max_trays: int = 200,
) -> Dict[str, Any]:
    """
    x–y McCabe-Thiele step-off for absorber with linear equilibrium y = m x
    and dilute straight operating line:
      y = y_out + (L'/V') (x - x_in)

    Returns a list of points for plotting the stepping.
    """
    _vf("y_in", y_in)
    _vf("y_out", y_out)
    _vf("x_in", x_in)
    _vp("m", m)
    _vp("Lprime_over_Vprime", Lprime_over_Vprime)
    _vp("max_trays", float(max_trays))

    pts: List[Tuple[float, float]] = []
    trays = 0

    y_curr = y_out
    pts.append((x_in, y_out))

    if y_curr >= y_in:
        return {"N_float": 0.0, "N_ceiling": 0, "points": pts}

    for _ in range(1, max_trays + 1):
        # horizontal to equilibrium
        x_eq = y_curr / m
        pts.append((x_eq, y_curr))

        # vertical to operating line
        y_next = y_out + Lprime_over_Vprime * (x_eq - x_in)
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
    _vp("L_in_total", spec.L_in_total)
    _vf("y_in", spec.y_in)
    _vf("x_in", spec.x_in)
    _vp("m", spec.m)
    if not (0.0 <= spec.recovery <= 1.0):
        raise ValueError("recovery must satisfy 0 <= recovery <= 1.")
    _vp("L_factor_over_min", spec.L_factor_over_min)

    # Target y_out from Y-basis recovery
    y_out = y_out_from_recovery_Ybasis(spec.y_in, spec.recovery)

    # Solute-free flows
    Vp, _Lp_in = compute_solutefree_flows(spec.V_in_total, spec.y_in, spec.L_in_total, spec.x_in)

    # Minimum solvent (solute-free) from pinch at top
    pinch = compute_Lmin_prime_pinch_top(Vp, spec.y_in, y_out, spec.m, spec.x_in)
    Lmin_prime = pinch["Lmin_prime"]

    # Design solvent
    Lp = spec.L_factor_over_min * Lmin_prime
    LV = Lp / Vp
    A = Lp / (spec.m * Vp)

    # Analytical trays (general closed form already used in this file)
    N_analytical = kremser_N_general(spec.y_in, y_out, spec.x_in, spec.m, Lp, Vp)

    # Graphical trays + step points in x–y
    N_graph = trays_graphical_stepoff_xy(
        y_in=spec.y_in,
        y_out=y_out,
        x_in=spec.x_in,
        m=spec.m,
        Lprime_over_Vprime=LV,
        max_trays=spec.max_trays_graphical,
    )

    x1_star = pinch["x1_star"]
    op_line = {
        "slope_Lprime_over_Vprime": LV,
        "point_top": (spec.x_in, y_out),
        "point_bottom_at_x1_star": (x1_star, y_out + LV * (x1_star - spec.x_in)),
    }

    return {
        "inlet": {"V_in_total": spec.V_in_total, "L_in_total": spec.L_in_total, "y_in": spec.y_in, "x_in": spec.x_in},
        "target": {"recovery": spec.recovery, "y_out": y_out},
        "equilibrium": {"m": spec.m},
        "solute_free": {"V_prime": Vp, "Lmin_prime": Lmin_prime, "L_prime": Lp, "L_over_V": LV, "A": A},
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