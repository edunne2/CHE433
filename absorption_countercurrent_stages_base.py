# bank/absorption_countercurrent_stages_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import math


@dataclass(frozen=True)
class CountercurrentAbsorptionSpec:
    """
    Countercurrent theoretical-stage absorber with linear equilibrium:

        yA = m xA

    Stagewise model uses solute-free ratios (slide style):
        X = x/(1-x)
        Y = y/(1-y)

    Exact equilibrium in ratio coordinates:
        y = m x
        x = X/(1+X)
        y = m*X/(1+X)
        Y* = y/(1-y) = [m X] / [1 + X(1-m)]

    Solute-free flowrates (assumed constant):
        V' = total_gas_in * (1 - y_in)        (inert gas)
        L' = total_liq_in * (1 - x_in_liq)    (solvent)

    Stage balance (solute) in ratio form for stage n:
        V' (Y_{n+1} - Y_n) = L' (X_n - X_{n-1})

    Countercurrent boundary conditions:
        Gas enters bottom:  Y_{N+1} = Y_in
        Gas exits top:      Y_1     = Y_out   (set from required removal or given)
        Liquid enters top:  X_0     = X_in_liq (often 0 for pure solvent)

    Units: consistent molar flow basis (mol, kmol, kgmol, etc.).
    """
    m: float                    # equilibrium slope y = m x
    V_in_total: float           # total inlet gas flow (A + inert)
    yA_in: float                # inlet gas mole fraction of solute A
    L_in_total: float           # total inlet liquid flow (A + solvent, usually solvent only)
    xA_in: float = 0.0          # inlet liquid mole fraction of solute A (0 for pure solvent)

    # Specify separation target either as outlet yA_out or fraction_absorbed
    yA_out: Optional[float] = None
    fraction_absorbed: Optional[float] = None  # e.g., 0.90 means 90% removed from gas


def _validate_fraction(name: str, v: float) -> None:
    if not (0.0 <= v < 1.0):
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1.")


def _validate_positive(name: str, v: float) -> None:
    if v <= 0:
        raise ValueError(f"{name} must be > 0.")


def Y_from_y(y: float) -> float:
    _validate_fraction("y", y)
    return y / (1.0 - y)


def X_from_x(x: float) -> float:
    _validate_fraction("x", x)
    return x / (1.0 - x)


def y_from_Y(Y: float) -> float:
    if Y < 0:
        raise ValueError("Y must be >= 0.")
    return Y / (1.0 + Y)


def x_from_X(X: float) -> float:
    if X < 0:
        raise ValueError("X must be >= 0.")
    return X / (1.0 + X)


def Yeq_from_X_linear(m: float, X: float) -> float:
    """
    Exact equilibrium curve in solute-free ratio coordinates for y = m x.
    """
    if X < 0:
        raise ValueError("X must be >= 0.")
    if m <= 0:
        raise ValueError("m must be > 0.")
    denom = 1.0 + X * (1.0 - m)
    if denom <= 0:
        # corresponds to y -> 1 (not physical for absorption design)
        return float("inf")
    return (m * X) / denom


def Xeq_from_Y_linear(m: float, Y: float) -> float:
    """
    Invert exact equilibrium: given Y, compute X such that Y = Yeq(X).

    Start with:
        y = Y/(1+Y)
        x = y/m
        X = x/(1-x)

    This is exact for y = m x.
    """
    if Y < 0:
        raise ValueError("Y must be >= 0.")
    if m <= 0:
        raise ValueError("m must be > 0.")
    y = y_from_Y(Y)
    x = y / m
    if x >= 1.0:
        return float("inf")
    return X_from_x(x)


def compute_solutefree_flows(V_in_total: float, yA_in: float, L_in_total: float, xA_in: float) -> Tuple[float, float]:
    _validate_positive("V_in_total", V_in_total)
    _validate_positive("L_in_total", L_in_total)
    _validate_fraction("yA_in", yA_in)
    _validate_fraction("xA_in", xA_in)

    V_prime = V_in_total * (1.0 - yA_in)
    L_prime = L_in_total * (1.0 - xA_in)

    if V_prime <= 0:
        raise ValueError("Computed V' (inert gas) must be > 0.")
    if L_prime <= 0:
        raise ValueError("Computed L' (solvent) must be > 0.")
    return V_prime, L_prime


def infer_yA_out(yA_in: float, yA_out: Optional[float], fraction_absorbed: Optional[float]) -> float:
    _validate_fraction("yA_in", yA_in)

    if yA_out is None and fraction_absorbed is None:
        raise ValueError("Provide either yA_out or fraction_absorbed.")
    if yA_out is not None and fraction_absorbed is not None:
        _validate_fraction("yA_out", yA_out)
        if not (0.0 <= fraction_absorbed <= 1.0):
            raise ValueError("fraction_absorbed must satisfy 0 <= fraction_absorbed <= 1.")
        # accept yA_out; ignore fraction_absorbed
        return yA_out
    if yA_out is not None:
        _validate_fraction("yA_out", yA_out)
        return yA_out

    if not (0.0 <= fraction_absorbed <= 1.0):
        raise ValueError("fraction_absorbed must satisfy 0 <= fraction_absorbed <= 1.")

    # Absorption fraction defined on solute moles in gas; use Y scaling:
    Y_in = Y_from_y(yA_in)
    Y_out = (1.0 - fraction_absorbed) * Y_in
    return y_from_Y(Y_out)


def stage_count_graphical_stepoff(
    spec: CountercurrentAbsorptionSpec,
    max_stages: int = 200,
) -> Dict[str, Any]:
    """
    Graphical/step-off equivalent (McCabe-Thiele in Y-X space) using exact ratio relations.

    Procedure:
      1) Compute Y_in from y_in.
      2) Compute Y_out from target (y_out or fraction_absorbed).
      3) Given X0 (from x_in), step:
           X_n = Xeq(Y_n)
           Y_{n+1} = Y_n + (L'/V') (X_n - X_{n-1})
         starting with X_0 and Y_1.
      4) Stop when Y_{n+1} >= Y_in; interpolate final fractional stage.

    Returns:
      - N_theoretical (float), N_ceiling (int)
      - arrays for plotting: X_points, Y_points (Y points are Y_1..Y_{k+1})
    """
    _validate_positive("max_stages", float(max_stages))
    if spec.m <= 0:
        raise ValueError("m must be > 0.")

    V_prime, L_prime = compute_solutefree_flows(spec.V_in_total, spec.yA_in, spec.L_in_total, spec.xA_in)
    LV = L_prime / V_prime

    y_out = infer_yA_out(spec.yA_in, spec.yA_out, spec.fraction_absorbed)

    Y_in = Y_from_y(spec.yA_in)   # Y_{N+1}
    Y_1 = Y_from_y(y_out)         # Y_1
    X_0 = X_from_x(spec.xA_in)    # X_0

    # stepping storage
    X_list: List[float] = [X_0]
    Y_list: List[float] = [Y_1]   # store Y_1 first
    stage_index = 0

    X_prev = X_0
    Y_n = Y_1

    if Y_n > Y_in:
        return {
            "N_theoretical": 0.0,
            "N_ceiling": 0,
            "V_prime": V_prime,
            "L_prime": L_prime,
            "Y_in": Y_in,
            "Y_out": Y_1,
            "X_in": X_0,
            "X_points": X_list,
            "Y_points": Y_list,
        }

    while stage_index < max_stages:
        stage_index += 1

        X_n = Xeq_from_Y_linear(spec.m, Y_n)
        if not math.isfinite(X_n):
            raise ValueError("Equilibrium inversion produced non-finite X; check m and compositions.")

        Y_np1 = Y_n + LV * (X_n - X_prev)

        X_list.append(X_n)
        Y_list.append(Y_np1)

        if Y_np1 >= Y_in:
            # fractional final stage between (stage_index-1) and stage_index
            if stage_index == 1:
                Y_before = Y_1
            else:
                Y_before = Y_list[-2]  # Y_n
            Y_after = Y_np1
            if Y_after == Y_before:
                frac = 1.0
            else:
                frac = (Y_in - Y_before) / (Y_after - Y_before)
                frac = max(0.0, min(1.0, frac))
            N_float = (stage_index - 1) + frac
            return {
                "N_theoretical": N_float,
                "N_ceiling": int(math.ceil(N_float - 1e-12)),
                "V_prime": V_prime,
                "L_prime": L_prime,
                "Y_in": Y_in,
                "Y_out": Y_1,
                "X_in": X_0,
                "X_points": X_list,
                "Y_points": Y_list,
            }

        X_prev = X_n
        Y_n = Y_np1

    raise RuntimeError("Exceeded max_stages without reaching inlet gas composition; increase max_stages or check feasibility.")


def kremser_required_stages_absorption(
    yA_in: float,
    fraction_absorbed: float,
    L_prime: float,
    V_prime: float,
    m: float,
    N_max: int = 200,
) -> Dict[str, Any]:
    """
    Kremser equation for absorption with pure solvent (x_in ~ 0), dilute assumption.
    Uses absorption factor:
        A = L' / (m V')

    For A != 1:
        Y1 / Y_{N+1} = (A - 1) / (A^{N+1} - 1)

    Target:
        Y1 = (1 - fraction_absorbed) * Y_{N+1}

    Returns smallest integer N such that removal target is met.
    """
    _validate_fraction("yA_in", yA_in)
    if not (0.0 <= fraction_absorbed <= 1.0):
        raise ValueError("fraction_absorbed must satisfy 0 <= fraction_absorbed <= 1.")
    _validate_positive("L_prime", L_prime)
    _validate_positive("V_prime", V_prime)
    _validate_positive("m", m)

    A = L_prime / (m * V_prime)
    Y_in = Y_from_y(yA_in)
    Y_out_target = (1.0 - fraction_absorbed) * Y_in

    ratio_target = Y_out_target / Y_in if Y_in > 0 else 0.0

    if Y_in == 0:
        return {"A": A, "N": 0, "ratio": 0.0}

    if abs(A - 1.0) < 1e-12:
        # Special case A ~ 1: use limit form
        # For A -> 1, Kremser gives: Y1/Y_{N+1} = 1/(N+1)
        for N in range(1, N_max + 1):
            ratio = 1.0 / (N + 1.0)
            if ratio <= ratio_target + 1e-15:
                return {"A": A, "N": N, "ratio": ratio}
        return {"A": A, "N": None, "ratio": None}

    for N in range(1, N_max + 1):
        ratio = (A - 1.0) / (A ** (N + 1) - 1.0)
        if ratio <= ratio_target + 1e-15:
            return {"A": A, "N": N, "ratio": ratio}

    return {"A": A, "N": None, "ratio": None}