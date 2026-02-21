# bank/absorption_single_stage_henry_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from scipy.optimize import brentq


@dataclass(frozen=True)
class SingleStageHenryAbsorptionSpec:
    """
    Single-stage equilibrium contact (gas + liquid), using the notation in the slides:

    Henry's law:
        pA = H xA
        yA = pA / P

    Solute-free ratios (slides):
        X = xA / (1 - xA)   = (moles A in liquid) / (moles solvent)
        Y = yA / (1 - yA)   = (moles A in gas)    / (moles inert gas)

    Inert streams assumed constant through the stage:
        L' = moles of solute-free absorbent (solvent, e.g., water)
        V' = moles of solute-free gas (inert, e.g., air)

    Single-stage solute balance in ratio form (with constant L', V'):
        L' X0 + V' Y2 = L' X1 + V' Y1

    Inputs accept either inlet yA2 or inlet pA2.
    Units:
      - P_total and H must be in the same pressure units.
      - Flows can be mol, kmol, kgmol, etc. (consistent basis).
    """
    P_total: float                 # total pressure
    H: float                       # Henry constant (same pressure units as P_total)

    # Gas inlet (total including solute + inert)
    V2_total: float                # total inlet gas flow
    yA2: Optional[float] = None    # inlet solute mole fraction in gas
    pA2: Optional[float] = None    # inlet solute partial pressure in gas (same units as P_total)

    # Liquid inlet (solute-free absorbent, e.g., pure water)
    L0_total: float = 0.0          # total inlet liquid flow (assumed solvent only unless xA0 given)
    xA0: float = 0.0               # inlet solute mole fraction in liquid (default 0 for pure solvent)


def _validate_positive(name: str, val: float, allow_zero: bool = False) -> None:
    if allow_zero:
        if val < 0:
            raise ValueError(f"{name} must be >= 0.")
    else:
        if val <= 0:
            raise ValueError(f"{name} must be > 0.")


def _infer_y_from_p(P_total: float, y: Optional[float], p: Optional[float]) -> float:
    if y is None and p is None:
        raise ValueError("Provide either yA2 or pA2 for the inlet gas.")
    if y is not None and p is not None:
        # sanity check
        if abs(p - y * P_total) / max(1.0, abs(p)) > 1e-6:
            raise ValueError("Provided yA2 and pA2 are inconsistent with P_total.")
        return float(y)
    if y is not None:
        return float(y)
    return float(p) / P_total


def _X_from_x(x: float) -> float:
    if not (0.0 <= x < 1.0):
        raise ValueError("x must satisfy 0 <= x < 1.")
    return x / (1.0 - x)


def _Y_from_y(y: float) -> float:
    if not (0.0 <= y < 1.0):
        raise ValueError("y must satisfy 0 <= y < 1.")
    return y / (1.0 - y)


def _x_from_X(X: float) -> float:
    if X < 0:
        raise ValueError("X must be >= 0.")
    return X / (1.0 + X)


def _y_from_Y(Y: float) -> float:
    if Y < 0:
        raise ValueError("Y must be >= 0.")
    return Y / (1.0 + Y)


def solve_single_stage_henry_absorption(spec: SingleStageHenryAbsorptionSpec) -> Dict[str, Any]:
    """
    Solves for outlet phase amounts and compositions for a single equilibrium mixer.

    Returns:
      - inlet: V', L', yA2, xA0, Y2, X0, nA_in
      - outlet: xA1, yA1, X1, Y1, V1_total, L1_total, nA_g_out, nA_l_out, nA_absorbed
    """
    _validate_positive("P_total", spec.P_total)
    _validate_positive("H", spec.H)
    _validate_positive("V2_total", spec.V2_total)
    _validate_positive("L0_total", spec.L0_total, allow_zero=False)
    if not (0.0 <= spec.xA0 < 1.0):
        raise ValueError("xA0 must satisfy 0 <= xA0 < 1.")

    yA2 = _infer_y_from_p(spec.P_total, spec.yA2, spec.pA2)
    if not (0.0 <= yA2 < 1.0):
        raise ValueError("yA2 must satisfy 0 <= yA2 < 1.")

    # Inerts (slides: V' constant inert gas, L' constant solvent)
    V_prime = spec.V2_total * (1.0 - yA2)     # inert gas moles
    L_prime = spec.L0_total * (1.0 - spec.xA0)  # solvent moles (≈ L0_total if xA0=0)

    _validate_positive("V_prime (inert gas)", V_prime)
    _validate_positive("L_prime (solvent)", L_prime)

    Y2 = _Y_from_y(yA2)
    X0 = _X_from_x(spec.xA0)

    # Equilibrium: yA1*P = H*xA1  => yA1 = (H/P) xA1
    m = spec.H / spec.P_total

    # Solve for xA1 using the ratio-form solute balance:
    #   L' X0 + V' Y2 = L' X1 + V' Y1
    # with:
    #   X1 = x/(1-x)
    #   y = m x
    #   Y1 = y/(1-y)
    def balance_residual(x: float) -> float:
        if x <= 0:
            x = 0.0
        y = m * x
        if y >= 1.0:
            return 1e300
        X1 = _X_from_x(x)
        Y1 = _Y_from_y(y)
        return (L_prime * X1 + V_prime * Y1) - (L_prime * X0 + V_prime * Y2)

    # Bracket x in [0, x_max) where y = m x < 1
    x_lo = 0.0
    x_hi = min(0.999999999999, 0.999999999999 / m) if m > 0 else 0.999999999999

    f_lo = balance_residual(x_lo + 1e-16)
    f_hi = balance_residual(x_hi)

    if f_lo * f_hi > 0:
        # No root in physical interval; choose endpoint with smaller |residual|
        xA1 = x_lo if abs(f_lo) <= abs(f_hi) else x_hi
        method = "no_bracket_used_best_endpoint"
    else:
        xA1 = float(brentq(balance_residual, x_lo + 1e-16, x_hi, maxiter=200, xtol=1e-14))
        method = "brentq"

    yA1 = m * xA1
    X1 = _X_from_x(xA1)
    Y1 = _Y_from_y(yA1)

    # Convert ratios back to actual solute moles in each phase:
    nA_l_out = L_prime * X1
    nA_g_out = V_prime * Y1

    # Totals:
    L1_total = L_prime + nA_l_out
    V1_total = V_prime + nA_g_out

    # Inlet solute (from ratios):
    nA_in_gas = V_prime * Y2
    nA_in_liq = L_prime * X0
    nA_in_total = nA_in_gas + nA_in_liq

    nA_out_total = nA_g_out + nA_l_out
    nA_absorbed = nA_l_out - nA_in_liq  # positive for absorption

    return {
        "method": method,
        "inlet": {
            "P_total": spec.P_total,
            "H": spec.H,
            "V2_total": spec.V2_total,
            "L0_total": spec.L0_total,
            "V_prime": V_prime,
            "L_prime": L_prime,
            "yA2": yA2,
            "xA0": spec.xA0,
            "Y2": Y2,
            "X0": X0,
            "nA_in_total": nA_in_total,
            "nA_in_gas": nA_in_gas,
            "nA_in_liq": nA_in_liq,
        },
        "outlet": {
            "xA1": xA1,
            "yA1": yA1,
            "X1": X1,
            "Y1": Y1,
            "L1_total": L1_total,
            "V1_total": V1_total,
            "nA_l_out": nA_l_out,
            "nA_g_out": nA_g_out,
            "nA_out_total": nA_out_total,
            "nA_absorbed_to_liquid": nA_absorbed,
            "pA1": yA1 * spec.P_total,
        },
    }