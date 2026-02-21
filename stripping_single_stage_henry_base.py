# bank/stripping_single_stage_henry_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from scipy.optimize import brentq


@dataclass(frozen=True)
class SingleStageHenryStrippingSpec:
    """
    Single-stage equilibrium contact for STRIPPING (gas removes solute A from liquid),
    using the same slide-style notation and Henry's law:

        pA = H xA
        yA = pA / P

    Solute-free ratios:
        X = xA / (1 - xA)   = (moles A in liquid) / (moles solvent)
        Y = yA / (1 - yA)   = (moles A in gas)    / (moles inert gas)

    Solute-free flowrates (assumed constant through the stage):
        L' = moles of solute-free liquid (solvent)
        V' = moles of solute-free gas (inert)

    Single-stage solute balance in ratio form:
        L' X_in + V' Y_in = L' X_out + V' Y_out

    STRIPPING convention:
      - Liquid inlet is typically solute-containing (xA_in > 0)
      - Gas inlet is typically lean (yA_in ~ 0)
      - Net transfer is from liquid -> gas (nA_stripped_to_gas > 0)

    Units:
      - P_total and H must be in the same pressure units.
      - Amounts may be mol, kmol, kgmol, etc. (consistent basis).
    """
    P_total: float                 # total pressure
    H: float                       # Henry constant (same pressure units as P_total)

    # Gas inlet (total including solute + inert)
    V_in_total: float              # total inlet gas
    yA_in: Optional[float] = None  # inlet solute mole fraction in gas
    pA_in: Optional[float] = None  # inlet solute partial pressure in gas (same units as P_total)

    # Liquid inlet (total including solute + solvent)
    L_in_total: float = 0.0        # total inlet liquid
    xA_in: float = 0.0             # inlet solute mole fraction in liquid


def _validate_positive(name: str, val: float, allow_zero: bool = False) -> None:
    if allow_zero:
        if val < 0:
            raise ValueError(f"{name} must be >= 0.")
    else:
        if val <= 0:
            raise ValueError(f"{name} must be > 0.")


def _infer_y_from_p(P_total: float, y: Optional[float], p: Optional[float]) -> float:
    if y is None and p is None:
        raise ValueError("Provide either yA_in or pA_in for the inlet gas.")
    if y is not None and p is not None:
        if abs(p - y * P_total) / max(1.0, abs(p)) > 1e-6:
            raise ValueError("Provided yA_in and pA_in are inconsistent with P_total.")
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


def solve_single_stage_henry_stripping(spec: SingleStageHenryStrippingSpec) -> Dict[str, Any]:
    """
    Solves outlet amounts and compositions for a single equilibrium stripping stage.

    Unknown chosen as xA_out; then:
        yA_out = (H/P) * xA_out
        X_out  = xA_out/(1-xA_out)
        Y_out  = yA_out/(1-yA_out)

    Apply solute balance in ratio form:
        L' X_in + V' Y_in = L' X_out + V' Y_out

    Returns dict with inlet/outlet and stripped amount.
    """
    _validate_positive("P_total", spec.P_total)
    _validate_positive("H", spec.H)
    _validate_positive("V_in_total", spec.V_in_total)
    _validate_positive("L_in_total", spec.L_in_total)
    if not (0.0 <= spec.xA_in < 1.0):
        raise ValueError("xA_in must satisfy 0 <= xA_in < 1.")

    yA_in = _infer_y_from_p(spec.P_total, spec.yA_in, spec.pA_in)
    if not (0.0 <= yA_in < 1.0):
        raise ValueError("yA_in must satisfy 0 <= yA_in < 1.")

    # Solute-free flowrates (constant)
    V_prime = spec.V_in_total * (1.0 - yA_in)      # inert gas
    L_prime = spec.L_in_total * (1.0 - spec.xA_in) # solvent

    _validate_positive("V_prime (inert gas)", V_prime)
    _validate_positive("L_prime (solvent)", L_prime)

    Y_in = _Y_from_y(yA_in)
    X_in = _X_from_x(spec.xA_in)

    # Equilibrium slope in y-x: y = (H/P) x
    m = spec.H / spec.P_total

    # Solve for x_out from ratio-balance + equilibrium
    def balance_residual(x_out: float) -> float:
        if x_out < 0.0:
            x_out = 0.0
        if x_out >= 1.0:
            return 1e300

        y_out = m * x_out
        if y_out >= 1.0:
            return 1e300

        X_out = _X_from_x(x_out)
        Y_out = _Y_from_y(y_out)

        return (L_prime * X_out + V_prime * Y_out) - (L_prime * X_in + V_prime * Y_in)

    # Physical upper bound so y_out < 1
    x_lo = 0.0
    x_hi = min(0.999999999999, 0.999999999999 / m) if m > 0 else 0.999999999999

    f_lo = balance_residual(x_lo + 1e-16)
    f_hi = balance_residual(x_hi)

    if f_lo * f_hi > 0:
        xA_out = x_lo if abs(f_lo) <= abs(f_hi) else x_hi
        method = "no_bracket_used_best_endpoint"
    else:
        xA_out = float(brentq(balance_residual, x_lo + 1e-16, x_hi, maxiter=200, xtol=1e-14))
        method = "brentq"

    yA_out = m * xA_out
    X_out = _X_from_x(xA_out)
    Y_out = _Y_from_y(yA_out)

    # Convert ratios to solute moles
    nA_l_in = L_prime * X_in
    nA_g_in = V_prime * Y_in

    nA_l_out = L_prime * X_out
    nA_g_out = V_prime * Y_out

    # Totals
    L_out_total = L_prime + nA_l_out
    V_out_total = V_prime + nA_g_out

    # Stripped amount (positive means liquid -> gas)
    nA_stripped_to_gas = nA_g_out - nA_g_in

    return {
        "method": method,
        "inlet": {
            "P_total": spec.P_total,
            "H": spec.H,
            "V_in_total": spec.V_in_total,
            "L_in_total": spec.L_in_total,
            "V_prime": V_prime,
            "L_prime": L_prime,
            "yA_in": yA_in,
            "xA_in": spec.xA_in,
            "Y_in": Y_in,
            "X_in": X_in,
            "nA_in_gas": nA_g_in,
            "nA_in_liq": nA_l_in,
            "nA_in_total": nA_g_in + nA_l_in,
        },
        "outlet": {
            "xA_out": xA_out,
            "yA_out": yA_out,
            "X_out": X_out,
            "Y_out": Y_out,
            "pA_out": yA_out * spec.P_total,
            "L_out_total": L_out_total,
            "V_out_total": V_out_total,
            "nA_out_gas": nA_g_out,
            "nA_out_liq": nA_l_out,
            "nA_out_total": nA_g_out + nA_l_out,
            "nA_stripped_to_gas": nA_stripped_to_gas,
        },
    }