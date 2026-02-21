# bank/single_stage_equilibrium_contact_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
import math


def _vp(name: str, v: float) -> None:
    if v is None or v <= 0:
        raise ValueError(f"{name} must be > 0.")


def _vf(name: str, v: float) -> None:
    if v is None or not (0.0 <= v < 1.0):
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1.")


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


@dataclass(frozen=True)
class SingleStageEqContactSpec:
    """
    Single-stage equilibrium contact (mixer-settler / one equilibrium stage).

    Two equations (mole-fraction basis, dilute is fine):
      (1) Overall solute balance:
            V (y_in - y_out) = L (x_out - x_in)
      (2) Exit streams in equilibrium:
            y_out = y_eq(x_out)

    Default equilibrium is linear: y = m x, but you can pass a custom y_eq(x).

    Optional solute-free exact balance (recommended if not extremely dilute):
      Use_XY_balance=True replaces (1) with:
            V' (Y_in - Y_out) = L' (X_out - X_in)
      where:
            V' = V_total (1 - y_in)         (inert gas)
            L' = L_total (1 - x_in)         (solvent)
            Y = y/(1-y),  X = x/(1-x)

      Equilibrium still applied in y-x at the outlet:
            y_out = y_eq(x_out)
      (then Y_out computed from y_out, X_out from x_out)

    Supports both absorption and stripping with the same equations:
      - Absorption: typically y_out < y_in
      - Stripping: typically x_out < x_in (solute moves from liquid to gas)
    """
    V_total: float
    L_total: float
    y_in: float
    x_in: float
    m: Optional[float] = None
    use_XY_balance: bool = False


def solve_single_stage_equilibrium_contact(
    spec: SingleStageEqContactSpec,
    y_eq: Optional[Callable[[float], float]] = None,
) -> Dict[str, Any]:
    _vp("V_total", spec.V_total)
    _vp("L_total", spec.L_total)
    _vf("y_in", spec.y_in)
    _vf("x_in", spec.x_in)

    if y_eq is None:
        if spec.m is None:
            raise ValueError("Provide either y_eq(x) or m for linear equilibrium.")
        _vp("m", spec.m)
        y_eq = lambda x: spec.m * x  # type: ignore[assignment]

    # Closed-form for linear equilibrium only
    is_linear = (spec.m is not None) and (y_eq is not None)

    if spec.use_XY_balance:
        # V'(Yin - Yout) = L'(Xout - Xin) with yout = y_eq(xout)
        Vp = spec.V_total * (1.0 - spec.y_in)
        Lp = spec.L_total * (1.0 - spec.x_in)
        _vp("V_prime", Vp)
        _vp("L_prime", Lp)

        Yin = Y_from_y(spec.y_in)
        Xin = X_from_x(spec.x_in)

        if spec.m is not None:
            m = spec.m
            # yout = m xout; solve in terms of Xout or xout.
            # Use xout unknown directly; compute yout; then solve V'(Yin - Yout)=L'(Xout - Xin)
            # Nonlinear due to transforms; solve by robust bisection.
            def f(xout: float) -> float:
                yout = m * xout
                if not (0.0 <= yout < 1.0):
                    return 1e9
                Yout = Y_from_y(yout)
                Xout = X_from_x(xout)
                return Vp * (Yin - Yout) - Lp * (Xout - Xin)

            # bracket xout in [0, 1) with reasonable bounds around x_in
            a = 0.0
            b = min(0.999999, max(spec.x_in, spec.y_in / max(1e-12, spec.m)) * 10.0 + 1e-6)
            fa = f(a)
            fb = f(b)
            if fa * fb > 0:
                # widen
                b = 0.999999
                fb = f(b)
                if fa * fb > 0:
                    raise ValueError("Could not bracket solution for x_out using X-Y balance.")
            for _ in range(120):
                c = 0.5 * (a + b)
                fc = f(c)
                if abs(fc) < 1e-12:
                    a = b = c
                    break
                if fa * fc <= 0:
                    b, fb = c, fc
                else:
                    a, fa = c, fc
            x_out = 0.5 * (a + b)
            y_out = spec.m * x_out
        else:
            raise ValueError("X-Y balance mode requires linear equilibrium (m) in this bank.")

        Vp = spec.V_total * (1.0 - spec.y_in)
        Lp = spec.L_total * (1.0 - spec.x_in)
        out = {
            "basis": "solute_free_XY",
            "flows": {"V_total": spec.V_total, "L_total": spec.L_total, "V_prime": Vp, "L_prime": Lp},
            "inlet": {"y_in": spec.y_in, "x_in": spec.x_in},
            "outlet": {
                "x_out": x_out,
                "y_out": y_out,
                "X_in": X_from_x(spec.x_in),
                "Y_in": Y_from_y(spec.y_in),
                "X_out": X_from_x(x_out),
                "Y_out": Y_from_y(y_out),
            },
            "equilibrium": {"m": spec.m, "y_eq_x_out": y_out},
        }
        return out

    # Simple mole-fraction balance: V(yin-yout)=L(xout-xin), yout=y_eq(xout)
    if spec.m is not None:
        m = spec.m
        # V(yin - m xout)=L(xout - xin) => xout = (V yin + L xin)/(L + V m)
        x_out = (spec.V_total * spec.y_in + spec.L_total * spec.x_in) / (spec.L_total + spec.V_total * m)
        _vf("x_out", x_out)
        y_out = m * x_out
        _vf("y_out", y_out)
    else:
        raise ValueError("Nonlinear equilibrium requires a numerical solver; not implemented in this bank.")

    return {
        "basis": "mole_fraction",
        "flows": {"V_total": spec.V_total, "L_total": spec.L_total},
        "inlet": {"y_in": spec.y_in, "x_in": spec.x_in},
        "outlet": {"x_out": x_out, "y_out": y_out},
        "equilibrium": {"m": spec.m, "y_eq_x_out": y_out},
        "checks": {"overall_balance_residual": spec.V_total * (spec.y_in - y_out) - spec.L_total * (x_out - spec.x_in)},
    }
# ----------------------------
# Henry-law single-stage (ratio form) - added
# ----------------------------

from dataclasses import dataclass
from typing import Optional, Dict, Any
import math


@dataclass(frozen=True)
class SingleStageHenryAbsorptionSpec:
    """
    Single-stage equilibrium contact using Henry's law (slide notation).

    Henry:
        pA = H xA
        yA = pA / P_total
      => yA = (H/P_total) xA  where m = H/P_total

    Solute-free ratios:
        X = x/(1-x)  (solute/solvent)
        Y = y/(1-y)  (solute/inert gas)

    Inerts constant through stage:
        V' = V2_total (1 - yA2)
        L' = L0_total (1 - xA0)

    Stage solute balance (ratio form):
        L' X0 + V' Y2 = L' X1 + V' Y1

    Inlet gas may be given as yA2 or pA2.
    P_total and H must be consistent pressure units.
    """
    P_total: float
    H: float
    V2_total: float
    L0_total: float
    xA0: float = 0.0
    yA2: Optional[float] = None
    pA2: Optional[float] = None


def solve_single_stage_henry_absorption(spec: SingleStageHenryAbsorptionSpec) -> Dict[str, Any]:
    _vp("P_total", spec.P_total)
    _vp("H", spec.H)
    _vp("V2_total", spec.V2_total)
    _vp("L0_total", spec.L0_total)
    _vf("xA0", spec.xA0)

    # infer yA2
    if spec.yA2 is None and spec.pA2 is None:
        raise ValueError("Provide either yA2 or pA2.")
    if spec.yA2 is not None and spec.pA2 is not None:
        # consistency check
        if abs(spec.pA2 - spec.yA2 * spec.P_total) / max(1.0, abs(spec.pA2)) > 1e-6:
            raise ValueError("Provided yA2 and pA2 are inconsistent with P_total.")
        yA2 = float(spec.yA2)
    elif spec.yA2 is not None:
        yA2 = float(spec.yA2)
    else:
        yA2 = float(spec.pA2) / spec.P_total

    _vf("yA2", yA2)

    # solute-free inerts
    V_prime = spec.V2_total * (1.0 - yA2)
    L_prime = spec.L0_total * (1.0 - spec.xA0)
    _vp("V_prime", V_prime)
    _vp("L_prime", L_prime)

    # ratios
    Y2 = Y_from_y(yA2)
    X0 = X_from_x(spec.xA0)

    # Henry slope
    m = spec.H / spec.P_total
    _vp("m", m)

    # residual in xA1 using ratio balance with yA1 = m xA1
    def resid(x: float) -> float:
        if x <= 0.0:
            x = 0.0
        if x >= 1.0:
            return 1e300
        y = m * x
        if y >= 1.0:
            return 1e300
        X1 = X_from_x(x)
        Y1 = Y_from_y(y)
        return (L_prime * X1 + V_prime * Y1) - (L_prime * X0 + V_prime * Y2)

    # bracket x in [0, x_hi) where y=m x < 1
    x_lo = 0.0
    x_hi = min(0.999999999999, 0.999999999999 / m)

    f_lo = resid(x_lo + 1e-18)
    f_hi = resid(x_hi)

    method = "bisection"
    if f_lo * f_hi > 0:
        # no bracket; choose best endpoint (matches your fallback idea)
        xA1 = x_lo if abs(f_lo) <= abs(f_hi) else x_hi
        method = "no_bracket_used_best_endpoint"
    else:
        a, b = x_lo + 1e-18, x_hi
        fa, fb = f_lo, f_hi
        for _ in range(200):
            c = 0.5 * (a + b)
            fc = resid(c)
            if abs(fc) < 1e-14:
                a = b = c
                break
            if fa * fc <= 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        xA1 = 0.5 * (a + b)

    yA1 = m * xA1
    X1 = X_from_x(xA1)
    Y1 = Y_from_y(yA1)

    # solute moles in phases (ratio definition)
    nA_l_out = L_prime * X1
    nA_g_out = V_prime * Y1

    # totals
    L1_total = L_prime + nA_l_out
    V1_total = V_prime + nA_g_out

    # inlet solute
    nA_in_gas = V_prime * Y2
    nA_in_liq = L_prime * X0
    nA_absorbed = nA_l_out - nA_in_liq

    return {
        "method": method,
        "inlet": {
            "P_total": spec.P_total,
            "H": spec.H,
            "m_equiv": m,
            "V2_total": spec.V2_total,
            "L0_total": spec.L0_total,
            "yA2": yA2,
            "pA2": yA2 * spec.P_total,
            "xA0": spec.xA0,
            "V_prime": V_prime,
            "L_prime": L_prime,
            "Y2": Y2,
            "X0": X0,
            "nA_in_gas": nA_in_gas,
            "nA_in_liq": nA_in_liq,
            "nA_in_total": nA_in_gas + nA_in_liq,
        },
        "outlet": {
            "xA1": xA1,
            "yA1": yA1,
            "pA1": yA1 * spec.P_total,
            "X1": X1,
            "Y1": Y1,
            "nA_l_out": nA_l_out,
            "nA_g_out": nA_g_out,
            "nA_out_total": nA_l_out + nA_g_out,
            "nA_absorbed_to_liquid": nA_absorbed,
            "L1_total": L1_total,
            "V1_total": V1_total,
        },
        "checks": {
            "ratio_balance_residual": (L_prime * X1 + V_prime * Y1) - (L_prime * X0 + V_prime * Y2),
        },
    }
# ----------------------------
# Henry-law single-stage STRIPPING (ratio form) - added
# ----------------------------

from dataclasses import dataclass
from typing import Optional, Dict, Any
import math


@dataclass(frozen=True)
class SingleStageHenryStrippingSpec:
    """
    Single-stage equilibrium contact for stripping using Henry's law.

      pA = H xA
      yA = pA/P_total = (H/P_total) xA  where m = H/P_total

    Ratios:
      X = x/(1-x),  Y = y/(1-y)

    Solute-free flows (constant):
      V' = V_in_total (1-y_in)
      L' = L_in_total (1-x_in)

    Ratio-form solute balance:
      L' X_in + V' Y_in = L' X_out + V' Y_out

    Positive stripped-to-gas:
      nA_stripped_to_gas = V' Y_out - V' Y_in
    """
    P_total: float
    H: float
    V_in_total: float
    L_in_total: float
    xA_in: float
    yA_in: Optional[float] = None
    pA_in: Optional[float] = None


def solve_single_stage_henry_stripping(spec: SingleStageHenryStrippingSpec) -> Dict[str, Any]:
    _vp("P_total", spec.P_total)
    _vp("H", spec.H)
    _vp("V_in_total", spec.V_in_total)
    _vp("L_in_total", spec.L_in_total)
    _vf("xA_in", spec.xA_in)

    # infer yA_in
    if spec.yA_in is None and spec.pA_in is None:
        raise ValueError("Provide either yA_in or pA_in.")
    if spec.yA_in is not None and spec.pA_in is not None:
        if abs(spec.pA_in - spec.yA_in * spec.P_total) / max(1.0, abs(spec.pA_in)) > 1e-6:
            raise ValueError("Provided yA_in and pA_in are inconsistent with P_total.")
        yA_in = float(spec.yA_in)
    elif spec.yA_in is not None:
        yA_in = float(spec.yA_in)
    else:
        yA_in = float(spec.pA_in) / spec.P_total

    _vf("yA_in", yA_in)

    # solute-free inerts
    V_prime = spec.V_in_total * (1.0 - yA_in)
    L_prime = spec.L_in_total * (1.0 - spec.xA_in)
    _vp("V_prime", V_prime)
    _vp("L_prime", L_prime)

    Y_in = Y_from_y(yA_in)
    X_in = X_from_x(spec.xA_in)

    m = spec.H / spec.P_total
    _vp("m", m)

    def resid(x_out: float) -> float:
        if x_out <= 0.0:
            x_out = 0.0
        if x_out >= 1.0:
            return 1e300
        y_out = m * x_out
        if y_out >= 1.0:
            return 1e300
        X_out = X_from_x(x_out)
        Y_out = Y_from_y(y_out)
        return (L_prime * X_out + V_prime * Y_out) - (L_prime * X_in + V_prime * Y_in)

    x_lo = 0.0
    x_hi = min(0.999999999999, 0.999999999999 / m)

    f_lo = resid(x_lo + 1e-18)
    f_hi = resid(x_hi)

    method = "bisection"
    if f_lo * f_hi > 0:
        x_out = x_lo if abs(f_lo) <= abs(f_hi) else x_hi
        method = "no_bracket_used_best_endpoint"
    else:
        a, b = x_lo + 1e-18, x_hi
        fa, fb = f_lo, f_hi
        for _ in range(200):
            c = 0.5 * (a + b)
            fc = resid(c)
            if abs(fc) < 1e-14:
                a = b = c
                break
            if fa * fc <= 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        x_out = 0.5 * (a + b)

    y_out = m * x_out
    X_out = X_from_x(x_out)
    Y_out = Y_from_y(y_out)

    nA_l_in = L_prime * X_in
    nA_g_in = V_prime * Y_in
    nA_l_out = L_prime * X_out
    nA_g_out = V_prime * Y_out

    nA_stripped_to_gas = nA_g_out - nA_g_in

    return {
        "method": method,
        "inlet": {
            "P_total": spec.P_total,
            "H": spec.H,
            "m_equiv": m,
            "V_in_total": spec.V_in_total,
            "L_in_total": spec.L_in_total,
            "yA_in": yA_in,
            "pA_in": yA_in * spec.P_total,
            "xA_in": spec.xA_in,
            "V_prime": V_prime,
            "L_prime": L_prime,
            "Y_in": Y_in,
            "X_in": X_in,
            "nA_in_gas": nA_g_in,
            "nA_in_liq": nA_l_in,
            "nA_in_total": nA_g_in + nA_l_in,
        },
        "outlet": {
            "xA_out": x_out,
            "yA_out": y_out,
            "pA_out": y_out * spec.P_total,
            "X_out": X_out,
            "Y_out": Y_out,
            "nA_out_gas": nA_g_out,
            "nA_out_liq": nA_l_out,
            "nA_out_total": nA_g_out + nA_l_out,
            "nA_stripped_to_gas": nA_stripped_to_gas,
            "L_out_total": L_prime + nA_l_out,
            "V_out_total": V_prime + nA_g_out,
        },
        "checks": {
            "ratio_balance_residual": (L_prime * X_out + V_prime * Y_out) - (L_prime * X_in + V_prime * Y_in),
        },
    }