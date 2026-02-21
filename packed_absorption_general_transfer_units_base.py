# bank/packed_absorption_general_transfer_units_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math


# ----------------------------
# validators + utilities
# ----------------------------

def _vp(name: str, v: float) -> None:
    if v is None or v <= 0:
        raise ValueError(f"{name} must be > 0.")


def _vf(name: str, v: float) -> None:
    if v is None or not (0.0 <= v < 1.0):
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1.")


def logmean(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        raise ValueError("logmean requires both arguments > 0.")
    if abs(a - b) < 1e-14:
        return a
    return (a - b) / math.log(a / b)


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


# ----------------------------
# spec
# ----------------------------

@dataclass(frozen=True)
class PackedAbsorptionGeneralSpec:
    """
    Robust packed-tower absorption toolkit (dilute; linear equilibrium y* = m x).

    Endpoints (mole fractions):
      Gas:    y1 (inlet, bottom/rich), y2 (outlet, top/lean)
      Liquid: x2 (inlet, top/lean), x1 (outlet, bottom/rich) optional

    Flows:
      V_total, L_total : total molar flowrates [kmol/h] (or any consistent mol/time)

    Equilibrium:
      m : slope in y = m x

    Geometry / height:
      z : packed height [m] (optional if you want to compute H = z/N)
      S : cross-sectional area [m^2] (optional; needed for H_G, H_L, H_OG, H_OL from k’s)

    Absorption factor basis for A = L/(m V):
      use_solute_free_for_A = True (default):
        V' = V_total (1 - y1)
        L' = L_total (1 - x2)
        A  = L'/(m V')
      If False:
        A  = L_total/(m V_total)
      You can also override by providing A_override.

    Transfer unit / coefficient inputs (optional):
      k_ya, k_xa : film coefficients per packed volume (k'ya, k'xa)
                  units: mol/(time·m^3·molfrac) consistent with V, L, S
      K_ya, K_xa : overall coefficients per packed volume (K'ya, K'xa)
                  if omitted but k_ya and k_xa provided, overall are computed by
                  simple resistances for linear equilibrium (dilute):
                    1/K_ya = 1/k_ya + m/k_xa
                    1/K_xa = 1/k_xa + 1/(m k_ya)

    Optional interface endpoint compositions (only if you want NG, NL explicitly):
      yi1, yi2 : interface gas mole fractions at y1 and y2 bulk points
      xi1, xi2 : interface liquid mole fractions at x1 and x2 bulk points

    What this module can compute (when sufficient inputs exist):
      - x1 (if not given) from overall balance on solute-free ratios (X,Y)
      - A (absorption factor)
      - N (equivalent equilibrium stages; Kremser analytical form)
      - HETP (if z is given)
      - NOG, NOL (overall transfer units) using the CLOSED-FORM screenshot equations
      - HOG, HOL from z/NOG and z/NOL (if z is given)
      - H_G, H_L, H_OG, H_OL from coefficients: H = V/(k a S) style
      - NG, NL from log-mean driving forces using yi endpoints / xi endpoints (if provided)
    """

    # required core
    V_total: float
    L_total: float
    y1: float
    y2: float
    x2: float
    m: float

    # optional
    x1: Optional[float] = None
    z: Optional[float] = None
    S: Optional[float] = None

    # absorption factor control
    use_solute_free_for_A: bool = True
    A_override: Optional[float] = None

    # coefficients (optional)
    k_ya: Optional[float] = None
    k_xa: Optional[float] = None
    K_ya: Optional[float] = None
    K_xa: Optional[float] = None

    # interface endpoints (optional; for NG/NL)
    yi1: Optional[float] = None
    yi2: Optional[float] = None
    xi1: Optional[float] = None
    xi2: Optional[float] = None


# ----------------------------
# core flow helpers
# ----------------------------

def V_prime(V_total: float, y1: float) -> float:
    _vp("V_total", V_total)
    _vf("y1", y1)
    Vp = V_total * (1.0 - y1)
    _vp("V_prime", Vp)
    return Vp


def L_prime(L_total: float, x2: float) -> float:
    _vp("L_total", L_total)
    _vf("x2", x2)
    Lp = L_total * (1.0 - x2)
    _vp("L_prime", Lp)
    return Lp


def x1_from_overall_balance(Vp: float, Lp: float, y1: float, y2: float, x2: float) -> float:
    """
    Solute-free overall balance (dilute form):
        L' X2 + V' Y1 = L' X1 + V' Y2
      => X1 = X2 + (V'/L')(Y1 - Y2)
    """
    _vp("V_prime", Vp)
    _vp("L_prime", Lp)
    _vf("y1", y1)
    _vf("y2", y2)
    _vf("x2", x2)

    X2 = X_from_x(x2)
    Y1 = Y_from_y(y1)
    Y2 = Y_from_y(y2)

    X1 = X2 + (Vp / Lp) * (Y1 - Y2)
    if X1 < 0:
        raise ValueError("Computed X1 < 0; check inputs.")
    x1 = x_from_X(X1)
    _vf("x1", x1)
    return x1


def absorption_factor_A(L_used: float, m: float, V_used: float) -> float:
    _vp("L_used", L_used)
    _vp("m", m)
    _vp("V_used", V_used)
    return L_used / (m * V_used)


# ----------------------------
# analytical equations (screenshots)
# ----------------------------

def _ratio_y_term(y1: float, y2: float, m: float, x2: float) -> float:
    num = y1 - m * x2
    den = y2 - m * x2
    if num <= 0 or den <= 0:
        raise ValueError("Require (y - m x2) > 0 at both ends.")
    return num / den


def N_theoretical(y1: float, y2: float, x2: float, m: float, A: float) -> float:
    """
    Equivalent equilibrium stages (Kremser analytical; same structure used in your tray module).
    """
    _vf("y1", y1); _vf("y2", y2); _vf("x2", x2); _vp("m", m); _vp("A", A)
    r = _ratio_y_term(y1, y2, m, x2)

    if abs(A - 1.0) < 1e-12:
        return r - 1.0

    inside = (1.0 - 1.0 / A) * r + 1.0 / A
    if inside <= 0:
        raise ValueError("Log argument <= 0 in N.")
    return math.log(inside) / math.log(A)


def NOG_overall_closed_form(y1: float, y2: float, x2: float, m: float, A: float) -> float:
    """
    Screenshot:
      NOG = 1/(1 - 1/A) * ln[ (1 - 1/A)*((y1 - m x2)/(y2 - m x2)) + 1/A ]
    """
    _vf("y1", y1); _vf("y2", y2); _vf("x2", x2); _vp("m", m); _vp("A", A)
    r = _ratio_y_term(y1, y2, m, x2)

    if abs(A - 1.0) < 1e-12:
        return r - 1.0

    inside = (1.0 - 1.0 / A) * r + 1.0 / A
    if inside <= 0:
        raise ValueError("Log argument <= 0 in NOG.")
    return (1.0 / (1.0 - 1.0 / A)) * math.log(inside)


def NOL_overall_closed_form(y1: float, x1: float, x2: float, m: float, A: float) -> float:
    """
    Screenshot:
      NOL = 1/(1 - A) * ln[ (1 - A)*((x2 - y1/m)/(x1 - y1/m)) + A ]
    """
    _vf("y1", y1); _vf("x1", x1); _vf("x2", x2); _vp("m", m); _vp("A", A)

    y1_over_m = y1 / m
    den = (x1 - y1_over_m)
    if abs(den) < 1e-18:
        raise ValueError("x1 - y1/m too close to 0; NOL singular.")
    ratio = (x2 - y1_over_m) / den

    if abs(1.0 - A) < 1e-12:
        return ratio - 1.0

    inside = (1.0 - A) * ratio + A
    if inside <= 0:
        raise ValueError("Log argument <= 0 in NOL.")
    return (1.0 / (1.0 - A)) * math.log(inside)


# ----------------------------
# coefficient-based H's + optional NG/NL
# ----------------------------

def overall_Kya_from_resistances(k_ya: float, k_xa: float, m: float) -> float:
    _vp("k_ya", k_ya); _vp("k_xa", k_xa); _vp("m", m)
    return 1.0 / (1.0 / k_ya + m / k_xa)


def overall_Kxa_from_resistances(k_ya: float, k_xa: float, m: float) -> float:
    _vp("k_ya", k_ya); _vp("k_xa", k_xa); _vp("m", m)
    return 1.0 / (1.0 / k_xa + 1.0 / (m * k_ya))


def NG_from_interface_endpoints(y1: float, y2: float, yi1: float, yi2: float) -> float:
    """
    Dilute slide result:
      NG = (y1 - y2) / (y - yi)_M
      (y - yi)_M = [(y1-yi1) - (y2-yi2)] / ln[(y1-yi1)/(y2-yi2)]
    """
    _vf("y1", y1); _vf("y2", y2); _vf("yi1", yi1); _vf("yi2", yi2)
    d1 = y1 - yi1
    d2 = y2 - yi2
    dM = logmean(d1, d2)
    return (y1 - y2) / dM


def NL_from_interface_endpoints(x1: float, x2: float, xi1: float, xi2: float) -> float:
    """
    Dilute slide result:
      NL = (x1 - x2) / (xi - x)_M
      (xi - x)_M = [(xi1-x1) - (xi2-x2)] / ln[(xi1-x1)/(xi2-x2)]
    """
    _vf("x1", x1); _vf("x2", x2); _vf("xi1", xi1); _vf("xi2", xi2)
    d1 = xi1 - x1
    d2 = xi2 - x2
    dM = logmean(d1, d2)
    return (x1 - x2) / dM


# ----------------------------
# main solve
# ----------------------------

def solve_packed_absorption_general(spec: PackedAbsorptionGeneralSpec) -> Dict[str, Any]:
    # validate core
    _vp("V_total", spec.V_total)
    _vp("L_total", spec.L_total)
    _vf("y1", spec.y1)
    _vf("y2", spec.y2)
    _vf("x2", spec.x2)
    _vp("m", spec.m)
    if spec.y2 >= spec.y1:
        raise ValueError("Require y1 > y2 for absorption.")

    # choose flows for A
    Vp = V_prime(spec.V_total, spec.y1)
    Lp = L_prime(spec.L_total, spec.x2)

    if spec.use_solute_free_for_A:
        V_used = Vp
        L_used = Lp
        A_basis = "solute_free"
    else:
        V_used = spec.V_total
        L_used = spec.L_total
        A_basis = "total_flow"

    A = spec.A_override if spec.A_override is not None else absorption_factor_A(L_used, spec.m, V_used)

    # compute x1 if missing
    x1 = spec.x1 if spec.x1 is not None else x1_from_overall_balance(Vp, Lp, spec.y1, spec.y2, spec.x2)

    # analytical counts
    N = N_theoretical(spec.y1, spec.y2, spec.x2, spec.m, A)
    NOG = NOG_overall_closed_form(spec.y1, spec.y2, spec.x2, spec.m, A)
    NOL = NOL_overall_closed_form(spec.y1, x1, spec.x2, spec.m, A)

    out: Dict[str, Any] = {
        "inputs": {
            "V_total": spec.V_total,
            "L_total": spec.L_total,
            "y1": spec.y1,
            "y2": spec.y2,
            "x2": spec.x2,
            "m": spec.m,
            "x1_given": spec.x1,
            "z_given": spec.z,
            "S_given": spec.S,
            "A_override": spec.A_override,
            "use_solute_free_for_A": spec.use_solute_free_for_A,
        },
        "flows": {
            "V_prime": Vp,
            "L_prime": Lp,
            "V_used_for_A": V_used,
            "L_used_for_A": L_used,
            "A_basis": A_basis,
        },
        "compositions": {
            "x1": x1,
            "x2": spec.x2,
            "y1": spec.y1,
            "y2": spec.y2,
            "y1_over_m": spec.y1 / spec.m,
        },
        "equilibrium": {
            "m": spec.m,
            "y_star1": spec.m * x1,
            "y_star2": spec.m * spec.x2,
            "x_star1": spec.y1 / spec.m,
            "x_star2": spec.y2 / spec.m,
        },
        "absorption_factor": {"A": A},
        "analytical": {
            "N_theoretical": N,
            "NOG_overall_closed_form": NOG,
            "NOL_overall_closed_form": NOL,
        },
    }

    # heights from z (if z given)
    if spec.z is not None:
        _vp("z", spec.z)
        out["heights_from_z"] = {
            "HETP": spec.z / N,
            "HOG_from_z_over_NOG": spec.z / NOG,
            "HOL_from_z_over_NOL": spec.z / NOL,
        }

    # coefficient-based H's (if k’s + S available)
    coeff_block: Dict[str, Any] = {}
    if spec.S is not None:
        _vp("S", spec.S)

        # prefer solute-free for V and L in H definitions unless user forces total by A_basis
        V_for_H = Vp if spec.use_solute_free_for_A else spec.V_total
        L_for_H = Lp if spec.use_solute_free_for_A else spec.L_total

        coeff_block["basis"] = {
            "V_for_H": V_for_H,
            "L_for_H": L_for_H,
            "note": "Heights H = V/(k a S) use V,L consistent with chosen basis.",
        }

        if spec.k_ya is not None:
            _vp("k_ya", spec.k_ya)
            coeff_block["HG"] = V_for_H / (spec.k_ya * spec.S)  # H_G = V/(k'ya S)

        if spec.k_xa is not None:
            _vp("k_xa", spec.k_xa)
            coeff_block["HL"] = L_for_H / (spec.k_xa * spec.S)  # H_L = L/(k'xa S)

        # overall coefficients
        K_ya = spec.K_ya
        K_xa = spec.K_xa

        if (K_ya is None or K_xa is None) and (spec.k_ya is not None) and (spec.k_xa is not None):
            if K_ya is None:
                K_ya = overall_Kya_from_resistances(spec.k_ya, spec.k_xa, spec.m)
            if K_xa is None:
                K_xa = overall_Kxa_from_resistances(spec.k_ya, spec.k_xa, spec.m)

        coeff_block["coefficients"] = {
            "k_ya": spec.k_ya,
            "k_xa": spec.k_xa,
            "K_ya": K_ya,
            "K_xa": K_xa,
        }

        if K_ya is not None:
            _vp("K_ya", K_ya)
            coeff_block["HOG"] = V_for_H / (K_ya * spec.S)  # H_OG = V/(K'ya S)

        if K_xa is not None:
            _vp("K_xa", K_xa)
            coeff_block["HOL"] = L_for_H / (K_xa * spec.S)  # H_OL = L/(K'xa S)

        # if z given, compute N's implied by coefficients: N = z/H
        if spec.z is not None:
            _vp("z", spec.z)
            implied: Dict[str, float] = {}
            if "HG" in coeff_block:
                implied["NG_implied_from_HG"] = spec.z / coeff_block["HG"]
            if "HL" in coeff_block:
                implied["NL_implied_from_HL"] = spec.z / coeff_block["HL"]
            if "HOG" in coeff_block:
                implied["NOG_implied_from_HOG"] = spec.z / coeff_block["HOG"]
            if "HOL" in coeff_block:
                implied["NOL_implied_from_HOL"] = spec.z / coeff_block["HOL"]
            if implied:
                coeff_block["implied_transfer_units_from_z"] = implied

    if coeff_block:
        out["coefficient_based"] = coeff_block

    # interface-based NG/NL (only if interface endpoints supplied)
    if (spec.yi1 is not None) and (spec.yi2 is not None):
        _vf("yi1", spec.yi1)
        _vf("yi2", spec.yi2)
        NG = NG_from_interface_endpoints(spec.y1, spec.y2, spec.yi1, spec.yi2)
        out["interface_based"] = out.get("interface_based", {})
        out["interface_based"]["NG"] = NG
        out["interface_based"]["yi1_yi2"] = {"yi1": spec.yi1, "yi2": spec.yi2}
        if spec.z is not None:
            out["interface_based"]["HG_from_z_over_NG"] = spec.z / NG

    if (spec.xi1 is not None) and (spec.xi2 is not None):
        _vf("xi1", spec.xi1)
        _vf("xi2", spec.xi2)
        NL = NL_from_interface_endpoints(x1, spec.x2, spec.xi1, spec.xi2)
        out["interface_based"] = out.get("interface_based", {})
        out["interface_based"]["NL"] = NL
        out["interface_based"]["xi1_xi2"] = {"xi1": spec.xi1, "xi2": spec.xi2}
        if spec.z is not None:
            out["interface_based"]["HL_from_z_over_NL"] = spec.z / NL

    # relationship checks (from your slide)
    # For A != 1: NOG = N ln(A) / (1 - 1/A)
    rel: Dict[str, Any] = {}
    if abs(A - 1.0) >= 1e-12:
        rel["NOG_from_N_relation"] = N * math.log(A) / (1.0 - 1.0 / A)
        rel["difference_NOG_closed_minus_relation"] = NOG - rel["NOG_from_N_relation"]
        if spec.z is not None:
            rel["HETP_from_HOG_relation"] = (spec.z / NOG) * (math.log(1.0 / A) / ((1.0 - A) / A))
    else:
        rel["A_equals_1_limit"] = {"NOG_equals_N": True, "HETP_equals_HOG_if_z_given": (spec.z is not None)}
    out["relations"] = rel

    return out