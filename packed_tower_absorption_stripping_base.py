# bank/packed_tower_absorption_stripping_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal
import math

from bank.Tower.two_film_interface_concentration_base import (
    TwoFilmInterfaceSpec,
    interface_concentrations,
    logmean,
)


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


def V_prime(V_total: float, y1: float) -> float:
    _vp("V_total", V_total); _vf("y1", y1)
    Vp = V_total * (1.0 - y1)
    _vp("V_prime", Vp)
    return Vp


def L_prime(L_total: float, x2: float) -> float:
    _vp("L_total", L_total); _vf("x2", x2)
    Lp = L_total * (1.0 - x2)
    _vp("L_prime", Lp)
    return Lp


def absorption_factor_A(L_used: float, m: float, V_used: float) -> float:
    _vp("L_used", L_used); _vp("m", m); _vp("V_used", V_used)
    return L_used / (m * V_used)


def Kya_from_resistances(k_ya: float, k_xa: float, m: float) -> float:
    _vp("k_ya", k_ya); _vp("k_xa", k_xa); _vp("m", m)
    return 1.0 / (1.0 / k_ya + m / k_xa)


def Kxa_from_resistances(k_ya: float, k_xa: float, m: float) -> float:
    _vp("k_ya", k_ya); _vp("k_xa", k_xa); _vp("m", m)
    return 1.0 / (1.0 / k_xa + 1.0 / (m * k_ya))


def NOG_closed_form(y1: float, y2: float, x2: float, m: float, A: float) -> float:
    # slide closed form (overall gas)
    num = y1 - m * x2
    den = y2 - m * x2
    if num <= 0 or den <= 0:
        raise ValueError("Require (y - m x2) > 0 at both ends.")
    r = num / den
    if abs(A - 1.0) < 1e-12:
        return r - 1.0
    inside = (1.0 - 1.0 / A) * r + 1.0 / A
    if inside <= 0:
        raise ValueError("Log argument <= 0 in NOG.")
    return (1.0 / (1.0 - 1.0 / A)) * math.log(inside)


def NOL_closed_form(y1: float, x1: float, x2: float, m: float, A: float) -> float:
    y1_over_m = y1 / m
    den = (x1 - y1_over_m)
    if abs(den) < 1e-18:
        raise ValueError("x1 - y1/m too close to 0.")
    ratio = (x2 - y1_over_m) / den
    if abs(1.0 - A) < 1e-12:
        return ratio - 1.0
    inside = (1.0 - A) * ratio + A
    if inside <= 0:
        raise ValueError("Log argument <= 0 in NOL.")
    return (1.0 / (1.0 - A)) * math.log(inside)


@dataclass(frozen=True)
class PackedTowerCoeffSpec:
    """
    Packed tower absorption/stripping with mass-transfer coefficients and interface concentrations.

    Core (required):
      V_total, L_total : molar flowrates [kmol/h] (or consistent mol/time)
      y1, y2           : gas inlet/outlet mole fractions (bottom/top)
      x2               : liquid inlet mole fraction (top)
      m                : equilibrium slope y* = m x
      S                : cross-sectional area [m^2]
      z                : packed height [m]

    Optional:
      x1               : liquid outlet mole fraction (bottom). If omitted, computed by solute-free balance.
      basis_for_VL     : "solute_free" (default) uses V', L' in HG/HL/HOG/HOL definitions
                         "total" uses V_total, L_total

    Coefficients:
      k_ya, k_xa : film coefficients per packed volume (k'ya, k'xa)
      K_ya, K_xa : overall coefficients per packed volume (K'ya, K'xa). If omitted and k's present, computed via resistances.

    Interface:
      If you want NG and NL explicitly, you can request interface endpoints via two-film closed form:
        compute interface at (x1,y1) and (x2,y2) using k's (dilute closed form),
        then NG=(y1-y2)/(y-yi)_M and NL=(x1-x2)/(xi-x)_M.
    """
    V_total: float
    L_total: float
    y1: float
    y2: float
    x2: float
    m: float
    S: float
    z: float
    x1: Optional[float] = None
    basis_for_VL: Literal["solute_free", "total"] = "solute_free"

    k_ya: Optional[float] = None
    k_xa: Optional[float] = None
    K_ya: Optional[float] = None
    K_xa: Optional[float] = None

    compute_interface_endpoints: bool = False


def solve_packed_tower_with_coeffs(spec: PackedTowerCoeffSpec) -> Dict[str, Any]:
    _vp("V_total", spec.V_total)
    _vp("L_total", spec.L_total)
    _vf("y1", spec.y1)
    _vf("y2", spec.y2)
    _vf("x2", spec.x2)
    _vp("m", spec.m)
    _vp("S", spec.S)
    _vp("z", spec.z)

    if spec.y2 >= spec.y1:
        raise ValueError("Require y1 > y2 for absorption (for stripping, swap definitions consistently).")

    Vp = V_prime(spec.V_total, spec.y1)
    Lp = L_prime(spec.L_total, spec.x2)

    # compute x1 if missing using solute-free overall balance
    if spec.x1 is None:
        X2 = X_from_x(spec.x2)
        Y1 = Y_from_y(spec.y1)
        Y2 = Y_from_y(spec.y2)
        X1 = X2 + (Vp / Lp) * (Y1 - Y2)
        if X1 < 0:
            raise ValueError("Computed X1 < 0.")
        x1 = x_from_X(X1)
    else:
        x1 = spec.x1
    _vf("x1", x1)

    # choose V,L for HTU definitions
    if spec.basis_for_VL == "solute_free":
        V_for_H = Vp
        L_for_H = Lp
        basis = "solute_free"
    else:
        V_for_H = spec.V_total
        L_for_H = spec.L_total
        basis = "total"

    # absorption factor (for closed-form overall TU equations)
    A = absorption_factor_A(L_for_H, spec.m, V_for_H)

    out: Dict[str, Any] = {
        "inputs": {
            "V_total": spec.V_total,
            "L_total": spec.L_total,
            "y1": spec.y1,
            "y2": spec.y2,
            "x2": spec.x2,
            "x1": x1,
            "m": spec.m,
            "S": spec.S,
            "z": spec.z,
            "basis_for_VL": basis,
        },
        "solute_free": {"V_prime": Vp, "L_prime": Lp},
        "equilibrium": {
            "y_star1": spec.m * x1,
            "y_star2": spec.m * spec.x2,
            "x_star1": spec.y1 / spec.m,
            "x_star2": spec.y2 / spec.m,
        },
        "absorption_factor": {"A": A},
    }

    # closed-form overall transfer units (slides)
    NOG_cf = NOG_closed_form(spec.y1, spec.y2, spec.x2, spec.m, A)
    NOL_cf = NOL_closed_form(spec.y1, x1, spec.x2, spec.m, A)
    out["overall_transfer_units_closed_form"] = {
        "NOG": NOG_cf,
        "HOG": spec.z / NOG_cf,
        "NOL": NOL_cf,
        "HOL": spec.z / NOL_cf,
    }

    # coefficient-based HTUs (HG, HL, HOG, HOL) and implied NTUs (z/H)
    coeffs: Dict[str, Any] = {"basis_for_HTU": basis}

    if spec.k_ya is not None:
        _vp("k_ya", spec.k_ya)
        HG = V_for_H / (spec.k_ya * spec.S)
        coeffs["HG"] = HG
        coeffs["NG_implied"] = spec.z / HG

    if spec.k_xa is not None:
        _vp("k_xa", spec.k_xa)
        HL = L_for_H / (spec.k_xa * spec.S)
        coeffs["HL"] = HL
        coeffs["NL_implied"] = spec.z / HL

    K_ya = spec.K_ya
    K_xa = spec.K_xa
    if (K_ya is None or K_xa is None) and (spec.k_ya is not None) and (spec.k_xa is not None):
        if K_ya is None:
            K_ya = Kya_from_resistances(spec.k_ya, spec.k_xa, spec.m)
        if K_xa is None:
            K_xa = Kxa_from_resistances(spec.k_ya, spec.k_xa, spec.m)

    coeffs["coefficients"] = {"k_ya": spec.k_ya, "k_xa": spec.k_xa, "K_ya": K_ya, "K_xa": K_xa}

    if K_ya is not None:
        _vp("K_ya", K_ya)
        HOG = V_for_H / (K_ya * spec.S)
        coeffs["HOG"] = HOG
        coeffs["NOG_implied"] = spec.z / HOG

    if K_xa is not None:
        _vp("K_xa", K_xa)
        HOL = L_for_H / (K_xa * spec.S)
        coeffs["HOL"] = HOL
        coeffs["NOL_implied"] = spec.z / HOL

    if len(coeffs) > 1:
        out["HTU_from_coefficients"] = coeffs

    # Optional: compute interface endpoints and NG/NL from log-mean driving forces (dilute)
    if spec.compute_interface_endpoints:
        if spec.k_ya is None or spec.k_xa is None:
            raise ValueError("Need k_ya and k_xa to compute interface endpoints.")

        top = interface_concentrations(TwoFilmInterfaceSpec(x=x1, y=spec.y1, m=spec.m, k_ya=spec.k_ya, k_xa=spec.k_xa))
        bot = interface_concentrations(TwoFilmInterfaceSpec(x=spec.x2, y=spec.y2, m=spec.m, k_ya=spec.k_ya, k_xa=spec.k_xa))

        yi1 = top["interface"]["y_i"]
        yi2 = bot["interface"]["y_i"]
        xi1 = top["interface"]["x_i"]
        xi2 = bot["interface"]["x_i"]

        dy1 = spec.y1 - yi1
        dy2 = spec.y2 - yi2
        dx1 = xi1 - x1
        dx2 = xi2 - spec.x2

        dyM = logmean(dy1, dy2)
        dxM = logmean(dx1, dx2)

        NG = (spec.y1 - spec.y2) / dyM
        NL = (x1 - spec.x2) / dxM

        out["interface_endpoints"] = {
            "top_bulk": {"x1": x1, "y1": spec.y1},
            "top_interface": {"xi1": xi1, "yi1": yi1},
            "bottom_bulk": {"x2": spec.x2, "y2": spec.y2},
            "bottom_interface": {"xi2": xi2, "yi2": yi2},
            "NG": NG,
            "NL": NL,
            "HG_from_z_over_NG": spec.z / NG,
            "HL_from_z_over_NL": spec.z / NL,
        }

    return out
# ============================================================
# Example 22.5-2 style packed-height solver (tabulated eq + interface construction)
# ============================================================

from dataclasses import dataclass
from typing import Sequence, List, Dict, Any, Optional
import math

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


def _logmean_pos(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        raise ValueError("log-mean inputs must be > 0.")
    if abs(a - b) < 1e-14:
        return a
    return (a - b) / math.log(a / b)


def logmean_one_minus(v1: float, v2: float) -> float:
    _vf("v1", v1)
    _vf("v2", v2)
    return _logmean_pos(1.0 - v1, 1.0 - v2)


def logmean_delta(a1: float, a2: float) -> float:
    if a1 <= 0 or a2 <= 0:
        raise ValueError("logmean_delta requires both arguments > 0.")
    return _logmean_pos(a1, a2)


def build_eq_pchip(x_eq: Sequence[float], y_eq: Sequence[float]) -> PchipInterpolator:
    x = np.asarray(x_eq, dtype=float)
    y = np.asarray(y_eq, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Equilibrium table must have same length >= 2.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x_eq must be strictly increasing.")
    if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
        raise ValueError("Equilibrium values must be in [0,1].")
    return PchipInterpolator(x, y, extrapolate=False)


def compute_x1_from_overall_balance_ratio(Vp: float, Lp: float, y1: float, y2: float, x2: float) -> float:
    # L'X2 + V'Y1 = L'X1 + V'Y2
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
    _vf("x1", x1)
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
    xs = np.linspace(x_min, x_max, int(n_scan))
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
    _vp("k_ya", k_ya)
    _vp("k_xa", k_xa)
    _vf("x_bulk", x_bulk)
    _vf("y_bulk", y_bulk)

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


@dataclass(frozen=True)
class PackedHeightExample2252Spec:
    """
    Adds Example 22.5-2 packed-height solver to this bank module.

    Inputs:
      V_prime, L_prime (solute-free flows)
      y1, y2, x2 (bulk endpoints)
      diameter_m
      equilibrium table x_eq, y_eq
      k_ya, k_xa
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


def packed_height_example2252_all(spec: PackedHeightExample2252Spec) -> Dict[str, Any]:
    _vp("V_prime", spec.V_prime)
    _vp("L_prime", spec.L_prime)
    _vp("diameter_m", spec.diameter_m)
    _vp("k_ya", spec.k_ya)
    _vp("k_xa", spec.k_xa)
    _vf("y1", spec.y1)
    _vf("y2", spec.y2)
    _vf("x2", spec.x2)

    eq = build_eq_pchip(spec.x_eq, spec.y_eq)
    x_min, x_max = float(min(spec.x_eq)), float(max(spec.x_eq))

    # 1) overall balance for x1 (ratio form)
    x1 = compute_x1_from_overall_balance_ratio(spec.V_prime, spec.L_prime, spec.y1, spec.y2, spec.x2)

    # 2) interface at rich end (x1,y1)
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

    # 3) interface at lean end (x2,y2)
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
    Lav = spec.L_prime

    # (a) z from k'ya
    dy1 = spec.y1 - yi1
    dy2 = spec.y2 - yi2
    dyM = logmean_delta(dy1, dy2)
    z_kya = (Vav / S) * (spec.y1 - spec.y2) / (spec.k_ya * dyM)

    # (b) z from k'xa
    dx1 = xi1 - x1
    dx2 = xi2 - spec.x2
    dxM = logmean_delta(dx1, dx2)
    z_kxa = (Lav / S) * (x1 - spec.x2) / (spec.k_xa * dxM)

    # (c) z from overall K'ya (rich end)
    y1_star = float(eq(x1))
    y2_star = float(eq(spec.x2))

    one_minus_y_RM = logmean_one_minus(y1_star, spec.y1)
    one_minus_y_M_top = logmean_one_minus(yi1, spec.y1)
    one_minus_x_M_top = logmean_one_minus(x1, xi1)

    ky_eff = spec.k_ya / one_minus_y_M_top
    kx_eff = spec.k_xa / one_minus_x_M_top
    m_local = float(eq.derivative()(x1))

    Ky_bracket = 1.0 / (1.0 / ky_eff + m_local / kx_eff)
    Kya_prime = Ky_bracket * one_minus_y_RM

    dy_star_1 = spec.y1 - y1_star
    dy_star_2 = spec.y2 - y2_star
    dy_star_M = logmean_delta(dy_star_1, dy_star_2)

    z_Kya = (Vav / S) * (spec.y1 - spec.y2) / (Kya_prime * dy_star_M)

    return {
        "bulk": {"x1": x1, "x2": spec.x2, "y1": spec.y1, "y2": spec.y2},
        "interface": {
            "top": {"x_i1": xi1, "y_i1": yi1, "slope": float(top["slope_final"])},
            "bottom": {"x_i2": xi2, "y_i2": yi2, "slope": float(bot["slope_final"])},
        },
        "equilibrium_star": {"y1_star": y1_star, "y2_star": y2_star, "m_local_at_x1": m_local},
        "averages": {"S_m2": S, "V1": V1, "V2": V2, "Vav": Vav, "Lav": Lav},
        "driving_forces": {"dyM_y_minus_yi": dyM, "dxM_xi_minus_x": dxM, "dyM_y_minus_ystar": dy_star_M},
        "overall_Kya": {
            "one_minus_y_RM": one_minus_y_RM,
            "one_minus_y_M_top": one_minus_y_M_top,
            "one_minus_x_M_top": one_minus_x_M_top,
            "ky_eff": ky_eff,
            "kx_eff": kx_eff,
            "Ky_bracket": Ky_bracket,
            "Kya_prime": Kya_prime,
        },
        "heights": {"z_using_kya": z_kya, "z_using_kxa": z_kxa, "z_using_Kya": z_Kya},
        "iteration_history": {"top": top["history"], "bottom": bot["history"]},
    }
# ----------------------------------------
# Hydraulics → mass-transfer linkage helpers
# ----------------------------------------

def cross_sectional_area_from_diameter(D: float) -> float:
    """
    Convert tower diameter to cross-sectional area.

    Parameters
    ----------
    D : float
        Tower diameter [m]

    Returns
    -------
    S : float
        Cross-sectional area [m^2]

    Equation:
        S = π D^2 / 4
    """
    if D <= 0:
        raise ValueError("Diameter must be > 0.")
    return math.pi * D**2 / 4.0


def superficial_gas_velocity(V_total: float, y: float, S: float) -> float:
    """
    Superficial gas velocity using total gas flow.

    Parameters
    ----------
    V_total : float
        Total gas molar flow [mol/time]
    y : float
        Gas-phase solute mole fraction
    S : float
        Cross-sectional area [m^2]

    Returns
    -------
    v_g : float
        Superficial gas velocity [mol/(time·m^2)]
    """
    if not (0.0 <= y < 1.0):
        raise ValueError("y must satisfy 0 <= y < 1.")
    if S <= 0:
        raise ValueError("Area must be > 0.")
    return V_total / S


def superficial_liquid_velocity(L_total: float, x: float, S: float) -> float:
    """
    Superficial liquid velocity using total liquid flow.

    Parameters
    ----------
    L_total : float
        Total liquid molar flow [mol/time]
    x : float
        Liquid-phase solute mole fraction
    S : float
        Cross-sectional area [m^2]

    Returns
    -------
    v_l : float
        Superficial liquid velocity [mol/(time·m^2)]
    """
    if not (0.0 <= x < 1.0):
        raise ValueError("x must satisfy 0 <= x < 1.")
    if S <= 0:
        raise ValueError("Area must be > 0.")
    return L_total / S
# ============================================================
# Two-film model with tabulated equilibrium (stagnant/nondiffusing B)
# Adds: interface (x_i,y_i), flux, overall Kx/Ky, resistance split
# ============================================================

from dataclasses import dataclass
from typing import Sequence, Dict, Any, List
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import math


@dataclass(frozen=True)
class TwoFilmTableSpec:
    """
    Wetted-wall / local two-film model with tabulated equilibrium y = f(x).

    Inputs:
      x_table, y_table : equilibrium data (strictly increasing x)
      k_y, k_x         : film coefficients k'y and k'x (per area basis)
      y_AG, x_AL       : bulk gas and bulk liquid mole fractions at point P
      max_iter, tol_slope : interface iteration controls

    Outputs:
      interface: x_Ai, y_Ai
      effective films: ky_eff=k_y/(1-y)iM, kx_eff=k_x/(1-x)iM
      overall: Ky_overall=K'y/(1-y)BM, Kx_overall=K'x/(1-x)BM, plus Ky_prime,Kx_prime
      flux: N_A from gas film, liquid film, overall-y, overall-x
      resistance: percent in gas film
    """
    x_table: Sequence[float]
    y_table: Sequence[float]
    k_y: float
    k_x: float
    y_AG: float
    x_AL: float
    max_iter: int = 50
    tol_slope: float = 1e-8


def _arr1(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    if a.ndim != 1:
        raise ValueError("Table inputs must be 1D.")
    return a


def _validate_table_xy(x: np.ndarray, y: np.ndarray) -> None:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x_table and y_table must have same length >= 2.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Equilibrium table has non-finite values.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x_table must be strictly increasing.")
    if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
        raise ValueError("x_table and y_table must be in [0,1].")


def _logmean_pos(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        raise ValueError("Log-mean inputs must be > 0.")
    if abs(a - b) < 1e-14:
        return a
    return (a - b) / math.log(a / b)


def _logmean_one_minus(v1: float, v2: float) -> float:
    if not (0.0 <= v1 < 1.0) or not (0.0 <= v2 < 1.0):
        raise ValueError("Values must satisfy 0 <= v < 1 for log-mean of (1-v).")
    return _logmean_pos(1.0 - v1, 1.0 - v2)


def _invert_y_to_x(eq: PchipInterpolator, y_target: float, x_min: float, x_max: float) -> float:
    def f(x: float) -> float:
        return float(eq(x)) - y_target

    xs = np.linspace(x_min, x_max, 4001)
    fs = np.array([f(float(xx)) for xx in xs])
    idx = np.where(np.sign(fs[:-1]) * np.sign(fs[1:]) <= 0)[0]
    if len(idx) == 0:
        raise ValueError("Could not invert y->x within table range.")
    i = int(idx[0])
    return float(brentq(f, float(xs[i]), float(xs[i + 1]), xtol=1e-14, maxiter=200))


def _find_intersection_x(eq: PchipInterpolator, xAL: float, yAG: float, slope: float, x_min: float, x_max: float) -> float:
    # Solve eq(x) = yAG + slope*(x - xAL)
    def f(x: float) -> float:
        return float(eq(x)) - (yAG + slope * (x - xAL))

    xs = np.linspace(x_min, x_max, 4001)
    fs = np.array([f(float(xx)) for xx in xs])
    idx = np.where(np.sign(fs[:-1]) * np.sign(fs[1:]) <= 0)[0]
    if len(idx) == 0:
        raise ValueError("No intersection found between construction line and equilibrium curve in table range.")
    i = int(idx[0])
    return float(brentq(f, float(xs[i]), float(xs[i + 1]), xtol=1e-14, maxiter=200))


def solve_two_film_from_table(spec: TwoFilmTableSpec) -> Dict[str, Any]:
    _vp("k_y", spec.k_y)
    _vp("k_x", spec.k_x)
    _vf("y_AG", spec.y_AG)
    _vf("x_AL", spec.x_AL)

    x = _arr1(spec.x_table)
    y = _arr1(spec.y_table)
    _validate_table_xy(x, y)

    eq = PchipInterpolator(x, y, extrapolate=False)
    x_min, x_max = float(x[0]), float(x[-1])

    if not (x_min <= spec.x_AL <= x_max):
        raise ValueError("x_AL is outside x_table range; extend equilibrium data.")

    # equilibrium endpoints for overall driving forces
    y_star = float(eq(spec.x_AL))                        # y_A* at x_AL
    x_star = _invert_y_to_x(eq, spec.y_AG, x_min, x_max) # x_A* at y_AG

    # Iteration 1: dilute => (1-y)iM ~ 1, (1-x)iM ~ 1
    one_minus_y_iM = 1.0
    one_minus_x_iM = 1.0
    slope = - (spec.k_x / one_minus_x_iM) / (spec.k_y / one_minus_y_iM)

    hist: List[Dict[str, float]] = []

    for it in range(1, int(spec.max_iter) + 1):
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

    x_Ai = float(hist[-1]["x_Ai"])
    y_Ai = float(hist[-1]["y_Ai"])

    # effective film coefficients (stagnant/nondiffusing components)
    ky_eff = spec.k_y / one_minus_y_iM
    kx_eff = spec.k_x / one_minus_x_iM

    # flux from each film (Eq. 22.1-37 form)
    N_A_g = ky_eff * (spec.y_AG - y_Ai)
    N_A_l = kx_eff * (x_Ai - spec.x_AL)
    N_A = 0.5 * (N_A_g + N_A_l)

    # chord slopes
    m_prime = (y_Ai - y_star) / (x_Ai - spec.x_AL)
    m_dprime = (spec.y_AG - y_Ai) / (x_star - x_Ai)

    # BM log-mean factors
    one_minus_y_BM = _logmean_one_minus(y_star, spec.y_AG)
    one_minus_x_BM = _logmean_one_minus(spec.x_AL, x_star)

    # overall bracketed coefficients (used directly with overall driving forces)
    Ky_overall = 1.0 / (1.0 / ky_eff + m_prime / kx_eff)               # = K'y/(1-y)BM
    Kx_overall = 1.0 / (1.0 / kx_eff + 1.0 / (m_dprime * ky_eff))      # = K'x/(1-x)BM

    # primed overall coefficients
    Ky_prime = Ky_overall * one_minus_y_BM
    Kx_prime = Kx_overall * one_minus_x_BM

    # flux consistency from overall driving forces
    N_A_overall_y = Ky_overall * (spec.y_AG - y_star)
    N_A_overall_x = Kx_overall * (x_star - spec.x_AL)

    # resistance split (Eq. 22.1-53 form)
    R_gas = 1.0 / ky_eff
    R_liq_equiv = m_prime / kx_eff
    pct_R_gas = (R_gas / (R_gas + R_liq_equiv)) * 100.0

    return {
        "iteration": {
            "history": hist,
            "slope_final": float(slope),
            "one_minus_y_iM": float(one_minus_y_iM),
            "one_minus_x_iM": float(one_minus_x_iM),
        },
        "equilibrium_endpoints": {
            "y_A_star_at_xAL": float(y_star),
            "x_A_star_at_yAG": float(x_star),
            "one_minus_y_BM": float(one_minus_y_BM),
            "one_minus_x_BM": float(one_minus_x_BM),
        },
        "interface": {
            "x_Ai": float(x_Ai),
            "y_Ai": float(y_Ai),
            "m_prime": float(m_prime),
            "m_dprime": float(m_dprime),
        },
        "film_effective": {
            "ky_eff": float(ky_eff),
            "kx_eff": float(kx_eff),
        },
        "overall": {
            "Ky_overall": float(Ky_overall),   # = K'y/(1-y)BM
            "Kx_overall": float(Kx_overall),   # = K'x/(1-x)BM
            "Ky_prime": float(Ky_prime),
            "Kx_prime": float(Kx_prime),
        },
        "flux": {
            "N_A": float(N_A),
            "N_A_from_gas_film": float(N_A_g),
            "N_A_from_liq_film": float(N_A_l),
            "N_A_from_overall_y": float(N_A_overall_y),
            "N_A_from_overall_x": float(N_A_overall_x),
        },
        "resistance": {
            "percent_R_gas": float(pct_R_gas),
            "percent_R_liq": float(100.0 - pct_R_gas),
        },
    }