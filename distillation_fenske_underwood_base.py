# bank/distillation_fenske_underwood_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Sequence, List, Optional

import math


def _vp(name: str, v: float) -> None:
    if v <= 0:
        raise ValueError(f"{name} must be > 0.")


def normalize(z: Sequence[float], name: str = "composition") -> List[float]:
    if len(z) < 2:
        raise ValueError(f"{name} must have length >= 2.")
    s = float(sum(z))
    if s <= 0:
        raise ValueError(f"{name} sum must be > 0.")
    out = [float(v) / s for v in z]
    for i, v in enumerate(out):
        if v < -1e-15:
            raise ValueError(f"{name}[{i}] must be >= 0.")
    return out


# -----------------------------
# Fenske (keys)
# -----------------------------
def fenske_Nmin(xD_LK: float, xB_LK: float, xD_HK: float, xB_HK: float, alpha_avg_LK_HK: float) -> float:
    # Nmin = ln[(xD_LK/xB_LK)*(xB_HK/xD_HK)] / ln(alpha_avg)
    _vp("alpha_avg_LK_HK", alpha_avg_LK_HK)
    _vp("xD_LK", xD_LK)
    _vp("xB_LK", xB_LK)
    _vp("xD_HK", xD_HK)
    _vp("xB_HK", xB_HK)
    arg = (xD_LK / xB_LK) * (xB_HK / xD_HK)
    _vp("log_argument", arg)
    return math.log(arg) / math.log(alpha_avg_LK_HK)


# -----------------------------
# Underwood
# -----------------------------
def underwood_theta(
    alpha: Sequence[float],
    zF: Sequence[float],
    q: float,
    theta_lo: float,
    theta_hi: float,
    maxiter: int = 200,
    tol: float = 1e-12,
) -> float:
    # 1 - q = sum (alpha_i z_i)/(alpha_i - theta)
    z = normalize(zF, "zF")
    if len(alpha) != len(z):
        raise ValueError("alpha length must match zF.")
    _vp("maxiter", float(maxiter))
    _vp("tol", tol)

    def f(theta: float) -> float:
        s = 0.0
        for ai, zi in zip(alpha, z):
            _vp("alpha_i", float(ai))
            denom = float(ai) - theta
            if abs(denom) < 1e-18:
                return float("inf")
            s += float(ai) * zi / denom
        return (1.0 - q) - s

    flo = f(theta_lo)
    fhi = f(theta_hi)
    if flo == 0.0:
        return theta_lo
    if fhi == 0.0:
        return theta_hi
    if flo * fhi > 0:
        raise ValueError("theta bracket does not straddle a root; adjust theta_lo/theta_hi.")

    lo, hi = theta_lo, theta_hi
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) <= tol:
            return mid
        if flo * fm <= 0:
            hi = mid
            fhi = fm
        else:
            lo = mid
            flo = fm
    return 0.5 * (lo + hi)


def underwood_Rmin(alpha: Sequence[float], xD: Sequence[float], theta: float) -> float:
    # Rmin = sum (alpha_i xD_i)/(alpha_i - theta) - 1
    xD_n = normalize(xD, "xD")
    if len(alpha) != len(xD_n):
        raise ValueError("alpha length must match xD.")
    s = 0.0
    for ai, xDi in zip(alpha, xD_n):
        denom = float(ai) - theta
        if abs(denom) < 1e-18:
            return float("inf")
        s += float(ai) * xDi / denom
    Rmin = s - 1.0
    _vp("Rmin", Rmin)
    return Rmin


# -----------------------------
# Gilliland (Eduljee explicit)
# -----------------------------
def gilliland_eduljee_N(Nmin: float, R: float, Rmin: float) -> Dict[str, float]:
    # X = (R-Rmin)/(R+1); Y = 1 - exp(((1+54.4X)/(11+117.2X))*(X-1))
    # N = (Nmin + Y)/(1 - Y)
    _vp("Nmin", Nmin)
    _vp("R", R)
    _vp("Rmin", Rmin)
    if R <= Rmin:
        raise ValueError("Require R > Rmin for Gilliland.")
    X = (R - Rmin) / (R + 1.0)
    _vp("X", X)
    Y = 1.0 - math.exp(((1.0 + 54.4 * X) / (11.0 + 117.2 * X)) * (X - 1.0))
    Y = max(0.0, min(0.999999999999, Y))
    N = (Nmin + Y) / (1.0 - Y)
    return {"X": X, "Y": Y, "N": N}


# -----------------------------
# Kirkbride feed location ratio
# -----------------------------
def kirkbride_Nb_over_Na(xF_LK: float, xF_HK: float, xD_LK: float, xB_HK: float, D: float, B: float) -> float:
    # log10(Nb/Na) = 0.206 log10[(xF_LK/xF_HK)*(B/D)*(xB_HK/xD_LK)]
    _vp("xF_LK", xF_LK); _vp("xF_HK", xF_HK)
    _vp("xD_LK", xD_LK); _vp("xB_HK", xB_HK)
    _vp("D", D); _vp("B", B)
    arg = (xF_LK / xF_HK) * (B / D) * (xB_HK / xD_LK)
    _vp("kirkbride_log_arg", arg)
    return 10.0 ** (0.206 * math.log10(arg))


@dataclass(frozen=True)
class ShortcutDistillationSpec:
    """
    Consolidated Fenske + Underwood + Gilliland + Kirkbride.

    Provide either:
      - alpha_rel_HK (alpha_HK=1) directly, OR
      - K_top and K_bottom so alpha_i can be built vs HK and geometric-averaged.

    Keys chosen by LK_index and HK_index.
    """
    zF: Sequence[float]
    xD: Sequence[float]
    xB: Sequence[float]

    LK_index: int
    HK_index: int

    q: float
    R: float

    alpha_rel_HK: Optional[Sequence[float]] = None
    K_top: Optional[Sequence[float]] = None
    K_bottom: Optional[Sequence[float]] = None

    theta_lo: float = 0.0
    theta_hi: float = 10.0

    # optional actual D,B for Kirkbride; if omitted, ratio returned for B=D=1
    D: Optional[float] = None
    B: Optional[float] = None


def _alpha_from_K(K: Sequence[float], ref_index: int) -> List[float]:
    if ref_index < 0 or ref_index >= len(K):
        raise ValueError("ref_index out of range.")
    _vp("K_ref", float(K[ref_index]))
    return [float(Ki) / float(K[ref_index]) for Ki in K]


def shortcut_design(spec: ShortcutDistillationSpec) -> Dict[str, Any]:
    zF = normalize(spec.zF, "zF")
    xD = normalize(spec.xD, "xD")
    xB = normalize(spec.xB, "xB")
    n = len(zF)
    if len(xD) != n or len(xB) != n:
        raise ValueError("zF, xD, xB must have same length.")
    if not (0 <= spec.LK_index < n and 0 <= spec.HK_index < n):
        raise ValueError("LK/HK index out of range.")
    if spec.LK_index == spec.HK_index:
        raise ValueError("LK_index and HK_index must differ.")
    _vp("R", spec.R)

    if spec.alpha_rel_HK is not None:
        alpha = [float(a) for a in spec.alpha_rel_HK]
        if len(alpha) != n:
            raise ValueError("alpha_rel_HK length must match compositions.")
        if abs(alpha[spec.HK_index] - 1.0) > 1e-6:
            alpha = [a / alpha[spec.HK_index] for a in alpha]
    else:
        if spec.K_top is None or spec.K_bottom is None:
            raise ValueError("Provide either alpha_rel_HK or both K_top and K_bottom.")
        Kt = [float(v) for v in spec.K_top]
        Kb = [float(v) for v in spec.K_bottom]
        if len(Kt) != n or len(Kb) != n:
            raise ValueError("K_top and K_bottom must match composition length.")
        alpha_top = _alpha_from_K(Kt, spec.HK_index)
        alpha_bot = _alpha_from_K(Kb, spec.HK_index)
        alpha = [math.sqrt(at * ab) for at, ab in zip(alpha_top, alpha_bot)]

    xD_LK, xB_LK = xD[spec.LK_index], xB[spec.LK_index]
    xD_HK, xB_HK = xD[spec.HK_index], xB[spec.HK_index]
    alpha_avg_LK_HK = alpha[spec.LK_index] / alpha[spec.HK_index]

    Nmin = fenske_Nmin(xD_LK, xB_LK, xD_HK, xB_HK, alpha_avg_LK_HK)

    theta = underwood_theta(alpha, zF, spec.q, spec.theta_lo, spec.theta_hi)
    Rmin = underwood_Rmin(alpha, xD, theta)

    gill = gilliland_eduljee_N(Nmin=Nmin, R=spec.R, Rmin=Rmin)

    # Kirkbride
    D = 1.0 if spec.D is None else float(spec.D)
    B = 1.0 if spec.B is None else float(spec.B)
    Nb_over_Na = kirkbride_Nb_over_Na(
        xF_LK=zF[spec.LK_index],
        xF_HK=zF[spec.HK_index],
        xD_LK=xD_LK,
        xB_HK=xB_HK,
        D=D,
        B=B,
    )

    return {
        "inputs": {
            "LK_index": spec.LK_index,
            "HK_index": spec.HK_index,
            "q": spec.q,
            "R": spec.R,
            "D_used": D,
            "B_used": B,
        },
        "compositions": {"zF": zF, "xD": xD, "xB": xB},
        "alpha_rel_HK": alpha,
        "keys": {
            "xD_LK": xD_LK, "xB_LK": xB_LK,
            "xD_HK": xD_HK, "xB_HK": xB_HK,
            "alpha_avg_LK_HK": alpha_avg_LK_HK,
        },
        "fenske": {"Nmin_total_reflux": Nmin},
        "underwood": {"theta": theta, "Rmin": Rmin},
        "gilliland_eduljee": gill,
        "kirkbride": {"Nb_over_Na": Nb_over_Na},
        "estimates": {"N_total_theoretical_est": gill["N"]},
    }