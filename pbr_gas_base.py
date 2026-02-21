# file: bank/pbr_gas_base.py
"""
Gas-phase Packed Bed Reactor (PBR) with pressure drop (Ergun) using W as the independent variable.

UPDATED (to match the flexibility added to cstr_gas_base.py and pfr_gas_base.py):
- Feed can be specified by inlet molar flows F0 [mol/time] OR inlet concentrations C0 [mol/L]
- If C0 is used, you must also provide v0 [L/time] so absolute flows exist (needed for dX/dW).
- Inerts optional: include inert in F0/C0 with nu_signed=0 (or omit from nu_signed).
- CT can be supplied or inferred:
    If F0 provided: CT = FT0/v0
    If C0 provided: CT = sum(C0)

Model ODEs (same as your current file):
    dX/dW  = (-rA') / FA0
    dPr/dW = -(alpha/(2*Pr)) * (T/T0) * (FT/FT0)
with FT/FT0 = 1 + epsilon*X, epsilon computed from stoich + inlet flows.

Conventions:
- nu_signed: reactants negative, products positive
- rate_minus_rAprime(C) returns (-rA') > 0 [mol/(mass_cat*time)]
- C provided to rate is local concentration dict [mol/L or mol/m^3 consistent with P0,R,T]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

from scipy.optimize import brentq


# -------------------------
# Feed resolution (F0 or C0)
# -------------------------

def resolve_feed_gas(
    v0: float,
    *,
    F0: Optional[Dict[str, float]] = None,   # mol/time
    C0: Optional[Dict[str, float]] = None,   # mol/L
    CT: Optional[float] = None,              # mol/L (optional override)
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """
    Returns:
        F0_resolved (mol/time), C0_resolved (mol/L), CT_resolved (mol/L)
    """
    if v0 <= 0:
        raise ValueError("v0 must be > 0.")

    if (F0 is None) == (C0 is None):
        raise ValueError("Provide exactly one of F0 or C0.")

    if F0 is not None:
        F0r = {sp: float(val) for sp, val in F0.items()}
        if any(val < 0 for val in F0r.values()):
            raise ValueError("All inlet molar flows must be >= 0.")
        FT0 = sum(F0r.values())
        if FT0 <= 0:
            raise ValueError("Total inlet molar flow must be > 0.")
        CTr = (FT0 / v0) if CT is None else float(CT)
        if CTr <= 0:
            raise ValueError("CT must be > 0.")
        C0r = {sp: F0r[sp] / v0 for sp in F0r.keys()}
        return F0r, C0r, CTr

    C0r = {sp: float(val) for sp, val in C0.items()}
    if any(val < 0 for val in C0r.values()):
        raise ValueError("All inlet concentrations must be >= 0.")
    CT_sum = sum(C0r.values())
    if CT_sum <= 0:
        raise ValueError("Sum of inlet concentrations must be > 0.")
    CTr = CT_sum if CT is None else float(CT)
    if CTr <= 0:
        raise ValueError("CT must be > 0.")
    F0r = {sp: v0 * C0r[sp] for sp in C0r.keys()}
    return F0r, C0r, CTr


# -------------------------
# Stoichiometry + gas mixing
# -------------------------

def epsilon_from_stoich(
    nu_signed: Dict[str, float],
    F0: Dict[str, float],
    basis: str,
) -> float:
    """
    epsilon in FT/FT0 = 1 + epsilon*X, using conversion X of basis reactant.

    epsilon = (sum(nu_i)/a) * (FA0/FT0)
    where a = -nu_basis (positive stoich number for basis reactant).
    """
    if basis not in nu_signed:
        raise KeyError("basis not in nu_signed.")
    nu_basis = float(nu_signed[basis])
    if nu_basis >= 0:
        raise ValueError("basis must be a reactant (negative nu).")
    a = -nu_basis

    FA0 = float(F0.get(basis, 0.0))
    FT0 = sum(float(v) for v in F0.values())
    if FA0 <= 0 or FT0 <= 0:
        raise ValueError("F0 must include positive basis flow and positive total flow.")

    sum_nu = sum(float(v) for v in nu_signed.values())
    return (sum_nu / a) * (FA0 / FT0)


def flows_from_X(
    F0: Dict[str, float],
    nu_signed: Dict[str, float],
    basis: str,
    X: float,
) -> Dict[str, float]:
    """
    Fi = Fi0 + nu_i*xi, with xi = FA0*X/a.
    """
    if not (0.0 <= X <= 1.0):
        raise ValueError("X must be in [0,1].")
    if basis not in nu_signed:
        raise KeyError("basis not in nu_signed.")
    nu_basis = float(nu_signed[basis])
    if nu_basis >= 0:
        raise ValueError("basis must be a reactant (negative nu).")
    a = -nu_basis

    FA0 = float(F0.get(basis, 0.0))
    if FA0 < 0:
        raise ValueError("Basis flow must be >= 0.")

    xi = (FA0 * X) / a

    species = set(F0.keys()) | set(nu_signed.keys())
    F: Dict[str, float] = {}
    for sp in species:
        F[sp] = float(F0.get(sp, 0.0)) + float(nu_signed.get(sp, 0.0)) * xi
    return F


def concentrations_from_X_Pr(
    F0: Dict[str, float],
    nu_signed: Dict[str, float],
    basis: str,
    X: float,
    Pr: float,
    P0: float,
    T: float,
    R: float,
) -> Dict[str, float]:
    """
    Ideal-gas concentrations using Pr = P/P0:
        P = Pr * P0
        CT = P/(R*T)
        yi = Fi/FT
        Ci = yi * CT
    """
    if Pr <= 0:
        raise ValueError("Pr must be > 0.")
    if P0 <= 0 or T <= 0 or R <= 0:
        raise ValueError("P0, T, R must be > 0.")

    F = flows_from_X(F0, nu_signed, basis, X)
    FT = sum(F.values())
    if FT <= 0:
        raise ValueError("Total flow became non-positive; check stoichiometry/X.")

    P = Pr * P0
    CT = P / (R * T)

    return {sp: (F[sp] / FT) * CT for sp in F.keys()}


# -------------------------
# Coupled ODEs: X(W), Pr(W)
# -------------------------

@dataclass(frozen=True)
class GasPBRParams:
    # Feed (must provide exactly one of F0 or C0)
    v0: float                              # L/time at operating T,P
    F0: Optional[Dict[str, float]] = None  # mol/time
    C0: Optional[Dict[str, float]] = None  # mol/L
    CT: Optional[float] = None             # mol/L (optional override)

    # Reaction + state
    nu_signed: Dict[str, float] = None     # signed stoich
    basis: str = "A"                       # basis reactant for conversion X

    # Ideal-gas mapping for concentrations from Pr
    P0: float = 1.0                        # pressure units consistent with R
    T: float = 298.15                      # K
    R: float = 0.082057                    # L·atm/(mol·K) if P0 in atm, etc.

    # Pressure drop
    alpha: float = 0.0                     # 1/mass_cat

    # Optional
    T0: Optional[float] = None
    epsilon: Optional[float] = None


def _resolve_F0_CT(p: GasPBRParams) -> Tuple[Dict[str, float], float]:
    F0r, _, CTr = resolve_feed_gas(p.v0, F0=p.F0, C0=p.C0, CT=p.CT)
    return F0r, CTr


def _dX_dW(
    X: float,
    Pr: float,
    p: GasPBRParams,
    rate_minus_rAprime: Callable[[Dict[str, float]], float],
) -> float:
    F0r, _ = _resolve_F0_CT(p)

    FA0 = float(F0r.get(p.basis, 0.0))
    if FA0 <= 0:
        raise ValueError("Basis inlet flow must be > 0.")

    C = concentrations_from_X_Pr(F0r, p.nu_signed, p.basis, X, Pr, p.P0, p.T, p.R)
    minus_rAprime = float(rate_minus_rAprime(C))
    if minus_rAprime < 0:
        raise ValueError("(-rA') must be >= 0.")
    return minus_rAprime / FA0


def _dPr_dW(X: float, Pr: float, p: GasPBRParams) -> float:
    if Pr <= 0:
        raise ValueError("Pr must be > 0.")

    F0r, _ = _resolve_F0_CT(p)

    T0 = float(p.T if p.T0 is None else p.T0)
    if T0 <= 0:
        raise ValueError("T0 must be > 0.")

    eps = p.epsilon
    if eps is None:
        eps = epsilon_from_stoich(p.nu_signed, F0r, p.basis)

    FT_ratio = 1.0 + float(eps) * float(X)

    return -(float(p.alpha) / (2.0 * float(Pr))) * (float(p.T) / T0) * FT_ratio


def rk4_integrate_X_Pr(
    params: GasPBRParams,
    rate_minus_rAprime: Callable[[Dict[str, float]], float],
    W_final: float,
    dW: float = 1e-3,
    X0: float = 0.0,
    Pr0: float = 1.0,
    stop_at_X: Optional[float] = None,
) -> Tuple[List[float], List[float], List[float]]:
    if W_final < 0:
        raise ValueError("W_final must be >= 0.")
    if dW <= 0:
        raise ValueError("dW must be > 0.")
    if not (0.0 <= X0 <= 1.0):
        raise ValueError("X0 must be in [0,1].")
    if Pr0 <= 0:
        raise ValueError("Pr0 must be > 0.")
    if stop_at_X is not None and not (0.0 <= stop_at_X <= 1.0):
        raise ValueError("stop_at_X must be in [0,1].")

    Ws: List[float] = [0.0]
    Xs: List[float] = [float(X0)]
    Prs: List[float] = [float(Pr0)]

    W = 0.0
    X = float(X0)
    Pr = float(Pr0)

    while W < W_final - 1e-15:
        h = min(dW, W_final - W)

        def fX(x: float, pr: float) -> float:
            return _dX_dW(x, pr, params, rate_minus_rAprime)

        def fPr(x: float, pr: float) -> float:
            return _dPr_dW(x, pr, params)

        k1x = fX(X, Pr)
        k1p = fPr(X, Pr)

        k2x = fX(X + 0.5*h*k1x, Pr + 0.5*h*k1p)
        k2p = fPr(X + 0.5*h*k1x, Pr + 0.5*h*k1p)

        k3x = fX(X + 0.5*h*k2x, Pr + 0.5*h*k2p)
        k3p = fPr(X + 0.5*h*k2x, Pr + 0.5*h*k2p)

        k4x = fX(X + h*k3x, Pr + h*k3p)
        k4p = fPr(X + h*k3x, Pr + h*k3p)

        X_next = X + (h/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
        Pr_next = Pr + (h/6.0)*(k1p + 2*k2p + 2*k3p + k4p)

        X_next = min(max(X_next, 0.0), 1.0)
        if Pr_next <= 0:
            raise RuntimeError("Pr dropped to non-positive; check alpha/W range.")

        W = W + h
        X, Pr = X_next, Pr_next

        Ws.append(W)
        Xs.append(X)
        Prs.append(Pr)

        if stop_at_X is not None and X >= stop_at_X - 1e-12:
            break

    return Ws, Xs, Prs


def catalyst_weight_for_target_conversion(
    params: GasPBRParams,
    rate_minus_rAprime: Callable[[Dict[str, float]], float],
    X_target: float,
    W_bounds: Tuple[float, float] = (0.0, 1e6),
    dW_profile: float = 1e-3,
) -> float:
    if not (0.0 <= X_target <= 1.0):
        raise ValueError("X_target must be in [0,1].")
    W_lo, W_hi = W_bounds
    if W_lo < 0 or W_hi <= W_lo:
        raise ValueError("Invalid W_bounds.")

    def resid(W: float) -> float:
        _, Xs, _ = rk4_integrate_X_Pr(
            params=params,
            rate_minus_rAprime=rate_minus_rAprime,
            W_final=W,
            dW=dW_profile,
            X0=0.0,
            Pr0=1.0,
            stop_at_X=None,
        )
        return Xs[-1] - X_target

    rlo = resid(W_lo)
    rhi = resid(W_hi)
    if abs(rlo) < 1e-10:
        return W_lo
    if abs(rhi) < 1e-10:
        return W_hi
    if rlo * rhi > 0:
        raise RuntimeError("Failed to bracket W for X_target; widen W_bounds or adjust dW_profile.")

    return brentq(resid, a=W_lo, b=W_hi, xtol=1e-10, rtol=1e-10, maxiter=2000)