# file: bank/cstr_base.py
"""
CSTR (Mixed Flow Reactor) base functions for liquid-phase, constant-density,
constant volumetric flow, steady state, perfectly mixed, no accumulation.

Core design equation (species A):
    F_A0 - F_A + r_A V = 0

With constant volumetric flow v0:
    F_A = v0 * C_A
    V = v0 * (C_A0 - C_A) / (-r_A)|evaluated at reactor (outlet) concentration

This module provides:
- Generic CSTR sizing using a user-supplied rate function.
- Standard rate-law evaluators (CA, CA*CB, CA*CB*CC, CA*CB*CC*CD) with powers,
  additive shifts, and extra multiplicative constants.
- Stoichiometry helpers so CB/CC/CD can be computed from CA (single reaction).

Conventions:
- Rate functions return (-rA) as a positive number in [mol/L/time].
- v0 in [L/time], concentrations in [mol/L], volume in [L].
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import math


# ----------------------------
# Generic CSTR sizing (A basis)
# ----------------------------

def cstr_volume_from_rate(
    v0: float,
    CA0: float,
    CA_out: float,
    rate_minus_rA: Callable[[float], float],
) -> float:
    """
    Compute CSTR volume from a supplied -rA(CA) evaluator.

    Parameters
    ----------
    v0 : float
        Volumetric flow rate [L/time]
    CA0 : float
        Inlet concentration of A [mol/L]
    CA_out : float
        Outlet (reactor) concentration of A [mol/L]
    rate_minus_rA : callable
        Function f(CA)->(-rA) evaluated at reactor concentration [mol/L/time]

    Returns
    -------
    V : float
        Reactor volume [L]
    """
    if v0 <= 0:
        raise ValueError("v0 must be > 0.")
    if CA0 < 0 or CA_out < 0:
        raise ValueError("Concentrations must be >= 0.")
    if CA_out > CA0 + 1e-15:
        raise ValueError("CA_out must be <= CA0 for consumption of A.")

    r = float(rate_minus_rA(CA_out))
    if r <= 0:
        raise ValueError("(-rA) at CA_out must be > 0.")

    return v0 * (CA0 - CA_out) / r


# --------------------------------
# Rate law evaluators (bank style)
# --------------------------------

def _extra_factor(extra_constants: Optional[Dict[str, float]]) -> float:
    if not extra_constants:
        return 1.0
    fac = 1.0
    for _, v in extra_constants.items():
        fac *= float(v)
    return fac


@dataclass(frozen=True)
class RateCAParams:
    k: float
    pow_A: float = 1.0
    add_A: float = 0.0
    extra_constants: Optional[Dict[str, float]] = None


def rate_CA(CA: float, p: RateCAParams) -> float:
    """Return (-rA) = k*(CA+add_A)^pow_A * Π(extra_constants)."""
    if CA < 0:
        raise ValueError("CA must be >= 0.")
    base = CA + p.add_A
    if base < 0:
        raise ValueError("CA + add_A must be >= 0.")
    return float(p.k) * (base ** float(p.pow_A)) * _extra_factor(p.extra_constants)


@dataclass(frozen=True)
class RateCACBParams:
    k: float
    pow_A: float = 1.0
    pow_B: float = 1.0
    add_A: float = 0.0
    add_B: float = 0.0
    extra_constants: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class RateCACBCCParams:
    k: float
    pow_A: float = 1.0
    pow_B: float = 1.0
    pow_C: float = 1.0
    add_A: float = 0.0
    add_B: float = 0.0
    add_C: float = 0.0
    extra_constants: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class RateCACBCCCDParams:
    k: float
    pow_A: float = 1.0
    pow_B: float = 1.0
    pow_C: float = 1.0
    pow_D: float = 1.0
    add_A: float = 0.0
    add_B: float = 0.0
    add_C: float = 0.0
    add_D: float = 0.0
    extra_constants: Optional[Dict[str, float]] = None


# --------------------------
# Single-reaction stoich map
# --------------------------

@dataclass(frozen=True)
class StoichAB:
    nuA: float
    nuB: float


@dataclass(frozen=True)
class StoichABC:
    nuA: float
    nuB: float
    nuC: float


@dataclass(frozen=True)
class StoichABCD:
    nuA: float
    nuB: float
    nuC: float
    nuD: float


def _extent_from_CA(CA: float, CA0: float, nuA: float) -> float:
    if abs(nuA) < 1e-15:
        raise ValueError("nuA must be nonzero.")
    return (CA - CA0) / nuA


def conc_B_from_CA(CA: float, CA0: float, CB0: float, s: StoichAB) -> float:
    xi = _extent_from_CA(CA, CA0, s.nuA)
    return CB0 + s.nuB * xi


def concs_BC_from_CA(CA: float, CA0: float, CB0: float, CC0: float, s: StoichABC) -> Tuple[float, float]:
    xi = _extent_from_CA(CA, CA0, s.nuA)
    return CB0 + s.nuB * xi, CC0 + s.nuC * xi


def concs_BCD_from_CA(CA: float, CA0: float, CB0: float, CC0: float, CD0: float, s: StoichABCD) -> Tuple[float, float, float]:
    xi = _extent_from_CA(CA, CA0, s.nuA)
    return CB0 + s.nuB * xi, CC0 + s.nuC * xi, CD0 + s.nuD * xi


# ----------------------------
# Rate with stoich dependencies
# ----------------------------

def rate_CACB_from_CA(
    CA: float, CA0: float, CB0: float, s: StoichAB, p: RateCACBParams
) -> float:
    CB = conc_B_from_CA(CA, CA0, CB0, s)
    if CB < -1e-12:
        raise ValueError("Computed CB < 0; check stoichiometry/initials.")
    CB = max(0.0, CB)

    a = CA + p.add_A
    b = CB + p.add_B
    if a < 0 or b < 0:
        raise ValueError("Shifted concentrations must be >= 0.")

    return float(p.k) * (a ** float(p.pow_A)) * (b ** float(p.pow_B)) * _extra_factor(p.extra_constants)


def rate_CACBCC_from_CA(
    CA: float, CA0: float, CB0: float, CC0: float, s: StoichABC, p: RateCACBCCParams
) -> float:
    CB, CC = concs_BC_from_CA(CA, CA0, CB0, CC0, s)
    if CB < -1e-12 or CC < -1e-12:
        raise ValueError("Computed concentration < 0; check stoichiometry/initials.")
    CB, CC = max(0.0, CB), max(0.0, CC)

    a = CA + p.add_A
    b = CB + p.add_B
    c = CC + p.add_C
    if a < 0 or b < 0 or c < 0:
        raise ValueError("Shifted concentrations must be >= 0.")

    return (
        float(p.k)
        * (a ** float(p.pow_A))
        * (b ** float(p.pow_B))
        * (c ** float(p.pow_C))
        * _extra_factor(p.extra_constants)
    )


def rate_CACBCCCD_from_CA(
    CA: float, CA0: float, CB0: float, CC0: float, CD0: float, s: StoichABCD, p: RateCACBCCCDParams
) -> float:
    CB, CC, CD = concs_BCD_from_CA(CA, CA0, CB0, CC0, CD0, s)
    if CB < -1e-12 or CC < -1e-12 or CD < -1e-12:
        raise ValueError("Computed concentration < 0; check stoichiometry/initials.")
    CB, CC, CD = max(0.0, CB), max(0.0, CC), max(0.0, CD)

    a = CA + p.add_A
    b = CB + p.add_B
    c = CC + p.add_C
    d = CD + p.add_D
    if a < 0 or b < 0 or c < 0 or d < 0:
        raise ValueError("Shifted concentrations must be >= 0.")

    return (
        float(p.k)
        * (a ** float(p.pow_A))
        * (b ** float(p.pow_B))
        * (c ** float(p.pow_C))
        * (d ** float(p.pow_D))
        * _extra_factor(p.extra_constants)
    )


# -----------------------------
# Convenience: CSTR volume calls
# -----------------------------

def cstr_volume_CA(v0: float, CA0: float, CA_out: float, p: RateCAParams) -> float:
    return cstr_volume_from_rate(v0, CA0, CA_out, lambda CA: rate_CA(CA, p))


def cstr_volume_CACB(
    v0: float, CA0: float, CB0: float, CA_out: float, stoich: StoichAB, p: RateCACBParams
) -> float:
    return cstr_volume_from_rate(v0, CA0, CA_out, lambda CA: rate_CACB_from_CA(CA, CA0, CB0, stoich, p))


def cstr_volume_CACBCC(
    v0: float, CA0: float, CB0: float, CC0: float, CA_out: float, stoich: StoichABC, p: RateCACBCCParams
) -> float:
    return cstr_volume_from_rate(v0, CA0, CA_out, lambda CA: rate_CACBCC_from_CA(CA, CA0, CB0, CC0, stoich, p))


def cstr_volume_CACBCCCD(
    v0: float, CA0: float, CB0: float, CC0: float, CD0: float, CA_out: float, stoich: StoichABCD, p: RateCACBCCCDParams
) -> float:
    return cstr_volume_from_rate(v0, CA0, CA_out, lambda CA: rate_CACBCCCD_from_CA(CA, CA0, CB0, CC0, CD0, stoich, p))


# -------------------------------------------------------
# Optional helper for rational forms used in many problems
# -------------------------------------------------------

def rate_rational_CA(CA: float, k1: float, a: float, K2: float) -> float:
    """
    Common form:
        (-rA) = (k1 * CA) / (a + K2 * CA)
    """
    if CA < 0:
        raise ValueError("CA must be >= 0.")
    denom = a + K2 * CA
    if denom <= 0:
        raise ValueError("a + K2*CA must be > 0.")
    return (k1 * CA) / denom


def cstr_volume_rational_CA(v0: float, CA0: float, CA_out: float, k1: float, a: float, K2: float) -> float:
    return cstr_volume_from_rate(v0, CA0, CA_out, lambda CA: rate_rational_CA(CA, k1, a, K2))


if __name__ == "__main__":
    # Example from your prompt:
    # -rA = (3.5*CA)/(1+0.5*CA), v0=25 L/min, CA0=22, CA_out=0.5
    v0 = 25.0
    CA0 = 22.0
    CA_out = 0.5

    r = lambda CA: (3.5 * CA) / (1.0 + 0.5 * CA)
    V = cstr_volume_from_rate(v0, CA0, CA_out, r)
    print("V (L) =", V, "Rounded =", round(V, 1))
