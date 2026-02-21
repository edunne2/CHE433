# file: bank/cstr_series_base.py
"""
CSTRs in series (constant-density liquid, constant volumetric flow, steady state)

Supports solving N identical ideal CSTRs in series for single-reaction systems using
rate laws expressed as (-rA) evaluated at each reactor outlet.

Design equation per reactor j:
    V = v0 * (CA_in - CA_out) / (-rA_out)

For general nonlinear (-rA_out) (powers, additive shifts, extra constants, stoich-based CB/CC),
CA_out is solved numerically on [0, CA_in].

This module provides:
- Forward simulation through N CSTRs: outlet concentrations after N reactors
- Solve for k to hit an overall conversion target

Uses evaluators/stoich helpers from bank.cstr_base.

Conventions:
- rate functions return (-rA) as a positive number [mol/L/time]
- v0 [L/time], V [L], concentrations [mol/L]
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, Tuple, Optional
from scipy.optimize import brentq

from bank.cstr_base import (
    RateCAParams,
    RateCACBParams,
    RateCACBCCParams,
    StoichAB,
    StoichABC,
    rate_CA,
    rate_CACB_from_CA,
    rate_CACBCC_from_CA,
    conc_B_from_CA,
    concs_BC_from_CA,
)

# --------------------------
# Core: solve one CSTR stage
# --------------------------

def _solve_stage_CA_out(
    v0: float,
    V: float,
    CA_in: float,
    rate_minus_rA_of_CAout: Callable[[float], float],
) -> float:
    """
    Solve CA_out for one CSTR stage from:
        v0*(CA_in - CA_out)/V = (-rA)(CA_out)
    """
    if v0 <= 0:
        raise ValueError("v0 must be > 0.")
    if V <= 0:
        raise ValueError("V must be > 0.")
    if CA_in < 0:
        raise ValueError("CA_in must be >= 0.")
    if CA_in == 0.0:
        return 0.0

    def g(CA_out: float) -> float:
        if CA_out < 0:
            return 1e30
        lhs = v0 * (CA_in - CA_out) / V
        rhs = float(rate_minus_rA_of_CAout(CA_out))
        return lhs - rhs

    # Bracket on [0, CA_in]. For physical monotone consumption, g(0) >= 0 and g(CA_in) <= 0.
    g0 = g(0.0)
    g1 = g(CA_in)

    if abs(g0) < 1e-14:
        return 0.0
    if abs(g1) < 1e-14:
        return CA_in

    # If bracketing fails due to pathological parameters, expand slightly toward 0.
    if g0 * g1 > 0:
        # Try a small positive epsilon instead of exactly 0
        eps = max(1e-12, 1e-9 * CA_in)
        g_eps = g(eps)
        if g_eps * g1 > 0:
            raise RuntimeError("Failed to bracket CA_out root in [0, CA_in]. Check kinetics/inputs.")
        return brentq(g, a=eps, b=CA_in, xtol=1e-12, rtol=1e-12, maxiter=2000)

    return brentq(g, a=0.0, b=CA_in, xtol=1e-12, rtol=1e-12, maxiter=2000)


# --------------------------
# One-stage wrappers (A only)
# --------------------------

def cstr_stage_CA(
    v0: float,
    V: float,
    CA_in: float,
    p: RateCAParams,
) -> float:
    return _solve_stage_CA_out(v0, V, CA_in, lambda CA: rate_CA(CA, p))


# ---------------------------------------
# One-stage wrappers with stoich (A,B,C)
# ---------------------------------------

def cstr_stage_CACB(
    v0: float,
    V: float,
    CA_in: float,
    CB_in: float,
    stoich: StoichAB,
    p: RateCACBParams,
) -> Tuple[float, float]:
    """
    Returns (CA_out, CB_out).
    Stoichiometry assumed single reaction; CB tied to CA by extent computed from CA.
    """
    def minus_rA(CA_out: float) -> float:
        return rate_CACB_from_CA(CA_out, CA_in, CB_in, stoich, p)

    CA_out = _solve_stage_CA_out(v0, V, CA_in, minus_rA)
    CB_out = conc_B_from_CA(CA_out, CA_in, CB_in, stoich)
    if CB_out < -1e-12:
        raise ValueError("Computed CB_out < 0; check stoichiometry/inputs.")
    return CA_out, max(0.0, CB_out)


def cstr_stage_CACBCC(
    v0: float,
    V: float,
    CA_in: float,
    CB_in: float,
    CC_in: float,
    stoich: StoichABC,
    p: RateCACBCCParams,
) -> Tuple[float, float, float]:
    """
    Returns (CA_out, CB_out, CC_out).
    """
    def minus_rA(CA_out: float) -> float:
        return rate_CACBCC_from_CA(CA_out, CA_in, CB_in, CC_in, stoich, p)

    CA_out = _solve_stage_CA_out(v0, V, CA_in, minus_rA)
    CB_out, CC_out = concs_BC_from_CA(CA_out, CA_in, CB_in, CC_in, stoich)
    if CB_out < -1e-12 or CC_out < -1e-12:
        raise ValueError("Computed outlet concentration < 0; check stoichiometry/inputs.")
    return CA_out, max(0.0, CB_out), max(0.0, CC_out)


# --------------------------
# N-stage series simulators
# --------------------------

def cstr_series_CA(
    v0: float,
    V_each: float,
    CA0: float,
    N: int,
    p: RateCAParams,
) -> float:
    if N <= 0:
        raise ValueError("N must be >= 1.")
    CA = float(CA0)
    for _ in range(N):
        CA = cstr_stage_CA(v0, V_each, CA, p)
    return CA


def cstr_series_CACB(
    v0: float,
    V_each: float,
    CA0: float,
    CB0: float,
    N: int,
    stoich: StoichAB,
    p: RateCACBParams,
) -> Tuple[float, float]:
    if N <= 0:
        raise ValueError("N must be >= 1.")
    CA, CB = float(CA0), float(CB0)
    for _ in range(N):
        CA, CB = cstr_stage_CACB(v0, V_each, CA, CB, stoich, p)
    return CA, CB


def cstr_series_CACBCC(
    v0: float,
    V_each: float,
    CA0: float,
    CB0: float,
    CC0: float,
    N: int,
    stoich: StoichABC,
    p: RateCACBCCParams,
) -> Tuple[float, float, float]:
    if N <= 0:
        raise ValueError("N must be >= 1.")
    CA, CB, CC = float(CA0), float(CB0), float(CC0)
    for _ in range(N):
        CA, CB, CC = cstr_stage_CACBCC(v0, V_each, CA, CB, CC, stoich, p)
    return CA, CB, CC


# --------------------------
# Helpers: conversion and k
# --------------------------

def conversion_from_CA(CA0: float, CA_out: float) -> float:
    if CA0 <= 0:
        raise ValueError("CA0 must be > 0.")
    if CA_out < 0:
        raise ValueError("CA_out must be >= 0.")
    return (CA0 - CA_out) / CA0


def solve_k_for_conversion_CA(
    v0: float,
    V_each: float,
    CA0: float,
    N: int,
    X_target: float,
    params_without_k: RateCAParams,
    k_bounds: Tuple[float, float] = (1e-12, 1e6),
) -> float:
    """
    Solve k so that N CSTRs in series achieve overall conversion X_target for an A-only rate.
    Supports any pow_A (first, second, arbitrary), additive shift add_A, and extra_constants.
    """
    if not (0.0 <= X_target < 1.0):
        raise ValueError("X_target must be in [0, 1).")
    CA_target = CA0 * (1.0 - X_target)

    k_lo, k_hi = k_bounds
    if k_lo <= 0 or k_hi <= 0 or k_hi <= k_lo:
        raise ValueError("Invalid k_bounds.")

    def h(k: float) -> float:
        p = replace(params_without_k, k=float(k))
        CA_out = cstr_series_CA(v0, V_each, CA0, N, p)
        return CA_out - CA_target

    # Ensure bracket: h(k_lo) > 0 (too little reaction), h(k_hi) < 0 (too much reaction)
    hlo = h(k_lo)
    hhi = h(k_hi)
    if hlo == 0.0:
        return k_lo
    if hhi == 0.0:
        return k_hi
    if hlo * hhi > 0:
        raise RuntimeError("Failed to bracket k. Widen k_bounds.")

    return brentq(h, a=k_lo, b=k_hi, xtol=1e-12, rtol=1e-12, maxiter=2000)


def solve_k_for_conversion_CACB(
    v0: float,
    V_each: float,
    CA0: float,
    CB0: float,
    N: int,
    X_target: float,
    stoich: StoichAB,
    params_without_k: RateCACBParams,
    k_bounds: Tuple[float, float] = (1e-12, 1e6),
) -> float:
    """
    Solve k so that N CSTRs in series achieve X_target for rate depending on CA and CB (CB from stoich).
    Works for first/second/arbitrary powers via pow_A, pow_B and additive shifts add_A/add_B.
    """
    if not (0.0 <= X_target < 1.0):
        raise ValueError("X_target must be in [0, 1).")
    CA_target = CA0 * (1.0 - X_target)

    k_lo, k_hi = k_bounds
    if k_lo <= 0 or k_hi <= 0 or k_hi <= k_lo:
        raise ValueError("Invalid k_bounds.")

    def h(k: float) -> float:
        p = replace(params_without_k, k=float(k))
        CA_out, _ = cstr_series_CACB(v0, V_each, CA0, CB0, N, stoich, p)
        return CA_out - CA_target

    hlo = h(k_lo)
    hhi = h(k_hi)
    if hlo == 0.0:
        return k_lo
    if hhi == 0.0:
        return k_hi
    if hlo * hhi > 0:
        raise RuntimeError("Failed to bracket k. Widen k_bounds.")

    return brentq(h, a=k_lo, b=k_hi, xtol=1e-12, rtol=1e-12, maxiter=2000)


def solve_k_for_conversion_CACBCC(
    v0: float,
    V_each: float,
    CA0: float,
    CB0: float,
    CC0: float,
    N: int,
    X_target: float,
    stoich: StoichABC,
    params_without_k: RateCACBCCParams,
    k_bounds: Tuple[float, float] = (1e-12, 1e6),
) -> float:
    """
    Solve k so that N CSTRs in series achieve X_target for rate depending on CA,CB,CC (CB/CC from stoich).
    Works for first/second/arbitrary powers via pow_A, pow_B, pow_C and additive shifts add_*.
    """
    if not (0.0 <= X_target < 1.0):
        raise ValueError("X_target must be in [0, 1).")
    CA_target = CA0 * (1.0 - X_target)

    k_lo, k_hi = k_bounds
    if k_lo <= 0 or k_hi <= 0 or k_hi <= k_lo:
        raise ValueError("Invalid k_bounds.")

    def h(k: float) -> float:
        p = replace(params_without_k, k=float(k))
        CA_out, _, _ = cstr_series_CACBCC(v0, V_each, CA0, CB0, CC0, N, stoich, p)
        return CA_out - CA_target

    hlo = h(k_lo)
    hhi = h(k_hi)
    if hlo == 0.0:
        return k_lo
    if hhi == 0.0:
        return k_hi
    if hlo * hhi > 0:
        raise RuntimeError("Failed to bracket k. Widen k_bounds.")

    return brentq(h, a=k_lo, b=k_hi, xtol=1e-12, rtol=1e-12, maxiter=2000)
