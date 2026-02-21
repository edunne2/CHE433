# file: bank/pfr_base.py
"""
PFR base functions.

Two common cases:

(A) Liquid-phase PFR / constant-density with constant volumetric flow v0:
    Design equation:
        dF_A/dV = r_A
    With F_A = v0 * C_A  (v0 constant):
        v0 dC_A/dV = r_A
        dC_A/dV = r_A / v0 = -(-rA)/v0

    Volume to reach CA_out:
        V = ∫_{CA_out}^{CA0} v0 / (-rA(CA)) dCA

(B) Gas-phase / variable volumetric flow:
    Not handled here (needs v(C), EOS, pressure drop, etc.).

This module provides:
- Generic PFR sizing from a supplied -rA(CA).
- Optional profile integration CA(V) using RK4.

Conventions:
- rate functions return (-rA) positive in [mol/L/time]
- v0 in [L/time], concentrations in [mol/L], reactor volume in [L]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import math


def pfr_volume_from_rate(
    v0: float,
    CA0: float,
    CA_out: float,
    rate_minus_rA: Callable[[float], float],
    n_steps: int = 20000,
) -> float:
    """
    Compute PFR volume with constant volumetric flow:
        V = ∫ v0/(-rA(CA)) dCA  from CA_out to CA0

    Uses composite Simpson's rule (requires even n_steps).
    """
    if v0 <= 0:
        raise ValueError("v0 must be > 0.")
    if CA0 < 0 or CA_out < 0:
        raise ValueError("Concentrations must be >= 0.")
    if CA_out > CA0 + 1e-15:
        raise ValueError("CA_out must be <= CA0 for consumption of A.")
    if abs(CA_out - CA0) <= 1e-15:
        return 0.0

    if n_steps < 2:
        raise ValueError("n_steps must be >= 2.")
    if n_steps % 2 == 1:
        n_steps += 1  # make it even for Simpson

    a = float(CA_out)
    b = float(CA0)
    h = (b - a) / n_steps

    def integrand(CA: float) -> float:
        r = float(rate_minus_rA(CA))
        if r <= 0:
            raise ValueError("(-rA) must be > 0 over the integration interval.")
        return v0 / r

    s = integrand(a) + integrand(b)
    for i in range(1, n_steps):
        x = a + i * h
        s += (4 if i % 2 == 1 else 2) * integrand(x)

    return (h / 3.0) * s


def rk4_integrate_scalar(
    f: Callable[[float, float], float],
    y0: float,
    x0: float,
    x1: float,
    dx: float,
    stop_when: Optional[Callable[[float, float], bool]] = None,
) -> Tuple[List[float], List[float]]:
    if dx <= 0:
        raise ValueError("dx must be > 0.")
    if x1 < x0:
        raise ValueError("x1 must be >= x0.")

    xs: List[float] = [x0]
    ys: List[float] = [y0]

    x = x0
    y = y0

    while x < x1 - 1e-15:
        h = min(dx, x1 - x)

        k1 = f(x, y)
        k2 = f(x + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(x + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(x + h, y + h * k3)

        y_next = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_next = x + h

        if y_next < 0:
            y_next = 0.0

        xs.append(x_next)
        ys.append(y_next)
        x, y = x_next, y_next

        if stop_when and stop_when(x, y):
            break

    return xs, ys


def pfr_profile_CA_vs_V(
    v0: float,
    CA0: float,
    V_final: float,
    rate_minus_rA: Callable[[float], float],
    dV: float = 1e-3,
) -> Tuple[List[float], List[float]]:
    """
    Integrate CA(V) using:
        dCA/dV = -( -rA(CA) ) / v0
    """
    if V_final < 0:
        raise ValueError("V_final must be >= 0.")
    if CA0 < 0:
        raise ValueError("CA0 must be >= 0.")
    if v0 <= 0:
        raise ValueError("v0 must be > 0.")

    def dCA_dV(V: float, CA: float) -> float:
        _ = V
        r = float(rate_minus_rA(CA))
        if r < 0:
            raise ValueError("(-rA) must be >= 0.")
        return -r / v0

    return rk4_integrate_scalar(dCA_dV, CA0, 0.0, V_final, dV)


if __name__ == "__main__":
    # Example check: same rational form used earlier
    v0 = 25.0
    CA0 = 22.0
    CA_out = 0.5
    r = lambda CA: (3.5 * CA) / (1.0 + 0.5 * CA)

    V = pfr_volume_from_rate(v0, CA0, CA_out, r, n_steps=20000)
    print("PFR V (L) =", V)
