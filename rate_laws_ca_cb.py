# file: rate_laws_ca_cb.py
"""
Constant-volume batch reactor solver for rate laws that depend on CA and CB.

Core idea:
- Solve for CA(t).
- Compute CB from stoichiometry each step (constant volume):
      CA = CA0 + nuA * xi
      CB = CB0 + nuB * xi
  where xi is extent, and nuA < 0 for a reactant A.

Rate law (base form):
    rate = k * (CA + add_A)**pow_A * (CB + add_B)**pow_B * Π(extra_constants)

You supply:
- initial concentrations CA0, CB0
- stoichiometric coefficients nuA, nuB (e.g., for A + 2B -> products: nuA=-1, nuB=-2)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import math


@dataclass(frozen=True)
class CACBParams:
    k: float
    pow_A: float = 1.0
    pow_B: float = 1.0
    add_A: float = 0.0
    add_B: float = 0.0
    extra_constants: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class StoichAB:
    nuA: float  # typically negative for reactant
    nuB: float  # typically negative for reactant


def _extra_factor(extra_constants: Optional[Dict[str, float]]) -> float:
    if not extra_constants:
        return 1.0
    fac = 1.0
    for _, v in extra_constants.items():
        fac *= float(v)
    return fac


def _extent_from_CA(CA: float, CA0: float, nuA: float) -> float:
    if abs(nuA) < 1e-15:
        raise ValueError("nuA must be nonzero.")
    return (CA - CA0) / nuA


def conc_B_from_CA(CA: float, CA0: float, CB0: float, s: StoichAB) -> float:
    xi = _extent_from_CA(CA, CA0, s.nuA)
    return CB0 + s.nuB * xi


def rate_CACB(CA: float, CA0: float, CB0: float, s: StoichAB, p: CACBParams) -> float:
    if CA < 0:
        raise ValueError("CA must be >= 0.")
    CB = conc_B_from_CA(CA, CA0, CB0, s)
    if CB < -1e-12:
        raise ValueError("Computed CB < 0 from stoichiometry; check nu's and initials.")
    CB = max(0.0, CB)

    a = CA + p.add_A
    b = CB + p.add_B
    if a < 0 or b < 0:
        raise ValueError("Shifted concentrations must be >= 0 to take real powers.")

    return float(p.k) * (a ** float(p.pow_A)) * (b ** float(p.pow_B)) * _extra_factor(p.extra_constants)


def dCA_dt(t: float, CA: float, CA0: float, CB0: float, s: StoichAB, p: CACBParams) -> float:
    _ = t
    return -rate_CACB(CA, CA0, CB0, s, p)


def rk4_integrate_scalar(
    f: Callable[[float, float], float],
    y0: float,
    t0: float,
    t1: float,
    dt: float,
    stop_when: Optional[Callable[[float, float], bool]] = None,
) -> Tuple[List[float], List[float]]:
    if dt <= 0:
        raise ValueError("dt must be > 0.")
    if t1 < t0:
        raise ValueError("t1 must be >= t0.")

    ts: List[float] = [t0]
    ys: List[float] = [y0]

    t = t0
    y = y0

    while t < t1 - 1e-15:
        h = min(dt, t1 - t)

        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)

        y_next = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t_next = t + h

        if y_next < 0:
            y_next = 0.0

        ts.append(t_next)
        ys.append(y_next)
        t, y = t_next, y_next

        if stop_when and stop_when(t, y):
            break

    return ts, ys


def solve_CA_over_time(
    CA0: float,
    CB0: float,
    stoich: StoichAB,
    params: CACBParams,
    t_final: float,
    dt: float = 1e-3,
) -> Tuple[List[float], List[float]]:
    if CA0 < 0 or CB0 < 0:
        raise ValueError("Initial concentrations must be >= 0.")
    if t_final < 0:
        raise ValueError("t_final must be >= 0.")
    f = lambda t, CA: dCA_dt(t, CA, CA0, CB0, stoich, params)
    return rk4_integrate_scalar(f, CA0, 0.0, t_final, dt)


def time_to_CA_target(
    CA0: float,
    CB0: float,
    stoich: StoichAB,
    params: CACBParams,
    CA_target: float,
    dt: float = 1e-3,
    t_max: float = 1e6,
) -> float:
    if CA_target < 0:
        raise ValueError("CA_target must be >= 0.")
    if CA_target > CA0 + 1e-15:
        raise ValueError("CA_target must be <= CA0 for decay problems.")
    if abs(CA_target - CA0) <= 1e-15:
        return 0.0

    def stop(t: float, CA: float) -> bool:
        return CA <= CA_target

    f = lambda t, CA: dCA_dt(t, CA, CA0, CB0, stoich, params)
    ts, ys = rk4_integrate_scalar(f, CA0, 0.0, t_max, dt, stop_when=stop)

    if ys[-1] > CA_target + 1e-9:
        raise RuntimeError("Did not reach CA_target within t_max. Increase t_max or dt resolution.")

    if len(ts) < 2:
        return ts[-1]
    t1, c1 = ts[-2], ys[-2]
    t2, c2 = ts[-1], ys[-1]
    if abs(c2 - c1) < 1e-15:
        return t2
    frac = (c1 - CA_target) / (c1 - c2)
    return t1 + frac * (t2 - t1)


if __name__ == "__main__":
    # Example: A + B -> products, rate = k*CA*CB
    s = StoichAB(nuA=-1.0, nuB=-1.0)
    p = CACBParams(k=0.5, pow_A=1.0, pow_B=1.0, extra_constants={"K1": 1.0, "K2": 1.0})
    t_hit = time_to_CA_target(CA0=1.0, CB0=2.0, stoich=s, params=p, CA_target=0.2, dt=1e-4, t_max=1e4)
    print("t to reach CA=0.2:", t_hit)
