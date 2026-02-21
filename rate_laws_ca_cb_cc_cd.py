# file: rate_laws_ca_cb_cc_cd.py
"""
Constant-volume batch reactor solver for rate laws that depend on CA, CB, CC, CD.

Solve CA(t), compute CB, CC, CD from stoichiometry.

Rate law (base form):
    rate = k * (CA + add_A)**pow_A * (CB + add_B)**pow_B * (CC + add_C)**pow_C * (CD + add_D)**pow_D
           * Π(extra_constants)

Stoichiometry (constant volume):
    CA = CA0 + nuA*xi
    CB = CB0 + nuB*xi
    CC = CC0 + nuC*xi
    CD = CD0 + nuD*xi
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import math


@dataclass(frozen=True)
class CACBCCCDParams:
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


@dataclass(frozen=True)
class StoichABCD:
    nuA: float
    nuB: float
    nuC: float
    nuD: float


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


def concs_from_CA(CA: float, CA0: float, CB0: float, CC0: float, CD0: float, s: StoichABCD) -> Tuple[float, float, float]:
    xi = _extent_from_CA(CA, CA0, s.nuA)
    CB = CB0 + s.nuB * xi
    CC = CC0 + s.nuC * xi
    CD = CD0 + s.nuD * xi
    return CB, CC, CD


def rate_CACBCCCD(CA: float, CA0: float, CB0: float, CC0: float, CD0: float, s: StoichABCD, p: CACBCCCDParams) -> float:
    if CA < 0:
        raise ValueError("CA must be >= 0.")
    CB, CC, CD = concs_from_CA(CA, CA0, CB0, CC0, CD0, s)
    if CB < -1e-12 or CC < -1e-12 or CD < -1e-12:
        raise ValueError("Computed concentration < 0 from stoichiometry; check nu's and initials.")
    CB, CC, CD = max(0.0, CB), max(0.0, CC), max(0.0, CD)

    a = CA + p.add_A
    b = CB + p.add_B
    c = CC + p.add_C
    d = CD + p.add_D
    if a < 0 or b < 0 or c < 0 or d < 0:
        raise ValueError("Shifted concentrations must be >= 0 to take real powers.")

    return (
        float(p.k)
        * (a ** float(p.pow_A))
        * (b ** float(p.pow_B))
        * (c ** float(p.pow_C))
        * (d ** float(p.pow_D))
        * _extra_factor(p.extra_constants)
    )


def dCA_dt(t: float, CA: float, CA0: float, CB0: float, CC0: float, CD0: float, s: StoichABCD, p: CACBCCCDParams) -> float:
    _ = t
    return -rate_CACBCCCD(CA, CA0, CB0, CC0, CD0, s, p)


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
    CC0: float,
    CD0: float,
    stoich: StoichABCD,
    params: CACBCCCDParams,
    t_final: float,
    dt: float = 1e-3,
) -> Tuple[List[float], List[float]]:
    if min(CA0, CB0, CC0, CD0) < 0:
        raise ValueError("Initial concentrations must be >= 0.")
    if t_final < 0:
        raise ValueError("t_final must be >= 0.")
    f = lambda t, CA: dCA_dt(t, CA, CA0, CB0, CC0, CD0, stoich, params)
    return rk4_integrate_scalar(f, CA0, 0.0, t_final, dt)


def time_to_CA_target(
    CA0: float,
    CB0: float,
    CC0: float,
    CD0: float,
    stoich: StoichABCD,
    params: CACBCCCDParams,
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

    f = lambda t, CA: dCA_dt(t, CA, CA0, CB0, CC0, CD0, stoich, params)
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
    # Example: A + B + C + D -> products, rate = k*CA*CB*CC*CD
    s = StoichABCD(nuA=-1.0, nuB=-1.0, nuC=-1.0, nuD=-1.0)
    p = CACBCCCDParams(k=0.1, pow_A=1.0, pow_B=1.0, pow_C=1.0, pow_D=1.0, extra_constants={"K1": 1.2, "K2": 0.8})
    t_hit = time_to_CA_target(CA0=1.0, CB0=1.0, CC0=1.0, CD0=1.0, stoich=s, params=p, CA_target=0.4, dt=1e-4, t_max=1e6)
    print("t to reach CA=0.4:", t_hit)
