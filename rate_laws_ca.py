# file: rate_laws_ca.py
"""
Constant-volume batch reactor solver for rate laws that depend on CA only.

Model:
    dCA/dt = - rate(CA)

Rate law (base form):
    rate = k * (CA + add_A)**pow_A * Π(extra_constants)

Notes:
- Set pow_A = 1 for first order, pow_A = 2 for second order, etc.
- Set add_A = 0 if not needed.
- extra_constants is a dict like {"K1": 3.2, "K2": 0.5}. They multiply into the rate.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import math


@dataclass(frozen=True)
class CAParams:
    k: float
    pow_A: float = 1.0
    add_A: float = 0.0
    extra_constants: Optional[Dict[str, float]] = None


def _extra_factor(extra_constants: Optional[Dict[str, float]]) -> float:
    if not extra_constants:
        return 1.0
    fac = 1.0
    for _, v in extra_constants.items():
        fac *= float(v)
    return fac


def rate_CA(CA: float, p: CAParams) -> float:
    if CA < 0:
        raise ValueError("CA must be >= 0.")
    base = CA + p.add_A
    if base < 0:
        raise ValueError("CA + add_A must be >= 0 to take real powers.")
    return float(p.k) * (base ** float(p.pow_A)) * _extra_factor(p.extra_constants)


def dCA_dt(t: float, CA: float, p: CAParams) -> float:
    _ = t
    return -rate_CA(CA, p)


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
    t_final: float,
    params: CAParams,
    dt: float = 1e-3,
) -> Tuple[List[float], List[float]]:
    if CA0 < 0:
        raise ValueError("CA0 must be >= 0.")
    if t_final < 0:
        raise ValueError("t_final must be >= 0.")
    return rk4_integrate_scalar(lambda t, CA: dCA_dt(t, CA, params), CA0, 0.0, t_final, dt)


def time_to_CA_target(
    CA0: float,
    CA_target: float,
    params: CAParams,
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

    ts, ys = rk4_integrate_scalar(lambda t, CA: dCA_dt(t, CA, params), CA0, 0.0, t_max, dt, stop_when=stop)

    if ys[-1] > CA_target + 1e-9:
        raise RuntimeError("Did not reach CA_target within t_max. Increase t_max or dt resolution.")

    # Linear interpolate final step for a tighter estimate
    if len(ts) < 2:
        return ts[-1]
    t1, c1 = ts[-2], ys[-2]
    t2, c2 = ts[-1], ys[-1]
    if abs(c2 - c1) < 1e-15:
        return t2
    frac = (c1 - CA_target) / (c1 - c2)
    return t1 + frac * (t2 - t1)


if __name__ == "__main__":
    # Example: 2nd order in CA: dCA/dt = -k CA^2
    p = CAParams(k=0.2, pow_A=2.0, add_A=0.0, extra_constants={"K1": 1.0})
    t_hit = time_to_CA_target(CA0=2.0, CA_target=0.5, params=p, dt=1e-4, t_max=1e4)
    print("t to reach CA=0.5:", t_hit)

    
def fit_k_batch_powerlaw(CA0: float, CAt: float, t: float, n: float) -> float:
    """
    Fit k for constant-volume batch reactor:

        dCA/dt = -k * CA^n

    Parameters
    ----------
    CA0 : initial concentration
    CAt : concentration at time t
    t   : time
    n   : reaction order

    Returns
    -------
    k   : rate constant
    """

    if CA0 <= 0 or CAt <= 0:
        raise ValueError("CA0 and CAt must be positive.")
    if t <= 0:
        raise ValueError("t must be > 0.")

    if abs(n - 1.0) < 1e-12:
        import math
        return math.log(CA0 / CAt) / t
    else:
        one_minus_n = 1.0 - n
        return (CA0**one_minus_n - CAt**one_minus_n) / (one_minus_n * t)
