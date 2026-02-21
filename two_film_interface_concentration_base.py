# bank/two_film_interface_concentrations_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math


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


def logmean_one_minus(v1: float, v2: float) -> float:
    _vf("v1", v1); _vf("v2", v2)
    return logmean(1.0 - v1, 1.0 - v2)


@dataclass(frozen=True)
class TwoFilmInterfaceSpec:
    """
    Two-film interface concentrations for linear equilibrium y = m x.

    Bulk point: (x, y)

    Film coefficients per packed volume (or per area if consistent):
      k_ya, k_xa

    Dilute (default):
      Flux equality: k_ya (y - y_i) = k_xa (x_i - x)
      Interface equilibrium: y_i = m x_i
      Closed form:
        x_i = (k_ya y + k_xa x) / (k_xa + k_ya m)
        y_i = m x_i

    Optional non-dilute correction (matches lecture style with log-mean one-minus factors):
      Use iterative slope update:
        slope = - (k_xa (1-x)_M) / (k_ya (1-y)_M)
      For linear equilibrium, intersection with y = m x is analytic each iteration:
        y + slope (x_i - x) = m x_i  ->  x_i = (y - slope x) / (m - slope)
    """
    x: float
    y: float
    m: float
    k_ya: float
    k_xa: float
    non_dilute_iterate: bool = False
    it_max: int = 50
    tol: float = 1e-10


def interface_concentrations(spec: TwoFilmInterfaceSpec) -> Dict[str, Any]:
    _vf("x", spec.x)
    _vf("y", spec.y)
    _vp("m", spec.m)
    _vp("k_ya", spec.k_ya)
    _vp("k_xa", spec.k_xa)

    if not spec.non_dilute_iterate:
        x_i = (spec.k_ya * spec.y + spec.k_xa * spec.x) / (spec.k_xa + spec.k_ya * spec.m)
        _vf("x_i", x_i)
        y_i = spec.m * x_i
        _vf("y_i", y_i)
        return {
            "method": "closed_form_dilute",
            "bulk": {"x": spec.x, "y": spec.y},
            "interface": {"x_i": x_i, "y_i": y_i},
            "params": {"m": spec.m, "k_ya": spec.k_ya, "k_xa": spec.k_xa},
        }

    # iterative non-dilute style
    slope = - (spec.k_xa * (1.0 - spec.x)) / (spec.k_ya * (1.0 - spec.y))
    hist = []

    x_i = None
    y_i = None

    for it in range(1, spec.it_max + 1):
        denom = (spec.m - slope)
        if abs(denom) < 1e-14:
            raise ValueError("m - slope too close to 0; cannot intersect reliably.")
        x_i_new = (spec.y - slope * spec.x) / denom
        if not (0.0 <= x_i_new < 1.0):
            raise ValueError("Iterated x_i out of [0,1); check inputs.")
        y_i_new = spec.m * x_i_new

        one_minus_y_M = logmean_one_minus(y_i_new, spec.y)
        one_minus_x_M = logmean_one_minus(spec.x, x_i_new)

        slope_new = - (spec.k_xa * one_minus_x_M) / (spec.k_ya * one_minus_y_M)

        hist.append(
            {
                "iter": it,
                "slope": slope,
                "x_i": x_i_new,
                "y_i": y_i_new,
                "one_minus_y_M": one_minus_y_M,
                "one_minus_x_M": one_minus_x_M,
                "slope_new": slope_new,
            }
        )

        if abs(slope_new - slope) <= spec.tol * max(1.0, abs(slope)):
            x_i, y_i, slope = x_i_new, y_i_new, slope_new
            break

        x_i, y_i, slope = x_i_new, y_i_new, slope_new

    assert x_i is not None and y_i is not None

    return {
        "method": "iterative_non_dilute_logmean_one_minus",
        "bulk": {"x": spec.x, "y": spec.y},
        "interface": {"x_i": x_i, "y_i": y_i},
        "params": {"m": spec.m, "k_ya": spec.k_ya, "k_xa": spec.k_xa},
        "history": hist,
    }