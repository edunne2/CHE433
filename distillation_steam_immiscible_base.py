# bank/distillation_steam_immiscible_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Dict, Any, Optional
import math


def _vp(name: str, v: float) -> None:
    if v <= 0:
        raise ValueError(f"{name} must be > 0.")


def _bisection(fn, a: float, b: float, tol: float = 1e-10, maxiter: int = 300) -> float:
    fa = fn(a)
    fb = fn(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError("Bisection requires sign change.")
    lo, hi = a, b
    flo, fhi = fa, fb
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fm = fn(mid)
        if abs(fm) <= tol:
            return mid
        if flo * fm <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class VaporPressureTable:
    """
    Tabulated Psat(T) for one component.
    - T: any consistent temperature unit (K recommended)
    - Psat: any consistent pressure unit (kPa recommended)
    - log_interp=True: ln(Psat) linear in T between points
    Extrapolates outside the table using the nearest segment.
    """
    T: Sequence[float]
    Psat: Sequence[float]
    log_interp: bool = True

    def __post_init__(self) -> None:
        if len(self.T) != len(self.Psat) or len(self.T) < 2:
            raise ValueError("T and Psat must have same length >= 2.")
        pairs = sorted((float(t), float(p)) for t, p in zip(self.T, self.Psat))
        Ts = [t for t, _ in pairs]
        Ps = [p for _, p in pairs]
        for i, p in enumerate(Ps):
            _vp(f"Psat[{i}]", p)
        object.__setattr__(self, "_T", Ts)
        object.__setattr__(self, "_P", Ps)

    def Psat_of_T(self, T: float) -> float:
        T = float(T)
        Ts = self._T
        Ps = self._P

        if T <= Ts[0]:
            i = 0
        elif T >= Ts[-1]:
            i = len(Ts) - 2
        else:
            lo, hi = 0, len(Ts) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if Ts[mid] <= T:
                    lo = mid
                else:
                    hi = mid
            i = lo

        T0, T1 = Ts[i], Ts[i + 1]
        P0, P1 = Ps[i], Ps[i + 1]
        if abs(T1 - T0) < 1e-18:
            return 0.5 * (P0 + P1)

        w = (T - T0) / (T1 - T0)
        if self.log_interp:
            lnP = math.log(P0) + w * (math.log(P1) - math.log(P0))
            return math.exp(lnP)
        return P0 + w * (P1 - P0)


@dataclass(frozen=True)
class ImmiscibleDistillationSpec:
    """
    Immiscible (steam) distillation at total pressure P_total:

      P_total = Psat_A(T) + Psat_B(T)

    Vapor composition:
      y_A = Psat_A(T)/P_total
      y_B = Psat_B(T)/P_total
    """
    P_total: float
    vp_A: VaporPressureTable
    vp_B: VaporPressureTable
    T_lo: Optional[float] = None
    T_hi: Optional[float] = None
    tol: float = 1e-10
    maxiter: int = 300


def solve_immiscible_distillation(spec: ImmiscibleDistillationSpec) -> Dict[str, Any]:
    _vp("P_total", spec.P_total)

    Tmin = min(spec.vp_A._T[0], spec.vp_B._T[0]) if spec.T_lo is None else float(spec.T_lo)
    Tmax = max(spec.vp_A._T[-1], spec.vp_B._T[-1]) if spec.T_hi is None else float(spec.T_hi)
    if Tmax <= Tmin:
        raise ValueError("Require T_hi > T_lo.")

    def f(T: float) -> float:
        return spec.vp_A.Psat_of_T(T) + spec.vp_B.Psat_of_T(T) - spec.P_total

    fa, fb = f(Tmin), f(Tmax)

    if fa == 0.0:
        T = Tmin
    elif fb == 0.0:
        T = Tmax
    else:
        if fa * fb > 0.0:
            # expand symmetrically until sign change
            center = 0.5 * (Tmin + Tmax)
            span = 0.5 * (Tmax - Tmin)
            expand = 1.6
            for _ in range(80):
                span *= expand
                a = center - span
                b = center + span
                fa2, fb2 = f(a), f(b)
                if fa2 == 0.0:
                    T = a
                    break
                if fb2 == 0.0:
                    T = b
                    break
                if fa2 * fb2 < 0.0:
                    T = _bisection(f, a, b, tol=spec.tol, maxiter=spec.maxiter)
                    break
            else:
                raise ValueError("Could not bracket root for Psat_A(T)+Psat_B(T)=P_total.")
        else:
            T = _bisection(f, Tmin, Tmax, tol=spec.tol, maxiter=spec.maxiter)

    PA = spec.vp_A.Psat_of_T(T)
    PB = spec.vp_B.Psat_of_T(T)

    return {
        "inputs": {"P_total": spec.P_total},
        "T": T,
        "partial_pressures": {"P_A": PA, "P_B": PB, "sum": PA + PB},
        "y": {"y_A": PA / spec.P_total, "y_B": PB / spec.P_total},
        "residual": (PA + PB) - spec.P_total,
    }