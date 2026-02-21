# bank/distillation_ponchon_savarit_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import math


def _vp(name: str, v: float) -> None:
    if v <= 0:
        raise ValueError(f"{name} must be > 0.")


def _vf01_open(name: str, v: float) -> None:
    if not (0.0 <= v < 1.0):
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1.")


@dataclass(frozen=True)
class SaturatedLiquidEnthalpySpec:
    # h = x CpA_L (T-Tref) + (1-x) CpB_L (T-Tref) + ΔH_soln
    xA: float
    T: float
    T_ref: float
    CpA_L: float
    CpB_L: float
    dH_soln: float = 0.0


def h_sat_liquid(spec: SaturatedLiquidEnthalpySpec) -> float:
    _vf01_open("xA", spec.xA)
    return spec.xA * spec.CpA_L * (spec.T - spec.T_ref) + (1.0 - spec.xA) * spec.CpB_L * (spec.T - spec.T_ref) + spec.dH_soln


@dataclass(frozen=True)
class SaturatedVaporEnthalpySpec:
    # H = y[λA + CpA_V(T-TbA)] + (1-y)[λB + CpB_V(T-TbB)]
    yA: float
    T: float
    TbA: float
    TbB: float
    CpA_V: float
    CpB_V: float
    lamA: float
    lamB: float


def H_sat_vapor(spec: SaturatedVaporEnthalpySpec) -> float:
    _vf01_open("yA", spec.yA)
    return spec.yA * (spec.lamA + spec.CpA_V * (spec.T - spec.TbA)) + (1.0 - spec.yA) * (spec.lamB + spec.CpB_V * (spec.T - spec.TbB))


@dataclass(frozen=True)
class PonchonOverallSpec:
    """
    Overall mass/energy balance blocks used in Ponchon–Savarit workflows.
    Convention: heat added positive. Condenser duty usually negative if removing heat.

      Total:      F = D + W
      Component:  F zF = D xD + W xW
      Energy:     F hF + qR + qC = D hD + W hW
    """
    F: float
    zF: float

    D: float
    xD: float

    W: float
    xW: float

    hF: float
    hD: float
    hW: float

    qC: Optional[float] = None
    qR: Optional[float] = None


def overall_mass_balances(spec: PonchonOverallSpec) -> Dict[str, float]:
    _vp("F", spec.F); _vp("D", spec.D); _vp("W", spec.W)
    _vf01_open("zF", spec.zF); _vf01_open("xD", spec.xD); _vf01_open("xW", spec.xW)
    return {
        "res_total": spec.F - spec.D - spec.W,
        "res_component": spec.F * spec.zF - spec.D * spec.xD - spec.W * spec.xW,
    }


def overall_energy_balance(spec: PonchonOverallSpec, qC: float, qR: float) -> float:
    return (spec.F * spec.hF + qR + qC) - (spec.D * spec.hD + spec.W * spec.hW)


def compute_missing_duty(spec: PonchonOverallSpec) -> Dict[str, Any]:
    mb = overall_mass_balances(spec)
    qC = spec.qC
    qR = spec.qR
    if qC is None and qR is None:
        return {
            "mass_balance": mb,
            "energy": {"qC": None, "qR": None, "residual_form": "F*hF + qR + qC = D*hD + W*hW"},
        }
    if qC is None and qR is not None:
        qC = (spec.D * spec.hD + spec.W * spec.hW) - (spec.F * spec.hF + qR)
    if qR is None and qC is not None:
        qR = (spec.D * spec.hD + spec.W * spec.hW) - (spec.F * spec.hF + qC)

    res = overall_energy_balance(spec, qC, qR)
    return {"mass_balance": mb, "energy": {"qC": qC, "qR": qR, "residual": res}}