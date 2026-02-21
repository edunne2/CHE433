# bank/distillation_tray_efficiencies_base.py
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


# -----------------------------
# Summary-sheet tray efficiency relations
# -----------------------------
def murphree_vapor_efficiency(y_out: float, y_in: float, y_eq: float) -> float:
    """
    Murphree vapor efficiency (common summary form):
      E_MV = (y_out - y_in) / (y_eq - y_in)
    """
    _vf01_open("y_out", y_out)
    _vf01_open("y_in", y_in)
    _vf01_open("y_eq", y_eq)
    denom = (y_eq - y_in)
    if abs(denom) < 1e-15:
        raise ValueError("Murphree denominator ~ 0 (y_eq == y_in).")
    return (y_out - y_in) / denom


def overall_efficiency_from_murphree(E_M: float, V_over_L: float) -> float:
    """
    Summary sheet relation:
      E_o = ln[1 + E_M*(V/L)] / ln(V/L)
    """
    _vp("E_M", E_M)
    _vp("V_over_L", V_over_L)
    if abs(V_over_L - 1.0) < 1e-14:
        # limit as V/L -> 1
        return min(1.0, max(0.0, E_M))
    return math.log(1.0 + E_M * V_over_L) / math.log(V_over_L)


def actual_stages_from_theoretical(N_theoretical: float, E_overall: float) -> float:
    _vp("N_theoretical", N_theoretical)
    _vp("E_overall", E_overall)
    return N_theoretical / E_overall


@dataclass(frozen=True)
class TrayEfficiencySpec:
    E_M: float
    V_over_L: float
    N_theoretical: float


def tray_efficiency_block(spec: TrayEfficiencySpec) -> Dict[str, Any]:
    Eo = overall_efficiency_from_murphree(spec.E_M, spec.V_over_L)
    N_actual = actual_stages_from_theoretical(spec.N_theoretical, Eo)
    return {
        "inputs": {"E_M": spec.E_M, "V_over_L": spec.V_over_L, "N_theoretical": spec.N_theoretical},
        "outputs": {"E_overall": Eo, "N_actual": N_actual},
    }