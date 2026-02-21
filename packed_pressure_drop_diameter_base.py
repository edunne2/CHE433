# bank/packed_pressure_drop_diameter_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import math


@dataclass(frozen=True)
class PackedTowerHydraulicsSpec:
    """
    Packed-tower hydraulics (Example 22.3-1 style chart workflow).

    Chart quantities:
      Flow parameter (abscissa):
        X = (G_L/G_G) * sqrt(rho_g/rho_l)

      Capacity parameter (ordinate):
        C = v_g * sqrt(rho_g/(rho_l-rho_g)) * (F_p)^0.5 * (mu_l/mu_w)^0.05

    Progressive solving implemented here:
      - Always compute X and ΔP_flood/ft (from F_p correlation)
      - If C_flood is given: compute v_g,flood and G_G,flood, G_L,flood
      - If fraction and mdot_g are also given: compute G_G,design, G_L,design,
        A_cs = (mdot_g in lbm/s) / G_G,design
        D = sqrt(4*A_cs/pi)
        mdot_L = (G_L/G_G) * mdot_g
    """
    GL_over_GG: float
    rho_g: float
    rho_l: float
    mu_l_over_mu_w: float
    F_p: float


def _vp(name: str, v: float) -> None:
    if v <= 0:
        raise ValueError(f"{name} must be > 0.")


def _vratio(name: str, v: float) -> None:
    if v <= 0:
        raise ValueError(f"{name} must be > 0.")


def lbm_per_h_to_lbm_per_s(mdot_lbm_per_h: float) -> float:
    _vp("mdot_lbm_per_h", mdot_lbm_per_h)
    return mdot_lbm_per_h / 3600.0


def flow_parameter_X(GL_over_GG: float, rho_g: float, rho_l: float) -> float:
    _vratio("GL_over_GG", GL_over_GG)
    _vp("rho_g", rho_g)
    _vp("rho_l", rho_l)
    return GL_over_GG * math.sqrt(rho_g / rho_l)


def deltaP_flood_inH2O_per_ft_from_Fp(F_p: float) -> float:
    """
    Example 22.3-1 correlation:
      ΔP_flood/ft = 0.115 (F_p)^0.7   [in H2O/ft]
    """
    _vp("F_p", F_p)
    return 0.115 * (F_p ** 0.7)


def capacity_parameter_C(
    v_g: float,
    rho_g: float,
    rho_l: float,
    F_p: float,
    mu_l_over_mu_w: float,
) -> float:
    _vp("v_g", v_g)
    _vp("rho_g", rho_g)
    _vp("rho_l", rho_l)
    _vp("F_p", F_p)
    _vp("mu_l_over_mu_w", mu_l_over_mu_w)
    if rho_l <= rho_g:
        raise ValueError("Require rho_l > rho_g.")
    return (
        v_g
        * math.sqrt(rho_g / (rho_l - rho_g))
        * (F_p ** 0.5)
        * (mu_l_over_mu_w ** 0.05)
    )


def v_g_from_capacity(
    C: float,
    rho_g: float,
    rho_l: float,
    F_p: float,
    mu_l_over_mu_w: float,
) -> float:
    _vp("C", C)
    _vp("rho_g", rho_g)
    _vp("rho_l", rho_l)
    _vp("F_p", F_p)
    _vp("mu_l_over_mu_w", mu_l_over_mu_w)
    if rho_l <= rho_g:
        raise ValueError("Require rho_l > rho_g.")
    denom = (
        math.sqrt(rho_g / (rho_l - rho_g))
        * (F_p ** 0.5)
        * (mu_l_over_mu_w ** 0.05)
    )
    return C / denom


def gas_mass_velocity_GG(v_g: float, rho_g: float) -> float:
    _vp("v_g", v_g)
    _vp("rho_g", rho_g)
    return v_g * rho_g


def liquid_mass_velocity_GL(GL_over_GG: float, GG: float) -> float:
    _vratio("GL_over_GG", GL_over_GG)
    _vp("GG", GG)
    return GL_over_GG * GG


def GG_at_fraction_of_flooding(GG_flood: float, fraction: float) -> float:
    _vp("GG_flood", GG_flood)
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must satisfy 0 < fraction <= 1.")
    return fraction * GG_flood


def GL_at_fraction_of_flooding(GL_flood: float, fraction: float) -> float:
    _vp("GL_flood", GL_flood)
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must satisfy 0 < fraction <= 1.")
    return fraction * GL_flood


def deltaP_final(C_flood: float, fraction: float, flow_parameter_X: float) -> float:
    _vp("C_flood", C_flood)
    _vp("fraction", fraction)
    _vp("flow_parameter_X", flow_parameter_X)
    return (C_flood * fraction) - flow_parameter_X


def cross_sectional_area_from_mdot_over_GG(mdot_g_lbm_per_h: float, GG_design: float) -> float:
    """
    Your required method:
      A_cs = (mdot_g in lbm/s) / GG_design
    """
    _vp("mdot_g_lbm_per_h", mdot_g_lbm_per_h)
    _vp("GG_design", GG_design)
    mdot_s = lbm_per_h_to_lbm_per_s(mdot_g_lbm_per_h)
    return mdot_s / GG_design


def diameter_from_cross_sectional_area(A_cs: float) -> float:
    _vp("A_cs", A_cs)
    return math.sqrt((4.0 * A_cs) / math.pi)


def liquid_total_mass_flow(mdot_g_lbm_per_h: float, GL_over_GG: float) -> float:
    """
    Your required method:
      mdot_L = (GL/GG) * mdot_g
    """
    _vp("mdot_g_lbm_per_h", mdot_g_lbm_per_h)
    _vratio("GL_over_GG", GL_over_GG)
    return GL_over_GG * mdot_g_lbm_per_h


def packed_tower_hydraulics_progress(
    spec: PackedTowerHydraulicsSpec,
    mdot_g_lbm_per_h: Optional[float] = None,
    fraction_of_flooding: Optional[float] = None,
    C_flood: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Progressive outputs:
      - Always: X and ΔP_flood/ft (from F_p)
      - If C_flood: flooding v_g, GG, GL
      - If mdot_g and fraction_of_flooding also given: GG_design, GL_design, A_cs, D, R, mdot_L
    """
    # FIX: flow_parameter_X takes (GL_over_GG, rho_g, rho_l) only.
    X = flow_parameter_X(spec.GL_over_GG, spec.rho_g, spec.rho_l)
    dP_flood = deltaP_flood_inH2O_per_ft_from_Fp(spec.F_p)

    out: Dict[str, Any] = {
        "flow_parameter_X": X,
        "deltaP_flood_inH2O_per_ft": dP_flood,
        "inputs": {
            "GL_over_GG": spec.GL_over_GG,
            "rho_g": spec.rho_g,
            "rho_l": spec.rho_l,
            "mu_l_over_mu_w": spec.mu_l_over_mu_w,
            "F_p": spec.F_p,
        },
    }

    if C_flood is not None:
        v_flood = v_g_from_capacity(C_flood, spec.rho_g, spec.rho_l, spec.F_p, spec.mu_l_over_mu_w)
        GG_flood = gas_mass_velocity_GG(v_flood, spec.rho_g)
        GL_flood = liquid_mass_velocity_GL(spec.GL_over_GG, GG_flood)

        out["flooding"] = {
            "C_flood": C_flood,
            "v_g_flood": v_flood,
            "GG_flood": GG_flood,
            "GL_flood": GL_flood,
        }

        if (mdot_g_lbm_per_h is not None) and (fraction_of_flooding is not None):
            GG_design = GG_at_fraction_of_flooding(GG_flood, fraction_of_flooding)
            GL_design = liquid_mass_velocity_GL(spec.GL_over_GG, GG_design)

            A_cs = cross_sectional_area_from_mdot_over_GG(mdot_g_lbm_per_h, GG_design)
            D = diameter_from_cross_sectional_area(A_cs)
            R = 0.5 * D

            mdot_L = liquid_total_mass_flow(mdot_g_lbm_per_h, spec.GL_over_GG)

            out["design"] = {
                "fraction_of_flooding": fraction_of_flooding,
                "GG_design": GG_design,
                "GL_design": GL_design,
                "mdot_g_lbm_per_h": mdot_g_lbm_per_h,
                "mdot_g_lbm_per_s": lbm_per_h_to_lbm_per_s(mdot_g_lbm_per_h),
                "A_cs": A_cs,
                "D": D,
                "R": R,
                "mdot_L_lbm_per_h": mdot_L,
            }

    return out