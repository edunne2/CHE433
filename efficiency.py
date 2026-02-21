# bank/separations/distillation/efficiency.py
"""Tray efficiency calculations"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

from bank.core.validation import check_positive, check_in_closed_01


def murphree_vapor_efficiency(
    y_out: float,
    y_in: float,
    y_eq: float,
) -> float:
    """
    Murphree vapor efficiency:
    E_MV = (y_out - y_in) / (y_eq - y_in)
    """
    check_in_closed_01("y_out", y_out)
    check_in_closed_01("y_in", y_in)
    check_in_closed_01("y_eq", y_eq)
    
    denom = y_eq - y_in
    if abs(denom) < 1e-15:
        raise ValueError("Murphree denominator near zero (y_eq ≈ y_in)")
    
    return (y_out - y_in) / denom


def murphree_liquid_efficiency(
    x_out: float,
    x_in: float,
    x_eq: float,
) -> float:
    """
    Murphree liquid efficiency:
    E_ML = (x_in - x_out) / (x_in - x_eq)
    """
    check_in_closed_01("x_out", x_out)
    check_in_closed_01("x_in", x_in)
    check_in_closed_01("x_eq", x_eq)
    
    denom = x_in - x_eq
    if abs(denom) < 1e-15:
        raise ValueError("Murphree denominator near zero (x_in ≈ x_eq)")
    
    return (x_in - x_out) / denom


def overall_efficiency_from_murphree(
    E_MV: float,
    V_over_L: float,
    m: Optional[float] = None,
) -> float:
    """
    Overall column efficiency from Murphree vapor efficiency.
    
    For dilute systems or constant m:
    E_o = ln[1 + E_MV*(λ - 1)] / ln(λ)  where λ = mV/L
    
    Args:
        E_MV: Murphree vapor efficiency
        V_over_L: Vapor to liquid ratio
        m: Slope of equilibrium line (if None, assumes m=1)
    
    Returns:
        Overall column efficiency
    """
    check_positive("E_MV", E_MV)
    check_positive("V_over_L", V_over_L)
    
    if m is not None:
        check_positive("m", m)
        lambda_factor = m * V_over_L
    else:
        lambda_factor = V_over_L
    
    if abs(lambda_factor - 1.0) < 1e-14:
        # Limit as λ → 1
        return min(1.0, E_MV)
    
    return math.log(1.0 + E_MV * (lambda_factor - 1.0)) / math.log(lambda_factor)


def overall_efficiency_empirical(
    alpha: float,
    mu: float,
    correlation: str = "o connell",
) -> float:
    """
    Empirical overall efficiency correlations.
    
    Args:
        alpha: Relative volatility
        mu: Viscosity (cP)
        correlation: One of:
            - "o connell": E_o = 0.5 / (αμ)^0.25
            - "drickamer": E_o = 0.17 - 0.616 log10(μ)
            - "chung": E_o = 0.1 + 0.5/(αμ)^0.25
    
    Returns:
        Overall column efficiency
    """
    check_positive("alpha", alpha)
    check_positive("mu", mu)
    
    correlation = correlation.lower()
    
    if correlation == "o connell":
        return 0.5 / (alpha * mu) ** 0.25
    elif correlation == "drickamer":
        return max(0.1, 0.17 - 0.616 * math.log10(mu))
    elif correlation == "chung":
        return 0.1 + 0.5 / (alpha * mu) ** 0.25
    else:
        raise ValueError(f"Unknown correlation: {correlation}")


def actual_stages(
    N_theoretical: float,
    efficiency: float,
    include_reboiler: bool = False,
) -> float:
    """
    Calculate actual number of stages from theoretical stages and efficiency.
    
    Args:
        N_theoretical: Number of theoretical stages
        efficiency: Overall column efficiency (fraction)
        include_reboiler: If True, count reboiler as a stage
    
    Returns:
        Number of actual stages
    """
    check_positive("N_theoretical", N_theoretical)
    check_in_closed_01("efficiency", efficiency)
    
    if efficiency <= 0:
        raise ValueError("Efficiency must be positive")
    
    N_actual = N_theoretical / efficiency
    
    if not include_reboiler:
        # Reboiler counts as one theoretical stage but not an actual tray
        N_actual = N_actual - 1.0
    
    return N_actual


@dataclass
class TrayEfficiencySpec:
    """Specification for tray efficiency calculations"""
    E_MV: float                 # Murphree vapor efficiency
    V_over_L: float             # Vapor to liquid ratio
    N_theoretical: float        # Theoretical stages
    
    # Optional
    m: Optional[float] = None   # Equilibrium line slope


def tray_efficiency_block(spec: TrayEfficiencySpec) -> Dict[str, Any]:
    """
    Complete tray efficiency calculation block.
    
    Returns:
        Dictionary with overall efficiency and actual stages
    """
    E_o = overall_efficiency_from_murphree(
        spec.E_MV,
        spec.V_over_L,
        spec.m
    )
    
    N_actual = actual_stages(spec.N_theoretical, E_o)
    
    return {
        "inputs": {
            "E_MV": spec.E_MV,
            "V_over_L": spec.V_over_L,
            "m": spec.m,
            "N_theoretical": spec.N_theoretical,
        },
        "outputs": {
            "E_overall": E_o,
            "N_actual": N_actual,
            "N_actual_ceiling": math.ceil(N_actual - 1e-12),
        }
    }