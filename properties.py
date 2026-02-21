# bank/core/properties.py
"""Physical property calculations for common substances"""

from typing import Optional, Dict, Union
import math
from .validation import check_positive, check_in_closed_01
from .conversions import C_to_K


# ============================================================================
# Water Properties
# ============================================================================

def water_density(T_C: Union[float, int]) -> float:
    """
    Density of water (kg/m³) as function of temperature (°C).
    Correlation valid 0-100°C.
    """
    T = float(T_C)
    # Polynomial fit to NIST data
    return 1000.0 * (1.0 - (T + 288.9414) * (T - 3.9863)**2 / (508929.2 * (T + 68.12963)))


def water_viscosity(T_C: Union[float, int]) -> float:
    """
    Dynamic viscosity of water (Pa·s) as function of temperature (°C).
    Correlation valid 0-100°C.
    """
    T = float(T_C)
    T_K = C_to_K(T)
    A = 2.414e-5
    B = 247.8
    C = 140.0
    return A * 10.0 ** (B / (T_K - C))


def water_heat_capacity(T_C: Union[float, int]) -> float:
    """
    Specific heat capacity of water (J/kg·K) as function of temperature (°C).
    """
    T = float(T_C)
    # Approximately constant in liquid range
    return 4184.0


def water_vapor_pressure(T_C: Union[float, int]) -> float:
    """
    Vapor pressure of water (kPa) as function of temperature (°C).
    Using Antoine equation.
    """
    T = float(T_C)
    # Antoine constants for water (log10(P) = A - B/(T + C), P in kPa, T in °C)
    A = 7.19621
    B = 1730.63
    C = 233.426
    return 10.0 ** (A - B / (T + C))


# ============================================================================
# Air Properties
# ============================================================================

def air_density(T_C: Union[float, int], P_kPa: Union[float, int] = 101.325) -> float:
    """
    Density of air (kg/m³) as function of temperature (°C) and pressure (kPa).
    Assumes ideal gas, molecular weight = 28.97 g/mol.
    """
    T = float(T_C)
    P = float(P_kPa)
    T_K = C_to_K(T)
    R = 8.314  # J/mol·K
    MW = 0.02897  # kg/mol
    
    return (P * 1000.0 * MW) / (R * T_K)


def air_viscosity(T_C: Union[float, int]) -> float:
    """
    Dynamic viscosity of air (Pa·s) as function of temperature (°C).
    Sutherland's formula.
    """
    T = float(T_C)
    T_K = C_to_K(T)
    mu_0 = 1.716e-5  # Pa·s at 273 K
    T_0 = 273.0  # K
    S = 111.0  # Sutherland's constant for air
    
    return mu_0 * (T_K / T_0)**1.5 * (T_0 + S) / (T_K + S)


def air_thermal_conductivity(T_C: Union[float, int]) -> float:
    """
    Thermal conductivity of air (W/m·K) as function of temperature (°C).
    """
    T = float(T_C)
    T_K = C_to_K(T)
    # Correlation from NIST
    return 2.334e-3 * T_K**0.78


# ============================================================================
# Ideal Gas Properties
# ============================================================================

def ideal_gas_density(MW: float, T_C: float, P_kPa: float = 101.325) -> float:
    """
    Density of ideal gas (kg/m³).
    
    Args:
        MW: Molecular weight (kg/mol)
        T_C: Temperature (°C)
        P_kPa: Pressure (kPa)
    """
    check_positive("MW", MW)
    T = float(T_C)
    P = float(P_kPa)
    T_K = C_to_K(T)
    R = 8.314  # J/mol·K
    
    return (P * 1000.0 * MW) / (R * T_K)


def ideal_gas_viscosity(MW: float, T_C: float, sigma: float = None) -> float:
    """
    Estimate gas viscosity using Chapman-Enskog theory.
    
    Args:
        MW: Molecular weight (kg/mol)
        T_C: Temperature (°C)
        sigma: Collision diameter (Å). If None, estimated from critical properties.
    """
    MW_g = MW * 1000.0  # Convert to g/mol
    T_K = C_to_K(T_C)
    
    if sigma is None:
        # Estimate sigma from MW (rough approximation)
        sigma = 2.44 * (MW_g / 100.0)**(1/3)
    
    # Chapman-Enskog formula (simplified)
    omega_v = 1.16145 * T_K**(-0.14874) + 0.52487 * math.exp(-0.77320 * T_K) + 2.16178 * math.exp(-2.43787 * T_K)
    
    mu = 2.6693e-6 * math.sqrt(MW_g * T_K) / (sigma**2 * omega_v)
    return mu / 1000.0  # Convert to Pa·s


# ============================================================================
# Mixture Properties
# ============================================================================

def mixture_density(compositions: Dict[str, float], densities: Dict[str, float]) -> float:
    """
    Calculate density of a mixture (mass/volume basis).
    
    Args:
        compositions: Mass fractions {component: fraction}
        densities: Pure component densities {component: density}
    
    Returns:
        Mixture density (same units as input densities)
    """
    # Sum of (mass fraction / density) = 1 / mixture density
    sum_vol_frac = 0.0
    for comp, w in compositions.items():
        if comp not in densities:
            raise KeyError(f"Density not provided for component: {comp}")
        sum_vol_frac += w / densities[comp]
    
    return 1.0 / sum_vol_frac


def mixture_viscosity(compositions: Dict[str, float], viscosities: Dict[str, float]) -> float:
    """
    Estimate liquid mixture viscosity using Grunberg-Nissan equation.
    
    Args:
        compositions: Mole fractions {component: fraction}
        viscosities: Pure component viscosities {component: viscosity}
    
    Returns:
        Mixture viscosity (same units as input viscosities)
    """
    # Grunberg-Nissan: ln(μ_mix) = Σ x_i ln(μ_i) + ΣΣ x_i x_j G_ij
    # Simplified version ignoring interaction parameters
    log_mu = 0.0
    for comp, x in compositions.items():
        if comp not in viscosities:
            raise KeyError(f"Viscosity not provided for component: {comp}")
        log_mu += x * math.log(viscosities[comp])
    
    return math.exp(log_mu)


def surface_tension_water(T_C: float) -> float:
    """
    Surface tension of water (N/m) as function of temperature (°C).
    """
    T = float(T_C)
    T_K = C_to_K(T)
    T_critical = 647.15  # Critical temperature of water (K)
    
    # Correlation from IAPWS
    sigma_0 = 0.2358  # N/m at reference
    return sigma_0 * (1.0 - T_K / T_critical)**1.256  # FIXED: T_critical not T_c