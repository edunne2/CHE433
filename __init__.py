# bank/core/__init__.py
"""Core utilities for all chemical engineering calculations"""

from .validation import (
    check_positive,
    check_non_negative,
    check_in_closed_01,
    check_in_open_01,
    normalize_composition,
    check_composition_match,
    ChemEngError,
    InputError,
    ConvergenceError,
    PhaseError,
)

from .numerical import (
    bisection,
    newton_raphson,
    integrate_trapezoid,
    integrate_adaptive,
    linear_interpolate,
    log_interpolate,
)

from .conversions import (
    # Pressure
    atm_to_kPa, kPa_to_atm, psi_to_kPa, kPa_to_psi, psi_to_atm, atm_to_psi,
    bar_to_kPa, kPa_to_bar, mmHg_to_kPa, kPa_to_mmHg,
    # Temperature
    C_to_K, K_to_C, F_to_C, C_to_F, F_to_K, K_to_F, R_to_K, K_to_R,
    # Mass and Length
    lb_to_kg, kg_to_lb, g_to_kg, kg_to_g, ft_to_m, m_to_ft, in_to_m, m_to_in, cm_to_m, m_to_cm,
    # Volume and Flow
    L_to_m3, m3_to_L, gal_to_m3, m3_to_gal, ft3_to_m3, m3_to_ft3, gpm_to_m3s, m3s_to_gpm,
    # Viscosity
    cP_to_Pa_s, Pa_s_to_cP, cP_to_lb_ft_hr, lb_ft_hr_to_cP,
    # Energy and Power
    cal_to_J, J_to_cal, BTU_to_J, J_to_BTU, kW_to_HP, HP_to_kW,
)

from .properties import (
    # Water
    water_density, water_viscosity, water_heat_capacity, water_vapor_pressure,
    # Air
    air_density, air_viscosity, air_thermal_conductivity,
    # Ideal Gas
    ideal_gas_density, ideal_gas_viscosity,
    # Mixtures
    mixture_density, mixture_viscosity, surface_tension_water,
)

from .base import (
    SolverBase,
    SpecificationBase,
    ChemEngError as BaseChemEngError,
)

__all__ = [
    # Validation
    'check_positive', 'check_non_negative', 'check_in_closed_01', 'check_in_open_01',
    'normalize_composition', 'check_composition_match',
    'ChemEngError', 'InputError', 'ConvergenceError', 'PhaseError',
    
    # Numerical
    'bisection', 'newton_raphson', 'integrate_trapezoid', 'integrate_adaptive',
    'linear_interpolate', 'log_interpolate',
    
    # Conversions - Pressure
    'atm_to_kPa', 'kPa_to_atm', 'psi_to_kPa', 'kPa_to_psi', 'psi_to_atm', 'atm_to_psi',
    'bar_to_kPa', 'kPa_to_bar', 'mmHg_to_kPa', 'kPa_to_mmHg',
    
    # Conversions - Temperature
    'C_to_K', 'K_to_C', 'F_to_C', 'C_to_F', 'F_to_K', 'K_to_F', 'R_to_K', 'K_to_R',
    
    # Conversions - Mass and Length
    'lb_to_kg', 'kg_to_lb', 'g_to_kg', 'kg_to_g', 'ft_to_m', 'm_to_ft', 
    'in_to_m', 'm_to_in', 'cm_to_m', 'm_to_cm',
    
    # Conversions - Volume and Flow
    'L_to_m3', 'm3_to_L', 'gal_to_m3', 'm3_to_gal', 'ft3_to_m3', 'm3_to_ft3',
    'gpm_to_m3s', 'm3s_to_gpm',
    
    # Conversions - Viscosity
    'cP_to_Pa_s', 'Pa_s_to_cP', 'cP_to_lb_ft_hr', 'lb_ft_hr_to_cP',
    
    # Conversions - Energy and Power
    'cal_to_J', 'J_to_cal', 'BTU_to_J', 'J_to_BTU', 'kW_to_HP', 'HP_to_kW',
    
    # Properties
    'water_density', 'water_viscosity', 'water_heat_capacity', 'water_vapor_pressure',
    'air_density', 'air_viscosity', 'air_thermal_conductivity',
    'ideal_gas_density', 'ideal_gas_viscosity',
    'mixture_density', 'mixture_viscosity', 'surface_tension_water',
    
    # Base Classes
    'SolverBase', 'SpecificationBase',
]