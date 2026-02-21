# bank/core/conversions.py
"""Unit conversion utilities for all chemical engineering calculations"""

from typing import Union


# ============================================================================
# Pressure Conversions
# ============================================================================

def atm_to_kPa(P_atm: Union[float, int]) -> float:
    """Convert atm to kPa"""
    return float(P_atm) * 101.325

def kPa_to_atm(P_kPa: Union[float, int]) -> float:
    """Convert kPa to atm"""
    return float(P_kPa) / 101.325

def psi_to_kPa(P_psi: Union[float, int]) -> float:
    """Convert psi to kPa"""
    return float(P_psi) * 6.89476

def kPa_to_psi(P_kPa: Union[float, int]) -> float:
    """Convert kPa to psi"""
    return float(P_kPa) / 6.89476

def psi_to_atm(P_psi: Union[float, int]) -> float:
    """Convert psi to atm"""
    return float(P_psi) / 14.6959

def atm_to_psi(P_atm: Union[float, int]) -> float:
    """Convert atm to psi"""
    return float(P_atm) * 14.6959

def bar_to_kPa(P_bar: Union[float, int]) -> float:
    """Convert bar to kPa"""
    return float(P_bar) * 100.0

def kPa_to_bar(P_kPa: Union[float, int]) -> float:
    """Convert kPa to bar"""
    return float(P_kPa) / 100.0

def mmHg_to_kPa(P_mmHg: Union[float, int]) -> float:
    """Convert mmHg to kPa"""
    return float(P_mmHg) * 0.133322

def kPa_to_mmHg(P_kPa: Union[float, int]) -> float:
    """Convert kPa to mmHg"""
    return float(P_kPa) / 0.133322


# ============================================================================
# Temperature Conversions
# ============================================================================

def C_to_K(T_C: Union[float, int]) -> float:
    """Convert Celsius to Kelvin"""
    return float(T_C) + 273.15

def K_to_C(T_K: Union[float, int]) -> float:
    """Convert Kelvin to Celsius"""
    return float(T_K) - 273.15

def F_to_C(T_F: Union[float, int]) -> float:
    """Convert Fahrenheit to Celsius"""
    return (float(T_F) - 32.0) * 5.0 / 9.0

def C_to_F(T_C: Union[float, int]) -> float:
    """Convert Celsius to Fahrenheit"""
    return float(T_C) * 9.0 / 5.0 + 32.0

def F_to_K(T_F: Union[float, int]) -> float:
    """Convert Fahrenheit to Kelvin"""
    return C_to_K(F_to_C(T_F))

def K_to_F(T_K: Union[float, int]) -> float:
    """Convert Kelvin to Fahrenheit"""
    return C_to_F(K_to_C(T_K))

def R_to_K(T_R: Union[float, int]) -> float:
    """Convert Rankine to Kelvin"""
    return float(T_R) / 1.8

def K_to_R(T_K: Union[float, int]) -> float:
    """Convert Kelvin to Rankine"""
    return float(T_K) * 1.8


# ============================================================================
# Mass and Length Conversions
# ============================================================================

def lb_to_kg(m_lb: Union[float, int]) -> float:
    """Convert pounds to kilograms"""
    return float(m_lb) * 0.453592

def kg_to_lb(m_kg: Union[float, int]) -> float:
    """Convert kilograms to pounds"""
    return float(m_kg) / 0.453592

def g_to_kg(m_g: Union[float, int]) -> float:
    """Convert grams to kilograms"""
    return float(m_g) / 1000.0

def kg_to_g(m_kg: Union[float, int]) -> float:
    """Convert kilograms to grams"""
    return float(m_kg) * 1000.0

def ft_to_m(L_ft: Union[float, int]) -> float:
    """Convert feet to meters"""
    return float(L_ft) * 0.3048

def m_to_ft(L_m: Union[float, int]) -> float:
    """Convert meters to feet"""
    return float(L_m) / 0.3048

def in_to_m(L_in: Union[float, int]) -> float:
    """Convert inches to meters"""
    return float(L_in) * 0.0254

def m_to_in(L_m: Union[float, int]) -> float:
    """Convert meters to inches"""
    return float(L_m) / 0.0254

def cm_to_m(L_cm: Union[float, int]) -> float:
    """Convert centimeters to meters"""
    return float(L_cm) / 100.0

def m_to_cm(L_m: Union[float, int]) -> float:
    """Convert meters to centimeters"""
    return float(L_m) * 100.0


# ============================================================================
# Volume and Flow Rate Conversions
# ============================================================================

def L_to_m3(V_L: Union[float, int]) -> float:
    """Convert liters to cubic meters"""
    return float(V_L) / 1000.0

def m3_to_L(V_m3: Union[float, int]) -> float:
    """Convert cubic meters to liters"""
    return float(V_m3) * 1000.0

def gal_to_m3(V_gal: Union[float, int]) -> float:
    """Convert US gallons to cubic meters"""
    return float(V_gal) * 0.00378541

def m3_to_gal(V_m3: Union[float, int]) -> float:
    """Convert cubic meters to US gallons"""
    return float(V_m3) / 0.00378541

def ft3_to_m3(V_ft3: Union[float, int]) -> float:
    """Convert cubic feet to cubic meters"""
    return float(V_ft3) * 0.0283168

def m3_to_ft3(V_m3: Union[float, int]) -> float:
    """Convert cubic meters to cubic feet"""
    return float(V_m3) / 0.0283168

def gpm_to_m3s(Q_gpm: Union[float, int]) -> float:
    """Convert US gallons per minute to cubic meters per second"""
    return float(Q_gpm) * 0.0000630902

def m3s_to_gpm(Q_m3s: Union[float, int]) -> float:
    """Convert cubic meters per second to US gallons per minute"""
    return float(Q_m3s) / 0.0000630902


# ============================================================================
# Viscosity Conversions
# ============================================================================

def cP_to_Pa_s(visc_cP: Union[float, int]) -> float:
    """Convert centipoise to Pa·s"""
    return float(visc_cP) / 1000.0

def Pa_s_to_cP(visc_Pa_s: Union[float, int]) -> float:
    """Convert Pa·s to centipoise"""
    return float(visc_Pa_s) * 1000.0

def cP_to_lb_ft_hr(visc_cP: Union[float, int]) -> float:
    """Convert centipoise to lb/(ft·hr)"""
    return float(visc_cP) * 2.4191

def lb_ft_hr_to_cP(visc_lb_ft_hr: Union[float, int]) -> float:
    """Convert lb/(ft·hr) to centipoise"""
    return float(visc_lb_ft_hr) / 2.4191


# ============================================================================
# Energy and Power Conversions
# ============================================================================

def cal_to_J(E_cal: Union[float, int]) -> float:
    """Convert calories to Joules"""
    return float(E_cal) * 4.184

def J_to_cal(E_J: Union[float, int]) -> float:
    """Convert Joules to calories"""
    return float(E_J) / 4.184

def BTU_to_J(E_BTU: Union[float, int]) -> float:
    """Convert BTU to Joules"""
    return float(E_BTU) * 1055.06

def J_to_BTU(E_J: Union[float, int]) -> float:
    """Convert Joules to BTU"""
    return float(E_J) / 1055.06

def kW_to_HP(P_kW: Union[float, int]) -> float:
    """Convert kilowatts to horsepower"""
    return float(P_kW) * 1.34102

def HP_to_kW(P_HP: Union[float, int]) -> float:
    """Convert horsepower to kilowatts"""
    return float(P_HP) / 1.34102