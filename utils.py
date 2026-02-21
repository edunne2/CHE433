"""Unit conversions for liquid-liquid extraction - Eqs. 27.3-7, 27.3-8"""


def dyn_per_cm_to_lbm_per_h2(sigma_dyn_per_cm: float) -> float:
    """
    Convert interfacial tension from dyn/cm to lb_m/h² - Eq. 27.3-7
    
    1.0 dyn/cm = 28,572 lb_m/h²
    """
    return sigma_dyn_per_cm * 28572.0


def lbm_per_h2_to_dyn_per_cm(sigma_lbm_per_h2: float) -> float:
    """Convert interfacial tension from lb_m/h² to dyn/cm"""
    return sigma_lbm_per_h2 / 28572.0


def cp_to_lbm_per_ft_hr(viscosity_cp: float) -> float:
    """
    Convert viscosity from centipoise to lb_m/ft·h - Eq. 27.3-8
    
    1 cp = 2.4191 lb_m/ft·h
    """
    return viscosity_cp * 2.4191


def lbm_per_ft_hr_to_cp(viscosity_lbm_per_ft_hr: float) -> float:
    """Convert viscosity from lb_m/ft·h to centipoise"""
    return viscosity_lbm_per_ft_hr / 2.4191


def ft3_per_h_to_m3_per_h(flow_ft3_per_h: float) -> float:
    """Convert volumetric flow from ft³/h to m³/h"""
    return flow_ft3_per_h * 0.0283168


def m3_per_h_to_ft3_per_h(flow_m3_per_h: float) -> float:
    """Convert volumetric flow from m³/h to ft³/h"""
    return flow_m3_per_h / 0.0283168


def ft_to_m(length_ft: float) -> float:
    """Convert length from feet to meters"""
    return length_ft * 0.3048


def m_to_ft(length_m: float) -> float:
    """Convert length from meters to feet"""
    return length_m / 0.3048