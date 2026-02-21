"""Equipment design equations - Eqs. 27.3-1, 27.3-2"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

from bank.core.validation import check_positive, check_in_closed_01


def tray_efficiency_perforated(
    T: float,
    VD_over_VC: float,
    sigma: float,
    d_o: float,
) -> float:
    """
    Tray efficiency for perforated-plate (sieve-tray) tower - Eq. 27.3-2
    
    E_o = 0.352 T^0.5 (V_D/V_C)^0.42 / (σ d_o^0.35)
    
    Args:
        T: Tray spacing (ft)
        VD_over_VC: Ratio of dispersed to continuous phase superficial velocities
        sigma: Interfacial tension (dyn/cm)
        d_o: Hole diameter (ft)
    
    Returns:
        Fractional tray efficiency E_o
    """
    check_positive("T", T)
    check_positive("VD_over_VC", VD_over_VC)
    check_positive("sigma", sigma)
    check_positive("d_o", d_o)
    
    numerator = 0.352 * math.sqrt(T) * (VD_over_VC) ** 0.42
    denominator = sigma * (d_o ** 0.35)
    
    return numerator / denominator


def interfacial_tension_estimation(
    x_AB: float,
    x_BA: float,
    x_CA: float,
    x_CB: float,
) -> float:
    """
    Estimate interfacial tension - Eq. 27.3-1
    
    σ = -7.34 ln[x_AB + x_BA + (x_CA + x_CB)/2] - 4.90
    
    Valid for [x_AB + x_BA + (x_CA + x_CB)/2] between 0.0004 and 0.30
    
    Args:
        x_AB: Mole fraction of solvent A in saturated solvent-rich B layer
        x_BA: Mole fraction of solvent B in saturated solvent-rich A layer
        x_CA: Mole fraction of solute C in A layer
        x_CB: Mole fraction of solute C in B layer
    
    Returns:
        Interfacial tension σ (dyn/cm)
    """
    check_positive("x_AB", x_AB)
    check_positive("x_BA", x_BA)
    check_positive("x_CA", x_CA)
    check_positive("x_CB", x_CB)
    
    arg = x_AB + x_BA + (x_CA + x_CB) / 2.0
    
    if arg < 0.0004 or arg > 0.30:
        # Warning but still calculate
        pass
    
    return -7.34 * math.log(arg) - 4.90


class MixerSettler:
    """Mixer-settler extraction unit"""
    
    def __init__(self, volume: float, efficiency: float = 0.85):
        self.volume = volume
        self.efficiency = check_in_closed_01("efficiency", efficiency)
    
    def stages_required(self, theoretical_stages: float) -> float:
        """Calculate actual stages needed given efficiency"""
        return theoretical_stages / self.efficiency


class SprayTower:
    """Spray extraction tower"""
    
    def __init__(self, height: float, diameter: float):
        self.height = height
        self.diameter = diameter
    
    @staticmethod
    def typical_hets() -> float:
        """Typical HETS for spray towers (m) from Table 27.3-1"""
        return 3.0  # average of 3-4 m


class PackedTower:
    """Packed extraction tower"""
    
    def __init__(self, height: float, diameter: float, packing_factor: float):
        self.height = height
        self.diameter = diameter
        self.packing_factor = packing_factor
    
    @staticmethod
    def typical_hets() -> float:
        """Typical HETS for packed towers (m) from Table 27.3-1"""
        return 0.95  # average of 0.4-1.5 m
    
    def height_from_stages(self, stages: float) -> float:
        """Calculate tower height from number of stages"""
        return stages * self.typical_hets()


class SieveTrayTower:
    """Sieve-tray (perforated-plate) extraction tower"""
    
    def __init__(
        self,
        tray_spacing: float,
        hole_diameter: float,
        open_area_fraction: float = 0.2,
    ):
        self.tray_spacing = tray_spacing
        self.hole_diameter = hole_diameter
        self.open_area_fraction = check_in_closed_01("open_area_fraction", open_area_fraction)
    
    def tray_efficiency(
        self,
        VD: float,
        VC: float,
        sigma: float,
    ) -> float:
        """Calculate tray efficiency using Eq. 27.3-2"""
        return tray_efficiency_perforated(
            T=self.tray_spacing,
            VD_over_VC=VD / VC,
            sigma=sigma,
            d_o=self.hole_diameter,
        )


class PulsedTower:
    """Pulsed packed or sieve-tray tower"""
    
    def __init__(self, amplitude: float, frequency: float):
        self.amplitude = amplitude
        self.frequency = frequency
    
    @staticmethod
    def typical_hets() -> float:
        """Typical HETS for pulsed towers (m) from Table 27.3-1"""
        return 0.23  # average of 0.15-0.3 m


class ScheibelTower:
    """Scheibel rotating-agitator tower"""
    
    def __init__(self, diameter: float):
        self.diameter = diameter
    
    @staticmethod
    def typical_hets() -> float:
        """Typical HETS for Scheibel towers (m) from Table 27.3-1"""
        return 0.2  # average of 0.1-0.3 m


class KarrTower:
    """Karr reciprocating-plate tower"""
    
    def __init__(self, diameter: float, strokes_per_min: float = 120):
        self.diameter = diameter
        self.strokes_per_min = strokes_per_min
    
    @staticmethod
    def typical_hets() -> float:
        """Typical HETS for Karr towers (m) from Table 27.3-1"""
        return 0.3  # average of 0.2-0.4 m