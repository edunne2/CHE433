# bank/separations/absorption/pressure_drop.py
"""Pressure drop and flooding calculations for packed towers"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

from bank.core.validation import check_positive, ChemEngError


@dataclass
class PressureDropSpec:
    """
    Specification for packed column pressure drop.
    
    Based on generalized pressure drop correlation (GPDC).
    """
    # Packing characteristics
    packing_factor: float      # F_p (ft^-1)
    void_fraction: float       # ε
    
    # Flow rates
    G: float                   # Gas mass flux (lb/hr·ft²)
    L: float                   # Liquid mass flux (lb/hr·ft²)
    
    # Physical properties
    rho_G: float               # Gas density (lb/ft³)
    rho_L: float               # Liquid density (lb/ft³)
    mu_L: float                # Liquid viscosity (cP)
    
    # Tower dimensions
    tower_diameter: Optional[float] = None  # ft
    packed_height: Optional[float] = None   # ft
    
    # Optional: flooding conditions
    flooding_fraction: float = 0.7


def flooding_velocity_correlation(
    packing_factor: float,
    L: float,
    G: float,
    rho_G: float,
    rho_L: float,
    mu_L: float,
) -> float:
    """
    Calculate flooding velocity using generalized correlation.
    
    Returns:
        Gas superficial velocity at flooding (ft/s)
    """
    # Calculate flow parameter
    flow_param = (L / G) * math.sqrt(rho_G / rho_L)
    
    # Capacity parameter at flooding (from GPDC chart correlation)
    if flow_param < 0.01:
        cap_param = 0.22
    elif flow_param < 0.1:
        cap_param = 0.22 - 0.15 * math.log10(flow_param / 0.01)
    else:
        cap_param = 0.12 - 0.08 * math.log10(flow_param / 0.1)
    
    # Correct for liquid viscosity
    cap_param *= (mu_L / 1.0) ** 0.1
    
    # Calculate flooding velocity
    v_flood = math.sqrt(
        cap_param / (packing_factor * math.sqrt(rho_G / (rho_L - rho_G)))
    )
    
    return v_flood


def pressure_drop_empirical(
    G: float,
    rho_G: float,
    packing_factor: float,
    liquid_correction: float = 1.0,
) -> float:
    """
    Estimate pressure drop using empirical correlation.
    
    Returns:
        Pressure drop (in H₂O/ft packing)
    """
    # Base gas phase pressure drop
    v_G = G / (3600 * rho_G)  # Convert to ft/s
    delta_P_dry = 0.1 * packing_factor * rho_G * v_G**2
    
    # Correct for liquid
    delta_P = delta_P_dry * liquid_correction
    
    return delta_P / 12  # Convert to in H₂O/ft


class PressureDropSolver:
    """Solver for packed column pressure drop"""
    
    def __init__(self, spec: PressureDropSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        """Validate inputs"""
        check_positive("packing_factor", self.spec.packing_factor)
        check_positive("void_fraction", self.spec.void_fraction)
        check_positive("G", self.spec.G)
        check_positive("L", self.spec.L)
        check_positive("rho_G", self.spec.rho_G)
        check_positive("rho_L", self.spec.rho_L)
        check_positive("mu_L", self.spec.mu_L)
        
        if self.spec.void_fraction >= 1.0:
            raise ChemEngError(f"Void fraction must be < 1, got {self.spec.void_fraction}")
    
    def calculate_flooding(self) -> Dict[str, float]:
        """Calculate flooding conditions"""
        v_flood = flooding_velocity_correlation(
            self.spec.packing_factor,
            self.spec.L,
            self.spec.G,
            self.spec.rho_G,
            self.spec.rho_L,
            self.spec.mu_L
        )
        
        # Operating velocity
        if self.spec.tower_diameter:
            area = math.pi * (self.spec.tower_diameter / 2) ** 2
            v_op = self.spec.G / (3600 * self.spec.rho_G * area)
            percent_flood = v_op / v_flood
        else:
            v_op = None
            percent_flood = None
        
        return {
            "flooding_velocity_ft_s": v_flood,
            "operating_velocity_ft_s": v_op,
            "percent_flood": percent_flood,
            "recommended_max_velocity": v_flood * self.spec.flooding_fraction,
        }
    
    def calculate_pressure_drop(self) -> float:
        """Calculate column pressure drop"""
        # Liquid correction factor (simplified)
        liquid_correction = math.exp(0.5 * self.spec.L / 1000)
        
        delta_P_per_ft = pressure_drop_empirical(
            self.spec.G,
            self.spec.rho_G,
            self.spec.packing_factor,
            liquid_correction
        )
        
        if self.spec.packed_height:
            return delta_P_per_ft * self.spec.packed_height
        else:
            return delta_P_per_ft
    
    def solve(self) -> Dict[str, Any]:
        """Main solving method"""
        flooding = self.calculate_flooding()
        delta_P = self.calculate_pressure_drop()
        
        # Recommended diameter for given flooding
        if self.spec.tower_diameter is None:
            v_rec = flooding["recommended_max_velocity"]
            area_rec = self.spec.G / (3600 * self.spec.rho_G * v_rec)
            diameter_rec = 2 * math.sqrt(area_rec / math.pi)
        else:
            diameter_rec = None
        
        return {
            "flooding_analysis": flooding,
            "pressure_drop": {
                "delta_P_inH2O_per_ft": delta_P if not self.spec.packed_height else None,
                "delta_P_total_inH2O": delta_P if self.spec.packed_height else None,
            },
            "design_recommendations": {
                "recommended_diameter_ft": diameter_rec,
                "recommended_flooding_fraction": self.spec.flooding_fraction,
            }
        }