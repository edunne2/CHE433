"""Tower hydraulics and diameter calculations - Eqs. 26.6-1 to 26.6-4"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

from bank.core.validation import check_positive


def flooding_velocity(
    Kv: float,
    sigma: float,
    rho_L: float,
    rho_V: float
) -> float:
    """
    Maximum allowable vapor velocity - Eq. 26.6-1
    
    vmax = Kv (σ/20)^0.2 √[(ρL - ρV)/ρV]
    
    Args:
        Kv: Constant from Fig. 26.6-1 (ft/s)
        sigma: Surface tension (dyn/cm)
        rho_L: Liquid density (kg/m³ or lb/ft³)
        rho_V: Vapor density (kg/m³ or lb/ft³)
    
    Returns:
        Maximum vapor velocity (same units as Kv input)
    """
    check_positive("Kv", Kv)
    check_positive("sigma", sigma)
    check_positive("rho_L", rho_L)
    check_positive("rho_V", rho_V)
    
    if rho_V >= rho_L:
        raise ValueError("Vapor density must be less than liquid density")
    
    sigma_factor = (sigma / 20.0) ** 0.2
    density_factor = math.sqrt((rho_L - rho_V) / rho_V)
    
    return Kv * sigma_factor * density_factor


def tower_diameter(
    V_flow: float,
    rho_V: float,
    v_max: float,
    downcomer_factor: float = 0.91,
    safety_factor: float = 0.80
) -> float:
    """
    Calculate tower diameter from vapor flow and velocity.
    
    Args:
        V_flow: Vapor mass flow rate (kg/h or lb/h)
        rho_V: Vapor density (kg/m³ or lb/ft³)
        v_max: Maximum vapor velocity from Eq. 26.6-1 (ft/s)
        downcomer_factor: Factor for downspout area (typically 0.91)
        safety_factor: Factor for operating below flooding (typically 0.80)
    
    Returns:
        Tower diameter (ft)
    """
    check_positive("V_flow", V_flow)
    check_positive("rho_V", rho_V)
    check_positive("v_max", v_max)
    
    # Convert velocity to ft/s if needed - assume input in ft/s
    v_design = v_max * downcomer_factor * safety_factor
    
    # Volumetric flow rate (ft³/s)
    # Convert V_flow from lb/h to lb/s, then to ft³/s
    V_vol = V_flow / 3600.0 / rho_V
    
    # Area = volumetric flow / velocity
    area = V_vol / v_design
    
    # Diameter = sqrt(4A/π)
    diameter = math.sqrt(4.0 * area / math.pi)
    
    return diameter


def condenser_duty(
    D: float,
    R: float,
    delta_H_vap: float,
    condenser_type: str = "total"
) -> float:
    """
    Condenser duty calculation - Eqs. 26.6-2 and 26.6-3
    
    Total condenser: qc = D (R + 1) ΔHvap           (26.6-2)
    Partial condenser: qc = D (R) ΔHvap             (26.6-3)
    
    Args:
        D: Distillate flow rate (mol/h)
        R: Reflux ratio
        delta_H_vap: Heat of vaporization (kJ/mol)
        condenser_type: "total" or "partial"
    
    Returns:
        Condenser duty (kJ/h)
    """
    check_positive("D", D)
    check_positive("R", R)
    check_positive("delta_H_vap", delta_H_vap)
    
    if condenser_type.lower() == "total":
        return D * (R + 1.0) * delta_H_vap
    elif condenser_type.lower() == "partial":
        return D * R * delta_H_vap
    else:
        raise ValueError("condenser_type must be 'total' or 'partial'")


def reboiler_duty(
    B: float,
    Vm: float,
    delta_H_vap: float
) -> float:
    """
    Reboiler duty calculation - Eq. 26.6-4
    
    qR = B Vm ΔH'vap
    
    Args:
        B: Bottoms flow rate (mol/h)
        Vm: Vapor flow in stripping section (mol/h)
        delta_H_vap: Heat of vaporization (kJ/mol)
    
    Returns:
        Reboiler duty (kJ/h)
    """
    check_positive("B", B)
    check_positive("Vm", Vm)
    check_positive("delta_H_vap", delta_H_vap)
    
    return B * Vm * delta_H_vap


@dataclass
class HydraulicsSpec:
    """Specification for tower hydraulics calculations"""
    V_flow: float           # Vapor flow rate
    L_flow: float           # Liquid flow rate
    rho_V: float            # Vapor density
    rho_L: float            # Liquid density
    sigma: float            # Surface tension
    tray_spacing: float     # Tray spacing (in)
    Kv: Optional[float] = None  # Kv from Fig. 26.6-1
    D: Optional[float] = None   # Distillate rate
    B: Optional[float] = None   # Bottoms rate
    R: Optional[float] = None   # Reflux ratio
    delta_H_vap: Optional[float] = None  # Heat of vaporization


class HydraulicsSolver:
    """Solver for tower hydraulics calculations"""
    
    def __init__(self, spec: HydraulicsSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        check_positive("V_flow", self.spec.V_flow)
        check_positive("L_flow", self.spec.L_flow)
        check_positive("rho_V", self.spec.rho_V)
        check_positive("rho_L", self.spec.rho_L)
        check_positive("sigma", self.spec.sigma)
        check_positive("tray_spacing", self.spec.tray_spacing)
    
    def estimate_Kv(self) -> float:
        """Estimate Kv from tray spacing and flow parameter"""
        # Flow parameter = (L/V) √(ρV/ρL)
        flow_param = (self.spec.L_flow / self.spec.V_flow) * math.sqrt(
            self.spec.rho_V / self.spec.rho_L
        )
        
        # Simplified correlation based on Fig. 26.6-1
        # For 24-inch tray spacing
        if abs(self.spec.tray_spacing - 24.0) < 1.0:
            if flow_param < 0.01:
                return 0.40
            elif flow_param < 0.1:
                return 0.40 - 0.15 * math.log10(flow_param / 0.01)
            else:
                return 0.28 - 0.08 * math.log10(flow_param / 0.1)
        else:
            # Scale for other tray spacings (simplified)
            base_Kv = 0.35
            spacing_factor = (self.spec.tray_spacing / 24.0) ** 0.5
            return base_Kv * spacing_factor
    
    def calculate_flooding_velocity(self) -> float:
        """Calculate maximum vapor velocity - Eq. 26.6-1"""
        Kv = self.spec.Kv if self.spec.Kv else self.estimate_Kv()
        return flooding_velocity(
            Kv, self.spec.sigma, self.spec.rho_L, self.spec.rho_V
        )
    
    def calculate_diameter(self) -> float:
        """Calculate tower diameter"""
        v_max = self.calculate_flooding_velocity()
        return tower_diameter(self.spec.V_flow, self.spec.rho_V, v_max)
    
    def calculate_condenser_duty(self, condenser_type: str = "total") -> float:
        """Calculate condenser duty - Eqs. 26.6-2 or 26.6-3"""
        if None in [self.spec.D, self.spec.R, self.spec.delta_H_vap]:
            raise ValueError("Need D, R, and delta_H_vap for condenser duty")
        return condenser_duty(
            self.spec.D, self.spec.R, self.spec.delta_H_vap, condenser_type
        )
    
    def calculate_reboiler_duty(self) -> float:
        """Calculate reboiler duty - Eq. 26.6-4"""
        if None in [self.spec.B, self.spec.V_flow, self.spec.delta_H_vap]:
            raise ValueError("Need B, V_flow, and delta_H_vap for reboiler duty")
        return reboiler_duty(self.spec.B, self.spec.V_flow, self.spec.delta_H_vap)
    
    def solve(self) -> Dict[str, Any]:
        """Complete hydraulics analysis"""
        v_max = self.calculate_flooding_velocity()
        diameter = self.calculate_diameter()
        
        result = {
            "flooding_velocity": v_max,
            "tower_diameter": diameter,
            "Kv_used": self.spec.Kv if self.spec.Kv else self.estimate_Kv(),
        }
        
        if self.spec.D and self.spec.R and self.spec.delta_H_vap:
            result["condenser_duty_total"] = self.calculate_condenser_duty("total")
            result["condenser_duty_partial"] = self.calculate_condenser_duty("partial")
        
        if self.spec.B and self.spec.V_flow and self.spec.delta_H_vap:
            result["reboiler_duty"] = self.calculate_reboiler_duty()
        
        return result