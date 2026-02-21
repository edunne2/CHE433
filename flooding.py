"""Flooding correlations for packed extraction towers - Fig. 27.3-3"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

from bank.core.validation import check_positive, ChemEngError
from .utils import dyn_per_cm_to_lbm_per_h2, cp_to_lbm_per_ft_hr


@dataclass
class PackedFloodingSpec:
    """Specification for packed tower flooding calculation"""
    # Packing characteristics
    surface_area: float      # a, ft²/ft³
    void_fraction: float     # ε
    
    # Phase velocities (unknown, to be solved)
    VC: Optional[float] = None  # Continuous phase velocity (ft/h)
    VD: Optional[float] = None  # Dispersed phase velocity (ft/h)
    VD_over_VC: Optional[float] = None  # Ratio VD/VC
    
    # Physical properties
    rho_C: float            # Continuous phase density (lb_m/ft³)
    rho_D: float            # Dispersed phase density (lb_m/ft³)
    mu_C: float             # Continuous phase viscosity (lb_m/ft·h)
    sigma: float            # Interfacial tension (lb_m/h²)
    
    # Operating parameters
    L_flow: Optional[float] = None  # Volumetric flow of continuous phase (ft³/h)
    V_flow: Optional[float] = None  # Volumetric flow of dispersed phase (ft³/h)
    
    # Correlation parameters
    k1: float = 1.0         # Empirical factor (usually 1.0)


class PackedFloodingSolver:
    """
    Solver for packed extraction tower flooding using Fig. 27.3-3 correlation.
    
    The correlation relates:
        Ordinate = (VC^0.5 + VD^0.5)² / (a ε) * (ρC / (Δρ))^0.2 * μC^0.16
        Abscissa = (VC + VD)² a μC / (g_c Δρ) * (ρC / (Δρ))^0.2
    """
    
    def __init__(self, spec: PackedFloodingSpec):
        self.spec = spec
        self._validate()
        self.g_c = 4.17e8  # ft/h² (gravitational constant)
    
    def _validate(self):
        """Validate inputs"""
        check_positive("surface_area", self.spec.surface_area)
        check_positive("void_fraction", self.spec.void_fraction)
        check_positive("rho_C", self.spec.rho_C)
        check_positive("rho_D", self.spec.rho_D)
        check_positive("mu_C", self.spec.mu_C)
        check_positive("sigma", self.spec.sigma)
        
        self.Δρ = abs(self.spec.rho_C - self.spec.rho_D)
        if self.Δρ < 1e-12:
            raise ChemEngError("Density difference too small")
    
    def abscissa(self, VC: float, VD: float) -> float:
        """Calculate abscissa value for Fig. 27.3-3"""
        V_sum = VC + VD
        term1 = (V_sum ** 2) * self.spec.surface_area * self.spec.mu_C
        term2 = self.g_c * self.Δρ
        term3 = (self.spec.rho_C / self.Δρ) ** 0.2
        return (term1 / term2) * term3
    
    def ordinate(self, VC: float, VD: float) -> float:
        """Calculate ordinate value for Fig. 27.3-3"""
        sqrt_term = (math.sqrt(VC) + math.sqrt(VD)) ** 2
        term1 = sqrt_term / (self.spec.surface_area * self.spec.void_fraction)
        term2 = (self.spec.rho_C / self.Δρ) ** 0.2
        term3 = self.spec.mu_C ** 0.16
        return term1 * term2 * term3
    
    def solve_flooding_velocity(self) -> Dict[str, float]:
        """
        Solve for flooding velocities VC and VD.
        
        Uses the correlation from Fig. 27.3-3.
        For a given VD/VC ratio, find velocities at flooding.
        """
        if self.spec.VD_over_VC is None and (self.spec.VD is None or self.spec.VC is None):
            raise ChemEngError("Must specify either VD/VC ratio or individual velocities")
        
        # Get VD/VC ratio
        if self.spec.VD_over_VC is not None:
            r = self.spec.VD_over_VC
        elif self.spec.VD is not None and self.spec.VC is not None:
            r = self.spec.VD / self.spec.VC
        else:
            raise ChemEngError("Insufficient velocity information")
        
        # At flooding, from Fig. 27.3-3, the abscissa value is approximately 170
        # This is an empirical value from the correlation
        abscissa_flood = 170.0
        
        # Solve for VC
        # abscissa = (VC + r*VC)² * a * μC / (g_c * Δρ) * (ρC/Δρ)^0.2 = 170
        VC2 = abscissa_flood * self.g_c * self.Δρ
        VC2 /= (self.spec.surface_area * self.spec.mu_C * (self.spec.rho_C / self.Δρ) ** 0.2)
        VC2 /= (1 + r) ** 2
        
        VC = math.sqrt(VC2)
        VD = r * VC
        
        # Verify ordinate is reasonable
        ord_val = self.ordinate(VC, VD)
        
        return {
            "VC_flood": VC,
            "VD_flood": VD,
            "V_sum_flood": VC + VD,
            "VD_over_VC": r,
            "abscissa": self.abscissa(VC, VD),
            "ordinate": ord_val,
        }
    
    def design_at_percent_flood(self, percent: float = 0.5) -> Dict[str, float]:
        """
        Calculate design velocities at given percentage of flooding.
        
        Args:
            percent: Fraction of flooding (e.g., 0.5 for 50%)
        
        Returns:
            Design velocities and required area/diameter
        """
        flood = self.solve_flooding_velocity()
        
        VC_design = flood["VC_flood"] * percent
        VD_design = flood["VD_flood"] * percent
        
        result = {
            "VC_design": VC_design,
            "VD_design": VD_design,
            "V_sum_design": VC_design + VD_design,
            "percent_flood": percent,
        }
        
        # Calculate tower area if flows are given
        if self.spec.L_flow is not None and self.spec.V_flow is not None:
            # Area based on continuous phase
            area_C = self.spec.L_flow / VC_design
            area_D = self.spec.V_flow / VD_design
            
            # Use the larger area
            area = max(area_C, area_D)
            diameter = 2 * math.sqrt(area / math.pi)
            
            result["area_ft2"] = area
            result["diameter_ft"] = diameter
            result["diameter_m"] = diameter * 0.3048
        
        return result


def flooding_velocity_packed(
    surface_area: float,
    void_fraction: float,
    rho_C: float,
    rho_D: float,
    mu_C: float,
    sigma: float,
    VD_over_VC: float,
    percent_design: float = 0.5,
    L_flow: Optional[float] = None,
    V_flow: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convenience function for packed tower flooding calculation.
    
    Args:
        surface_area: Packing surface area a (ft²/ft³)
        void_fraction: Packing void fraction ε
        rho_C: Continuous phase density (lb_m/ft³)
        rho_D: Dispersed phase density (lb_m/ft³)
        mu_C: Continuous phase viscosity (lb_m/ft·h)
        sigma: Interfacial tension (lb_m/h²)
        VD_over_VC: Ratio of dispersed to continuous phase velocities
        percent_design: Design fraction of flooding (default 0.5)
        L_flow: Continuous phase volumetric flow (ft³/h)
        V_flow: Dispersed phase volumetric flow (ft³/h)
    
    Returns:
        Dictionary with flooding and design velocities
    """
    spec = PackedFloodingSpec(
        surface_area=surface_area,
        void_fraction=void_fraction,
        VD_over_VC=VD_over_VC,
        rho_C=rho_C,
        rho_D=rho_D,
        mu_C=mu_C,
        sigma=sigma,
        L_flow=L_flow,
        V_flow=V_flow,
    )
    
    solver = PackedFloodingSolver(spec)
    return solver.design_at_percent_flood(percent_design)