# bank/separations/mass_transfer/flux_equations.py
"""Molar flux equations for mass transfer"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
import math

from bank.core.validation import check_positive, check_in_closed_01, ChemEngError


@dataclass
class FluxSpec:
    """
    Specification for molar flux calculations.
    
    Handles:
    - Equimolar counter-diffusion (NA = -NB)
    - Unimolar diffusion (NB = 0)
    - Multicomponent (simplified)
    """
    # Driving force
    y1: Optional[float] = None  # Mole fraction at point 1
    y2: Optional[float] = None  # Mole fraction at point 2
    x1: Optional[float] = None  # Liquid mole fraction at point 1
    x2: Optional[float] = None  # Liquid mole fraction at point 2
    
    # Mass transfer coefficient
    kG: Optional[float] = None  # Gas phase coefficient
    kL: Optional[float] = None  # Liquid phase coefficient
    
    # Total concentration
    c_total: Optional[float] = None  # Total molar concentration
    P_total: Optional[float] = None  # Total pressure (for gas)
    
    # Diffusion type
    diffusion_type: str = "equimolar"  # "equimolar" or "unimolar"
    
    # For unimolar diffusion (NB = 0)
    y_BM: Optional[float] = None  # Log mean of inert


class MolarFlux:
    """Calculator for molar flux equations"""
    
    def __init__(self, spec: FluxSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        """Validate inputs"""
        if self.spec.diffusion_type not in ["equimolar", "unimolar"]:
            raise ValueError("diffusion_type must be 'equimolar' or 'unimolar'")
        
        # Check we have driving force
        if self.spec.y1 is None or self.spec.y2 is None:
            if self.spec.x1 is None or self.spec.x2 is None:
                raise ChemEngError("Need either (y1,y2) or (x1,x2)")
    
    def _driving_force_gas(self) -> float:
        """Calculate gas phase driving force"""
        if self.spec.y1 is not None and self.spec.y2 is not None:
            return self.spec.y1 - self.spec.y2
        else:
            # Would need equilibrium to convert liquid to gas
            raise NotImplementedError("Need gas compositions")
    
    def _driving_force_liquid(self) -> float:
        """Calculate liquid phase driving force"""
        if self.spec.x1 is not None and self.spec.x2 is not None:
            return self.spec.x1 - self.spec.x2
        else:
            raise NotImplementedError("Need liquid compositions")
    
    def _log_mean(self, a: float, b: float) -> float:
        """Calculate log mean"""
        if abs(a - b) < 1e-12:
            return a
        return (a - b) / math.log(a / b)
    
    def calculate_flux_equimolar(self) -> Dict[str, float]:
        """
        Flux for equimolar counter-diffusion (NA = -NB).
        N = kG (y1 - y2) for gas phase
        N = kL (x1 - x2) for liquid phase
        """
        result = {}
        
        if self.spec.kG is not None:
            dy = self._driving_force_gas()
            N = self.spec.kG * dy
            result["N_gas"] = N
            result["driving_force_gas"] = dy
        
        if self.spec.kL is not None:
            dx = self._driving_force_liquid()
            N = self.spec.kL * dx
            result["N_liquid"] = N
            result["driving_force_liquid"] = dx
        
        return result
    
    def calculate_flux_unimolar(self) -> Dict[str, float]:
        """
        Flux for unimolar diffusion (NB = 0, e.g., absorption).
        N = kG' (y1 - y2) where kG' = kG / y_BM
        y_BM is log mean of inert gas fraction.
        """
        if self.spec.y1 is None or self.spec.y2 is None:
            raise ChemEngError("Need y1 and y2 for unimolar diffusion")
        
        if self.spec.y_BM is not None:
            y_BM = self.spec.y_BM
        else:
            # Calculate log mean of (1-y)
            y_BM = self._log_mean(1 - self.spec.y1, 1 - self.spec.y2)
        
        if self.spec.kG is not None:
            kG_prime = self.spec.kG / y_BM
            dy = self.spec.y1 - self.spec.y2
            N = kG_prime * dy
            
            return {
                "N": N,
                "kG_prime": kG_prime,
                "y_BM": y_BM,
                "driving_force": dy,
                "enhancement_factor": 1.0 / y_BM,
            }
        else:
            raise ChemEngError("Need kG for unimolar diffusion")
    
    def calculate_flux_with_concentration(self) -> Dict[str, float]:
        """
        Calculate flux using concentration driving force.
        N = kG (c1 - c2) for gas
        N = kL (c1 - c2) for liquid
        """
        if self.spec.c_total is None:
            raise ChemEngError("Need total concentration")
        
        result = {}
        
        if self.spec.kG is not None and self.spec.y1 is not None and self.spec.y2 is not None:
            c1 = self.spec.c_total * self.spec.y1
            c2 = self.spec.c_total * self.spec.y2
            N = self.spec.kG * (c1 - c2)
            result["N_gas"] = N
            result["dc_gas"] = c1 - c2
        
        if self.spec.kL is not None and self.spec.x1 is not None and self.spec.x2 is not None:
            c1 = self.spec.c_total * self.spec.x1
            c2 = self.spec.c_total * self.spec.x2
            N = self.spec.kL * (c1 - c2)
            result["N_liquid"] = N
            result["dc_liquid"] = c1 - c2
        
        return result
    
    def solve(self) -> Dict[str, Any]:
        """Main solving method"""
        
        if self.spec.diffusion_type == "equimolar":
            result = self.calculate_flux_equimolar()
        else:  # unimolar
            result = self.calculate_flux_unimolar()
        
        return {
            "specification": self.spec.__dict__,
            "results": result,
            "flux_type": self.spec.diffusion_type,
        }