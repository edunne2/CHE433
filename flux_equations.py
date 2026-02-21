"""Molar flux equations for mass transfer - Eqs. 22.1-31, 22.1-33, 22.1-37"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

from bank.core.validation import check_positive, check_in_closed_01, ChemEngError
from bank.core.numerical import log_mean


@dataclass
class FluxSpec:
    """Specification for molar flux calculations"""
    y1: Optional[float] = None
    y2: Optional[float] = None
    x1: Optional[float] = None
    x2: Optional[float] = None
    kG: Optional[float] = None
    kL: Optional[float] = None
    c_total: Optional[float] = None
    P_total: Optional[float] = None
    diffusion_type: str = "equimolar"  # "equimolar" or "unimolar"
    y_BM: Optional[float] = None


class MolarFlux:
    """Calculator for molar flux equations - Eqs. 22.1-31, 22.1-33, 22.1-37"""
    
    def __init__(self, spec: FluxSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        if self.spec.diffusion_type not in ["equimolar", "unimolar"]:
            raise ValueError("diffusion_type must be 'equimolar' or 'unimolar'")
        
        if self.spec.y1 is None or self.spec.y2 is None:
            if self.spec.x1 is None or self.spec.x2 is None:
                raise ChemEngError("Need either (y1,y2) or (x1,x2)")
    
    def _driving_force_gas(self) -> float:
        """Calculate gas phase driving force"""
        if self.spec.y1 is not None and self.spec.y2 is not None:
            return self.spec.y1 - self.spec.y2
        raise NotImplementedError("Need gas compositions")
    
    def _driving_force_liquid(self) -> float:
        """Calculate liquid phase driving force"""
        if self.spec.x1 is not None and self.spec.x2 is not None:
            return self.spec.x1 - self.spec.x2
        raise NotImplementedError("Need liquid compositions")
    
    def calculate_flux_equimolar(self) -> Dict[str, float]:
        """
        Flux for equimolar counter-diffusion (NA = -NB) - Eq. 22.1-31
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
        Flux for unimolar diffusion (NB = 0) - Eqs. 22.1-33, 22.1-37
        N = kG' (y1 - y2) where kG' = kG / y_BM
        y_BM is log mean of inert gas fraction.
        """
        if self.spec.y1 is None or self.spec.y2 is None:
            raise ChemEngError("Need y1 and y2 for unimolar diffusion")
        
        if self.spec.y_BM is not None:
            y_BM = self.spec.y_BM
        else:
            y_BM = log_mean(1 - self.spec.y1, 1 - self.spec.y2)
        
        if self.spec.kG is not None:
            kG_prime = self.spec.kG / y_BM
            dy = self.spec.y1 - self.spec.y2
            N = kG_prime * dy
            
            return {
                "N": N, "kG_prime": kG_prime, "y_BM": y_BM,
                "driving_force": dy, "enhancement_factor": 1.0 / y_BM,
            }
        else:
            raise ChemEngError("Need kG for unimolar diffusion")
    
    def solve(self) -> Dict[str, Any]:
        """Main solving method"""
        if self.spec.diffusion_type == "equimolar":
            result = self.calculate_flux_equimolar()
        else:
            result = self.calculate_flux_unimolar()
        
        return {"results": result, "flux_type": self.spec.diffusion_type}