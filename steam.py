# bank/separations/distillation/steam.py
"""Steam distillation for immiscible systems"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

from bank.core.validation import (
    check_positive, check_in_closed_01, ChemEngError, InputError
)
from bank.core.numerical import bisection
from bank.separations.equilibrium import VaporPressureTable  # Fixed import


@dataclass
class SteamDistillationSpec:
    """
    Specification for steam distillation.
    
    For immiscible systems, total pressure = P_water(T) + P_organic(T)
    """
    P_total: float                    # Total pressure
    vp_water: VaporPressureTable      # Vapor pressure table for water
    vp_organic: VaporPressureTable    # Vapor pressure table for organic
    
    # Feed
    F_organic: Optional[float] = None  # Moles of organic
    F_water: Optional[float] = None    # Moles of water (if present)
    S_steam: Optional[float] = None    # Steam injected
    
    T: Optional[float] = None          # Operating temperature
    T_guess: float = 100.0              # Initial guess
    
    tol: float = 1e-10
    maxiter: int = 300


class SteamDistillationSolver:
    """Solver for steam distillation"""
    
    def __init__(self, spec: SteamDistillationSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        """Validate inputs"""
        check_positive("P_total", self.spec.P_total)
        
        # Check that vapor pressure tables are consistent
        if not self.spec.vp_water.check_consistency():
            raise ChemEngError("Water vapor pressure table not monotonic")
        if not self.spec.vp_organic.check_consistency():
            raise ChemEngError("Organic vapor pressure table not monotonic")
    
    def boiling_temperature(self) -> float:
        """Find T where P_water(T) + P_organic(T) = P_total"""
        
        def f(T: float) -> float:
            Pw = self.spec.vp_water.Psat_of_T(T)
            Po = self.spec.vp_organic.Psat_of_T(T)
            return (Pw + Po) - self.spec.P_total
        
        # Find temperature range from tables
        T_min = min(self.spec.vp_water.T[0], self.spec.vp_organic.T[0])
        T_max = max(self.spec.vp_water.T[-1], self.spec.vp_organic.T[-1])
        
        return bisection(f, T_min, T_max, tol=self.spec.tol, 
                        maxiter=self.spec.maxiter, expand_bracket=True)
    
    def solve(self) -> Dict[str, Any]:
        """Solve steam distillation problem"""
        
        # Determine temperature
        if self.spec.T is not None:
            T = self.spec.T
        else:
            T = self.boiling_temperature()
        
        # Vapor pressures at T
        P_water = self.spec.vp_water.Psat_of_T(T)
        P_organic = self.spec.vp_organic.Psat_of_T(T)
        
        # Vapor compositions
        y_water = P_water / self.spec.P_total
        y_organic = P_organic / self.spec.P_total
        
        # Normalize (should already sum to 1, but just in case)
        total = y_water + y_organic
        y_water /= total
        y_organic /= total
        
        # Steam requirements
        steam_ratio = y_water / y_organic
        
        if self.spec.F_organic is not None:
            steam_needed = steam_ratio * self.spec.F_organic
            water_from_feed = self.spec.F_water or 0.0
            steam_external = max(0.0, steam_needed - water_from_feed)
        else:
            steam_needed = None
            steam_external = None
        
        return {
            "operating_temperature": T,
            "vapor_pressures": {
                "P_water": P_water,
                "P_organic": P_organic,
                "sum": P_water + P_organic,
                "residual": (P_water + P_organic) - self.spec.P_total,
            },
            "vapor_composition": {
                "y_water": y_water,
                "y_organic": y_organic,
                "steam_to_organic_ratio": steam_ratio,
            },
            "steam_requirements": {
                "steam_needed_mol": steam_needed,
                "steam_external_mol": steam_external,
                "water_from_feed_mol": self.spec.F_water,
            }
        }