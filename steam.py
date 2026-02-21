"""Steam distillation - Eqs. 26.3-12 to 26.3-15"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from bank.core.validation import check_positive, check_in_closed_01
from bank.core.numerical import bisection


@dataclass
class SteamSpec:
    """Specification for steam distillation"""
    P_total: float           # Total pressure
    water_vp_func: callable  # Function P_water(T)
    organic_vp_func: callable # Function P_organic(T)
    n_organic: Optional[float] = None  # Moles of organic
    n_water_initial: Optional[float] = None  # Initial water
    T: Optional[float] = None
    tol: float = 1e-10
    maxiter: int = 300


class SteamDistillation:
    """
    Steam distillation for immiscible systems.
    
    Eqs. 26.3-12 to 26.3-15:
    PA + PB = P                                           (26.3-13)
    yA = PA/P, yB = PB/P                                   (26.3-14)
    nB/nA = PB/PA                                          (26.3-15)
    """
    
    def __init__(self, spec: SteamSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        check_positive("P_total", self.spec.P_total)
    
    def boiling_temperature(self) -> float:
        """Find T where PA(T) + PB(T) = P_total - Eq. 26.3-13"""
        
        def f(T: float) -> float:
            Pw = self.spec.water_vp_func(T)
            Po = self.spec.organic_vp_func(T)
            return Pw + Po - self.spec.P_total
        
        return bisection(f, 0, 200, tol=self.spec.tol, 
                        maxiter=self.spec.maxiter, expand_bracket=True)
    
    def solve(self) -> Dict[str, Any]:
        """Solve steam distillation problem"""
        
        T = self.spec.T if self.spec.T else self.boiling_temperature()
        
        P_water = self.spec.water_vp_func(T)
        P_organic = self.spec.organic_vp_func(T)
        
        # Vapor compositions - Eq. 26.3-14
        y_water = P_water / self.spec.P_total
        y_organic = P_organic / self.spec.P_total
        
        # Normalize
        total = y_water + y_organic
        y_water /= total
        y_organic /= total
        
        # Mole ratio - Eq. 26.3-15
        steam_ratio = y_water / y_organic
        
        # Steam requirements
        if self.spec.n_organic is not None:
            steam_needed = steam_ratio * self.spec.n_organic
            water_from_feed = self.spec.n_water_initial or 0.0
            steam_external = max(0.0, steam_needed - water_from_feed)
        else:
            steam_needed = None
            steam_external = None
        
        return {
            "operating_temperature": T,
            "vapor_pressures": {
                "P_water": P_water, "P_organic": P_organic,
                "sum": P_water + P_organic,
            },
            "vapor_composition": {
                "y_water": y_water, "y_organic": y_organic,
                "steam_to_organic_ratio": steam_ratio,
            },
            "steam_requirements": {
                "steam_needed": steam_needed,
                "steam_external": steam_external,
            }
        }