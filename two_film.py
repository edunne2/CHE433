"""Two-film theory for interphase mass transfer - Eqs. 22.1-31 to 22.1-38"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

from bank.core.validation import (
    check_positive, check_in_closed_01, ChemEngError, InputError
)
from bank.core.numerical import bisection, log_mean
from bank.separations.equilibrium import EquilibriumModel


@dataclass
class TwoFilmSpec:
    """Specification for two-film mass transfer model - Eqs. 22.1-31 to 22.1-38"""
    kG: Optional[float] = None
    kL: Optional[float] = None
    KG: Optional[float] = None
    KL: Optional[float] = None
    eq: Optional[EquilibriumModel] = None
    y_bulk: Optional[float] = None
    x_bulk: Optional[float] = None
    y_interface: Optional[float] = None
    x_interface: Optional[float] = None
    m: Optional[float] = None
    tol: float = 1e-12
    maxiter: int = 400


class TwoFilmModel:
    """
    Two-film theory solver - Eqs. 22.1-31 to 22.1-38, 22.1-44 to 22.1-47
    
    Key relationships:
    1/KG = 1/kG + m/kL (Eq. 22.1-44)
    1/KL = 1/(m*kG) + 1/kL (Eq. 22.1-47)
    N = kG(y_b - y_i) = kL(x_i - x_b) = KG(y_b - y*) = KL(x* - x_b)
    """
    
    def __init__(self, spec: TwoFilmSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        n_coefficients = sum([
            self.spec.kG is not None,
            self.spec.kL is not None,
            self.spec.KG is not None,
            self.spec.KL is not None
        ])
        
        if n_coefficients == 0:
            raise InputError("Must provide at least one mass transfer coefficient")
        
        if (self.spec.KG is not None or self.spec.KL is not None) and self.spec.m is None:
            if self.spec.eq is None:
                raise InputError("Need equilibrium model or m for overall coefficients")
    
    def _get_m(self) -> float:
        """Get slope of equilibrium line m' or m'' (Eqs. 22.1-42, 22.1-46)"""
        if self.spec.m is not None:
            return self.spec.m
        
        if self.spec.eq is not None and self.spec.x_bulk is not None:
            x = self.spec.x_bulk
            y = self.spec.eq.y_of_x(x)
            dx = 1e-6
            y_plus = self.spec.eq.y_of_x(min(1.0, x + dx))
            return (y_plus - y) / dx
        
        raise ChemEngError("Cannot determine slope m")
    
    def calculate_overall_coefficients(self) -> Dict[str, float]:
        """Calculate overall coefficients from film coefficients - Eqs. 22.1-44, 22.1-47"""
        if self.spec.kG is None or self.spec.kL is None:
            raise ChemEngError("Need both kG and kL to calculate overall coefficients")
        
        m = self._get_m()
        
        KG = 1.0 / (1.0/self.spec.kG + m/self.spec.kL)  # Eq. 22.1-44
        KL = 1.0 / (1.0/(m*self.spec.kG) + 1.0/self.spec.kL)  # Eq. 22.1-47
        
        return {
            "KG": KG, "KL": KL, "m": m,
            "resistance_gas_fraction": (1.0/self.spec.kG) / (1.0/self.spec.kG + m/self.spec.kL),
            "resistance_liquid_fraction": (m/self.spec.kL) / (1.0/self.spec.kG + m/self.spec.kL),
        }
    
    def calculate_interface_concentrations(self) -> Dict[str, float]:
        """
        Calculate interface concentrations - Eqs. 22.1-31, 22.1-32, 22.1-38
        
        At interface: y_i and x_i are in equilibrium
        Flux equality: kG(y_b - y_i) = kL(x_i - x_b)
        """
        if None in [self.spec.kG, self.spec.kL, self.spec.y_bulk, self.spec.x_bulk]:
            raise InputError("Need kG, kL, y_bulk, and x_bulk")
        
        if self.spec.eq is None:
            raise InputError("Need equilibrium relationship")
        
        def residual(x_i: float):
            y_i = self.spec.eq.y_of_x(x_i)
            return self.spec.kG * (self.spec.y_bulk - y_i) - self.spec.kL * (x_i - self.spec.x_bulk)
        
        x_low, x_high = 0.0, 1.0
        try:
            x_i = bisection(residual, x_low, x_high, tol=self.spec.tol, expand_bracket=True)
        except:
            x_i = bisection(residual, x_low, x_high, expand_bracket=True, expand_factor=2.0)
        
        y_i = self.spec.eq.y_of_x(x_i)
        N = self.spec.kG * (self.spec.y_bulk - y_i)
        
        return {
            "interface": {"x_i": x_i, "y_i": y_i},
            "flux": N,
            "driving_forces": {"gas": self.spec.y_bulk - y_i, "liquid": x_i - self.spec.x_bulk},
        }
    
    def solve(self) -> Dict[str, Any]:
        """Main solving method"""
        result = {}
        
        if self.spec.kG is not None and self.spec.kL is not None:
            result["overall_coefficients"] = self.calculate_overall_coefficients()
        
        if all([self.spec.kG, self.spec.kL, self.spec.y_bulk, self.spec.x_bulk]):
            result["interface"] = self.calculate_interface_concentrations()
        
        return result