# bank/separations/mass_transfer/two_film.py
"""Two-film theory for interphase mass transfer"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import math

from bank.core.validation import (
    check_positive, check_in_closed_01, ChemEngError, InputError
)
from bank.separations.equilibrium import EquilibriumModel


@dataclass
class TwoFilmSpec:
    """
    Specification for two-film mass transfer model.
    
    The two-film theory assumes:
    - Resistance to mass transfer is in two thin films (gas and liquid)
    - Equilibrium at the interface
    - Bulk phases are well-mixed
    """
    # Mass transfer coefficients
    kG: Optional[float] = None  # Gas film coefficient (mol/s·m²·Pa or similar)
    kL: Optional[float] = None  # Liquid film coefficient (m/s or similar)
    KG: Optional[float] = None  # Overall gas coefficient
    KL: Optional[float] = None  # Overall liquid coefficient
    
    # Equilibrium
    eq: Optional[EquilibriumModel] = None  # For calculating m = dy/dx
    
    # Concentrations
    y_bulk: Optional[float] = None  # Gas bulk mole fraction
    x_bulk: Optional[float] = None  # Liquid bulk mole fraction
    y_interface: Optional[float] = None  # Gas interface mole fraction
    x_interface: Optional[float] = None  # Liquid interface mole fraction
    
    # Physical properties
    m: Optional[float] = None  # Slope of equilibrium line (dy/dx)
    H: Optional[float] = None  # Henry's constant if applicable
    
    # Numerical
    tol: float = 1e-12
    maxiter: int = 400


class TwoFilmModel:
    """
    Two-film theory solver.
    
    Relates:
    - Individual film coefficients (kG, kL)
    - Overall coefficients (KG, KL)
    - Interface concentrations (y_i, x_i)
    - Bulk concentrations (y_b, x_b)
    
    Key relationships:
    1/KG = 1/kG + m/kL
    1/KL = 1/(m*kG) + 1/kL
    N = kG(y_b - y_i) = kL(x_i - x_b) = KG(y_b - y*) = KL(x* - x_b)
    """
    
    def __init__(self, spec: TwoFilmSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        """Validate inputs"""
        # Check we have enough information
        n_coefficients = sum([
            self.spec.kG is not None,
            self.spec.kL is not None,
            self.spec.KG is not None,
            self.spec.KL is not None
        ])
        
        if n_coefficients == 0:
            raise InputError("Must provide at least one mass transfer coefficient")
        
        # Need equilibrium relationship for overall coefficients
        if (self.spec.KG is not None or self.spec.KL is not None) and self.spec.m is None:
            if self.spec.eq is None:
                raise InputError("Need equilibrium model or m for overall coefficients")
    
    def _get_m(self) -> float:
        """Get slope of equilibrium line"""
        if self.spec.m is not None:
            return self.spec.m
        
        if self.spec.eq is not None and self.spec.x_bulk is not None:
            # Approximate m at bulk conditions
            x = self.spec.x_bulk
            y = self.spec.eq.y_of_x(x)
            # Small perturbation for derivative
            dx = 1e-6
            y_plus = self.spec.eq.y_of_x(min(1.0, x + dx))
            return (y_plus - y) / dx
        
        raise ChemEngError("Cannot determine slope m")
    
    def calculate_overall_coefficients(self) -> Dict[str, float]:
        """Calculate overall coefficients from film coefficients"""
        if self.spec.kG is None or self.spec.kL is None:
            raise ChemEngError("Need both kG and kL to calculate overall coefficients")
        
        m = self._get_m()
        
        # Overall gas coefficient
        KG = 1.0 / (1.0/self.spec.kG + m/self.spec.kL)
        
        # Overall liquid coefficient
        KL = 1.0 / (1.0/(m*self.spec.kG) + 1.0/self.spec.kL)
        
        return {
            "KG": KG,
            "KL": KL,
            "m": m,
            "resistance_gas_fraction": (1.0/self.spec.kG) / (1.0/self.spec.kG + m/self.spec.kL),
            "resistance_liquid_fraction": (m/self.spec.kL) / (1.0/self.spec.kG + m/self.spec.kL),
        }
    
    def calculate_film_coefficients(self) -> Dict[str, float]:
        """Calculate film coefficients from overall coefficients"""
        if self.spec.KG is None and self.spec.KL is None:
            raise ChemEngError("Need at least one overall coefficient")
        
        m = self._get_m()
        
        if self.spec.KG is not None:
            # Have KG, need relationship between kG and kL
            # This is underdetermined - need additional info
            if self.spec.kG is not None:
                kL = 1.0 / ((1.0/self.spec.KG - 1.0/self.spec.kG) / m)
                return {"kL": kL, "kG": self.spec.kG, "m": m}
            elif self.spec.kL is not None:
                kG = 1.0 / (1.0/self.spec.KG - m/self.spec.kL)
                return {"kG": kG, "kL": self.spec.kL, "m": m}
            else:
                raise InputError("Need one film coefficient to determine the other")
        
        else:  # self.spec.KL is not None
            if self.spec.kL is not None:
                kG = 1.0 / (m * (1.0/self.spec.KL - 1.0/self.spec.kL))
                return {"kG": kG, "kL": self.spec.kL, "m": m}
            elif self.spec.kG is not None:
                kL = 1.0 / (1.0/self.spec.KL - 1.0/(m*self.spec.kG))
                return {"kG": self.spec.kG, "kL": kL, "m": m}
            else:
                raise InputError("Need one film coefficient to determine the other")
    
    def calculate_interface_concentrations(self) -> Dict[str, float]:
        """
        Calculate interface concentrations from bulk concentrations.
        
        At interface: y_i and x_i are in equilibrium
        Flux equality: kG(y_b - y_i) = kL(x_i - x_b)
        """
        if None in [self.spec.kG, self.spec.kL, self.spec.y_bulk, self.spec.x_bulk]:
            raise InputError("Need kG, kL, y_bulk, and x_bulk")
        
        if self.spec.eq is None and self.spec.H is None:
            raise InputError("Need equilibrium relationship")
        
        from bank.core.numerical import bisection
        
        def residual(y_i: float):
            # From flux equality, find corresponding x_i
            x_i = self.spec.x_bulk + (self.spec.kG / self.spec.kL) * (self.spec.y_bulk - y_i)
            
            # Check equilibrium
            if self.spec.eq is not None:
                y_eq = self.spec.eq.y_of_x(x_i)
            else:  # Henry's law
                y_eq = self.spec.H * x_i
            
            return y_i - y_eq
        
        # Find y_i
        y_low = 0.0
        y_high = max(self.spec.y_bulk, 0.1)
        
        try:
            y_i = bisection(residual, y_low, y_high, tol=self.spec.tol)
        except:
            # Try expanding bracket
            y_i = bisection(residual, y_low, y_high, expand_bracket=True, tol=self.spec.tol)
        
        x_i = self.spec.x_bulk + (self.spec.kG / self.spec.kL) * (self.spec.y_bulk - y_i)
        
        # Calculate flux
        N = self.spec.kG * (self.spec.y_bulk - y_i)
        
        return {
            "interface": {
                "y_interface": y_i,
                "x_interface": x_i,
            },
            "flux": N,
            "driving_forces": {
                "gas_phase": self.spec.y_bulk - y_i,
                "liquid_phase": x_i - self.spec.x_bulk,
            }
        }
    
    def solve(self) -> Dict[str, Any]:
        """Main solving method"""
        result = {}
        
        # Determine what we can calculate
        if self.spec.kG is not None and self.spec.kL is not None:
            # Can calculate overall coefficients
            result["overall_coefficients"] = self.calculate_overall_coefficients()
        
        if (self.spec.KG is not None or self.spec.KL is not None) and \
           (self.spec.kG is not None or self.spec.kL is not None):
            # Can calculate missing film coefficients
            result["film_coefficients"] = self.calculate_film_coefficients()
        
        if all([self.spec.kG, self.spec.kL, self.spec.y_bulk, self.spec.x_bulk]):
            # Can calculate interface concentrations
            result["interface"] = self.calculate_interface_concentrations()
        
        return result