# bank/separations/mass_transfer/coefficients.py
"""Mass transfer coefficients correlations"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import math

from bank.core.validation import check_positive, ChemEngError


@dataclass
class CoefficientSpec:
    """
    Specification for mass transfer coefficient correlations.
    
    Common correlations:
    - Sherwood = f(Re, Sc) for forced convection
    - Chilton-Colburn j-factor analogy
    - Onda et al. for packed columns
    """
    # Flow conditions
    Re: Optional[float] = None      # Reynolds number
    Sc: Optional[float] = None      # Schmidt number
    Sh: Optional[float] = None      # Sherwood number (to be calculated)
    
    # Geometry
    d_p: Optional[float] = None     # Particle/packing diameter
    L: Optional[float] = None       # Characteristic length
    
    # Physical properties
    D: Optional[float] = None       # Diffusivity
    rho: Optional[float] = None     # Density
    mu: Optional[float] = None      # Viscosity
    u: Optional[float] = None       # Velocity
    
    # Correlation type
    correlation: str = "default"


def sherwood_correlation(
    Re: float,
    Sc: float,
    correlation: str = "default",
) -> float:
    """
    Calculate Sherwood number from Reynolds and Schmidt numbers.
    
    Common correlations:
    - default: Sh = 2 + 0.6 Re^0.5 Sc^0.33 (sphere)
    - laminar: Sh = 0.664 Re^0.5 Sc^0.33 (flat plate)
    - turbulent: Sh = 0.037 Re^0.8 Sc^0.33
    - packed_bed: Sh = 2 + 1.1 Re^0.6 Sc^0.33
    """
    check_positive("Re", Re)
    check_positive("Sc", Sc)
    
    correlation = correlation.lower()
    
    if correlation == "default" or correlation == "sphere":
        return 2.0 + 0.6 * math.sqrt(Re) * Sc**(1.0/3.0)
    elif correlation == "laminar":
        return 0.664 * math.sqrt(Re) * Sc**(1.0/3.0)
    elif correlation == "turbulent":
        return 0.037 * Re**0.8 * Sc**(1.0/3.0)
    elif correlation == "packed_bed":
        return 2.0 + 1.1 * Re**0.6 * Sc**(1.0/3.0)
    else:
        raise ValueError(f"Unknown correlation: {correlation}")


def chilton_colburn_j_factor(
    Sh: float,
    Re: float,
    Sc: float,
) -> float:
    """
    Calculate Chilton-Colburn j-factor for mass transfer.
    j_D = Sh/(Re * Sc^(1/3))
    """
    check_positive("Sh", Sh)
    check_positive("Re", Re)
    check_positive("Sc", Sc)
    
    return Sh / (Re * Sc**(1.0/3.0))


class MassTransferCoefficients:
    """Calculator for mass transfer coefficients"""
    
    def __init__(self, spec: CoefficientSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        """Validate inputs"""
        # Check we can calculate Re if not provided
        if self.spec.Re is None:
            if all([self.spec.rho, self.spec.u, self.spec.L, self.spec.mu]):
                self.Re = self.spec.rho * self.spec.u * self.spec.L / self.spec.mu
            else:
                raise ChemEngError("Cannot calculate Re - need rho, u, L, mu")
        else:
            self.Re = self.spec.Re
        
        # Check we can calculate Sc if not provided
        if self.spec.Sc is None:
            if all([self.spec.mu, self.spec.rho, self.spec.D]):
                self.Sc = self.spec.mu / (self.spec.rho * self.spec.D)
            else:
                raise ChemEngError("Cannot calculate Sc - need mu, rho, D")
        else:
            self.Sc = self.spec.Sc
    
    def calculate_Sh(self) -> float:
        """Calculate Sherwood number"""
        if self.spec.Sh is not None:
            return self.spec.Sh
        
        return sherwood_correlation(
            self.Re,
            self.Sc,
            correlation=self.spec.correlation
        )
    
    def calculate_k(self) -> float:
        """
        Calculate mass transfer coefficient from Sherwood number.
        k = Sh * D / L
        """
        Sh = self.calculate_Sh()
        
        if not all([self.spec.D, self.spec.L]):
            raise ChemEngError("Need D and L to calculate k from Sh")
        
        return Sh * self.spec.D / self.spec.L
    
    def calculate_j_factor(self) -> float:
        """Calculate Chilton-Colburn j-factor"""
        Sh = self.calculate_Sh()
        return chilton_colburn_j_factor(Sh, self.Re, self.Sc)
    
    def solve(self) -> Dict[str, Any]:
        """Main solving method"""
        Sh = self.calculate_Sh()
        
        result = {
            "dimensionless_numbers": {
                "Re": self.Re,
                "Sc": self.Sc,
                "Sh": Sh,
            }
        }
        
        # Calculate k if possible
        try:
            k = self.calculate_k()
            result["mass_transfer_coefficient"] = k
        except:
            result["mass_transfer_coefficient"] = None
        
        # Calculate j-factor
        result["j_factor"] = self.calculate_j_factor()
        
        return result


# Additional correlations for specific equipment

def packed_column_kG(
    G: float,
    Sc_G: float,
    d_p: float,
    correlation: str = "onda",
) -> float:
    """
    Gas phase mass transfer coefficient for packed columns.
    
    Onda correlation:
    kG * RT / (a * D_G) = C * (Re_G)^0.7 * (Sc_G)^(1/3) * (a*d_p)^-2.0
    """
    if correlation == "onda":
        # Simplified Onda correlation
        a = 100  # m²/m³ (typical)
        C = 5.23
        Re_G = G * d_p / (1.8e-5)  # Simplified
        
        kG = C * Re_G**0.7 * Sc_G**(1/3) * (a*d_p)**-2.0
        return kG
    else:
        raise ValueError(f"Unknown correlation: {correlation}")


def packed_column_kL(
    L: float,
    Sc_L: float,
    d_p: float,
    correlation: str = "onda",
) -> float:
    """
    Liquid phase mass transfer coefficient for packed columns.
    
    Onda correlation:
    kL * (ρ_L/(μ_L*g))^(1/3) = 0.0051 * (Re_L)^(2/3) * (Sc_L)^(-0.5) * (a*d_p)^0.4
    """
    if correlation == "onda":
        # Simplified Onda correlation
        g = 9.81  # m/s²
        rho_L = 1000  # kg/m³
        mu_L = 0.001  # Pa·s
        
        Re_L = L * d_p / mu_L
        
        kL = 0.0051 * Re_L**(2/3) * Sc_L**(-0.5) * (100*d_p)**0.4
        kL = kL / (rho_L/(mu_L*g))**(1/3)
        return kL
    else:
        raise ValueError(f"Unknown correlation: {correlation}")