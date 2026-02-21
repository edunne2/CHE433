"""Mass transfer coefficients correlations - Eqs. 22.8-1 and 22.8-2"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

from bank.core.validation import check_positive, ChemEngError


@dataclass
class CoefficientSpec:
    """Specification for mass transfer coefficient correlations"""
    Re: Optional[float] = None
    Sc: Optional[float] = None
    Sh: Optional[float] = None
    d_p: Optional[float] = None
    L: Optional[float] = None
    D: Optional[float] = None
    rho: Optional[float] = None
    mu: Optional[float] = None
    u: Optional[float] = None
    correlation: str = "default"


def sherwood_correlation(Re: float, Sc: float, correlation: str = "default") -> float:
    """Calculate Sherwood number from Reynolds and Schmidt numbers"""
    check_positive("Re", Re)
    check_positive("Sc", Sc)
    
    correlation = correlation.lower()
    
    if correlation == "default" or correlation == "sphere":
        return 2.0 + 0.6 * math.sqrt(Re) * Sc**(1.0/3.0)
    elif correlation == "laminar":
        return 0.664 * math.sqrt(Re) * Sc**(1.0/3.0)
    elif correlation == "turbulent":
        return 0.037 * Re**0.8 * Sc**(1.0/3.0)
    else:
        raise ValueError(f"Unknown correlation: {correlation}")


def chilton_colburn_j_factor(Sh: float, Re: float, Sc: float) -> float:
    """Calculate Chilton-Colburn j-factor for mass transfer"""
    check_positive("Sh", Sh)
    check_positive("Re", Re)
    check_positive("Sc", Sc)
    return Sh / (Re * Sc**(1.0/3.0))


def HG_correlation(
    fp: float,
    NSc: float,
    Gx: float,
    Gy: float,
    mu: Optional[float] = None
) -> float:
    """
    Gas phase HTU correlation - Eq. 22.8-1
    
    HG = (0.226/fp) (NSc/0.660)^0.5 (Gx/6.782)^(-0.5) (Gy/0.678)^0.35 (SI)
    """
    check_positive("fp", fp)
    check_positive("NSc", NSc)
    check_positive("Gx", Gx)
    check_positive("Gy", Gy)
    
    return (0.226 / fp) * (NSc / 0.660)**0.5 * (Gx / 6.782)**(-0.5) * (Gy / 0.678)**0.35


def HL_correlation(
    fp: float,
    NSc: float,
    Gx: float,
    mu: float
) -> float:
    """
    Liquid phase HTU correlation - Eq. 22.8-2
    
    HL = (0.357/fp) (NSc/372)^0.5 [ (Gx/μ) / (6.782/0.8937e-3) ]^0.3 (SI)
    """
    check_positive("fp", fp)
    check_positive("NSc", NSc)
    check_positive("Gx", Gx)
    check_positive("mu", mu)
    
    term = (Gx / mu) / (6.782 / 0.8937e-3)
    return (0.357 / fp) * (NSc / 372)**0.5 * term**0.3


class MassTransferCoefficients:
    """Calculator for mass transfer coefficients"""
    
    def __init__(self, spec: CoefficientSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        if self.spec.Re is None:
            if all([self.spec.rho, self.spec.u, self.spec.L, self.spec.mu]):
                self.Re = self.spec.rho * self.spec.u * self.spec.L / self.spec.mu
            else:
                raise ChemEngError("Cannot calculate Re - need rho, u, L, mu")
        else:
            self.Re = self.spec.Re
        
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
        return sherwood_correlation(self.Re, self.Sc, correlation=self.spec.correlation)
    
    def calculate_k(self) -> float:
        """Calculate mass transfer coefficient from Sherwood number"""
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
            "dimensionless_numbers": {"Re": self.Re, "Sc": self.Sc, "Sh": Sh},
            "j_factor": self.calculate_j_factor(),
        }
        
        try:
            result["mass_transfer_coefficient"] = self.calculate_k()
        except:
            result["mass_transfer_coefficient"] = None
        
        return result