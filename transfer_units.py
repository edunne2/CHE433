"""HTU/NTU method for packed columns - Eqs. 22.5-32 to 22.5-55"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
import math

from bank.core.validation import (
    check_positive, check_in_closed_01, ChemEngError
)
from bank.core.numerical import integrate_trapezoid, bisection, log_mean


@dataclass
class TransferUnitsSpec:
    """Specification for HTU/NTU method - Eqs. 22.5-32 to 22.5-35"""
    NOG: Optional[float] = None
    NOL: Optional[float] = None
    HOG: Optional[float] = None
    HOL: Optional[float] = None
    H: Optional[float] = None
    y_in: Optional[float] = None
    y_out: Optional[float] = None
    x_in: Optional[float] = None
    x_out: Optional[float] = None
    L: Optional[float] = None
    G: Optional[float] = None
    y_eq_func: Optional[Callable] = None
    x_eq_func: Optional[Callable] = None
    S: float = 1.0  # Cross-sectional area
    tol: float = 1e-12
    n_int: int = 1000


class TransferUnitsSolver:
    """
    Solver for HTU/NTU method - Eqs. 22.5-32 to 22.5-55
    
    Key relationships:
    - z = HG * NG = HL * NL = HOG * NOG = HOL * NOL (Eq. 22.5-40)
    - HG = V/(k'ya S) (Eq. 22.5-36)
    - HL = L/(k'xa S) (Eq. 22.5-37)
    - HOG = V/(K'ya S) (Eq. 22.5-38)
    - HOL = L/(K'xa S) (Eq. 22.5-39)
    - NOG = ∫ dy/(y - y*) (Eq. 22.5-43)
    - NOL = ∫ dx/(x* - x) (Eq. 22.5-44)
    """
    
    def __init__(self, spec: TransferUnitsSpec):
        self.spec = spec
        self._validate()
        self._x_out_calc = None
        self._y_out_calc = None
        self._NOG = None
        self._NOL = None
    
    def _validate(self):
        """Validate inputs"""
        n_transfer = sum([
            self.spec.NOG is not None,
            self.spec.NOL is not None,
            self.spec.H is not None
        ])
        
        n_htu = sum([
            self.spec.HOG is not None,
            self.spec.HOL is not None
        ])
        
        # If we don't have NTU or HTU, we need operating line data for integration
        if n_transfer == 0 and n_htu == 0:
            if not all([
                self.spec.y_in is not None,
                self.spec.y_out is not None,
                self.spec.x_in is not None,
                self.spec.x_out is not None,
                self.spec.L is not None,
                self.spec.G is not None,
                self.spec.y_eq_func is not None
            ]):
                raise ChemEngError(
                    "Insufficient data. Provide either NTU/HTU values or "
                    "complete operating line data for integration"
                )
    
    def _operating_line(self, y: float) -> float:
        """Get x at given y from operating line - Eq. 22.5-2"""
        if self.spec.L is None or self.spec.G is None:
            raise ChemEngError("L and G must be specified for operating line")
        return self.spec.x_in + (self.spec.L / self.spec.G) * (self.spec.y_in - y)
    
    def _operating_line_y(self, x: float) -> float:
        """Get y at given x from operating line - Eq. 22.5-2 rearranged"""
        if self.spec.L is None or self.spec.G is None:
            raise ChemEngError("L and G must be specified for operating line")
        return self.spec.y_in + (self.spec.G / self.spec.L) * (self.spec.x_in - x)
    
    def calculate_NOG_from_integral(self) -> float:
        """
        Calculate NOG by integrating ∫ dy/(y - y*) - Eq. 22.5-43
        
        For dilute systems with straight operating and equilibrium lines,
        this can also be calculated using Eq. 22.5-48.
        """
        if not all([
            self.spec.y_in, self.spec.y_out,
            self.spec.x_in, self.spec.x_out,
            self.spec.y_eq_func
        ]):
            raise ChemEngError("Missing data for NOG integration")
        
        def integrand(y: float):
            x = self._operating_line(y)
            y_star = self.spec.y_eq_func(x)
            return 1.0 / (y - y_star)
        
        return integrate_trapezoid(
            integrand, 
            self.spec.y_out, 
            self.spec.y_in, 
            n=self.spec.n_int
        )
    
    def calculate_NOL_from_integral(self) -> float:
        """
        Calculate NOL by integrating ∫ dx/(x* - x) - Eq. 22.5-44
        
        Note: Requires inverse equilibrium function x* = f_inv(y)
        """
        if not all([
            self.spec.x_in, self.spec.x_out,
            self.spec.y_in, self.spec.y_out,
            self.spec.x_eq_func
        ]):
            raise ChemEngError("Missing data for NOL integration")
        
        def integrand(x: float):
            y = self._operating_line_y(x)
            x_star = self.spec.x_eq_func(y)
            return 1.0 / (x_star - x)
        
        return integrate_trapezoid(
            integrand, 
            self.spec.x_in, 
            self.spec.x_out, 
            n=self.spec.n_int
        )
    
    def calculate_NOG_analytical(self) -> float:
        """
        Calculate NOG using analytical equation for straight lines - Eq. 22.5-48
        
        NOG = [1/(1-1/A)] ln[ (1-1/A)((y1 - m x2)/(y2 - m x2)) + 1/A ]
        """
        if not all([
            self.spec.y_in, self.spec.y_out,
            self.spec.x_in, self.spec.x_out,
            self.spec.L, self.spec.G
        ]):
            raise ChemEngError("Missing data for analytical NOG calculation")
        
        # Estimate average m from equilibrium
        if self.spec.y_eq_func is None:
            raise ChemEngError("Need equilibrium function for analytical NOG")
        
        x_avg = (self.spec.x_in + self.spec.x_out) / 2
        try:
            # Approximate m as derivative
            m = (self.spec.y_eq_func(x_avg + 1e-6) - self.spec.y_eq_func(x_avg)) / 1e-6
        except:
            m = 1.0
        
        A = self.spec.L / (m * self.spec.G)
        
        if abs(A - 1.0) < 1e-10:
            return (self.spec.y_in - self.spec.y_out) / (self.spec.y_out - m * self.spec.x_in)
        
        term = (self.spec.y_in - m * self.spec.x_in) / (self.spec.y_out - m * self.spec.x_in)
        return (1 / (1 - 1/A)) * math.log(term * (1 - 1/A) + 1/A)
    
    def relate_NOG_and_NOL(self) -> Dict[str, float]:
        """
        Relate gas-phase and liquid-phase transfer units - Eqs. 22.5-54, 22.5-55
        
        HOG = HG + (mV/L) HL
        HOL = HL + (L/(mV)) HG
        """
        if not all([self.spec.L, self.spec.G, self.spec.y_eq_func]):
            return {}
        
        x_avg = (self.spec.x_in + self.spec.x_out) / 2 if self.spec.x_out else self.spec.x_in
        try:
            # Approximate m as derivative
            m = (self.spec.y_eq_func(x_avg + 1e-6) - self.spec.y_eq_func(x_avg)) / 1e-6
        except:
            m = 1.0
        
        lambda_factor = m * self.spec.G / self.spec.L
        
        return {
            "lambda": lambda_factor,
            "NOL_to_NOG_ratio": 1.0 / lambda_factor if lambda_factor > 0 else float('inf'),
            "NOG_to_NOL_ratio": lambda_factor,
        }
    
    def solve(self) -> Dict[str, Any]:
        """Main solving method - implements Eqs. 22.5-32 to 22.5-55"""
        result = {}
        
        # Calculate missing values for gas phase
        if self.spec.NOG is None and self.spec.H is not None and self.spec.HOG is not None:
            result["NOG"] = self.spec.H / self.spec.HOG
            self._NOG = result["NOG"]
        elif self.spec.NOG is None and self.spec.y_eq_func is not None:
            try:
                result["NOG"] = self.calculate_NOG_from_integral()
                self._NOG = result["NOG"]
            except Exception as e:
                result["NOG"] = None
                result["NOG_warning"] = str(e)
        else:
            result["NOG"] = self.spec.NOG
            self._NOG = self.spec.NOG
        
        # Calculate missing values for liquid phase
        if self.spec.NOL is None and self.spec.H is not None and self.spec.HOL is not None:
            result["NOL"] = self.spec.H / self.spec.HOL
            self._NOL = result["NOL"]
        elif self.spec.NOL is None and self.spec.x_eq_func is not None:
            try:
                result["NOL"] = self.calculate_NOL_from_integral()
                self._NOL = result["NOL"]
            except Exception as e:
                result["NOL"] = None
                result["NOL_warning"] = str(e)
        else:
            result["NOL"] = self.spec.NOL
            self._NOL = self.spec.NOL
        
        # Calculate HOG if possible
        if self.spec.HOG is None and self.spec.H is not None and result["NOG"] is not None:
            result["HOG"] = self.spec.H / result["NOG"]
        else:
            result["HOG"] = self.spec.HOG
        
        # Calculate HOL if possible
        if self.spec.HOL is None and self.spec.H is not None and result.get("NOL") is not None:
            result["HOL"] = self.spec.H / result["NOL"]
        else:
            result["HOL"] = self.spec.HOL
        
        # Calculate column height if possible
        if self.spec.H is None and result["NOG"] is not None and result["HOG"] is not None:
            result["H"] = result["NOG"] * result["HOG"]
        elif self.spec.H is None and result.get("NOL") is not None and result.get("HOL") is not None:
            result["H"] = result["NOL"] * result["HOL"]
        else:
            result["H"] = self.spec.H
        
        # Calculate relationships between NOG and NOL
        result["relationships"] = self.relate_NOG_and_NOL()
        
        # Calculate HETP if possible (Eq. 22.5-52)
        if result["HOG"] is not None and result["NOG"] is not None:
            try:
                x_avg = (self.spec.x_in + self.spec.x_out) / 2 if self.spec.x_out else self.spec.x_in
                if self.spec.y_eq_func:
                    m = (self.spec.y_eq_func(x_avg + 1e-6) - self.spec.y_eq_func(x_avg)) / 1e-6
                    A = self.spec.L / (m * self.spec.G) if self.spec.L and self.spec.G else 1.0
                    if abs(A - 1.0) > 1e-12:
                        result["HETP"] = result["HOG"] * math.log(1/A) / ((1 - A) / A)
                    else:
                        result["HETP"] = result["HOG"]
                else:
                    result["HETP"] = None
            except:
                result["HETP"] = None
        else:
            result["HETP"] = None
        
        return {
            "inputs": {
                "x_in": self.spec.x_in,
                "x_out": self.spec.x_out,
                "y_in": self.spec.y_in,
                "y_out": self.spec.y_out,
                "L": self.spec.L,
                "G": self.spec.G,
                "S": self.spec.S,
            },
            "transfer_units": result,
        }