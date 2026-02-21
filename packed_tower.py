# bank/separations/absorption/packed_tower.py
"""Packed tower design for absorption/stripping"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

from bank.core.validation import (
    check_positive, check_in_closed_01, ChemEngError, InputError
)
from bank.separations.equilibrium import EquilibriumModel


@dataclass
class PackedTowerSpec:
    """
    Specification for packed tower design.
    
    Key variables:
    - H: Packed height
    - HOG: Height of a transfer unit (gas phase)
    - HETP: Height equivalent to a theoretical plate
    - NOG: Number of transfer units (gas phase)
    """
    # Tower parameters
    H: Optional[float] = None      # Packed height
    HOG: Optional[float] = None    # HTU - gas
    HETP: Optional[float] = None   # HETP
    NOG: Optional[float] = None    # NTU - gas
    
    # Operating conditions
    G: float                        # Gas flow rate
    L: float                        # Liquid flow rate
    
    # Compositions
    y_in: float                     # Gas inlet
    y_out: Optional[float] = None   # Gas outlet
    x_in: float                     # Liquid inlet
    x_out: Optional[float] = None   # Liquid outlet
    
    # Equilibrium
    eq: EquilibriumModel
    
    # Physical properties (for HOG/HETP correlations)
    rho_G: Optional[float] = None   # Gas density
    rho_L: Optional[float] = None   # Liquid density
    mu_G: Optional[float] = None    # Gas viscosity
    mu_L: Optional[float] = None    # Liquid viscosity
    D_G: Optional[float] = None     # Gas diffusivity
    D_L: Optional[float] = None     # Liquid diffusivity
    
    # Numerical
    tol: float = 1e-12
    maxiter: int = 400


class PackedTowerSolver:
    """Solver for packed tower design"""
    
    def __init__(self, spec: PackedTowerSpec):
        self.spec = spec
        self._validate()
    
    def _validate(self):
        """Validate inputs"""
        check_positive("G", self.spec.G)
        check_positive("L", self.spec.L)
        check_in_closed_01("y_in", self.spec.y_in)
        check_in_closed_01("x_in", self.spec.x_in)
        
        # Check specification completeness
        n_outlets = sum([
            self.spec.y_out is not None,
            self.spec.x_out is not None
        ])
        
        if n_outlets == 0:
            raise InputError("Must specify at least one outlet composition")
        
        if n_outlets == 1:
            # Calculate the other from mass balance
            self._calculate_missing_outlet()
    
    def _calculate_missing_outlet(self):
        """Calculate missing outlet from mass balance"""
        if self.spec.y_out is not None and self.spec.x_out is None:
            # Calculate x_out
            solute_removed = self.spec.G * (self.spec.y_in - self.spec.y_out)
            self.x_out_calc = self.spec.x_in + solute_removed / self.spec.L
            self.y_out_calc = self.spec.y_out
        elif self.spec.x_out is not None and self.spec.y_out is None:
            # Calculate y_out
            solute_removed = self.spec.L * (self.spec.x_out - self.spec.x_in)
            self.y_out_calc = self.spec.y_in - solute_removed / self.spec.G
            self.x_out_calc = self.spec.x_out
        else:
            self.x_out_calc = self.spec.x_out
            self.y_out_calc = self.spec.y_out
    
    def _driving_force(self, y: float, x: float) -> float:
        """Calculate driving force for mass transfer"""
        y_eq = self.spec.eq.y_of_x(x)
        return y - y_eq
    
    def calculate_NOG(self) -> float:
        """
        Calculate number of transfer units (gas phase):
        NOG = ∫ dy/(y - y*)
        """
        from bank.core.numerical import integrate_trapezoid
        
        y_start = self.spec.y_in
        y_end = self.y_out_calc
        
        if abs(y_start - y_end) < 1e-12:
            return 0.0
        
        # Need to relate x to y along operating line
        def integrand(y: float):
            # Find x at this y from operating line
            x = self.spec.x_in + (self.spec.L / self.spec.G) * (self.spec.y_in - y)
            y_eq = self.spec.eq.y_of_x(x)
            return 1.0 / (y - y_eq)
        
        return integrate_trapezoid(integrand, y_end, y_start, n=1000)
    
    def calculate_HOG(self) -> float:
        """
        Calculate height of a transfer unit.
        Uses simplified Onda et al. correlation for demonstration.
        For production use, replace with validated correlation.
        """
        if self.spec.HOG is not None:
            return self.spec.HOG
        
        # Empirical correlation (simplified Onda et al.)
        if all(v is not None for v in [self.spec.rho_G, self.spec.mu_G, self.spec.D_G]):
            # Gas phase HTU correlation
            Sc_G = self.spec.mu_G / (self.spec.rho_G * self.spec.D_G)
            Re_G = 4 * self.spec.G / (math.pi * 0.5 * self.spec.mu_G)  # Simplified
            
            HOG = 0.011 * Re_G**0.3 * Sc_G**0.5
            return HOG
        else:
            # Default value if no properties
            return 0.5  # meters
    
    def calculate_HETP(self) -> float:
        """
        Calculate HETP from HOG if needed.
        HETP = HOG * ln(λ)/(λ-1) where λ = mG/L
        """
        if self.spec.HETP is not None:
            return self.spec.HETP
        
        HOG = self.calculate_HOG()
        
        # Average m from equilibrium
        x_avg = (self.spec.x_in + self.x_out_calc) / 2
        m = self.spec.eq.K_value(x_avg)
        lambda_factor = m * self.spec.G / self.spec.L
        
        if abs(lambda_factor - 1.0) < 1e-12:
            return HOG
        
        HETP = HOG * math.log(lambda_factor) / (lambda_factor - 1)
        return HETP
    def calculate_NOL_from_integral(self) -> float:
        """
        Calculate number of liquid-phase transfer units (NOL) by integration.
    
        NOL = ∫ dx / (x* - x)  from x_out to x_in
    
         where:
             x* is the liquid composition in equilibrium with the bulk gas
             x is the bulk liquid composition
    
        The relationship between x and y along the column comes from the operating line:
             y = y_in + (L/G)(x - x_in)  (for absorption)
    
        And x* is found from equilibrium: x* = x_of_y(y)
    
        Returns:
            Number of liquid-phase transfer units (NOL)
    
        Raises:
            NotImplementedError: If inverse equilibrium function not available
            ValueError: If required data is missing
        """
        from bank.core.numerical import integrate_trapezoid, bisection
        from bank.core.validation import check_in_closed_01
    
        # Check that we have all necessary data
        if not all([
            self.spec.x_in is not None,
            self.spec.x_out is not None,
            self.spec.y_in is not None,
            self.spec.L is not None,
            self.spec.G is not None,
            self.spec.y_eq_func is not None
        ]):
            raise ValueError(
                "Missing data for NOL integration. Need: x_in, x_out, y_in, L, G, and y_eq_func"
            )
    
        x_in = self.spec.x_in
        x_out = self.spec.x_out
        y_in = self.spec.y_in
        L = self.spec.L
        G = self.spec.G
    
        # Determine integration direction
        if x_out > x_in:
            # Stripping: x decreases along column
            x_lower = x_in
            x_upper = x_out
        else:
            # Absorption: x increases along column
            x_lower = x_out
            x_upper = x_in
    
        def equilibrium_x_at_y(y: float) -> float:
            """
            Find x* that is in equilibrium with given y.
            This requires solving y = y_eq_func(x) for x.
            """
            if hasattr(self.spec.y_eq_func, 'x_of_y'):
                # If the equilibrium model provides direct inverse
                return self.spec.y_eq_func.x_of_y(y)
            else:
                # Need to solve numerically
                def f(x):
                    return self.spec.y_eq_func(x) - y
            
                try:
                    return bisection(f, 0.0, 1.0, tol=self.spec.tol)
                except:
                    raise NotImplementedError(
                        "Cannot calculate NOL: equilibrium model does not provide x_of_y(y) "
                        "and numerical inversion failed. Use NOG method instead."
                    )
    
        def integrand(x: float) -> float:
            """
            Integrand for NOL: 1/(x* - x)
            where x* is liquid composition in equilibrium with bulk gas at this point.
            """
            # Find y at this x from operating line
            # For absorption: y = y_in + (L/G)(x - x_in)
            y = y_in + (L / G) * (x - x_in)
        
            # Clip y to valid range
            y = max(0.0, min(1.0, y))
        
            # Find equilibrium liquid composition x*
            x_star = equilibrium_x_at_y(y)
        
            # Calculate driving force
            driving_force = x_star - x
        
            # Check for zero or negative driving force
            if driving_force <= 1e-12:
                # At the top of column, driving force should approach zero
                # Return a large number to make integral finite
                return 1e12
        
            return 1.0 / driving_force
    
        # Perform integration
        try:
            NOL = integrate_trapezoid(
                integrand, 
                x_lower, 
                x_upper, 
                n=self.spec.n_int if hasattr(self.spec, 'n_int') else 1000
            )
        except Exception as e:
            raise RuntimeError(f"Integration failed for NOL calculation: {e}")
    
        return NOL


def calculate_HOL_from_correlation(self) -> float:
    """
    Calculate height of a liquid-phase transfer unit (HOL) using correlation.
    
    Returns:
        Height of liquid transfer unit (m)
    """
    if self.spec.HOL is not None:
        return self.spec.HOL
    
    # Default correlation (simplified)
    # In practice, use Onda et al. or other validated correlation
    return 0.3  # meters, placeholder


def calculate_NOL_from_HTU(self) -> float:
    """
    Calculate NOL from column height and HOL.
    
    Returns:
        Number of liquid-phase transfer units
    """
    if self.spec.H is not None and self.spec.HOL is not None:
        return self.spec.H / self.spec.HOL
    else:
        return self.calculate_NOL_from_integral()


def calculate_HTU_from_NTU(self) -> Dict[str, float]:
    """
    Calculate HTU values from NTU values and column height.
    
    Returns:
        Dictionary with HOG and HOL if calculable
    """
    result = {}
    
    if self.spec.H is not None:
        if hasattr(self, '_NOG') or self.spec.NOG is not None:
            NOG = self.spec.NOG if self.spec.NOG is not None else self._NOG
            result["HOG"] = self.spec.H / NOG
        
        if hasattr(self, '_NOL') or self.spec.NOL is not None:
            NOL = self.spec.NOL if self.spec.NOL is not None else self._NOL
            result["HOL"] = self.spec.H / NOL
    
    return result


def relate_NOG_and_NOL(self) -> Dict[str, float]:
    """
    Relate gas-phase and liquid-phase transfer units.
    
    For dilute systems: NOL = (L/G) * NOG * (average slope factor)
    
    Returns:
        Dictionary with relationship factors
    """
    if not all([self.spec.L, self.spec.G, self.spec.y_eq_func]):
        return {}
    
    # Average slope of equilibrium line
    x_avg = (self.spec.x_in + self.spec.x_out) / 2
    try:
        m = self.spec.y_eq_func.K_value(x_avg) if hasattr(self.spec.y_eq_func, 'K_value') else 1.0
    except:
        m = 1.0
    
    lambda_factor = m * self.spec.G / self.spec.L
    
    return {
        "lambda": lambda_factor,
        "NOL_to_NOG_ratio": 1.0 / lambda_factor if lambda_factor > 0 else float('inf'),
        "NOG_to_NOL_ratio": lambda_factor,
    }

def solve(self) -> Dict[str, Any]:
    """Main solving method for transfer units calculation"""
    import math
    from bank.core.validation import ChemEngError
    
    result = {}
    
    # Calculate mass balance if outlets not both specified
    if self.spec.x_out is None or self.spec.y_out is None:
        self._calculate_missing_outlet()
        x_out = self.x_out_calc
        y_out = self.y_out_calc
    else:
        x_out = self.spec.x_out
        y_out = self.spec.y_out
    
    # Store for later use
    self.x_out = x_out
    self.y_out = y_out
    
    # Calculate solute mass balance
    solute_in = self.spec.L * self.spec.x_in + self.spec.G * self.spec.y_in
    solute_out = self.spec.L * x_out + self.spec.G * y_out
    mass_balance_error = solute_in - solute_out
    
    # Calculate missing values for gas phase
    if self.spec.NOG is None and self.spec.H is not None and self.spec.HOG is not None:
        result["NOG"] = self.spec.H / self.spec.HOG
        self._NOG = result["NOG"]
    elif self.spec.NOG is None and self.spec.y_eq_func is not None:
        result["NOG"] = self.calculate_NOG_from_integral()
        self._NOG = result["NOG"]
    else:
        result["NOG"] = self.spec.NOG
        self._NOG = self.spec.NOG
    
    # Calculate missing values for liquid phase
    if self.spec.NOL is None and self.spec.H is not None and self.spec.HOL is not None:
        result["NOL"] = self.spec.H / self.spec.HOL
        self._NOL = result["NOL"]
    elif self.spec.NOL is None and self.spec.y_eq_func is not None:
        try:
            result["NOL"] = self.calculate_NOL_from_integral()
            self._NOL = result["NOL"]
        except (NotImplementedError, ValueError) as e:
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
    elif self.spec.HOL is None:
        result["HOL"] = self.calculate_HOL_from_correlation()
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
    result["htu_from_ntu"] = self.calculate_HTU_from_NTU()
    
    # Calculate HETP if possible
    if result["HOG"] is not None and result["NOG"] is not None:
        lambda_factor = result["relationships"].get("lambda", 1.0)
        if abs(lambda_factor - 1.0) > 1e-12:
            result["HETP"] = result["HOG"] * math.log(lambda_factor) / (lambda_factor - 1)
        else:
            result["HETP"] = result["HOG"]
    else:
        result["HETP"] = None
    
    # Calculate theoretical stages from HETP
    if result["HETP"] is not None and result["HETP"] > 0 and result["H"] is not None:
        N_theoretical = result["H"] / result["HETP"]
        N_actual = math.ceil(N_theoretical - 1e-12)
    else:
        N_theoretical = None
        N_actual = None
    
    # Final return dictionary with consistent structure
    return {
        "inputs": {
            "x_in": self.spec.x_in,
            "x_out": x_out,
            "y_in": self.spec.y_in,
            "y_out": y_out,
            "L": self.spec.L,
            "G": self.spec.G,
            "H_specified": self.spec.H,
            "HOG_specified": self.spec.HOG,
            "HOL_specified": self.spec.HOL,
            "NOG_specified": self.spec.NOG,
            "NOL_specified": self.spec.NOL,
        },
        "transfer_units": {
            "NOG": result["NOG"],
            "NOL": result.get("NOL"),
            "HOG": result["HOG"],
            "HOL": result.get("HOL"),
            "H": result["H"],
            "HETP": result["HETP"],
            "NOG_warning": result.get("NOG_warning"),
            "NOL_warning": result.get("NOL_warning"),
        },
        "relationships": result["relationships"],
        "stages": {
            "N_theoretical": N_theoretical,
            "N_actual": N_actual,
        },
        "verification": {
            "mass_balance_error": mass_balance_error,
            "mass_balance_relative": mass_balance_error / (solute_in + 1e-15),
        }
    }