"""Wetted-wall column two-film theory calculations"""

from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from bank.core.validation import check_positive, check_in_closed_01
from bank.core.numerical import bisection
from bank.separations.equilibrium import TabulatedEquilibrium


class WettedWallColumnSolver:
    """
    Solver for wetted-wall column two-film theory problems.
    
    Finds interface compositions (x_i, y_i) such that:
    1. y_i and x_i are in equilibrium (from table)
    2. Flux equality: k_y(y_b - y_i) = k_x(x_i - x_b)
    """
    
    def __init__(
        self,
        x_table: List[float],
        y_table: List[float],
        k_y: float,
        k_x: float,
        y_b: float,
        x_b: float,
        tol: float = 1e-10,
        maxiter: int = 100
    ):
        """
        Initialize solver.
        
        Args:
            x_table: Liquid composition data points
            y_table: Vapor composition data points (in equilibrium with x)
            k_y: Gas film coefficient (kgmol/s·m²·mol frac)
            k_x: Liquid film coefficient (kgmol/s·m²·mol frac)
            y_b: Bulk gas composition
            x_b: Bulk liquid composition
            tol: Tolerance for convergence
            maxiter: Maximum iterations
        """
        self.x_table = x_table
        self.y_table = y_table
        self.k_y = check_positive("k_y", k_y)
        self.k_x = check_positive("k_x", k_x)
        self.y_b = check_in_closed_01("y_b", y_b)
        self.x_b = check_in_closed_01("x_b", x_b)
        self.tol = tol
        self.maxiter = maxiter
        
        # Create equilibrium model for interpolation
        self.eq = TabulatedEquilibrium(x_table, y_table)
        
        # Storage for results
        self._results = None
    
    def _log_mean(self, a: float, b: float) -> float:
        """Calculate log mean of two values"""
        if abs(a - b) < 1e-12:
            return a
        if a <= 0 or b <= 0:
            return (a + b) / 2
        return (a - b) / np.log(a / b)
    
    def _find_interface(self) -> Tuple[float, float]:
        """
        Find interface concentrations using trial-and-error method from Example 22.1-5.
        
        The method:
        1. First assume (1-y)_iM and (1-x)_iM = 1.0
        2. Find slope = -k_x/k_y
        3. Find intersection with equilibrium curve
        4. Recalculate log means using new y_i, x_i
        5. Repeat until convergence
        """
        
        # First trial: assume (1-y)_iM = (1-x)_iM = 1.0
        slope = -self.k_x / self.k_y
        
        # Find intersection of line through P with this slope
        def line_intersection(x):
            return self.y_b + slope * (x - self.x_b) - self.eq.y_of_x(x)
        
        x_i = bisection(line_intersection, self.x_b, 1.0, tol=self.tol, maxiter=self.maxiter)
        y_i = self.eq.y_of_x(x_i)
        
        # Second trial: calculate log means
        one_minus_y_iM = self._log_mean(1 - y_i, 1 - self.y_b)
        one_minus_x_iM = self._log_mean(1 - self.x_b, 1 - x_i)
        
        # New slope with corrected terms
        slope = -(self.k_x / one_minus_x_iM) / (self.k_y / one_minus_y_iM)
        
        # Find new intersection
        def line_intersection2(x):
            return self.y_b + slope * (x - self.x_b) - self.eq.y_of_x(x)
        
        x_i = bisection(line_intersection2, self.x_b, 1.0, tol=self.tol, maxiter=self.maxiter)
        y_i = self.eq.y_of_x(x_i)
        
        # Third trial (usually sufficient)
        one_minus_y_iM = self._log_mean(1 - y_i, 1 - self.y_b)
        one_minus_x_iM = self._log_mean(1 - self.x_b, 1 - x_i)
        
        slope = -(self.k_x / one_minus_x_iM) / (self.k_y / one_minus_y_iM)
        
        def line_intersection3(x):
            return self.y_b + slope * (x - self.x_b) - self.eq.y_of_x(x)
        
        x_i = bisection(line_intersection3, self.x_b, 1.0, tol=self.tol, maxiter=self.maxiter)
        y_i = self.eq.y_of_x(x_i)
        
        return x_i, y_i
    
    def solve(self) -> Dict[str, Any]:
        """Solve complete wetted-wall column problem"""
        
        # Find interface concentrations
        x_i, y_i = self._find_interface()
        
        # Calculate equilibrium endpoints
        y_star_at_x_b = self.eq.y_of_x(self.x_b)  # y in equilibrium with bulk liquid
        x_star_at_y_b = self.eq.x_of_y(self.y_b)  # x in equilibrium with bulk gas
        
        # Calculate log means for film coefficients
        one_minus_y_iM = self._log_mean(1 - y_i, 1 - self.y_b)
        one_minus_x_iM = self._log_mean(1 - self.x_b, 1 - x_i)
        
        # Calculate log means for overall coefficients
        one_minus_y_BM = self._log_mean(1 - y_star_at_x_b, 1 - self.y_b)
        one_minus_x_BM = self._log_mean(1 - self.x_b, 1 - x_star_at_y_b)
        
        # Calculate slope m' for liquid basis (using x_star)
        if abs(x_star_at_y_b - x_i) > self.tol:
            m_prime = (self.y_b - y_i) / (x_star_at_y_b - x_i)
        else:
            m_prime = float('inf')
        
        # Calculate resistance terms for liquid basis
        # 1/[K'x/(1-x)_BM] = 1/[m' * k'y/(1-y)_iM] + 1/[k'x/(1-x)_iM]
        gas_term_liquid = 1.0 / (m_prime * self.k_y / one_minus_y_iM) if m_prime != float('inf') else 0
        liquid_term = 1.0 / (self.k_x / one_minus_x_iM)
        total_resistance_liquid = gas_term_liquid + liquid_term
        
        # Overall coefficient - liquid basis
        Kx_over_one_minus_x = 1.0 / total_resistance_liquid if total_resistance_liquid > 0 else float('inf')
        Kx_prime = Kx_over_one_minus_x * one_minus_x_BM
        
        # Calculate slope m for gas basis (using y_star)
        # m = (y_star_at_x_b - y_i)/(x_b - x_i)
        if abs(self.x_b - x_i) > self.tol:
            m_gas = (y_star_at_x_b - y_i) / (self.x_b - x_i)
        else:
            m_gas = float('inf')
        
        # Calculate resistance terms for gas basis
        # 1/[K'y/(1-y)_BM] = 1/[k'y/(1-y)_iM] + m/[k'x/(1-x)_iM]
        gas_term = 1.0 / (self.k_y / one_minus_y_iM)
        liquid_term_gas = m_gas / (self.k_x / one_minus_x_iM) if m_gas != float('inf') else 0
        total_resistance_gas = gas_term + liquid_term_gas
        
        # Overall coefficient - gas basis
        Ky_over_one_minus_y = 1.0 / total_resistance_gas if total_resistance_gas > 0 else float('inf')
        Ky_prime = Ky_over_one_minus_y * one_minus_y_BM
        
        # Flux calculations
        N_A_film = self.k_y * (self.y_b - y_i)
        N_A_from_overall_x = Kx_over_one_minus_x * (x_star_at_y_b - self.x_b)
        N_A_from_overall_y = Ky_over_one_minus_y * (self.y_b - y_star_at_x_b)
        
        # Use overall liquid basis flux as primary
        N_A = N_A_from_overall_x
        
        # Store results
        self._results = {
            "interface": {
                "x_i": x_i,
                "y_i": y_i,
            },
            "flux": {
                "N_A": N_A,
                "N_A_film": N_A_film,
                "N_A_from_overall_y": N_A_from_overall_y,
                "N_A_from_overall_x": N_A_from_overall_x,
            },
            "overall_coefficients": {
                # Liquid basis
                "Kx_prime": Kx_prime,
                "Kx_overall": Kx_over_one_minus_x,
                "x_BM": one_minus_x_BM,
                # Gas basis
                "Ky_prime": Ky_prime,
                "Ky_overall": Ky_over_one_minus_y,
                "y_BM": one_minus_y_BM,
            },
            "film_coefficients": {
                "k_y": self.k_y,
                "k_x": self.k_x,
                "one_minus_x_iM": one_minus_x_iM,
                "one_minus_y_iM": one_minus_y_iM,
            },
            "resistances": {
                "gas_term_liquid_basis": gas_term_liquid,
                "liquid_term": liquid_term,
                "total_liquid_basis": total_resistance_liquid,
                "gas_term": gas_term,
                "liquid_term_gas_basis": liquid_term_gas,
                "total_gas_basis": total_resistance_gas,
                "percent_gas": (gas_term_liquid / total_resistance_liquid) * 100 if total_resistance_liquid > 0 else 0,
            },
            "slopes": {
                "m_prime": m_prime,
                "m_gas": m_gas,
            },
            "equilibrium_endpoints": {
                "y_star_at_x_b": y_star_at_x_b,
                "x_star_at_y_b": x_star_at_y_b,
            },
            "bulk": {
                "y_b": self.y_b,
                "x_b": self.x_b,
            }
        }
        
        return self._results
    
    def print_summary(self) -> None:
        """Print a formatted summary of results"""
        if self._results is None:
            raise RuntimeError("Must call solve() before printing summary")
        
        r = self._results
        
        print("=== Interface ===")
        print(f"x_i = {r['interface']['x_i']:.6f}")
        print(f"y_i = {r['interface']['y_i']:.6f}")
        
        print("\n=== Equilibrium Endpoints ===")
        print(f"x* = {r['equilibrium_endpoints']['x_star_at_y_b']:.6f}")
        print(f"y* = {r['equilibrium_endpoints']['y_star_at_x_b']:.6f}")
        
        print("\n=== Log Means ===")
        print(f"(1-y)_iM = {r['film_coefficients']['one_minus_y_iM']:.6f}")
        print(f"(1-x)_iM = {r['film_coefficients']['one_minus_x_iM']:.6f}")
        print(f"(1-x)*M = {r['overall_coefficients']['x_BM']:.6f}")
        print(f"(1-y)*M = {r['overall_coefficients']['y_BM']:.6f}")
        
        print("\n=== Slopes ===")
        print(f"m' (liquid basis) = {r['slopes']['m_prime']:.4f}")
        print(f"m (gas basis) = {r['slopes']['m_gas']:.4f}")
        
        print("\n=== Resistances (Liquid Basis) ===")
        print(f"Gas film term = {r['resistances']['gas_term_liquid_basis']:.1f}")
        print(f"Liquid film term = {r['resistances']['liquid_term']:.1f}")
        print(f"Total = {r['resistances']['total_liquid_basis']:.1f}")
        print(f"% resistance in gas film = {r['resistances']['percent_gas']:.1f}%")
        
        print("\n=== Resistances (Gas Basis) ===")
        print(f"Gas film term = {r['resistances']['gas_term']:.1f}")
        print(f"Liquid film term = {r['resistances']['liquid_term_gas_basis']:.1f}")
        print(f"Total = {r['resistances']['total_gas_basis']:.1f}")
        
        print("\n=== Overall Coefficients ===")
        print(f"K'x = {r['overall_coefficients']['Kx_prime']:.6e}")
        print(f"Kx = {r['overall_coefficients']['Kx_overall']:.6e}")
        print(f"K'y = {r['overall_coefficients']['Ky_prime']:.6e}")
        print(f"Ky = {r['overall_coefficients']['Ky_overall']:.6e}")
        
        print("\n=== Flux ===")
        print(f"N_A (film) = {r['flux']['N_A_film']:.6e}")
        print(f"N_A (from overall x) = {r['flux']['N_A_from_overall_x']:.6e}")
        print(f"N_A (from overall y) = {r['flux']['N_A_from_overall_y']:.6e}")
    
    def plot_construction(self) -> None:
        """Plot the wetted wall column construction"""
        if self._results is None:
            raise RuntimeError("Must call solve() before plotting")
        
        r = self._results
        
        # Extract results
        x_i = r["interface"]["x_i"]
        y_i = r["interface"]["y_i"]
        y_star = r["equilibrium_endpoints"]["y_star_at_x_b"]
        x_star = r["equilibrium_endpoints"]["x_star_at_y_b"]
        
        # Create smooth equilibrium curve
        x_grid = np.linspace(0, max(self.x_table), 200)
        y_grid = np.interp(x_grid, self.x_table, self.y_table)
        
        # Slope of line through P and M
        slope_PM = -self.k_x / self.k_y
        x_line = np.linspace(min(self.x_b, x_i) - 0.05, max(self.x_b, x_i) + 0.05, 50)
        y_line_PM = self.y_b + slope_PM * (x_line - self.x_b)
        
        plt.figure(figsize=(10, 8))
        
        # Plot equilibrium curve
        plt.plot(x_grid, y_grid, 'b-', linewidth=2, label='Equilibrium curve')
        
        # Plot line through P and M
        plt.plot(x_line, y_line_PM, 'r--', linewidth=1.5, 
                label=f'Line through P and M (slope = -k_x/k_y = {-self.k_x/self.k_y:.3f})')
        
        # Plot points
        plt.plot([self.x_b], [self.y_b], 'ko', markersize=8, label=f'P (bulk)')
        plt.plot([x_i], [y_i], 'ro', markersize=8, label=f'M (interface)')
        plt.plot([self.x_b], [y_star], 'go', markersize=8, label=f'E (y*)')
        plt.plot([x_star], [self.y_b], 'mo', markersize=8, label=f'F (x*)')
        
        # Add construction lines
        plt.plot([self.x_b, x_i], [self.y_b, self.y_b], 'k:', linewidth=1)
        plt.plot([x_i, x_i], [self.y_b, y_i], 'k:', linewidth=1)
        plt.plot([self.x_b, x_star], [self.y_b, self.y_b], 'k:', linewidth=1, alpha=0.5)
        plt.plot([x_star, x_star], [self.y_b, y_star], 'k:', linewidth=1, alpha=0.5)
        
        plt.xlabel('x_A (liquid mole fraction)')
        plt.ylabel('y_A (gas mole fraction)')
        plt.title('Wetted-Wall Column: Two-Film Theory Construction')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()