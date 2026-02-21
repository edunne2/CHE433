"""Liquid-Liquid equilibrium relations - Eqs. 27.1-1 to 27.1-3"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from bank.core.validation import (
    check_positive, check_in_closed_01, ChemEngError, InputError
)


@dataclass
class PhaseEquilibrium:
    """
    Liquid-liquid equilibrium data for ternary systems.
    
    Eqs. 27.1-1 to 27.1-3:
    F = C - P + 2 = 3 - 2 + 2 = 3                    (27.1-1)
    x_A + x_B + x_C = 1.0                            (27.1-2)
    y_A + y_B + y_C = 1.0                             (27.1-2)
    x_B = 1.0 - x_A - x_C                             (27.1-3)
    y_B = 1.0 - y_A - y_C                             (27.1-3)
    """
    
    def __init__(
        self,
        x_A_points: List[float],
        x_C_points: List[float],
        y_A_points: List[float],
        y_C_points: List[float],
        tie_line_pairs: Optional[List[Tuple[int, int]]] = None
    ):
        """
        Initialize with equilibrium data.
        
        Args:
            x_A_points: Raffinate phase A compositions (x_A)
            x_C_points: Raffinate phase C compositions (x_C)
            y_A_points: Extract phase A compositions (y_A)
            y_C_points: Extract phase C compositions (y_C)
            tie_line_pairs: List of (raffinate_idx, extract_idx) pairs for tie lines
        """
        if len(x_A_points) != len(x_C_points):
            raise InputError("x_A and x_C must have same length")
        if len(y_A_points) != len(y_C_points):
            raise InputError("y_A and y_C must have same length")
        
        self.x_A = np.array(x_A_points)
        self.x_C = np.array(x_C_points)
        self.x_B = 1.0 - self.x_A - self.x_C
        
        self.y_A = np.array(y_A_points)
        self.y_C = np.array(y_C_points)
        self.y_B = 1.0 - self.y_A - self.y_C
        
        # Validate compositions
        for i, (xA, xC, xB) in enumerate(zip(self.x_A, self.x_C, self.x_B)):
            if not (0 <= xA <= 1 and 0 <= xC <= 1 and 0 <= xB <= 1):
                raise InputError(f"Raffinate composition at index {i} invalid")
        
        for i, (yA, yC, yB) in enumerate(zip(self.y_A, self.y_C, self.y_B)):
            if not (0 <= yA <= 1 and 0 <= yC <= 1 and 0 <= yB <= 1):
                raise InputError(f"Extract composition at index {i} invalid")
        
        self.tie_lines = tie_line_pairs or list(zip(range(len(x_A_points)), range(len(y_A_points))))
    
    def get_tie_line(self, idx: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Get tie line by index"""
        if idx >= len(self.tie_lines):
            raise IndexError(f"Only {len(self.tie_lines)} tie lines available")
        
        r_idx, e_idx = self.tie_lines[idx]
        
        raffinate = {
            'x_A': float(self.x_A[r_idx]),
            'x_B': float(self.x_B[r_idx]),
            'x_C': float(self.x_C[r_idx]),
        }
        extract = {
            'y_A': float(self.y_A[e_idx]),
            'y_B': float(self.y_B[e_idx]),
            'y_C': float(self.y_C[e_idx]),
        }
        return raffinate, extract
    
    def find_tie_line_through_M(self, x_AM: float, x_CM: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Find the tie line that passes through point M (x_AM, x_CM).
        
        This uses linear interpolation between adjacent tie lines.
        """
        # Find raffinate compositions bracketing x_AM
        idx = None
        for i in range(len(self.x_A) - 1):
            if self.x_A[i] >= x_AM >= self.x_A[i + 1] or self.x_A[i] <= x_AM <= self.x_A[i + 1]:
                idx = i
                break
        
        if idx is None:
            raise ChemEngError(f"No tie line found through M ({x_AM:.4f}, {x_CM:.4f})")
        
        # Linear interpolation
        r1, e1 = self.get_tie_line(idx)
        r2, e2 = self.get_tie_line(idx + 1)
        
        # Interpolate based on x_A
        if abs(r2['x_A'] - r1['x_A']) < 1e-12:
            w = 0.5
        else:
            w = (x_AM - r1['x_A']) / (r2['x_A'] - r1['x_A'])
        
        raffinate = {
            'x_A': r1['x_A'] + w * (r2['x_A'] - r1['x_A']),
            'x_C': r1['x_C'] + w * (r2['x_C'] - r1['x_C']),
        }
        raffinate['x_B'] = 1.0 - raffinate['x_A'] - raffinate['x_C']
        
        extract = {
            'y_A': e1['y_A'] + w * (e2['y_A'] - e1['y_A']),
            'y_C': e1['y_C'] + w * (e2['y_C'] - e1['y_C']),
        }
        extract['y_B'] = 1.0 - extract['y_A'] - extract['y_C']
        
        return raffinate, extract


class TriangularDiagram:
    """Helper class for triangular coordinate calculations"""
    
    @staticmethod
    def composition_from_point(x: float, y: float) -> Dict[str, float]:
        """
        Convert triangular coordinates (x, y) to mass fractions.
        For equilateral triangle with side length 1.
        """
        xA = 1 - x - y/np.sqrt(3)
        xB = x - y/np.sqrt(3)
        xC = 2*y/np.sqrt(3)
        return {'xA': xA, 'xB': xB, 'xC': xC}
    
    @staticmethod
    def point_from_composition(xA: float, xB: float, xC: float) -> Tuple[float, float]:
        """Convert mass fractions to triangular coordinates"""
        x = (xB + xA/2)
        y = (np.sqrt(3)/2) * xA
        return x, y


class RectangularDiagram:
    """Helper class for rectangular coordinate calculations (Fig. 27.1-3)"""
    
    @staticmethod
    def raffinate_composition(xA: float, xC: float) -> Dict[str, float]:
        """Get complete raffinate composition from xA and xC"""
        xB = 1.0 - xA - xC
        return {'xA': xA, 'xB': xB, 'xC': xC}
    
    @staticmethod
    def extract_composition(yA: float, yC: float) -> Dict[str, float]:
        """Get complete extract composition from yA and yC"""
        yB = 1.0 - yA - yC
        return {'yA': yA, 'yB': yB, 'yC': yC}


def tie_line_compositions(
    eq: PhaseEquilibrium,
    x_AM: float,
    x_CM: float
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Convenience function to find tie line compositions through point M.
    
    Args:
        eq: PhaseEquilibrium object
        x_AM: Mass fraction of A in mixture M
        x_CM: Mass fraction of C in mixture M
    
    Returns:
        (raffinate_composition, extract_composition)
    """
    return eq.find_tie_line_through_M(x_AM, x_CM)