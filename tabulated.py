"""Tabulated equilibrium data"""

from typing import List, Optional
from bank.core.validation import check_in_closed_01
from bank.core.numerical import linear_interpolate
from .base import EquilibriumModel


class TabulatedEquilibrium(EquilibriumModel):
    """Equilibrium from tabulated (x,y) data - implements Eq. 22.1-30"""
    
    def __init__(
        self,
        x_points: List[float],
        y_points: List[float],
        extrapolate: bool = False
    ):
        if len(x_points) != len(y_points):
            raise ValueError("x_points and y_points must have same length")
        if len(x_points) < 2:
            raise ValueError("At least 2 points required")
        
        pairs = sorted(zip(x_points, y_points))
        self.x_points = [p[0] for p in pairs]
        self.y_points = [p[1] for p in pairs]
        self.extrapolate = extrapolate
        
        inv_pairs = sorted((y, x) for x, y in pairs)
        self.y_inv = [p[0] for p in inv_pairs]
        self.x_inv = [p[1] for p in inv_pairs]
    
    def y_of_x(self, x: float) -> float:
        check_in_closed_01("x", x)
        return linear_interpolate(x, self.x_points, self.y_points, self.extrapolate)
    
    def x_of_y(self, y: float) -> float:
        check_in_closed_01("y", y)
        return linear_interpolate(y, self.y_inv, self.x_inv, self.extrapolate)  