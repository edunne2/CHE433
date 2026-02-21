"""Vapor pressure table for steam distillation"""

from typing import Sequence, List
from bank.core.validation import check_positive, ChemEngError
from bank.core.numerical import linear_interpolate, log_interpolate


class VaporPressureTable:
    """Tabulated vapor pressure data for a pure component"""
    
    def __init__(
        self,
        T: Sequence[float],
        Psat: Sequence[float],
        log_interp: bool = True,
        extrapolate: bool = True
    ):
        if len(T) != len(Psat):
            raise ValueError(f"T and Psat must have same length")
        if len(T) < 2:
            raise ValueError(f"At least 2 points required")
        
        pairs = sorted((float(t), float(p)) for t, p in zip(T, Psat))
        self.T = [p[0] for p in pairs]
        self.Psat = [p[1] for p in pairs]
        
        for i, p in enumerate(self.Psat):
            if p <= 0:
                raise ChemEngError(f"Psat[{i}] must be > 0")
        
        self.log_interp = log_interp
        self.extrapolate = extrapolate
    
    def Psat_of_T(self, T: float) -> float:
        """Calculate vapor pressure at given temperature"""
        if self.log_interp:
            return log_interpolate(T, self.T, self.Psat, self.extrapolate)
        else:
            return linear_interpolate(T, self.T, self.Psat, self.extrapolate)
    
    def T_of_Psat(self, P: float) -> float:
        """Calculate temperature at given vapor pressure"""
        check_positive("P", P)
        
        inv_pairs = sorted((p, t) for t, p in zip(self.T, self.Psat))
        T_inv = [p[1] for p in inv_pairs]
        P_inv = [p[0] for p in inv_pairs]
        
        if self.log_interp:
            return log_interpolate(P, P_inv, T_inv, self.extrapolate)
        else:
            return linear_interpolate(P, P_inv, T_inv, self.extrapolate)
    
    def check_consistency(self) -> bool:
        """Check that vapor pressure is increasing with temperature"""
        for i in range(1, len(self.T)):
            if self.Psat[i] <= self.Psat[i-1]:
                return False
        return True