# bank/separations/equilibrium/vapor_pressure.py
"""Vapor pressure table for steam distillation and other applications"""

from typing import Sequence, List, Optional
import math

from bank.core.validation import check_positive, ChemEngError
from bank.core.numerical import linear_interpolate, log_interpolate


class VaporPressureTable:
    """
    Tabulated vapor pressure data for a pure component.
    
    Provides Psat(T) interpolation with options for linear or log interpolation.
    Can extrapolate outside the table range using the nearest segment.
    """
    
    def __init__(
        self,
        T: Sequence[float],
        Psat: Sequence[float],
        log_interp: bool = True,
        extrapolate: bool = True,
        units_T: str = "°C",
        units_P: str = "kPa"
    ):
        """
        Initialize vapor pressure table.
        
        Args:
            T: Temperature points (must be increasing)
            Psat: Vapor pressure points at corresponding temperatures
            log_interp: If True, interpolate ln(Psat) linearly with T
                       If False, interpolate Psat linearly with T
            extrapolate: If True, extrapolate outside table range
            units_T: Temperature units (for reference only)
            units_P: Pressure units (for reference only)
        """
        if len(T) != len(Psat):
            raise ValueError(f"T and Psat must have same length, got {len(T)} and {len(Psat)}")
        if len(T) < 2:
            raise ValueError(f"At least 2 points required, got {len(T)}")
        
        # Sort by temperature
        pairs = sorted((float(t), float(p)) for t, p in zip(T, Psat))
        self.T = [p[0] for p in pairs]
        self.Psat = [p[1] for p in pairs]
        
        # Validate pressures are positive
        for i, p in enumerate(self.Psat):
            if p <= 0:
                raise ChemEngError(f"Psat[{i}] must be > 0, got {p}")
        
        self.log_interp = log_interp
        self.extrapolate = extrapolate
        self.units_T = units_T
        self.units_P = units_P
    
    def Psat_of_T(self, T: float) -> float:
        """
        Calculate vapor pressure at given temperature.
        
        Args:
            T: Temperature
        
        Returns:
            Vapor pressure at T
        """
        if self.log_interp:
            return log_interpolate(T, self.T, self.Psat, self.extrapolate)
        else:
            return linear_interpolate(T, self.T, self.Psat, self.extrapolate)
    
    def T_of_Psat(self, P: float) -> float:
        """
        Calculate temperature at given vapor pressure.
        
        Args:
            P: Vapor pressure
        
        Returns:
            Temperature at which Psat = P
        """
        check_positive("P", P)
        
        # Create inverse table (sort by Psat)
        if self.log_interp:
            # For log interpolation, we need to invert carefully
            # This is approximate - for precise inversion, use numerical method
            inv_pairs = sorted((p, t) for t, p in zip(self.T, self.Psat))
            T_inv = [p[1] for p in inv_pairs]
            P_inv = [p[0] for p in inv_pairs]
            
            # Use log interpolation on inverse
            return log_interpolate(P, P_inv, T_inv, self.extrapolate)
        else:
            # Simple linear interpolation on inverse
            inv_pairs = sorted((p, t) for t, p in zip(self.T, self.Psat))
            T_inv = [p[1] for p in inv_pairs]
            P_inv = [p[0] for p in inv_pairs]
            
            return linear_interpolate(P, P_inv, T_inv, self.extrapolate)
    
    def check_consistency(self) -> bool:
        """
        Check that vapor pressure is increasing with temperature.
        
        Returns:
            True if monotonic increasing
        """
        for i in range(1, len(self.T)):
            if self.Psat[i] <= self.Psat[i-1]:
                return False
        return True
    
    def __repr__(self) -> str:
        return (f"VaporPressureTable({len(self.T)} points, "
                f"log_interp={self.log_interp}, "
                f"extrapolate={self.extrapolate})")


__all__ = ['VaporPressureTable']