"""Single-stage equilibrium extraction - Eqs. 27.2-1 to 27.2-11"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import math

from bank.core.validation import (
    check_positive, check_in_closed_01, ChemEngError, InputError
)
from .equilibrium import PhaseEquilibrium


def lever_arm_rule(
    L: float,
    V: float,
    M: float,
    dist_LM: float,
    dist_VM: float,
    dist_LV: float,
    mode: str = "L_from_VM_LM"
) -> float:
    """
    Lever-arm rule - Eqs. 27.2-7 and 27.2-8
    
    L/V = length_VM / length_LM                              (27.2-7)
    L/M = length_VM / length_LV                              (27.2-8)
    
    Args:
        L: Mass of L phase (kg)
        V: Mass of V phase (kg)
        M: Total mass (kg)
        dist_LM: Distance between points L and M
        dist_VM: Distance between points V and M
        dist_LV: Distance between points L and V
        mode: Which ratio to calculate:
            - "L_from_VM_LM": L/V = VM/LM
            - "L_from_VM_LV": L/M = VM/LV
            - "V_from_LM_LV": V/M = LM/LV
    
    Returns:
        Calculated ratio or mass
    """
    if mode == "L_from_VM_LM":
        return V * (dist_VM / dist_LM)
    elif mode == "L_from_VM_LV":
        return M * (dist_VM / dist_LV)
    elif mode == "V_from_LM_LV":
        return M * (dist_LM / dist_LV)
    else:
        raise ValueError(f"Unknown mode: {mode}")


@dataclass
class SingleStageSpec:
    """Specification for single-stage equilibrium extraction"""
    L0: float           # Raffinate phase inlet (kg)
    xA0: float          # Raffinate phase A composition
    xC0: float          # Raffinate phase C composition
    V2: float           # Extract phase inlet (kg)
    yA2: float          # Extract phase A composition
    yC2: float          # Extract phase C composition
    eq: PhaseEquilibrium  # Equilibrium data
    
    @property
    def xB0(self) -> float:
        return 1.0 - self.xA0 - self.xC0
    
    @property
    def yB2(self) -> float:
        return 1.0 - self.yA2 - self.yC2


class SingleStageExtractor:
    """
    Single-stage equilibrium extraction solver.
    
    Implements Eqs. 27.2-1 to 27.2-11.
    """
    
    def __init__(self, spec: SingleStageSpec):
        self.spec = spec
        self._validate()
        self._calculate_M()
    
    def _validate(self):
        """Validate inputs"""
        check_positive("L0", self.spec.L0)
        check_positive("V2", self.spec.V2)
        check_in_closed_01("xA0", self.spec.xA0)
        check_in_closed_01("xC0", self.spec.xC0)
        check_in_closed_01("yA2", self.spec.yA2)
        check_in_closed_01("yC2", self.spec.yC2)
        
        # Validate composition sums
        if abs(self.spec.xA0 + self.spec.xC0 + self.spec.xB0 - 1.0) > 1e-10:
            raise InputError(f"Raffinate inlet composition does not sum to 1: "
                           f"{self.spec.xA0:.4f} + {self.spec.xC0:.4f} + {self.spec.xB0:.4f} = "
                           f"{self.spec.xA0 + self.spec.xC0 + self.spec.xB0:.4f}")
        
        if abs(self.spec.yA2 + self.spec.yC2 + self.spec.yB2 - 1.0) > 1e-10:
            raise InputError(f"Extract inlet composition does not sum to 1")
    
    def _calculate_M(self):
        """Calculate mixture M composition - Eqs. 27.2-9 to 27.2-11"""
        self.M = self.spec.L0 + self.spec.V2
        
        # Component A balance - Eq. 27.2-10
        self.x_AM = (self.spec.L0 * self.spec.xA0 + self.spec.V2 * self.spec.yA2) / self.M
        
        # Component C balance - Eq. 27.2-11
        self.x_CM = (self.spec.L0 * self.spec.xC0 + self.spec.V2 * self.spec.yC2) / self.M
        
        # Component B by difference
        self.x_BM = 1.0 - self.x_AM - self.x_CM
    
    def find_equilibrium_phases(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Find equilibrium phases L1 and V1 using tie lines.
        
        Returns:
            (raffinate_composition, extract_composition)
        """
        return self.spec.eq.find_tie_line_through_M(self.x_AM, self.x_CM)
    
    def solve_mass_balance(
        self,
        raffinate: Dict[str, float],
        extract: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Solve for L1 and V1 masses using mass balance.
        
        Args:
            raffinate: Composition of raffinate phase L1
            extract: Composition of extract phase V1
        
        Returns:
            Dictionary with L1, V1, and verification
        """
        # Solve using component A balance - Eq. 27.2-10
        # L1 * xA1 + V1 * yA1 = M * x_AM
        # with L1 + V1 = M
        
        denom = raffinate['x_A'] - extract['y_A']
        if abs(denom) < 1e-12:
            # Try using component C instead
            denom = raffinate['x_C'] - extract['y_C']
            if abs(denom) < 1e-12:
                raise ChemEngError("Cannot solve mass balance - compositions too close")
            
            L1 = self.M * (self.x_CM - extract['y_C']) / denom
        else:
            L1 = self.M * (self.x_AM - extract['y_A']) / denom
        
        V1 = self.M - L1
        
        # Verify with component C balance
        check_C = abs(L1 * raffinate['x_C'] + V1 * extract['y_C'] - self.M * self.x_CM)
        
        return {
            "L1": L1,
            "V1": V1,
            "raffinate": raffinate,
            "extract": extract,
            "verification": {
                "A_balance": abs(L1 * raffinate['x_A'] + V1 * extract['y_A'] - self.M * self.x_AM),
                "C_balance": check_C,
                "total_balance": abs(L1 + V1 - self.M),
            }
        }
    
    def solve(self) -> Dict[str, Any]:
        """Complete single-stage extraction solution"""
        # Find equilibrium phases
        raffinate, extract = self.find_equilibrium_phases()
        
        # Solve mass balance
        result = self.solve_mass_balance(raffinate, extract)
        
        # Calculate L/V ratio using lever-arm rule (approximate)
        # This would require distances on the actual plot
        
        return {
            "inputs": {
                "L0": self.spec.L0,
                "V2": self.spec.V2,
                "xA0": self.spec.xA0,
                "xC0": self.spec.xC0,
                "yA2": self.spec.yA2,
                "yC2": self.spec.yC2,
            },
            "mixture": {
                "M": self.M,
                "x_AM": self.x_AM,
                "x_CM": self.x_CM,
                "x_BM": self.x_BM,
            },
            "results": result,
        }