# file: bank/limiting_reactant.py
"""
Limiting Reactant Engine (single reaction)

Works for:
- Any number of reactants
- Any stoichiometric coefficients
- Amount basis: concentration, moles, molar flow (consistent units required)

Method:
For each reactant i:
    max_extent_i = amount_i / stoich_i

The smallest max_extent determines the limiting reactant.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class LimitingReactantSolver:
    reactant_stoich: Dict[str, float]   # positive stoich numbers
    reactant_amounts: Dict[str, float]  # matching units (mol, mol/L, mol/time, etc.)

    def _validate(self):
        for sp, coeff in self.reactant_stoich.items():
            if coeff <= 0:
                raise ValueError("Stoichiometric coefficients must be positive.")
            if sp not in self.reactant_amounts:
                raise KeyError(f"Missing amount for reactant '{sp}'.")
            if self.reactant_amounts[sp] < 0:
                raise ValueError("Reactant amounts must be >= 0.")

    def max_extent(self) -> Dict[str, float]:
        """
        Returns max reaction extent allowed by each reactant.
        """
        self._validate()
        extent_limits = {}
        for sp, coeff in self.reactant_stoich.items():
            amount = self.reactant_amounts[sp]
            extent_limits[sp] = amount / coeff
        return extent_limits

    def limiting_reactant(self) -> Tuple[str, float]:
        """
        Returns (limiting_species, max_extent_value)
        """
        extent_limits = self.max_extent()
        limiting_species = min(extent_limits, key=extent_limits.get)
        return limiting_species, extent_limits[limiting_species]

    def excess_ratios(self) -> Dict[str, float]:
        """
        Returns how many times each reactant exceeds the limiting requirement.
        Ratio = extent_i / extent_limiting
        """
        limiting_species, limiting_extent = self.limiting_reactant()
        extent_limits = self.max_extent()
        return {sp: extent_limits[sp] / limiting_extent for sp in extent_limits}
