# file: stoich_base.py
"""
General single-reaction stoichiometry engine.

Handles:
- Up to 4 reactants
- Up to 6 products
- Any species names
- Rate conversions between species
- Time unit conversions (s, min, hr)

Conventions:
- Reactant coefficients entered as positive numbers.
- Product coefficients entered as positive numbers.
- Internally:
      reactants -> negative signed stoich
      products  -> positive signed stoich

Core relation:
      r_i = ν_i * R

Where:
      r_i  = net rate of production of species i
      ν_i  = signed stoichiometric coefficient
      R    = extent rate of reaction
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


# ----------------------------
# Time Unit Conversion
# ----------------------------

_TIME_TO_SECONDS = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
}


def convert_time_in_rate(value: float, from_time: str, to_time: str) -> float:
    """
    Convert rate time unit.
    Example: mol/L/min -> mol/L/s (divide by 60)
    """
    f = from_time.strip().lower()
    t = to_time.strip().lower()

    if f not in _TIME_TO_SECONDS or t not in _TIME_TO_SECONDS:
        raise ValueError("Unsupported time unit.")

    sec_from = _TIME_TO_SECONDS[f]
    sec_to = _TIME_TO_SECONDS[t]

    # value is "per from_time"
    # convert to "per to_time": value * (sec_to / sec_from)
    return value * (sec_to / sec_from)


# ----------------------------
# Stoichiometry Engine
# ----------------------------

@dataclass(frozen=True)
class ReactionStoich:
    reactants: Dict[str, float]
    products: Dict[str, float]

    def nu_signed(self) -> Dict[str, float]:
        """
        Return signed stoichiometric coefficients.
        """
        nu = {}

        for sp, coeff in self.reactants.items():
            if coeff <= 0:
                raise ValueError("Reactant coefficients must be > 0.")
            nu[sp] = -float(coeff)

        for sp, coeff in self.products.items():
            if coeff <= 0:
                raise ValueError("Product coefficients must be > 0.")
            if sp in nu:
                raise ValueError(f"Species '{sp}' appears twice.")
            nu[sp] = float(coeff)

        return nu

    def extent_rate_from_species_rate(self, species: str, r_species: float) -> float:
        """
        Compute extent rate R from r_species.
        """
        nu = self.nu_signed()
        if species not in nu:
            raise KeyError(f"Species '{species}' not found.")
        return r_species / nu[species]

    def species_rate_from_extent(self, species: str, R: float) -> float:
        """
        Compute r_species from extent rate R.
        """
        nu = self.nu_signed()
        if species not in nu:
            raise KeyError(f"Species '{species}' not found.")
        return nu[species] * R

    def species_rate_from_species(self, given_species: str, r_given: float, target_species: str) -> float:
        """
        Convert rate of one species into rate of another species.
        """
        R = self.extent_rate_from_species_rate(given_species, r_given)
        return self.species_rate_from_extent(target_species, R)


# ----------------------------
# Sign Convention Helpers
# ----------------------------

def net_rate_from_destruction(destruction_rate: float) -> float:
    """
    Convert positive destruction rate (-r_i) into net rate r_i.
    """
    return -float(destruction_rate)


def destruction_from_net(r_species: float) -> float:
    """
    Convert net rate into positive destruction rate (-r_i).
    """
    return -float(r_species)
