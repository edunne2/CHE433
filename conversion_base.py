# file: bank/conversion_base.py
"""
Conversion utilities (single reaction, constant volume / concentration basis).

Purpose:
- Identify limiting reactant on a chosen "amount" basis (moles or concentrations)
- Convert a limiting-reactant conversion X into:
  - moles of any species
  - concentrations of any species (given constant volume)

Assumptions:
- Single reaction
- Stoichiometric coefficients provided as positive numbers on each side
- Constant reactor volume if using concentrations

Definitions:
For reaction extent xi (in moles of reaction, on a mole basis):
    N_i = N_i0 + nu_i_signed * xi
Limiting reactant L:
    X_L = (N_L0 - N_L) / N_L0
For reactant L with nu_L_signed < 0:
    N_L = N_L0 + nu_L_signed * xi
    => xi = (X_L * N_L0) / (-nu_L_signed)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Reaction:
    reactants: Dict[str, float]  # positive stoich numbers
    products: Dict[str, float]   # positive stoich numbers

    def nu_signed(self) -> Dict[str, float]:
        nu: Dict[str, float] = {}
        for sp, c in self.reactants.items():
            if c <= 0:
                raise ValueError("Reactant coefficients must be > 0.")
            nu[sp] = -float(c)
        for sp, c in self.products.items():
            if c <= 0:
                raise ValueError("Product coefficients must be > 0.")
            if sp in nu:
                raise ValueError(f"Species '{sp}' appears twice.")
            nu[sp] = float(c)
        return nu


def limiting_reactant(reactant_stoich: Dict[str, float], amounts: Dict[str, float]) -> Tuple[str, float]:
    """
    Limiting reactant based on amount/stoich. amounts can be moles, molar flows, or concentrations.
    Returns (limiting_species, extent_limit_basis_value).
    """
    for sp, coeff in reactant_stoich.items():
        if coeff <= 0:
            raise ValueError("Stoichiometric coefficients must be > 0.")
        if sp not in amounts:
            raise KeyError(f"Missing amount for reactant '{sp}'.")
        if amounts[sp] < 0:
            raise ValueError("Amounts must be >= 0.")

    extent_limits = {sp: amounts[sp] / reactant_stoich[sp] for sp in reactant_stoich}
    L = min(extent_limits, key=extent_limits.get)
    return L, extent_limits[L]


def extent_from_conversion(
    rxn: Reaction,
    limiting_species: str,
    X_lim: float,
    N0: Dict[str, float],
) -> float:
    """
    Compute extent xi (moles of reaction) from limiting reactant conversion X_lim.
    Uses mole basis (N0 in moles).
    """
    if not (0 <= X_lim <= 1):
        raise ValueError("X_lim must be between 0 and 1.")
    if limiting_species not in N0:
        raise KeyError("limiting_species missing from N0.")

    nu = rxn.nu_signed()
    if limiting_species not in nu:
        raise KeyError("limiting_species missing from reaction stoichiometry.")

    nuL = nu[limiting_species]
    if nuL >= 0:
        raise ValueError("limiting_species must be a reactant (negative signed stoich).")

    NL0 = N0[limiting_species]
    if NL0 <= 0:
        raise ValueError("Initial moles of limiting reactant must be > 0.")

    return (X_lim * NL0) / (-nuL)


def moles_from_extent(rxn: Reaction, N0: Dict[str, float], xi: float) -> Dict[str, float]:
    """
    Return moles of all species present in N0 or stoichiometry keys after extent xi.
    """
    nu = rxn.nu_signed()
    species = set(N0.keys()) | set(nu.keys())
    N = {}
    for sp in species:
        N[sp] = float(N0.get(sp, 0.0)) + float(nu.get(sp, 0.0)) * xi
        # Do not auto-clip; negative indicates inconsistent inputs/conversion.
    return N


def concentration_from_moles(N: Dict[str, float], V_L: float) -> Dict[str, float]:
    if V_L <= 0:
        raise ValueError("V_L must be > 0.")
    return {sp: n / V_L for sp, n in N.items()}
