# bank/henry_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class HenryProblem:
    """
    Convention used here (common in chem eng appendices):
      P = H * x

    where:
      P  = solute partial pressure in gas phase (same pressure units as H)
      H  = Henry's law constant (pressure units)
      x  = solute mole fraction in liquid phase (dimensionless)

    This module also supports building x from common liquid concentration specs
    (e.g., kg solute per kg solvent), but does NOT store any substance constants.
    Those belong in the notebook.
    """
    H: Optional[float] = None                 # Henry constant (e.g., atm)
    P: Optional[float] = None                 # partial pressure (e.g., atm)
    x: Optional[float] = None                 # mole fraction in liquid

    # Optional concentration specs to infer x (provide what you have)
    kg_solute_per_kg_solvent: Optional[float] = None  # e.g., kg CO2 / kg H2O
    mol_solute_per_kg_solvent: Optional[float] = None # mol solute / kg solvent

    # Molecular weights needed only when converting mass-based specs -> moles
    MW_solute: Optional[float] = None         # kg/mol
    MW_solvent: Optional[float] = None        # kg/mol


def mole_fraction_from_kg_per_kg(
    kg_solute_per_kg_solvent: float,
    MW_solute: float,
    MW_solvent: float,
) -> float:
    """
    Given:
      w = kg solute / kg solvent

    Take 1 kg solvent basis:
      n_solute  = w / MW_solute
      n_solvent = 1 / MW_solvent

      x_solute = n_solute / (n_solute + n_solvent)
    """
    if MW_solute <= 0 or MW_solvent <= 0:
        raise ValueError("MW_solute and MW_solvent must be > 0.")
    if kg_solute_per_kg_solvent < 0:
        raise ValueError("kg_solute_per_kg_solvent must be >= 0.")

    n_solute = kg_solute_per_kg_solvent / MW_solute
    n_solvent = 1.0 / MW_solvent
    denom = n_solute + n_solvent
    if denom <= 0:
        raise ValueError("Invalid composition; denominator <= 0.")
    return n_solute / denom


def mole_fraction_from_molality(mol_solute_per_kg_solvent: float, MW_solvent: float) -> float:
    """
    Given:
      m = mol solute / kg solvent

    Take 1 kg solvent basis:
      n_solute  = m
      n_solvent = 1 / MW_solvent
      x = n_solute / (n_solute + n_solvent)
    """
    if MW_solvent <= 0:
        raise ValueError("MW_solvent must be > 0.")
    if mol_solute_per_kg_solvent < 0:
        raise ValueError("mol_solute_per_kg_solvent must be >= 0.")

    n_solute = mol_solute_per_kg_solvent
    n_solvent = 1.0 / MW_solvent
    denom = n_solute + n_solvent
    if denom <= 0:
        raise ValueError("Invalid composition; denominator <= 0.")
    return n_solute / denom


def solve_henry(problem: HenryProblem) -> Dict[str, Any]:
    """
    Solves for any missing variable among (P, H, x), and can infer x from
    common concentration inputs if x is not directly provided.

    Accepted pathways:
      - If x missing:
          * compute from kg_solute_per_kg_solvent with MWs
          * OR compute from mol_solute_per_kg_solvent with MW_solvent
      - Then apply Henry:
          P = H*x  (solve any one missing among P,H,x if the other two exist)

    Returns a dict with:
      P, H, x, plus flags describing what was inferred.
    """
    H = problem.H
    P = problem.P
    x = problem.x

    inferred = {
        "x_inferred_from": None,
        "henry_solved_for": None,
    }

    # Infer x if needed
    if x is None:
        if problem.kg_solute_per_kg_solvent is not None:
            if problem.MW_solute is None or problem.MW_solvent is None:
                raise ValueError("MW_solute and MW_solvent required to convert kg/kg -> x.")
            x = mole_fraction_from_kg_per_kg(
                problem.kg_solute_per_kg_solvent,
                problem.MW_solute,
                problem.MW_solvent,
            )
            inferred["x_inferred_from"] = "kg_solute_per_kg_solvent"
        elif problem.mol_solute_per_kg_solvent is not None:
            if problem.MW_solvent is None:
                raise ValueError("MW_solvent required to convert mol/kg -> x.")
            x = mole_fraction_from_molality(problem.mol_solute_per_kg_solvent, problem.MW_solvent)
            inferred["x_inferred_from"] = "mol_solute_per_kg_solvent"

    # Basic validation if present
    if x is not None and not (0.0 <= x <= 1.0):
        raise ValueError("Mole fraction x must be between 0 and 1.")
    if H is not None and H <= 0:
        raise ValueError("Henry constant H must be > 0.")
    if P is not None and P < 0:
        raise ValueError("Partial pressure P must be >= 0.")

    # Solve Henry relation for the missing variable (if possible)
    known = {"P": P is not None, "H": H is not None, "x": x is not None}

    if known["H"] and known["x"] and not known["P"]:
        P = H * x
        inferred["henry_solved_for"] = "P"
    elif known["P"] and known["x"] and not known["H"]:
        if x == 0:
            raise ValueError("Cannot solve for H when x = 0 (division by zero).")
        H = P / x
        inferred["henry_solved_for"] = "H"
    elif known["P"] and known["H"] and not known["x"]:
        x = P / H
        inferred["henry_solved_for"] = "x"
    elif known["P"] and known["H"] and known["x"]:
        # nothing to solve; can optionally check consistency
        pass
    else:
        raise ValueError(
            "Insufficient information. Provide enough data to determine two of (P,H,x) "
            "or provide a concentration spec to infer x plus one of (P or H)."
        )

    return {"P": P, "H": H, "x": x, **inferred}