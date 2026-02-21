# file: bank/cstr_gas_base.py
"""
Ideal-gas CSTR (isothermal + isobaric) for a single reaction, outlet-rate evaluation.

Now supports specifying the inlet either by:
- molar flows F0 [mol/time]  (recommended), OR
- concentrations C0 [mol/L] with inlet volumetric flow v0 [L/time]

Inerts:
- No special handling needed. Include inert species in F0 or C0 with nu_signed = 0
  (or omit from nu_signed; it will be treated as 0).

Key relations:
- Total concentration is constant at operating T,P:
      CT = FT0 / v0   (if you provide F0)
      CT = sum(C0)    (if you provide C0; overrides CT inferred from F0)
- Outlet flows at extent-rate xi (mol/time):
      Fi = Fi0 + nu_i * xi
- Outlet concentrations:
      Ci = Fi * CT / FT

CSTR volume on basis species "A" for (-rA) provided:
      V = (FA0 - FA_out) / (-rA_out)

Conventions:
- nu_signed: reactants negative, products positive
- rate function returns (-r_basis) > 0 in mol/L/time
- Consistent time units required across F0, k, and v0.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple


def _coerce_nu_signed(nu_signed: Dict[str, float]) -> Dict[str, float]:
    return {sp: float(v) for sp, v in nu_signed.items()}


def resolve_feed_gas(
    v0: float,
    *,
    F0: Optional[Dict[str, float]] = None,
    C0: Optional[Dict[str, float]] = None,
    CT: Optional[float] = None,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """
    Resolve inlet description.

    Returns:
        F0_resolved : mol/time
        C0_resolved : mol/L
        CT_resolved : mol/L
    """
    if v0 <= 0:
        raise ValueError("v0 must be > 0.")

    if (F0 is None) == (C0 is None):
        raise ValueError("Provide exactly one of F0 or C0.")

    if F0 is not None:
        F0r = {sp: float(val) for sp, val in F0.items()}
        if any(val < 0 for val in F0r.values()):
            raise ValueError("All inlet molar flows must be >= 0.")
        FT0 = sum(F0r.values())
        if FT0 <= 0:
            raise ValueError("Total inlet molar flow must be > 0.")
        CTr = (FT0 / v0) if CT is None else float(CT)
        if CTr <= 0:
            raise ValueError("CT must be > 0.")
        # concentrations from flows
        C0r = {sp: F0r[sp] / v0 for sp in F0r.keys()}
        # If CT was supplied and differs from sum(C0r), that's okay; CT governs ideal-gas mapping
        return F0r, C0r, CTr

    # C0 provided
    C0r = {sp: float(val) for sp, val in C0.items()}
    if any(val < 0 for val in C0r.values()):
        raise ValueError("All inlet concentrations must be >= 0.")
    CT_sum = sum(C0r.values())
    if CT_sum <= 0:
        raise ValueError("Sum of inlet concentrations must be > 0.")
    CTr = CT_sum if CT is None else float(CT)
    if CTr <= 0:
        raise ValueError("CT must be > 0.")
    # flows from concentrations using v0
    F0r = {sp: v0 * C0r[sp] for sp in C0r.keys()}
    return F0r, C0r, CTr


def outlet_flows_from_xi(
    F0: Dict[str, float],
    nu_signed: Dict[str, float],
    xi: float,
) -> Dict[str, float]:
    """
    Fi = Fi0 + nu_i*xi
    """
    if xi < 0:
        raise ValueError("xi must be >= 0.")
    species = set(F0.keys()) | set(nu_signed.keys())
    F = {}
    for sp in species:
        F[sp] = float(F0.get(sp, 0.0)) + float(nu_signed.get(sp, 0.0)) * float(xi)
    return F


def outlet_concentrations_from_xi(
    v0: float,
    *,
    F0: Optional[Dict[str, float]] = None,
    C0: Optional[Dict[str, float]] = None,
    CT: Optional[float] = None,
    nu_signed: Dict[str, float],
    xi: float,
) -> Dict[str, float]:
    """
    Ci = Fi * CT / FT, with CT constant at operating T,P.
    Feed can be specified by F0 or C0; CT is resolved accordingly.
    """
    F0r, _, CTr = resolve_feed_gas(v0, F0=F0, C0=C0, CT=CT)
    nu = _coerce_nu_signed(nu_signed)

    F = outlet_flows_from_xi(F0r, nu, xi)
    FT = sum(F.values())
    if FT <= 0:
        raise ValueError("Total outlet molar flow became non-positive; check xi/stoichiometry.")
    return {sp: (F[sp] * CTr / FT) for sp in F.keys()}


def extent_out_from_conversion(
    *,
    v0: float,
    reactant_stoich: Dict[str, float],   # positive stoich for reactants
    limiting_species: str,
    X_lim: float,
    F0: Optional[Dict[str, float]] = None,
    C0: Optional[Dict[str, float]] = None,
) -> float:
    """
    xi_out = (F_L0 * X_lim) / alpha_L

    If you pass concentrations C0, the inlet molar flow is F_L0 = v0*C_L0.
    """
    if not (0.0 <= X_lim <= 1.0):
        raise ValueError("X_lim must be in [0,1].")
    if limiting_species not in reactant_stoich:
        raise KeyError("limiting_species not in reactant_stoich.")

    alpha_L = float(reactant_stoich[limiting_species])
    if alpha_L <= 0:
        raise ValueError("Reactant stoich must be > 0.")

    F0r, C0r, _ = resolve_feed_gas(v0, F0=F0, C0=C0, CT=None)

    if limiting_species in F0r:
        FL0 = float(F0r[limiting_species])
    else:
        # allow missing species (treated as 0)
        FL0 = 0.0

    if FL0 < 0:
        raise ValueError("Initial molar flow must be >= 0.")
    return (FL0 * X_lim) / alpha_L


def cstr_volume_isothermal_isobaric_ideal_gas(
    v0: float,                                # L/time at operating conditions
    nu_signed: Dict[str, float],              # signed stoich
    basis_species_for_rate: str,              # e.g. "A" for (-rA)
    rate_minus_r_basis: Callable[[Dict[str, float]], float],  # local C dict -> (-r_basis)
    xi_out: float,                            # mol/time
    *,
    F0: Optional[Dict[str, float]] = None,    # mol/time
    C0: Optional[Dict[str, float]] = None,    # mol/L
    CT: Optional[float] = None,               # mol/L (optional override)
) -> float:
    """
    Volume:
        V = (F_basis0 - F_basis_out)/(-r_basis_out)

    Feed can be specified by F0 or C0.
    """
    if xi_out < 0:
        raise ValueError("xi_out must be >= 0.")
    nu = _coerce_nu_signed(nu_signed)

    F0r, _, CTr = resolve_feed_gas(v0, F0=F0, C0=C0, CT=CT)
    if basis_species_for_rate not in F0r:
        raise KeyError("basis species missing from inlet feed (F0/C0).")

    F_out = outlet_flows_from_xi(F0r, nu, xi_out)
    Fbasis0 = float(F0r[basis_species_for_rate])
    Fbasis = float(F_out.get(basis_species_for_rate, 0.0))
    if Fbasis > Fbasis0 + 1e-12:
        raise ValueError("Basis species increased; check stoichiometry/signs.")

    FT = sum(F_out.values())
    if FT <= 0:
        raise ValueError("Total outlet molar flow became non-positive; check xi/stoichiometry.")
    C_out = {sp: (F_out[sp] * CTr / FT) for sp in F_out.keys()}

    minus_r = float(rate_minus_r_basis(C_out))
    if minus_r <= 0:
        raise ValueError("(-r) at outlet must be > 0.")

    return (Fbasis0 - Fbasis) / minus_r
