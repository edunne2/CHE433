# file: bank/pfr_gas_base.py
"""
Ideal-gas PFR base (isothermal + isobaric) using extent integration.

UPDATED:
- Feed can be specified by inlet molar flows F0 [mol/time] OR inlet concentrations C0 [mol/L]
- Inerts optional: include inert in F0/C0 with nu_signed=0 (or omit from nu_signed)
- CT can be supplied or inferred:
    If F0 provided: CT = FT0/v0
    If C0 provided: CT = sum(C0)

Back-compat:
- The original pfr_volume_isothermal_isobaric_ideal_gas signature is kept.
- New wrapper pfr_volume_isothermal_isobaric_ideal_gas_feed supports F0/C0.

Conventions:
- rate_minus_r_basis takes local concentration dict -> (-r_basis) > 0 [mol/L/time]
- nu_signed has reactants negative
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple


def resolve_feed_gas(
    v0: float,
    *,
    F0: Optional[Dict[str, float]] = None,
    C0: Optional[Dict[str, float]] = None,
    CT: Optional[float] = None,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """
    Returns:
        F0_resolved (mol/time), C0_resolved (mol/L), CT_resolved (mol/L)
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
        C0r = {sp: F0r[sp] / v0 for sp in F0r.keys()}
        return F0r, C0r, CTr

    C0r = {sp: float(val) for sp, val in C0.items()}
    if any(val < 0 for val in C0r.values()):
        raise ValueError("All inlet concentrations must be >= 0.")
    CT_sum = sum(C0r.values())
    if CT_sum <= 0:
        raise ValueError("Sum of inlet concentrations must be > 0.")
    CTr = CT_sum if CT is None else float(CT)
    if CTr <= 0:
        raise ValueError("CT must be > 0.")
    F0r = {sp: v0 * C0r[sp] for sp in C0r.keys()}
    return F0r, C0r, CTr


def extent_out_from_conversion(
    F0: Dict[str, float],
    reactant_stoich: Dict[str, float],  # positive stoich for reactants
    limiting_species: str,
    X_lim: float,
) -> float:
    if not (0.0 <= X_lim <= 1.0):
        raise ValueError("X_lim must be in [0,1].")
    if limiting_species not in F0:
        raise KeyError("limiting_species not in F0.")
    if limiting_species not in reactant_stoich:
        raise KeyError("limiting_species not in reactant stoich.")

    alpha_L = float(reactant_stoich[limiting_species])  # positive
    if alpha_L <= 0:
        raise ValueError("Reactant stoich must be > 0.")

    FL0 = float(F0[limiting_species])
    if FL0 < 0:
        raise ValueError("Initial molar flows must be >= 0.")

    return (FL0 * X_lim) / alpha_L


def pfr_volume_isothermal_isobaric_ideal_gas(
    v0: float,                                # L/time (inlet volumetric flow)
    C0: Dict[str, float],                     # mol/L (inlet concentrations)
    nu_signed: Dict[str, float],              # signed stoich coeffs
    CT: float,                                # mol/L constant total concentration
    basis_species_for_rate: str,              # e.g., "A"
    rate_minus_r_basis: Callable[[Dict[str, float]], float],  # local C dict -> (-r_basis)
    xi_out: float,                            # mol/time (extent rate at outlet)
    n_steps: int = 200000,                    # Simpson steps (even)
) -> float:
    # --- original implementation kept (unchanged) ---
    if v0 <= 0:
        raise ValueError("v0 must be > 0.")
    if CT <= 0:
        raise ValueError("CT must be > 0.")
    if basis_species_for_rate not in nu_signed:
        raise KeyError("basis species missing from nu_signed.")

    nu_basis = float(nu_signed[basis_species_for_rate])
    if nu_basis >= 0:
        raise ValueError("basis_species_for_rate must be a reactant (negative nu).")

    # inlet molar flows
    F0 = {sp: v0 * float(C0.get(sp, 0.0)) for sp in set(C0.keys()) | set(nu_signed.keys())}
    FT0 = sum(F0.values())

    if abs(FT0 - v0 * CT) / (v0 * CT) > 1e-9:
        raise ValueError("Inconsistent CT vs inlet concentrations: expected CT ≈ sum(C0).")

    if xi_out < 0:
        raise ValueError("xi_out must be >= 0.")

    if n_steps < 2:
        raise ValueError("n_steps must be >= 2.")
    if n_steps % 2 == 1:
        n_steps += 1

    a = 0.0
    b = float(xi_out)
    h = (b - a) / n_steps

    def dxi_dV(xi: float) -> float:
        F = {sp: F0.get(sp, 0.0) + nu_signed.get(sp, 0.0) * xi for sp in F0.keys()}
        FT = sum(F.values())
        if FT <= 0:
            raise ValueError("Total molar flow became non-positive; check stoichiometry/conversion.")
        C = {sp: (F[sp] * CT / FT) for sp in F.keys()}
        minus_r = float(rate_minus_r_basis(C))
        if minus_r <= 0:
            raise ValueError("(-r) must be > 0 over the integration interval.")
        return -minus_r / nu_basis  # nu_basis negative

    def integrand(xi: float) -> float:
        return 1.0 / dxi_dV(xi)

    s = integrand(a) + integrand(b)
    for i in range(1, n_steps):
        x = a + i * h
        s += (4 if i % 2 == 1 else 2) * integrand(x)

    return (h / 3.0) * s


def pfr_volume_isothermal_isobaric_ideal_gas_feed(
    *,
    v0: float,                                   # L/time
    nu_signed: Dict[str, float],
    basis_species_for_rate: str,
    rate_minus_r_basis: Callable[[Dict[str, float]], float],
    xi_out: float,                                # mol/time
    F0: Optional[Dict[str, float]] = None,         # mol/time
    C0: Optional[Dict[str, float]] = None,         # mol/L
    CT: Optional[float] = None,                    # mol/L (optional override)
    n_steps: int = 200000,
) -> float:
    """
    Wrapper that accepts either F0 or C0 and infers CT consistently, then calls the core integrator.
    """
    F0r, C0r, CTr = resolve_feed_gas(v0, F0=F0, C0=C0, CT=CT)

    # If C0 was derived from F0, sum(C0r) may not equal CT when CT override was supplied.
    # For the integrator, pass inlet concentrations and the CT that governs ideal-gas mapping.
    return pfr_volume_isothermal_isobaric_ideal_gas(
        v0=v0,
        C0=C0r,
        nu_signed=nu_signed,
        CT=CTr,
        basis_species_for_rate=basis_species_for_rate,
        rate_minus_r_basis=rate_minus_r_basis,
        xi_out=xi_out,
        n_steps=n_steps,
    )