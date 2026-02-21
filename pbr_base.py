# file: bank/pbr_base.py
"""
PBR (Packed Bed Reactor) base functions.

Standard design equation on catalyst weight W:
    dF_A/dW = r_A'   (rate per mass of catalyst)

For liquid-phase / constant volumetric flow v0 and concentration CA:
    F_A = v0 * C_A   (v0 constant)
    dF_A/dW = v0 dC_A/dW

With r_A' negative for consumption, define (-rA') > 0:
    v0 dC_A/dW = -(-rA'(CA))
    dC_A/dW = -(-rA'(CA)) / v0

Sizing to reach CA_out:
    W = ∫_{CA_out}^{CA0} v0 / (-rA'(CA)) dCA

Then convert catalyst weight to bed volume if needed:
    V_bed = W / rho_bulk
where rho_bulk is bulk density of catalyst bed [mass/volume].

Conventions:
- rate functions return (-rA') positive in [mol/(mass_cat*time)]
- v0 in [L/time], concentrations in [mol/L]
- W in [mass_cat]
"""

from __future__ import annotations
from typing import Callable, List, Optional, Tuple
import math


def pbr_catalyst_weight_from_rate(
    v0: float,
    CA0: float,
    CA_out: float,
    rate_minus_rAprime: Callable[[float], float],
    n_steps: int = 20000,
) -> float:
    """
    Compute required catalyst weight:
        W = ∫ v0/(-rA'(CA)) dCA from CA_out to CA0

    Composite Simpson's rule (even n_steps).
    """
    if v0 <= 0:
        raise ValueError("v0 must be > 0.")
    if CA0 < 0 or CA_out < 0:
        raise ValueError("Concentrations must be >= 0.")
    if CA_out > CA0 + 1e-15:
        raise ValueError("CA_out must be <= CA0 for consumption of A.")
    if abs(CA_out - CA0) <= 1e-15:
        return 0.0

    if n_steps < 2:
        raise ValueError("n_steps must be >= 2.")
    if n_steps % 2 == 1:
        n_steps += 1

    a = float(CA_out)
    b = float(CA0)
    h = (b - a) / n_steps

    def integrand(CA: float) -> float:
        r = float(rate_minus_rAprime(CA))
        if r <= 0:
            raise ValueError("(-rA') must be > 0 over the integration interval.")
        return v0 / r

    s = integrand(a) + integrand(b)
    for i in range(1, n_steps):
        x = a + i * h
        s += (4 if i % 2 == 1 else 2) * integrand(x)

    return (h / 3.0) * s


def bed_volume_from_catalyst_weight(W: float, rho_bulk: float) -> float:
    """
    Convert catalyst weight to packed bed volume:
        V_bed = W / rho_bulk

    rho_bulk: bulk density of bed [mass_cat/volume_bed]
    """
    if W < 0:
        raise ValueError("W must be >= 0.")
    if rho_bulk <= 0:
        raise ValueError("rho_bulk must be > 0.")
    return W / rho_bulk


def rk4_integrate_scalar(
    f: Callable[[float, float], float],
    y0: float,
    x0: float,
    x1: float,
    dx: float,
    stop_when: Optional[Callable[[float, float], bool]] = None,
) -> Tuple[List[float], List[float]]:
    if dx <= 0:
        raise ValueError("dx must be > 0.")
    if x1 < x0:
        raise ValueError("x1 must be >= x0.")

    xs: List[float] = [x0]
    ys: List[float] = [y0]

    x = x0
    y = y0

    while x < x1 - 1e-15:
        h = min(dx, x1 - x)

        k1 = f(x, y)
        k2 = f(x + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(x + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(x + h, y + h * k3)

        y_next = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_next = x + h

        if y_next < 0:
            y_next = 0.0

        xs.append(x_next)
        ys.append(y_next)
        x, y = x_next, y_next

        if stop_when and stop_when(x, y):
            break

    return xs, ys


def pbr_profile_CA_vs_W(
    v0: float,
    CA0: float,
    W_final: float,
    rate_minus_rAprime: Callable[[float], float],
    dW: float = 1e-3,
) -> Tuple[List[float], List[float]]:
    """
    Integrate CA(W) using:
        dCA/dW = -(-rA'(CA)) / v0
    """
    if W_final < 0:
        raise ValueError("W_final must be >= 0.")
    if CA0 < 0:
        raise ValueError("CA0 must be >= 0.")
    if v0 <= 0:
        raise ValueError("v0 must be > 0.")

    def dCA_dW(W: float, CA: float) -> float:
        _ = W
        r = float(rate_minus_rAprime(CA))
        if r < 0:
            raise ValueError("(-rA') must be >= 0.")
        return -r / v0

    return rk4_integrate_scalar(dCA_dW, CA0, 0.0, W_final, dW)


if __name__ == "__main__":
    # Example placeholder: user supplies a -rA'(CA) in mol/(kg_cat*min)
    v0 = 25.0
    CA0 = 22.0
    CA_out = 0.5
    rprime = lambda CA: 0.8 * CA  # dummy

    W = pbr_catalyst_weight_from_rate(v0, CA0, CA_out, rprime, n_steps=20000)
    print("PBR W =", W)
