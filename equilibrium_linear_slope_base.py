# bank/equilibrium_linear_slope_from_table_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Sequence, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class LinearSlopeFitSpec:
    """
    Compute a single linear-equilibrium slope m for y* = m x from tabulated (x,y).

    Supported methods:
      - "point":         m = y(x_ref)/x_ref using linear interpolation on table
      - "fit_origin":    least-squares fit constrained through origin (recommended for y*=mx)
      - "fit_intercept": least-squares fit y = a + m x (returns both a and m; m still returned)

    Range selection:
      - Use x_range=(xmin,xmax) or x_max to restrict to dilute region.
    """
    x: Sequence[float]
    y: Sequence[float]
    method: str = "fit_origin"
    x_ref: Optional[float] = None
    x_range: Optional[Tuple[float, float]] = None
    x_max: Optional[float] = None


def _as_1d_float(a: Sequence[float], name: str) -> np.ndarray:
    v = np.asarray(a, dtype=float)
    if v.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    if len(v) < 2:
        raise ValueError(f"{name} must have length >= 2.")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} contains non-finite values.")
    return v


def _validate_xy(x: np.ndarray, y: np.ndarray) -> None:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if np.any(x < 0) or np.any(x >= 1) or np.any(y < 0) or np.any(y >= 1):
        raise ValueError("x and y must be mole fractions in [0,1).")
    # sort by x if needed
    if not np.all(np.diff(x) >= 0):
        idx = np.argsort(x)
        x[:] = x[idx]
        y[:] = y[idx]


def _apply_range(x: np.ndarray, y: np.ndarray, x_range: Optional[Tuple[float, float]], x_max: Optional[float]) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones_like(x, dtype=bool)
    if x_range is not None:
        xmin, xmax = x_range
        if xmin >= xmax:
            raise ValueError("x_range must satisfy xmin < xmax.")
        mask &= (x >= xmin) & (x <= xmax)
    if x_max is not None:
        mask &= (x <= x_max)
    xf = x[mask]
    yf = y[mask]
    if len(xf) < 2:
        raise ValueError("Range filter left <2 points; widen x_range/x_max.")
    return xf, yf


def linear_equilibrium_slope_from_table(spec: LinearSlopeFitSpec) -> Dict[str, Any]:
    x = _as_1d_float(spec.x, "x")
    y = _as_1d_float(spec.y, "y")
    _validate_xy(x, y)

    xf, yf = _apply_range(x, y, spec.x_range, spec.x_max)

    method = spec.method.lower().strip()

    if method == "point":
        if spec.x_ref is None:
            raise ValueError("x_ref must be provided for method='point'.")
        x_ref = float(spec.x_ref)
        if x_ref <= 0:
            raise ValueError("x_ref must be > 0 for method='point'.")
        # interpolate on full sorted table (not filtered) to ensure bracket exists
        y_ref = float(np.interp(x_ref, x, y))
        m = y_ref / x_ref
        return {
            "m": float(m),
            "method": "point",
            "x_ref": x_ref,
            "y_ref": y_ref,
            "n_points_used": 1,
            "note": "m computed as y_ref/x_ref using linear interpolation on the table.",
        }

    if method == "fit_origin":
        # least squares through origin: m = (x·y)/(x·x)
        denom = float(np.dot(xf, xf))
        if denom <= 0:
            raise ValueError("Degenerate x values for fit.")
        m = float(np.dot(xf, yf) / denom)
        return {
            "m": m,
            "method": "fit_origin",
            "x_range_used": (float(xf.min()), float(xf.max())),
            "n_points_used": int(len(xf)),
            "SSE": float(np.sum((yf - m * xf) ** 2)),
        }

    if method == "fit_intercept":
        # y = a + m x
        A = np.vstack([xf, np.ones_like(xf)]).T
        m, a = np.linalg.lstsq(A, yf, rcond=None)[0]
        m = float(m)
        a = float(a)
        return {
            "m": m,
            "a": a,
            "method": "fit_intercept",
            "x_range_used": (float(xf.min()), float(xf.max())),
            "n_points_used": int(len(xf)),
            "SSE": float(np.sum((yf - (m * xf + a)) ** 2)),
            "note": "For strict y*=m x equilibrium, prefer fit_origin unless instructed otherwise.",
        }

    raise ValueError("method must be one of: 'point', 'fit_origin', 'fit_intercept'.")