# bank/distillation_mccabe_thiele_binary_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import math


def _vp(name: str, v: float) -> None:
    if v <= 0:
        raise ValueError(f"{name} must be > 0.")


def _vf01_open(name: str, v: float) -> None:
    if not (0.0 <= v < 1.0):
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1.")


# -----------------------------
# Binary constant-alpha VLE
# -----------------------------
def y_eq_const_alpha(x: float, alpha: float) -> float:
    _vf01_open("x", x)
    _vp("alpha", alpha)
    return (alpha * x) / (1.0 + (alpha - 1.0) * x)


def x_eq_from_y_const_alpha(y: float, alpha: float) -> float:
    _vf01_open("y", y)
    _vp("alpha", alpha)
    denom = alpha - (alpha - 1.0) * y
    _vp("denom", denom)
    x = y / denom
    # guard for numeric rounding
    if x >= 1.0:
        x = 1.0 - 1e-15
    _vf01_open("x_eq", x)
    return x


# -----------------------------
# Operating lines & q-line
# -----------------------------
def rectifying_line_params(R: float, x_D: float) -> Tuple[float, float]:
    # y = (R/(R+1)) x + xD/(R+1)
    _vp("R", R)
    _vf01_open("x_D", x_D)
    m = R / (R + 1.0)
    b = x_D / (R + 1.0)
    return m, b


def q_line_params(q: float, x_F: float) -> Tuple[float, float]:
    # y = (q/(q-1)) x - xF/(q-1)
    _vf01_open("x_F", x_F)
    if abs(q - 1.0) < 1e-14:
        # vertical at x = xF
        return float("inf"), x_F
    m = q / (q - 1.0)
    b = -x_F / (q - 1.0)
    return m, b


def intersect_lines(m1: float, b1: float, m2: float, b2: float) -> Tuple[float, float]:
    if not math.isfinite(m1) and not math.isfinite(m2):
        raise ValueError("Both lines vertical; no unique intersection.")
    if not math.isfinite(m1):
        x = b1
        y = m2 * x + b2
        return x, y
    if not math.isfinite(m2):
        x = b2
        y = m1 * x + b1
        return x, y
    if abs(m1 - m2) < 1e-14:
        raise ValueError("Lines are parallel; no unique intersection.")
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y


def stripping_line_from_feedpoint_and_bottom(x_W: float, feed_pt: Tuple[float, float]) -> Tuple[float, float]:
    # Common teaching construction: line through (xW, y=xW) and (x_int,y_int)
    _vf01_open("x_W", x_W)
    x_int, y_int = feed_pt
    _vf01_open("x_int", x_int)
    _vf01_open("y_int", y_int)
    if abs(x_int - x_W) < 1e-14:
        raise ValueError("Cannot form stripping line: feedpoint x equals xW.")
    m = (y_int - x_W) / (x_int - x_W)
    b = y_int - m * x_int
    return m, b


# -----------------------------
# McCabe-Thiele stepping
# -----------------------------
def step_off_stages(
    alpha: float,
    x_D: float,
    x_W: float,
    rect: Tuple[float, float],
    strip: Tuple[float, float],
    x_int: float,
    max_stages: int = 400,
) -> Dict[str, Any]:
    """
    Start at (xD, y=xD) on diagonal (total condenser).
    Horizontal to equilibrium, vertical to op line.
    Switch lines at x_int.
    """
    _vp("alpha", alpha)
    _vf01_open("x_D", x_D)
    _vf01_open("x_W", x_W)
    _vf01_open("x_int", x_int)
    _vp("max_stages", float(max_stages))

    mR, bR = rect
    mS, bS = strip

    pts: List[Tuple[float, float]] = []
    x_curr, y_curr = x_D, x_D
    pts.append((x_curr, y_curr))

    for n in range(1, max_stages + 1):
        x_eq = x_eq_from_y_const_alpha(y_curr, alpha)
        pts.append((x_eq, y_curr))

        if x_eq >= x_int:
            y_next = mR * x_eq + bR
        else:
            y_next = mS * x_eq + bS
        pts.append((x_eq, y_next))

        if x_eq <= x_W + 1e-14:
            # fractional last stage using last two equilibrium x's
            if len(pts) < 5:
                N = 1.0
            else:
                x_prev = pts[-5][0]
                x_now = x_eq
                if abs(x_prev - x_now) < 1e-14:
                    frac = 1.0
                else:
                    frac = (x_prev - x_W) / (x_prev - x_now)
                    frac = max(0.0, min(1.0, frac))
                N = (n - 1) + frac
            return {"N_theoretical": N, "N_ceiling": int(math.ceil(N - 1e-12)), "points_xy": pts}

        y_curr = y_next

    raise RuntimeError("Exceeded max_stages without reaching xW; check feasibility or increase max_stages.")


# -----------------------------
# Helper: total reflux minimum stages (binary Fenske form)
# -----------------------------
def Nmin_total_reflux_binary(alpha_avg: float, x_D: float, x_W: float) -> float:
    # Nmin = ln[(xD/(1-xD))*((1-xW)/xW)] / ln(alpha_avg)
    _vp("alpha_avg", alpha_avg)
    _vf01_open("x_D", x_D)
    _vf01_open("x_W", x_W)
    arg = (x_D / (1.0 - x_D)) * ((1.0 - x_W) / x_W)
    _vp("log_argument", arg)
    return math.log(arg) / math.log(alpha_avg)


@dataclass(frozen=True)
class McCabeThieleBinarySpec:
    alpha: float
    x_D: float
    x_W: float
    x_F: float
    R: float
    q: float
    max_stages: int = 400

    # Overrides for side streams / steam injection variants
    rect_slope: Optional[float] = None
    rect_intercept: Optional[float] = None
    strip_slope: Optional[float] = None
    strip_intercept: Optional[float] = None


def design_mccabe_thiele_binary(spec: McCabeThieleBinarySpec) -> Dict[str, Any]:
    _vp("alpha", spec.alpha)
    _vf01_open("x_D", spec.x_D)
    _vf01_open("x_W", spec.x_W)
    _vf01_open("x_F", spec.x_F)
    _vp("R", spec.R)
    _vp("max_stages", float(spec.max_stages))

    # rectifying line
    if spec.rect_slope is None or spec.rect_intercept is None:
        mR, bR = rectifying_line_params(spec.R, spec.x_D)
    else:
        mR, bR = float(spec.rect_slope), float(spec.rect_intercept)

    # q-line
    mq, bq = q_line_params(spec.q, spec.x_F)

    # intersection
    x_int, y_int = intersect_lines(mR, bR, mq, bq)

    # stripping line
    if spec.strip_slope is None or spec.strip_intercept is None:
        mS, bS = stripping_line_from_feedpoint_and_bottom(spec.x_W, (x_int, y_int))
    else:
        mS, bS = float(spec.strip_slope), float(spec.strip_intercept)

    stepping = step_off_stages(
        alpha=spec.alpha,
        x_D=spec.x_D,
        x_W=spec.x_W,
        rect=(mR, bR),
        strip=(mS, bS),
        x_int=x_int,
        max_stages=spec.max_stages,
    )

    return {
        "inputs": {
            "alpha": spec.alpha,
            "x_D": spec.x_D,
            "x_W": spec.x_W,
            "x_F": spec.x_F,
            "R": spec.R,
            "q": spec.q,
        },
        "lines": {
            "rectifying": {"slope": mR, "intercept": bR},
            "q_line": {"slope": mq, "intercept": bq, "vertical_x_if_q_eq_1": bq if not math.isfinite(mq) else None},
            "stripping": {"slope": mS, "intercept": bS},
        },
        "intersection": {"x_int": x_int, "y_int": y_int},
        "stages": stepping,
    }