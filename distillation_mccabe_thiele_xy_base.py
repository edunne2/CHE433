# bank/distillation_mccabe_thiele_xy_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Sequence, List, Tuple
import math


def _vp(name: str, v: float) -> None:
    if v <= 0:
        raise ValueError(f"{name} must be > 0.")


def _vf01_open(name: str, v: float) -> None:
    if not (0.0 <= v < 1.0):
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1.")


def _vf01(name: str, v: float) -> None:
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"{name} must be in [0,1].")


def _sort_pairs_by_first(a: Sequence[float], b: Sequence[float]) -> Tuple[List[float], List[float]]:
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Sequences must have same length >= 2.")
    pairs = sorted((float(ai), float(bi)) for ai, bi in zip(a, b))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _interp_piecewise_linear(x: float, xs: List[float], ys: List[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0, x1 = xs[lo], xs[lo + 1]
    y0, y1 = ys[lo], ys[lo + 1]
    if abs(x1 - x0) < 1e-18:
        return 0.5 * (y0 + y1)
    w = (x - x0) / (x1 - x0)
    return y0 + w * (y1 - y0)


@dataclass(frozen=True)
class XYEquilibrium:
    """
    Binary equilibrium y(x) (light component) from tabulated points.
    Provides y_of_x(x) and x_of_y(y) via piecewise-linear interpolation.
    """
    x: Sequence[float]
    y: Sequence[float]

    def __post_init__(self) -> None:
        xs, ys = _sort_pairs_by_first(self.x, self.y)
        for i, (xi, yi) in enumerate(zip(xs, ys)):
            _vf01(f"x[{i}]", xi)
            _vf01(f"y[{i}]", yi)
        object.__setattr__(self, "_xs", xs)
        object.__setattr__(self, "_ys", ys)

        ys2, xs2 = _sort_pairs_by_first(ys, xs)
        object.__setattr__(self, "_ys2", ys2)
        object.__setattr__(self, "_xs2", xs2)

    def y_of_x(self, x: float) -> float:
        _vf01("x", x)
        return float(_interp_piecewise_linear(float(x), self._xs, self._ys))

    def x_of_y(self, y: float) -> float:
        _vf01("y", y)
        return float(_interp_piecewise_linear(float(y), self._ys2, self._xs2))


def rectifying_line_params(R: float, x_D: float) -> Tuple[float, float]:
    _vp("R", R)
    _vf01_open("x_D", x_D)
    m = R / (R + 1.0)
    b = x_D / (R + 1.0)
    return m, b


def q_line_params(q: float, x_F: float) -> Tuple[float, float]:
    _vf01_open("x_F", x_F)
    if abs(q - 1.0) < 1e-14:
        return float("inf"), x_F  # vertical at x=x_F
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
    _vf01_open("x_W", x_W)
    x_int, y_int = feed_pt
    _vf01("x_int", x_int)
    _vf01("y_int", y_int)
    if abs(x_int - x_W) < 1e-14:
        raise ValueError("Cannot form stripping line: x_int equals x_W.")
    m = (y_int - x_W) / (x_int - x_W)
    b = y_int - m * x_int
    return m, b


def step_off_stages_xy(
    eq: XYEquilibrium,
    x_D: float,
    x_W: float,
    rect: Tuple[float, float],
    strip: Tuple[float, float],
    x_int: float,
    max_stages: int = 600,
) -> Dict[str, Any]:
    """
    Stepping convention:
      start at (x_D, y=x_D) (total condenser),
      horizontal to equilibrium (x from y),
      vertical to operating line,
      switch rect/strip at x_int.
    Returns points and theoretical stage count (with fractional last stage).
    """
    _vf01_open("x_D", x_D)
    _vf01_open("x_W", x_W)
    _vf01("x_int", x_int)
    _vp("max_stages", float(max_stages))

    mR, bR = rect
    mS, bS = strip

    pts: List[Tuple[float, float]] = [(x_D, x_D)]
    y_curr = x_D

    xeq_history: List[float] = []

    for n in range(1, max_stages + 1):
        x_eq = eq.x_of_y(y_curr)
        pts.append((x_eq, y_curr))
        xeq_history.append(x_eq)
        if len(xeq_history) > 2:
            xeq_history = xeq_history[-2:]

        if x_eq >= x_int:
            y_next = mR * x_eq + bR
        else:
            y_next = mS * x_eq + bS
        y_next = max(0.0, min(1.0, y_next))
        pts.append((x_eq, y_next))

        if x_eq <= x_W + 1e-14:
            if len(xeq_history) < 2:
                N = float(n)
            else:
                x_prev, x_now = xeq_history[0], xeq_history[1]
                if abs(x_prev - x_now) < 1e-14:
                    frac = 1.0
                else:
                    frac = (x_prev - x_W) / (x_prev - x_now)
                    frac = max(0.0, min(1.0, frac))
                N = (n - 1) + frac
            return {"N_theoretical": N, "N_ceiling": int(math.ceil(N - 1e-12)), "points_xy": pts}

        y_curr = y_next

    raise RuntimeError("Exceeded max_stages without reaching x_W.")


def feed_tray_from_points(points_xy: Sequence[Tuple[float, float]], x_int: float) -> Dict[str, int]:
    """
    Compute feed tray number from stepping points.

    points_xy pattern:
      index 0: (xD,xD)
      index 1: (x_eq1, y1)
      index 2: (x_eq1, y2)
      index 3: (x_eq2, y2)
      ...

    Equilibrium hits are at indices 1,3,5,... (odd indices).
    Rectifying section counted where x_eq >= x_int.

    Returns:
      N_rect: number of trays above feed (integer)
      feed_tray_from_top: N_rect + 1
    """
    _vf01("x_int", x_int)
    pts = list(points_xy)
    x_eq_hits = [pts[i][0] for i in range(1, len(pts), 2)]
    N_rect = sum(1 for x in x_eq_hits if x >= x_int - 1e-14)
    return {"N_rect": int(N_rect), "feed_tray_from_top": int(N_rect) + 1}


@dataclass(frozen=True)
class McCabeThieleXYSpec:
    eq: XYEquilibrium
    F: float
    zF: float
    x_D: float
    x_W: float
    R: float
    q: float
    max_stages: int = 600


def solve_mccabe_thiele_xy(spec: McCabeThieleXYSpec) -> Dict[str, Any]:
    _vp("F", spec.F)
    _vf01_open("zF", spec.zF)
    _vf01_open("x_D", spec.x_D)
    _vf01_open("x_W", spec.x_W)
    _vp("R", spec.R)
    _vp("max_stages", float(spec.max_stages))

    denom = (spec.x_D - spec.x_W)
    if abs(denom) < 1e-14:
        raise ValueError("x_D must differ from x_W.")

    D = spec.F * (spec.zF - spec.x_W) / denom
    W = spec.F - D
    _vp("D", D)
    _vp("W", W)

    mR, bR = rectifying_line_params(spec.R, spec.x_D)
    mq, bq = q_line_params(spec.q, spec.zF)
    x_int, y_int = intersect_lines(mR, bR, mq, bq)
    mS, bS = stripping_line_from_feedpoint_and_bottom(spec.x_W, (x_int, y_int))

    stepping = step_off_stages_xy(
        eq=spec.eq,
        x_D=spec.x_D,
        x_W=spec.x_W,
        rect=(mR, bR),
        strip=(mS, bS),
        x_int=x_int,
        max_stages=spec.max_stages,
    )

    feed_info = feed_tray_from_points(stepping["points_xy"], x_int)

    return {
        "inputs": {
            "F": spec.F,
            "zF": spec.zF,
            "x_D": spec.x_D,
            "x_W": spec.x_W,
            "R": spec.R,
            "q": spec.q,
        },
        "flows": {"D": D, "W": W},
        "lines": {
            "rectifying": {"slope": mR, "intercept": bR},
            "q_line": {"slope": mq, "intercept": bq, "vertical_x_if_q_eq_1": bq if not math.isfinite(mq) else None},
            "stripping": {"slope": mS, "intercept": bS},
        },
        "intersection": {"x_int": x_int, "y_int": y_int},
        "stages": stepping,
        "feed": feed_info,
    }