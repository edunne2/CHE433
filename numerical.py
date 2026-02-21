"""Unified numerical methods"""
from typing import Callable, Optional, Tuple, List, Any
import math
from .validation import ConvergenceError, check_positive


def log_mean(a: float, b: float) -> float:
    """
    Calculate the log mean of two values.
    
    The log mean is defined as (a - b) / ln(a/b) for a ≠ b,
    and equals a when a = b.
    
    Used in Eqs. 22.1-35, 22.1-36, 22.1-54, 22.1-56, 22.5-24, 22.5-25
    """
    if abs(a - b) < 1e-12:
        return a
    if a <= 0 or b <= 0:
        raise ValueError(f"Log mean requires positive values, got {a}, {b}")
    return (a - b) / math.log(a / b)


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    args: tuple = (),
    tol: float = 1e-12,
    maxiter: int = 400,
    expand_bracket: bool = False,
    expand_factor: float = 1.6,
    bracket_expand_max: int = 20,
) -> float:
    """Find root using bisection with optional bracket expansion"""
    check_positive("tol", tol)
    check_positive("maxiter", maxiter)
    
    def f_wrapped(x: float) -> float:
        return f(x, *args) if args else f(x)
    
    fa, fb = f_wrapped(a), f_wrapped(b)
    
    if abs(fa) <= tol:
        return a
    if abs(fb) <= tol:
        return b
    
    # Expand bracket if needed
    if fa * fb > 0 and expand_bracket:
        center, span = 0.5 * (a + b), 0.5 * (b - a)
        for _ in range(bracket_expand_max):
            span *= expand_factor
            a_new, b_new = center - span, center + span
            fa_new, fb_new = f_wrapped(a_new), f_wrapped(b_new)
            if fa_new * fb_new <= 0:
                a, b, fa, fb = a_new, b_new, fa_new, fb_new
                break
        else:
            raise ConvergenceError(f"Cannot bracket root after expansion")
    elif fa * fb > 0:
        raise ConvergenceError(f"No sign change in [{a}, {b}]")
    
    lo, hi = a, b
    flo, fhi = fa, fb
    
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fm = f_wrapped(mid)
        
        if abs(fm) <= tol:
            return mid
        
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    
    final = 0.5 * (lo + hi)
    if abs(f_wrapped(final)) <= tol * 10:
        return final
    
    raise ConvergenceError(f"Bisection failed after {maxiter} iterations")


def newton_raphson(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    args: tuple = (),
    tol: float = 1e-12,
    maxiter: int = 100,
) -> float:
    """Newton-Raphson root finding"""
    def f_wrapped(x): return f(x, *args) if args else f(x)
    def df_wrapped(x): return df(x, *args) if args else df(x)
    
    x = x0
    for _ in range(maxiter):
        fx = f_wrapped(x)
        if abs(fx) <= tol:
            return x
        
        dfx = df_wrapped(x)
        if abs(dfx) < 1e-15:
            raise ConvergenceError("Zero derivative")
        
        x_new = x - fx / dfx
        if abs(x_new - x) <= tol:
            return x_new
        x = x_new
    
    raise ConvergenceError(f"Newton-Raphson failed after {maxiter} iterations")


def integrate_trapezoid(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 1000,
    args: tuple = (),
) -> float:
    """Composite trapezoid rule"""
    if n < 2:
        raise ValueError("n must be >= 2")
    if a >= b:
        raise ValueError(f"Require a < b, got a={a}, b={b}")
    
    def f_wrapped(x): return f(x, *args) if args else f(x)
    
    dx = (b - a) / (n - 1)
    integral = 0.5 * (f_wrapped(a) + f_wrapped(b))
    
    for i in range(1, n - 1):
        integral += f_wrapped(a + i * dx)
    
    return integral * dx


def integrate_adaptive(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_depth: int = 10,
    args: tuple = (),
) -> float:
    """
    Adaptive Simpson's rule integration.
    
    Args:
        f: Function to integrate
        a: Lower limit
        b: Upper limit
        tol: Relative tolerance
        max_depth: Maximum recursion depth
        args: Additional arguments to pass to f
    
    Returns:
        Approximate integral
    """
    def f_wrapped(x: float) -> float:
        return f(x, *args) if args else f(x)
    
    def _simpson_rule(lo: float, hi: float) -> Tuple[float, float]:
        """Compute Simpson's rule and mid-point refinement"""
        mid = 0.5 * (lo + hi)
        h = hi - lo
        
        # Simpson's rule with 3 points
        s = (h / 6) * (f_wrapped(lo) + 4 * f_wrapped(mid) + f_wrapped(hi))
        
        # Refined with 2 subintervals
        left_mid = 0.5 * (lo + mid)
        right_mid = 0.5 * (mid + hi)
        s_refined = (h / 12) * (
            f_wrapped(lo) + 4 * f_wrapped(left_mid) +
            2 * f_wrapped(mid) + 4 * f_wrapped(right_mid) +
            f_wrapped(hi)
        )
        
        return s, s_refined
    
    def _adaptive(lo: float, hi: float, depth: int) -> float:
        s, s_refined = _simpson_rule(lo, hi)
        
        # Check convergence
        if depth >= max_depth or abs(s_refined - s) <= tol * abs(s_refined) + 1e-15:
            return s_refined
        
        # Recursively refine
        mid = 0.5 * (lo + hi)
        left = _adaptive(lo, mid, depth + 1)
        right = _adaptive(mid, hi, depth + 1)
        return left + right
    
    return _adaptive(a, b, 0)


def linear_interpolate(
    x: float,
    x_points: List[float],
    y_points: List[float],
    extrapolate: bool = False,
) -> float:
    """Piecewise linear interpolation"""
    if len(x_points) != len(y_points):
        raise ValueError("x_points and y_points must have same length")
    if len(x_points) < 2:
        raise ValueError("At least 2 points required")
    
    # Check sorting
    for i in range(1, len(x_points)):
        if x_points[i] < x_points[i-1]:
            raise ValueError("x_points must be sorted")
    
    # Handle endpoints
    if x <= x_points[0]:
        return y_points[0] if not extrapolate else _extrapolate(x, x_points[:2], y_points[:2])
    if x >= x_points[-1]:
        return y_points[-1] if not extrapolate else _extrapolate(x, x_points[-2:], y_points[-2:])
    
    # Binary search
    lo, hi = 0, len(x_points) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x_points[mid] <= x:
            lo = mid
        else:
            hi = mid
    
    x0, x1 = x_points[lo], x_points[lo+1]
    y0, y1 = y_points[lo], y_points[lo+1]
    
    if abs(x1 - x0) < 1e-18:
        return 0.5 * (y0 + y1)
    
    w = (x - x0) / (x1 - x0)
    return y0 + w * (y1 - y0)


def log_interpolate(
    x: float,
    x_points: List[float],
    y_points: List[float],
    extrapolate: bool = False,
) -> float:
    """
    Interpolate with ln(y) linear in x.
    Useful for vapor pressure interpolation.
    """
    if len(x_points) != len(y_points):
        raise ValueError("x_points and y_points must have same length")
    if len(x_points) < 2:
        raise ValueError("At least 2 points required")
    
    # Check if x_points is sorted
    for i in range(1, len(x_points)):
        if x_points[i] < x_points[i-1]:
            raise ValueError("x_points must be sorted in increasing order")
    
    # Convert y to log space
    y_log = [math.log(y) for y in y_points]
    
    # Use linear_interpolate
    log_y = linear_interpolate(x, x_points, y_log, extrapolate)
    
    return math.exp(log_y)


def _extrapolate(x, xp, yp):
    """Linear extrapolation"""
    slope = (yp[1] - yp[0]) / (xp[1] - xp[0]) if abs(xp[1] - xp[0]) > 1e-18 else 0
    return yp[0] + slope * (x - xp[0])