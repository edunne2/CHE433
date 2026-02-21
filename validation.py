# bank/core/validation.py
"""Unified validation for all modules"""
from typing import Union, Sequence, List, Optional, Dict, Any
import math

class ChemEngError(Exception):
    """Base exception for all chemical engineering calculations"""
    pass

class InputError(ChemEngError):
    """Invalid input parameters"""
    pass

class ConvergenceError(ChemEngError):
    """Numerical method failed to converge"""
    pass

class PhaseError(ChemEngError):
    """Invalid phase condition"""
    pass

def check_positive(name: str, value: Union[float, int]) -> float:
    """Check value is positive"""
    v = float(value)
    if v <= 0:
        raise InputError(f"{name} must be > 0, got {v}")
    return v

def check_non_negative(name: str, value: Union[float, int]) -> float:
    """Check value is non-negative"""
    v = float(value)
    if v < 0:
        raise InputError(f"{name} must be >= 0, got {v}")
    return v

def check_in_closed_01(name: str, value: float) -> float:
    """Check value in [0, 1]"""
    v = float(value)
    if not (0.0 <= v <= 1.0):
        raise InputError(f"{name} must be in [0, 1], got {v}")
    return v

def check_in_open_01(name: str, value: float) -> float:
    """Check value in [0, 1)"""
    v = float(value)
    if not (0.0 <= v < 1.0):
        raise InputError(f"{name} must satisfy 0 <= {name} < 1, got {v}")
    return v

def normalize_composition(
    z: Sequence[float],
    name: str = "composition",
    tolerance: float = 1e-10
) -> List[float]:
    """Normalize composition to sum to 1.0"""
    if len(z) < 2:
        raise InputError(f"{name} must have length >= 2, got {len(z)}")
    
    z_float = [float(v) for v in z]
    for i, v in enumerate(z_float):
        if v < -tolerance:
            raise InputError(f"{name}[{i}] cannot be negative, got {v}")
        z_float[i] = max(0.0, v)
    
    total = sum(z_float)
    if total <= 0:
        raise InputError(f"{name} sum must be > 0, got {total}")
    
    if abs(total - 1.0) <= tolerance:
        return z_float
    
    return [v / total for v in z_float]

def check_composition_match(
    compositions: Dict[str, Sequence[float]],
    expected_length: Optional[int] = None
) -> int:
    """Check all compositions have same length"""
    lengths = {name: len(comp) for name, comp in compositions.items()}
    unique = set(lengths.values())
    
    if len(unique) > 1:
        raise InputError(f"Composition length mismatch: {lengths}")
    
    n = next(iter(unique))
    if expected_length is not None and n != expected_length:
        raise InputError(f"Expected length {expected_length}, got {n}")
    
    return n