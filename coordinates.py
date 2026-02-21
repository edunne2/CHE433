"""Utilities for solute-free coordinates (X, Y) in absorption calculations"""

from typing import Union
import numpy as np

from bank.core.validation import check_in_closed_01


def Y_from_y(y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert mole fraction y to mole ratio Y = y/(1-y)"""
    if isinstance(y, np.ndarray):
        return np.where(y >= 1.0, np.inf, y / (1 - y))
    else:
        if y >= 1.0:
            return float('inf')
        check_in_closed_01("y", y)
        return y / (1 - y)


def X_from_x(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert mole fraction x to mole ratio X = x/(1-x)"""
    if isinstance(x, np.ndarray):
        return np.where(x >= 1.0, np.inf, x / (1 - x))
    else:
        if x >= 1.0:
            return float('inf')
        check_in_closed_01("x", x)
        return x / (1 - x)


def y_from_Y(Y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert mole ratio Y to mole fraction y = Y/(1+Y)"""
    if isinstance(Y, np.ndarray):
        return Y / (1 + Y)
    else:
        if Y < 0:
            raise ValueError(f"Y must be >= 0, got {Y}")
        return Y / (1 + Y)


def x_from_X(X: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert mole ratio X to mole fraction x = X/(1+X)"""
    if isinstance(X, np.ndarray):
        return X / (1 + X)
    else:
        if X < 0:
            raise ValueError(f"X must be >= 0, got {X}")
        return X / (1 + X)