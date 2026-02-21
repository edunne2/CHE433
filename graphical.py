# bank/separations/absorption/graphical.py
"""Basic absorption calculation methods"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import math

from bank.core.validation import check_positive, check_in_closed_01


def kremser_stages(
    y_in: float,
    y_out: float,
    x_in: float,
    m: float,
    A: float
) -> float:
    """
    Kremser equation for number of stages in absorption.
    
    N = log[ ((y_in - m*x_in)/(y_out - m*x_in)) * (1 - 1/A) + 1/A ] / log A
    
    Args:
        y_in: Inlet gas mole fraction
        y_out: Outlet gas mole fraction
        x_in: Inlet liquid mole fraction
        m: Slope of equilibrium line (y = m*x)
        A: Absorption factor
    
    Returns:
        Number of theoretical stages
    """
    if A <= 0 or abs(A - 1.0) < 1e-10:
        return float('inf')
    
    numerator = ((y_in - m * x_in) / (y_out - m * x_in)) * (1 - 1/A) + 1/A
    
    if numerator <= 0:
        return float('inf')
    
    return math.log(numerator) / math.log(A)


def absorption_factor(
    L: float,
    G: float,
    m: float
) -> float:
    """
    Calculate absorption factor A = L/(m*G)
    
    Args:
        L: Liquid flow rate
        G: Gas flow rate
        m: Slope of equilibrium line
    
    Returns:
        Absorption factor A
    """
    return L / (m * G)