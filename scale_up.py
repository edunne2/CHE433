"""Scale-up equations for extraction towers - Eqs. 27.3-3 to 27.3-6"""

from typing import Dict, Any, Tuple
import math

from bank.core.validation import check_positive


def scheibel_scale_up(
    D1: float,
    D2: float,
    Q1: float,
    HETS1: float,
) -> Dict[str, float]:
    """
    Scale-up Scheibel tower - Eqs. 27.3-3 and 27.3-4
    
    Q2 / Q1 = (D2 / D1)^2.4                                    (27.3-3)
    (HETS)_2 / (HETS)_1 = (D2 / D1)^0.5                         (27.3-4)
    
    Args:
        D1: Small tower diameter (m)
        D2: Large tower diameter (m)
        Q1: Small tower throughput (m³/s)
        HETS1: Small tower HETS (m)
    
    Returns:
        Dictionary with scaled-up Q2 and HETS2
    """
    check_positive("D1", D1)
    check_positive("D2", D2)
    check_positive("Q1", Q1)
    check_positive("HETS1", HETS1)
    
    diameter_ratio = D2 / D1
    
    Q2 = Q1 * (diameter_ratio ** 2.4)
    HETS2 = HETS1 * math.sqrt(diameter_ratio)
    
    return {
        "Q2": Q2,
        "HETS2": HETS2,
        "diameter_ratio": diameter_ratio,
        "Q_ratio": Q2 / Q1,
        "HETS_ratio": HETS2 / HETS1,
    }


def karr_scale_up(
    D1: float,
    D2: float,
    HETS1: float,
    SPM1: float,
) -> Dict[str, float]:
    """
    Scale-up Karr reciprocating-plate tower - Eqs. 27.3-5 and 27.3-6
    
    (HETS_2) / (HETS_1) = (D2 / D1)^0.38                        (27.3-5)
    (SPM_2) / (SPM_1) = (D1 / D2)^0.14                          (27.3-6)
    
    Args:
        D1: Small tower diameter (m)
        D2: Large tower diameter (m)
        HETS1: Small tower HETS (m)
        SPM1: Small tower strokes per minute
    
    Returns:
        Dictionary with scaled-up HETS2 and SPM2
    """
    check_positive("D1", D1)
    check_positive("D2", D2)
    check_positive("HETS1", HETS1)
    check_positive("SPM1", SPM1)
    
    diameter_ratio = D2 / D1
    inverse_ratio = D1 / D2
    
    HETS2 = HETS1 * (diameter_ratio ** 0.38)
    SPM2 = SPM1 * (inverse_ratio ** 0.14)
    
    return {
        "HETS2": HETS2,
        "SPM2": SPM2,
        "diameter_ratio": diameter_ratio,
        "HETS_ratio": HETS2 / HETS1,
        "SPM_ratio": SPM2 / SPM1,
    }


def scale_up_throughput(
    D1: float,
    D2: float,
    VC1: float,
    VD1: float,
) -> Tuple[float, float]:
    """
    Scale up velocities keeping total throughput per unit area constant.
    
    For many tower types, (VC + VD) is kept constant during scale-up.
    
    Args:
        D1: Small tower diameter
        D2: Large tower diameter
        VC1: Small tower continuous phase velocity
        VD1: Small tower dispersed phase velocity
    
    Returns:
        (VC2, VD2) for large tower
    """
    # Keep total superficial velocity constant
    VC2 = VC1
    VD2 = VD1
    return VC2, VD2