"""Multicomponent distillation calculations - Eqs. 26.8-1 to 26.8-18"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Callable
import math
import numpy as np

from bank.core.validation import (
    check_positive, check_in_closed_01, normalize_composition, ChemEngError
)
from bank.core.numerical import bisection


@dataclass
class Component:
    """Component properties for multicomponent distillation"""
    name: str
    xiF: float      # Feed composition
    Ki_func: Callable  # K = f(T) function
    alpha: Optional[float] = None  # Relative volatility (if constant)


class MulticomponentEquilibrium:
    """
    Vapor-liquid equilibrium for multicomponent systems.
    
    Eqs. 26.8-1 to 26.8-4:
    yi = Ki xi                                             (26.8-1, 26.8-2, 26.8-3)
    αi = Ki / KC, αC = 1.0                                  (26.8-4)
    """
    
    def __init__(self, components: List[Component], base_component_index: int):
        self.components = components
        self.base_idx = base_component_index
        self.n_comp = len(components)
    
    def K_values(self, T: float) -> List[float]:
        """Get K values for all components at temperature T"""
        return [comp.Ki_func(T) for comp in self.components]
    
    def alpha_values(self, T: float) -> List[float]:
        """Get relative volatilities at temperature T - Eq. 26.8-4"""
        K = self.K_values(T)
        K_base = K[self.base_idx]
        return [Ki / K_base for Ki in K]
    
    def bubble_point(self, x: List[float], T_guess: float = 100.0) -> Dict[str, Any]:
        """
        Calculate bubble point temperature - Eq. 26.8-5
        
        Σ yi = Σ Ki xi = Kc Σ αi xi = 1.0
        """
        x_norm = normalize_composition(x, "x")
        
        def f(T: float) -> float:
            alpha = self.alpha_values(T)
            sum_alpha_x = sum(alpha[i] * x_norm[i] for i in range(self.n_comp))
            Kc = 1.0 / sum_alpha_x
            # Get actual Kc at this T
            K_actual = self.K_values(T)[self.base_idx]
            return K_actual - Kc
        
        T_bubble = bisection(f, T_guess - 50, T_guess + 50, expand_bracket=True)
        
        # Calculate vapor compositions - Eq. 26.8-6
        alpha = self.alpha_values(T_bubble)
        K = self.K_values(T_bubble)
        sum_alpha_x = sum(alpha[i] * x_norm[i] for i in range(self.n_comp))
        y = [(alpha[i] * x_norm[i]) / sum_alpha_x for i in range(self.n_comp)]
        
        return {
            "T": T_bubble,
            "y": y,
            "K": K,
            "alpha": alpha,
        }
    
    def dew_point(self, y: List[float], T_guess: float = 100.0) -> Dict[str, Any]:
        """
        Calculate dew point temperature - Eq. 26.8-7
        
        Σ xi = Σ (yi / Ki) = (1/Kc) Σ (yi / αi) = 1.0
        """
        y_norm = normalize_composition(y, "y")
        
        def f(T: float) -> float:
            alpha = self.alpha_values(T)
            sum_y_over_alpha = sum(y_norm[i] / alpha[i] for i in range(self.n_comp))
            Kc = 1.0 / sum_y_over_alpha
            K_actual = self.K_values(T)[self.base_idx]
            return K_actual - Kc
        
        T_dew = bisection(f, T_guess - 50, T_guess + 50, expand_bracket=True)
        
        # Calculate liquid compositions - Eq. 26.8-8
        alpha = self.alpha_values(T_dew)
        K = self.K_values(T_dew)
        sum_y_over_alpha = sum(y_norm[i] / alpha[i] for i in range(self.n_comp))
        x = [(y_norm[i] / alpha[i]) / sum_y_over_alpha for i in range(self.n_comp)]
        
        return {
            "T": T_dew,
            "x": x,
            "K": K,
            "alpha": alpha,
        }
    
    def flash(
        self,
        z: List[float],
        f: float,
        T_guess: float = 100.0
    ) -> Dict[str, Any]:
        """
        Flash distillation for multicomponent mixture - Eqs. 26.8-9 to 26.8-11
        
        yi = ((f-1)xi)/f + xiF/f                             (26.8-9)
        yi = KC αi xi = ((f-1)xi)/f + xiF/f                   (26.8-10)
        Σ xi = Σ {xiF / (f(KCαi - 1) + 1)} = 1.0             (26.8-11)
        """
        z_norm = normalize_composition(z, "z")
        
        def equation(T: float) -> float:
            K = self.K_values(T)
            alpha = self.alpha_values(T)
            Kc = K[self.base_idx]
            
            sum_x = 0.0
            for i in range(self.n_comp):
                denominator = f * (Kc * alpha[i] - 1.0) + 1.0
                sum_x += z_norm[i] / denominator
            
            return sum_x - 1.0
        
        T_flash = bisection(equation, T_guess - 50, T_guess + 50, expand_bracket=True)
        
        # Calculate compositions
        K = self.K_values(T_flash)
        alpha = self.alpha_values(T_flash)
        Kc = K[self.base_idx]
        
        x = []
        y = []
        for i in range(self.n_comp):
            denominator = f * (Kc * alpha[i] - 1.0) + 1.0
            xi = z_norm[i] / denominator
            x.append(xi)
            y.append(K[i] * xi)
        
        # Normalize
        sum_x = sum(x)
        sum_y = sum(y)
        x = [xi / sum_x for xi in x]
        y = [yi / sum_y for yi in y]
        
        return {
            "T": T_flash,
            "x": x,
            "y": y,
            "K": K,
            "alpha": alpha,
        }


def fenske_multicomponent(
    xLD: float, xLW: float,  # Light key in distillate and bottoms
    xHD: float, xHW: float,  # Heavy key in distillate and bottoms
    alpha_L_av: float        # Average relative volatility of light key
) -> float:
    """
    Fenske equation for minimum stages in multicomponent distillation - Eq. 26.8-12
    
    Nm = log [(xLD D / xHD D)(xHW W / xLW W)] / log αL,av
    """
    check_in_closed_01("xLD", xLD)
    check_in_closed_01("xLW", xLW)
    check_in_closed_01("xHD", xHD)
    check_in_closed_01("xHW", xHW)
    check_positive("alpha_L_av", alpha_L_av)
    
    numerator = math.log((xLD / xHD) * (xHW / xLW))
    denominator = math.log(alpha_L_av)
    
    return numerator / denominator


def component_distribution(
    alpha_i_av: float,
    Nm: float,
    xHD: float,
    xHW: float,
    D: float,
    W: float
) -> Tuple[float, float]:
    """
    Distribution of non-key components at total reflux - Eq. 26.8-14
    
    (xiD D) / (xiW W) = (αi,av)^Nm (xHD D) / (xHW W)
    
    Returns:
        xiD_D, xiW_W (moles of component in distillate and bottoms)
    """
    check_positive("alpha_i_av", alpha_i_av)
    check_positive("Nm", Nm)
    check_in_closed_01("xHD", xHD)
    check_in_closed_01("xHW", xHW)
    check_positive("D", D)
    check_positive("W", W)
    
    ratio = (alpha_i_av ** Nm) * (xHD * D) / (xHW * W)
    
    # Need overall balance to solve for individual amounts
    # This requires additional information
    return ratio


def underwood_equations(
    alpha: List[float],
    xiF: List[float],
    xiD: List[float],
    q: float
) -> Dict[str, float]:
    """
    Underwood equations for minimum reflux - Eqs. 26.8-15 and 26.8-16
    
    1 - q = Σ (αi xiF) / (αi - θ)                            (26.8-15)
    Rm + 1 = Σ (αi xiD) / (αi - θ)                           (26.8-16)
    """
    n = len(alpha)
    if len(xiF) != n or len(xiD) != n:
        raise ValueError("All lists must have same length")
    
    # Find root θ between α_HK and α_LK
    # This is a simplified implementation
    def f(theta: float) -> float:
        sum_val = 0.0
        for i in range(n):
            denominator = alpha[i] - theta
            if abs(denominator) < 1e-12:
                return float('inf')
            sum_val += alpha[i] * xiF[i] / denominator
        return (1.0 - q) - sum_val
    
    # Search for root
    theta = None
    alpha_sorted = sorted(alpha)
    for i in range(len(alpha_sorted) - 1):
        a1, a2 = alpha_sorted[i], alpha_sorted[i+1]
        if f(a1 + 1e-6) * f(a2 - 1e-6) <= 0:
            theta = bisection(f, a1 + 1e-6, a2 - 1e-6)
            break
    
    if theta is None:
        raise ChemEngError("Could not find root θ")
    
    # Calculate Rm
    Rm1 = -1.0
    for i in range(n):
        denominator = alpha[i] - theta
        Rm1 += alpha[i] * xiD[i] / denominator
    
    return {
        "theta": theta,
        "Rm": Rm1,
    }


def erbar_maddox(
    R: float,
    Rm: float,
    Nm: float
) -> float:
    """
    Erbar-Maddox correlation for number of stages - Fig. 26.8-3
    
    This is an approximation of the graphical correlation.
    """
    X = Rm / (Rm + 1.0)
    Y = R / (R + 1.0)
    
    # Simplified correlation - in practice would use table lookup
    # This is a rough approximation
    Nm_over_N = 0.5 + 0.3 * (Y - X)
    Nm_over_N = max(0.2, min(0.8, Nm_over_N))
    
    return Nm / Nm_over_N


def kirkbride_feed_location(
    xHF: float,  # Heavy key in feed
    xLF: float,  # Light key in feed
    xLW: float,  # Light key in bottoms
    xHD: float,  # Heavy key in distillate
    W: float,
    D: float
) -> float:
    """
    Kirkbride equation for feed plate location - Eq. 26.8-18
    
    log (Ne / Ns) = 0.206 log [(xHF / xLF) (W/D) (xLW / xHD)]
    """
    check_in_closed_01("xHF", xHF)
    check_in_closed_01("xLF", xLF)
    check_in_closed_01("xLW", xLW)
    check_in_closed_01("xHD", xHD)
    check_positive("W", W)
    check_positive("D", D)
    
    arg = (xHF / xLF) * (W / D) * (xLW / xHD)
    log_arg = math.log10(arg)
    
    Ne_over_Ns = 10.0 ** (0.206 * log_arg)
    return Ne_over_Ns


class MulticomponentDistillation:
    """
    Complete multicomponent distillation design using shortcut methods.
    """
    
    def __init__(
        self,
        components: List[Component],
        base_component_index: int,
        feed_composition: List[float],
        q: float,
        light_key_index: int,
        heavy_key_index: int,
        recovery_LK_in_distillate: float,
        recovery_HK_in_bottoms: float,
        F: float = 100.0
    ):
        self.components = components
        self.base_idx = base_component_index
        self.feed = feed_composition
        self.q = q
        self.LK_idx = light_key_index
        self.HK_idx = heavy_key_index
        self.rec_LK = recovery_LK_in_distillate
        self.rec_HK = recovery_HK_in_bottoms
        self.F = F
        
        self.eq = MulticomponentEquilibrium(components, base_component_index)
        self._solve_material_balance()
    
    def _solve_material_balance(self):
        """Initial material balance assuming no trace components"""
        n = len(self.components)
        
        # First estimate of D and W
        # Using key components only
        LK_feed = self.F * self.feed[self.LK_idx]
        HK_feed = self.F * self.feed[self.HK_idx]
        
        LK_dist = self.rec_LK * LK_feed
        LK_bott = LK_feed - LK_dist
        
        HK_bott = self.rec_HK * HK_feed
        HK_dist = HK_feed - HK_bott
        
        # Assume all lighter than LK go to distillate
        # All heavier than HK go to bottoms
        self.D = LK_dist + HK_dist
        self.W = LK_bott + HK_bott
        
        for i in range(n):
            if i < self.LK_idx:  # Lighter than LK
                self.D += self.F * self.feed[i]
            elif i > self.HK_idx:  # Heavier than HK
                self.W += self.F * self.feed[i]
        
        # Distillate and bottoms compositions (initial)
        self.xD = [0.0] * n
        self.xW = [0.0] * n
        
        for i in range(n):
            if i < self.LK_idx:
                self.xD[i] = self.F * self.feed[i] / self.D
            elif i == self.LK_idx:
                self.xD[i] = LK_dist / self.D
                self.xW[i] = LK_bott / self.W
            elif i == self.HK_idx:
                self.xD[i] = HK_dist / self.D
                self.xW[i] = HK_bott / self.W
            else:  # i > HK_idx
                self.xW[i] = self.F * self.feed[i] / self.W
    
    def calculate_top_bottom_temperatures(self) -> Dict[str, Any]:
        """Calculate dew point of distillate and bubble point of bottoms"""
        top = self.eq.dew_point(self.xD)
        bottom = self.eq.bubble_point(self.xW)
        
        return {
            "top_temperature": top["T"],
            "bottom_temperature": bottom["T"],
            "top_composition": top["x"],
            "bottom_composition": bottom["y"],
        }
    
    def minimum_stages_total_reflux(self) -> float:
        """Calculate minimum stages using Fenske equation"""
        # Average alpha for light key
        temps = self.calculate_top_bottom_temperatures()
        alpha_top = self.eq.alpha_values(temps["top_temperature"])[self.LK_idx]
        alpha_bottom = self.eq.alpha_values(temps["bottom_temperature"])[self.LK_idx]
        alpha_av = math.sqrt(alpha_top * alpha_bottom)
        
        return fenske_multicomponent(
            self.xD[self.LK_idx], self.xW[self.LK_idx],
            self.xD[self.HK_idx], self.xW[self.HK_idx],
            alpha_av
        )
    
    def minimum_reflux(self) -> float:
        """Calculate minimum reflux using Underwood equations"""
        # Average temperature for alpha
        temps = self.calculate_top_bottom_temperatures()
        T_avg = (temps["top_temperature"] + temps["bottom_temperature"]) / 2.0
        alpha = self.eq.alpha_values(T_avg)
        
        result = underwood_equations(alpha, self.feed, self.xD, self.q)
        return result["Rm"]
    
    def kirkbride_feed_tray(self, N: float) -> Dict[str, float]:
        """Calculate feed tray location using Kirkbride equation"""
        Ne_over_Ns = kirkbride_feed_location(
            self.feed[self.HK_idx],
            self.feed[self.LK_idx],
            self.xW[self.LK_idx],
            self.xD[self.HK_idx],
            self.W,
            self.D
        )
        
        Ns = N / (1.0 + Ne_over_Ns)
        Ne = N - Ns
        
        return {
            "Ne_over_Ns": Ne_over_Ns,
            "Ne": Ne,
            "Ns": Ns,
            "feed_tray_from_top": Ne,
        }
    
    def design(self, R_factor: float = 1.3) -> Dict[str, Any]:
        """Complete shortcut design"""
        Nm = self.minimum_stages_total_reflux()
        Rm = self.minimum_reflux()
        R_op = R_factor * Rm
        
        # Erbar-Maddox for actual stages
        N = erbar_maddox(R_op, Rm, Nm)
        
        # Feed tray location
        feed = self.kirkbride_feed_tray(N)
        
        return {
            "flows": {"D": self.D, "W": self.W},
            "compositions": {
                "distillate": self.xD,
                "bottoms": self.xW,
            },
            "temperatures": self.calculate_top_bottom_temperatures(),
            "minimum_stages": Nm,
            "minimum_reflux": Rm,
            "operating_reflux": R_op,
            "actual_stages": N,
            "feed_tray": feed,
        }