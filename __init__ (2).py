# bank/separations/distillation/__init__.py
"""Distillation calculations"""

# McCabe-Thiele
from .mccabe_thiele import (
    McCabeThieleSolver,
    McCabeThieleSpec,
    mccabe_thiele_binary,
)

# Shortcut methods (Fenske-Underwood-Gilliland-Kirkbride)
from .shortcut import (
    ShortcutSolver,
    ShortcutSpec,
    fenske_nmin,
    underwood_theta,
    underwood_rmin,
    gilliland_eduljee,
    kirkbride_feed_location,
    shortcut_design,  # Backward compatibility function
)

# Flash and Rayleigh distillation
from .flash import (
    FlashSolver,
    FlashSpec,
    RayleighSolver,
    RayleighSpec,
    solve_flash,      # Backward compatibility
    solve_rayleigh,   # Backward compatibility
)

# Tray efficiency
from .efficiency import (
    TrayEfficiencySpec,
    tray_efficiency_block,
    murphree_vapor_efficiency,
    murphree_liquid_efficiency,
    overall_efficiency_from_murphree,
    overall_efficiency_empirical,
    actual_stages,
)

# Steam distillation
from .steam import (
    SteamDistillationSolver,
    SteamDistillationSpec,
)

# Ponchon-Savarit (enthalpy-based method)
from .ponchon_savarit import (
    PonchonSavaritSolver,
    EnthalpySpec,
)

__all__ = [
    # McCabe-Thiele
    'McCabeThieleSolver',
    'McCabeThieleSpec',
    'mccabe_thiele_binary',
    
    # Shortcut
    'ShortcutSolver',
    'ShortcutSpec',
    'fenske_nmin',
    'underwood_theta',
    'underwood_rmin',
    'gilliland_eduljee',
    'kirkbride_feed_location',
    'shortcut_design',
    
    # Flash
    'FlashSolver',
    'FlashSpec',
    'RayleighSolver',
    'RayleighSpec',
    'solve_flash',
    'solve_rayleigh',
    
    # Efficiency
    'TrayEfficiencySpec',
    'tray_efficiency_block',
    'murphree_vapor_efficiency',
    'murphree_liquid_efficiency',
    'overall_efficiency_from_murphree',
    'overall_efficiency_empirical',
    'actual_stages',
    
    # Steam Distillation
    'SteamDistillationSolver',
    'SteamDistillationSpec',
    
    # Ponchon-Savarit
    'PonchonSavaritSolver',
    'EnthalpySpec',
]