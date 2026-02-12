"""
TITANIUM Bot - Signals Package
==============================
Sinyal stratejileri modülü.
"""

from signals.reversal import (
    calculate_momentum_reversal,
    check_rsi_divergence,
    check_volatility_spike,
    calculate_reversal_score
)

from signals.range import (
    is_ranging_market,
    calculate_range_score
)

from signals.rapid import (
    detect_flash_move,
    detect_volume_spike,
    detect_wick_rejection,
    detect_rsi_extreme_bounce,
    calculate_rapid_score
)

from signals.mean_reversion import (
    calculate_mean_reversion_signals,
    execute_strategy_switch
)

__all__ = [
    'calculate_momentum_reversal',
    'check_rsi_divergence', 
    'check_volatility_spike',
    'calculate_reversal_score',
    'is_ranging_market',
    'calculate_range_score',
    'detect_flash_move',
    'detect_volume_spike',
    'detect_wick_rejection',
    'detect_rsi_extreme_bounce',
    'calculate_rapid_score',
    'calculate_mean_reversion_signals',
    'execute_strategy_switch',
]

