# strategy/__init__.py
"""
TITANIUM Bot - Strategy Module
===============================
Backtest ve live trading için paylaşılan strateji fonksiyonları.

Bu modül hem main.py hem de backtest script'leri tarafından
import edilebilir, böylece aynı strateji mantığı kullanılır.
"""

from .indicators import (
    calculate_ema,
    calculate_sma,
    calculate_rsi,
    calculate_atr,
    calculate_adx,
    calculate_bollinger,
)

from .scoring import (
    calculate_reversal_score,
    calculate_rapid_score,
    calculate_range_score,
    is_ranging_market,
)

__all__ = [
    # Indicators
    'calculate_ema',
    'calculate_sma',
    'calculate_rsi',
    'calculate_atr',
    'calculate_adx',
    'calculate_bollinger',
    # Scoring
    'calculate_reversal_score',
    'calculate_rapid_score',
    'calculate_range_score',
    'is_ranging_market',
]
