"""
TITANIUM Bot - Shared Scoring Functions
=========================================
Backtest ve live trading için paylaşılan skorlama fonksiyonları.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional

from .indicators import (
    calculate_rsi,
    calculate_atr,
    calculate_adx,
    calculate_bollinger,
)


# ==========================================
# REVERSAL DETECTION
# ==========================================

def calculate_momentum_reversal(
    df: pd.DataFrame,
    lookback: int = 5,
    threshold: float = 2.0
) -> Tuple[Optional[str], float]:
    """
    Son X mumda ani momentum değişimi var mı?
    
    Returns:
        (reversal_type, change_pct)
    """
    closes = df['close'].tail(lookback)
    
    curr_close = closes.iloc[-1]
    avg_prev = closes.iloc[:-1].mean()
    change_pct = ((curr_close - avg_prev) / avg_prev) * 100
    
    if change_pct > threshold:
        return 'REVERSAL_UP', change_pct
    elif change_pct < -threshold:
        return 'REVERSAL_DOWN', change_pct
    return None, change_pct


def check_rsi_divergence(
    df: pd.DataFrame,
    lookback: int = 14
) -> Tuple[Optional[str], float]:
    """
    RSI Divergence tespit et.
    
    Returns:
        (divergence_type, strength)
    """
    price = df['close'].tail(lookback)
    rsi = calculate_rsi(df['close']).tail(lookback)
    
    curr_rsi = rsi.iloc[-1]
    
    # Bullish Divergence
    price_falling = price.iloc[-1] < price.iloc[-3]
    rsi_rising = rsi.iloc[-1] > rsi.iloc[-3]
    
    if price_falling and rsi_rising and curr_rsi < 40:
        strength = min(100, abs(rsi.iloc[-1] - rsi.iloc[-3]) * 5)
        return 'BULLISH_DIV', strength
    
    # Bearish Divergence
    price_rising = price.iloc[-1] > price.iloc[-3]
    rsi_falling = rsi.iloc[-1] < rsi.iloc[-3]
    
    if price_rising and rsi_falling and curr_rsi > 60:
        strength = min(100, abs(rsi.iloc[-3] - rsi.iloc[-1]) * 5)
        return 'BEARISH_DIV', strength
    
    return None, 0


def check_volatility_spike(
    df: pd.DataFrame,
    period: int = 14,
    multiplier: float = 2.0
) -> Tuple[Optional[str], float]:
    """
    Volatilite patlaması tespit et.
    
    Returns:
        (spike_type, atr_ratio)
    """
    atr = calculate_atr(df, period)
    avg_atr = atr.rolling(50).mean().iloc[-1]
    curr_atr = atr.iloc[-1]
    
    if pd.isna(avg_atr) or avg_atr == 0:
        return None, 1.0
    
    atr_ratio = curr_atr / avg_atr
    
    if atr_ratio > multiplier:
        is_bullish_candle = df['close'].iloc[-1] > df['open'].iloc[-1]
        if is_bullish_candle:
            return 'SPIKE_UP', atr_ratio
        else:
            return 'SPIKE_DOWN', atr_ratio
    
    return None, atr_ratio


def calculate_reversal_score(df: pd.DataFrame) -> Tuple[int, int, List[str]]:
    """
    Tüm reversal indikatörlerini birleştirerek skor hesapla.
    
    Returns:
        (long_score, short_score, details)
    """
    long_score = 0
    short_score = 0
    details = []
    
    # 1. Momentum Reversal (max 10 puan)
    mom_type, mom_pct = calculate_momentum_reversal(df, lookback=5, threshold=1.5)
    if mom_type == 'REVERSAL_UP':
        score = min(10, int(abs(mom_pct) * 3))
        long_score += score
        details.append(f"MOM↑:{score}")
    elif mom_type == 'REVERSAL_DOWN':
        score = min(10, int(abs(mom_pct) * 3))
        short_score += score
        details.append(f"MOM↓:{score}")
    
    # 2. RSI Divergence (max 12 puan)
    div_type, div_strength = check_rsi_divergence(df, lookback=14)
    if div_type == 'BULLISH_DIV':
        score = min(12, int(div_strength / 8))
        long_score += score
        details.append(f"DIV↑:{score}")
    elif div_type == 'BEARISH_DIV':
        score = min(12, int(div_strength / 8))
        short_score += score
        details.append(f"DIV↓:{score}")
    
    # 3. Volatility Spike (max 8 puan)
    spike_type, atr_ratio = check_volatility_spike(df, period=14, multiplier=1.8)
    if spike_type == 'SPIKE_UP':
        score = min(8, int((atr_ratio - 1) * 5))
        long_score += score
        details.append(f"VOL↑:{score}")
    elif spike_type == 'SPIKE_DOWN':
        score = min(8, int((atr_ratio - 1) * 5))
        short_score += score
        details.append(f"VOL↓:{score}")
    
    return long_score, short_score, details


# ==========================================
# RAPID REVERSAL
# ==========================================

def detect_flash_move(
    df: pd.DataFrame,
    threshold_pct: float = 3.0,
    lookback: int = 3
) -> Tuple[Optional[str], float]:
    """Ani fiyat hareketi tespiti."""
    if len(df) < lookback + 1:
        return None, 0
    
    closes = df['close'].tail(lookback + 1)
    start_price = closes.iloc[0]
    end_price = closes.iloc[-1]
    
    change_pct = ((end_price - start_price) / start_price) * 100
    
    last_candle_bullish = df['close'].iloc[-1] > df['open'].iloc[-1]
    
    if change_pct < -threshold_pct and last_candle_bullish:
        return 'FLASH_UP', abs(change_pct)
    
    if change_pct > threshold_pct and not last_candle_bullish:
        return 'FLASH_DOWN', abs(change_pct)
    
    return None, abs(change_pct)


def detect_volume_spike(
    df: pd.DataFrame,
    multiplier: float = 3.0,
    lookback: int = 20
) -> Tuple[Optional[str], float]:
    """Hacim patlaması tespiti."""
    if len(df) < lookback:
        return None, 1.0
    
    vol_sma = df['volume'].tail(lookback).mean()
    curr_vol = df['volume'].iloc[-1]
    
    if vol_sma == 0:
        return None, 1.0
    
    vol_ratio = curr_vol / vol_sma
    
    if vol_ratio >= multiplier:
        is_bullish = df['close'].iloc[-1] > df['open'].iloc[-1]
        if is_bullish:
            return 'VOL_SPIKE_UP', vol_ratio
        else:
            return 'VOL_SPIKE_DOWN', vol_ratio
    
    return None, vol_ratio


def detect_wick_rejection(
    df: pd.DataFrame,
    wick_body_ratio: float = 2.0
) -> Tuple[Optional[str], float]:
    """Fitil reddi tespiti."""
    row = df.iloc[-1]
    
    body = abs(row['close'] - row['open'])
    upper_wick = row['high'] - max(row['close'], row['open'])
    lower_wick = min(row['close'], row['open']) - row['low']
    
    if body == 0:
        body = 0.0001
    
    upper_ratio = upper_wick / body
    lower_ratio = lower_wick / body
    
    if lower_ratio >= wick_body_ratio and lower_ratio > upper_ratio:
        return 'WICK_UP', lower_ratio
    
    if upper_ratio >= wick_body_ratio and upper_ratio > lower_ratio:
        return 'WICK_DOWN', upper_ratio
    
    return None, max(upper_ratio, lower_ratio)


def detect_rsi_extreme_bounce(
    df: pd.DataFrame,
    oversold: int = 25,
    overbought: int = 75
) -> Tuple[Optional[str], float]:
    """RSI aşırı bölgeden dönüş tespiti."""
    if len(df) < 15:
        return None, 50
    
    rsi = calculate_rsi(df['close'])
    curr_rsi = rsi.iloc[-1]
    prev_rsi = rsi.iloc[-2]
    
    is_bullish = df['close'].iloc[-1] > df['open'].iloc[-1]
    
    if prev_rsi < oversold and curr_rsi > prev_rsi and is_bullish:
        return 'RSI_BOUNCE_UP', curr_rsi
    
    if prev_rsi > overbought and curr_rsi < prev_rsi and not is_bullish:
        return 'RSI_BOUNCE_DOWN', curr_rsi
    
    return None, curr_rsi


def calculate_rapid_score(
    df: pd.DataFrame
) -> Tuple[int, int, List[str], List[str]]:
    """
    Rapid Reversal için skor hesapla (0-100).
    
    Returns:
        (long_score, short_score, details, triggers)
    """
    long_score = 0
    short_score = 0
    details = []
    triggers = []
    
    # 1. Flash Move (25 puan)
    flash_type, flash_pct = detect_flash_move(df, threshold_pct=3.0)
    if flash_type == 'FLASH_UP':
        score = min(25, int(flash_pct * 5))
        long_score += score
        details.append(f"Flash:{score}")
        triggers.append(f"Flash Move {flash_pct:.1f}%")
    elif flash_type == 'FLASH_DOWN':
        score = min(25, int(flash_pct * 5))
        short_score += score
        details.append(f"Flash:{score}")
        triggers.append(f"Flash Move {flash_pct:.1f}%")
    
    # 2. Volume Spike (25 puan)
    vol_type, vol_ratio = detect_volume_spike(df, multiplier=3.0)
    if vol_type == 'VOL_SPIKE_UP':
        score = min(25, int((vol_ratio - 1) * 8))
        long_score += score
        details.append(f"Vol:{score}")
        triggers.append(f"Volume {vol_ratio:.1f}x")
    elif vol_type == 'VOL_SPIKE_DOWN':
        score = min(25, int((vol_ratio - 1) * 8))
        short_score += score
        details.append(f"Vol:{score}")
        triggers.append(f"Volume {vol_ratio:.1f}x")
    
    # 3. RSI Extreme Bounce (20 puan)
    rsi_type, rsi_val = detect_rsi_extreme_bounce(df)
    if rsi_type == 'RSI_BOUNCE_UP':
        long_score += 20
        details.append("RSI:20")
        triggers.append(f"RSI Bounce ({rsi_val:.0f})")
    elif rsi_type == 'RSI_BOUNCE_DOWN':
        short_score += 20
        details.append("RSI:20")
        triggers.append(f"RSI Bounce ({rsi_val:.0f})")
    
    # 4. ATR Explosion (15 puan)
    spike_type, atr_ratio = check_volatility_spike(df, period=14, multiplier=2.5)
    if spike_type == 'SPIKE_UP':
        score = min(15, int((atr_ratio - 1) * 6))
        long_score += score
        details.append(f"ATR:{score}")
    elif spike_type == 'SPIKE_DOWN':
        score = min(15, int((atr_ratio - 1) * 6))
        short_score += score
        details.append(f"ATR:{score}")
    
    # 5. Wick Rejection (15 puan)
    wick_type, wick_ratio = detect_wick_rejection(df, wick_body_ratio=2.0)
    if wick_type == 'WICK_UP':
        score = min(15, int(wick_ratio * 3))
        long_score += score
        details.append(f"Wick:{score}")
        triggers.append("Wick Rejection")
    elif wick_type == 'WICK_DOWN':
        score = min(15, int(wick_ratio * 3))
        short_score += score
        details.append(f"Wick:{score}")
        triggers.append("Wick Rejection")
    
    return long_score, short_score, details, triggers


# ==========================================
# RANGE MARKET
# ==========================================

def is_ranging_market(
    df: pd.DataFrame,
    adx_threshold: int = 20
) -> Tuple[bool, Dict]:
    """
    Düz piyasa tespiti.
    
    Returns:
        (is_ranging, details)
    """
    adx = calculate_adx(df).iloc[-1]
    atr_series = calculate_atr(df)
    atr = atr_series.iloc[-1]
    atr_sma = atr_series.rolling(50).mean().iloc[-1]
    
    if pd.isna(adx) or pd.isna(atr):
        return False, {'adx': 0, 'atr_ratio': 1, 'bb_width': 0, 'criteria_met': 0}
    
    lower, mid, upper = calculate_bollinger(df)
    bb_width = ((upper.iloc[-1] - lower.iloc[-1]) / mid.iloc[-1]) * 100 if mid.iloc[-1] > 0 else 0
    bb_width_avg = ((upper - lower) / mid * 100).rolling(50).mean().iloc[-1]
    
    is_low_adx = adx < adx_threshold
    is_low_volatility = atr < atr_sma * 0.9 if not pd.isna(atr_sma) and atr_sma > 0 else False
    is_bb_tight = bb_width < bb_width_avg * 0.9 if not pd.isna(bb_width_avg) and bb_width_avg > 0 else False
    
    criteria_met = sum([is_low_adx, is_low_volatility, is_bb_tight])
    is_ranging = criteria_met >= 2
    
    details = {
        'adx': adx,
        'atr_ratio': atr / atr_sma if atr_sma and atr_sma > 0 else 1,
        'bb_width': bb_width,
        'is_low_adx': is_low_adx,
        'is_low_volatility': is_low_volatility,
        'is_bb_tight': is_bb_tight,
        'criteria_met': criteria_met
    }
    
    return is_ranging, details


def calculate_range_score(
    df: pd.DataFrame
) -> Tuple[int, int, List[str], List[str], Dict]:
    """
    Range Trading için sinyal skoru (0-60).
    
    Returns:
        (long_score, short_score, long_breakdown, short_breakdown, tp_sl_info)
    """
    long_score = 0
    short_score = 0
    long_breakdown = []
    short_breakdown = []
    
    price = df['close'].iloc[-1]
    
    # 1. Bollinger Band Position (max 20 puan)
    lower, mid, upper = calculate_bollinger(df)
    bb_lower = lower.iloc[-1]
    bb_mid = mid.iloc[-1]
    bb_upper = upper.iloc[-1]
    
    if bb_upper != bb_lower:
        bb_position = (price - bb_lower) / (bb_upper - bb_lower)
    else:
        bb_position = 0.5
    
    if bb_position < 0.15:
        long_score += 20
        long_breakdown.append("BB:20")
    elif bb_position < 0.25:
        long_score += 15
        long_breakdown.append("BB:15")
    elif bb_position < 0.35:
        long_score += 8
        long_breakdown.append("BB:8")
    
    if bb_position > 0.85:
        short_score += 20
        short_breakdown.append("BB:20")
    elif bb_position > 0.75:
        short_score += 15
        short_breakdown.append("BB:15")
    elif bb_position > 0.65:
        short_score += 8
        short_breakdown.append("BB:8")
    
    # 2. RSI (max 15 puan)
    rsi = calculate_rsi(df['close']).iloc[-1]
    
    if rsi < 30:
        long_score += 15
        long_breakdown.append("RSI:15")
    elif rsi < 35:
        long_score += 12
        long_breakdown.append("RSI:12")
    elif rsi < 40:
        long_score += 8
        long_breakdown.append("RSI:8")
    
    if rsi > 70:
        short_score += 15
        short_breakdown.append("RSI:15")
    elif rsi > 65:
        short_score += 12
        short_breakdown.append("RSI:12")
    elif rsi > 60:
        short_score += 8
        short_breakdown.append("RSI:8")
    
    # 3. SMA20 Deviation (max 12 puan)
    sma20 = df['close'].rolling(20).mean().iloc[-1]
    deviation_pct = ((price - sma20) / sma20) * 100 if sma20 and sma20 > 0 else 0
    
    if deviation_pct < -2.0:
        long_score += 12
        long_breakdown.append("DEV:12")
    elif deviation_pct < -1.5:
        long_score += 8
        long_breakdown.append("DEV:8")
    elif deviation_pct < -1.0:
        long_score += 5
        long_breakdown.append("DEV:5")
    
    if deviation_pct > 2.0:
        short_score += 12
        short_breakdown.append("DEV:12")
    elif deviation_pct > 1.5:
        short_score += 8
        short_breakdown.append("DEV:8")
    elif deviation_pct > 1.0:
        short_score += 5
        short_breakdown.append("DEV:5")
    
    # 4. Wick Rejection (max 8 puan)
    row = df.iloc[-1]
    body = abs(row['close'] - row['open'])
    upper_wick = row['high'] - max(row['close'], row['open'])
    lower_wick = min(row['close'], row['open']) - row['low']
    
    if body > 0:
        lower_ratio = lower_wick / body
        upper_ratio = upper_wick / body
        
        if lower_ratio > 2.0:
            long_score += 8
            long_breakdown.append("WICK:8")
        elif lower_ratio > 1.5:
            long_score += 5
            long_breakdown.append("WICK:5")
        
        if upper_ratio > 2.0:
            short_score += 8
            short_breakdown.append("WICK:8")
        elif upper_ratio > 1.5:
            short_score += 5
            short_breakdown.append("WICK:5")
    
    # 5. Stochastic-like check (max 5 puan)
    low_14 = df['low'].tail(14).min()
    high_14 = df['high'].tail(14).max()
    
    if high_14 != low_14:
        stoch_k = ((price - low_14) / (high_14 - low_14)) * 100
    else:
        stoch_k = 50
    
    if stoch_k < 20:
        long_score += 5
        long_breakdown.append("STOCH:5")
    elif stoch_k < 30:
        long_score += 3
        long_breakdown.append("STOCH:3")
    
    if stoch_k > 80:
        short_score += 5
        short_breakdown.append("STOCH:5")
    elif stoch_k > 70:
        short_score += 3
        short_breakdown.append("STOCH:3")
    
    # TP/SL info
    atr = calculate_atr(df).iloc[-1]
    tp_sl_info = {
        'bb_mid': bb_mid,
        'bb_lower': bb_lower,
        'bb_upper': bb_upper,
        'atr': atr,
        'bb_position': bb_position
    }
    
    return long_score, short_score, long_breakdown, short_breakdown, tp_sl_info
