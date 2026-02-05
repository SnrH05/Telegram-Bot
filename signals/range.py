"""
TITANIUM Bot - Range Trading Signals
=====================================
Düz piyasa tespiti ve range trading fonksiyonları.
"""

import pandas as pd
import logging
from strategy import calculate_rsi, calculate_atr, calculate_adx, calculate_bollinger

logger = logging.getLogger(__name__)


def is_ranging_market(df, adx_threshold=20):
    """
    Düz piyasa tespiti
    
    Kriterler:
    - ADX < threshold (trend yok)
    - ATR ortalamanın altında (düşük volatilite)
    - Bollinger bantları daralmış
    
    Returns:
        is_ranging (bool), range_details (dict)
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


def calculate_range_score(df):
    """
    Range Trading için sinyal skoru (0-60 üzerinden)
    
    Puanlama:
    - Bollinger Alt/Üst Bant: 20 puan
    - RSI Oversold/Overbought: 15 puan
    - SMA20'den Sapma: 12 puan
    - Wick Rejection: 8 puan
    - Stochastic: 5 puan
    
    Eşik: 35/60 = Range Sinyal
    
    Returns:
        long_score, short_score, long_breakdown, short_breakdown, tp_sl_info
    """
    long_score = 0
    short_score = 0
    long_breakdown = []
    short_breakdown = []
    
    price = df['close'].iloc[-1]
    
    # 1. Bollinger Bant Pozisyonu (max 20 puan)
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
    
    # 3. SMA20'den Sapma (max 12 puan)
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
    
    # 5. Stochastic benzeri kontrol (max 5 puan)
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
    
    # TP/SL için range bilgisi
    atr = calculate_atr(df).iloc[-1]
    tp_sl_info = {
        'bb_mid': bb_mid,
        'bb_lower': bb_lower,
        'bb_upper': bb_upper,
        'atr': atr,
        'bb_position': bb_position
    }
    
    return long_score, short_score, long_breakdown, short_breakdown, tp_sl_info
