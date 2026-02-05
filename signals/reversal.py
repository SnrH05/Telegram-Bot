"""
TITANIUM Bot - Reversal Signals
===============================
Ani yön değişimi tespiti fonksiyonları.
"""

import pandas as pd
import logging
from strategy import calculate_rsi, calculate_atr

logger = logging.getLogger(__name__)


def calculate_momentum_reversal(df, lookback=5, threshold=2.0):
    """
    Son X mumda ani momentum değişimi var mı?
    
    Args:
        df: OHLCV DataFrame
        lookback: Kaç mum geriye bakılacak
        threshold: Yüzde değişim eşiği
    
    Returns:
        reversal_type: 'REVERSAL_UP', 'REVERSAL_DOWN', veya None
        change_pct: Yüzde değişim
    """
    closes = df['close'].tail(lookback)
    
    # Son mum ile önceki mumların ortalaması arasındaki fark
    curr_close = closes.iloc[-1]
    avg_prev = closes.iloc[:-1].mean()
    change_pct = ((curr_close - avg_prev) / avg_prev) * 100
    
    # Eşiği geçen değişim = Ani hareket
    if change_pct > threshold:
        return 'REVERSAL_UP', change_pct
    elif change_pct < -threshold:
        return 'REVERSAL_DOWN', change_pct
    return None, change_pct


def check_rsi_divergence(df, lookback=14):
    """
    RSI Divergence tespit et - En güvenilir reversal sinyali
    
    Bullish Divergence: Fiyat düşük dip yaparken RSI yüksek dip yapar
    Bearish Divergence: Fiyat yüksek zirve yaparken RSI düşük zirve yapar
    
    Returns:
        divergence_type: 'BULLISH_DIV', 'BEARISH_DIV', veya None
        strength: Divergence gücü (0-100)
    """
    price = df['close'].tail(lookback)
    rsi = calculate_rsi(df['close']).tail(lookback)
    
    curr_price = price.iloc[-1]
    curr_rsi = rsi.iloc[-1]
    
    # Bullish Divergence: Fiyat düşüyor ama RSI yükseliyor
    price_falling = price.iloc[-1] < price.iloc[-3]
    rsi_rising = rsi.iloc[-1] > rsi.iloc[-3]
    
    if price_falling and rsi_rising and curr_rsi < 40:
        strength = min(100, abs(rsi.iloc[-1] - rsi.iloc[-3]) * 5)
        return 'BULLISH_DIV', strength
    
    # Bearish Divergence: Fiyat çıkıyor ama RSI düşüyor
    price_rising = price.iloc[-1] > price.iloc[-3]
    rsi_falling = rsi.iloc[-1] < rsi.iloc[-3]
    
    if price_rising and rsi_falling and curr_rsi > 60:
        strength = min(100, abs(rsi.iloc[-3] - rsi.iloc[-1]) * 5)
        return 'BEARISH_DIV', strength
    
    return None, 0


def check_volatility_spike(df, period=14, multiplier=2.0):
    """
    Volatilite patlaması tespit et - ATR normal seviyenin X katı üstündeyse
    
    Returns:
        spike_type: 'SPIKE_UP', 'SPIKE_DOWN', veya None
        atr_ratio: Mevcut ATR / Ortalama ATR
    """
    atr = calculate_atr(df, period)
    avg_atr = atr.rolling(50).mean().iloc[-1]
    curr_atr = atr.iloc[-1]
    
    if pd.isna(avg_atr) or avg_atr == 0:
        return None, 1.0
    
    atr_ratio = curr_atr / avg_atr
    
    if atr_ratio > multiplier:
        # Mum rengine göre yön belirle
        is_bullish_candle = df['close'].iloc[-1] > df['open'].iloc[-1]
        if is_bullish_candle:
            return 'SPIKE_UP', atr_ratio
        else:
            return 'SPIKE_DOWN', atr_ratio
    
    return None, atr_ratio


def calculate_reversal_score(df):
    """
    Tüm reversal indikatörlerini birleştirerek skor hesapla
    
    Returns:
        reversal_long_score: LONG reversal skoru (0-30)
        reversal_short_score: SHORT reversal skoru (0-30)
        reversal_details: Detay bilgisi
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
    
    # 2. RSI Divergence (max 12 puan) - En güvenilir
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
