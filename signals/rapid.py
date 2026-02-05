"""
TITANIUM Bot - Rapid Reversal Signals
======================================
Ani fiyat hareketi ve rapid reversal stratejisi fonksiyonları.
"""

import logging
from strategy import calculate_rsi, calculate_atr
from signals.reversal import check_volatility_spike

logger = logging.getLogger(__name__)


def detect_flash_move(df, threshold_pct=3.0, lookback=3):
    """
    Ani fiyat hareketi tespiti - Son X mumda %threshold+ değişim
    
    Returns:
        flash_type: 'FLASH_UP', 'FLASH_DOWN', veya None
        change_pct: Yüzde değişim
    """
    if len(df) < lookback + 1:
        return None, 0
    
    closes = df['close'].tail(lookback + 1)
    start_price = closes.iloc[0]
    end_price = closes.iloc[-1]
    
    change_pct = ((end_price - start_price) / start_price) * 100
    
    last_candle_bullish = df['close'].iloc[-1] > df['open'].iloc[-1]
    
    # Flash DOWN sonrası yeşil mum = LONG fırsatı
    if change_pct < -threshold_pct and last_candle_bullish:
        return 'FLASH_UP', abs(change_pct)
    
    # Flash UP sonrası kırmızı mum = SHORT fırsatı
    if change_pct > threshold_pct and not last_candle_bullish:
        return 'FLASH_DOWN', abs(change_pct)
    
    return None, abs(change_pct)


def detect_volume_spike(df, multiplier=3.0, lookback=20):
    """
    Hacim patlaması tespiti - Ortalamanın X katı hacim
    
    Returns:
        spike_type: 'VOL_SPIKE_UP', 'VOL_SPIKE_DOWN', veya None
        vol_ratio: Hacim oranı
    """
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


def detect_wick_rejection(df, wick_body_ratio=2.0):
    """
    Fitil reddi tespiti - Uzun fitil = Reddedilen seviye
    
    Returns:
        wick_type: 'WICK_UP', 'WICK_DOWN', veya None
        wick_ratio: Fitil/gövde oranı
    """
    row = df.iloc[-1]
    
    body = abs(row['close'] - row['open'])
    upper_wick = row['high'] - max(row['close'], row['open'])
    lower_wick = min(row['close'], row['open']) - row['low']
    
    if body == 0:
        body = 0.0001  # Doji durumu
    
    upper_ratio = upper_wick / body
    lower_ratio = lower_wick / body
    
    if lower_ratio >= wick_body_ratio and lower_ratio > upper_ratio:
        return 'WICK_UP', lower_ratio
    
    if upper_ratio >= wick_body_ratio and upper_ratio > lower_ratio:
        return 'WICK_DOWN', upper_ratio
    
    return None, max(upper_ratio, lower_ratio)


def detect_rsi_extreme_bounce(df, oversold=25, overbought=75):
    """
    RSI aşırı bölgeden dönüş tespiti
    
    Returns:
        bounce_type: 'RSI_BOUNCE_UP', 'RSI_BOUNCE_DOWN', veya None
        rsi_value: Mevcut RSI değeri
    """
    if len(df) < 15:
        return None, 50
    
    rsi = calculate_rsi(df['close'])
    curr_rsi = rsi.iloc[-1]
    prev_rsi = rsi.iloc[-2]
    
    is_bullish = df['close'].iloc[-1] > df['open'].iloc[-1]
    
    # Oversold'dan dönüş + yeşil mum
    if prev_rsi < oversold and curr_rsi > prev_rsi and is_bullish:
        return 'RSI_BOUNCE_UP', curr_rsi
    
    # Overbought'tan dönüş + kırmızı mum
    if prev_rsi > overbought and curr_rsi < prev_rsi and not is_bullish:
        return 'RSI_BOUNCE_DOWN', curr_rsi
    
    return None, curr_rsi


def calculate_rapid_score(df):
    """
    Rapid Reversal için ayrı skor hesapla (0-100)
    
    Puanlama:
    - Flash Move: 25 puan
    - Volume Spike: 25 puan
    - RSI Extreme: 20 puan
    - ATR Explosion: 15 puan
    - Wick Rejection: 15 puan
    
    Returns:
        rapid_long_score, rapid_short_score, rapid_details, tetikleyici
    """
    long_score = 0
    short_score = 0
    details = []
    tetikleyici = []
    
    # 1. Flash Move (25 puan)
    flash_type, flash_pct = detect_flash_move(df, threshold_pct=3.0)
    if flash_type == 'FLASH_UP':
        score = min(25, int(flash_pct * 5))
        long_score += score
        details.append(f"Flash:{score}")
        tetikleyici.append(f"Flash Move {flash_pct:.1f}%")
    elif flash_type == 'FLASH_DOWN':
        score = min(25, int(flash_pct * 5))
        short_score += score
        details.append(f"Flash:{score}")
        tetikleyici.append(f"Flash Move {flash_pct:.1f}%")
    
    # 2. Volume Spike (25 puan)
    vol_type, vol_ratio = detect_volume_spike(df, multiplier=3.0)
    if vol_type == 'VOL_SPIKE_UP':
        score = min(25, int((vol_ratio - 1) * 8))
        long_score += score
        details.append(f"Vol:{score}")
        tetikleyici.append(f"Volume {vol_ratio:.1f}x")
    elif vol_type == 'VOL_SPIKE_DOWN':
        score = min(25, int((vol_ratio - 1) * 8))
        short_score += score
        details.append(f"Vol:{score}")
        tetikleyici.append(f"Volume {vol_ratio:.1f}x")
    
    # 3. RSI Extreme Bounce (20 puan)
    rsi_type, rsi_val = detect_rsi_extreme_bounce(df)
    if rsi_type == 'RSI_BOUNCE_UP':
        long_score += 20
        details.append("RSI:20")
        tetikleyici.append(f"RSI Bounce ({rsi_val:.0f})")
    elif rsi_type == 'RSI_BOUNCE_DOWN':
        short_score += 20
        details.append("RSI:20")
        tetikleyici.append(f"RSI Bounce ({rsi_val:.0f})")
    
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
        tetikleyici.append("Wick Rejection")
    elif wick_type == 'WICK_DOWN':
        score = min(15, int(wick_ratio * 3))
        short_score += score
        details.append(f"Wick:{score}")
        tetikleyici.append("Wick Rejection")
    
    return long_score, short_score, details, tetikleyici
