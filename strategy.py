"""
TITANIUM Bot - Strategy Module
==============================
Paylaşılan strateji ve indikatör fonksiyonları.
main.py ve backtest.py tarafından ortak kullanım için.

Versiyon: 6.1
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional


# ==========================================
# 📊 TEMEL İNDİKATÖRLER
# ==========================================

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average hesapla"""
    return series.ewm(span=span, adjust=False).mean()


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average hesapla"""
    return series.rolling(window=window).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index hesapla.
    
    Args:
        series: Fiyat serisi (close)
        period: RSI periyodu
    
    Returns:
        RSI değerleri (0-100 arası, NaN değerler 50 ile doldurulur)
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    
    # Sıfıra bölme önleme
    loss = loss.replace(0, 1e-10)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Edge case'ler
    rsi = rsi.clip(0, 100)
    return rsi.fillna(50)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range hesapla.
    
    Args:
        df: OHLC DataFrame
        period: ATR periyodu
    
    Returns:
        ATR değerleri
    """
    df = df.copy()
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    return df['tr'].rolling(period).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index hesapla.
    
    Args:
        df: OHLC DataFrame
        period: ADX periyodu
    
    Returns:
        ADX değerleri (0-100 arası)
    """
    df = df.copy()
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    
    df['atr'] = df['tr'].rolling(period).mean()
    
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    
    df['plus_dm'] = np.where(
        (df['up_move'] > df['down_move']) & (df['up_move'] > 0), 
        df['up_move'], 0
    )
    df['minus_dm'] = np.where(
        (df['down_move'] > df['up_move']) & (df['down_move'] > 0), 
        df['down_move'], 0
    )
    
    # Sıfıra bölme önleme
    atr_safe = df['atr'].replace(0, 1e-10)
    df['plus_di'] = 100 * (df['plus_dm'].ewm(alpha=1/period).mean() / atr_safe)
    df['minus_di'] = 100 * (df['minus_dm'].ewm(alpha=1/period).mean() / atr_safe)
    
    di_sum = df['plus_di'] + df['minus_di']
    di_sum = di_sum.replace(0, 1e-10)
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / di_sum
    df['adx'] = df['dx'].ewm(alpha=1/period).mean()
    
    return df['adx'].fillna(0).replace([np.inf, -np.inf], 0)


def calculate_bollinger(
    df: pd.DataFrame, 
    period: int = 20, 
    std_dev: int = 2
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bantları hesapla.
    
    Returns:
        (lower_band, middle_band, upper_band)
    """
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return lower, sma, upper


# ==========================================
# 🔄 REVERSAL TESPİTİ
# ==========================================

def calculate_momentum_reversal(
    df: pd.DataFrame, 
    lookback: int = 5, 
    threshold: float = 2.0
) -> Tuple[Optional[str], float]:
    """
    Momentum reversal tespit et.
    
    Returns:
        (reversal_type, change_pct)
        reversal_type: 'REVERSAL_UP', 'REVERSAL_DOWN', veya None
    """
    if len(df) < lookback + 1:
        return None, 0.0
    
    recent = df.tail(lookback)
    first_close = recent.iloc[0]['close']
    last_close = recent.iloc[-1]['close']
    
    change_pct = ((last_close - first_close) / first_close) * 100
    
    if change_pct > threshold:
        return 'REVERSAL_UP', change_pct
    elif change_pct < -threshold:
        return 'REVERSAL_DOWN', change_pct
    
    return None, change_pct


def check_rsi_divergence(
    df: pd.DataFrame, 
    lookback: int = 14
) -> Tuple[Optional[str], int]:
    """
    RSI Divergence tespit et.
    
    Returns:
        (divergence_type, strength)
        divergence_type: 'BULLISH_DIV', 'BEARISH_DIV', veya None
    """
    if len(df) < lookback + 1:
        return None, 0
    
    recent_df = df.tail(lookback)
    rsi = calculate_rsi(df['close']).tail(lookback)
    
    # Fiyat ve RSI diplerini bul
    price_lows = recent_df['close'].nsmallest(3)
    rsi_lows = rsi.iloc[price_lows.index - recent_df.index[0]].nsmallest(3) if len(price_lows) > 0 else pd.Series()
    
    # Fiyat ve RSI zirvelerini bul
    price_highs = recent_df['close'].nlargest(3)
    rsi_highs = rsi.iloc[price_highs.index - recent_df.index[0]].nlargest(3) if len(price_highs) > 0 else pd.Series()
    
    # Bullish Divergence: Fiyat düşük dip, RSI yüksek dip
    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        if price_lows.iloc[0] < price_lows.iloc[1] and rsi_lows.iloc[0] > rsi_lows.iloc[1]:
            strength = min(100, int(abs(rsi_lows.iloc[0] - rsi_lows.iloc[1]) * 5))
            return 'BULLISH_DIV', strength
    
    # Bearish Divergence: Fiyat yüksek zirve, RSI düşük zirve
    if len(price_highs) >= 2 and len(rsi_highs) >= 2:
        if price_highs.iloc[0] > price_highs.iloc[1] and rsi_highs.iloc[0] < rsi_highs.iloc[1]:
            strength = min(100, int(abs(rsi_highs.iloc[0] - rsi_highs.iloc[1]) * 5))
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
    if atr.isna().all():
        return None, 1.0
    
    current_atr = atr.iloc[-1]
    avg_atr = atr.rolling(50).mean().iloc[-1]
    
    if pd.isna(avg_atr) or avg_atr == 0:
        return None, 1.0
    
    ratio = current_atr / avg_atr
    
    # Son mumun yönü
    last_candle_bullish = df['close'].iloc[-1] > df['open'].iloc[-1]
    
    if ratio > multiplier:
        if last_candle_bullish:
            return 'SPIKE_UP', ratio
        else:
            return 'SPIKE_DOWN', ratio
    
    return None, ratio


# ==========================================
# 🔥 VOLATILITY SQUEEZE
# ==========================================

def detect_volatility_squeeze(
    df: pd.DataFrame, 
    bb_period: int = 20, 
    vol_lookback: int = 10
) -> Tuple[bool, int, Dict]:
    """
    Volatility Squeeze tespit et.
    BB daralması + Hacim artışı + ATR normal = Büyük hareket öncüsü
    
    Returns:
        (is_squeeze, squeeze_score, details)
    """
    try:
        # Bollinger hesapla
        sma = df['close'].rolling(bb_period).mean()
        std = df['close'].rolling(bb_period).std()
        bb_width = ((std * 2) / sma * 100).iloc[-1]
        
        # BB genişliği tarihsel olarak düşük mü?
        bb_width_history = ((std * 2) / sma * 100).tail(50)
        bb_percentile = (bb_width_history < bb_width).sum() / len(bb_width_history) * 100
        is_bb_tight = bb_percentile < 20
        
        # Hacim artıyor mu?
        vol_sma = df['volume'].rolling(20).mean().iloc[-1]
        recent_vol_avg = df['volume'].tail(3).mean()
        vol_ratio = recent_vol_avg / vol_sma if vol_sma > 0 else 1
        vol_expanding = vol_ratio > 1.5
        
        # ATR Z-score kontrolü
        atr = calculate_atr(df)
        atr_current = atr.iloc[-1]
        atr_sma = atr.rolling(50).mean().iloc[-1]
        atr_std = atr.rolling(50).std().iloc[-1]
        
        atr_z_score = 0
        is_atr_normal = True
        if not pd.isna(atr_std) and atr_std > 0:
            atr_z_score = (atr_current - atr_sma) / atr_std
            is_atr_normal = atr_z_score < 1.5
        
        # Üçü birden = SQUEEZE!
        is_squeeze = is_bb_tight and vol_expanding and is_atr_normal
        
        # Squeeze skoru hesapla (0-15 puan bonus)
        squeeze_score = 0
        if is_squeeze:
            bb_score = min(8, int((20 - bb_percentile) / 2.5))
            vol_score = min(7, int((vol_ratio - 1) * 3.5))
            squeeze_score = bb_score + vol_score
        
        details = {
            'bb_width': round(bb_width, 2),
            'bb_percentile': round(bb_percentile, 1),
            'vol_ratio': round(vol_ratio, 2),
            'atr_z_score': round(atr_z_score, 2),
            'is_bb_tight': is_bb_tight,
            'vol_expanding': vol_expanding,
            'is_atr_normal': is_atr_normal,
            'squeeze_score': squeeze_score
        }
        
        return is_squeeze, squeeze_score, details
        
    except Exception:
        return False, 0, {}


# ==========================================
# 🛡️ DİNAMİK STOP-LOSS
# ==========================================

def calculate_trend_aware_sl_multiplier(
    df: pd.DataFrame, 
    direction: str
) -> Tuple[float, str]:
    """
    Trend gücüne göre dinamik SL çarpanı hesapla.
    
    Returns:
        (sl_multiplier, trend_strength)
    """
    try:
        adx_val = calculate_adx(df).iloc[-1]
        
        ema9 = calculate_ema(df['close'], 9).iloc[-1]
        ema21 = calculate_ema(df['close'], 21).iloc[-1]
        ema50 = calculate_ema(df['close'], 50).iloc[-1]
        
        bullish_alignment = ema9 > ema21 > ema50
        bearish_alignment = ema9 < ema21 < ema50
        
        trend_aligned = (direction == "LONG" and bullish_alignment) or \
                       (direction == "SHORT" and bearish_alignment)
        
        rsi = calculate_rsi(df['close']).iloc[-1]
        rsi_confirms = (direction == "LONG" and 40 < rsi < 70) or \
                      (direction == "SHORT" and 30 < rsi < 60)
        
        if adx_val > 35 and trend_aligned:
            return 4.0, "ÇOK GÜÇLÜ"
        elif adx_val > 25 and trend_aligned and rsi_confirms:
            return 2.5, "GÜÇLÜ"
        elif adx_val > 20:
            return 2.0, "NORMAL"
        else:
            return 2.0, "ZAYIF"
            
    except Exception:
        return 2.0, "DEFAULT"


# ==========================================
# 📊 SKORLAMA SABİTLERİ
# ==========================================

# Maksimum teorik puan
MAX_TEORIK_PUAN = 123

# Sinyal eşik oranları
ESIK_ORAN = 0.60       # %60 - sinyal için
YAKIN_ESIK_ORAN = 0.40  # %40 - yakın için

# Hesaplanmış eşikler
SINYAL_ESIK = int(MAX_TEORIK_PUAN * ESIK_ORAN)        # 74
YAKIN_ESIK = int(MAX_TEORIK_PUAN * YAKIN_ESIK_ORAN)   # 49
