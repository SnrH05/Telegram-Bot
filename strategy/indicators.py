"""
TITANIUM Bot - Shared Indicators
==================================
Backtest ve live trading için paylaşılan indikatör fonksiyonları.

Bu modül hem main.py hem de backtest script'leri tarafından kullanılır.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential Moving Average hesapla.
    
    Args:
        series: Fiyat serisi
        span: EMA periyodu
        
    Returns:
        EMA serisi
    """
    return series.ewm(span=span, adjust=False).mean()


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average hesapla.
    
    Args:
        series: Fiyat serisi
        window: SMA periyodu
        
    Returns:
        SMA serisi
    """
    return series.rolling(window=window).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index hesapla.
    
    Division by zero korumalı versiyon.
    
    Args:
        series: Fiyat serisi
        period: RSI periyodu (default: 14)
        
    Returns:
        RSI serisi (0-100 arasında)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Division by zero önleme
    loss = loss.replace(0, 1e-10)
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Edge case handling
    rsi = rsi.clip(0, 100)
    return rsi.fillna(50)  # Yetersiz veri için nötr


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range hesapla.
    
    Args:
        df: OHLCV DataFrame (high, low, close sütunları gerekli)
        period: ATR periyodu (default: 14)
        
    Returns:
        ATR serisi
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
    
    Division by zero korumalı versiyon.
    
    Args:
        df: OHLCV DataFrame
        period: ADX periyodu (default: 14)
        
    Returns:
        ADX serisi (0-100 arasında)
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
    
    # Division by zero önleme
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
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands hesapla.
    
    Args:
        df: OHLCV DataFrame (close sütunu gerekli)
        period: SMA periyodu (default: 20)
        std_dev: Standart sapma çarpanı (default: 2)
        
    Returns:
        (lower_band, middle_band, upper_band) tuple
    """
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return lower, sma, upper


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume hesapla.
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        OBV serisi
    """
    obv = ((df['close'].diff() > 0).astype(int) * 2 - 1) * df['volume']
    return obv.cumsum()


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD hesapla.
    
    Args:
        series: Fiyat serisi
        fast: Hızlı EMA periyodu
        slow: Yavaş EMA periyodu
        signal: Signal line periyodu
        
    Returns:
        (macd_line, signal_line, histogram) tuple
    """
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_stochastic_rsi(
    series: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic RSI hesapla.
    
    RSI üzerine Stochastic formülü uygulayarak aşırı alım/satım
    sinyallerini daha hassas tespit eder.
    
    Args:
        series: Fiyat serisi (close)
        rsi_period: RSI periyodu (default: 14)
        stoch_period: Stochastic pencere periyodu (default: 14)
        k_smooth: %K düzleştirme periyodu (default: 3)
        d_smooth: %D düzleştirme periyodu (default: 3)
        
    Returns:
        (stoch_k, stoch_d) tuple — her ikisi de 0-100 arasında
    """
    # Step 1: RSI hesapla
    rsi = calculate_rsi(series, period=rsi_period)
    
    # Step 2: RSI üzerinde Stochastic formülü
    rsi_low = rsi.rolling(window=stoch_period).min()
    rsi_high = rsi.rolling(window=stoch_period).max()
    
    # Division by zero koruması (flat RSI durumlarında high == low)
    denominator = rsi_high - rsi_low
    denominator = denominator.replace(0, 1e-10)
    
    stoch_rsi = ((rsi - rsi_low) / denominator) * 100
    
    # Step 3: Düzleştirme — %K ve %D
    stoch_k = stoch_rsi.rolling(window=k_smooth).mean()
    stoch_d = stoch_k.rolling(window=d_smooth).mean()
    
    # Sınırlandır ve NaN temizle
    stoch_k = stoch_k.clip(0, 100).fillna(50)
    stoch_d = stoch_d.clip(0, 100).fillna(50)
    
    return stoch_k, stoch_d


def calculate_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Chaikin Money Flow (CMF) hesapla.
    
    Para akışı yönünü tespit eder. Pozitif CMF = alım baskısı,
    negatif CMF = satış baskısı.
    
    Formül:
        MF Multiplier = ((close - low) - (high - close)) / (high - low)
        MF Volume     = MF Multiplier × volume
        CMF           = sum(MF Volume, period) / sum(volume, period)
    
    Args:
        df: OHLCV DataFrame (high, low, close, volume sütunları gerekli)
        period: CMF periyodu (default: 20)
        
    Returns:
        CMF serisi (-1 ile +1 arasında)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # Money Flow Multiplier: ((C - L) - (H - C)) / (H - L)
    hl_range = high - low
    hl_range = hl_range.replace(0, 1e-10)  # Division by zero koruması
    
    mf_multiplier = ((close - low) - (high - close)) / hl_range
    
    # Money Flow Volume
    mf_volume = mf_multiplier * volume
    
    # CMF = rolling sum(MF Volume) / rolling sum(Volume)
    vol_sum = volume.rolling(window=period).sum()
    vol_sum = vol_sum.replace(0, 1e-10)  # Division by zero koruması
    
    cmf = mf_volume.rolling(window=period).sum() / vol_sum
    
    return cmf.clip(-1, 1).fillna(0)


def calculate_supertrend(
    df: pd.DataFrame, 
    period: int = 10, 
    multiplier: float = 3.0
) -> Tuple[pd.Series, pd.Series]:
    """
    SuperTrend İndikatörü hesapla.
    
    Args:
        df: OHLCV DataFrame (high, low, close)
        period: ATR periyodu
        multiplier: ATR çarpanı
        
    Returns:
        (supertrend_line, trend_direction)
        trend_direction: 1 (UP), -1 (DOWN)
    """
    # ATR hesapla
    df = df.copy()
    atr = calculate_atr(df, period)
    
    # Basic Upper/Lower Bands
    hl2 = (df['high'] + df['low']) / 2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)
    
    # Final Upper/Lower Bands başlat
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    trend = pd.Series(0, index=df.index)
    
    # İteratif hesaplama (SuperTrend doğası gereği önceki değere bakar)
    # Pandas ile vektörel yapmak zordur, loop kullanacağız
    for i in range(1, len(df)):
        # Final Upper Band
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or df['close'].iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
            
        # Final Lower Band
        if basic_lower.iloc[i] > final_lower.iloc[i-1] or df['close'].iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]
            
        # Trend Yönü
        # Önceki trend devam ediyor mu?
        prev_trend = trend.iloc[i-1] if i > 0 else 1
        
        if prev_trend == 1: # Uptrend
            if df['close'].iloc[i] < final_lower.iloc[i]:
                trend.iloc[i] = -1 # Downtrend'e dön
            else:
                trend.iloc[i] = 1
        else: # Downtrend
            if df['close'].iloc[i] > final_upper.iloc[i]:
                trend.iloc[i] = 1 # Uptrend'e dön
            else:
                trend.iloc[i] = -1
                
    # SuperTrend Line
    supertrend = pd.Series(index=df.index, dtype='float64')
    supertrend.loc[trend == 1] = final_lower
    supertrend.loc[trend == -1] = final_upper
    
    return supertrend, trend

