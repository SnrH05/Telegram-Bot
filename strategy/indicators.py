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
