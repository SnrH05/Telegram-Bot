"""
TITANIUM Bot - Vectorized Scoring
=================================
Calculates the 11-factor score for the entire DataFrame at once.
Optimized for backtesting.
Match the logic in strategy/scoring.py V6.2.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import strategy modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.indicators import (
    calculate_rsi, calculate_atr, calculate_adx,
    calculate_bollinger, calculate_ema, calculate_sma,
    calculate_macd, calculate_stochastic_rsi, calculate_obv
)

def calculate_vectorized_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 11-factor scores for LONG and SHORT logic.
    Adds 'total_score_long' and 'total_score_short' columns.
    Target Max Score: 123
    """
    df = df.copy()
    
    # ----------------------------------------------------
    # 1. Indicators Calculation
    # ----------------------------------------------------
    # Price Stuff
    df['ema20'] = calculate_ema(df['close'], 20)
    df['ema50'] = calculate_ema(df['close'], 50)
    df['ema100'] = calculate_ema(df['close'], 100)
    df['ema200'] = calculate_ema(df['close'], 200)
    df['sma200'] = calculate_sma(df['close'], 200)
    
    # Oscillators
    df['rsi'] = calculate_rsi(df['close'], 14)
    # df['rsi_4h'] = ... (Simulated by 4x period RSI approx or resampling? 
    # For backtest speed, let's use coarser RSI approximation: RSI(56) on 1h ~= RSI(14) on 4h? 
    # No, that's not accurate but acceptable for fast vectorization, or we just resample)
    # Better: Use RSI(14) but look at trend of RSI? 
    # Let's use simple RSI(56) as a proxy for "Higher Timeframe RSI" to avoid resampling complexity in pure vector calc
    df['rsi_htf'] = calculate_rsi(df['close'], 14*4) 
    
    macd, signal, hist = calculate_macd(df['close'])
    df['macd_hist'] = hist
    
    stoch_k, stoch_d = calculate_stochastic_rsi(df['close'])
    df['stoch_k'] = stoch_k
    
    # Volume & Volatility
    lower, middle, upper = calculate_bollinger(df, 20, 2.0)
    df['bb_lower'] = lower
    df['bb_upper'] = upper
    df['bb_width'] = (upper - lower) / middle
    
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['obv'] = calculate_obv(df)
    df['obv_sma10'] = df['obv'].rolling(10).mean()
    
    df['adx'] = calculate_adx(df, 14)

    # initialize score columns
    df['score_long'] = 0
    df['score_short'] = 0

    # ----------------------------------------------------
    # 2. Factor Scoring (11 Factors)
    # ----------------------------------------------------
    
    # F1: BTC Trend (Approximated by EMA Alignment here as we don't have separate BTC data in single coin backtest easily)
    # In live bot this comes from BTC pair. In backtest, we'll assume "General Trend" factor based on own chart for now
    # or skip BTC factor? Let's use EMA alignment as a proxy for ecosystem trend strength.
    # EMA20 > EMA50 > EMA100
    long_trend = (df['ema20'] > df['ema50']) & (df['ema50'] > df['ema100'])
    short_trend = (df['ema20'] < df['ema50']) & (df['ema50'] < df['ema100'])
    
    df['score_long'] += np.where(long_trend, 20, 0)
    df['score_short'] += np.where(short_trend, 20, 0)
    
    # F2: Reversal (MACD Hist + RSI)
    # LONG: RSI < 40 and MACD Hist > prev Hist (momentum turning up)
    # SHORT: RSI > 60 and MACD Hist < prev Hist
    hist_up = df['macd_hist'] > df['macd_hist'].shift(1)
    hist_down = df['macd_hist'] < df['macd_hist'].shift(1)
    
    df['score_long'] += np.where((df['rsi'] < 45) & hist_up, 18, 0)
    df['score_short'] += np.where((df['rsi'] > 55) & hist_down, 18, 0)
    
    # F3: HTF Trend (EMA50/100/200)
    long_htf = (df['ema50'] > df['ema100']) & (df['ema100'] > df['ema200'])
    short_htf = (df['ema50'] < df['ema100']) & (df['ema100'] < df['ema200'])
    
    df['score_long'] += np.where(long_htf, 15, 0)
    df['score_short'] += np.where(short_htf, 15, 0)
    
    # F4: Squeeze (BB Width narrowed + Vol up?)
    # Simply BB Width < Percentile 20? 
    # Let's say BB Width < 0.10 (arbitrary) or recent low width
    is_squeeze = df['bb_width'] < df['bb_width'].rolling(50).quantile(0.20)
    vol_up = df['volume'] > df['vol_sma20']
    
    df['score_long'] += np.where(is_squeeze & vol_up, 15, 0)
    df['score_short'] += np.where(is_squeeze & vol_up, 15, 0) # Squeeze works both ways
    
    # F5: SMA200
    df['score_long'] += np.where(df['close'] > df['sma200'], 12, 0)
    df['score_short'] += np.where(df['close'] < df['sma200'], 12, 0)
    
    # F6: USDT Dom (Proxy: Inverse price action? Skip in single coin backtest)
    # Let's award points if recent momentum is strong to compensate
    df['score_long'] += 5 # Base points for neutrality
    df['score_short'] += 5
    
    # F7: RSI Sweet Spot
    # LONG: 30-50, SHORT: 50-70
    df['score_long'] += np.where((df['rsi'] >= 30) & (df['rsi'] <= 50), 10, 0)
    df['score_short'] += np.where((df['rsi'] >= 50) & (df['rsi'] <= 70), 10, 0)
    
    # F8: RSI 4H (HTF)
    # LONG: HTF RSI > 40, SHORT: HTF RSI < 60
    df['score_long'] += np.where(df['rsi_htf'] > 40, 5, 0)
    df['score_short'] += np.where(df['rsi_htf'] < 60, 5, 0)
    
    # F9: Volume
    # Vol > SMA20
    df['score_long'] += np.where(df['volume'] > df['vol_sma20'], 8, 0)
    df['score_short'] += np.where(df['volume'] > df['vol_sma20'], 8, 0)
    
    # F10: OBV
    # OBV > SMA
    df['score_long'] += np.where(df['obv'] > df['obv_sma10'], 3, 0)
    df['score_short'] += np.where(df['obv'] < df['obv_sma10'], 3, 0)
    
    # F11: ADX
    # ADX > 25
    df['score_long'] += np.where(df['adx'] > 25, 7, 0)
    df['score_short'] += np.where(df['adx'] > 25, 7, 0)
    
    return df

if __name__ == "__main__":
    # Test
    dates = pd.date_range("2025-01-01", periods=100, freq='h')
    df = pd.DataFrame({
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100,
        'low': np.random.rand(100) * 100,
        'close': np.random.rand(100) * 100,
        'volume': np.random.rand(100) * 1000
    }, index=dates)
    
    scored = calculate_vectorized_scores(df)
    print(scored[['close', 'score_long', 'score_short']].tail())
