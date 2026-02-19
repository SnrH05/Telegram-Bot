"""
TITANIUM Bot - Backtest Engine
==============================
Simulates trading based on vectorized scores.
Generates performance report.
Usage: python backtest/engine.py [coin_pair]
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SINYAL_ESIK, MAX_TEORIK_PUAN
from backtest.scoring import calculate_vectorized_scores
from strategy.indicators import calculate_atr

def run_backtest(df: pd.DataFrame, coin: str = "TEST"):
    """
    Run backtest simulation on a single DataFrame.
    Returns trade list and stats.
    """
    # 1. Calculate Scores
    print(f"[INFO] Calculating scores for {coin}...")
    df = calculate_vectorized_scores(df)
    
    # 2. Calculate ATR for Stops/Targets (Dynamic TP/SL logic)
    df['atr'] = calculate_atr(df, 14)
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    
    trades = []
    active_trade = None # {'entry_price': ..., 'sl': ..., 'tp1': ..., 'tp2': ..., 'tp3': ..., 'direction': ...}
    
    # Iterative loop for trade management (easier for complex TP/SL logic)
    print(f"[INFO] Simulating trades for {len(df)} candles...")
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        timestamp = df.index[i]
        price = row['close']
        high = row['high']
        low = row['low']
        
        # --- Manage Active Trade ---
        if active_trade:
            entry = active_trade['entry_price']
            direction = active_trade['direction']
            
            # Check Exit Conditions
            exit_price = None
            exit_reason = None
            pnl = 0
            
            if direction == "LONG":
                # Check SL
                if low <= active_trade['sl']:
                    exit_price = active_trade['sl']
                    exit_reason = "SL"
                # Check TP3
                elif high >= active_trade['tp3']:
                    exit_price = active_trade['tp3']
                    exit_reason = "TP3"
                # Check TP2
                elif high >= active_trade['tp2']:
                     # Trailing stop logic could go here
                     pass 
                # Check TP1
                elif high >= active_trade['tp1']:
                    pass
                    
            elif direction == "SHORT":
                # Check SL
                if high >= active_trade['sl']:
                    exit_price = active_trade['sl']
                    exit_reason = "SL"
                # Check TP3
                elif low <= active_trade['tp3']:
                    exit_price = active_trade['tp3']
                    exit_reason = "TP3"
            
            # Close Trade
            if exit_price:
                if direction == "LONG":
                    pnl = (exit_price - entry) / entry * 100
                else:
                    pnl = (entry - exit_price) / entry * 100
                    
                trades.append({
                    'coin': coin,
                    'direction': direction,
                    'entry_time': active_trade['entry_time'],
                    'exit_time': timestamp,
                    'entry_price': entry,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'reason': exit_reason,
                    'score': active_trade['score']
                })
                active_trade = None
                continue # Trade closed, look for new signal next candle (or same?)
        
        # --- Check New Signal ---
        if not active_trade:
            # Determine logic
            # Use dynamic volume factor
            vol_factor = 1.0
            if row['vol_sma20'] > 0:
                vol_factor = max(0.7, min(1.5, row['volume'] / row['vol_sma20']))
            
            # Base Params (MAJOR defaults)
            base_sl = 1.0
            base_tp1, base_tp2, base_tp3 = 2.0, 4.0, 6.0
            
            sl_mult = base_sl * vol_factor
            tp1_mult = base_tp1 * vol_factor
            tp2_mult = base_tp2 * vol_factor
            tp3_mult = base_tp3 * vol_factor
            
            atr = row['atr']
            
            if row['score_long'] >= SINYAL_ESIK:
                active_trade = {
                    'direction': 'LONG',
                    'entry_time': timestamp,
                    'entry_price': price,
                    'sl': price - (atr * sl_mult),
                    'tp1': price + (atr * tp1_mult),
                    'tp2': price + (atr * tp2_mult),
                    'tp3': price + (atr * tp3_mult),
                    'score': row['score_long']
                }
            elif row['score_short'] >= SINYAL_ESIK:
                active_trade = {
                    'direction': 'SHORT',
                    'entry_time': timestamp,
                    'entry_price': price,
                    'sl': price + (atr * sl_mult),
                    'tp1': price - (atr * tp1_mult),
                    'tp2': price - (atr * tp2_mult),
                    'tp3': price - (atr * tp3_mult),
                    'score': row['score_short']
                }
                
    return pd.DataFrame(trades)

def print_report(trades_df):
    if trades_df.empty:
        print("[WARN] No trades generated.")
        return
        
    print("\n" + "="*40)
    print("BACKTEST PERFORMANCE REPORT")
    print("="*40)
    
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['pnl'] > 0])
    losses = len(trades_df[trades_df['pnl'] <= 0])
    win_rate = (wins / total_trades) * 100
    
    total_pnl = trades_df['pnl'].sum()
    avg_pnl = trades_df['pnl'].mean()
    max_dd = trades_df['pnl'].cumsum().min()
    
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate:     {win_rate:.2f}%")
    print(f"Total PnL:    {total_pnl:.2f}%")
    print(f"Avg PnL:      {avg_pnl:.2f}%")
    print(f"Max Drawdown: {max_dd:.2f}% (approx)")
    
    print("\nStrategy Breakdown:")
    print(trades_df.groupby('direction')['pnl'].agg(['count', 'mean', 'sum']))

def main():
    target_coin = "BTC"
    if len(sys.argv) > 1:
        target_coin = sys.argv[1]
        
    data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", f"{target_coin}_1h.csv")
    
    if os.path.exists(data_file):
        print(f"[INFO] Loading data from {data_file}...")
        df = pd.read_csv(data_file, parse_dates=['date'], index_col='date')
    else:
        print(f"[WARN] Data file not found: {data_file}")
        print("Generating mock data for testing...")
        dates = pd.date_range("2025-01-01", periods=1000, freq='h')
        df = pd.DataFrame({
            'open': np.linspace(50000, 60000, 1000), # Simple trend
            'high': np.linspace(50000, 60000, 1000) + 100,
            'low': np.linspace(50000, 60000, 1000) - 100,
            'close': np.linspace(50000, 60000, 1000) + np.random.randn(1000) * 50,
            'volume': np.random.rand(1000) * 1000 + 500
        }, index=dates)
        
    trades = run_backtest(df, target_coin)
    print_report(trades)

if __name__ == "__main__":
    main()
