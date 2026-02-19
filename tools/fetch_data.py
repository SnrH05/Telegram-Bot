"""
TITANIUM Bot - Historical Data Fetcher
======================================
Downloads historical OHLCV data from KuCoin for backtesting.
Usage: python tools/fetch_data.py
"""

import sys
import os
import time
import ccxt
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COIN_LIST, COIN_GROUPS

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TIMEFRAME = "1h"
DAYS_BACK = 365  # 1 Year of data

def fetch_ohlcv(exchange, symbol, timeframe, since):
    """Fetch OHLCV data with pagination."""
    all_ohlcv = []
    limit = 1000  # KuCoin limit per request
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Update since timestamp to last candle time + 1ms
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1
            
            # Progress indicator
            print(f"  [INFO] Fetched {len(ohlcv)} candles... Total: {len(all_ohlcv)}")
            
            if len(ohlcv) < limit:
                break
                
            # Rate limit handling
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"[ERROR] fetching {symbol}: {e}")
            time.sleep(5)  # Retry delay
            continue
            
    return all_ohlcv

def save_data(symbol, ohlcv):
    """Save data to CSV."""
    if not ohlcv:
        return
        
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    
    # Save to CSV
    filename = f"{symbol.replace('/USDT', '')}_{TIMEFRAME}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath)
    print(f"[OK] Saved {len(df)} candles to {filepath}")

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    exchange = ccxt.kucoin()
    
    # Get distinct list of all coins from groups
    if len(sys.argv) > 1:
        # User specified a coin (e.g. python tools/fetch_data.py BTC)
        user_coin = sys.argv[1].upper()
        sorted_coins = [user_coin]
        print(f"[INFO] Single coin mode selected: {user_coin}")
    else:
        # Default: All coins
        all_coins = set()
        for group in COIN_GROUPS.values():
            all_coins.update(group['coins'])
        sorted_coins = sorted(list(all_coins))
    
    print(f"[START] Starting download for {len(sorted_coins)} coins...")
    print(f"Timeframe: {TIMEFRAME}, Days: {DAYS_BACK}")
    
    since = exchange.parse8601((datetime.now() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d 00:00:00"))
    
    for idx, coin in enumerate(sorted_coins):
        symbol = f"{coin}/USDT"
        print(f"\n[{idx+1}/{len(sorted_coins)}] Fetching {symbol}...")
        
        try:
            ohlcv = fetch_ohlcv(exchange, symbol, TIMEFRAME, since)
            save_data(symbol, ohlcv)
        except Exception as e:
            print(f"[ERROR] Failed to process {symbol}: {e}")
            
    print("\n[DONE] Download complete!")

if __name__ == "__main__":
    main()
