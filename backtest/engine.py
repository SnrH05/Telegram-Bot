"""
TITANIUM Bot - Backtest Engine (V7.0: Broker İyileştirmeleri)
=============================================================
Simulates trading based on vectorized scores.
Generates performance report.

V7.0 Improvements:
  - Trailing Stop: Break-even at 1×ATR, lock profit at 2×ATR
  - R:R Guard: Min 1.5:1 — SL=1.5×ATR, TP1=2.5, TP2=4.0, TP3=6.0
  - Strategy-aware thresholds (TREND=67, RANGE=62, MEME=64)

Usage:
  python backtest/engine.py [coin_pair] [--baseline | --improved]
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MAX_TEORIK_PUAN, ESIK_ORAN_TREND, ESIK_ORAN_RANGE, ESIK_ORAN_MEME
from backtest.scoring import calculate_vectorized_scores
from strategy.indicators import calculate_atr


# ==========================================
# BASELINE CONFIG (Eski Ayarlar)
# ==========================================
BASELINE_CONFIG = {
    'sl_atr_mult':  2.0,   # ATR * 2.0 = Stop Loss
    'tp1_atr_mult': 2.0,   # ATR * 2.0 = TP1
    'tp2_atr_mult': 4.0,   # ATR * 4.0 = TP2
    'tp3_atr_mult': 6.0,   # ATR * 6.0 = TP3
    'signal_threshold': int(MAX_TEORIK_PUAN * 0.60),  # Eski eşik: 74
    'trailing_stop': False,  # Mevcut: Yok
}

# ==========================================
# IMPROVED CONFIG (Broker İyileştirmeleri)
# ==========================================
IMPROVED_CONFIG = {
    'sl_atr_mult':  1.5,   # İyileştirme #7: 2.0→1.5 (daha az kayıp)
    'tp1_atr_mult': 2.5,   # İyileştirme #2: Asimetrik R:R (min 1.5:1)
    'tp2_atr_mult': 4.0,
    'tp3_atr_mult': 6.0,
    'signal_threshold': int(MAX_TEORIK_PUAN * ESIK_ORAN_TREND),  # İyileştirme #8: 74→67
    'trailing_stop': True,  # İyileştirme #3: Trailing Stop aktif
}


def run_backtest(df: pd.DataFrame, coin: str = "TEST", config: dict = None):
    """
    Run backtest simulation on a single DataFrame.

    Args:
        df: OHLCV DataFrame
        coin: Coin name
        config: BASELINE_CONFIG or IMPROVED_CONFIG

    Returns:
        pd.DataFrame of trades
    """
    if config is None:
        config = IMPROVED_CONFIG

    sl_mult   = config['sl_atr_mult']
    tp1_mult  = config['tp1_atr_mult']
    tp2_mult  = config['tp2_atr_mult']
    tp3_mult  = config['tp3_atr_mult']
    threshold = config['signal_threshold']
    use_trail = config['trailing_stop']

    # 1. Calculate Scores
    print(f"[INFO] Calculating scores for {coin}... (threshold={threshold})")
    df = calculate_vectorized_scores(df)

    # 2. Calculate ATR
    df['atr'] = calculate_atr(df, 14)
    df['vol_sma20'] = df['volume'].rolling(20).mean()

    trades = []
    active_trade = None

    print(f"[INFO] Simulating {len(df)} candles...")

    for i in range(50, len(df)):
        row      = df.iloc[i]
        timestamp = df.index[i]
        price    = row['close']
        high     = row['high']
        low      = row['low']
        atr      = row['atr'] if not pd.isna(row['atr']) else (price * 0.01)

        # ==========================================
        # MANAGE ACTIVE TRADE
        # ==========================================
        if active_trade:
            entry     = active_trade['entry_price']
            direction = active_trade['direction']
            sl        = active_trade['sl']
            tp1       = active_trade['tp1']
            tp2       = active_trade['tp2']
            tp3       = active_trade['tp3']
            entry_atr = active_trade['entry_atr']

            exit_price  = None
            exit_reason = None

            if direction == "LONG":
                # ── İyileştirme #3: Trailing Stop Logic ──────────────────
                if use_trail:
                    unrealized = high - entry  # En yüksek kâr
                    if unrealized >= 2.0 * entry_atr:
                        # 2×ATR kârda → SL = entry + 1×ATR (kâr kilitle)
                        new_sl = entry + 1.0 * entry_atr
                        if new_sl > active_trade['sl']:
                            active_trade['sl'] = new_sl
                            sl = new_sl
                    elif unrealized >= 1.0 * entry_atr:
                        # 1×ATR kârda → SL = entry (break-even)
                        new_sl = entry
                        if new_sl > active_trade['sl']:
                            active_trade['sl'] = new_sl
                            sl = new_sl
                # ─────────────────────────────────────────────────────────

                if low <= sl:
                    exit_price  = sl
                    exit_reason = "SL"
                elif high >= tp3:
                    exit_price  = tp3
                    exit_reason = "TP3"
                elif high >= tp2 and not active_trade.get('tp2_hit'):
                    active_trade['tp2_hit'] = True
                    # TP2'de trailing stop sıkıştır
                    if use_trail:
                        new_sl = entry + 1.5 * entry_atr
                        if new_sl > active_trade['sl']:
                            active_trade['sl'] = new_sl
                elif high >= tp1 and not active_trade.get('tp1_hit'):
                    active_trade['tp1_hit'] = True

            elif direction == "SHORT":
                # ── İyileştirme #3: Trailing Stop Logic ──────────────────
                if use_trail:
                    unrealized = entry - low  # En yüksek kâr (short)
                    if unrealized >= 2.0 * entry_atr:
                        new_sl = entry - 1.0 * entry_atr
                        if new_sl < active_trade['sl']:
                            active_trade['sl'] = new_sl
                            sl = new_sl
                    elif unrealized >= 1.0 * entry_atr:
                        new_sl = entry
                        if new_sl < active_trade['sl']:
                            active_trade['sl'] = new_sl
                            sl = new_sl
                # ─────────────────────────────────────────────────────────

                if high >= sl:
                    exit_price  = sl
                    exit_reason = "SL"
                elif low <= tp3:
                    exit_price  = tp3
                    exit_reason = "TP3"
                elif low <= tp2 and not active_trade.get('tp2_hit'):
                    active_trade['tp2_hit'] = True
                    if use_trail:
                        new_sl = entry - 1.5 * entry_atr
                        if new_sl < active_trade['sl']:
                            active_trade['sl'] = new_sl

            # Close Trade if exit triggered
            if exit_price is not None:
                if direction == "LONG":
                    pnl = (exit_price - entry) / entry * 100
                else:
                    pnl = (entry - exit_price) / entry * 100

                trades.append({
                    'coin':        coin,
                    'direction':   direction,
                    'entry_time':  active_trade['entry_time'],
                    'exit_time':   timestamp,
                    'entry_price': entry,
                    'exit_price':  exit_price,
                    'pnl':         pnl,
                    'reason':      exit_reason,
                    'score':       active_trade['score'],
                    'tp1_hit':     active_trade.get('tp1_hit', False),
                    'tp2_hit':     active_trade.get('tp2_hit', False),
                })
                active_trade = None
                continue

        # ==========================================
        # CHECK NEW SIGNAL
        # ==========================================
        if not active_trade:
            if row['score_long'] >= threshold:
                active_trade = {
                    'direction':   'LONG',
                    'entry_time':  timestamp,
                    'entry_price': price,
                    'entry_atr':   atr,
                    'sl':          price - (atr * sl_mult),
                    'tp1':         price + (atr * tp1_mult),
                    'tp2':         price + (atr * tp2_mult),
                    'tp3':         price + (atr * tp3_mult),
                    'score':       row['score_long'],
                    'tp1_hit':     False,
                    'tp2_hit':     False,
                }
            elif row['score_short'] >= threshold:
                active_trade = {
                    'direction':   'SHORT',
                    'entry_time':  timestamp,
                    'entry_price': price,
                    'entry_atr':   atr,
                    'sl':          price + (atr * sl_mult),
                    'tp1':         price - (atr * tp1_mult),
                    'tp2':         price - (atr * tp2_mult),
                    'tp3':         price - (atr * tp3_mult),
                    'score':       row['score_short'],
                    'tp1_hit':     False,
                    'tp2_hit':     False,
                }

    return pd.DataFrame(trades)


def print_report(trades_df: pd.DataFrame, label: str = ""):
    print(f"\n{'='*45}")
    print(f"  BACKTEST RAPORU — {label}")
    print(f"{'='*45}")

    if trades_df.empty:
        print("[WARN] Hiç işlem üretilmedi.")
        return

    total  = len(trades_df)
    wins   = len(trades_df[trades_df['pnl'] > 0])
    losses = len(trades_df[trades_df['pnl'] <= 0])
    wr     = (wins / total) * 100 if total > 0 else 0

    total_pnl = trades_df['pnl'].sum()
    avg_pnl   = trades_df['pnl'].mean()
    best      = trades_df['pnl'].max()
    worst     = trades_df['pnl'].min()

    # Max Drawdown (kümülatif)
    cumsum     = trades_df['pnl'].cumsum()
    rolling_max = cumsum.cummax()
    dd_series   = cumsum - rolling_max
    max_dd      = dd_series.min()

    # Profit factor
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss   = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    sl_count  = len(trades_df[trades_df['reason'] == 'SL'])
    tp3_count = len(trades_df[trades_df['reason'] == 'TP3'])

    print(f"  Toplam İşlem    : {total}")
    print(f"  Kazanç / Kayıp  : {wins} / {losses}")
    print(f"  Win Rate        : {wr:.1f}%")
    print(f"  ─────────────────────────────────")
    print(f"  Toplam PnL      : {total_pnl:+.2f}%")
    print(f"  Ortalama PnL    : {avg_pnl:+.2f}%")
    print(f"  En İyi İşlem    : {best:+.2f}%")
    print(f"  En Kötü İşlem   : {worst:+.2f}%")
    print(f"  Max Drawdown    : {max_dd:.2f}%")
    print(f"  Profit Factor   : {pf:.2f}x")
    print(f"  ─────────────────────────────────")
    print(f"  SL'e Çarpan     : {sl_count} ({sl_count/total*100:.0f}%)")
    print(f"  TP3'e Ulaşan    : {tp3_count} ({tp3_count/total*100:.0f}%)")

    if 'direction' in trades_df.columns:
        print(f"\n  Yön Dağılımı:")
        breakdown = trades_df.groupby('direction')['pnl'].agg(['count', 'mean', 'sum'])
        breakdown.columns = ['Adet', 'Ort PnL', 'Toplam PnL']
        print(breakdown.to_string())


def _make_mock_data(periods=1500):
    """Gerçekçi mock OHLCV veri üret (trend + ranging bölgeler)."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=periods, freq='h')

    # Fiyat yolculuğu: Trend + Ranging dönemleri
    price = 45000.0
    prices = []
    for i in range(periods):
        phase = (i // 200) % 3
        if phase == 0:   # Yükseliş trendi
            drift = 0.0008
        elif phase == 1: # Düşüş trendi
            drift = -0.0005
        else:            # Yatay hareket
            drift = 0.0

        noise = np.random.randn() * 0.012
        price = price * (1 + drift + noise)
        price = max(price, 100)
        prices.append(price)

    closes = np.array(prices)
    opens  = closes * (1 + np.random.randn(periods) * 0.003)
    highs  = np.maximum(closes, opens) * (1 + abs(np.random.randn(periods) * 0.005))
    lows   = np.minimum(closes, opens) * (1 - abs(np.random.randn(periods) * 0.005))
    vols   = np.random.rand(periods) * 5000 + 1000
    # Hacim spike'ları
    spike_idx = np.random.choice(periods, size=80, replace=False)
    vols[spike_idx] *= np.random.uniform(2, 5, size=80)

    df = pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows,
        'close': closes, 'volume': vols
    }, index=dates)
    return df


def main():
    parser = argparse.ArgumentParser(description="TITANIUM Backtest Engine V7.0")
    parser.add_argument("coin", nargs="?", default="BTC")
    parser.add_argument("--baseline", action="store_true", help="Sadece baseline çalıştır")
    parser.add_argument("--improved", action="store_true", help="Sadece improved çalıştır")
    parser.add_argument("--compare", action="store_true", help="Her ikisini karşılaştır (varsayılan)")
    args = parser.parse_args()

    target_coin = args.coin
    data_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", f"{target_coin}_1h.csv"
    )

    if os.path.exists(data_file):
        print(f"[INFO] Gerçek veri yükleniyor: {data_file}")
        df = pd.read_csv(data_file, parse_dates=['date'], index_col='date')
    else:
        print(f"[INFO] Gerçek veri bulunamadı, gerçekçi mock veri üretiliyor ({1500} mum)...")
        df = _make_mock_data(1500)

    run_baseline = args.baseline or args.compare or (not args.improved)
    run_improved = args.improved or args.compare or (not args.baseline)

    baseline_trades = None
    improved_trades = None

    if run_baseline:
        print(f"\n{'─'*45}")
        print(f"  📊 BASELINE (Eski Ayarlar)")
        print(f"  SL: 2.0×ATR | TP1: 2.0×ATR | Trailing: ❌")
        print(f"  Eşik: {BASELINE_CONFIG['signal_threshold']} puan")
        print(f"{'─'*45}")
        baseline_trades = run_backtest(df.copy(), target_coin, BASELINE_CONFIG)
        print_report(baseline_trades, "BASELINE (Eski Ayarlar)")

    if run_improved:
        print(f"\n{'─'*45}")
        print(f"  🚀 IMPROVED (Broker İyileştirmeleri V7.0)")
        print(f"  SL: 1.5×ATR | TP1: 2.5×ATR | Trailing: ✅")
        print(f"  Eşik: {IMPROVED_CONFIG['signal_threshold']} puan")
        print(f"{'─'*45}")
        improved_trades = run_backtest(df.copy(), target_coin, IMPROVED_CONFIG)
        print_report(improved_trades, "IMPROVED (V7.0)")

    # Karşılaştırma
    if baseline_trades is not None and improved_trades is not None:
        print(f"\n{'='*45}")
        print(f"  ⚖️  KARŞILAŞTIRMA TABLOSU")
        print(f"{'='*45}")

        def safe(df, col, fn):
            try:
                return fn(df[col])
            except Exception:
                return 0

        b_wr = (len(baseline_trades[baseline_trades['pnl'] > 0]) / max(len(baseline_trades), 1)) * 100 if not baseline_trades.empty else 0
        i_wr = (len(improved_trades[improved_trades['pnl'] > 0]) / max(len(improved_trades), 1)) * 100 if not improved_trades.empty else 0

        b_pnl = safe(baseline_trades, 'pnl', sum)
        i_pnl = safe(improved_trades, 'pnl', sum)

        b_dd = (baseline_trades['pnl'].cumsum() - baseline_trades['pnl'].cumsum().cummax()).min() if not baseline_trades.empty else 0
        i_dd = (improved_trades['pnl'].cumsum() - improved_trades['pnl'].cumsum().cummax()).min() if not improved_trades.empty else 0

        b_cnt = len(baseline_trades)
        i_cnt = len(improved_trades)

        def arrow(new, old, higher_is_better=True):
            if higher_is_better:
                return "✅ ↑" if new > old else ("🔴 ↓" if new < old else "⚪")
            else:
                return "✅ ↓" if new < old else ("🔴 ↑" if new > old else "⚪")

        print(f"  {'Metrik':<24} {'Baseline':>10} {'Improved':>10} {'Sonuç':>6}")
        print(f"  {'─'*54}")
        print(f"  {'İşlem Sayısı':<24} {b_cnt:>10} {i_cnt:>10} {arrow(i_cnt, b_cnt, True):>6}")
        print(f"  {'Win Rate %':<24} {b_wr:>9.1f}% {i_wr:>9.1f}% {arrow(i_wr, b_wr, True):>6}")
        print(f"  {'Toplam PnL %':<24} {b_pnl:>+10.2f} {i_pnl:>+10.2f} {arrow(i_pnl, b_pnl, True):>6}")
        print(f"  {'Max Drawdown %':<24} {b_dd:>+10.2f} {i_dd:>+10.2f} {arrow(i_dd, b_dd, False):>6}")
        print(f"{'='*45}")


if __name__ == "__main__":
    main()
