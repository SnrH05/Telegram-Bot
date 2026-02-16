"""Win rate analysis script - examine trade outcomes and patterns"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

try:
    with sqlite3.connect("titanium_live.db") as conn:
        # 1. Overall stats
        df_all = pd.read_sql_query("""
            SELECT coin, yon, durum, pnl_yuzde, tp1_hit, tp2_hit, 
                   giris_fiyat, tp1, tp2, tp3, sl,
                   sinyal_zamani, kapanis_zamani
            FROM islemler 
            WHERE durum IN ('KAZANDI', 'KAYBETTI', 'PARTIAL')
            ORDER BY kapanis_zamani DESC
        """, conn)
        
        if df_all.empty:
            print("No closed trades found!")
        else:
            total = len(df_all)
            wins = len(df_all[df_all['durum'] == 'KAZANDI'])
            losses = len(df_all[df_all['durum'] == 'KAYBETTI'])
            partials = len(df_all[df_all['durum'] == 'PARTIAL'])
            
            print("=" * 60)
            print("TITANIUM BOT - WIN RATE ANALİZİ")
            print("=" * 60)
            print(f"\nToplam İşlem: {total}")
            print(f"  ✅ Kazandı: {wins} ({wins/total*100:.1f}%)")
            print(f"  ❌ Kaybetti: {losses} ({losses/total*100:.1f}%)")
            print(f"  ⚠️ Partial: {partials} ({partials/total*100:.1f}%)")
            print(f"\nWin Rate (K+P): {(wins+partials)/total*100:.1f}%")
            print(f"Pure Win Rate: {wins/total*100:.1f}%")
            
            # 2. PnL stats
            avg_pnl = df_all['pnl_yuzde'].mean()
            avg_win = df_all[df_all['pnl_yuzde'] > 0]['pnl_yuzde'].mean() if len(df_all[df_all['pnl_yuzde'] > 0]) > 0 else 0
            avg_loss = df_all[df_all['pnl_yuzde'] < 0]['pnl_yuzde'].mean() if len(df_all[df_all['pnl_yuzde'] < 0]) > 0 else 0
            total_pnl = df_all['pnl_yuzde'].sum()
            
            print(f"\n--- PnL İstatistikleri ---")
            print(f"Toplam PnL: {total_pnl:.2f}%")
            print(f"Ortalama PnL: {avg_pnl:.2f}%")
            print(f"Ortalama Kazanç: +{avg_win:.2f}%")
            print(f"Ortalama Kayıp: {avg_loss:.2f}%")
            
            # 3. By direction
            print(f"\n--- Yön Bazlı ---")
            for yon in ['LONG', 'SHORT']:
                df_yon = df_all[df_all['yon'] == yon]
                if len(df_yon) > 0:
                    yon_wins = len(df_yon[df_yon['durum'] == 'KAZANDI'])
                    yon_losses = len(df_yon[df_yon['durum'] == 'KAYBETTI'])
                    yon_partials = len(df_yon[df_yon['durum'] == 'PARTIAL'])
                    yon_wr = (yon_wins + yon_partials) / len(df_yon) * 100
                    yon_pnl = df_yon['pnl_yuzde'].sum()
                    print(f"  {yon}: {len(df_yon)} işlem | WR: {yon_wr:.1f}% | PnL: {yon_pnl:+.2f}%")
                    print(f"    K:{yon_wins} L:{yon_losses} P:{yon_partials}")
            
            # 4. By coin - worst performers
            print(f"\n--- Coin Bazlı (En Kötü) ---")
            coin_stats = df_all.groupby('coin').agg(
                trades=('durum', 'count'),
                pnl_total=('pnl_yuzde', 'sum'),
                losses=('durum', lambda x: (x == 'KAYBETTI').sum())
            ).sort_values('pnl_total')
            print(coin_stats.head(10))
            
            # 5. Recent performance (last 20 trades)
            print(f"\n--- Son 20 İşlem ---")
            recent = df_all.head(20)
            recent_wins = len(recent[recent['durum'] == 'KAZANDI'])
            recent_losses = len(recent[recent['durum'] == 'KAYBETTI'])
            recent_partials = len(recent[recent['durum'] == 'PARTIAL'])
            recent_pnl = recent['pnl_yuzde'].sum()
            recent_wr = (recent_wins + recent_partials) / len(recent) * 100 if len(recent) > 0 else 0
            print(f"WR: {recent_wr:.1f}% | PnL: {recent_pnl:+.2f}%")
            print(f"K:{recent_wins} L:{recent_losses} P:{recent_partials}")
            
            # 6. SL analysis - how tight are stops?
            print(f"\n--- SL Analizi ---")
            df_all['sl_distance_pct'] = abs(df_all['giris_fiyat'] - df_all['sl']) / df_all['giris_fiyat'] * 100
            df_all['tp1_distance_pct'] = abs(df_all['tp1'] - df_all['giris_fiyat']) / df_all['giris_fiyat'] * 100
            
            avg_sl_dist = df_all['sl_distance_pct'].mean()
            avg_tp1_dist = df_all['tp1_distance_pct'].mean()
            rr_ratio = avg_tp1_dist / avg_sl_dist if avg_sl_dist > 0 else 0
            
            print(f"Ortalama SL Mesafesi: {avg_sl_dist:.2f}%")
            print(f"Ortalama TP1 Mesafesi: {avg_tp1_dist:.2f}%")
            print(f"Risk/Reward Oranı (TP1): {rr_ratio:.2f}")
            
            # 7. TP hit rates
            print(f"\n--- TP Hit Oranları ---")
            tp1_hits = df_all['tp1_hit'].sum()
            tp2_hits = df_all['tp2_hit'].sum()
            print(f"TP1 Hit: {tp1_hits}/{total} ({tp1_hits/total*100:.1f}%)")
            print(f"TP2 Hit: {tp2_hits}/{total} ({tp2_hits/total*100:.1f}%)")
            
            # 8. Time analysis  
            print(f"\n--- Haftalık Performans ---")
            df_all['kapanis_zamani'] = pd.to_datetime(df_all['kapanis_zamani'], errors='coerce')
            df_all['week'] = df_all['kapanis_zamani'].dt.isocalendar().week
            weekly = df_all.groupby('week').agg(
                trades=('durum', 'count'),
                wins=('durum', lambda x: (x == 'KAZANDI').sum()),
                losses=('durum', lambda x: (x == 'KAYBETTI').sum()),
                pnl=('pnl_yuzde', 'sum')
            )
            for idx, row in weekly.iterrows():
                wr = row['wins'] / row['trades'] * 100 if row['trades'] > 0 else 0
                print(f"  Hafta {idx}: {row['trades']} işlem | WR: {wr:.0f}% | PnL: {row['pnl']:+.2f}%")
            
            # 9. Last 10 trades detail
            print(f"\n--- Son 10 İşlem Detay ---")
            for _, t in df_all.head(10).iterrows():
                pnl_icon = "✅" if t['pnl_yuzde'] > 0 else "❌"
                print(f"  {pnl_icon} {t['coin']:6s} {t['yon']:5s} | PnL: {t['pnl_yuzde']:+.2f}% | SL%: {t['sl_distance_pct']:.2f}% | {t['durum']} | {t['kapanis_zamani']}")
        
        # 10. Open positions
        df_open = pd.read_sql_query("SELECT coin, yon, giris_fiyat, sl, tp1 FROM islemler WHERE durum='ACIK'", conn)
        print(f"\n--- Açık Pozisyonlar: {len(df_open)} ---")
        if not df_open.empty:
            for _, p in df_open.iterrows():
                sl_pct = abs(p['giris_fiyat'] - p['sl']) / p['giris_fiyat'] * 100
                print(f"  {p['coin']:6s} {p['yon']:5s} | Giriş: ${p['giris_fiyat']:.4f} | SL%: {sl_pct:.2f}%")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
