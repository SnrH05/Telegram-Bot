
import sqlite3
import pandas as pd
from datetime import datetime
import asyncio
import os

# Mock broadcast
async def broadcast_message(text, parse_mode=None, disable_web_page_preview=None):
    print("="*20 + " TELEGRAM MESSAGE " + "="*20)
    print(text)
    print("="*60)

async def gunluk_rapor_test():
    try:
        bugun = datetime.now().strftime("%Y-%m-%d")
        print(f"Generating report for: {bugun}")

        with sqlite3.connect("titanium_live.db") as conn:
            query = """
            SELECT coin, yon, durum, pnl_yuzde, tp1_hit, tp2_hit, kapanis_zamani 
            FROM islemler 
            WHERE durum IN ('KAZANDI', 'KAYBETTI', 'PARTIAL') 
            AND date(kapanis_zamani) = ?
            """
            df_rapor = pd.read_sql_query(query, conn, params=(bugun,))

        if df_rapor.empty:
            print("No trades found for today.")
            return

        toplam_pnl = df_rapor['pnl_yuzde'].sum()
        
        full_win = len(df_rapor[df_rapor['durum'] == 'KAZANDI'])
        partial_win = len(df_rapor[df_rapor['durum'] == 'PARTIAL'])
        loss_count = len(df_rapor[df_rapor['durum'] == 'KAYBETTI'])
        total_count = len(df_rapor)
        
        win_count = full_win + partial_win
        win_rate = (win_count / total_count) * 100 if total_count > 0 else 0
        pnl_ikon = "✅" if toplam_pnl > 0 else "🔻"
        
        mesaj = f"📅 <b>GÜNLÜK RAPOR ({bugun})</b>\n\n"
        
        for index, row in df_rapor.iterrows():
            is_win = row['durum'] in ['KAZANDI', 'PARTIAL']
            wl = "W" if is_win else "L"
            
            tp_list = []
            if row.get('tp1_hit', 0): tp_list.append("TP1")
            if row.get('tp2_hit', 0): tp_list.append("TP2")
            if row['durum'] == 'KAZANDI': tp_list.append("TP3")
            
            tp_str = ",".join(tp_list) if tp_list else "-"
            
            pnl_val = row['pnl_yuzde']
            pnl_str = f"+{pnl_val:.1f}" if pnl_val >= 0 else f"{pnl_val:.1f}"
            
            mesaj += f"<code>{row['coin'][:4]:<5}|{row['yon'][0]}|{wl}|{tp_str}|{pnl_str}%</code>\n"
        
        mesaj += f"\n📊 <b>İSTATİSTİKLER</b>\n"
        mesaj += f"🏆 <b>Tam Kazanç:</b> {full_win} | ⚡ <b>Kısmi:</b> {partial_win} | ❌ <b>Kayıp:</b> {loss_count}\n"
        mesaj += f"🔢 <b>Toplam:</b> {total_count} | 🎯 <b>WR:</b> %{win_rate:.0f}\n"
        mesaj += f"💰 <b>NET PNL:</b> {pnl_ikon} <b>%{toplam_pnl:.2f}</b>\n"
        
        await broadcast_message(text=mesaj)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(gunluk_rapor_test())
