import sqlite3
import pandas as pd
from datetime import datetime

# Adjust this if your system time is different from what you expect for "today"
# The user's metadata says local time is 16:40+03:00 on 2026-02-05
bugun = datetime.now().strftime("%Y-%m-%d")
print(f"Checking trades for date: {bugun}")

try:
    with sqlite3.connect("titanium_live.db") as conn:
        query = """
        SELECT coin, yon, durum, pnl_yuzde, tp1_hit, tp2_hit, kapanis_zamani 
        FROM islemler 
        WHERE durum IN ('KAZANDI', 'KAYBETTI', 'PARTIAL') 
        AND date(kapanis_zamani) = ?
        """
        df_rapor = pd.read_sql_query(query, conn, params=(bugun,))
        
        if df_rapor.empty:
            print("No closed trades found for today.")
        else:
            print(f"Found {len(df_rapor)} closed trades.")
            print(df_rapor)
            
        # Also check open trades just in case
        query_open = "SELECT coin, yon, giris_fiyat FROM islemler WHERE durum='ACIK'"
        df_open = pd.read_sql_query(query_open, conn)
        print(f"\nOpen trades: {len(df_open)}")
        if not df_open.empty:
            print(df_open)

except Exception as e:
    print(f"Error accessing DB: {e}")
