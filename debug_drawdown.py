
import sqlite3
import pandas as pd

INITIAL_EQUITY = 1000.0

try:
    with sqlite3.connect("titanium_live.db") as conn:
        # Get total PnL percentage sum
        cursor = conn.execute("""
            SELECT COALESCE(SUM(pnl_yuzde), 0) 
            FROM islemler 
            WHERE durum IN ('KAZANDI', 'KAYBETTI', 'PARTIAL')
        """)
        total_pnl_pct = cursor.fetchone()[0]
        
        current_equity = INITIAL_EQUITY * (1 + total_pnl_pct / 100)
        
        # Calculate max equity seen (simplified, assumes checking against 1000)
        # To be accurate like DrawdownMonitor, we would need to track equity history,
        # but DrawdownMonitor tracks peak equity in memory (or assumes start is peak if no gains).
        # Let's check if the total PnL is negative enough.
        
        peak_equity = max(INITIAL_EQUITY, current_equity) 
        # But DrawdownMonitor might have seen a higher peak if PnL was positive at some point.
        # However, checking the current total PnL is the best proxy we have for persistent state.
        
        # DrawdownMonitor logic:
        # return ((self.peak_equity - self.current_equity) / self.peak_equity) * 100
        
        # If the bot restarted, peak_equity resets to current_equity unless stored?
        # DrawdownMonitor in risk_manager.py initializes with initial_equity=1000.
        # It updates equity from DB *sum of all trades*.
        # So peak_equity starts at 1000. If current_equity > 1000, peak updates.
        
        if current_equity > INITIAL_EQUITY:
            peak_equity = current_equity
        else:
            peak_equity = INITIAL_EQUITY
            
        drawdown = ((peak_equity - current_equity) / peak_equity) * 100
        
        print(f"Total PnL % Sum: {total_pnl_pct:.2f}%")
        print(f"Current Equity (Estimated): {current_equity:.2f}")
        print(f"Peak Equity (Estimated): {peak_equity:.2f}")
        print(f"Calculated Drawdown: {drawdown:.2f}%")
        
        if drawdown >= 15.0:
            print("ALERT: Drawdown is >= 15% (LEVEL 2 HALT)")
        elif drawdown >= 10.0:
            print("WARNING: Drawdown is >= 10% (LEVEL 1 REDUCE)")
        else:
            print("STATUS: NORMAL")
            
except Exception as e:
    print(f"Error: {e}")
