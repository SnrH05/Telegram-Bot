"""
TITANIUM Risk Manager Module
============================
Production-grade risk management for live crypto trading.

Components:
1. KillSwitch - Halt trading during extreme volatility
2. DrawdownMonitor - Track and limit portfolio drawdown
3. DailyLimitTracker - Enforce daily loss limits

Author: TITANIUM Bot V5.9
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional
import sqlite3


class KillSwitch:
    """
    Volatility-based trading halt mechanism.
    
    Triggers:
    - ATR Z-score > 3.0 (volatility explosion)
    - BTC 1H change > ±5% (flash crash/pump)
    - BTC 4H change > ±8% (major move - emergency close all)
    """
    
    def __init__(self, db_path: str = "titanium_live.db"):
        self.db_path = db_path
        self.is_active = False
        self.activation_time: Optional[datetime] = None
        self.cooldown_hours = 2
        self.reason = ""
        
        # Thresholds
        self.ATR_Z_THRESHOLD = 3.0
        self.BTC_1H_THRESHOLD = 5.0  # %
        self.BTC_4H_THRESHOLD = 8.0  # %
        
    def check_volatility(self, df_btc: pd.DataFrame, atr_series: pd.Series) -> Tuple[bool, str]:
        """
        Check if volatility conditions trigger kill-switch.
        
        Args:
            df_btc: BTC OHLCV DataFrame (1H timeframe)
            atr_series: Pre-calculated ATR series
            
        Returns:
            (should_halt, reason)
        """
        # Skip if already active and in cooldown
        if self.is_active and self.activation_time:
            elapsed = (datetime.now() - self.activation_time).total_seconds() / 3600
            if elapsed < self.cooldown_hours:
                return True, f"Kill-switch active ({self.cooldown_hours - elapsed:.1f}h remaining)"
            else:
                self.is_active = False
                self.reason = ""
        
        # 1. ATR Z-Score Check
        if len(atr_series) >= 50:
            atr_current = atr_series.iloc[-1]
            atr_sma = atr_series.rolling(50).mean().iloc[-1]
            atr_std = atr_series.rolling(50).std().iloc[-1]
            
            if atr_std > 0:
                atr_z = (atr_current - atr_sma) / atr_std
                
                if atr_z > self.ATR_Z_THRESHOLD:
                    self._activate(f"ATR Z-Score: {atr_z:.2f} > {self.ATR_Z_THRESHOLD}")
                    return True, self.reason
        
        # 2. BTC 1H Change Check
        if len(df_btc) >= 2:
            btc_1h_change = ((df_btc['close'].iloc[-1] - df_btc['close'].iloc[-2]) / 
                            df_btc['close'].iloc[-2]) * 100
            
            if abs(btc_1h_change) > self.BTC_1H_THRESHOLD:
                self._activate(f"BTC 1H Flash: {btc_1h_change:+.2f}%")
                return True, self.reason
        
        # 3. BTC 4H Change Check (emergency - close all positions)
        if len(df_btc) >= 5:
            btc_4h_change = ((df_btc['close'].iloc[-1] - df_btc['close'].iloc[-5]) / 
                            df_btc['close'].iloc[-5]) * 100
            
            if abs(btc_4h_change) > self.BTC_4H_THRESHOLD:
                self._activate(f"BTC 4H EMERGENCY: {btc_4h_change:+.2f}%", emergency=True)
                return True, self.reason
        
        return False, ""
    
    def _activate(self, reason: str, emergency: bool = False):
        """Activate the kill-switch."""
        self.is_active = True
        self.activation_time = datetime.now()
        self.reason = reason
        self.cooldown_hours = 6 if emergency else 2
        print(f"🚨 KILL-SWITCH ACTIVATED: {reason}")
    
    def force_deactivate(self):
        """Manual override to deactivate kill-switch."""
        self.is_active = False
        self.activation_time = None
        self.reason = ""
        print("✅ Kill-switch manually deactivated")


class DrawdownMonitor:
    """
    Track portfolio drawdown and enforce limits.
    
    Levels:
    - 10% DD: Reduce position size by 50%
    - 15% DD: Halt all new signals
    - 20% DD: Close all positions + halt for 24h
    """
    
    def __init__(self, initial_equity: float = 1000.0, db_path: str = "titanium_live.db"):
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity
        self.current_equity = initial_equity
        self.db_path = db_path
        
        # Thresholds
        self.LEVEL_1_DD = 10.0  # Reduce size
        self.LEVEL_2_DD = 15.0  # Halt signals
        self.LEVEL_3_DD = 20.0  # Emergency close
        
        # State
        self.halt_until: Optional[datetime] = None
        
    def update_equity(self, new_equity: float):
        """Update current equity and peak."""
        self.current_equity = new_equity
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
            
    def get_drawdown(self) -> float:
        """Calculate current drawdown percentage."""
        if self.peak_equity <= 0:
            return 0.0
        return ((self.peak_equity - self.current_equity) / self.peak_equity) * 100
    
    def check_status(self) -> Tuple[str, float]:
        """
        Check drawdown status and return action.
        
        Returns:
            (action, position_size_multiplier)
            action: "NORMAL", "REDUCE_SIZE", "HALT", "EMERGENCY"
        """
        # Check if we're in a halt period
        if self.halt_until and datetime.now() < self.halt_until:
            remaining = (self.halt_until - datetime.now()).total_seconds() / 3600
            return "HALT", 0.0
        
        dd = self.get_drawdown()
        
        if dd >= self.LEVEL_3_DD:
            self.halt_until = datetime.now() + timedelta(hours=24)
            print(f"🚨 DRAWDOWN EMERGENCY: {dd:.1f}% - Halting for 24h")
            return "EMERGENCY", 0.0
        elif dd >= self.LEVEL_2_DD:
            print(f"⚠️ DRAWDOWN HIGH: {dd:.1f}% - Signals halted")
            return "HALT", 0.0
        elif dd >= self.LEVEL_1_DD:
            print(f"⚠️ DRAWDOWN WARNING: {dd:.1f}% - Reducing position size")
            return "REDUCE_SIZE", 0.5
        else:
            return "NORMAL", 1.0
    
    def calculate_equity_from_db(self) -> float:
        """Calculate current equity from database trades."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Sum all PnL from closed trades
                cursor = conn.execute("""
                    SELECT COALESCE(SUM(pnl_yuzde), 0) 
                    FROM islemler 
                    WHERE durum IN ('KAZANDI', 'KAYBETTI', 'PARTIAL')
                """)
                total_pnl_pct = cursor.fetchone()[0]
                
                # Calculate equity
                equity = self.initial_equity * (1 + total_pnl_pct / 100)
                self.update_equity(equity)
                return equity
        except Exception as e:
            print(f"⚠️ Equity calculation error: {e}")
            return self.current_equity


class DailyLimitTracker:
    """
    Track and enforce daily loss limits.
    
    Levels:
    - 5% daily loss: Halt signals for rest of day
    - 8% daily loss: Close positions + alert admin
    """
    
    def __init__(self, db_path: str = "titanium_live.db"):
        self.db_path = db_path
        self.LEVEL_1_LOSS = -5.0  # Halt signals
        self.LEVEL_2_LOSS = -8.0  # Emergency
        
        self.halted_until: Optional[datetime] = None
        
    def get_daily_pnl(self) -> float:
        """Get today's total PnL percentage."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT COALESCE(SUM(pnl_yuzde), 0)
                    FROM islemler
                    WHERE date(kapanis_zamani) = ?
                    AND durum IN ('KAZANDI', 'KAYBETTI', 'PARTIAL')
                """, (today,))
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"⚠️ Daily PnL calculation error: {e}")
            return 0.0
    
    def check_status(self) -> Tuple[str, bool]:
        """
        Check daily limit status.
        
        Returns:
            (action, should_close_positions)
        """
        # Check if already halted for today
        if self.halted_until:
            if datetime.now() < self.halted_until:
                return "HALTED", False
            else:
                self.halted_until = None
        
        daily_pnl = self.get_daily_pnl()
        
        if daily_pnl <= self.LEVEL_2_LOSS:
            print(f"🚨 DAILY LOSS EMERGENCY: {daily_pnl:.1f}% - Closing all positions!")
            self._halt_until_tomorrow()
            return "EMERGENCY", True
        elif daily_pnl <= self.LEVEL_1_LOSS:
            print(f"⚠️ DAILY LOSS LIMIT: {daily_pnl:.1f}% - Halting for today")
            self._halt_until_tomorrow()
            return "HALTED", False
        
        return "NORMAL", False
    
    def _halt_until_tomorrow(self):
        """Set halt until next day 00:00."""
        tomorrow = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow += timedelta(days=1)
        self.halted_until = tomorrow


class RiskManager:
    """
    Unified risk management interface.
    
    Combines all risk components into a single check.
    """
    
    def __init__(self, initial_equity: float = 1000.0, db_path: str = "titanium_live.db"):
        self.kill_switch = KillSwitch(db_path)
        self.drawdown_monitor = DrawdownMonitor(initial_equity, db_path)
        self.daily_limit = DailyLimitTracker(db_path)
        
    def pre_signal_check(self, df_btc: pd.DataFrame, atr_series: pd.Series) -> Tuple[bool, float, str]:
        """
        Run all risk checks before generating signals.
        
        Args:
            df_btc: BTC OHLCV DataFrame
            atr_series: ATR series for BTC
            
        Returns:
            (can_trade, position_size_multiplier, reason)
        """
        # 1. Kill-switch check
        kill_active, kill_reason = self.kill_switch.check_volatility(df_btc, atr_series)
        if kill_active:
            return False, 0.0, f"Kill-switch: {kill_reason}"
        
        # 2. Drawdown check
        self.drawdown_monitor.calculate_equity_from_db()
        dd_status, size_mult = self.drawdown_monitor.check_status()
        if dd_status in ("HALT", "EMERGENCY"):
            return False, 0.0, f"Drawdown: {dd_status}"
        
        # 3. Daily limit check
        daily_status, should_close = self.daily_limit.check_status()
        if daily_status in ("HALTED", "EMERGENCY"):
            return False, 0.0, f"Daily limit: {daily_status}"
        
        return True, size_mult, "OK"
    
    def get_status_summary(self) -> dict:
        """Get a summary of all risk metrics."""
        return {
            "kill_switch_active": self.kill_switch.is_active,
            "kill_switch_reason": self.kill_switch.reason,
            "current_drawdown": self.drawdown_monitor.get_drawdown(),
            "daily_pnl": self.daily_limit.get_daily_pnl(),
            "can_trade": not self.kill_switch.is_active
        }
