"""
TITANIUM Bot - Risk Manager Unit Testleri
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from risk_manager import KillSwitch, DrawdownMonitor, DailyLimitTracker, RiskManager


class TestKillSwitch:
    """KillSwitch testleri"""
    
    @pytest.fixture
    def kill_switch(self, tmp_path):
        """Test için KillSwitch instance"""
        db_path = str(tmp_path / "test.db")
        return KillSwitch(db_path=db_path)
    
    @pytest.fixture
    def normal_btc_data(self):
        """Normal volatilite BTC verisi"""
        n = 100
        close = pd.Series([50000 + np.random.randn() * 100 for _ in range(n)])
        high = close + 50
        low = close - 50
        
        return pd.DataFrame({
            'open': close,
            'high': high,
            'low': low,
            'close': close
        })
    
    @pytest.fixture
    def extreme_btc_data(self):
        """Aşırı volatilite BTC verisi (crash simülasyonu)"""
        n = 100
        # Son 10 mumda %10 düşüş
        close = [50000] * 90 + [50000 - i * 500 for i in range(10)]
        close = pd.Series(close)
        high = close + 100
        low = close - 100
        
        return pd.DataFrame({
            'open': close,
            'high': high,
            'low': low,
            'close': close
        })
    
    def test_kill_switch_normal(self, kill_switch, normal_btc_data):
        """Normal koşullarda kill switch aktif olmamalı"""
        # ATR hesapla
        df = normal_btc_data.copy()
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = abs(df['high'] - df['close'].shift(1))
        df['l-pc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        atr = df['tr'].rolling(14).mean()
        
        should_halt, reason = kill_switch.check_volatility(normal_btc_data, atr)
        assert not should_halt, "Normal koşullarda halt olmamalı"
    
    def test_kill_switch_initial_state(self, kill_switch):
        """Başlangıçta kill switch aktif olmamalı"""
        assert not kill_switch.is_active, "Başlangıçta aktif olmamalı"
    
    def test_kill_switch_force_deactivate(self, kill_switch):
        """Manuel deaktive edilebilmeli"""
        kill_switch.is_active = True
        kill_switch.force_deactivate()
        assert not kill_switch.is_active, "Deaktive edilebilmeli"


class TestDrawdownMonitor:
    """DrawdownMonitor testleri"""
    
    @pytest.fixture
    def monitor(self, tmp_path):
        """Test için DrawdownMonitor"""
        db_path = str(tmp_path / "test.db")
        return DrawdownMonitor(initial_equity=1000.0, db_path=db_path)
    
    def test_initial_drawdown_zero(self, monitor):
        """Başlangıçta drawdown 0 olmalı"""
        assert monitor.get_drawdown() == 0, "Başlangıç drawdown 0 olmalı"
    
    def test_drawdown_calculation(self, monitor):
        """Drawdown doğru hesaplanmalı"""
        monitor.update_equity(1000)  # Peak
        monitor.update_equity(900)   # %10 düşüş
        
        dd = monitor.get_drawdown()
        assert abs(dd - 10.0) < 0.1, f"Drawdown %10 olmalı, got {dd}"
    
    def test_drawdown_levels(self, monitor):
        """Drawdown seviyeleri doğru tespit edilmeli"""
        # %10 altı - NORMAL
        monitor.update_equity(1000)
        monitor.update_equity(950)
        action, multiplier = monitor.check_status()
        assert action == "NORMAL", "<%10 = NORMAL"
        
        # %10-15 arası - REDUCE_SIZE
        monitor.peak_equity = 1000
        monitor.current_equity = 880
        action, multiplier = monitor.check_status()
        assert action == "REDUCE_SIZE", "%10-15 = REDUCE_SIZE"
        
        # %15-20 arası - HALT
        monitor.current_equity = 820
        action, multiplier = monitor.check_status()
        assert action == "HALT", "%15-20 = HALT"
        
        # >%20 - EMERGENCY
        monitor.current_equity = 750
        action, multiplier = monitor.check_status()
        assert action == "EMERGENCY", ">%20 = EMERGENCY"


class TestDailyLimitTracker:
    """DailyLimitTracker testleri"""
    
    @pytest.fixture
    def tracker(self, tmp_path):
        """Test için DailyLimitTracker"""
        db_path = str(tmp_path / "test.db")
        return DailyLimitTracker(db_path=db_path)
    
    def test_initial_not_halted(self, tracker):
        """Başlangıçta halt olmamalı"""
        assert tracker.halted_until is None, "Başlangıçta halt olmamalı"
    
    def test_halt_until_tomorrow(self, tracker):
        """Yarına kadar halt ayarlanabilmeli"""
        tracker._halt_until_tomorrow()
        assert tracker.halted_until is not None, "Halt ayarlanmalı"
        assert tracker.halted_until > datetime.now(), "Halt gelecekte olmalı"


class TestRiskManager:
    """Unified RiskManager testleri"""
    
    @pytest.fixture
    def risk_manager(self, tmp_path):
        """Test için RiskManager"""
        db_path = str(tmp_path / "test.db")
        return RiskManager(initial_equity=1000.0, db_path=db_path)
    
    def test_initial_can_trade(self, risk_manager):
        """Başlangıçta trade yapılabilmeli"""
        # Basit veri oluştur
        n = 100
        close = pd.Series([50000] * n)
        df = pd.DataFrame({
            'open': close,
            'high': close + 100,
            'low': close - 100,
            'close': close
        })
        
        df_copy = df.copy()
        df_copy['h-l'] = df_copy['high'] - df_copy['low']
        df_copy['h-pc'] = abs(df_copy['high'] - df_copy['close'].shift(1))
        df_copy['l-pc'] = abs(df_copy['low'] - df_copy['close'].shift(1))
        df_copy['tr'] = df_copy[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        atr = df_copy['tr'].rolling(14).mean()
        
        can_trade, multiplier, reason = risk_manager.pre_signal_check(df, atr)
        assert can_trade, f"Başlangıçta trade yapılabilmeli, reason: {reason}"
    
    def test_status_summary(self, risk_manager):
        """Status özeti döndürmeli"""
        summary = risk_manager.get_status_summary()
        
        assert 'current_drawdown' in summary
        assert 'daily_pnl' in summary
        assert 'kill_switch_active' in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
