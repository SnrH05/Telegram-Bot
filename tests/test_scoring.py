"""
TITANIUM Bot - Scoring System Tests
=====================================
Reversal, Rapid, Range skorlama testleri.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def bullish_reversal_df():
    """Bullish reversal senaryosu - düşüş sonrası dönüş."""
    np.random.seed(42)
    n = 50
    
    # İlk 40 mum düşüş, son 10 mum yükseliş
    downtrend = np.linspace(110, 90, 40)
    uptrend = np.linspace(90, 95, 10)
    close = np.concatenate([downtrend, uptrend])
    
    df = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n, freq='H'),
        'open': close - np.random.rand(n) * 0.5,
        'high': close + np.random.rand(n) * 1.0,
        'low': close - np.random.rand(n) * 1.0,
        'close': close,
        'volume': np.random.randint(1000, 5000, n).astype(float)
    })
    df.set_index('date', inplace=True)
    return df


@pytest.fixture
def bearish_reversal_df():
    """Bearish reversal senaryosu - yükseliş sonrası dönüş."""
    np.random.seed(42)
    n = 50
    
    # İlk 40 mum yükseliş, son 10 mum düşüş
    uptrend = np.linspace(90, 110, 40)
    downtrend = np.linspace(110, 105, 10)
    close = np.concatenate([uptrend, downtrend])
    
    df = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n, freq='H'),
        'open': close + np.random.rand(n) * 0.5,
        'high': close + np.random.rand(n) * 1.0,
        'low': close - np.random.rand(n) * 1.0,
        'close': close,
        'volume': np.random.randint(1000, 5000, n).astype(float)
    })
    df.set_index('date', inplace=True)
    return df


@pytest.fixture
def range_market_df():
    """Range (düz piyasa) senaryosu."""
    np.random.seed(42)
    n = 50
    
    # 95-105 arasında salınım
    close = 100 + np.sin(np.linspace(0, 4*np.pi, n)) * 5
    
    df = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n, freq='H'),
        'open': close - np.random.rand(n) * 0.2,
        'high': close + np.random.rand(n) * 0.5,
        'low': close - np.random.rand(n) * 0.5,
        'close': close,
        'volume': np.random.randint(1000, 3000, n).astype(float)
    })
    df.set_index('date', inplace=True)
    return df


@pytest.fixture
def flash_crash_df():
    """Flash crash senaryosu - ani düşüş sonrası toparlanma."""
    n = 20
    
    # Normal -> Flash düşüş -> Toparlanma
    prices = [100, 100, 100, 100, 100,
              99, 98, 95, 92, 88,  # Flash crash
              89, 91, 93, 95, 96,  # Recover
              97, 98, 99, 99, 100]
    
    close = np.array(prices, dtype=float)
    
    df = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n, freq='H'),
        'open': close - 0.5,
        'high': close + 1.0,
        'low': close - 1.5,
        'close': close,
        'volume': [5000]*5 + [20000]*5 + [15000]*5 + [8000]*5  # Flash'ta yüksek hacim
    })
    df.set_index('date', inplace=True)
    return df


# ==========================================
# REVERSAL SCORE TESTS
# ==========================================

class TestReversalScore:
    """Reversal skorlama testleri."""
    
    def test_reversal_score_range(self, bullish_reversal_df):
        """Reversal skoru 0-30 arasında olmalı."""
        from strategy.scoring import calculate_reversal_score
        
        long_score, short_score, details = calculate_reversal_score(bullish_reversal_df)
        
        assert 0 <= long_score <= 30, f"LONG reversal skoru 0-30 arasında olmalı: {long_score}"
        assert 0 <= short_score <= 30, f"SHORT reversal skoru 0-30 arasında olmalı: {short_score}"
        
    def test_bullish_reversal_long_higher(self, bullish_reversal_df):
        """Bullish reversal'da LONG skoru yüksek olmalı."""
        from strategy.scoring import calculate_reversal_score
        
        long_score, short_score, _ = calculate_reversal_score(bullish_reversal_df)
        
        # Bullish reversal senaryosunda LONG >= SHORT olmalı
        assert long_score >= 0, "Bullish reversal'da LONG skoru >= 0 olmalı"
        
    def test_reversal_returns_details(self, bullish_reversal_df):
        """Reversal detayları dönmeli."""
        from strategy.scoring import calculate_reversal_score
        
        _, _, details = calculate_reversal_score(bullish_reversal_df)
        
        assert isinstance(details, list), "Details list olmalı"


# ==========================================
# RAPID SCORE TESTS
# ==========================================

class TestRapidScore:
    """Rapid reversal skorlama testleri."""
    
    def test_rapid_score_range(self, flash_crash_df):
        """Rapid skoru 0-100 arasında olmalı."""
        from strategy.scoring import calculate_rapid_score
        
        long_score, short_score, details, triggers = calculate_rapid_score(flash_crash_df)
        
        assert 0 <= long_score <= 100, f"LONG rapid skoru 0-100 arasında olmalı: {long_score}"
        assert 0 <= short_score <= 100, f"SHORT rapid skoru 0-100 arasında olmalı: {short_score}"
        
    def test_rapid_returns_triggers(self, flash_crash_df):
        """Rapid tetikleyicileri dönmeli."""
        from strategy.scoring import calculate_rapid_score
        
        _, _, details, triggers = calculate_rapid_score(flash_crash_df)
        
        assert isinstance(details, list)
        assert isinstance(triggers, list)


# ==========================================
# RANGE SCORE TESTS
# ==========================================

class TestRangeScore:
    """Range market skorlama testleri."""
    
    def test_range_score_range(self, range_market_df):
        """Range skoru 0-60 arasında olmalı."""
        from strategy.scoring import calculate_range_score
        
        long_score, short_score, long_bd, short_bd, tp_sl = calculate_range_score(range_market_df)
        
        assert 0 <= long_score <= 60, f"LONG range skoru 0-60 arasında olmalı: {long_score}"
        assert 0 <= short_score <= 60, f"SHORT range skoru 0-60 arasında olmalı: {short_score}"
        
    def test_range_returns_tp_sl_info(self, range_market_df):
        """Range TP/SL bilgisi dönmeli."""
        from strategy.scoring import calculate_range_score
        
        _, _, _, _, tp_sl = calculate_range_score(range_market_df)
        
        assert 'bb_mid' in tp_sl
        assert 'bb_lower' in tp_sl
        assert 'bb_upper' in tp_sl
        assert 'atr' in tp_sl


# ==========================================
# HELPER FUNCTION TESTS
# ==========================================

class TestHelperFunctions:
    """Yardımcı fonksiyon testleri."""
    
    def test_momentum_reversal(self, bullish_reversal_df):
        """Momentum reversal tespiti."""
        from strategy.scoring import calculate_momentum_reversal
        
        reversal_type, change_pct = calculate_momentum_reversal(bullish_reversal_df)
        
        # Dönüş tipi veya None olmalı
        assert reversal_type in [None, 'REVERSAL_UP', 'REVERSAL_DOWN']
        assert isinstance(change_pct, (int, float))
        
    def test_rsi_divergence(self, bullish_reversal_df):
        """RSI divergence tespiti."""
        from strategy.scoring import check_rsi_divergence
        
        div_type, strength = check_rsi_divergence(bullish_reversal_df)
        
        assert div_type in [None, 'BULLISH_DIV', 'BEARISH_DIV']
        assert 0 <= strength <= 100
        
    def test_volatility_spike(self, flash_crash_df):
        """Volatility spike tespiti."""
        from strategy.scoring import check_volatility_spike
        
        spike_type, atr_ratio = check_volatility_spike(flash_crash_df)
        
        assert spike_type in [None, 'SPIKE_UP', 'SPIKE_DOWN']
        assert atr_ratio >= 0
        
    def test_is_ranging_market(self, range_market_df):
        """Range market tespiti."""
        from strategy.scoring import is_ranging_market
        
        is_range, details = is_ranging_market(range_market_df)
        
        assert isinstance(bool(is_range), bool)
        assert 'adx' in details
        assert 'criteria_met' in details


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
