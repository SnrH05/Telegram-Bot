"""
TITANIUM Bot - Strategy Tests (LONG & SHORT)
==============================================
Her strateji fonksiyonunun LONG ve SHORT sinyal üretimini test eder.
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
def uptrend_df():
    """Güçlü yükseliş trendi - EMA200 üzeri, EMA50'ye pullback."""
    np.random.seed(42)
    n = 250
    
    # Uzun vadeli yükseliş + son birkaç mumda pullback
    base = np.linspace(80, 120, n - 5)
    pullback = np.array([119.5, 118.8, 118.2, 118.5, 119.0])  # EMA50'ye yaklaşım
    close = np.concatenate([base, pullback])
    
    df = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=n, freq='h'),
        'open': close - np.random.rand(n) * 0.3,
        'high': close + np.random.rand(n) * 1.0,
        'low': close - np.random.rand(n) * 1.0,
        'close': close,
        'volume': np.random.randint(1000, 5000, n).astype(float)
    })
    df.set_index('date', inplace=True)
    return df


@pytest.fixture
def downtrend_df():
    """Güçlü düşüş trendi - EMA200 altı, EMA50'ye rally."""
    np.random.seed(42)
    n = 250
    
    # Uzun vadeli düşüş + son birkaç mumda rally (resistance'a yaklaşım)
    base = np.linspace(120, 80, n - 5)
    rally = np.array([80.5, 81.2, 81.8, 81.5, 81.0])  # EMA50'ye yaklaşım
    close = np.concatenate([base, rally])
    
    df = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=n, freq='h'),
        'open': close + np.random.rand(n) * 0.3,
        'high': close + np.random.rand(n) * 1.0,
        'low': close - np.random.rand(n) * 1.0,
        'close': close,
        'volume': np.random.randint(1000, 5000, n).astype(float)
    })
    df.set_index('date', inplace=True)
    return df


@pytest.fixture
def bb_lower_touch_df():
    """Bollinger alt bandına dokunan senaryo."""
    np.random.seed(42)
    n = 50
    
    # Yatay piyasa + son mumda ani düşüş (alt banda temas)
    base = np.full(n - 3, 100.0)
    dip = np.array([97.0, 96.0, 97.5])  # Alt banda dokunuş ve toparlanma
    close = np.concatenate([base, dip])
    
    df = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=n, freq='h'),
        'open': close + np.random.rand(n) * 0.3,
        'high': close + np.random.rand(n) * 1.5,
        'low': close - np.random.rand(n) * 2.0,
        'close': close,
        'volume': np.random.randint(1000, 5000, n).astype(float)
    })
    df.set_index('date', inplace=True)
    return df


@pytest.fixture
def volume_spike_up_df():
    """Hacim patlaması + yukarı yönlü senaryo."""
    np.random.seed(42)
    n = 50
    
    close = np.linspace(100, 110, n)
    volume = np.full(n, 1000.0)
    volume[-1] = 5000.0  # 5x hacim spike
    
    df = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=n, freq='h'),
        'open': close - 0.5,
        'high': close + 1.0,
        'low': close - 1.0,
        'close': close,
        'volume': volume
    })
    df.set_index('date', inplace=True)
    return df


@pytest.fixture
def volume_spike_down_df():
    """Hacim patlaması + aşağı yönlü senaryo."""
    np.random.seed(42)
    n = 50
    
    close = np.linspace(110, 90, n)
    volume = np.full(n, 1000.0)
    volume[-1] = 5000.0  # 5x hacim spike
    
    df = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=n, freq='h'),
        'open': close + 0.5,
        'high': close + 1.0,
        'low': close - 1.0,
        'close': close,
        'volume': volume
    })
    df.set_index('date', inplace=True)
    return df


# ==========================================
# STRATEGY RETURN FORMAT TESTS
# ==========================================

class TestStrategyReturnFormat:
    """Tüm stratejiler doğru formatta dönüş yapmalı."""
    
    def test_trend_strategy_returns_4_tuple(self, uptrend_df):
        from strategy.strategies import check_trend_strategy
        
        result = check_trend_strategy(uptrend_df, {})
        
        assert len(result) == 4, f"Trend strategy 4-tuple dönmeli, {len(result)} döndü"
        is_signal, direction, score, details = result
        assert isinstance(is_signal, bool)
        assert direction in ("", "LONG", "SHORT")
        assert isinstance(score, (int, float))
        assert isinstance(details, dict)
    
    def test_swing_strategy_returns_4_tuple(self, bb_lower_touch_df):
        from strategy.strategies import check_swing_strategy
        
        result = check_swing_strategy(bb_lower_touch_df, {})
        
        assert len(result) == 4, f"Swing strategy 4-tuple dönmeli, {len(result)} döndü"
        is_signal, direction, score, details = result
        assert isinstance(is_signal, bool)
        assert direction in ("", "LONG", "SHORT")
        assert isinstance(score, (int, float))
        assert isinstance(details, dict)
    
    def test_rocket_strategy_returns_4_tuple(self, volume_spike_up_df):
        from strategy.strategies import check_rocket_strategy
        
        result = check_rocket_strategy(volume_spike_up_df, {})
        
        assert len(result) == 4, f"Rocket strategy 4-tuple dönmeli, {len(result)} döndü"
        is_signal, direction, score, details = result
        assert isinstance(is_signal, bool)
        assert direction in ("", "LONG", "SHORT")
        assert isinstance(score, (int, float))
        assert isinstance(details, dict)


# ==========================================
# DIRECTION TESTS
# ==========================================

class TestDirectionLogic:
    """Strateji yönünün doğru ayarlandığını test et."""
    
    def test_trend_no_signal_returns_empty_direction(self):
        """Sinyal yoksa yön boş olmalı."""
        from strategy.strategies import check_trend_strategy
        
        # Yatay piyasa — sinyal üretmemeli
        np.random.seed(42)
        n = 250
        close = np.full(n, 100.0) + np.random.rand(n) * 0.5
        
        df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=n, freq='h'),
            'open': close - 0.1,
            'high': close + 0.5,
            'low': close - 0.5,
            'close': close,
            'volume': np.full(n, 1000.0)
        })
        df.set_index('date', inplace=True)
        
        is_signal, direction, _, _ = check_trend_strategy(df, {})
        
        if not is_signal:
            assert direction == "", "Sinyal yoksa direction boş olmalı"
    
    def test_long_signal_has_long_direction(self, uptrend_df):
        """LONG sinyal LONG direction dönmeli."""
        from strategy.strategies import check_trend_strategy
        
        is_signal, direction, _, _ = check_trend_strategy(uptrend_df, {})
        
        if is_signal:
            assert direction == "LONG", f"Uptrend'de sinyal LONG olmalı, {direction} döndü"
    
    def test_short_signal_has_short_direction(self, downtrend_df):
        """SHORT sinyal SHORT direction dönmeli."""
        from strategy.strategies import check_trend_strategy
        
        is_signal, direction, _, _ = check_trend_strategy(downtrend_df, {})
        
        if is_signal:
            assert direction == "SHORT", f"Downtrend'de sinyal SHORT olmalı, {direction} döndü"


# ==========================================
# SCORE RANGE TESTS
# ==========================================

class TestScoreRanges:
    """Tüm skorlar 0-100 arasında olmalı."""
    
    def test_trend_score_range(self, uptrend_df):
        from strategy.strategies import check_trend_strategy
        _, _, score, _ = check_trend_strategy(uptrend_df, {})
        assert 0 <= score <= 100, f"Trend skoru 0-100 arasında olmalı: {score}"
    
    def test_swing_score_range(self, bb_lower_touch_df):
        from strategy.strategies import check_swing_strategy
        _, _, score, _ = check_swing_strategy(bb_lower_touch_df, {})
        assert 0 <= score <= 100, f"Swing skoru 0-100 arasında olmalı: {score}"
    
    def test_rocket_score_range(self, volume_spike_up_df):
        from strategy.strategies import check_rocket_strategy
        _, _, score, _ = check_rocket_strategy(volume_spike_up_df, {})
        assert 0 <= score <= 100, f"Rocket skoru 0-100 arasında olmalı: {score}"


# ==========================================
# TOTAL SCORE WITH DIRECTION
# ==========================================

class TestTotalScoreDirection:
    """calculate_total_score direction parametresini doğru işlemeli."""
    
    def test_total_score_long(self, uptrend_df):
        from strategy.scoring import calculate_total_score
        
        score, breakdown = calculate_total_score(uptrend_df, "MAJOR", base_score=50, direction="LONG")
        assert 0 <= score <= 123, f"LONG total score: {score}"
        assert isinstance(breakdown, dict), "Breakdown dict olmalı"
        assert len(breakdown) == 11, f"11 faktör olmalı, {len(breakdown)} var"
    
    def test_total_score_short(self, downtrend_df):
        from strategy.scoring import calculate_total_score
        
        score, breakdown = calculate_total_score(downtrend_df, "MAJOR", base_score=50, direction="SHORT")
        assert 0 <= score <= 123, f"SHORT total score: {score}"
        assert isinstance(breakdown, dict), "Breakdown dict olmalı"
    
    def test_total_score_default_long(self, uptrend_df):
        """Default direction LONG olmalı (backward compat)."""
        from strategy.scoring import calculate_total_score
        
        score, breakdown = calculate_total_score(uptrend_df, "MAJOR", base_score=50)
        assert 0 <= score <= 123, f"Default total score: {score}"
        # Breakdown faktörleri kontrol et
        expected_factors = {'BTC', 'Reversal', 'HTF', 'Squeeze', 'SMA200', 'USDT', 'RSI', 'RSI4H', 'VOL', 'OBV', 'ADX'}
        assert set(breakdown.keys()) == expected_factors, f"Eksik faktörler: {expected_factors - set(breakdown.keys())}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
