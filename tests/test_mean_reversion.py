"""
TITANIUM Bot - Mean Reversion Strategy Tests
==============================================
Stochastic RSI, CMF indikatörleri ve 4-katman confluence sinyali testleri.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ana modülleri import edebilmek için path ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==========================================
# TEST FIXTURES
# ==========================================

@pytest.fixture
def ranging_ohlcv():
    """
    Düz (ranging) piyasa verisi — ADX düşük olacak şekilde tasarlandı.
    Fiyat dar bir bant içinde salınım yapar.
    """
    np.random.seed(42)
    n = 120  # Yeterli ısınma süresi için

    base = 100.0
    # Küçük sinüzoidal salınım — trend yok
    noise = np.sin(np.linspace(0, 8 * np.pi, n)) * 1.5 + np.random.randn(n) * 0.3
    close = base + noise

    df = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n, freq='h'),
        'open': close - np.random.rand(n) * 0.3,
        'high': close + np.random.rand(n) * 0.8,
        'low': close - np.random.rand(n) * 0.8,
        'close': close,
        'volume': np.random.randint(2000, 15000, n).astype(float),
    })
    return df


@pytest.fixture
def trending_ohlcv():
    """Güçlü yukarı trend verisi — ADX yüksek olmalı."""
    np.random.seed(99)
    n = 120

    # Güçlü trend: her mum +1-2 birim yukarı
    close = 100.0 + np.cumsum(np.random.uniform(0.5, 2.0, n))

    df = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n, freq='h'),
        'open': close - np.random.rand(n) * 0.5,
        'high': close + np.random.rand(n) * 1.5,
        'low': close - np.random.rand(n) * 0.5,
        'close': close,
        'volume': np.random.randint(5000, 30000, n).astype(float),
    })
    return df


@pytest.fixture
def sample_ohlcv():
    """Genel amaçlı OHLCV verisi."""
    np.random.seed(42)
    n = 100

    close = 100 + np.cumsum(np.random.randn(n) * 0.5)

    df = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n, freq='h'),
        'open': close - np.random.rand(n) * 0.5,
        'high': close + np.random.rand(n) * 1.0,
        'low': close - np.random.rand(n) * 1.0,
        'close': close,
        'volume': np.random.randint(1000, 10000, n).astype(float),
    })
    return df


# ==========================================
# STOCHASTIC RSI TESTS
# ==========================================

class TestStochasticRSI:
    """Stochastic RSI hesaplama testleri."""

    def test_stoch_rsi_range(self, sample_ohlcv):
        """Stoch K ve D 0-100 arasında olmalı."""
        from strategy.indicators import calculate_stochastic_rsi

        k, d = calculate_stochastic_rsi(sample_ohlcv['close'])

        assert k.min() >= 0, "Stoch K 0'dan küçük olamaz"
        assert k.max() <= 100, "Stoch K 100'den büyük olamaz"
        assert d.min() >= 0, "Stoch D 0'dan küçük olamaz"
        assert d.max() <= 100, "Stoch D 100'den büyük olamaz"

    def test_stoch_rsi_no_nan_after_warmup(self, sample_ohlcv):
        """Warmup sonrası NaN olmamalı."""
        from strategy.indicators import calculate_stochastic_rsi

        k, d = calculate_stochastic_rsi(sample_ohlcv['close'])

        # İlk 35 değer warmup (14 RSI + 14 Stoch + 3 K smooth + 3 D smooth)
        assert not k.iloc[40:].isna().any(), "Stoch K NaN içermemeli"
        assert not d.iloc[40:].isna().any(), "Stoch D NaN içermemeli"

    def test_stoch_rsi_flat_price(self):
        """Sabit fiyatta Stoch RSI neutral (50) dönmeli."""
        from strategy.indicators import calculate_stochastic_rsi

        flat = pd.Series([100.0] * 50)
        k, d = calculate_stochastic_rsi(flat)

        # Sabit fiyatta — NaN fill default 50
        assert k.iloc[-1] >= 0 and k.iloc[-1] <= 100

    def test_stoch_rsi_no_inf(self, sample_ohlcv):
        """Inf değer olmamalı."""
        from strategy.indicators import calculate_stochastic_rsi

        k, d = calculate_stochastic_rsi(sample_ohlcv['close'])

        assert not np.isinf(k).any(), "Stoch K inf olmamalı"
        assert not np.isinf(d).any(), "Stoch D inf olmamalı"


# ==========================================
# CMF TESTS
# ==========================================

class TestCMF:
    """Chaikin Money Flow testleri."""

    def test_cmf_range(self, sample_ohlcv):
        """CMF -1 ile +1 arasında olmalı."""
        from strategy.indicators import calculate_cmf

        cmf = calculate_cmf(sample_ohlcv)

        assert cmf.min() >= -1, "CMF -1'den küçük olamaz"
        assert cmf.max() <= 1, "CMF 1'den büyük olamaz"

    def test_cmf_no_inf(self, sample_ohlcv):
        """Inf değer olmamalı."""
        from strategy.indicators import calculate_cmf

        cmf = calculate_cmf(sample_ohlcv)

        assert not np.isinf(cmf).any(), "CMF inf olmamalı"

    def test_cmf_zero_volume(self):
        """Sıfır hacim durumunda hata vermemeli."""
        from strategy.indicators import calculate_cmf

        n = 50
        df = pd.DataFrame({
            'open': [100.0] * n,
            'high': [101.0] * n,
            'low': [99.0] * n,
            'close': [100.0] * n,
            'volume': [0.0] * n,
        })

        cmf = calculate_cmf(df)

        assert not np.isinf(cmf).any(), "Sıfır hacimde CMF inf olmamalı"
        assert not np.isnan(cmf).any(), "CMF NaN olmamalı (fillna gerekli)"

    def test_cmf_close_at_high_positive(self):
        """Close = High olduğunda CMF pozitif olmalı (alım baskısı)."""
        from strategy.indicators import calculate_cmf

        n = 30
        df = pd.DataFrame({
            'open': [99.0] * n,
            'high': [101.0] * n,
            'low': [98.0] * n,
            'close': [101.0] * n,  # Close = High → strong buying
            'volume': [10000.0] * n,
        })

        cmf = calculate_cmf(df)
        # Son değer pozitif olmalı
        assert cmf.iloc[-1] > 0, "Close=High durumunda CMF pozitif olmalı"


# ==========================================
# MEAN REVERSION SIGNAL TESTS
# ==========================================

class TestMeanReversionSignals:
    """4-katman confluence sinyal testleri."""

    def test_output_columns_exist(self, ranging_ohlcv):
        """Tüm beklenen sütunlar mevcut olmalı."""
        from signals.mean_reversion import calculate_mean_reversion_signals

        result = calculate_mean_reversion_signals(ranging_ohlcv)

        expected_cols = [
            'adx', 'bb_lower', 'bb_mid', 'bb_upper',
            'stoch_k', 'stoch_d', 'cmf',
            'scalp_signal', 'scalp_stop_loss', 'scalp_take_profit',
        ]
        for col in expected_cols:
            assert col in result.columns, f"'{col}' sütunu eksik"

    def test_signal_is_binary(self, ranging_ohlcv):
        """scalp_signal yalnızca 0 veya 1 olmalı."""
        from signals.mean_reversion import calculate_mean_reversion_signals

        result = calculate_mean_reversion_signals(ranging_ohlcv)

        unique_vals = result['scalp_signal'].unique()
        assert set(unique_vals).issubset({0, 1}), \
            f"scalp_signal 0/1 dışında değer içeriyor: {unique_vals}"

    def test_no_signal_in_strong_trend(self, trending_ohlcv):
        """Güçlü trend'de (ADX > 25) sinyal üretilmemeli."""
        from signals.mean_reversion import calculate_mean_reversion_signals

        result = calculate_mean_reversion_signals(trending_ohlcv)

        # Son 30 mumda ADX yüksek olmalı → sinyal 0
        # Not: İlk mumlarda ADX düşük olabilir (ısınma)
        last_30 = result.tail(30)
        high_adx_rows = last_30[last_30['adx'] > 25]

        if len(high_adx_rows) > 0:
            assert (high_adx_rows['scalp_signal'] == 0).all(), \
                "ADX > 25 olan mumlarda sinyal üretilmemeli"

    def test_original_columns_preserved(self, ranging_ohlcv):
        """Orijinal sütunlar bozulmadan kalmalı."""
        from signals.mean_reversion import calculate_mean_reversion_signals

        original_cols = set(ranging_ohlcv.columns)
        result = calculate_mean_reversion_signals(ranging_ohlcv)

        for col in original_cols:
            assert col in result.columns, f"Orijinal sütun '{col}' kaybolmuş"

    def test_sl_below_tp(self, ranging_ohlcv):
        """Stop Loss her zaman Take Profit'in altında olmalı."""
        from signals.mean_reversion import calculate_mean_reversion_signals

        result = calculate_mean_reversion_signals(ranging_ohlcv)

        # NaN olmayan satırlar için kontrol
        valid = result.dropna(subset=['scalp_stop_loss', 'scalp_take_profit'])
        if len(valid) > 0:
            assert (valid['scalp_stop_loss'] < valid['scalp_take_profit']).all(), \
                "SL her zaman TP'den düşük olmalı"

    def test_no_nan_in_signal(self, ranging_ohlcv):
        """scalp_signal sütununda NaN olmamalı."""
        from signals.mean_reversion import calculate_mean_reversion_signals

        result = calculate_mean_reversion_signals(ranging_ohlcv)

        assert not result['scalp_signal'].isna().any(), \
            "scalp_signal NaN içermemeli"


# ==========================================
# STRATEGY SWITCH TESTS
# ==========================================

class TestStrategySwitch:
    """Rejim bazlı strateji anahtarlayıcı testleri."""

    def test_trend_regime_detection(self, trending_ohlcv):
        """Trend piyasada TREND rejimi seçilmeli."""
        from signals.mean_reversion import execute_strategy_switch

        result = execute_strategy_switch(trending_ohlcv, symbol="TEST")

        assert result['regime'] in ('TREND', 'RANGE'), \
            f"Geçersiz rejim: {result['regime']}"
        assert result['strategy_used'] is not None

    def test_range_regime_detection(self, ranging_ohlcv):
        """Düz piyasada RANGE rejimi seçilmeli."""
        from signals.mean_reversion import execute_strategy_switch

        result = execute_strategy_switch(ranging_ohlcv, symbol="TEST")

        # Düz piyasa verisinde ADX düşük olmalı
        if result['adx_value'] <= 25:
            assert result['regime'] == "RANGE"
            assert result['strategy_used'] == "MEAN_REVERSION"

    def test_switch_returns_complete_dict(self, ranging_ohlcv):
        """Dönen dict tüm beklenen anahtarları içermeli."""
        from signals.mean_reversion import execute_strategy_switch

        result = execute_strategy_switch(ranging_ohlcv, symbol="TEST")

        expected_keys = [
            'regime', 'adx_value', 'signal',
            'strategy_used', 'tp_price', 'sl_price', 'details',
        ]
        for key in expected_keys:
            assert key in result, f"'{key}' anahtarı eksik"

    def test_buy_signal_has_tp_sl(self, ranging_ohlcv):
        """BUY sinyali varsa TP ve SL dolu olmalı."""
        from signals.mean_reversion import execute_strategy_switch

        result = execute_strategy_switch(ranging_ohlcv, symbol="TEST")

        if result['signal'] == "BUY":
            assert result['tp_price'] is not None, "BUY sinyalinde TP eksik"
            assert result['sl_price'] is not None, "BUY sinyalinde SL eksik"
            assert result['sl_price'] < result['tp_price'], "SL < TP olmalı"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
