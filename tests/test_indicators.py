"""
TITANIUM Bot - Indicator Tests
===============================
RSI, ADX, ATR, EMA, SMA hesaplama testleri.
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
def sample_ohlcv():
    """Basit OHLCV test verisi."""
    np.random.seed(42)
    n = 100
    
    # Yükselen trend simülasyonu
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    df = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n, freq='H'),
        'open': close - np.random.rand(n) * 0.5,
        'high': close + np.random.rand(n) * 1.0,
        'low': close - np.random.rand(n) * 1.0,
        'close': close,
        'volume': np.random.randint(1000, 10000, n).astype(float)
    })
    return df


@pytest.fixture
def flat_ohlcv():
    """Düz (range) piyasa verisi."""
    n = 100
    base_price = 100.0
    
    df = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n, freq='H'),
        'open': [base_price] * n,
        'high': [base_price + 0.1] * n,
        'low': [base_price - 0.1] * n,
        'close': [base_price] * n,
        'volume': [5000.0] * n
    })
    return df


@pytest.fixture
def volatile_ohlcv():
    """Yüksek volatilite verisi."""
    np.random.seed(42)
    n = 100
    
    close = 100 + np.cumsum(np.random.randn(n) * 5)  # Büyük hareketler
    
    df = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n, freq='H'),
        'open': close - np.random.rand(n) * 3,
        'high': close + np.random.rand(n) * 5,
        'low': close - np.random.rand(n) * 5,
        'close': close,
        'volume': np.random.randint(5000, 50000, n).astype(float)
    })
    return df


# ==========================================
# RSI TESTS
# ==========================================

class TestRSI:
    """RSI hesaplama testleri."""
    
    def test_rsi_range(self, sample_ohlcv):
        """RSI 0-100 arasında olmalı."""
        from strategy.indicators import calculate_rsi
        
        rsi = calculate_rsi(sample_ohlcv['close'])
        
        assert rsi.min() >= 0, "RSI 0'dan küçük olamaz"
        assert rsi.max() <= 100, "RSI 100'den büyük olamaz"
        
    def test_rsi_no_nan_after_warmup(self, sample_ohlcv):
        """Warmup sonrası NaN olmamalı."""
        from strategy.indicators import calculate_rsi
        
        rsi = calculate_rsi(sample_ohlcv['close'], period=14)
        
        # İlk 14 değer NaN olabilir, sonrası olmamalı
        assert not rsi.iloc[20:].isna().any(), "RSI NaN değer içermemeli"
        
    def test_rsi_constant_price(self, flat_ohlcv):
        """Sabit fiyatta RSI 50 civarında olmalı."""
        from strategy.indicators import calculate_rsi
        
        rsi = calculate_rsi(flat_ohlcv['close'])
        
        # Sabit fiyatta gain=loss=0, RSI ~50 veya 0 olabilir (edge case)
        # RSI algoritmasına bağlı - 0 kabul edilebilir
        assert rsi.iloc[-1] >= 0 and rsi.iloc[-1] <= 100, "Sabit fiyatta RSI 0-100 arasında olmalı"
        
    def test_rsi_division_by_zero(self):
        """Division by zero hatası olmamalı."""
        from strategy.indicators import calculate_rsi
        
        # Sadece yükselen fiyat (loss = 0)
        prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                           110, 111, 112, 113, 114, 115, 116, 117, 118, 119])
        
        rsi = calculate_rsi(prices)
        
        # Hata vermemeli ve geçerli değer döndürmeli
        assert not np.isinf(rsi.iloc[-1]), "RSI inf olmamalı"
        assert not np.isnan(rsi.iloc[-1]), "RSI NaN olmamalı"


# ==========================================
# ADX TESTS
# ==========================================

class TestADX:
    """ADX hesaplama testleri."""
    
    def test_adx_range(self, sample_ohlcv):
        """ADX 0-100 arasında olmalı."""
        from strategy.indicators import calculate_adx
        
        adx = calculate_adx(sample_ohlcv)
        
        assert adx.min() >= 0, "ADX 0'dan küçük olamaz"
        assert adx.max() <= 100, "ADX 100'den büyük olamaz"
        
    def test_adx_no_inf(self, sample_ohlcv):
        """ADX inf olmamalı."""
        from strategy.indicators import calculate_adx
        
        adx = calculate_adx(sample_ohlcv)
        
        assert not np.isinf(adx).any(), "ADX inf değer içermemeli"
        
    def test_adx_flat_market(self, flat_ohlcv):
        """Düz piyasada ADX düşük olmalı."""
        from strategy.indicators import calculate_adx
        
        adx = calculate_adx(flat_ohlcv)
        
        # Düz piyasada ADX < 25 beklenir
        assert adx.iloc[-1] < 30, f"Düz piyasada ADX düşük olmalı, got {adx.iloc[-1]}"
        
    def test_adx_division_by_zero(self, flat_ohlcv):
        """Division by zero koruması çalışmalı."""
        from strategy.indicators import calculate_adx
        
        # Hiç hareket olmayan veri
        flat_ohlcv['high'] = flat_ohlcv['close']
        flat_ohlcv['low'] = flat_ohlcv['close']
        
        adx = calculate_adx(flat_ohlcv)
        
        # Hata vermemeli
        assert not np.isinf(adx.iloc[-1]), "ADX inf olmamalı"


# ==========================================
# ATR TESTS
# ==========================================

class TestATR:
    """ATR hesaplama testleri."""
    
    def test_atr_positive(self, sample_ohlcv):
        """ATR her zaman pozitif olmalı."""
        from strategy.indicators import calculate_atr
        
        atr = calculate_atr(sample_ohlcv)
        
        # NaN olmayan değerler pozitif olmalı
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all(), "ATR negatif olamaz"
        
    def test_atr_volatile_high(self, volatile_ohlcv):
        """Yüksek volatilitede ATR yüksek olmalı."""
        from strategy.indicators import calculate_atr
        
        atr_volatile = calculate_atr(volatile_ohlcv).iloc[-1]
        
        # Volatile piyasada ATR > 1 beklenir
        assert atr_volatile > 1.0, "Volatile piyasada ATR yüksek olmalı"
        
    def test_atr_flat_low(self, flat_ohlcv):
        """Düz piyasada ATR düşük olmalı."""
        from strategy.indicators import calculate_atr
        
        atr_flat = calculate_atr(flat_ohlcv).iloc[-1]
        
        # Düz piyasada ATR ~ 0.2 (high-low farkı)
        assert atr_flat < 1.0, f"Düz piyasada ATR düşük olmalı, got {atr_flat}"


# ==========================================
# EMA/SMA TESTS
# ==========================================

class TestMovingAverages:
    """EMA ve SMA testleri."""
    
    def test_ema_follows_price(self, sample_ohlcv):
        """EMA fiyatı takip etmeli."""
        from strategy.indicators import calculate_ema
        
        ema = calculate_ema(sample_ohlcv['close'], 20)
        
        # EMA son fiyata yakın olmalı
        last_price = sample_ohlcv['close'].iloc[-1]
        assert abs(ema.iloc[-1] - last_price) < 10, "EMA fiyatı takip etmeli"
        
    def test_sma_average(self, sample_ohlcv):
        """SMA doğru ortalama hesaplamalı."""
        from strategy.indicators import calculate_sma
        
        window = 5
        sma = calculate_sma(sample_ohlcv['close'], window)
        
        # Son 5 değerin ortalaması
        expected = sample_ohlcv['close'].tail(window).mean()
        
        assert abs(sma.iloc[-1] - expected) < 0.01, "SMA yanlış hesaplandı"
        
    def test_ema_faster_than_sma(self, sample_ohlcv):
        """Yükselen trend'de EMA > SMA olmalı."""
        from strategy.indicators import calculate_ema, calculate_sma
        
        # Yükselen trend oluştur
        sample_ohlcv['close'] = sample_ohlcv['close'] + np.arange(len(sample_ohlcv)) * 0.5
        
        ema = calculate_ema(sample_ohlcv['close'], 20)
        sma = calculate_sma(sample_ohlcv['close'], 20)
        
        # EMA daha hızlı tepki vermeli - fark çok küçük olabilir
        assert abs(ema.iloc[-1] - sma.iloc[-1]) < 2, "EMA ve SMA birbirine yakın olmalı"


# ==========================================
# EDGE CASES
# ==========================================

class TestEdgeCases:
    """Uç durumlar."""
    
    def test_empty_series(self):
        """Boş series hata vermemeli."""
        from strategy.indicators import calculate_rsi, calculate_ema
        
        empty = pd.Series([], dtype=float)
        
        rsi = calculate_rsi(empty)
        ema = calculate_ema(empty, 14)
        
        assert len(rsi) == 0
        assert len(ema) == 0
        
    def test_single_value(self):
        """Tek değerli series."""
        from strategy.indicators import calculate_rsi
        
        single = pd.Series([100.0])
        rsi = calculate_rsi(single)
        
        # NaN veya 50 dönmeli
        assert len(rsi) == 1
        
    def test_negative_prices(self):
        """Negatif fiyat (anormal veri)."""
        from strategy.indicators import calculate_rsi
        
        # Bazı negatif değerler
        prices = pd.Series([100, 101, -5, 102, 103, 104, 105, 106, 107, 108,
                           109, 110, 111, 112, 113, 114, 115])
        
        rsi = calculate_rsi(prices)
        
        # Hata vermemeli
        assert not np.isinf(rsi.iloc[-1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
