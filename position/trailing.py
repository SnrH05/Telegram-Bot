"""
TITANIUM Bot - Trailing Stop Management
========================================
Dinamik trailing stop ve trend bazlı SL yönetimi.
"""

import logging
from strategy import calculate_rsi, calculate_adx, calculate_ema

logger = logging.getLogger(__name__)


def calculate_trend_aware_sl_multiplier(df, direction):
    """
    🛡️ TREND-UYUMLU DİNAMİK SL ÇARPANI (V6.1)
    
    Güçlü trend dönemlerinde SL'i genişlet, zayıf trendde daralt.
    
    Kriterler:
    - ADX > 35: Çok güçlü trend → 4.0x ATR
    - ADX > 25 + EMA dizilimi: Güçlü trend → 2.5x ATR
    - ADX > 20: Normal trend → 2.0x ATR
    - ADX < 20: Zayıf/Sideways → 2.0x ATR
    
    EMA Dizilimi:
    - LONG: EMA9 > EMA21 > EMA50 (bullish)
    - SHORT: EMA9 < EMA21 < EMA50 (bearish)
    
    Args:
        df: OHLCV DataFrame
        direction: 'LONG' veya 'SHORT'
    
    Returns:
        sl_multiplier (float), trend_strength (str)
    """
    try:
        # ADX hesapla
        adx_val = calculate_adx(df).iloc[-1]
        
        # EMA'ları hesapla
        ema9 = calculate_ema(df['close'], 9).iloc[-1]
        ema21 = calculate_ema(df['close'], 21).iloc[-1]
        ema50 = calculate_ema(df['close'], 50).iloc[-1]
        
        # EMA dizilimi kontrolü
        bullish_alignment = ema9 > ema21 > ema50
        bearish_alignment = ema9 < ema21 < ema50
        
        # Trend yönüyle uyumlu mu?
        trend_aligned = (direction == "LONG" and bullish_alignment) or \
                       (direction == "SHORT" and bearish_alignment)
        
        # RSI momentum kontrolü (trend devam ediyor mu?)
        rsi = calculate_rsi(df['close']).iloc[-1]
        rsi_confirms = (direction == "LONG" and 40 < rsi < 70) or \
                      (direction == "SHORT" and 30 < rsi < 60)
        
        # SL çarpanını belirle
        if adx_val > 35 and trend_aligned:
            # Çok güçlü trend - geniş SL
            return 4.0, "ÇOK GÜÇLÜ"
        elif adx_val > 25 and trend_aligned and rsi_confirms:
            # Güçlü onaylı trend
            return 2.5, "GÜÇLÜ"
        elif adx_val > 20:
            # Normal trend
            return 2.0, "NORMAL"
        else:
            # Zayıf/sideways piyasa - standart SL
            return 2.0, "ZAYIF"
            
    except Exception as e:
        logger.warning(f"⚠️ Trend SL hesaplama hatası: {e}")
        return 2.0, "DEFAULT"
