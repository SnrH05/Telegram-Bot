"""
TITANIUM Bot - Mean Reversion Scalping Signals
================================================
Düz (ranging) piyasalarda yüksek kazanma oranıyla scalp sinyalleri.

4 Katmanlı Confluence Mimarisi:
  1. Regime Filter  → ADX(14) < 25
  2. Setup          → Fiyat Bollinger Alt Bandına temas
  3. Trigger        → StochRSI oversold + K/D crossover
  4. Validation     → CMF > 0 (para girişi)

Tüm katmanlar aynı anda sağlanırsa BUY sinyali üretilir.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple

from strategy.indicators import (
    calculate_adx,
    calculate_bollinger,
    calculate_stochastic_rsi,
    calculate_cmf,
)

logger = logging.getLogger(__name__)


# ==========================================
# 📊 MEAN REVERSION SİNYAL ÜRETİCİ
# ==========================================

def calculate_mean_reversion_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    4-Katmanlı Confluence ile Mean Reversion BUY sinyalleri üret.
    
    Tüm hesaplamalar vektörize edilmiştir (döngü yok).
    
    Args:
        df: OHLCV DataFrame ('open', 'high', 'low', 'close', 'volume')
            Minimum ~40 mum önerilir (indikatör ısınma süresi).
            
    Returns:
        DataFrame — orijinal sütunlar + indikatör sütunları + sinyal sütunları:
          - adx            : ADX(14) değerleri
          - bb_lower       : Bollinger Alt Bant
          - bb_mid         : Bollinger Orta Bant (SMA20)
          - bb_upper       : Bollinger Üst Bant
          - stoch_k        : Stochastic RSI %K
          - stoch_d        : Stochastic RSI %D
          - cmf            : Chaikin Money Flow
          - scalp_signal   : 1 = BUY, 0 = sinyal yok
          - scalp_stop_loss  : Önerilen Stop Loss fiyatı
          - scalp_take_profit: Önerilen Take Profit fiyatı
    """
    result = df.copy()
    
    # ------------------------------------------
    # Layer 1: Regime Filter — ADX(14) < 25
    # ------------------------------------------
    # Piyasa trend'de değilse (düz) devam et
    result['adx'] = calculate_adx(result, period=14)
    layer_1_pass = result['adx'] < 25
    
    # ------------------------------------------
    # Layer 2: Setup — Fiyat Bollinger Alt Bandına temas
    # ------------------------------------------
    # Low fiyatı alt banta değdi veya aşağı sardı
    bb_lower, bb_mid, bb_upper = calculate_bollinger(result, period=20, std_dev=2.0)
    result['bb_lower'] = bb_lower
    result['bb_mid'] = bb_mid
    result['bb_upper'] = bb_upper
    layer_2_pass = result['low'] <= result['bb_lower']
    
    # ------------------------------------------
    # Layer 3: Trigger — StochRSI oversold + K crosses above D
    # ------------------------------------------
    # Momentum aşırı satım bölgesinden yukarı dönüyor
    stoch_k, stoch_d = calculate_stochastic_rsi(
        result['close'], rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3
    )
    result['stoch_k'] = stoch_k
    result['stoch_d'] = stoch_d
    
    # Koşul A: K < 20 (oversold bölgesi)
    stoch_oversold = result['stoch_k'] < 20
    
    # Koşul B: K, D'yi yukarı kesiyor (crossover)
    # Önceki mumda K <= D idi, şimdi K > D
    k_crosses_above_d = (result['stoch_k'] > result['stoch_d']) & \
                        (result['stoch_k'].shift(1) <= result['stoch_d'].shift(1))
    
    layer_3_pass = stoch_oversold & k_crosses_above_d
    
    # ------------------------------------------
    # Layer 4: Volume Confirmation — CMF(20) > 0
    # ------------------------------------------
    # Para varlığa giriyor (dead cat bounce'ı filtrele)
    result['cmf'] = calculate_cmf(result, period=20)
    layer_4_pass = result['cmf'] > 0
    
    # ------------------------------------------
    # 🎯 CONFLUENCE: Tüm 4 katman aynı anda geçerli
    # ------------------------------------------
    result['scalp_signal'] = (
        layer_1_pass & layer_2_pass & layer_3_pass & layer_4_pass
    ).astype(int)
    
    # ------------------------------------------
    # 💰 RİSK YÖNETİMİ: Dinamik SL / TP
    # ------------------------------------------
    # Stop Loss  = Alt Bant × 0.99 (alt bandın %1 altı)
    # Take Profit = Orta Bant (SMA20 — ortalamaya dönüş hedefi)
    result['scalp_stop_loss'] = result['bb_lower'] * 0.99
    result['scalp_take_profit'] = result['bb_mid']
    
    # NaN temizlik — sinyal sütununda NaN → 0
    result['scalp_signal'] = result['scalp_signal'].fillna(0).astype(int)
    
    logger.debug(
        f"Mean Reversion: {result['scalp_signal'].sum()} sinyal üretildi "
        f"({len(result)} mum içinde)"
    )
    
    return result


# ==========================================
# 🔄 REJİM TABANLI STRATEJİ ANAHTARLAYICI
# ==========================================

def execute_strategy_switch(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    adx_threshold: float = 25.0,
) -> Dict:
    """
    ADX bazlı strateji anahtarlayıcı — Ana döngüye entegre edilecek.
    
    Akış:
      1. ADX hesapla
      2. ADX > threshold → Trend strateji (mevcut mantık)
      3. ADX <= threshold → Mean Reversion scalp sinyali
    
    Args:
        df: OHLCV DataFrame (minimum ~40 mum)
        symbol: İşlem çifti (ör. "BTCUSDT") — loglama amaçlı
        adx_threshold: Trend/Range ayrım eşiği (default: 25)
        
    Returns:
        dict:
          - regime       : "TREND" veya "RANGE"
          - adx_value    : Güncel ADX değeri
          - signal       : "BUY", None
          - strategy_used: "TREND_FOLLOWING" veya "MEAN_REVERSION"
          - tp_price     : Take Profit fiyatı (sadece sinyal varsa)
          - sl_price     : Stop Loss fiyatı (sadece sinyal varsa)
          - details      : Ek bilgiler (indikatör değerleri)
    """
    result: Dict = {
        'regime': None,
        'adx_value': 0.0,
        'signal': None,
        'strategy_used': None,
        'tp_price': None,
        'sl_price': None,
        'details': {},
    }
    
    # ---- Step 1: ADX hesapla ----
    adx_series = calculate_adx(df, period=14)
    current_adx = adx_series.iloc[-1] if len(adx_series) > 0 else 0.0
    result['adx_value'] = round(float(current_adx), 2)
    
    # ---- Step 2: THE SWITCH ----
    if current_adx > adx_threshold:
        # ===========================
        #  🟢 TREND MODE
        # ===========================
        result['regime'] = "TREND"
        result['strategy_used'] = "TREND_FOLLOWING"
        
        logger.info(
            f"[{symbol}] 📈 Market is Trending (ADX: {result['adx_value']}). "
            f"Using Trend Strategy."
        )
        
        # -------------------------------------------------------
        # EXISTING TREND LOGIC HERE
        # -------------------------------------------------------
        # Mevcut trend stratejinizi buraya bağlayın.
        # Örnek:
        #   trend_signal = calculate_trend_signal(df)
        #   if trend_signal == "BUY":
        #       result['signal'] = "BUY"
        #       result['tp_price'] = ...
        #       result['sl_price'] = ...
        # -------------------------------------------------------
        
    else:
        # ===========================
        #  🔵 RANGE / MEAN REVERSION MODE
        # ===========================
        result['regime'] = "RANGE"
        result['strategy_used'] = "MEAN_REVERSION"
        
        logger.info(
            f"[{symbol}] 📊 Market is Flat (ADX: {result['adx_value']}). "
            f"Switching to Mean Reversion Scalping."
        )
        
        # Mean Reversion sinyallerini hesapla
        mr_df = calculate_mean_reversion_signals(df)
        
        # Son mumun sinyaline bak
        latest = mr_df.iloc[-1]
        
        result['details'] = {
            'stoch_k': round(float(latest['stoch_k']), 2),
            'stoch_d': round(float(latest['stoch_d']), 2),
            'cmf': round(float(latest['cmf']), 4),
            'bb_lower': round(float(latest['bb_lower']), 4),
            'bb_mid': round(float(latest['bb_mid']), 4),
            'bb_upper': round(float(latest['bb_upper']), 4),
        }
        
        if latest['scalp_signal'] == 1:
            result['signal'] = "BUY"
            result['tp_price'] = round(float(latest['scalp_take_profit']), 4)
            result['sl_price'] = round(float(latest['scalp_stop_loss']), 4)
            
            logger.info(
                f"[{symbol}] 🎯 SCALP BUY Signal! "
                f"TP: {result['tp_price']} (Mid BB) | "
                f"SL: {result['sl_price']} (1% below Lower BB)"
            )
        else:
            logger.debug(
                f"[{symbol}] Mean Reversion: Confluence sağlanamadı — sinyal yok. "
                f"StochK={result['details']['stoch_k']}, "
                f"CMF={result['details']['cmf']}"
            )
    
    return result
