"""
TITANIUM Bot - Strategy Logic
=============================
Bu modül, 3 farklı coin grubu için özelleştirilmiş stratejileri içerir.
Her strateji hem LONG hem SHORT sinyal üretebilir.

Skorlama yön-spesifiktir:
- LONG: Bullish EMA dizilimi, RSI dip dönüşü, pozitif CMF, yeşil mum bonusu
- SHORT: Bearish EMA dizilimi, RSI tepe dönüşü, negatif CMF, kırmızı mum bonusu
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

from .indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger,
    calculate_stochastic_rsi,
    calculate_supertrend,
    calculate_cmf,
    calculate_adx
)


# ==========================================
# ORTAK: YÖN-SPESİFİK BONUS PUAN HESAPLA
# ==========================================
def _calculate_direction_bonus(df: pd.DataFrame, direction: str) -> Tuple[float, Dict[str, Any]]:
    """
    Yön-spesifik ek puanlar hesapla. Her strateji bu bonusu kullanır.
    
    LONG Bonusları:
      +5  EMA9 > EMA21 (kısa vadeli bullish dizilim)
      +5  RSI 40-65 arası (sağlıklı momentum, aşırı alım değil)
      +5  CMF > 0 (para girişi)
      +3  Son mum yeşil (kapanış teyidi)
      +2  MACD histogram artıyorsa (ivme)
      
    SHORT Bonusları:
      +5  EMA9 < EMA21 (kısa vadeli bearish dizilim)
      +5  RSI 35-60 arası (sağlıklı momentum, aşırı satım değil)
      +5  CMF < 0 (para çıkışı)
      +3  Son mum kırmızı (kapanış teyidi)
      +2  MACD histogram azalıyorsa (ivme)

    Max bonus: 20 puan
    
    Returns:
        (bonus, bonus_details)
    """
    bonus = 0.0
    bonus_details = {}
    
    try:
        # EMA dizilim kontrolü
        ema9 = calculate_ema(df['close'], 9).iloc[-1]
        ema21 = calculate_ema(df['close'], 21).iloc[-1]
        
        if direction == "LONG":
            if ema9 > ema21:
                bonus += 5
                bonus_details['ema_align'] = '✅ Bullish'
        else:  # SHORT
            if ema9 < ema21:
                bonus += 5
                bonus_details['ema_align'] = '✅ Bearish'
    except:
        pass
    
    try:
        # RSI yön-spesifik momentum
        rsi_val = calculate_rsi(df['close']).iloc[-1]
        
        if direction == "LONG":
            # LONG: RSI 40-65 = sağlıklı momentum (aşırı alımda değil)
            if 40 <= rsi_val <= 65:
                bonus += 5
                bonus_details['rsi_zone'] = f'✅ {rsi_val:.0f}'
        else:  # SHORT
            # SHORT: RSI 35-60 = sağlıklı momentum (aşırı satımda değil)
            if 35 <= rsi_val <= 60:
                bonus += 5
                bonus_details['rsi_zone'] = f'✅ {rsi_val:.0f}'
    except:
        pass
    
    try:
        # CMF (Chaikin Money Flow) — para akışı yönü
        cmf = calculate_cmf(df).iloc[-1]
        
        if direction == "LONG" and cmf > 0.05:
            bonus += 5
            bonus_details['cmf'] = f'✅ +{cmf:.2f}'
        elif direction == "SHORT" and cmf < -0.05:
            bonus += 5
            bonus_details['cmf'] = f'✅ {cmf:.2f}'
    except:
        pass
    
    try:
        # Son mum rengi teyidi
        last_close = df['close'].iloc[-1]
        last_open = df['open'].iloc[-1]
        
        if direction == "LONG" and last_close > last_open:
            bonus += 3
            bonus_details['candle'] = '✅ Yeşil'
        elif direction == "SHORT" and last_close < last_open:
            bonus += 3
            bonus_details['candle'] = '✅ Kırmızı'
    except:
        pass
    
    try:
        # MACD histogram ivmesi
        _, _, hist = calculate_macd(df['close'])
        hist_now = hist.iloc[-1]
        hist_prev = hist.iloc[-2]
        
        if direction == "LONG" and hist_now > hist_prev:
            bonus += 2
            bonus_details['macd_mom'] = '✅ Artıyor'
        elif direction == "SHORT" and hist_now < hist_prev:
            bonus += 2
            bonus_details['macd_mom'] = '✅ Azalıyor'
    except:
        pass
    
    return bonus, bonus_details


# ==========================================
# 1. GROUP STRATEGY: TREND SETTERS (MAJORS)
# ==========================================
def check_trend_strategy(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[bool, str, float, Dict[str, Any]]:
    """
    EMA Pullback Strategy for Majors (LONG & SHORT).
    
    LONG:
    1. Trend Filter: Close > EMA 200
    2. Pullback Trigger: Price touches/nears EMA 50
    3. Confirmation: RSI 40-60 + MACD Positive/Crossing
    
    SHORT:
    1. Trend Filter: Close < EMA 200
    2. Rally Trigger: Price rallies up to EMA 50 (resistance)
    3. Confirmation: RSI 40-60 + MACD Negative/Crossing Down
    
    Skorlama (LONG):
      Base: 45 | Trend Gücü: max 15 | Pullback Kalitesi: max 20 | Yön Bonus: max 20
    Skorlama (SHORT):
      Base: 40 | Trend Gücü: max 15 | Rally Kalitesi: max 20 | Yön Bonus: max 20
      SHORT base biraz düşük (kripto genelde bullish bias)
    
    Returns:
        (is_signal, direction, score, details)
    """
    # İndikatörleri hesapla
    ema200 = calculate_ema(df['close'], params.get('ema_trend', 200))
    ema50 = calculate_ema(df['close'], params.get('ema_pullback', 50))
    rsi = calculate_rsi(df['close'])
    macd_line, signal_line, hist = calculate_macd(df['close'])
    
    current_close = df['close'].iloc[-1]
    current_low = df['low'].iloc[-1]
    current_high = df['high'].iloc[-1]
    current_ema200 = ema200.iloc[-1]
    current_ema50 = ema50.iloc[-1]
    current_rsi = rsi.iloc[-1]
    current_hist = hist.iloc[-1]
    
    rsi_min = params.get('rsi_min', 40)
    rsi_max = params.get('rsi_max', 60)
    
    # ========== LONG CHECK ==========
    if current_close > current_ema200:
        dist_to_ema50 = (current_low - current_ema50) / current_ema50
        is_pullback = dist_to_ema50 <= 0.005 and current_close >= current_ema50 * 0.99
        
        if is_pullback and (rsi_min <= current_rsi <= rsi_max):
            is_macd_bullish = current_hist > 0 or (hist.iloc[-1] > hist.iloc[-2])
            
            if is_macd_bullish:
                # LONG Skorlama
                score = 45  # Base
                trend_score = min(15, (current_close / current_ema200 - 1) * 800)
                pullback_score = min(20, (1 - abs(dist_to_ema50)) * 20)
                
                # Yön-spesifik bonus
                dir_bonus, dir_details = _calculate_direction_bonus(df, "LONG")
                
                final_score = score + trend_score + pullback_score + dir_bonus
                
                details = {
                    "strategy": "Trend Pullback",
                    "direction": "LONG",
                    "close": round(current_close, 4),
                    "ema200": round(current_ema200, 4),
                    "ema50": round(current_ema50, 4),
                    "rsi": round(current_rsi, 1),
                    "macd_hist": round(current_hist, 6),
                    "dir_bonus": round(dir_bonus, 1),
                }
                details.update(dir_details)
                return True, "LONG", min(100, final_score), details
    
    # ========== SHORT CHECK ==========
    if current_close < current_ema200:
        dist_to_ema50 = (current_ema50 - current_high) / current_ema50
        is_rally = dist_to_ema50 <= 0.005 and current_close <= current_ema50 * 1.01
        
        if is_rally and (rsi_min <= current_rsi <= rsi_max):
            is_macd_bearish = current_hist < 0 or (hist.iloc[-1] < hist.iloc[-2])
            
            if is_macd_bearish:
                # SHORT Skorlama — biraz düşük base (kripto bullish bias)
                score = 40  # Base (LONG'dan 5 düşük)
                trend_score = min(15, (1 - current_close / current_ema200) * 800)
                rally_score = min(20, (1 - abs(dist_to_ema50)) * 20)
                
                # Ekstra: ADX güçlü trend teyidi SHORT için çok önemli
                try:
                    adx_val = calculate_adx(df).iloc[-1]
                    if adx_val > 30:
                        score += 5  # Güçlü trend = SHORT daha güvenilir
                except:
                    pass
                
                # Yön-spesifik bonus
                dir_bonus, dir_details = _calculate_direction_bonus(df, "SHORT")
                
                final_score = score + trend_score + rally_score + dir_bonus
                
                details = {
                    "strategy": "Trend Rally Short",
                    "direction": "SHORT",
                    "close": round(current_close, 4),
                    "ema200": round(current_ema200, 4),
                    "ema50": round(current_ema50, 4),
                    "rsi": round(current_rsi, 1),
                    "macd_hist": round(current_hist, 6),
                    "dir_bonus": round(dir_bonus, 1),
                }
                details.update(dir_details)
                return True, "SHORT", min(100, final_score), details
    
    return False, "", 0.0, {"fail": "No signal"}


# ==========================================
# 2. GROUP STRATEGY: SWING PLAYERS (MID-CAPS)
# ==========================================
def check_swing_strategy(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[bool, str, float, Dict[str, Any]]:
    """
    Bollinger Reversion Strategy for Mid-Caps (LONG & SHORT).
    
    LONG:
    1. Filter: BB Width not too tight
    2. Trigger: Price <= Lower Band
    3. Confirmation: StochRSI Cross UP (Oversold Area)
    
    SHORT:
    1. Filter: BB Width not too tight
    2. Trigger: Price >= Upper Band
    3. Confirmation: StochRSI Cross DOWN (Overbought Area)
    
    Skorlama (LONG):
      Base: 55 | BB Sapma: max 15 | StochRSI: max 10 | Yön Bonus: max 20
    Skorlama (SHORT):
      Base: 50 | BB Sapma: max 15 | StochRSI: max 10 | Üst Fitil: max 5 | Yön Bonus: max 20
    
    Returns:
        (is_signal, direction, score, details)
    """
    period = params.get('bb_period', 20)
    std_dev = params.get('bb_std', 2)
    lower, mid, upper = calculate_bollinger(df, period, std_dev)
    stoch_k, stoch_d = calculate_stochastic_rsi(df['close'])
    
    current_close = df['close'].iloc[-1]
    current_low = df['low'].iloc[-1]
    current_high = df['high'].iloc[-1]
    current_lower_band = lower.iloc[-1]
    current_upper_band = upper.iloc[-1]
    current_k = stoch_k.iloc[-1]
    current_d = stoch_d.iloc[-1]
    prev_k = stoch_k.iloc[-2]
    prev_d = stoch_d.iloc[-2]
    
    # BB Width Check
    bb_width = (upper.iloc[-1] - lower.iloc[-1]) / mid.iloc[-1]
    if bb_width < 0.03:
        return False, "", 0.0, {"fail": "BB Too Tight"}
    
    oversold_limit = params.get('stoch_rsi_oversold', 20)
    overbought_limit = params.get('stoch_rsi_overbought', 80)
    
    # ========== LONG CHECK ==========
    if current_low <= current_lower_band * 1.005:
        is_crossover = (prev_k < prev_d) and (current_k > current_d)
        is_oversold = current_k <= oversold_limit or prev_k <= oversold_limit
        
        long_confirmed = (is_crossover and is_oversold) or (current_k < 20 and current_k > prev_k)
        
        if long_confirmed:
            score = 55  # LONG base
            
            # BB sapma puanı (alt bandın ne kadar altında)
            deviation_score = 0
            if current_low < current_lower_band:
                deviation_score = min(15, (current_lower_band / current_low - 1) * 1000)
            
            # StochRSI dipten dönüş puanı
            stoch_score = min(10, (20 - current_k)) if current_k < 20 else 0
            
            # LONG bonus: Alt fitil reddi (dip alıcı baskısı)
            wick_bonus = 0
            body = abs(current_close - df['open'].iloc[-1])
            lower_wick = min(current_close, df['open'].iloc[-1]) - current_low
            if body > 0 and lower_wick / body > 1.5:
                wick_bonus = 5
            
            # Yön-spesifik bonus
            dir_bonus, dir_details = _calculate_direction_bonus(df, "LONG")
            
            final_score = score + deviation_score + stoch_score + wick_bonus + dir_bonus
            
            details = {
                "strategy": "Bollinger Reversion",
                "direction": "LONG",
                "low": round(current_low, 4),
                "bb_lower": round(current_lower_band, 4),
                "stoch_k": round(current_k, 1),
                "bb_width": round(bb_width, 4),
                "dir_bonus": round(dir_bonus, 1),
            }
            if wick_bonus > 0:
                details['wick_reject'] = '✅ Alıcı'
            details.update(dir_details)
            return True, "LONG", min(100, final_score), details
    
    # ========== SHORT CHECK ==========
    if current_high >= current_upper_band * 0.995:
        is_crossdown = (prev_k > prev_d) and (current_k < current_d)
        is_overbought = current_k >= overbought_limit or prev_k >= overbought_limit
        
        short_confirmed = (is_crossdown and is_overbought) or (current_k > 80 and current_k < prev_k)
        
        if short_confirmed:
            score = 50  # SHORT base (LONG'dan 5 düşük)
            
            # BB sapma puanı (üst bandın ne kadar üstünde)
            deviation_score = 0
            if current_high > current_upper_band:
                deviation_score = min(15, (current_high / current_upper_band - 1) * 1000)
            
            # StochRSI tepeden dönüş puanı
            stoch_score = min(10, (current_k - 80)) if current_k > 80 else 0
            
            # SHORT bonus: Üst fitil reddi (tepe satıcı baskısı)
            wick_bonus = 0
            body = abs(current_close - df['open'].iloc[-1])
            upper_wick = current_high - max(current_close, df['open'].iloc[-1])
            if body > 0 and upper_wick / body > 1.5:
                wick_bonus = 5
            
            # Yön-spesifik bonus
            dir_bonus, dir_details = _calculate_direction_bonus(df, "SHORT")
            
            final_score = score + deviation_score + stoch_score + wick_bonus + dir_bonus
            
            details = {
                "strategy": "Bollinger Reversion Short",
                "direction": "SHORT",
                "high": round(current_high, 4),
                "bb_upper": round(current_upper_band, 4),
                "stoch_k": round(current_k, 1),
                "bb_width": round(bb_width, 4),
                "dir_bonus": round(dir_bonus, 1),
            }
            if wick_bonus > 0:
                details['wick_reject'] = '✅ Satıcı'
            details.update(dir_details)
            return True, "SHORT", min(100, final_score), details
    
    return False, "", 0.0, {"fail": "No signal"}


# ==========================================
# 3. GROUP STRATEGY: ROCKETS (MEME/VOLATILITY)
# ==========================================
def check_rocket_strategy(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[bool, str, float, Dict[str, Any]]:
    """
    Volume Breakout & SuperTrend Strategy for Memes (LONG & SHORT).
    
    LONG:
    1. Volume Spike: Current Volume > 2.5x Avg Volume (20)
    2. Trend: Price > SuperTrend(10, 3)
    3. Momentum: RSI(7) > 60
    
    SHORT:
    1. Volume Spike: Current Volume > 2.5x Avg Volume (20)
    2. Trend: Price < SuperTrend(10, 3)
    3. Momentum: RSI(7) < 40
    
    Skorlama (LONG):
      Base: 60 | Hacim: max 20 | RSI: max 10 | Yön Bonus: max 20
    Skorlama (SHORT):
      Base: 55 | Hacim: max 20 | RSI: max 10 | Ardışık Kırmızı: max 5 | Yön Bonus: max 20
    
    Returns:
        (is_signal, direction, score, details)
    """
    # İndikatörler
    supertrend, trend_dir = calculate_supertrend(
        df, 
        params.get('supertrend_period', 10), 
        params.get('supertrend_multiplier', 3)
    )
    
    # Volume MA
    vol_ma_period = params.get('volume_ma', 20)
    vol_ma = df['volume'].rolling(vol_ma_period).mean()
    
    # RSI (7)
    rsi = calculate_rsi(df['close'], params.get('rsi_period', 7))
    
    current_vol = df['volume'].iloc[-1]
    current_vol_ma = vol_ma.iloc[-1]
    current_close = df['close'].iloc[-1]
    current_st = supertrend.iloc[-1]
    current_rsi = rsi.iloc[-1]
    
    # Volume Spike Check (hem LONG hem SHORT için gerekli)
    spike_mult = params.get('volume_spike_mult', 2.5)
    if current_vol < (current_vol_ma * spike_mult):
        return False, "", 0.0, {"fail": f"No Vol Spike ({current_vol/current_vol_ma:.1f}x)"}
    
    vol_mult = current_vol / current_vol_ma
    
    # ========== LONG CHECK ==========
    if current_close > current_st and current_rsi > params.get('rsi_min', 60):
        score = 60  # LONG base
        vol_score = min(20, (vol_mult - spike_mult) * 10)
        rsi_score = min(10, (current_rsi - 60) / 2)
        
        # LONG bonus: Ardışık yeşil mumlar (momentum teyidi)
        green_streak = 0
        for i in range(-1, max(-4, -len(df)), -1):
            if df['close'].iloc[i] > df['open'].iloc[i]:
                green_streak += 1
            else:
                break
        streak_bonus = min(5, green_streak * 2) if green_streak >= 2 else 0
        
        # Yön-spesifik bonus
        dir_bonus, dir_details = _calculate_direction_bonus(df, "LONG")
        
        final_score = score + vol_score + rsi_score + streak_bonus + dir_bonus
        
        details = {
            "strategy": "Volume Rocket",
            "direction": "LONG",
            "vol_mult": round(vol_mult, 2),
            "rsi": round(current_rsi, 1),
            "supertrend": round(current_st, 4),
            "dir_bonus": round(dir_bonus, 1),
        }
        if streak_bonus > 0:
            details['green_streak'] = f'✅ {green_streak} mum'
        details.update(dir_details)
        return True, "LONG", min(100, final_score), details
    
    # ========== SHORT CHECK ==========
    rsi_short_max = params.get('rsi_short_max', 40)
    if current_close < current_st and current_rsi < rsi_short_max:
        score = 55  # SHORT base (LONG'dan 5 düşük)
        vol_score = min(20, (vol_mult - spike_mult) * 10)
        rsi_score = min(10, (40 - current_rsi) / 2)
        
        # SHORT bonus: Ardışık kırmızı mumlar (düşüş momentum teyidi)
        red_streak = 0
        for i in range(-1, max(-4, -len(df)), -1):
            if df['close'].iloc[i] < df['open'].iloc[i]:
                red_streak += 1
            else:
                break
        streak_bonus = min(5, red_streak * 2) if red_streak >= 2 else 0
        
        # Yön-spesifik bonus
        dir_bonus, dir_details = _calculate_direction_bonus(df, "SHORT")
        
        final_score = score + vol_score + rsi_score + streak_bonus + dir_bonus
        
        details = {
            "strategy": "Volume Rocket Short",
            "direction": "SHORT",
            "vol_mult": round(vol_mult, 2),
            "rsi": round(current_rsi, 1),
            "supertrend": round(current_st, 4),
            "dir_bonus": round(dir_bonus, 1),
        }
        if streak_bonus > 0:
            details['red_streak'] = f'✅ {red_streak} mum'
        details.update(dir_details)
        return True, "SHORT", min(100, final_score), details
    
    return False, "", 0.0, {"fail": "No signal"}
