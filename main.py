import feedparser  
import asyncio
import os
import sys
import sqlite3
import time
import re
import logging
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
import mplfinance as mpf
import io
from datetime import datetime, timedelta, timezone
from google import genai
from telegram import Bot
from telegram.constants import ParseMode

# ==========================================
# 📋 LOGGING YAPILANDIRMASI
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Konsola yaz
        logging.FileHandler('titanium_bot.log', encoding='utf-8')  # Dosyaya yaz
    ]
)
logger = logging.getLogger(__name__)

# 🛡️ PRODUCTION RISK MANAGEMENT
from risk_manager import RiskManager
from regime_detector import RegimeDetector, PositionSizer, SlippageModel, MarketRegime
from state_manager import state_manager, periodic_save

logger.info("⚙️ TITANIUM PREMIUM BOT (V6.1: PRODUCTION HARDENED) BAŞLATILIYOR...")

# ==========================================
# 🔧 AYARLAR
# ==========================================
TOKEN = os.getenv("BOT_TOKEN", "").strip()
KANAL_ID = int(os.getenv("KANAL_ID", "0"))
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

if not TOKEN or not GEMINI_KEY or not KANAL_ID:
    logger.error("❌ HATA: ENV bilgileri eksik! (BOT_TOKEN, KANAL_ID, GEMINI_KEY)")
    # sys.exit(1) 

# Gemini Client
client = None
try:
    client = genai.Client(api_key=GEMINI_KEY, http_options={"api_version": "v1"})
except Exception as e:
    logger.warning(f"⚠️ Gemini Client başlatılamadı: {e}")

bot = Bot(token=TOKEN)

exchange_config = {
    'enableRateLimit': True,
    'rateLimit': 50,  # 50ms bekleme - Binance rate limit koruması
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True,
    },
    # Retry ayarları
    'timeout': 30000,  # 30 saniye timeout
}

# TITANIUM COIN LISTESI
COIN_LIST = [
    "BTC","ETH","SOL","XRP","BNB","ADA","AVAX","DOGE",
    "TON","LINK","DOT","POL","LTC","BCH","PEPE","FET",
    "SUI","APT","ARB","OP", "TIA", "INJ", "RENDER"
]

RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed"
]

SON_SINYAL_ZAMANI = {}
SON_RAPOR_TARIHI = None 

# ==========================================
# 🎯 SİNYAL OPTİMİZASYONU AYARLARI (V5.9)
# ==========================================
COIN_COOLDOWN_SAAT = 4      # Aynı coin için minimum bekleme süresi (saat)
GUNLUK_SINYAL_LIMIT = 999   # Günlük limit KALDIRILDI (eski: 8)
BUGUNUN_SINYALLERI = []     # Bugün üretilen sinyallerin listesi

# ==========================================
# 🧮 BÖLÜM 1: İNDİKATÖRLER
# ==========================================
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_sma(series, window):
    return series.rolling(window=window).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    # Prevent division by zero: replace 0 loss with small epsilon
    loss = loss.replace(0, 1e-10)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # Handle edge cases: clamp RSI to valid range and replace NaN/inf
    rsi = rsi.clip(0, 100)
    return rsi.fillna(50)  # Neutral RSI on insufficient data

def calculate_atr(df, period=14):
    """Calculate Average True Range for dynamic stops"""
    df = df.copy()
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    return df['tr'].rolling(period).mean()

# ADX İndikatörü
def calculate_adx(df, period=14):
    df = df.copy()
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    
    df['atr'] = df['tr'].rolling(period).mean()
    
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    # Prevent division by zero in DI calculations
    atr_safe = df['atr'].replace(0, 1e-10)
    df['plus_di'] = 100 * (df['plus_dm'].ewm(alpha=1/period).mean() / atr_safe)
    df['minus_di'] = 100 * (df['minus_dm'].ewm(alpha=1/period).mean() / atr_safe)
    
    # Prevent division by zero in DX calculation
    di_sum = df['plus_di'] + df['minus_di']
    di_sum = di_sum.replace(0, 1e-10)
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / di_sum
    df['adx'] = df['dx'].ewm(alpha=1/period).mean()
    
    # Replace NaN/inf with 0 (no trend)
    return df['adx'].fillna(0).replace([np.inf, -np.inf], 0)

def calculate_trend_aware_sl_multiplier(df, direction):
    """
    🛡️ TREND-UYUMLU DİNAMİK SL ÇARPANI (V6.1)
    
    Güçlü trend dönemlerinde SL'i genişlet, zayıf trendde daralt.
    
    Kriterler:
    - ADX > 35: Çok güçlü trend → 3.0x ATR
    - ADX > 25 + EMA dizilimi: Güçlü trend → 2.5x ATR
    - ADX > 20: Normal trend → 2.0x ATR
    - ADX < 20: Zayıf/Sideways → 1.5x ATR
    
    EMA Dizilimi:
    - LONG: EMA9 > EMA21 > EMA50 (bullish)
    - SHORT: EMA9 < EMA21 < EMA50 (bearish)
    
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

# ==========================================
# 🔄 BÖLÜM 1.5: ANİ YÖN DEĞİŞİMİ TESPİTİ (REVERSAL)
# ==========================================
def calculate_momentum_reversal(df, lookback=5, threshold=2.0):
    """
    Son X mumda ani momentum değişimi var mı?
    
    Args:
        df: OHLCV DataFrame
        lookback: Kaç mum geriye bakılacak
        threshold: Yüzde değişim eşiği
    
    Returns:
        reversal_type: 'REVERSAL_UP', 'REVERSAL_DOWN', veya None
        change_pct: Yüzde değişim
    """
    closes = df['close'].tail(lookback)
    
    # Son mum ile önceki mumların ortalaması arasındaki fark
    curr_close = closes.iloc[-1]
    avg_prev = closes.iloc[:-1].mean()
    change_pct = ((curr_close - avg_prev) / avg_prev) * 100
    
    # Eşiği geçen değişim = Ani hareket
    if change_pct > threshold:
        return 'REVERSAL_UP', change_pct
    elif change_pct < -threshold:
        return 'REVERSAL_DOWN', change_pct
    return None, change_pct

def check_rsi_divergence(df, lookback=14):
    """
    RSI Divergence tespit et - En güvenilir reversal sinyali
    
    Bullish Divergence: Fiyat düşük dip yaparken RSI yüksek dip yapar
    Bearish Divergence: Fiyat yüksek zirve yaparken RSI düşük zirve yapar
    
    Returns:
        divergence_type: 'BULLISH_DIV', 'BEARISH_DIV', veya None
        strength: Divergence gücü (0-100)
    """
    price = df['close'].tail(lookback)
    rsi = calculate_rsi(df['close']).tail(lookback)
    
    # Son 2 lokal minimum/maksimum bul
    price_min_idx = price.idxmin()
    price_max_idx = price.idxmax()
    
    curr_price = price.iloc[-1]
    curr_rsi = rsi.iloc[-1]
    
    # Bullish Divergence: Fiyat düşüyor ama RSI yükseliyor
    # Son 3 mumda fiyat düşerken RSI çıkıyorsa
    price_falling = price.iloc[-1] < price.iloc[-3]
    rsi_rising = rsi.iloc[-1] > rsi.iloc[-3]
    
    if price_falling and rsi_rising and curr_rsi < 40:
        strength = min(100, abs(rsi.iloc[-1] - rsi.iloc[-3]) * 5)
        return 'BULLISH_DIV', strength
    
    # Bearish Divergence: Fiyat çıkıyor ama RSI düşüyor
    price_rising = price.iloc[-1] > price.iloc[-3]
    rsi_falling = rsi.iloc[-1] < rsi.iloc[-3]
    
    if price_rising and rsi_falling and curr_rsi > 60:
        strength = min(100, abs(rsi.iloc[-3] - rsi.iloc[-1]) * 5)
        return 'BEARISH_DIV', strength
    
    return None, 0

def check_volatility_spike(df, period=14, multiplier=2.0):
    """
    Volatilite patlaması tespit et - ATR normal seviyenin X katı üstündeyse
    
    Returns:
        spike_type: 'SPIKE_UP', 'SPIKE_DOWN', veya None
        atr_ratio: Mevcut ATR / Ortalama ATR
    """
    atr = calculate_atr(df, period)
    avg_atr = atr.rolling(50).mean().iloc[-1]
    curr_atr = atr.iloc[-1]
    
    if pd.isna(avg_atr) or avg_atr == 0:
        return None, 1.0
    
    atr_ratio = curr_atr / avg_atr
    
    if atr_ratio > multiplier:
        # Mum rengine göre yön belirle
        is_bullish_candle = df['close'].iloc[-1] > df['open'].iloc[-1]
        if is_bullish_candle:
            return 'SPIKE_UP', atr_ratio
        else:
            return 'SPIKE_DOWN', atr_ratio
    
    return None, atr_ratio

def calculate_reversal_score(df):
    """
    Tüm reversal indikatörlerini birleştirerek skor hesapla
    
    Returns:
        reversal_long_score: LONG reversal skoru (0-30)
        reversal_short_score: SHORT reversal skoru (0-30)
        reversal_details: Detay bilgisi
    """
    long_score = 0
    short_score = 0
    details = []
    
    # 1. Momentum Reversal (max 10 puan)
    mom_type, mom_pct = calculate_momentum_reversal(df, lookback=5, threshold=1.5)
    if mom_type == 'REVERSAL_UP':
        score = min(10, int(abs(mom_pct) * 3))
        long_score += score
        details.append(f"MOM↑:{score}")
    elif mom_type == 'REVERSAL_DOWN':
        score = min(10, int(abs(mom_pct) * 3))
        short_score += score
        details.append(f"MOM↓:{score}")
    
    # 2. RSI Divergence (max 12 puan) - En güvenilir
    div_type, div_strength = check_rsi_divergence(df, lookback=14)
    if div_type == 'BULLISH_DIV':
        score = min(12, int(div_strength / 8))
        long_score += score
        details.append(f"DIV↑:{score}")
    elif div_type == 'BEARISH_DIV':
        score = min(12, int(div_strength / 8))
        short_score += score
        details.append(f"DIV↓:{score}")
    
    # 3. Volatility Spike (max 8 puan)
    spike_type, atr_ratio = check_volatility_spike(df, period=14, multiplier=1.8)
    if spike_type == 'SPIKE_UP':
        score = min(8, int((atr_ratio - 1) * 5))
        long_score += score
        details.append(f"VOL↑:{score}")
    elif spike_type == 'SPIKE_DOWN':
        score = min(8, int((atr_ratio - 1) * 5))
        short_score += score
        details.append(f"VOL↓:{score}")
    
    return long_score, short_score, details

# ==========================================
# 📊 BÖLÜM 1.55: RANGE TRADING MODU (DÜZ PİYASA)
# ==========================================
def calculate_bollinger(df, period=20, std_dev=2):
    """
    Bollinger Bantları hesapla
    
    Returns:
        lower_band, middle_band (SMA), upper_band
    """
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return lower, sma, upper

def detect_volatility_squeeze(df, bb_period=20, vol_lookback=10):
    """
    🔥 VOLATILITY SQUEEZE TESPİTİ (V6.1: ATR Z-score eklendi)
    
    Bollinger Squeeze + Volume Spike + ATR Normal = Patlama Öncesi Tespit
    
    Kriterler:
    - BB genişliği son 50 mumun en dar %20'sinde
    - Son 3 mumda hacim ortalamanın 1.5x+ üstüne çıkmış
    - ATR Z-score < 1.5 (volatilite henüz patlamadı)
    
    Returns:
        is_squeeze (bool), squeeze_score (int), squeeze_details (dict)
    """
    try:
        # Bollinger hesapla
        sma = df['close'].rolling(bb_period).mean()
        std = df['close'].rolling(bb_period).std()
        bb_width = ((std * 2) / sma * 100).iloc[-1]
        
        # BB genişliği tarihsel olarak düşük mü?
        bb_width_history = ((std * 2) / sma * 100).tail(50)
        bb_percentile = (bb_width_history < bb_width).sum() / len(bb_width_history) * 100
        is_bb_tight = bb_percentile < 20  # En dar %20'de
        
        # Hacim artıyor mu?
        vol_sma = df['volume'].rolling(20).mean().iloc[-1]
        recent_vol_avg = df['volume'].tail(3).mean()
        vol_ratio = recent_vol_avg / vol_sma if vol_sma > 0 else 1
        vol_expanding = vol_ratio > 1.5
        
        # 🆕 V6.1: ATR Z-score kontrolü (volatilite henüz patlamadı mı?)
        atr = calculate_atr(df)
        atr_current = atr.iloc[-1]
        atr_sma = atr.rolling(50).mean().iloc[-1]
        atr_std = atr.rolling(50).std().iloc[-1]
        
        atr_z_score = 0
        is_atr_normal = True
        if not pd.isna(atr_std) and atr_std > 0:
            atr_z_score = (atr_current - atr_sma) / atr_std
            is_atr_normal = atr_z_score < 1.5  # Volatilite henüz patlamadı
        
        # Üçü birden = SQUEEZE! (V6.1: ATR koşulu eklendi)
        is_squeeze = is_bb_tight and vol_expanding and is_atr_normal
        
        # Squeeze skoru hesapla (0-15 puan bonus)
        squeeze_score = 0
        if is_squeeze:
            # BB ne kadar dar? (max 8 puan)
            bb_score = min(8, int((20 - bb_percentile) / 2.5))
            # Hacim ne kadar yüksek? (max 7 puan)
            vol_score = min(7, int((vol_ratio - 1) * 3.5))
            squeeze_score = bb_score + vol_score
        
        details = {
            'bb_width': round(bb_width, 2),
            'bb_percentile': round(bb_percentile, 1),
            'vol_ratio': round(vol_ratio, 2),
            'atr_z_score': round(atr_z_score, 2),
            'is_bb_tight': is_bb_tight,
            'vol_expanding': vol_expanding,
            'is_atr_normal': is_atr_normal,
            'squeeze_score': squeeze_score
        }
        
        return is_squeeze, squeeze_score, details
        
    except Exception as e:
        logger.warning(f"⚠️ Squeeze detection error: {e}")
        return False, 0, {}


def is_ranging_market(df, adx_threshold=20):
    """
    Düz piyasa tespiti
    
    Kriterler:
    - ADX < threshold (trend yok)
    - ATR ortalamanın altında (düşük volatilite)
    - Bollinger bantları daralmış
    
    Returns:
        is_ranging (bool), range_details (dict)
    """
    # Tek seferde hesapla (optimizasyon)
    adx = calculate_adx(df).iloc[-1]
    atr_series = calculate_atr(df)
    atr = atr_series.iloc[-1]
    atr_sma = atr_series.rolling(50).mean().iloc[-1]
    
    # NaN kontrolü
    if pd.isna(adx) or pd.isna(atr):
        return False, {'adx': 0, 'atr_ratio': 1, 'bb_width': 0, 'criteria_met': 0}
    
    # Bollinger genişliği
    lower, mid, upper = calculate_bollinger(df)
    bb_width = ((upper.iloc[-1] - lower.iloc[-1]) / mid.iloc[-1]) * 100 if mid.iloc[-1] > 0 else 0
    bb_width_avg = ((upper - lower) / mid * 100).rolling(50).mean().iloc[-1]
    
    # Düz piyasa kriterleri
    is_low_adx = adx < adx_threshold
    is_low_volatility = atr < atr_sma * 0.9 if not pd.isna(atr_sma) and atr_sma > 0 else False
    is_bb_tight = bb_width < bb_width_avg * 0.9 if not pd.isna(bb_width_avg) and bb_width_avg > 0 else False
    
    # En az 2 kriter sağlanmalı
    criteria_met = sum([is_low_adx, is_low_volatility, is_bb_tight])
    is_ranging = criteria_met >= 2
    
    details = {
        'adx': adx,
        'atr_ratio': atr / atr_sma if atr_sma and atr_sma > 0 else 1,
        'bb_width': bb_width,
        'is_low_adx': is_low_adx,
        'is_low_volatility': is_low_volatility,
        'is_bb_tight': is_bb_tight,
        'criteria_met': criteria_met
    }
    
    return is_ranging, details

def calculate_range_score(df):
    """
    Range Trading için sinyal skoru (0-60 üzerinden)
    
    Puanlama:
    - Bollinger Alt/Üst Bant: 20 puan
    - RSI Oversold/Overbought: 15 puan
    - SMA20'den Sapma: 12 puan
    - Wick Rejection: 8 puan
    - Stochastic: 5 puan
    
    Eşik: 35/60 = Range Sinyal
    
    Returns:
        long_score, short_score, breakdown, tp_sl_info
    """
    long_score = 0
    short_score = 0
    long_breakdown = []
    short_breakdown = []
    
    price = df['close'].iloc[-1]
    
    # 1. Bollinger Bant Pozisyonu (max 20 puan)
    lower, mid, upper = calculate_bollinger(df)
    bb_lower = lower.iloc[-1]
    bb_mid = mid.iloc[-1]
    bb_upper = upper.iloc[-1]
    
    if bb_upper != bb_lower:
        bb_position = (price - bb_lower) / (bb_upper - bb_lower)
    else:
        bb_position = 0.5
    
    if bb_position < 0.15:  # Alt banda çok yakın
        long_score += 20
        long_breakdown.append("BB:20")
    elif bb_position < 0.25:  # Alt banda yakın
        long_score += 15
        long_breakdown.append("BB:15")
    elif bb_position < 0.35:
        long_score += 8
        long_breakdown.append("BB:8")
    
    if bb_position > 0.85:  # Üst banda çok yakın
        short_score += 20
        short_breakdown.append("BB:20")
    elif bb_position > 0.75:  # Üst banda yakın
        short_score += 15
        short_breakdown.append("BB:15")
    elif bb_position > 0.65:
        short_score += 8
        short_breakdown.append("BB:8")
    
    # 2. RSI (max 15 puan)
    rsi = calculate_rsi(df['close']).iloc[-1]
    
    if rsi < 30:
        long_score += 15
        long_breakdown.append("RSI:15")
    elif rsi < 35:
        long_score += 12
        long_breakdown.append("RSI:12")
    elif rsi < 40:
        long_score += 8
        long_breakdown.append("RSI:8")
    
    if rsi > 70:
        short_score += 15
        short_breakdown.append("RSI:15")
    elif rsi > 65:
        short_score += 12
        short_breakdown.append("RSI:12")
    elif rsi > 60:
        short_score += 8
        short_breakdown.append("RSI:8")
    
    # 3. SMA20'den Sapma (max 12 puan)
    sma20 = df['close'].rolling(20).mean().iloc[-1]
    deviation_pct = ((price - sma20) / sma20) * 100 if sma20 and sma20 > 0 else 0
    
    if deviation_pct < -2.0:
        long_score += 12
        long_breakdown.append("DEV:12")
    elif deviation_pct < -1.5:
        long_score += 8
        long_breakdown.append("DEV:8")
    elif deviation_pct < -1.0:
        long_score += 5
        long_breakdown.append("DEV:5")
    
    if deviation_pct > 2.0:
        short_score += 12
        short_breakdown.append("DEV:12")
    elif deviation_pct > 1.5:
        short_score += 8
        short_breakdown.append("DEV:8")
    elif deviation_pct > 1.0:
        short_score += 5
        short_breakdown.append("DEV:5")
    
    # 4. Wick Rejection (max 8 puan) - Son mumda fitil reddi
    row = df.iloc[-1]
    body = abs(row['close'] - row['open'])
    upper_wick = row['high'] - max(row['close'], row['open'])
    lower_wick = min(row['close'], row['open']) - row['low']
    
    if body > 0:
        lower_ratio = lower_wick / body
        upper_ratio = upper_wick / body
        
        if lower_ratio > 2.0:  # Uzun alt fitil = alıcı baskısı
            long_score += 8
            long_breakdown.append("WICK:8")
        elif lower_ratio > 1.5:
            long_score += 5
            long_breakdown.append("WICK:5")
        
        if upper_ratio > 2.0:  # Uzun üst fitil = satıcı baskısı
            short_score += 8
            short_breakdown.append("WICK:8")
        elif upper_ratio > 1.5:
            short_score += 5
            short_breakdown.append("WICK:5")
    
    # 5. Stochastic benzeri kontrol (max 5 puan)
    # Son 14 mumun en düşük ve en yükseğine göre pozisyon
    low_14 = df['low'].tail(14).min()
    high_14 = df['high'].tail(14).max()
    
    if high_14 != low_14:
        stoch_k = ((price - low_14) / (high_14 - low_14)) * 100
    else:
        stoch_k = 50
    
    if stoch_k < 20:
        long_score += 5
        long_breakdown.append("STOCH:5")
    elif stoch_k < 30:
        long_score += 3
        long_breakdown.append("STOCH:3")
    
    if stoch_k > 80:
        short_score += 5
        short_breakdown.append("STOCH:5")
    elif stoch_k > 70:
        short_score += 3
        short_breakdown.append("STOCH:3")
    
    # TP/SL için range bilgisi
    atr = calculate_atr(df).iloc[-1]
    tp_sl_info = {
        'bb_mid': bb_mid,
        'bb_lower': bb_lower,
        'bb_upper': bb_upper,
        'atr': atr,
        'bb_position': bb_position
    }
    
    return long_score, short_score, long_breakdown, short_breakdown, tp_sl_info

# ==========================================
# ⚡ BÖLÜM 1.6: RAPID REVERSAL STRATEJİSİ
# ==========================================
def detect_flash_move(df, threshold_pct=3.0, lookback=3):
    """
    Ani fiyat hareketi tespiti - Son X mumda %threshold+ değişim
    
    Returns:
        flash_type: 'FLASH_UP', 'FLASH_DOWN', veya None
        change_pct: Yüzde değişim
    """
    if len(df) < lookback + 1:
        return None, 0
    
    closes = df['close'].tail(lookback + 1)
    start_price = closes.iloc[0]
    end_price = closes.iloc[-1]
    
    change_pct = ((end_price - start_price) / start_price) * 100
    
    # Son mum yeşil mi kırmızı mı?
    last_candle_bullish = df['close'].iloc[-1] > df['open'].iloc[-1]
    
    # Flash DOWN sonrası yeşil mum = LONG fırsatı
    if change_pct < -threshold_pct and last_candle_bullish:
        return 'FLASH_UP', abs(change_pct)
    
    # Flash UP sonrası kırmızı mum = SHORT fırsatı
    if change_pct > threshold_pct and not last_candle_bullish:
        return 'FLASH_DOWN', abs(change_pct)
    
    return None, abs(change_pct)

def detect_volume_spike(df, multiplier=3.0, lookback=20):
    """
    Hacim patlaması tespiti - Ortalamanın X katı hacim
    
    Returns:
        spike_type: 'VOL_SPIKE_UP', 'VOL_SPIKE_DOWN', veya None
        vol_ratio: Hacim oranı
    """
    if len(df) < lookback:
        return None, 1.0
    
    vol_sma = df['volume'].tail(lookback).mean()
    curr_vol = df['volume'].iloc[-1]
    
    if vol_sma == 0:
        return None, 1.0
    
    vol_ratio = curr_vol / vol_sma
    
    if vol_ratio >= multiplier:
        # Mum rengine göre yön belirle
        is_bullish = df['close'].iloc[-1] > df['open'].iloc[-1]
        if is_bullish:
            return 'VOL_SPIKE_UP', vol_ratio
        else:
            return 'VOL_SPIKE_DOWN', vol_ratio
    
    return None, vol_ratio

def detect_wick_rejection(df, wick_body_ratio=2.0):
    """
    Fitil reddi tespiti - Uzun fitil = Reddedilen seviye
    
    Returns:
        wick_type: 'WICK_UP' (uzun alt fitil=alıcı), 'WICK_DOWN' (uzun üst fitil=satıcı), veya None
        wick_ratio: Fitil/gövde oranı
    """
    row = df.iloc[-1]
    
    body = abs(row['close'] - row['open'])
    upper_wick = row['high'] - max(row['close'], row['open'])
    lower_wick = min(row['close'], row['open']) - row['low']
    
    if body == 0:
        body = 0.0001  # Doji durumu
    
    upper_ratio = upper_wick / body
    lower_ratio = lower_wick / body
    
    # Uzun alt fitil = Alıcı baskısı (LONG sinyali)
    if lower_ratio >= wick_body_ratio and lower_ratio > upper_ratio:
        return 'WICK_UP', lower_ratio
    
    # Uzun üst fitil = Satıcı baskısı (SHORT sinyali)
    if upper_ratio >= wick_body_ratio and upper_ratio > lower_ratio:
        return 'WICK_DOWN', upper_ratio
    
    return None, max(upper_ratio, lower_ratio)

def detect_rsi_extreme_bounce(df, oversold=25, overbought=75):
    """
    RSI aşırı bölgeden dönüş tespiti
    
    Returns:
        bounce_type: 'RSI_BOUNCE_UP', 'RSI_BOUNCE_DOWN', veya None
        rsi_value: Mevcut RSI değeri
    """
    if len(df) < 15:
        return None, 50
    
    rsi = calculate_rsi(df['close'])
    curr_rsi = rsi.iloc[-1]
    prev_rsi = rsi.iloc[-2]
    
    # Son mum yönü
    is_bullish = df['close'].iloc[-1] > df['open'].iloc[-1]
    
    # Oversold'dan dönüş + yeşil mum
    if prev_rsi < oversold and curr_rsi > prev_rsi and is_bullish:
        return 'RSI_BOUNCE_UP', curr_rsi
    
    # Overbought'tan dönüş + kırmızı mum
    if prev_rsi > overbought and curr_rsi < prev_rsi and not is_bullish:
        return 'RSI_BOUNCE_DOWN', curr_rsi
    
    return None, curr_rsi

def calculate_rapid_score(df):
    """
    Rapid Reversal için ayrı skor hesapla (0-100)
    
    Puanlama:
    - Flash Move: 25 puan
    - Volume Spike: 25 puan
    - RSI Extreme: 20 puan
    - ATR Explosion: 15 puan
    - Wick Rejection: 15 puan
    
    Returns:
        rapid_long_score, rapid_short_score, rapid_details, tetikleyici
    """
    long_score = 0
    short_score = 0
    details = []
    tetikleyici = []
    
    # 1. Flash Move (25 puan)
    flash_type, flash_pct = detect_flash_move(df, threshold_pct=3.0)
    if flash_type == 'FLASH_UP':
        score = min(25, int(flash_pct * 5))
        long_score += score
        details.append(f"Flash:{score}")
        tetikleyici.append(f"Flash Move {flash_pct:.1f}%")
    elif flash_type == 'FLASH_DOWN':
        score = min(25, int(flash_pct * 5))
        short_score += score
        details.append(f"Flash:{score}")
        tetikleyici.append(f"Flash Move {flash_pct:.1f}%")
    
    # 2. Volume Spike (25 puan)
    vol_type, vol_ratio = detect_volume_spike(df, multiplier=3.0)
    if vol_type == 'VOL_SPIKE_UP':
        score = min(25, int((vol_ratio - 1) * 8))
        long_score += score
        details.append(f"Vol:{score}")
        tetikleyici.append(f"Volume {vol_ratio:.1f}x")
    elif vol_type == 'VOL_SPIKE_DOWN':
        score = min(25, int((vol_ratio - 1) * 8))
        short_score += score
        details.append(f"Vol:{score}")
        tetikleyici.append(f"Volume {vol_ratio:.1f}x")
    
    # 3. RSI Extreme Bounce (20 puan)
    rsi_type, rsi_val = detect_rsi_extreme_bounce(df)
    if rsi_type == 'RSI_BOUNCE_UP':
        long_score += 20
        details.append("RSI:20")
        tetikleyici.append(f"RSI Bounce ({rsi_val:.0f})")
    elif rsi_type == 'RSI_BOUNCE_DOWN':
        short_score += 20
        details.append("RSI:20")
        tetikleyici.append(f"RSI Bounce ({rsi_val:.0f})")
    
    # 4. ATR Explosion (15 puan)
    spike_type, atr_ratio = check_volatility_spike(df, period=14, multiplier=2.5)
    if spike_type == 'SPIKE_UP':
        score = min(15, int((atr_ratio - 1) * 6))
        long_score += score
        details.append(f"ATR:{score}")
    elif spike_type == 'SPIKE_DOWN':
        score = min(15, int((atr_ratio - 1) * 6))
        short_score += score
        details.append(f"ATR:{score}")
    
    # 5. Wick Rejection (15 puan)
    wick_type, wick_ratio = detect_wick_rejection(df, wick_body_ratio=2.0)
    if wick_type == 'WICK_UP':
        score = min(15, int(wick_ratio * 3))
        long_score += score
        details.append(f"Wick:{score}")
        tetikleyici.append("Wick Rejection")
    elif wick_type == 'WICK_DOWN':
        score = min(15, int(wick_ratio * 3))
        short_score += score
        details.append(f"Wick:{score}")
        tetikleyici.append("Wick Rejection")
    
    return long_score, short_score, details, tetikleyici

# ==========================================
# 🎨 BÖLÜM 2: GRAFİK
# ==========================================
def _grafik_olustur_sync(coin, df_gelen, tp, sl, yon):
    try:
        df = df_gelen.copy()
        apds = [
            mpf.make_addplot(df['sma50'], panel=0, color='cyan', width=0.8),
            mpf.make_addplot(df['sma200'], panel=0, color='white', width=1.0),
            mpf.make_addplot(df['rsi'], panel=1, color='#FF6D00', width=1.0, title="RSI")
        ]
        
        h_lines_rsi = dict(hlines=[30, 80], colors=['green', 'red'], linewidths=[0.5, 0.5], linestyle='--')
        
        buf = io.BytesIO()
        theme_color = '#131722'
        grid_color = '#363c4e'
        text_color = '#b2b5be'
        my_style = mpf.make_mpf_style(
            base_mpf_style='binance', facecolor=theme_color, figcolor=theme_color, edgecolor=theme_color,
            gridcolor=grid_color, gridstyle=':', rc={'axes.labelcolor': text_color, 'xtick.color': text_color, 'ytick.color': text_color, 'text.color': text_color}
        )
        
        h_lines = dict(
            hlines=[tp, sl], 
            colors=['#00FF00', '#FF0000'],
            linewidths=[1.5, 1.5], alpha=0.9, linestyle='-.'
        )
        
        mpf.plot(
            df, type='candle', style=my_style, title=f"\n{coin}/USDT - TITANIUM {yon}",
            ylabel='Fiyat ($)', ylabel_lower='RSI', addplot=apds, hlines=h_lines, volume=False,
            panel_ratios=(3, 1), savefig=dict(fname=buf, dpi=120, bbox_inches='tight', facecolor=theme_color)
        )
        buf.seek(0)
        return buf
    except Exception as e:
        logger.error(f"Grafik Hatası: {e}")
        return None

async def grafik_olustur_async(coin, df, tp, sl, yon):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _grafik_olustur_sync, coin, df, tp, sl, yon)

# ==========================================
# 📊 BÖLÜM 3: DB & SİNYAL YÖNETİMİ
# ==========================================
def db_ilk_kurulum():
    with sqlite3.connect("titanium_live.db") as conn:
        # Islemler Tablosu (Multi-TP Support)
        conn.execute("""CREATE TABLE IF NOT EXISTS islemler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT, yon TEXT, giris_fiyat REAL, 
            tp1 REAL, tp2 REAL, tp3 REAL, sl REAL,
            tp1_hit INTEGER DEFAULT 0, tp2_hit INTEGER DEFAULT 0,
            durum TEXT DEFAULT 'ACIK', pnl_yuzde REAL DEFAULT 0,
            acilis_zamani DATETIME, kapanis_zamani DATETIME
        )""")
        
        # Migrate old table if tp2/tp3 columns missing
        try:
            conn.execute("ALTER TABLE islemler ADD COLUMN tp2 REAL")
            conn.execute("ALTER TABLE islemler ADD COLUMN tp3 REAL")
            conn.execute("ALTER TABLE islemler ADD COLUMN tp1_hit INTEGER DEFAULT 0")
            conn.execute("ALTER TABLE islemler ADD COLUMN tp2_hit INTEGER DEFAULT 0")
            # Rename old tp column to tp1
            conn.execute("UPDATE islemler SET tp1 = tp, tp2 = tp, tp3 = tp WHERE tp1 IS NULL") # Assuming 'tp' was the old column name
            conn.execute("ALTER TABLE islemler DROP COLUMN tp") # Drop the old single TP column
        except sqlite3.OperationalError as e:
            # This error occurs if the column already exists or if 'tp' column doesn't exist to drop
            if "duplicate column name" not in str(e) and "no such column" not in str(e):
                logger.warning(f"DB Migration Error: {e}")
        except Exception as e:
            logger.error(f"Unexpected DB Migration Error: {e}")
        
        # Haberler Tablosu (Haber Hafızası)
        conn.execute("CREATE TABLE IF NOT EXISTS haberler (link TEXT PRIMARY KEY)")

def short_var_mi(coin):
    """Check if there's an open SHORT position for a coin"""
    with sqlite3.connect("titanium_live.db") as conn:
        c = conn.cursor()
        c.execute("SELECT count(*) FROM islemler WHERE coin=? AND yon='SHORT' AND durum='ACIK'", (coin,))
        count = c.fetchone()[0]
        return count > 0

def pozisyon_acik_mi(coin):
    """Check if there's ANY open position (LONG or SHORT) for a coin - Anti-Spam"""
    with sqlite3.connect("titanium_live.db") as conn:
        c = conn.cursor()
        c.execute("SELECT count(*) FROM islemler WHERE coin=? AND durum='ACIK'", (coin,))
        count = c.fetchone()[0]
        return count > 0

def islem_kaydet(coin, yon, giris, tp1, tp2, tp3, sl):
    """Save trade with multiple take profit levels"""
    with sqlite3.connect("titanium_live.db") as conn:
        conn.execute("INSERT INTO islemler (coin, yon, giris_fiyat, tp1, tp2, tp3, sl, acilis_zamani) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
                  (coin, yon, giris, tp1, tp2, tp3, sl, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

async def gunluk_rapor_gonder(tarih=None):
    try:
        bugun = tarih if tarih else datetime.now().strftime("%Y-%m-%d")
        logger.info(f"📊 {bugun} Günlük Rapor Hazırlanıyor...")

        with sqlite3.connect("titanium_live.db") as conn:
            # TP hit bilgilerini de çek
            query = """
            SELECT coin, yon, durum, pnl_yuzde, tp1_hit, tp2_hit, kapanis_zamani 
            FROM islemler 
            WHERE durum IN ('KAZANDI', 'KAYBETTI', 'PARTIAL') 
            AND date(kapanis_zamani) = ?
            """
            df_rapor = pd.read_sql_query(query, conn, params=(bugun,))

        if df_rapor.empty:
            logger.info(f"ℹ️ {bugun} için raporlanacak işlem yok, boş rapor gönderiliyor.")
            mesaj = f"📅 <b>GÜNLÜK RAPOR ({bugun})</b>\n\nℹ️ <i>Bugün herhangi bir işlem sonlanmadı.</i>\n\n💰 <b>NET PNL:</b> ➖ <b>%0.00</b>"
            await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
            return

        toplam_pnl = df_rapor['pnl_yuzde'].sum()
        
        # Detaylı istatistikler
        full_win = len(df_rapor[df_rapor['durum'] == 'KAZANDI'])  # TP3 tam kazanç
        partial_win = len(df_rapor[df_rapor['durum'] == 'PARTIAL'])  # Kısmi kazanç (TP1/TP2 hit + SL)
        loss_count = len(df_rapor[df_rapor['durum'] == 'KAYBETTI'])  # Tam kayıp
        total_count = len(df_rapor)
        
        # Kazanç oranı (tam + kısmi kazanç)
        win_count = full_win + partial_win
        win_rate = (win_count / total_count) * 100 if total_count > 0 else 0
        pnl_ikon = "✅" if toplam_pnl > 0 else "🔻"
        
        mesaj = f"📅 <b>GÜNLÜK RAPOR ({bugun})</b>\n\n"
        
        for index, row in df_rapor.iterrows():
            # W veya L
            is_win = row['durum'] in ['KAZANDI', 'PARTIAL']
            wl = "W" if is_win else "L"
            
            # Hangi TP'ler vuruldu
            tp_list = []
            if row.get('tp1_hit', 0):
                tp_list.append("TP1")
            if row.get('tp2_hit', 0):
                tp_list.append("TP2")
            if row['durum'] == 'KAZANDI':
                tp_list.append("TP3")
            
            tp_str = ",".join(tp_list) if tp_list else "-"
            
            # PnL
            pnl_val = row['pnl_yuzde']
            pnl_str = f"+{pnl_val:.1f}" if pnl_val >= 0 else f"{pnl_val:.1f}"
            
            mesaj += f"<code>{row['coin'][:4]:<5}|{row['yon'][0]}|{wl}|{tp_str}|{pnl_str}%</code>\n"
        
        mesaj += f"\n📊 <b>İSTATİSTİKLER</b>\n"
        mesaj += f"🏆 <b>Tam Kazanç:</b> {full_win} | ⚡ <b>Kısmi:</b> {partial_win} | ❌ <b>Kayıp:</b> {loss_count}\n"
        mesaj += f"🔢 <b>Toplam:</b> {total_count} | 🎯 <b>WR:</b> %{win_rate:.0f}\n"
        mesaj += f"💰 <b>NET PNL:</b> {pnl_ikon} <b>%{toplam_pnl:.2f}</b>\n"
        
        # 💵 $100 SİMÜLASYONU
        yatirim_per_sinyal = 100  # Her sinyale $100
        toplam_yatirim = total_count * yatirim_per_sinyal
        toplam_kar = sum([(row['pnl_yuzde'] / 100) * yatirim_per_sinyal for _, row in df_rapor.iterrows()])
        final_bakiye = toplam_yatirim + toplam_kar
        kar_ikon = "📈" if toplam_kar >= 0 else "📉"
        
        mesaj += f"\n💵 <b>$100 SİMÜLASYONU</b>\n"
        mesaj += f"🏦 <b>Yatırım:</b> ${toplam_yatirim} ({total_count} x $100)\n"
        mesaj += f"{kar_ikon} <b>Kâr/Zarar:</b> ${toplam_kar:+.2f}\n"
        mesaj += f"💎 <b>Final:</b> <b>${final_bakiye:.2f}</b>"

        await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"❌ Günlük Rapor Hatası: {e}")



# ==========================================
# 🧠 BÖLÜM 4: AI HABER ANALİZİ
# ==========================================
def link_kontrol(link):
    with sqlite3.connect("titanium_live.db") as conn:
        c = conn.cursor()
        try:
            c.execute("INSERT INTO haberler VALUES (?)", (link,))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

def _ai_analiz_sync(prompt):
    try:
        if client is None:
            return "AI client mevcut değil.", 0
        r = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        text = r.text.strip()
        ozet_match = re.search(r"ÖZET:(.*)", text, re.DOTALL)
        skor_match = re.search(r"SKOR:\s*(-?\d)", text)
        temiz_ozet = ozet_match.group(1).strip() if ozet_match else "Özet oluşturulamadı."
        skor = int(skor_match.group(1)) if skor_match else 0
        return temiz_ozet, skor
    except Exception as e:
        logger.warning(f"⚠️ AI Analiz Hatası: {e}")
        return "Analiz yapılamadı.", 0

async def ai_analiz(baslik, ozet):
    prompt = f"GÖREV: Haber analizi.\nBAŞLIK: {baslik}\nÖZET: {ozet}\nFORMAT:\nÖZET:[Kısa özet]\nSKOR:[-2 ile +2 arası tamsayı]"
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _ai_analiz_sync, prompt)

async def haberleri_kontrol_et():
    logger.info("📰 Haberler taranıyor...")
    for rss in RSS_LIST:
        try:
            feed = feedparser.parse(rss)
            for entry in feed.entries[:2]:
                if not link_kontrol(entry.link): continue 
                if entry.published_parsed:
                    t = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    if (datetime.now() - t) > timedelta(minutes=45): continue
                
                clean_text = re.sub('<[^<]+?>', '', entry.get("summary", ""))
                ai_text, skor = await ai_analiz(entry.title, clean_text[:500])
                if abs(skor) < 2: continue
                
                skor_icon = "🟢" if skor > 0 else "🔴"
                mesaj = f"<b>{entry.title}</b>\n\n{ai_text}\n\n🎯 <b>Etki:</b> {skor_icon} <b>({skor})</b>\n🔗 <a href='{entry.link}'>Link</a>"
                try:
                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                except Exception as tg_err:
                    logger.warning(f"⚠️ Telegram Haber Gönderim Hatası: {tg_err}")
                await asyncio.sleep(2)
        except Exception as rss_err:
            logger.warning(f"⚠️ RSS Okuma Hatası ({rss}): {rss_err}")

# ==========================================
# 🧠 BÖLÜM 4.5: AKILLI TRAILING - TREND GÜCÜ ANALİZİ
# ==========================================
async def trend_gucunu_analiz_et(exchange, coin, yon, mevcut_fiyat):
    """
    TP hit sonrası trend gücünü analiz et
    
    Kriterler (her biri 25 puan):
    1. RSI: LONG için >55, SHORT için <45
    2. Fiyat vs SMA20: Trendle uyumlu mu?
    3. ADX: >25 ise trend güçlü
    4. Hacim: Ortalama üstü mü?
    
    Returns:
        trend_gucu (str): 'GUCLU', 'ORTA', 'ZAYIF'
        sl_multiplier (float): ATR çarpanı
        analiz_detay (dict): Detaylı analiz bilgisi
    """
    try:
        ohlcv = await exchange.fetch_ohlcv(f"{coin}/USDT", '1h', limit=50)
        df = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        # İndikatörleri hesapla
        rsi_now = calculate_rsi(df['close']).iloc[-1]
        sma20 = df['close'].rolling(20).mean().iloc[-1]
        adx_now = calculate_adx(df).iloc[-1]
        atr_now = calculate_atr(df, 14).iloc[-1]
        
        # Hacim analizi
        vol_sma = df['volume'].rolling(20).mean().iloc[-1]
        curr_vol = df['volume'].iloc[-1]
        vol_ratio = curr_vol / vol_sma if vol_sma > 0 else 1
        
        puan = 0
        detay = {
            'rsi': rsi_now,
            'rsi_ok': False,
            'sma20_ok': False,
            'adx': adx_now,
            'adx_ok': False,
            'vol_ratio': vol_ratio,
            'vol_ok': False,
            'atr': atr_now
        }
        
        # 1. RSI Kontrolü (25 puan)
        if yon == "LONG":
            if rsi_now > 55:
                puan += 25
                detay['rsi_ok'] = True
            elif rsi_now > 45:
                puan += 15  # Nötr bölge
        else:  # SHORT
            if rsi_now < 45:
                puan += 25
                detay['rsi_ok'] = True
            elif rsi_now < 55:
                puan += 15  # Nötr bölge
        
        # 2. Fiyat vs SMA20 (25 puan)
        if yon == "LONG":
            if mevcut_fiyat > sma20:
                puan += 25
                detay['sma20_ok'] = True
        else:  # SHORT
            if mevcut_fiyat < sma20:
                puan += 25
                detay['sma20_ok'] = True
        
        # 3. ADX Kontrolü (25 puan)
        if adx_now > 30:
            puan += 25
            detay['adx_ok'] = True
        elif adx_now > 25:
            puan += 18
            detay['adx_ok'] = True
        elif adx_now > 20:
            puan += 10
        
        # 4. Hacim Kontrolü (25 puan)
        if vol_ratio > 1.5:
            puan += 25
            detay['vol_ok'] = True
        elif vol_ratio > 1.2:
            puan += 18
            detay['vol_ok'] = True
        elif vol_ratio > 1.0:
            puan += 10
        
        # Trend gücü ve SL çarpanı belirleme
        if puan >= 75:
            trend_gucu = "GUCLU"
            sl_multiplier = 1.5  # Geniş SL - Gideni tutma!
        elif puan >= 40:
            trend_gucu = "ORTA"
            sl_multiplier = 1.0  # Dengeli SL
        else:
            trend_gucu = "ZAYIF"
            sl_multiplier = 0.0  # Buffer moduna geç
        
        detay['puan'] = puan
        
        logger.debug(f"📊 TREND ANALİZİ: {coin} ({yon})")
        logger.debug(f"   RSI: {rsi_now:.1f} {'✅' if detay['rsi_ok'] else '❌'}")
        logger.debug(f"   SMA20: {'✅' if detay['sma20_ok'] else '❌'}")
        logger.debug(f"   ADX: {adx_now:.1f} {'✅' if detay['adx_ok'] else '❌'}")
        logger.debug(f"   Hacim: {vol_ratio:.2f}x {'✅' if detay['vol_ok'] else '❌'}")
        logger.debug(f"   TOPLAM: {puan}/100 → {trend_gucu}")
        
        return trend_gucu, sl_multiplier, detay
        
    except Exception as e:
        logger.error(f"⚠️ Trend Analiz Hatası ({coin}): {e}")
        # Hata durumunda güvenli mod - orta seviye
        return "ORTA", 1.0, {'puan': 50, 'atr': 0}

# ==========================================
# 🚀 BÖLÜM 5: STRATEJİ MOTORU (VOLUME + SCORING)
# ==========================================

async def usdt_hacim_akisi_analiz(exchange):
    """
    USDT Hacim Akışı Analizi (CVD - Cumulative Volume Delta)
    
    CVD Mantığı:
    - Her mum için delta = (close - open) / (high - low) * volume * price
    - Bu oran mum içindeki gerçek alış/satış oranını yaklaşık gösterir
    - Toplam CVD pozitifse -> Net alış baskısı (BULLISH)
    - Toplam CVD negatifse -> Net satış baskısı (BEARISH)
    
    Returns:
        usdt_score: -1.0 ile +1.0 arası puan
        cvd_millions: Net CVD (milyon $ cinsinden)
        flow_direction: 'INFLOW', 'OUTFLOW', 'NEUTRAL'
    """
    try:
        # Majör coinler - yüksek hacimli
        major_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT"]
        
        total_cvd = 0.0
        total_volume_usd = 0.0
        
        for pair in major_pairs:
            try:
                ohlcv = await exchange.fetch_ohlcv(pair, '1h', limit=24)  # Son 24 saat
                if not ohlcv:
                    continue
                
                df = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                
                for i in range(len(df)):
                    row = df.iloc[i]
                    
                    # Mum aralığı
                    candle_range = row['high'] - row['low']
                    
                    # USDT cinsinden hacim
                    usdt_vol = row['volume'] * row['close']
                    total_volume_usd += usdt_vol
                    
                    if candle_range > 0:
                        # CVD Delta = Mum içindeki alış-satış oranı
                        # (close - open) / range = -1 ile +1 arası değer
                        # +1 = Tam alış hakimiyeti (close = high)
                        # -1 = Tam satış hakimiyeti (close = low)
                        delta_ratio = (row['close'] - row['open']) / candle_range
                        
                        # Ağırlıklı CVD: Delta oranı × USDT hacmi
                        weighted_cvd = delta_ratio * usdt_vol
                        total_cvd += weighted_cvd
                    else:
                        # Doji durumu - mum açılış/kapanışa göre karar ver
                        if row['close'] > row['open']:
                            total_cvd += usdt_vol * 0.5  # Hafif alış
                        elif row['close'] < row['open']:
                            total_cvd -= usdt_vol * 0.5  # Hafif satış
                        
            except Exception as e:
                continue
        
        if total_volume_usd == 0:
            return 0.0, 0.0, 'NEUTRAL'
        
        # CVD oranı hesapla (-1 ile +1 arası)
        cvd_ratio = total_cvd / total_volume_usd
        
        # Skor hesapla (max ±1.0) - 3x amplify (CVD daha hassas)
        usdt_score = max(-1.0, min(1.0, cvd_ratio * 3))
        
        # Yön belirle
        if cvd_ratio > 0.03:  # %3+ net alış baskısı
            flow_direction = 'INFLOW'
        elif cvd_ratio < -0.03:  # %3+ net satış baskısı
            flow_direction = 'OUTFLOW'
        else:
            flow_direction = 'NEUTRAL'
        
        # Milyon $ cinsinden CVD
        cvd_millions = total_cvd / 1_000_000
        
        # Detaylı log
        logger.debug(f"📊 CVD AKIŞ: TotalVol=${total_volume_usd/1e9:.2f}B | CVD={cvd_millions:+.1f}M$ | Ratio={cvd_ratio*100:+.2f}% | {flow_direction}")
        
        return usdt_score, cvd_millions, flow_direction
        
    except Exception as e:
        logger.warning(f"⚠️ CVD Hacim Analiz Hatası: {e}")
        return 0.0, 0.0, 'NEUTRAL'

async def btc_piyasa_puani_hesapla(exchange):
    """
    BTC için Piyasa Puanı (-2.5 ile +2.5 arası)
    
    Kriterler:
    1. SMA 200 (Ana Trend): +/- 1.0
    2. SMA 50 (Kısa Trend): +/- 0.5
    3. RSI 50 (Momentum):   +/- 0.5
    4. HACİM (Volume):      +/- 0.5 (Teyitçi)
    """
    try:
        # BTC verisini çek
        ohlcv = await exchange.fetch_ohlcv("BTC/USDT", '1h', limit=210)
        if not ohlcv: return 0
        
        df = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        # Son değerleri al
        price = df['close'].iloc[-1]
        open_price = df['open'].iloc[-1]
        sma50 = df['close'].rolling(50).mean().iloc[-1]
        sma200 = df['close'].rolling(200).mean().iloc[-1]
        rsi = calculate_rsi(df['close']).iloc[-1]
        
        # --- HACİM ANALİZİ ---
        vol_sma = df['volume'].rolling(20).mean().iloc[-1] # Son 20 mum ortalaması
        curr_vol = df['volume'].iloc[-1] # Şu anki hacim
        vol_ratio = curr_vol / vol_sma
        
        score = 0.0
        
        # 1. ANA TREND
        if price > sma200: score += 1.0
        else: score -= 1.0
            
        # 2. KISA TREND
        if price > sma50: score += 0.5
        else: score -= 0.5
            
        # 3. MOMENTUM
        if rsi > 50: score += 0.5
        else: score -= 0.5
        
        # 4. HACİM TEYİDİ (YENİ EKLENEN KISIM)
        # Hacim ortalamanın %25 üzerindeyse güçlü kabul et
        if vol_ratio > 1.25:
            # Yeşil Mum + Yüksek Hacim = Güçlü Alış (+0.5)
            if price > open_price: 
                score += 0.5
            # Kırmızı Mum + Yüksek Hacim = Güçlü Satış (-0.5)
            else: 
                score -= 0.5
                
        return score
    except Exception as e:
        logger.warning(f"⚠️ BTC Puan Hatası: {e}")
        return 0

async def piyasayi_tarama(exchange):
    logger.info(f"🔍 ({datetime.now().strftime('%H:%M')}) TITANIUM V5.9 OPTİMİZE TARAMA (Eşik:75)...")
    
    # 1. BTC PUANINI HESAPLA (Volume Destekli)
    btc_score = await btc_piyasa_puani_hesapla(exchange)
    
    # İkon Belirleme
    if btc_score >= 1.5: btc_ikon = "🟢🟢 (Güçlü Bull)"
    elif btc_score >= 0.5: btc_ikon = "🟢 (Bull)"
    elif btc_score <= -1.5: btc_ikon = "🔴🔴 (Güçlü Bear)"
    elif btc_score <= -0.5: btc_ikon = "🔴 (Bear)"
    else: btc_ikon = "⚪ (Nötr)"

    logger.info(f"🌍 BTC SKORU: {btc_score} -> {btc_ikon}")
    
    # 2. USDT HACİM AKIŞI ANALİZİ (YENİ!)
    usdt_score, usdt_flow_m, usdt_direction = await usdt_hacim_akisi_analiz(exchange)
    
    # USDT İkon Belirleme
    if usdt_direction == 'INFLOW':
        usdt_ikon = "💹 INFLOW" if usdt_score > 0.5 else "📈 Hafif Giriş"
    elif usdt_direction == 'OUTFLOW':
        usdt_ikon = "📉 OUTFLOW" if usdt_score < -0.5 else "💸 Hafif Çıkış"
    else:
        usdt_ikon = "➡️ Nötr"
    
    logger.info(f"💵 USDT AKIŞ SKORU: {usdt_score:.2f} | Net: {usdt_flow_m:+.1f}M$ | {usdt_ikon}")
    
    # 2. COIN VERILERINI CEK
    async def fetch_candle(s):
        try:
            ohlcv = await exchange.fetch_ohlcv(f"{s}/USDT", '1h', limit=300)
            return s, ohlcv
        except Exception as e:
            logger.warning(f"⚠️ {s} veri çekme hatası: {e}")
            return s, None

    tasks = [fetch_candle(c) for c in COIN_LIST]
    results = await asyncio.gather(*tasks)
    
    # ========== HTF (4H) TREND DATA ==========
    async def fetch_htf_candle(s):
        """4H mum verisi çek - HTF trend teyidi için"""
        try:
            ohlcv_4h = await exchange.fetch_ohlcv(f"{s}/USDT", '4h', limit=60)
            return s, ohlcv_4h
        except Exception as e:
            logger.warning(f"⚠️ {s} HTF veri hatası: {e}")
            return s, None
    
    htf_tasks = [fetch_htf_candle(c) for c in COIN_LIST]
    htf_results = await asyncio.gather(*htf_tasks)
    htf_data = {coin: data for coin, data in htf_results}
    
    for coin, bars in results:
        if not bars: continue
        
        df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df.set_index('date', inplace=True)
        
        # İndikatörler
        df['sma50'] = calculate_sma(df['close'], 50)
        df['sma200'] = calculate_sma(df['close'], 200)
        df['rsi'] = calculate_rsi(df['close'])
        df['adx'] = calculate_adx(df)
        df['atr'] = calculate_atr(df)  # ATR for dynamic stops
        
        curr = df.iloc[-1]
        price = curr['close']
        rsi_val = curr['rsi']
        adx_val = curr['adx']
        atr_val = curr['atr']  # Current ATR value
        
        # 🛡️ DEFENSIVE: Skip if any indicator is NaN or inf
        if pd.isna(rsi_val) or pd.isna(adx_val) or pd.isna(atr_val):
            logger.warning(f"⚠️ SKIP {coin}: NaN indicator değeri (RSI={rsi_val}, ADX={adx_val}, ATR={atr_val})")
            continue
        if np.isinf(rsi_val) or np.isinf(adx_val) or np.isinf(atr_val):
            logger.warning(f"⚠️ SKIP {coin}: Inf indicator değeri")
            continue
        
        trend_guclu = adx_val > 25
        
        # ========== HTF (4H) TREND TEYİDİ ==========
        htf_bullish = False
        htf_bearish = False
        
        if coin in htf_data and htf_data[coin]:
            df_4h = pd.DataFrame(htf_data[coin], columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df_4h['sma50'] = calculate_sma(df_4h['close'], 50)
            df_4h['sma20'] = calculate_sma(df_4h['close'], 20)
            
            curr_4h = df_4h.iloc[-1]
            htf_price = curr_4h['close']
            htf_sma50 = curr_4h['sma50']
            htf_sma20 = curr_4h['sma20']
            
            # HTF Bullish: Fiyat > SMA50 VE SMA20 > SMA50 (uptrend)
            htf_bullish = (htf_price > htf_sma50) and (htf_sma20 > htf_sma50)
            # HTF Bearish: Fiyat < SMA50 VE SMA20 < SMA50 (downtrend)
            htf_bearish = (htf_price < htf_sma50) and (htf_sma20 < htf_sma50)
        
        sinyal = None
        setup = ""
        
        # 🚫 ANTI-SPAM: Skip if there's already an open position for this coin
        if pozisyon_acik_mi(coin):
            continue  # Wait until current position closes
        
        # 🕐 V5.9 COOLDOWN: Son sinyalden bu yana COIN_COOLDOWN_SAAT saat geçmeli
        if coin in SON_SINYAL_ZAMANI:
            gecen_sure = (datetime.now() - SON_SINYAL_ZAMANI[coin]).total_seconds() / 3600
            if gecen_sure < COIN_COOLDOWN_SAAT:
                continue  # Bu coin için henüz yeterli süre geçmedi
        
        # 📊 V5.9 GÜNLÜK LİMİT: Maksimum GUNLUK_SINYAL_LIMIT sinyal/gün
        bugun_str = datetime.now().strftime("%Y-%m-%d")
        bugunun_sinyal_sayisi = len([s for s in BUGUNUN_SINYALLERI if s[0] == bugun_str])
        if bugunun_sinyal_sayisi >= GUNLUK_SINYAL_LIMIT:
            continue  # Günlük limite ulaşıldı
        
        # ========== 📊 PUANLIK SKORLAMA SİSTEMİ V5.9 (100 ÜZERİNDEN) ==========
        # Ağırlıklar (Toplam: 100 puan):
        # - BTC Skoru: 20 puan (piyasa yönü - en önemli)
        # - � REVERSAL: 18 puan (ani yön değişimi)
        # - 4H HTF Trend: 15 puan (yüksek zaman dilimi teyidi)
        # - SMA200 Trend: 12 puan (ana fiyat trendi)
        # - 💵 USDT Akışı: 10 puan (para giriş/çıkışı)
        # - RSI Seviye: 10 puan (momentum)
        # - Hacim: 8 puan (coin bazlı volume)
        # - ADX: 7 puan (trend gücü)
        # TOPLAM: 100 puan max, 60+ = Sinyal
        
        long_score = 0
        short_score = 0
        long_breakdown = []
        short_breakdown = []
        
        # 🔄 REVERSAL SKORU HESAPLA (max 18 puan)
        rev_long, rev_short, rev_details = calculate_reversal_score(df)
        
        # 💵 USDT AKIŞ SKORU (max 10 puan)
        if usdt_score >= 0.7:
            long_score += 10
            long_breakdown.append("USDT:10")
        elif usdt_score >= 0.4:
            long_score += 7
            long_breakdown.append("USDT:7")
        elif usdt_score >= 0.1:
            long_score += 3
            long_breakdown.append("USDT:3")
        
        if usdt_score <= -0.7:
            short_score += 10
            short_breakdown.append("USDT:10")
        elif usdt_score <= -0.4:
            short_score += 7
            short_breakdown.append("USDT:7")
        elif usdt_score <= -0.1:
            short_score += 3
            short_breakdown.append("USDT:3")
        
        # 1️⃣ BTC SKORU (max 20 puan) - V6.1: Nötr bölge eklendi
        if btc_score >= 1.5:
            long_score += 20
            long_breakdown.append("BTC:20")
        elif btc_score >= 1.0:
            long_score += 15
            long_breakdown.append("BTC:15")
        elif btc_score >= 0.5:
            long_score += 10
            long_breakdown.append("BTC:10")
        elif btc_score >= 0.3:  # 🆕 Nötr-pozitif bölge
            long_score += 5
            long_breakdown.append("BTC:5")
            
        if btc_score <= -1.5:
            short_score += 20
            short_breakdown.append("BTC:20")
        elif btc_score <= -1.0:
            short_score += 15
            short_breakdown.append("BTC:15")
        elif btc_score <= -0.5:
            short_score += 10
            short_breakdown.append("BTC:10")
        elif btc_score <= -0.3:  # 🆕 Nötr-negatif bölge
            short_score += 5
            short_breakdown.append("BTC:5")
        
        # 2️⃣ 4H HTF TREND (max 15 puan)
        if htf_bullish:
            long_score += 15
            long_breakdown.append("HTF:15")
        if htf_bearish:
            short_score += 15
            short_breakdown.append("HTF:15")
        
        # 3️⃣ SMA200 TREND (max 12 puan) - V6.1: %1.5 mesafe koşulu eklendi
        sma200_val = curr['sma200']
        sma200_distance_pct = abs((price - sma200_val) / sma200_val) * 100 if sma200_val > 0 else 0
        
        # Sadece %1.5+ mesafede puan ver (whipsaw önleme)
        if price > sma200_val and sma200_distance_pct >= 1.5:
            long_score += 12
            long_breakdown.append("SMA200:12")
        elif price > sma200_val and sma200_distance_pct >= 0.5:
            long_score += 6  # Yakın bölgede yarım puan
            long_breakdown.append("SMA200:6")
            
        if price < sma200_val and sma200_distance_pct >= 1.5:
            short_score += 12
            short_breakdown.append("SMA200:12")
        elif price < sma200_val and sma200_distance_pct >= 0.5:
            short_score += 6  # Yakın bölgede yarım puan
            short_breakdown.append("SMA200:6")
        
        # 4️⃣ ADX GÜÇ (max 7 puan)
        if adx_val > 30:
            long_score += 7
            short_score += 7
            long_breakdown.append("ADX:7")
            short_breakdown.append("ADX:7")
        elif adx_val > 25:
            long_score += 5
            short_score += 5
            long_breakdown.append("ADX:5")
            short_breakdown.append("ADX:5")
        elif adx_val > 20:
            long_score += 3
            short_score += 3
            long_breakdown.append("ADX:3")
            short_breakdown.append("ADX:3")
        
        # 5️⃣ RSI SEVİYE (max 10 puan) - V5.9: GEVŞETİLMİŞ
        # LONG için oversold: 30, 35, 40, 45
        if rsi_val < 30:
            long_score += 10
            long_breakdown.append("RSI:10")
        elif rsi_val < 35:
            long_score += 8
            long_breakdown.append("RSI:8")
        elif rsi_val < 40:
            long_score += 5
            long_breakdown.append("RSI:5")
        elif rsi_val < 45:
            long_score += 3
            long_breakdown.append("RSI:3")
        
        # SHORT için overbought: 70, 65, 60, 55
        if rsi_val > 70:
            short_score += 10
            short_breakdown.append("RSI:10")
        elif rsi_val > 65:
            short_score += 8
            short_breakdown.append("RSI:8")
        elif rsi_val > 60:
            short_score += 5
            short_breakdown.append("RSI:5")
        elif rsi_val > 55:
            short_score += 3
            short_breakdown.append("RSI:3")
        
        # 🆕 V6.1: RSI 4H TEYİDİ (+5 bonus)
        # 1H RSI ile 4H RSI aynı yönü gösteriyorsa bonus puan
        if coin in htf_data and htf_data[coin]:
            try:
                df_4h = pd.DataFrame(htf_data[coin], columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                rsi_4h = calculate_rsi(df_4h['close']).iloc[-1]
                
                # LONG: 1H RSI < 45 VE 4H RSI < 50 = Teyit
                if rsi_val < 45 and rsi_4h < 50:
                    long_score += 5
                    long_breakdown.append("RSI4H:5")
                
                # SHORT: 1H RSI > 55 VE 4H RSI > 50 = Teyit
                if rsi_val > 55 and rsi_4h > 50:
                    short_score += 5
                    short_breakdown.append("RSI4H:5")
            except Exception:
                pass  # 4H RSI hesaplanamadı, bonus yok
        
        # 6️⃣ HACİM ANALİZİ (max 8 puan)
        vol_sma20 = df['volume'].rolling(20).mean().iloc[-1]
        curr_vol = df['volume'].iloc[-1]
        vol_ratio = curr_vol / vol_sma20 if vol_sma20 > 0 else 1
        
        if vol_ratio > 1.5:  # %50 üzeri hacim
            long_score += 8
            short_score += 8
            long_breakdown.append("VOL:8")
            short_breakdown.append("VOL:8")
        elif vol_ratio > 1.25:  # %25 üzeri hacim
            long_score += 5
            short_score += 5
            long_breakdown.append("VOL:5")
            short_breakdown.append("VOL:5")
        elif vol_ratio > 1.0:  # Ortalama üzeri
            long_score += 2
            short_score += 2
            long_breakdown.append("VOL:2")
            short_breakdown.append("VOL:2")
        
        # 🆕 V6.1: OBV (On-Balance Volume) TRENDİ (+3 bonus)
        # Hacim yönünün fiyat yönüyle uyumunu kontrol et
        try:
            obv = ((df['close'].diff() > 0).astype(int) * 2 - 1) * df['volume']
            obv_cumsum = obv.cumsum()
            obv_sma10 = obv_cumsum.rolling(10).mean()
            
            # OBV son 10 mumda yükseliyor mu?
            obv_rising = obv_cumsum.iloc[-1] > obv_sma10.iloc[-1]
            obv_falling = obv_cumsum.iloc[-1] < obv_sma10.iloc[-1]
            
            if obv_rising:
                long_score += 3
                long_breakdown.append("OBV:3")
            if obv_falling:
                short_score += 3
                short_breakdown.append("OBV:3")
        except Exception:
            pass  # OBV hesaplanamadı
        
        # 7️⃣ ANİ YÖN DEĞİŞİMİ - REVERSAL (max 18 puan)
        # Reversal skorunu 30'dan 18'e normalize et (18/30 = 0.6 oranı)
        # V6.1: ADX < 20 ise reversal puanını yarıya indir (choppy piyasa koruması)
        reversal_multiplier = 0.3 if adx_val < 20 else 0.6  # 🆕 ADX kontrolü
        
        if rev_long > 0:
            normalized_rev_long = int(rev_long * reversal_multiplier)
            long_score += normalized_rev_long
            long_breakdown.extend([f"{d.split(':')[0]}:{int(int(d.split(':')[1])*reversal_multiplier)}" for d in rev_details])
        if rev_short > 0:
            normalized_rev_short = int(rev_short * reversal_multiplier)
            short_score += normalized_rev_short
            short_breakdown.extend([f"{d.split(':')[0]}:{int(int(d.split(':')[1])*reversal_multiplier)}" for d in rev_details])
        
        # Debug log for reversal detection
        if rev_long > 0 or rev_short > 0:
            adx_note = " [ADX<20: YARIM]" if adx_val < 20 else ""
            logger.debug(f"🔄 REVERSAL TESPİT: {coin} | LONG+{int(rev_long*reversal_multiplier)} | SHORT+{int(rev_short*reversal_multiplier)} | {rev_details}{adx_note}")
        
        # 🔥 8️⃣ VOLATILITY SQUEEZE BONUS (max 15 puan)
        # BB daralması + Hacim artışı = Büyük hareket öncüsü
        is_squeeze, squeeze_bonus, squeeze_details = detect_volatility_squeeze(df)
        if is_squeeze and squeeze_bonus > 0:
            long_score += squeeze_bonus
            short_score += squeeze_bonus
            long_breakdown.append(f"SQUEEZE:{squeeze_bonus}")
            short_breakdown.append(f"SQUEEZE:{squeeze_bonus}")
            logger.info(f"🔥 SQUEEZE TESPİT: {coin} | BB%={squeeze_details.get('bb_percentile', 0):.0f} | Vol={squeeze_details.get('vol_ratio', 1):.1f}x | Bonus: +{squeeze_bonus}")
        
        # ========== SİNYAL KARARI (V6.1: DİNAMİK EŞİK) ==========
        # Maksimum teorik puan hesapla (tüm yön bağımsız + en yüksek yön bağımlı puanlar)
        # BTC:20 + Reversal:18 + HTF:15 + Squeeze:15 + SMA200:12 + USDT:10 + RSI:10 + RSI4H:5 + VOL:8 + OBV:3 + ADX:7 = 123
        MAX_TEORIK_PUAN = 123
        ESIK_ORAN = 0.60  # %60 eşik
        
        ESIK = int(MAX_TEORIK_PUAN * ESIK_ORAN)  # 123 * 0.60 = 74
        YAKIN_ESIK = int(MAX_TEORIK_PUAN * 0.40)  # 123 * 0.40 = 49
        
        # 📊 SKORLARI LOGLA (Eşiğe yakın olanları göster)
        max_score = max(long_score, short_score)
        best_direction = "LONG" if long_score >= short_score else "SHORT"
        best_breakdown = long_breakdown if long_score >= short_score else short_breakdown
        
        if max_score >= ESIK:
            # Sinyal üretilecek - detaylı log
            sinyal_ikon = "🟢" if best_direction == "LONG" else "🔴"
            logger.info(f"{sinyal_ikon} SİNYAL! {coin}: {best_direction} {max_score}/100 ({'+'.join(best_breakdown)})")
        elif max_score >= YAKIN_ESIK:
            # Eşiğe yakın - uyarı log
            eksik = ESIK - max_score
            logger.debug(f"⏳ YAKIN: {coin} {best_direction} {max_score}/100 (Eksik: {eksik}p) [{'+'.join(best_breakdown)}]")
        
        if long_score >= ESIK and long_score > short_score:
            sinyal = "LONG"
            setup = f"Score: {long_score}/100 ({'+'.join(long_breakdown)})"
        elif short_score >= ESIK and short_score > long_score:
            sinyal = "SHORT"
            setup = f"Score: {short_score}/100 ({'+'.join(short_breakdown)})"
        
        if sinyal:
            # ========== ATR-BASED TP/SL CALCULATION ==========
            # V5.9: ATR yüzdesi %0.80'den düşükse sinyal verme (düşük volatilite)
            atr_pct = (atr_val / price) * 100
            if atr_pct < 0.80:
                logger.info(f"⏸️ ATR DÜŞÜK: {coin} ATR={atr_pct:.2f}% < 0.80% - Sinyal iptal")
                continue  # Volatilite yetersiz, sinyal verme
            
            # 🆕 V6.1: Trend-Uyumlu Dinamik SL Çarpanı
            sl_multiplier, trend_strength = calculate_trend_aware_sl_multiplier(df, sinyal)
            
            # ATR Multipliers: SL=dinamik, TP1=2.5x, TP2=4.5x, TP3=7x
            atr_sl = atr_val * sl_multiplier  # Dinamik SL çarpanı
            atr_tp1 = atr_val * 2.5   # TP1: 2.5x ATR
            atr_tp2 = atr_val * 4.5   # TP2: 4.5x ATR
            atr_tp3 = atr_val * 7.0   # TP3: 7x ATR
            
            logger.debug(f"📊 {coin} SL: {sl_multiplier}x ATR (Trend: {trend_strength})")
            
            if sinyal == "LONG":
                tp1_price = price + atr_tp1
                tp2_price = price + atr_tp2
                tp3_price = price + atr_tp3
                sl_price = price - atr_sl
            else:  # SHORT
                tp1_price = price - atr_tp1
                tp2_price = price - atr_tp2
                tp3_price = price - atr_tp3
                sl_price = price + atr_sl
            
            # Calculate percentages for display
            tp1_pct = abs(tp1_price - price) / price * 100
            tp2_pct = abs(tp2_price - price) / price * 100
            tp3_pct = abs(tp3_price - price) / price * 100
            sl_pct = abs(sl_price - price) / price * 100
            atr_pct = (atr_val / price) * 100  # ATR as % of price
            
            p_fmt = ".8f" if price < 0.01 else ".4f"
            
            # Save with all TP levels
            islem_kaydet(coin, sinyal, price, tp1_price, tp2_price, tp3_price, sl_price)
            SON_SINYAL_ZAMANI[coin] = datetime.now()
            
            # V5.9: Günlük sinyal listesine ekle
            BUGUNUN_SINYALLERI.append((datetime.now().strftime("%Y-%m-%d"), coin, sinyal))
            
            logger.info(f"🎯 {sinyal}: {coin} (Score: {long_score if sinyal == 'LONG' else short_score}/100, ATR: {atr_pct:.2f}%)")
            
            resim = await grafik_olustur_async(coin, df.tail(100), tp1_price, sl_price, sinyal)
            ikon = "🟢" if sinyal == "LONG" else "🔴"
            
            # Skor bilgisi
            skor_deger = long_score if sinyal == "LONG" else short_score
            skor_breakdown = '+'.join(long_breakdown) if sinyal == "LONG" else '+'.join(short_breakdown)
            
            # Reversal bilgisi
            rev_info = "🔄 Reversal: " + "+".join(rev_details) if rev_details else ""
            
            # Yüzdelik değişimleri hesapla (SHORT için mutlak değer - kazanç olarak göster)
            tp1_pct = abs((tp1_price - price) / price) * 100
            tp2_pct = abs((tp2_price - price) / price) * 100
            tp3_pct = abs((tp3_price - price) / price) * 100
            sl_pct = abs((sl_price - price) / price) * 100
            
            # TP'ler + (kazanç), SL - (risk) olarak göster
            tp_sign = "+"
            sl_sign = "-"
            
            mesaj = f"""
{ikon} <b>TITANIUM SİNYAL ({sinyal})</b> #V6.1

🪙 <b>Coin:</b> #{coin}
🌍 <b>BTC:</b> {btc_score} {btc_ikon}
💵 <b>USDT Akış:</b> {usdt_flow_m:+.1f}M$ {usdt_ikon}
⏰ <b>Trend:</b> {'✅ Bullish' if htf_bullish else '🔴 Bearish' if htf_bearish else '⚪ Nötr'}
{rev_info}

💰 <b>Giriş:</b> ${price:{p_fmt}}

🎯 <b>TP1 (33%):</b> ${tp1_price:{p_fmt}} ({tp_sign}{tp1_pct:.2f}%)
🎯 <b>TP2 (33%):</b> ${tp2_price:{p_fmt}} ({tp_sign}{tp2_pct:.2f}%)
🎯 <b>TP3 (34%):</b> ${tp3_price:{p_fmt}} ({tp_sign}{tp3_pct:.2f}%)
🛑 <b>STOP (SL):</b> ${sl_price:{p_fmt}} ({sl_sign}{sl_pct:.2f}%)

📌 <i>%{skor_deger} Güven Skoru</i>
"""
            try:
                if resim:
                    await bot.send_photo(chat_id=KANAL_ID, photo=resim, caption=mesaj, parse_mode=ParseMode.HTML)
                else:
                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
            except Exception as e:
                logger.error(f"Telegram Hatasi: {e}")
        
        # V5.9: Range Trading KALDIRILDI - Sadece Trend + Rapid stratejileri aktif

# ==========================================
# ⚡ BÖLÜM 5.5: RAPID REVERSAL TARAMA
# ==========================================
async def rapid_strateji_tarama(exchange):
    """
    Rapid Reversal stratejisi - Ani piyasa değişimlerini tara
    Mevcut trend stratejisinden BAĞIMSIZ çalışır
    """
    logger.info(f"⚡ ({datetime.now().strftime('%H:%M')}) RAPID REVERSAL TARAMA...")
    
    # ESKİ: RAPID_ESIK = 50  # Çok fazla sinyal üretiyordu
    RAPID_ESIK = 65  # YÜKSEK KALİTE: Güçlü reversal sinyalleri için artırıldı
    
    # Coin verilerini çek
    async def fetch_candle(s):
        try:
            ohlcv = await exchange.fetch_ohlcv(f"{s}/USDT", '1h', limit=50)
            return s, ohlcv
        except: 
            return s, None

    tasks = [fetch_candle(c) for c in COIN_LIST]
    results = await asyncio.gather(*tasks)
    
    for coin, bars in results:
        if not bars: 
            continue
        
        # 🚫 ANTI-SPAM: Açık pozisyon varsa atla
        if pozisyon_acik_mi(coin):
            continue
        
        # 🕐 V5.9 COOLDOWN: Rapid için de aynı cooldown uygula
        if coin in SON_SINYAL_ZAMANI:
            gecen_sure = (datetime.now() - SON_SINYAL_ZAMANI[coin]).total_seconds() / 3600
            if gecen_sure < COIN_COOLDOWN_SAAT:
                continue
        
        # 📊 V5.9 GÜNLÜK LİMİT: Rapid sinyalleri de limiti kullanır
        bugun_str = datetime.now().strftime("%Y-%m-%d")
        bugunun_sinyal_sayisi = len([s for s in BUGUNUN_SINYALLERI if s[0] == bugun_str])
        if bugunun_sinyal_sayisi >= GUNLUK_SINYAL_LIMIT:
            continue
        
        df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df.set_index('date', inplace=True)
        
        # İndikatörleri hesapla (grafik için gerekli)
        df['sma50'] = calculate_sma(df['close'], 50)
        df['sma200'] = calculate_sma(df['close'], 50)  # 50 kullan (50 mum var sadece)
        df['rsi'] = calculate_rsi(df['close'])
        df['atr'] = calculate_atr(df)
        atr_val = df['atr'].iloc[-1]
        price = df['close'].iloc[-1]
        
        # Rapid skor hesapla
        rapid_long, rapid_short, rapid_details, tetikleyiciler = calculate_rapid_score(df)
        
        sinyal = None
        
        # RAPID LONG
        if rapid_long >= RAPID_ESIK and rapid_long > rapid_short:
            sinyal = "LONG"
            skor = rapid_long
        # RAPID SHORT
        elif rapid_short >= RAPID_ESIK and rapid_short > rapid_long:
            sinyal = "SHORT"
            skor = rapid_short
        
        if sinyal:
            # ========== RAPID TP/SL (SIKІ RİSK YÖNETİMİ) ==========
            atr_sl = atr_val * 1.5   # Stop Loss: 1.5x ATR (daha sıkı)
            atr_tp1 = atr_val * 2.0  # TP1: 2x ATR
            atr_tp2 = atr_val * 3.0  # TP2: 3x ATR
            
            if sinyal == "LONG":
                tp1_price = price + atr_tp1
                tp2_price = price + atr_tp2
                sl_price = price - atr_sl
            else:  # SHORT
                tp1_price = price - atr_tp1
                tp2_price = price - atr_tp2
                sl_price = price + atr_sl
            
            # Yüzdeler
            tp1_pct = abs(tp1_price - price) / price * 100
            tp2_pct = abs(tp2_price - price) / price * 100
            sl_pct = abs(sl_price - price) / price * 100
            
            p_fmt = ".8f" if price < 0.01 else ".4f"
            
            # Kaydet (TP3'ü TP2 ile aynı tut - sadece 2 TP var)
            islem_kaydet(coin, sinyal, price, tp1_price, tp2_price, tp2_price, sl_price)
            SON_SINYAL_ZAMANI[coin] = datetime.now()
            
            # V5.9: Günlük sinyal listesine ekle
            BUGUNUN_SINYALLERI.append((datetime.now().strftime("%Y-%m-%d"), coin, f"RAPID-{sinyal}"))
            
            logger.info(f"⚡ RAPID {sinyal}: {coin} (Score: {skor}/100, Tetik: {', '.join(tetikleyiciler)})")
            
            # 🎨 GRAFİK OLUŞTUR (YENİ!)
            resim = await grafik_olustur_async(coin, df.tail(50), tp1_price, sl_price, f"RAPID {sinyal}")
            
            ikon = "🟢" if sinyal == "LONG" else "🔴"
            detail_str = '+'.join(rapid_details)
            tetik_str = ' + '.join(tetikleyiciler) if tetikleyiciler else "Multi-trigger"
            
            # Yüzdelik değişimleri hesapla (mutlak değer - kazanç olarak göster)
            tp1_pct = abs((tp1_price - price) / price) * 100
            tp2_pct = abs((tp2_price - price) / price) * 100
            sl_pct = abs((sl_price - price) / price) * 100
            
            mesaj = f"""
⚡ <b>RAPID REVERSAL SİNYAL ({sinyal})</b> #V6.1-RAPID

🪙 <b>Coin:</b> #{coin}
🔥 <b>Rapid Skor:</b> {skor}/100 ({detail_str})
📊 <b>Tetikleyici:</b> {tetik_str}

💰 <b>Giriş:</b> ${price:{p_fmt}}

🎯 <b>TP1 (50%):</b> ${tp1_price:{p_fmt}} (+{tp1_pct:.2f}%)
🎯 <b>TP2 (50%):</b> ${tp2_price:{p_fmt}} (+{tp2_pct:.2f}%)
🛑 <b>SL:</b> ${sl_price:{p_fmt}} (-{sl_pct:.2f}%)

⚠️ <i>RAPID sinyal - Hızlı hareket bekleniyor!</i>
"""
            try:
                if resim:
                    await bot.send_photo(chat_id=KANAL_ID, photo=resim, caption=mesaj, parse_mode=ParseMode.HTML)
                else:
                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
            except Exception as e:
                logger.error(f"Telegram Hatasi (Rapid): {e}")

# ==========================================
# 🛡️ BÖLÜM 6: POZİSYON TAKİBİ (MULTI-TP)
# ==========================================
async def pozisyonlari_yokla(exchange):
    """Track open positions with multi-level TP support"""
    with sqlite3.connect("titanium_live.db") as conn:
        c = conn.cursor()
        # Updated query for multi-TP columns
        c.execute("""SELECT id, coin, yon, giris_fiyat, tp1, tp2, tp3, sl, tp1_hit, tp2_hit 
                     FROM islemler WHERE durum='ACIK'""")
        acik_islemler = c.fetchall()
        
    if not acik_islemler: return

    for islem in acik_islemler:
        id, coin, yon, giris, tp1, tp2, tp3, sl, tp1_hit, tp2_hit = islem
        try:
            ticker = await exchange.fetch_ticker(f"{coin}/USDT")
            fiyat = ticker['last']
            p_fmt = ".8f" if fiyat < 0.01 else ".4f"
            
            # DEBUG LOG
            logger.debug(f"🔍 {coin}: Fiyat=${fiyat:{p_fmt}} | SL=${sl:{p_fmt}} | TP1=${tp1:{p_fmt}} (hit={tp1_hit}) | TP2=${tp2:{p_fmt}} (hit={tp2_hit}) | TP3=${tp3:{p_fmt}}")
            
            # --- TP1 CHECK (10% TOLERANS + AKILLI TRAILING) ---
            if not tp1_hit:
                # %10 tolerans: TP1'in %90'ına ulaşınca da sayılır
                tp1_tolerance = abs(tp1 - giris) * 0.90
                tp1_target = giris + tp1_tolerance if yon == "LONG" else giris - tp1_tolerance
                tp1_reached = (fiyat >= tp1_target) if yon == "LONG" else (fiyat <= tp1_target)
                if tp1_reached:
                    # 🧠 AKILLI TRAILING: Trend gücünü analiz et
                    trend_gucu, sl_multiplier, analiz = await trend_gucunu_analiz_et(exchange, coin, yon, fiyat)
                    atr_now = analiz.get('atr', abs(tp1 - giris) * 0.5)  # Fallback ATR
                    trend_puan = analiz.get('puan', 50)
                    
                    # SL'i trend gücüne göre belirle
                    if trend_gucu == "GUCLU" or trend_gucu == "ORTA":
                        # ATR bazlı geniş SL - Gideni tutma!
                        if yon == "LONG":
                            new_sl = fiyat - (atr_now * sl_multiplier)
                        else:  # SHORT
                            new_sl = fiyat + (atr_now * sl_multiplier)
                    else:
                        # ZAYIF trend - Sıkı SL, kârı koru!
                        buffer = 0.005  # %0.5 buffer
                        if yon == "LONG":
                            new_sl = tp1 * (1 - buffer)
                        else:  # SHORT
                            new_sl = tp1 * (1 + buffer)
                    
                    with sqlite3.connect("titanium_live.db") as conn:
                        conn.execute("UPDATE islemler SET tp1_hit=1, sl=? WHERE id=?", (new_sl, id))
                    
                    pnl1 = ((tp1 - giris) / giris * 100) if yon == "LONG" else ((giris - tp1) / giris * 100)
                    
                    # Trend göstergeleri
                    trend_ikon = "🟢" if trend_gucu == "GUCLU" else ("🟡" if trend_gucu == "ORTA" else "🔴")
                    sl_tipi = f"ATR×{sl_multiplier}" if sl_multiplier > 0 else "TP1±0.5%"
                    gideni_tutma = "🚀 Gideni Tutmuyoruz!" if trend_gucu == "GUCLU" else ("📊 Dengeli" if trend_gucu == "ORTA" else "🔒 Kârı Koruyoruz!")
                    
                    mesaj = f"""
🎯 <b>TP1 ULAŞILDI!</b> ✅

🪙 <b>#{coin}</b> ({yon})
💰 <b>Giriş:</b> ${giris:{p_fmt}}
🎯 <b>TP1:</b> ${tp1:{p_fmt}}
📈 <b>Kâr:</b> +{pnl1:.2f}%



{gideni_tutma}
🔒 <b>YENİ SL:</b> ${new_sl:{p_fmt}} ({sl_tipi})

"""
                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                    continue  # Check other TPs next cycle
            
            # --- MOMENTUM KONTROLÜ (TP1 SONRASI) ---
            momentum_strong = False
            if tp1_hit:
                try:
                    ohlcv_mom = await exchange.fetch_ohlcv(f"{coin}/USDT", '1h', limit=50)
                    df_mom = pd.DataFrame(ohlcv_mom, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                    rsi_now = calculate_rsi(df_mom['close']).iloc[-1]
                    sma20 = df_mom['close'].rolling(20).mean().iloc[-1]
                    sma50 = df_mom['close'].rolling(50).mean().iloc[-1]
                    
                    if yon == "LONG":
                        # LONG için: RSI > 45 VE fiyat > SMA20 = momentum güçlü
                        momentum_strong = (rsi_now > 45) and (fiyat > sma20)
                    else:  # SHORT
                        # SHORT için: RSI < 55 VE fiyat < SMA20 = momentum güçlü
                        momentum_strong = (rsi_now < 55) and (fiyat < sma20)
                    
                    logger.debug(f"📊 MOMENTUM {coin}: RSI={rsi_now:.1f}, SMA20={'✅' if (fiyat > sma20 if yon == 'LONG' else fiyat < sma20) else '❌'} -> {'GÜÇLÜ' if momentum_strong else 'ZAYIF'}")
                except Exception as mom_err:
                    logger.warning(f"⚠️ Momentum Hatası ({coin}): {mom_err}")
                    momentum_strong = True  # Hata olursa default olarak devam et
            
            # --- TP2 CHECK (10% TOLERANS + AKILLI TRAILING) ---
            if tp1_hit and not tp2_hit:
                # %10 tolerans: TP2'nin %90'ına ulaşınca da sayılır
                tp2_tolerance = abs(tp2 - giris) * 0.90
                tp2_target = giris + tp2_tolerance if yon == "LONG" else giris - tp2_tolerance
                tp2_reached = (fiyat >= tp2_target) if yon == "LONG" else (fiyat <= tp2_target)
                if tp2_reached:
                    # 🧠 AKILLI TRAILING: Trend gücünü analiz et
                    trend_gucu, sl_multiplier, analiz = await trend_gucunu_analiz_et(exchange, coin, yon, fiyat)
                    atr_now = analiz.get('atr', abs(tp2 - giris) * 0.3)  # Fallback ATR
                    trend_puan = analiz.get('puan', 50)
                    
                    # TP2'de daha agresif trailing - çarpanı artır
                    sl_multiplier_tp2 = sl_multiplier * 0.8  # TP2'de biraz daha sıkı
                    
                    # SL'i trend gücüne göre belirle
                    if trend_gucu == "GUCLU" or trend_gucu == "ORTA":
                        # ATR bazlı SL
                        if yon == "LONG":
                            new_sl = fiyat - (atr_now * sl_multiplier_tp2)
                        else:  # SHORT
                            new_sl = fiyat + (atr_now * sl_multiplier_tp2)
                    else:
                        # ZAYIF trend - TP2 + buffer
                        buffer = 0.003  # %0.3 buffer (TP2'de daha sıkı)
                        if yon == "LONG":
                            new_sl = tp2 * (1 - buffer)
                        else:  # SHORT
                            new_sl = tp2 * (1 + buffer)
                    
                    with sqlite3.connect("titanium_live.db") as conn:
                        conn.execute("UPDATE islemler SET tp2_hit=1, sl=? WHERE id=?", (new_sl, id))
                    
                    pnl2 = ((tp2 - giris) / giris * 100) if yon == "LONG" else ((giris - tp2) / giris * 100)
                    
                    # Trend göstergeleri
                    trend_ikon = "🟢" if trend_gucu == "GUCLU" else ("🟡" if trend_gucu == "ORTA" else "🔴")
                    devam_msg = "🚀 TREND GÜÇLÜ - TP3'e Devam!" if trend_gucu == "GUCLU" else ("� TP3'e Bırakıldı" if trend_gucu == "ORTA" else "🔒 %66 Kilitlendi")
                    
                    mesaj = f"""
🎯🎯 <b>TP2 ULAŞILDI!</b> ✅✅

🪙 <b>#{coin}</b> ({yon})
💰 <b>Giriş:</b> ${giris:{p_fmt}}
🎯 <b>TP2:</b> ${tp2:{p_fmt}}
📈 <b>Kâr:</b> +{pnl2:.2f}%

{trend_ikon} <b>Trend:</b> {trend_gucu} ({trend_puan}/100)
🔒 <b>YENİ SL:</b> ${new_sl:{p_fmt}}
{devam_msg}
"""
                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                    continue
            
            # --- TP2 SONRASI TRAILING STOP (YENİ!) ---
            if tp1_hit and tp2_hit:
                # ATR hesapla - trailing için
                try:
                    ohlcv = await exchange.fetch_ohlcv(f"{coin}/USDT", '1h', limit=20)
                    df_trail = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                    atr_trail = calculate_atr(df_trail, 14).iloc[-1]
                    trailing_distance = atr_trail * 1.5  # 1.5x ATR trailing mesafesi
                    
                    if yon == "LONG":
                        # Yeni potansiyel SL = Mevcut fiyat - trailing mesafe
                        new_trail_sl = fiyat - trailing_distance
                        # Sadece yukarı doğru güncelle (sl'den yüksekse)
                        if new_trail_sl > sl:
                            with sqlite3.connect("titanium_live.db") as conn:
                                conn.execute("UPDATE islemler SET sl=? WHERE id=?", (new_trail_sl, id))
                            logger.info(f"📈 TRAILING: {coin} SL güncellendi: ${sl:{p_fmt}} -> ${new_trail_sl:{p_fmt}}")
                            sl = new_trail_sl  # Güncel SL'i kullan
                    elif yon == "SHORT":  # FIX: Properly handle SHORT (was incorrectly nested)
                        new_trail_sl = fiyat + trailing_distance
                        if new_trail_sl < sl:
                            with sqlite3.connect("titanium_live.db") as conn:
                                conn.execute("UPDATE islemler SET sl=? WHERE id=?", (new_trail_sl, id))
                            logger.info(f"📉 TRAILING: {coin} SL güncellendi: ${sl:{p_fmt}} -> ${new_trail_sl:{p_fmt}}")
                            sl = new_trail_sl
                except Exception as trail_err:
                    logger.warning(f"⚠️ Trailing Hesaplama Hatası ({coin}): {trail_err}")
            
            # --- TP3 CHECK (FULL EXIT - 10% TOLERANS) ---
            if tp1_hit and tp2_hit:
                # %10 tolerans: TP3'ün %90'ına ulaşınca da sayılır
                tp3_tolerance = abs(tp3 - giris) * 0.90
                tp3_target = giris + tp3_tolerance if yon == "LONG" else giris - tp3_tolerance
                tp3_reached = (fiyat >= tp3_target) if yon == "LONG" else (fiyat <= tp3_target)
                if tp3_reached:
                    pnl3 = ((tp3 - giris) / giris * 100) if yon == "LONG" else ((giris - tp3) / giris * 100)
                    # Calculate weighted average PnL (33% + 33% + 34%)
                    pnl1 = ((tp1 - giris) / giris * 100) if yon == "LONG" else ((giris - tp1) / giris * 100)
                    pnl2 = ((tp2 - giris) / giris * 100) if yon == "LONG" else ((giris - tp2) / giris * 100)
                    total_pnl = (pnl1 * 0.33) + (pnl2 * 0.33) + (pnl3 * 0.34)
                    
                    with sqlite3.connect("titanium_live.db") as conn:
                        conn.execute("""UPDATE islemler SET durum='KAZANDI', pnl_yuzde=?, 
                                      kapanis_zamani=? WHERE id=?""", 
                                  (total_pnl, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), id))
                    
                    mesaj = f"""
🏆🏆🏆 <b>TÜM HEDEFLER TAMAMLANDI!</b> 🎉

🪙 <b>#{coin}</b> ({yon})
💰 <b>Giriş:</b> ${giris:{p_fmt}}
🎯 <b>TP1:</b> ${tp1:{p_fmt}} ✅
🎯 <b>TP2:</b> ${tp2:{p_fmt}} ✅
🎯 <b>TP3:</b> ${tp3:{p_fmt}} ✅
📈 <b>Toplam Kâr:</b> +{total_pnl:.2f}%

🤖 <i>Titanium V5.4 - Trailing TP Success!</i>
"""
                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                    continue
            
            # --- STOP LOSS CHECK ---
            sl_hit = (fiyat <= sl) if yon == "LONG" else (fiyat >= sl)
            if sl_hit:
                # Calculate actual PnL based on which TPs were hit
                partial_profit = 0.0
                if tp1_hit:
                    pnl1 = ((tp1 - giris) / giris * 100) if yon == "LONG" else ((giris - tp1) / giris * 100)
                    partial_profit += pnl1 * 0.33
                if tp2_hit:
                    pnl2 = ((tp2 - giris) / giris * 100) if yon == "LONG" else ((giris - tp2) / giris * 100)
                    partial_profit += pnl2 * 0.33
                
                # Remaining position hit SL
                remaining_pct = 1.0 - (0.33 if tp1_hit else 0) - (0.33 if tp2_hit else 0)
                sl_pnl = ((sl - giris) / giris * 100) if yon == "LONG" else ((giris - sl) / giris * 100)
                total_pnl = partial_profit + (sl_pnl * remaining_pct)
                
                durum = "PARTIAL" if (tp1_hit or tp2_hit) else "KAYBETTI"
                
                with sqlite3.connect("titanium_live.db") as conn:
                    conn.execute("""UPDATE islemler SET durum=?, pnl_yuzde=?, 
                                  kapanis_zamani=? WHERE id=?""", 
                              (durum, total_pnl, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), id))
                
                ikon = "⚠️" if durum == "PARTIAL" else "❌"
                tp_status = f"TP1: {'✅' if tp1_hit else '❌'} | TP2: {'✅' if tp2_hit else '❌'}"
                
                mesaj = f"""
🏁 <b>POZİSYON KAPANDI</b> {ikon}

🪙 <b>#{coin}</b> ({yon})
🏷️ <b>Sonuç:</b> {durum}
{tp_status}

💰 <b>Giriş:</b> ${giris:{p_fmt}}
🚪 <b>SL Çıkış:</b> ${fiyat:{p_fmt}}
📉 <b>Net Kâr/Zarar:</b> {total_pnl:+.2f}%

🤖 <i>Titanium Bot</i>
"""
                await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                
        except Exception as e:
            logger.error(f"Pozisyon Takip Hatası ({coin}): {e}")
            continue

# ==========================================
# 🏁 MAIN LOOP (V6.0 - PRODUCTION HARDENED)
# ==========================================
async def main():
    global SON_RAPOR_TARIHI
    db_ilk_kurulum()
    logger.info("🚀 Titanium PREMIUM Bot V6.0 Aktif! (Production Hardened)")
    
    exchange = ccxt.kucoin(exchange_config)
    
    # 🛡️ INITIALIZE PRODUCTION LAYERS
    risk_manager = RiskManager(initial_equity=1000.0, db_path="titanium_live.db")
    regime_detector = RegimeDetector()
    position_sizer = PositionSizer(account_balance=1000.0)
    slippage_model = SlippageModel()
    
    logger.info("🛡️ Risk Manager: Initialized")
    logger.info("🧠 Regime Detector: Initialized")
    logger.info("⚙️ Position Sizer: Initialized")
    
    try:
        startup_msg = """🚀 <b>TITANIUM BOT V6.0 BAŞLATILDI!</b>

🛡️ <b>YENİ: Production Hardened</b>
• Kill-Switch: ATR Z-Score + BTC Flash
• Drawdown Monitor: 10%/15%/20% Limits
• Regime Detection: TREND/RANGE/NO_TRADE
• Position Sizing: Kelly-Inspired

✅ Sistem: Aktif
🎯 Sinyal Eşiği: 60/100
⚡ Rapid Eşiği: 65/100
✅ Borsa: KuCoin

<i>Survival > Profitability</i>"""
        await bot.send_message(chat_id=KANAL_ID, text=startup_msg, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"❌ Telegram Test Mesajı Hatası: {e}")

    if "ETH" in COIN_LIST:
        COIN_LIST.remove("ETH")

    try:
        while True:
            # İstanbul saati (UTC+3)
            istanbul_tz = timezone(timedelta(hours=3))
            simdi = datetime.now(istanbul_tz)
            bugun_str = simdi.strftime("%Y-%m-%d")
            
            # 🛡️ PRE-LOOP RISK CHECK
            try:
                # Fetch BTC data for risk checks
                btc_ohlcv = await exchange.fetch_ohlcv("BTC/USDT", '1h', limit=60)
                df_btc = pd.DataFrame(btc_ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                atr_btc = calculate_atr(df_btc)
                
                can_trade, size_mult, risk_reason = risk_manager.pre_signal_check(df_btc, atr_btc)
                
                if not can_trade:
                    logger.warning(f"🚨 TRADING HALTED: {risk_reason}")
                    logger.info("💤 Waiting 5 minutes before retry...")
                    await asyncio.sleep(300)  # Wait 5 min and retry
                    continue
                
                # 🧠 REGIME DETECTION
                btc_4h = await exchange.fetch_ohlcv("BTC/USDT", '4h', limit=60)
                df_btc_4h = pd.DataFrame(btc_4h, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                current_regime, regime_details = regime_detector.detect_regime(df_btc_4h)
                
                regime_icons = {
                    MarketRegime.TREND: "📈 TREND",
                    MarketRegime.RANGE: "↔️ RANGE",
                    MarketRegime.NO_TRADE: "🚫 NO_TRADE",
                    MarketRegime.MIXED: "🔀 MIXED"
                }
                logger.info(f"🧠 REGIME: {regime_icons.get(current_regime, 'UNKNOWN')} | {regime_details.get('reason', '')}")
                
                # Skip signal generation in NO_TRADE regime
                if current_regime == MarketRegime.NO_TRADE:
                    logger.info("⏸️ NO_TRADE regime - skipping signal generation")
                    await asyncio.sleep(300)
                    continue
                    
            except Exception as risk_err:
                logger.warning(f"⚠️ Risk check error: {risk_err}")
                # Continue with trading but log the error
            
            # Gün sonu raporu - İstanbul saati 23:55
            # Gün sonu raporu - İstanbul saati 23:55
            if simdi.hour == 23 and simdi.minute >= 55 and SON_RAPOR_TARIHI != bugun_str:
                logger.info(f"📊 Gün sonu raporu gönderiliyor... (İstanbul: {simdi.strftime('%H:%M')})")
                await gunluk_rapor_gonder(bugun_str)
                SON_RAPOR_TARIHI = bugun_str
                # Hemen kaydet
                periodic_save(last_report_date=SON_RAPOR_TARIHI)
            
            await haberleri_kontrol_et()
            await piyasayi_tarama(exchange)
            await rapid_strateji_tarama(exchange)  # ⚡ RAPID REVERSAL
            await pozisyonlari_yokla(exchange)
            
            # 📊 Periodic status log
            status = risk_manager.get_status_summary()
            logger.info(f"📊 DD: {status['current_drawdown']:.1f}% | Daily: {status['daily_pnl']:.1f}% | KS: {'🔴' if status['kill_switch_active'] else '🟢'}")
            
            # 💾 Periyodik state kaydetme (her döngüde)
            # 💾 Periyodik state kaydetme (her döngüde)
            periodic_save(
                positions=None,  # DB'den çekilir
                signals=BUGUNUN_SINYALLERI,
                cooldowns=SON_SINYAL_ZAMANI,
                last_report_date=SON_RAPOR_TARIHI
            )
            
            logger.debug("💤 Bekleme (1dk)...")
            await asyncio.sleep(60) 
    except KeyboardInterrupt:
        logger.info("🛑 Bot Durduruluyor...")
    finally:
        # 💾 Son state kaydet
        if hasattr(state_manager, 'save'):
            state_manager.save()
        else:
            # Fallback if method different
            state_manager.save_state()
        logger.info("💾 State kaydedildi")
        await exchange.close()

if __name__ == "__main__":
    # 🔄 Crash recovery kontrolü
    # 🔄 State Yükle (Recovery)
    # state_manager.load() is called in get_state_manager(), so state is already loaded if initialized
    if state_manager.son_sinyal_zamani:
        logger.info(f"🔄 RECOVERY: {len(state_manager.son_sinyal_zamani)} coin cooldown yüklendi")
        SON_SINYAL_ZAMANI.update(state_manager.son_sinyal_zamani)
        
    if state_manager.bugunun_sinyalleri:
        logger.info(f"🔄 RECOVERY: {len(state_manager.bugunun_sinyalleri)} günlük sinyal yüklendi")
        # Extend current list avoiding duplicates
        existing_set = set((s[1], s[2]) for s in BUGUNUN_SINYALLERI) # coin, yon
        for s in state_manager.bugunun_sinyalleri:
            if len(s) >= 3 and (s[1], s[2]) not in existing_set:
                BUGUNUN_SINYALLERI.append(s)
        
    if state_manager.son_rapor_tarihi:
        SON_RAPOR_TARIHI = state_manager.son_rapor_tarihi
        logger.info(f"🔄 RECOVERY: Son rapor tarihi yüklendi: {SON_RAPOR_TARIHI}")
    
    # Shutdown handler kaydet

    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

