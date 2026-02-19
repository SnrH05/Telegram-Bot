"""
TITANIUM Bot - Konfigürasyon Modülü
===================================
Tüm sabitler, ayarlar ve konfigürasyonlar burada tanımlanır.
"""

import os
import logging

logger = logging.getLogger(__name__)

# ==========================================
# 🔧 BOT TEMEL AYARLARI
# ==========================================
TOKEN = os.getenv("BOT_TOKEN", "").strip()
# KANAL_ID artık liste olarak tutulacak
KANAL_ID_RAW = os.getenv("KANAL_ID", "0")
KANAL_ID = [int(x.strip()) for x in KANAL_ID_RAW.split(",") if x.strip()]
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

# ==========================================
# 📊 EXCHANGE AYARLARI
# ==========================================
EXCHANGE_CONFIG = {
    'enableRateLimit': True,
    'rateLimit': 50,  # 50ms bekleme - Binance rate limit koruması
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True,
    },
    'timeout': 30000,  # 30 saniye timeout
}

# ==========================================
# 🪙 COIN GRUPLARI VE AYARLARI (V6.2)
# ==========================================
# Grup 1: Trend Setters (Majors) - Trend Takibi — Yüksek market cap, düşük volatilite
# Grup 2: Swing Players (Mid-Caps) - Kanal İçi (Mean Reversion) — Orta market cap, orta volatilite
# Grup 3: Rockets (Meme/High Vol) - Hacim Patlaması — Düşük market cap, yüksek volatilite
# NOT: Tüm coinler KuCoin spot USDT paritesinde aktif olarak doğrulanmıştır.
COIN_GROUPS = {
    "MAJOR": {
        "coins": [
            "BTC", "ETH", "SOL", "BNB", "XRP",
            "ADA", "AVAX", "DOT", "TON", "TRX"
        ],
        "strategy": "TREND_FOLLOWING",
        "params": {
            "ema_trend": 200,
            "ema_pullback": 50,
            "rsi_min": 40,
            "rsi_max": 60,
            "stop_loss_mult": 1.0
        }
    },
    "SWING": {
        "coins": [
            "LINK", "LTC", "ATOM", "NEAR", "UNI",
            "AAVE", "INJ", "SEI", "TIA", "ARB",
            "OP", "STX", "RUNE", "ICP", "HBAR",
            "FIL", "ALGO", "VET", "PENDLE", "ONDO"
        ],
        "strategy": "MEAN_REVERSION",
        "params": {
            "bb_period": 20,
            "bb_std": 2,
            "stoch_rsi_overbought": 80,
            "stoch_rsi_oversold": 20,
            "stop_loss_atr": 1.5
        }
    },
    "MEME": {
        "coins": [
            "DOGE", "SHIB", "PEPE", "WIF", "FLOKI",
            "BONK", "SUI", "FET", "APT", "RENDER",
            "JUP", "WLD", "TAO", "ORDI", "PYTH",
            "IMX", "SAND", "GRT", "BLUR", "POPCAT",
            "MEW", "TURBO", "BOME", "BRETT", "PEOPLE",
            "NEIRO", "PNUT", "GOAT", "GRASS", "EIGEN"
        ],
        "strategy": "VOLATILITY_BREAKOUT",
        "params": {
            "supertrend_period": 10,
            "supertrend_multiplier": 3,
            "volume_ma": 20,
            "volume_spike_mult": 2.5,
            "rsi_period": 7,
            "rsi_min": 60
        }
    }
}

# Geriye dönük uyumluluk için düz liste
COIN_LIST = []
for group in COIN_GROUPS.values():
    COIN_LIST.extend(group["coins"])

# ==========================================
# 📊 SKORLAMA VE LİMİT AYARLARI (V6.2: DİNAMİK EŞİK)
# ==========================================
# Maksimum teorik puan (tüm yön bağımsız + en yüksek yön bağımlı puanlar)
# BTC:20 + Reversal:18 + HTF:15 + Squeeze:15 + SMA200:12 + USDT:10 + RSI:10 + RSI4H:5 + VOL:8 + OBV:3 + ADX:7 = 123
MAX_TEORIK_PUAN = 123
ESIK_ORAN = 0.60  # %60 eşik (Kalite odaklı)

SINYAL_ESIK = int(MAX_TEORIK_PUAN * ESIK_ORAN)        # 123 * 0.60 = 74
YAKIN_ESIK = int(MAX_TEORIK_PUAN * 0.45)               # 123 * 0.45 = 55

MIN_SCORE_THRESHOLD = SINYAL_ESIK  # Signal manager ve diğer kontroller için


RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed"
]

# ==========================================
# 🎯 SİNYAL OPTİMİZASYONU AYARLARI
# ==========================================
COIN_COOLDOWN_SAAT = 4      # Aynı coin için minimum bekleme süresi
GUNLUK_SINYAL_LIMIT = 999   # Günlük limit (pratik olarak sınırsız)

# ==========================================
# ⏱️ ZAMANLAMA AYARLARI
# ==========================================
TARAMA_BEKLEME_SANIYE = 60         # Her tarama arası bekleme
POZISYON_KONTROL_SANIYE = 30       # Pozisyon takip sıklığı
HABER_KONTROL_SANIYE = 300         # Haber kontrolü (5 dk)
RAPOR_SAATI = 23                   # Günlük rapor saati (23:55)
RAPOR_DAKIKA = 55

# ==========================================
# 🛡️ RİSK YÖNETİMİ AYARLARI
# ==========================================
MAX_AYNI_ANDA_ISLEM = 23            # Aynı anda açık olabilecek işlem
VARSAYILAN_SL_CARPANI = 2.0        # Varsayılan ATR çarpanı
MIN_ATR_YUZDE = 0.8                # Minimum ATR% (volatilite kontrolü)

# ==========================================
# 🌐 GLOBAL DEĞİŞKENLER (Runtime state)
# ==========================================
SON_SINYAL_ZAMANI = {}             # {coin: datetime} - Cooldown takibi
SON_RAPOR_TARIHI = None            # Son gönderilen günlük rapor tarihi
BUGUNUN_SINYALLERI = []            # Bugün üretilen sinyallerin listesi


def validate_config():
    """Konfigürasyon doğrulaması"""
    errors = []
    
    if not TOKEN:
        errors.append("BOT_TOKEN eksik")
    if not KANAL_ID:
        errors.append("KANAL_ID eksik")
    if not GEMINI_KEY:
        errors.append("GEMINI_KEY eksik")
    
    if errors:
        logger.error(f"❌ Konfigürasyon hataları: {', '.join(errors)}")
        return False
    
    logger.info("✅ Konfigürasyon doğrulandı")
    return True
