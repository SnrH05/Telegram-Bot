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
            "stop_loss_atr": 1.5,
            "signal_threshold_ratio": 0.50  # Range için daha düşük eşik
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
            "rsi_min": 50  # 60→50: Düşük volatilite dönemlerinde de sinyal üretsin
        }
    }
}

# Geriye dönük uyumluluk için düz liste
COIN_LIST = []
for group in COIN_GROUPS.values():
    COIN_LIST.extend(group["coins"])

# ==========================================
# 📊 SKORLAMA VE LİMİT AYARLARI (V7.0: STRATEJİ BAZLI EŞİK)
# ==========================================
# Maksimum teorik puan (tüm yön bağımsız + en yüksek yön bağımlı puanlar)
# BTC:20 + Reversal:18 + HTF:15 + Squeeze:15 + SMA200:12 + USDT:10 + RSI:10 + RSI4H:5 + VOL:8 + OBV:3 + ADX:7 = 123
MAX_TEORIK_PUAN = 123

# Strateji bazlı eşik oranları — Broker İyileştirmesi #1
ESIK_ORAN_TREND = 0.55   # Trend stratejisi: daha fazla sinyal üretsin
ESIK_ORAN_RANGE = 0.50   # Range/MR: yüksek WR'ye güven, daha gevşek eşik
ESIK_ORAN_MEME  = 0.52   # Meme/Volatility: ortada
ESIK_ORAN = ESIK_ORAN_TREND  # Geriye dönük uyumluluk

SINYAL_ESIK = int(MAX_TEORIK_PUAN * ESIK_ORAN_TREND)  # 123 * 0.55 = 67
YAKIN_ESIK = int(MAX_TEORIK_PUAN * 0.42)               # 123 * 0.42 = 51

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
# 🛡️ RİSK YÖNETİMİ AYARLARI (V7.0: Broker İyileştirmesi)
# ==========================================
MAX_AYNI_ANDA_ISLEM = 8             # Aynı anda açık olabilecek işlem (23→8: Korelasyon koruması)
MAX_AYNI_GRUP_ISLEM = 3             # Aynı gruptan (MAJOR/SWING/MEME) max açık işlem sayısı
VARSAYILAN_SL_CARPANI = 1.5        # Varsayılan ATR çarpanı (2.0→1.5: Daha az kayıp)
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
