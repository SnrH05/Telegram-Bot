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
# 🪙 COIN VE RSS LİSTELERİ
# ==========================================
COIN_LIST = [
    "BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "AVAX", "DOGE",
    "TON", "LINK", "DOT", "POL", "LTC", "BCH", "PEPE", "FET",
    "SUI", "APT", "ARB", "OP", "TIA", "INJ", "RENDER"
]

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
# 📈 SKOR EŞİKLERİ (strategy.py'den import edilebilir)
# ==========================================
# Bu değerler strategy.py'de de tanımlı, merkezi tutarlılık için buradan kullan
MAX_TEORIK_PUAN = 100
SINYAL_ESIK = 65
YAKIN_ESIK = 50

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
MIN_ATR_YUZDE = 0.5                # Minimum ATR% (volatilite kontrolü)

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
