import feedparser
import asyncio
import os
import sys
import sqlite3
import time
import re
import ccxt
import numpy as np
from datetime import datetime, timedelta
from google import genai
from telegram import Bot
from telegram.constants import ParseMode

print("⚙️ Görsel Odaklı Premium Bot Başlatılıyor...")

# --- ENV ---
TOKEN = os.getenv("BOT_TOKEN", "").strip()
KANAL_ID = int(os.getenv("KANAL_ID", "0"))
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

if not TOKEN or not GEMINI_KEY or not KANAL_ID:
    print("❌ ENV eksik! Railway Variables kısmını kontrol et.")
    sys.exit(1)

# --- AYARLAR ---
client = genai.Client(api_key=GEMINI_KEY, http_options={"api_version": "v1"})
bot = Bot(token=TOKEN)

# Binance Bağlantısı (Sadece fiyat okumak için, API Key gerekmez)
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# --- RSS LİSTESİ ---
RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed"
]

# --- COIN LİSTESİ ---
COIN_LIST = [
    "BTC","ETH","SOL","XRP","BNB","ADA","AVAX","DOGE",
    "TON","LINK","DOT","MATIC","LTC","BCH","PEPE","FET"
]

# ==========================================
# 🎨 BÖLÜM 1: GÖRSEL FORMAT VE AI (HABER)
# ==========================================

def db_baslat():
    conn = sqlite3.connect("haber_hafizasi.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS gonderilenler (link TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()

def link_var_mi(link):
    conn = sqlite3.connect("haber_hafizasi.db")
    c = conn.cursor()
    c.execute("SELECT 1 FROM gonderilenler WHERE link=?", (link,))
    r = c.fetchone()
    conn.close()
    return r is not None

def link_kaydet(link):
    conn = sqlite3.connect("haber_hafizasi.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO gonderilenler VALUES (?)", (link,))
        conn.commit()
    except: pass
    conn.close()

def haber_yeni_mi(entry):
    try:
        if entry.published_parsed:
            t = datetime.fromtimestamp(time.mktime(entry.published_parsed))
            return (datetime.now() - t) < timedelta(minutes=20)
    except: pass
    return True

def coinleri_bul(text):
    bulunanlar = []
    for coin in COIN_LIST:
        if re.search(rf"\b{coin}\b", text, re.IGNORECASE):
            bulunanlar.append(coin)
    return bulunanlar[:5]

def skor_etiketi(s):
    if s >= 2: return "🟢 Güçlü Bullish"
    if s == 1: return "🟢 Bullish"
    if s == 0: return "⚖️ Nötr"
    if s == -1: return "🔴 Bearish"
    return "🔴 Güçlü Bearish"

async def ai_analiz(baslik, ozet, coinler):
    coin_text = ", ".join(coinler) if coinler else "Genel Piyasa"
    
    # SENİN İSTEDİĞİN GÖRSEL FORMATI BURADA OLUŞTURUYORUZ
    prompt = f"""
Sen bir kripto uzmanısın. Şu haberi analiz et:
HABER: {baslik}
{ozet}
COINLER: {coin_text}

Çıktıyı TAM OLARAK aşağıdaki şablonla ver (Emojileri kullan):

🔥 Özet: [Haberin tek cümlelik vurucu özeti]

💡 Kritik Nokta: [Yatırımcı için en önemli detay]

🪙 Coin Etkisi:
- [Coin Sembolü]: [Etki]

🎯 Skor Analizi:
Skor: [ -2 ile 2 arası bir tam sayı]
Yorum: [Kısa gerekçe]
"""
    try:
        r = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return r.text.strip()
    except: return "🔥 Özet: Analiz yapılamadı.\n🎯 Skor Analizi:\nSkor: 0"

async def haberleri_kontrol_et():
    for rss in RSS_LIST:
        try:
            feed = feedparser.parse(rss)
            for entry in feed.entries[:2]:
                link = entry.link.strip()
                if link_var_mi(link): continue
                
                # Tarih kontrolü (Eski haberleri atla)
                if not haber_yeni_mi(entry): 
                    link_kaydet(link)
                    continue

                link_kaydet(link)
                
                # Veri hazırlığı
                ozet = entry.get("summary", "")[:300]
                metin = entry.title + " " + ozet
                coinler = coinleri_bul(metin)
                
                # AI Analizi
                ai_text = await ai_analiz(entry.title, ozet, coinler)
                
                # Skoru metnin içinden çekip başlığa koymak için regex
                skor_match = re.search(r"Skor:\s*(-?\d)", ai_text)
                skor = int(skor_match.group(1)) if skor_match else 0

                # 📨 TELEGRAM MESAJ TASLAĞI (GÖRSELDEKİ FORMAT)
                mesaj = f"""
📰 <b>{entry.title}</b>

🧠 PİYASA ANALİZİ
<b>Skor:</b> {skor} | {skor_etiketi(skor)}

{ai_text}

🔗 <a href="{link}">Kaynak</a>
"""
                await bot.send_message(
                    chat_id=KANAL_ID, 
                    text=mesaj, 
                    parse_mode=ParseMode.HTML, 
                    disable_web_page_preview=True
                )
                print(f"✅ Haber Atıldı: {entry.title[:30]}")
                await asyncio.sleep(5)
        except Exception as e:
            print(f"RSS Hatası: {e}")

# ==========================================
# 📈 BÖLÜM 2: TEKNİK SİNYAL (TRADINGVIEW'SIZ)
# ==========================================

def rsi_hesapla(fiyatlar, periyot=14):
    deltalar = np.diff(fiyatlar)
    seed = deltalar[:periyot+1]
    up = seed[seed >= 0].sum()/periyot
    down = -seed[seed < 0].sum()/periyot
    rs = up/down
    rsi = np.zeros_like(fiyatlar)
    rsi[:periyot] = 100. - 100./(1. + rs)

    for i in range(periyot, len(fiyatlar)):
        delta = deltalar[i-1]
        if delta > 0: upval = delta; downval = 0.
        else: upval = 0.; downval = -delta
        
        up = (up * (periyot - 1) + upval) / periyot
        down = (down * (periyot - 1) + downval) / periyot
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)
        
    return rsi[-1]

async def piyasayi_tarama():
    print(f"🔍 ({datetime.now().strftime('%H:%M')}) Teknik Tarama...")
    for coin in COIN_LIST:
        symbol = f"{coin}/USDT"
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=20)
            closes = np.array([x[4] for x in bars])
            guncel_rsi = rsi_hesapla(closes)
            fiyat = closes[-1]
            
            sinyal = None
            if guncel_rsi < 30:
                sinyal = "LONG (AL) 🟢"
                yorum = f"RSI Dipte ({guncel_rsi:.1f})"
            elif guncel_rsi > 70:
                sinyal = "SHORT (SAT) 🔴"
                yorum = f"RSI Tepede ({guncel_rsi:.1f})"

            if sinyal:
                mesaj = f"""
🚨 <b>SİNYAL ALINDI</b>

🪙 <b>#{coin}</b>
📊 <b>Yön:</b> {sinyal}
💰 <b>Fiyat:</b> ${fiyat}
📉 <b>Sebep:</b> {yorum}
"""
                await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                await asyncio.sleep(1)
        except: continue

# ==========================================
# 🏁 MAIN LOOP
# ==========================================
async def main():
    db_baslat()
    print("🚀 Bot Aktif!")
    
    sayac = 0
    while True:
        # Her dakika haber bak
        await haberleri_kontrol_et()
        
        # Her 15 dakikada bir (15. döngüde) teknik analiz yap
        if sayac % 15 == 0:
            await piyasayi_tarama()
        
        sayac += 1
        print("💤 Bekleme (60sn)...")
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
