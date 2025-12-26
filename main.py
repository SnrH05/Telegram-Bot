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

print("⚙️ Tam Otomatik Hibrit Bot (Haber + RSI Sinyal) Başlatılıyor...")

# --- ENV ---
TOKEN = os.getenv("BOT_TOKEN", "").strip()
KANAL_ID = int(os.getenv("KANAL_ID", "0"))
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

if not TOKEN or not GEMINI_KEY or not KANAL_ID:
    print("❌ ENV eksik! Lütfen Railway Variables kısmını kontrol et.")
    sys.exit(1)

# --- AYARLAR ---
client = genai.Client(api_key=GEMINI_KEY, http_options={"api_version": "v1"})
bot = Bot(token=TOKEN)

# Borsa Bağlantısı (API Key gerekmez, sadece fiyat okuyoruz)
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'} # Vadeli işlem fiyatları
})

# --- LİSTELER ---
RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed"
]

# Takip Edilecek Coinler (USDT paritesi varsayılır)
COIN_LIST = [
    "BTC","ETH","SOL","XRP","BNB","ADA","AVAX","DOGE",
    "TON","LINK","DOT","MATIC","LTC","BCH","PEPE","FET"
]

# ==========================================
# 📈 MODÜL 1: TEKNİK ANALİZ (SİNYAL)
# ==========================================

def rsi_hesapla(fiyatlar, periyot=14):
    """Basit RSI Hesaplama Fonksiyonu"""
    deltalar = np.diff(fiyatlar)
    seed = deltalar[:periyot+1]
    up = seed[seed >= 0].sum()/periyot
    down = -seed[seed < 0].sum()/periyot
    rs = up/down
    rsi = np.zeros_like(fiyatlar)
    rsi[:periyot] = 100. - 100./(1. + rs)

    for i in range(periyot, len(fiyatlar)):
        delta = deltalar[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        
        up = (up * (periyot - 1) + upval) / periyot
        down = (down * (periyot - 1) + downval) / periyot
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)
        
    return rsi[-1] # Son RSI değerini döndür

async def piyasayi_tarama():
    print(f"🔍 ({datetime.now().strftime('%H:%M')}) Teknik Analiz Taraması Başladı...")
    
    for coin in COIN_LIST:
        symbol = f"{coin}/USDT"
        try:
            # Son 20 mumluk veriyi çek (1 Saatlik grafik)
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=20)
            closes = [x[4] for x in bars] # Sadece kapanış fiyatlarını al
            
            # RSI Hesapla
            guncel_rsi = rsi_hesapla(np.array(closes))
            fiyat = closes[-1]
            
            sinyal_yonu = None
            mesaj_ek = ""

            # --- STRATEJİ (RSI 30-70) ---
            if guncel_rsi < 30:
                sinyal_yonu = "LONG (AL) 🟢"
                mesaj_ek = f"📉 RSI Aşırı Satımda ({guncel_rsi:.2f}). Tepki gelebilir!"
            elif guncel_rsi > 70:
                sinyal_yonu = "SHORT (SAT) 🔴"
                mesaj_ek = f"📈 RSI Aşırı Alımda ({guncel_rsi:.2f}). Düzeltme gelebilir!"

            # Sinyal varsa gönder
            if sinyal_yonu:
                mesaj = f"""
🚨 <b>TEKNİK SİNYAL TESPİT EDİLDİ</b>

🪙 <b>#{coin}</b>
📊 <b>Sinyal:</b> {sinyal_yonu}
💰 <b>Fiyat:</b> ${fiyat}
📉 <b>İndikatör:</b> RSI (1s)

ℹ️ <i>{mesaj_ek}</i>
"""
                await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                print(f"🚀 Sinyal Gönderildi: {coin}")
                
            await asyncio.sleep(1) # API limitine takılmamak için bekle

        except Exception as e:
            print(f"Hata ({coin}): {e}")
            continue

# ==========================================
# 📰 MODÜL 2: HABER VE AI 
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
    if s >= 2: return "🟢 Güçlü Bullish 🚀"
    if s == 1: return "🟢 Bullish"
    if s == 0: return "⚖️ Nötr"
    if s == -1: return "🔴 Bearish"
    return "🔴 Güçlü Bearish 🔻"

async def ai_analiz(baslik, ozet, coinler):
    coin_text = ", ".join(coinler) if coinler else "Genel Piyasa"
    prompt = f"""
Sen elit bir kripto analistisin. HABER: {baslik}\n{ozet}
COINLER: {coin_text}
FORMAT DIŞINA ÇIKMA!
FORMAT:
🔥 Özet: (max 10 kelime)
💡 Kritik: (tek cümle)
🎯 Skor: (-2 ile 2 arası sadece rakam)
Yorum: Bullish 🚀 / Bearish 🔻 / Nötr ⚖️
"""
    try:
        r = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return r.text.strip()
    except: return "AI Analiz Hatası"

async def haberleri_kontrol_et():
    for rss in RSS_LIST:
        try:
            feed = feedparser.parse(rss)
            for entry in feed.entries[:2]:
                link = entry.link.strip()
                if link_var_mi(link): continue
                if not haber_yeni_mi(entry): 
                    link_kaydet(link)
                    continue

                link_kaydet(link)
                ozet = entry.get("summary", "")[:300]
                metin = entry.title + " " + ozet
                coinler = coinleri_bul(metin)
                ai_text = await ai_analiz(entry.title, ozet, coinler)
                
                # Basit skor parse
                skor_match = re.search(r"Skor:\s*(-?\d)", ai_text)
                skor = int(skor_match.group(1)) if skor_match else 0

                mesaj = f"""
📰 <b>{entry.title}</b>
{skor_etiketi(skor)}

{ai_text}
🔗 <a href="{link}">Haberin Devamı</a>
"""
                await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                await asyncio.sleep(5)
        except Exception as e:
            print(f"RSS Hatası: {e}")

# ==========================================
# 🏁 ANA DÖNGÜ
# ==========================================
async def main():
    db_baslat()
    print("🚀 Bot Aktif! (Hem Haber Hem Teknik Analiz)")

    # Sayaçlar
    rss_sayac = 0
    teknik_sayac = 0

    while True:
        # Her 60 saniyede bir döngü döner
        
        # 1. Haberleri Kontrol Et (Her dakika)
        await haberleri_kontrol_et()
        
        # 2. Teknik Analiz Yap (Her 15 dakikada bir - 15 * 60sn)
        # Sürekli analiz yaparsa sunucuyu yorar ve spam yapar.
        if teknik_sayac % 15 == 0: 
            await piyasayi_tarama()
        
        teknik_sayac += 1
        print("💤 Bekleniyor... (60sn)")
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
