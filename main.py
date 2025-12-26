import feedparser
import asyncio
import os
import sys
import sqlite3
import time
import re
from datetime import datetime, timedelta
from collections import defaultdict
from google import genai
from telegram import Bot
from telegram.constants import ParseMode

print("⚙️ Premium Skorlu Analist Botu Başlatılıyor...")

# --- ENV ---
TOKEN = os.getenv("BOT_TOKEN", "").strip()
KANAL_ID = int(os.getenv("KANAL_ID", "0"))
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

if not TOKEN or not GEMINI_KEY or not KANAL_ID:
    print("❌ ENV eksik")
    sys.exit(1)

# --- GEMINI ---
client = genai.Client(
    api_key=GEMINI_KEY,
    http_options={"api_version": "v1"}
)

# --- TELEGRAM ---
bot = Bot(token=TOKEN)

# --- RSS ---
RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://decrypt.co/feed",
    "https://cryptoslate.com/feed",
    "https://bitcoinmagazine.com/news/feed"
]

# --- COIN EVRENİ (PRO) ---
COIN_LIST = [
    "BTC","ETH","SOL","XRP","BNB","ADA","AVAX","DOGE",
    "TON","LINK","DOT","MATIC","ARB","OP","LTC","BCH"
]

# --- DB ---
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
    except:
        pass
    conn.close()

# --- TIME FILTER ---
def haber_yeni_mi(entry):
    try:
        if entry.published_parsed:
            t = datetime.fromtimestamp(time.mktime(entry.published_parsed))
            return (datetime.now() - t) < timedelta(minutes=15)
    except:
        pass
    return True

# --- COIN DETECT (PRO) ---
def coinleri_bul(text):
    bulunanlar = []
    for coin in COIN_LIST:
        if re.search(rf"\b{coin}\b", text, re.IGNORECASE):
            bulunanlar.append(coin)
    return bulunanlar[:5]

# --- SKOR ---
def skor_etiketi(s):
    if s >= 2: return "🟢 Güçlü Bullish 🚀"
    if s == 1: return "🟢 Bullish"
    if s == 0: return "⚖️ Nötr"
    if s == -1: return "🔴 Bearish"
    return "🔴 Güçlü Bearish 🔻"

# --- AI ANALİZ ---
async def ai_analiz(baslik, ozet, coinler):
    coin_text = ", ".join(coinler) if coinler else "Genel Piyasa"

    prompt = f"""
Sen elit bir kripto hedge-fund analistisin.

HABER:
{baslik}
{ozet}

COINLER:
{coin_text}

FORMAT DIŞINA ÇIKMA!

🔥 Özet: (max 12 kelime)
💡 Kritik Nokta: (tek cümle)
🪙 Coin Etkisi:
- Coin: Bullish/Bearish/Nötr (max 6 kelime)
🎯 Skor Analizi:
Skor: -2,-1,0,1,2
Yorum: Bullish 🚀 / Bearish 🔻 / Nötr ⚖️
Gerekçe: max 6 kelime
"""

    r = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    text = r.text.strip()
    return "\n".join(text.splitlines()[:12])  # taşmayı kes

# --- RSS LOOP ---
async def haberleri_kontrol_et():
    for rss in RSS_LIST:
        feed = feedparser.parse(rss)

        for entry in feed.entries[:3]:
            link = entry.link.strip()
            if link_var_mi(link): 
                continue
            if not haber_yeni_mi(entry):
                link_kaydet(link)
                continue

            link_kaydet(link)

            ozet = entry.get("summary", "")[:400]
            metin = entry.title + " " + ozet
            coinler = coinleri_bul(metin)

            ai_text = await ai_analiz(entry.title, ozet, coinler)
            skor_match = re.search(r"Skor:\s*(-?\d)", ai_text)
            skor = int(skor_match.group(1)) if skor_match else 0

            mesaj = f"""
📰 <b>{entry.title}</b>

🧠 <b>PİYASA ANALİZİ</b>
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

            print("✅ Paylaşıldı:", entry.title[:60])
            await asyncio.sleep(8)

# --- MAIN ---
async def main():
    db_baslat()
    print("🚀 Premium Bot Aktif")

    while True:
        print(f"🔄 ({datetime.now().strftime('%H:%M:%S')}) RSS Taraması Başlıyor...")
        await haberleri_kontrol_et()
        print("💤 Döngü tamamlandı | 60 sn bekleniyor\n")
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
