import feedparser
import asyncio
import os
import sys
import sqlite3
import time
from datetime import datetime, timedelta
from google import genai
from google.genai import types
from telegram import Bot
from telegram.constants import ParseMode

print("⚙️ Sistem Başlatılıyor (Skorlu Analist Modu)...")

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

# --- RSS ---
RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://decrypt.co/feed",
    "https://cryptoslate.com/feed",
    "https://bitcoinmagazine.com/news/feed",
    "https://tr.cointelegraph.com/rss"
]

bot = Bot(token=TOKEN)

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
            return (datetime.now() - t) < timedelta(hours=24)
    except:
        pass
    return True

# --- SKOR PARSER ---
def skor_ayikla(text):
    for line in text.splitlines():
        if "Skor:" in line:
            try:
                return int(line.replace("Skor:", "").strip())
            except:
                return 0
    return 0

def skor_etiketi(s):
    if s >= 2: return "🟢 Güçlü Bullish 🚀"
    if s == 1: return "🟢 Bullish"
    if s == 0: return "⚖️ Nötr"
    if s == -1: return "🔴 Bearish"
    return "🔴 Güçlü Bearish 🔻"

# --- AI ANALİZ ---
async def ai_analiz(baslik, ozet):
    try:
        prompt = f"""
Sen profesyonel bir kripto para analisti ve trader'sın.

HABER:
{baslik}
{ozet}

FORMATI AYNEN KORU:

🔥 Özet: (tek cümle, emoji)
💡 Önemli Detay: (kritik bilgi)
🎯 Skor Analizi:
Skor: [-2,-1,0,1,2]
Yorum: Bullish 🚀 / Bearish 🔻 / Nötr ⚖️
Gerekçe: en fazla 6 kelime

SKOR KURALLARI:
+2 = Resmi ve güçlü pozitif
+1 = Sınırlı pozitif
0 = Belirsizlik / jeopolitik risk
-1 = Risk artışı
-2 = Açık negatif
"""

        config = types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE")
            ]
        )

        r = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=config
        )

        return r.text.strip() if r.text else None

    except Exception as e:
        print("AI Hata:", e)
        return None

# --- RSS LOOP ---
async def haberleri_kontrol_et():
    for rss in RSS_LIST:
        feed = feedparser.parse(rss)

        for entry in feed.entries[:5]:
            link = entry.link.strip()
            if link_var_mi(link): continue
            if not haber_yeni_mi(entry):
                link_kaydet(link)
                continue

            link_kaydet(link)

            ozet = entry.get("summary", "")[:500]
            ai_text = await ai_analiz(entry.title, ozet)

            if not ai_text: continue

            skor = skor_ayikla(ai_text)
            etiket = skor_etiketi(skor)

            mesaj = f"""
📰 <b>{entry.title}</b>

🧠 <b>PİYASA ANALİZİ</b>
<b>Skor:</b> {skor} | {etiket}

{ai_text}

🔗 <a href="{link}">Kaynak</a>
"""

            await bot.send_message(
                chat_id=KANAL_ID,
                text=mesaj,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True
            )

            print("✅ Paylaşıldı:", entry.title[:40])
            await asyncio.sleep(5)

# --- MAIN ---
async def main():
    db_baslat()
    print("🚀 Bot aktif (Skor Sistemi ON)")

    while True:
        # ŞİMDİKİ SAATİ LOGA YAZ
        print(f"🔄 ({datetime.now().strftime('%H:%M:%S')}) RSS Taraması Başlıyor...")

        await haberleri_kontrol_et()

        # TARAMA BİTTİ LOGU
        print("💤 Tüm kontroller tamamlandı. 60 saniye bekleniyor...\n")

        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
