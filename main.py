import feedparser
import asyncio
import os
import sys
import sqlite3
import time
from datetime import datetime, timedelta
from dateutil import parser as date_parser 
from google import genai
from google.genai import types 
from telegram import Bot
from telegram.constants import ParseMode

# --- Ayarlar ---
print("⚙️ Sistem Başlatılıyor (Gemini 2.0 Motoru)...")

TOKEN = os.getenv("BOT_TOKEN", "").strip()
KANAL_ID_RAW = os.getenv("KANAL_ID", "").strip()
KANAL_ID = int(KANAL_ID_RAW) if KANAL_ID_RAW else None
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

if not TOKEN or not GEMINI_KEY:
    print("❌ HATA: Token veya Key eksik!")
    sys.exit(1)

# --- İstemci Başlatma ---
try:
    # v1 API sürümü ile bağlanıyoruz
    client = genai.Client(
        api_key=GEMINI_KEY,
        http_options={'api_version': 'v1'} 
    )
except Exception as e:
    print(f"❌ İstemci Hatası: {e}")
    sys.exit(1)

RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://feeds.bbci.co.uk/turkce/rss.xml",
    "https://www.ntv.com.tr/ekonomi.rss",
    "https://www.reddit.com/r/CryptoCurrency.rss",
    "https://www.milliyet.com.tr/rss/rssnew/dunyarss.xml"
]
RSS_LIST = [url.strip() for url in RSS_LIST]

bot = Bot(token=TOKEN)

# --- VERİTABANI ---
def db_baslat():
    conn = sqlite3.connect("haber_hafizasi.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS gonderilenler (link TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()

def link_var_mi(link):
    conn = sqlite3.connect("haber_hafizasi.db")
    cursor = conn.cursor()
    cursor.execute("SELECT link FROM gonderilenler WHERE link=?", (link,))
    sonuc = cursor.fetchone()
    conn.close()
    return sonuc is not None

def link_kaydet(link):
    conn = sqlite3.connect("haber_hafizasi.db")
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO gonderilenler (link) VALUES (?)", (link,))
        conn.commit()
    except sqlite3.IntegrityError:
        pass 
    conn.close()

# --- TARİH KONTROLÜ ---
def haber_yeni_mi(entry):
    try:
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            haber_zamani = datetime.fromtimestamp(time.mktime(entry.published_parsed))
            if (datetime.now() - haber_zamani) > timedelta(hours=24):
                return False
        return True
    except:
        return True 

# --- GÜÇLENDİRİLMİŞ AI FONKSİYONU (Gemini 2.0) ---
async def ai_ozetle(baslik, icerik):
    try:
        metin_kaynak = icerik if len(icerik) > 50 else baslik
        
        # SANSÜRLERİ KALDIR (BLOCK_NONE)
        config = types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            ]
        )

        # BURASI DEĞİŞTİ: Listendeki 'gemini-2.0-flash' modelini kullanıyoruz
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Bu haberi tarafsız, profesyonel bir dille ve 2 kısa cümleyle Türkçe özetle:\n\n{metin_kaynak}",
            config=config
        )
        
        if response and response.text:
            return response.text.strip()
        return None 

    except Exception as e:
        print(f"⚠️ AI Hatası: {e}")
        return None

async def haberleri_kontrol_et():
    for rss in RSS_LIST:
        try:
            feed = feedparser.parse(rss)
            for entry in feed.entries[:5]:
                link = entry.link.strip()
                
                if link_var_mi(link): continue 
                if not haber_yeni_mi(entry):
                    link_kaydet(link)
                    continue

                link_kaydet(link) 

                try:
                    orjinal_ozet = entry.get("summary", entry.get("description", "Detaylar için linke tıklayın."))
                    ai_sonuc = await ai_ozetle(entry.title, orjinal_ozet)

                    if ai_sonuc:
                        final_metin = f"🤖 <b>AI ÖZETİ (v2.0):</b>\n{ai_sonuc}"
                    else:
                        temiz_ozet = orjinal_ozet.replace("<p>", "").replace("</p>", "").replace("<br>", "\n")[:250]
                        final_metin = f"📝 <b>HABER ÖZETİ:</b>\n{temiz_ozet}..."

                    mesaj = (
                        f"📰 <b>{entry.title}</b>\n\n"
                        f"{final_metin}\n\n"
                        f"🔗 <a href='{link}'>Haberin Tamamı</a>"
                    )

                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                    print(f"✅ Paylaşıldı: {entry.title[:20]}...")
                    await asyncio.sleep(5) 

                except Exception as e:
                    print(f"❌ Mesaj Hatası: {e}")

        except Exception as e:
            print(f"⚠️ Akış hatası: {e}")

async def main():
    db_baslat() 
    print("🚀 Bot Gemini 2.0 Flash ile Başladı...")
    while True:
        await haberleri_kontrol_et()
        await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(main())
