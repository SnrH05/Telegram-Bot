import feedparser
import asyncio
import os
import sys
import sqlite3
import time
from datetime import datetime, timedelta
from dateutil import parser as date_parser # Tarih formatlarını anlamak için
from google import genai
from telegram import Bot
from telegram.constants import ParseMode

# --- Debug ve Ayarlar ---
print("⚙️ Sistem Başlatılıyor...")

TOKEN = os.getenv("BOT_TOKEN", "").strip()
KANAL_ID_RAW = os.getenv("KANAL_ID", "").strip()
KANAL_ID = int(KANAL_ID_RAW) if KANAL_ID_RAW else None
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

# --- Değişken Kontrolleri ---
if not TOKEN:
    print("❌ HATA: BOT_TOKEN eksik!")
    sys.exit(1)
if not GEMINI_KEY:
    print("❌ HATA: GEMINI_KEY eksik!")
    sys.exit(1)

# --- İstemci Başlatma ---
try:
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
    "https://www.ntv.com.tr/ekonomi.rss"
]
RSS_LIST = [url.strip() for url in RSS_LIST]

bot = Bot(token=TOKEN)

# --- VERİTABANI (SQLite) KURULUMU ---
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
        pass # Zaten varsa hata verme
    conn.close()

# --- YENİ EKLENTİ: ESKİ HABER FİLTRESİ ---
def haber_yeni_mi(entry):
    """Haber 24 saatten eskiyse False döner"""
    try:
        # Feedparser genelde zamanı 'published_parsed' içinde verir
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            haber_zamani = datetime.fromtimestamp(time.mktime(entry.published_parsed))
            su_an = datetime.now()
            fark = su_an - haber_zamani
            # Eğer haber 24 saatten (1 gün) eskiyse gönderme
            if fark > timedelta(hours=24):
                return False
        return True
    except:
        return True # Tarih okuyamazsak güvenli taraf seçip 'yeni' sayalım

async def ai_ozetle(baslik, icerik):
    try:
        metin_kaynak = icerik if len(icerik) > 50 else baslik
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"Bu haberi 2 kısa cümleyle Türkçe özetle:\n\n{metin_kaynak}"
        )
        if response and response.text:
            return response.text.strip()
        return "Özet oluşturulamadı."
    except Exception:
        return "AI şu an özetleyemedi."

async def haberleri_kontrol_et():
    for rss in RSS_LIST:
        try:
            feed = feedparser.parse(rss)
            # İlk 5 habere bakalım (daha derin tarama)
            for entry in feed.entries[:5]:
                link = entry.link.strip()
                
                # 1. Kontrol: Veritabanında var mı?
                if link_var_mi(link):
                    continue # Varsa atla

                # 2. Kontrol: Haber çok mu eski? (Örn: Bot yeni açıldı, dünkü haberi atmasın)
                if not haber_yeni_mi(entry):
                    # Veritabanına yine de kaydedelim ki bir daha sormasın
                    link_kaydet(link)
                    continue

                # --- GÖNDERME İŞLEMİ ---
                body = entry.get("summary", entry.get("description", ""))
                ozet = await ai_ozetle(entry.title, body)
                
                mesaj = (
                    f"📰 <b>{entry.title}</b>\n\n"
                    f"🤖 <b>AI ÖZETİ:</b>\n{ozet}\n\n"
                    f"🔗 <a href='{link}'>Haberin Tamamı</a>"
                )

                await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                
                # Başarılı olursa kaydet
                link_kaydet(link)
                print(f"✅ Paylaşıldı: {entry.title[:20]}...")
                await asyncio.sleep(5) 

        except Exception as e:
            print(f"⚠️ Akış hatası: {e}")

async def main():
    db_baslat() # Veritabanını oluştur
    print("🚀 Bot Akıllı Hafıza Modunda Başlatıldı...")
    while True:
        await haberleri_kontrol_et()
        await asyncio.sleep(600)

if __name__ == "__main__":
    asyncio.run(main())
