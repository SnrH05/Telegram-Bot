import feedparser
import asyncio
import os
import sys # Sistemi durdurmak için gerekli
from google import genai
from telegram import Bot
from telegram.constants import ParseMode

# --- Debug ve Ayarlar ---
print("⚙️ Sistem Değişkenleri Kontrol Ediliyor...")

TOKEN = os.getenv("BOT_TOKEN", "").strip()
KANAL_ID_RAW = os.getenv("KANAL_ID", "").strip()
KANAL_ID = int(KANAL_ID_RAW) if KANAL_ID_RAW else None
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

# 1. Sigorta: Token Kontrolü
if not TOKEN:
    print("❌ HATA: BOT_TOKEN bulunamadı! Railway Variables kısmını kontrol et.")
    sys.exit(1)

# 2. Sigorta: API Key Kontrolü (Hatanın Sebebi Burası)
if not GEMINI_KEY:
    print("❌ HATA: GEMINI_KEY Railway'den okunamadı! Boş geliyor.")
    print("👉 İpucu: Railway'de değişken adını tam olarak 'GEMINI_KEY' yazdığından emin ol.")
    sys.exit(1)
else:
    # Güvenlik için sadece ilk 4 karakteri yazdıralım
    print(f"✅ API Key Başarıyla Okundu: {GEMINI_KEY[:4]}****")

# --- İstemci Başlatma ---
try:
    client = genai.Client(
        api_key=GEMINI_KEY,
        http_options={'api_version': 'v1'} 
    )
except Exception as e:
    print(f"❌ İstemci Başlatma Hatası: {e}")
    sys.exit(1)

RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://feeds.bbci.co.uk/turkce/rss.xml",
    "https://www.ntv.com.tr/ekonomi.rss"
]
RSS_LIST = [url.strip() for url in RSS_LIST]

bot = Bot(token=TOKEN)
gonderilenler = set()

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

    except Exception as e:
        print(f"⚠️ AI Anlık Hata: {e}")
        return "AI şu an özetleyemedi."

async def haberleri_kontrol_et():
    for rss in RSS_LIST:
        try:
            feed = feedparser.parse(rss)
            for entry in feed.entries[:3]:
                link = entry.link.strip()
                if link not in gonderilenler:
                    body = entry.get("summary", entry.get("description", ""))
                    ozet = await ai_ozetle(entry.title, body)
                    
                    mesaj = (
                        f"📰 <b>{entry.title}</b>\n\n"
                        f"🤖 <b>AI ÖZETİ:</b>\n{ozet}\n\n"
                        f"🔗 <a href='{link}'>Haberin Tamamı</a>"
                    )

                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                    gonderilenler.add(link)
                    print(f"✅ Paylaşıldı: {entry.title[:20]}...")
                    await asyncio.sleep(5) 
        except Exception as e:
            print(f"⚠️ Akış hatası: {e}")

async def main():
    print("🚀 Bot Başlatılıyor...")
    while True:
        await haberleri_kontrol_et()
        await asyncio.sleep(600)

if __name__ == "__main__":
    asyncio.run(main())
