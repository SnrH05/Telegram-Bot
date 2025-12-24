import feedparser
import asyncio
import os
from google import genai
from telegram import Bot
from telegram.constants import ParseMode

# --- Ayarlar ---
TOKEN = os.getenv("BOT_TOKEN", "").strip()
KANAL_ID_RAW = os.getenv("KANAL_ID", "").strip()
KANAL_ID = int(KANAL_ID_RAW) if KANAL_ID_RAW else None
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

# --- Gemini İstemcisi (Hatanın Çözümü Burada) ---
# 'http_options' parametresi ile AI Studio üzerinden çalışmasını zorluyoruz
client = genai.Client(
    api_key=GEMINI_KEY,
    http_options={'api_version': 'v1'} # Beta olmayan stabil sürüm
)

RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://feeds.bbci.co.uk/turkce/rss.xml",
    "https://www.ntv.com.tr/ekonomi.rss"
]
RSS_LIST = [url.strip() for url in RSS_LIST]

bot = Bot(token=TOKEN)
gonderilenler = set()

async def ai_ozetle(baslik, icerik):
    if not GEMINI_KEY: return "API Key eksik."
    try:
        # Özetlenecek metni hazırla
        metin_kaynak = icerik if len(icerik) > 50 else baslik
        
        # Model ismini tırnak içinde direkt veriyoruz
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"Bu haberi 2 kısa cümleyle Türkçe özetle:\n\n{metin_kaynak}"
        )
        
        if response and response.text:
            return response.text.strip()
        return "Özet oluşturulamadı."

    except Exception as e:
        print(f"❌ Gemini Hatası: {e}")
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
                    print(f"✅ Paylaşıldı: {entry.title[:30]}")
                    await asyncio.sleep(5) 
        except Exception as e:
            print(f"⚠️ Hata: {e}")

async def main():
    if not KANAL_ID or not TOKEN:
        print("❌ HATA: KANAL_ID veya TOKEN eksik!")
        return
    
    print("🚀 Bot ve AI Motoru Stabil Modda Başlatıldı...")
    while True:
        await haberleri_kontrol_et()
        await asyncio.sleep(600)

if __name__ == "__main__":
    asyncio.run(main())
