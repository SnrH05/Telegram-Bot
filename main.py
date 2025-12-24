import feedparser
import asyncio
import os
from google import genai # Yeni kütüphane yapısı
from telegram import Bot
from telegram.constants import ParseMode

# --- Ayarlar ---
TOKEN = os.getenv("BOT_TOKEN", "").strip()
KANAL_ID = int(os.getenv("KANAL_ID").strip())
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

# Yeni Gemini İstemcisi
client = genai.Client(api_key=GEMINI_KEY)

RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://feeds.bbci.co.uk/turkce/rss.xml",
    "https://www.ntv.com.tr/ekonomi.rss"
]

bot = Bot(token=TOKEN)
gonderilenler = set()

async def ai_ozetle(baslik, icerik):
    if not GEMINI_KEY: return "API Key eksik."
    try:
        metin = f"Başlık: {baslik}\nİçerik: {icerik}"
        # Yeni SDK'da fonksiyon ismi ve parametreler değişti
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"Şu haberi 2 kısa cümleyle Türkçe özetle:\n\n{metin}"
        )
        return response.text.strip()
    except Exception as e:
        print(f"❌ Yeni AI Hatası: {e}")
        return "AI şu an bu haberi özetleyemedi."

async def haberleri_kontrol_et():
    for rss in RSS_LIST:
        try:
            feed = feedparser.parse(rss.strip())
            for entry in feed.entries[:3]:
                link = entry.link.strip()
                if link not in gonderilenler:
                    icerik = entry.get("summary", entry.get("description", ""))
                    ozet = await ai_ozetle(entry.title, icerik)
                    
                    mesaj = (
                        f"📰 <b>{entry.title}</b>\n\n"
                        f"🤖 <b>AI ÖZETİ:</b>\n{ozet}\n\n"
                        f"🔗 <a href='{link}'>Haberin Tamamı</a>"
                    )

                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                    gonderilenler.add(link)
                    await asyncio.sleep(3)
        except Exception as e:
            print(f"Hata: {e}")

async def main():
    print("🚀 Bot Yeni SDK ile Başlatıldı...")
    while True:
        await haberleri_kontrol_et()
        await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(main())
