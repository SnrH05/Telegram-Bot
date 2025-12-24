import feedparser
import asyncio
import os
import google.generativeai as genai # AI Kütüphanesi
from telegram import Bot
from telegram.constants import ParseMode

# --- Ayarlar ---
TOKEN = os.getenv("BOT_TOKEN", "").strip()
KANAL_ID = int(os.getenv("KANAL_ID").strip()) if os.getenv("KANAL_ID") else None
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

# Gemini Kurulumu
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    ai_model = genai.GenerativeModel('gemini-1.5-flash') # Hızlı ve stabil model

RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://feeds.bbci.co.uk/turkce/rss.xml",
    "https://www.ntv.com.tr/ekonomi.rss"
]

bot = Bot(token=TOKEN)
gonderilenler = set()

async def ai_ozetle(baslik, icerik):
    if not GEMINI_KEY:
        return "Özet hazırlanamadı (API Anahtarı eksik)."
    try:
        prompt = f"Aşağıdaki haberi dikkat çekici ve profesyonel bir dille 2 kısa cümleyle Türkçe özetle. Başlık: {baslik} İçerik: {icerik}"
        response = ai_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"AI Hatası: {e}")
        return "Özet çıkarılamadı."

async def haberleri_kontrol_et():
    for rss in RSS_LIST:
        try:
            feed = feedparser.parse(rss.strip())
            kaynak = feed.feed.get("title", "Haber Kaynağı")

            for entry in feed.entries[:3]:
                link = entry.link.strip()
                if link not in gonderilenler:
                    # AI Özetini Alıyoruz
                    ozet = await ai_ozetle(entry.title, entry.get("summary", ""))
                    
                    mesaj = (
                        f"📰 <b>{entry.title}</b>\n\n"
                        f"🤖 <b>AI ÖZETİ:</b>\n{ozet}\n\n"
                        f"📌 <i>{kaynak}</i>\n"
                        f"🔗 <a href='{link}'>Haberin Tamamı</a>"
                    )

                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                    gonderilenler.add(link)
                    await asyncio.sleep(2)
        except Exception as e:
            print(f"Hata: {e}")

async def main():
    print("🤖 AI Destekli Bot Çalışıyor...")
    while True:
        await haberleri_kontrol_et()
        await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(main())
