import feedparser
import asyncio
import os
import google.generativeai as genai
from telegram import Bot
from telegram.constants import ParseMode

# --- Ayarlar ---
TOKEN = os.getenv("BOT_TOKEN", "").strip()
KANAL_ID_STR = os.getenv("KANAL_ID", "").strip()
KANAL_ID = int(KANAL_ID_STR) if KANAL_ID_STR else None
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

# Gemini Kurulumu (Hata veren kısmı düzelttim)
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    # Model adını 'gemini-1.5-flash-latest' yaparak 404 hatasını bypass ediyoruz
    ai_model = genai.GenerativeModel('gemini-1.5-flash-latest')

# RSS listesini tertemiz yapalım, o ASCII hatası gelmesin
RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://feeds.bbci.co.uk/turkce/rss.xml",
    "https://www.ntv.com.tr/ekonomi.rss"
]
RSS_LIST = [url.strip() for url in RSS_LIST]

bot = Bot(token=TOKEN)
gonderilenler = set()

async def ai_ozetle(baslik, icerik):
    if not GEMINI_KEY: return "Özet yok."
    try:
        # İçerik çok kısaysa başlığı kullan
        metin = icerik if len(icerik) > 30 else baslik
        prompt = f"Şu haberi 2 kısa cümleyle Türkçe özetle:\n\n{metin}"
        
        response = ai_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ AI Hatası detay: {e}")
        return "AI şu an bu haberi özetleyemedi."

async def haberleri_kontrol_et():
    for rss in RSS_LIST:
        try:
            feed = feedparser.parse(rss)
            kaynak = feed.feed.get("title", "Haber Kaynağı")

            for entry in feed.entries[:3]:
                link = entry.link.strip()
                if link not in gonderilenler:
                    # Başlık ve özet/içerik bilgisini birleştirip gönderiyoruz
                    icerik = entry.get("summary", entry.get("description", ""))
                    ozet = await ai_ozetle(entry.title, icerik)
                    
                    mesaj = (
                        f"📰 <b>{entry.title}</b>\n\n"
                        f"🤖 <b>AI ÖZETİ:</b>\n{ozet}\n\n"
                        f"📌 <i>{kaynak}</i>\n"
                        f"🔗 <a href='{link}'>Devamını Oku</a>"
                    )

                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                    gonderilenler.add(link)
                    await asyncio.sleep(3) # Limitlere takılmamak için biraz daha yavaş
        except Exception as e:
            print(f"❌ Akış Hatası ({rss}): {e}")

async def main():
    print("🚀 Bot ve AI Motoru Başlatıldı...")
    while True:
        await haberleri_kontrol_et()
        await asyncio.sleep(600) # 10 dakikada bir kontrol iyidir

if __name__ == "__main__":
    asyncio.run(main())
