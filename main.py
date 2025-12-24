import feedparser
import asyncio
import os
from google import genai
from telegram import Bot
from telegram.constants import ParseMode

# --- Ayarlar ---
TOKEN = os.getenv("BOT_TOKEN", "").strip()
KANAL_ID_VAL = os.getenv("KANAL_ID", "").strip()
KANAL_ID = int(KANAL_ID_VAL) if KANAL_ID_VAL else None
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

# Yeni SDK İstemcisi - Versiyonu v1 olarak sabitleyebiliriz
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
        # Metin çok kısa gelirse başlığı kullanıyoruz
        metin_kaynak = icerik if len(icerik) > 50 else baslik
        
        # Model isminin önündeki 'models/' ekini kaldırarak deniyoruz
        # Bu genelde v1beta hatalarını çözer
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=f"Aşağıdaki haberi 2 kısa cümleyle Türkçe özetle:\n\n{metin_kaynak}"
        )
        
        if response and response.text:
            return response.text.strip()
        return "Özet oluşturulamadı."
        
    except Exception as e:
        print(f"❌ Gemini Hatası: {e}")
        return "AI şu an bu haberi özetleyemedi."

async def haberleri_kontrol_et():
    for rss in RSS_LIST:
        try:
            feed = feedparser.parse(rss.strip())
            for entry in feed.entries[:3]:
                link = entry.link.strip()
                if link not in gonderilenler:
                    # İçerik kısmını daha güvenli alalım
                    ozet_metni = entry.get("summary", entry.get("description", ""))
                    ai_sonuc = await ai_ozetle(entry.title, ozet_metni)
                    
                    mesaj = (
                        f"📰 <b>{entry.title}</b>\n\n"
                        f"🤖 <b>AI ÖZETİ:</b>\n{ai_sonuc}\n\n"
                        f"🔗 <a href='{link}'>Haberin Tamamı</a>"
                    )

                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                    gonderilenler.add(link)
                    await asyncio.sleep(5) # Hız sınırına takılmamak için süreyi artırdık
        except Exception as e:
            print(f"Hata: {e}")

async def main():
    if not KANAL_ID or not TOKEN:
        print("❌ KANAL_ID veya BOT_TOKEN eksik!")
        return
        
    print("🚀 Bot stabil modda başlatıldı...")
    while True:
        await haberleri_kontrol_et()
        await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(main())
