import feedparser
import asyncio
import os
from google import genai
from telegram import Bot
from telegram.constants import ParseMode

# --- Ayarlar ---
TOKEN = os.getenv("BOT_TOKEN", "").strip()
# Railway'den gelen ID bazen tırnaklı olabilir, temizleyip int yapalım
KANAL_ID_RAW = os.getenv("KANAL_ID", "").strip()
KANAL_ID = int(KANAL_ID_RAW) if KANAL_ID_RAW else None
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

# YENİ SDK İstemcisi
client = genai.Client(api_key=GEMINI_KEY)

RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://feeds.bbci.co.uk/turkce/rss.xml",
    "https://www.ntv.com.tr/ekonomi.rss"
]
# Linklerdeki gizli karakterleri temizle (ASCII hatasını bitirir)
RSS_LIST = [url.strip() for url in RSS_LIST]

bot = Bot(token=TOKEN)
gonderilenler = set()

async def ai_ozetle(baslik, icerik):
    if not GEMINI_KEY: return "API Key eksik."
    try:
        # Özetlenecek metni hazırla
        input_text = icerik if len(icerik) > 50 else baslik
        
        # models/ ekini kullanmadan direkt model ismini veriyoruz
        # Bu yeni SDK'da v1beta hatasını otomatik çözer
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"Bu haberi 2 kısa cümleyle Türkçe özetle:\n\n{input_text}"
        )
        
        if response and response.text:
            return response.text.strip()
        return "Özet içeriği boş döndü."

    except Exception as e:
        # Railway loglarında hatayı tam görmek için:
        print(f"❌ Gemini Motor Hatası: {e}")
        return "AI şu an özetleyemedi."

async def haberleri_kontrol_et():
    for rss in RSS_LIST:
        try:
            feed = feedparser.parse(rss)
            for entry in feed.entries[:3]:
                link = entry.link.strip()
                if link not in gonderilenler:
                    # Haberin gövdesini al
                    body = entry.get("summary", entry.get("description", ""))
                    ozet = await ai_ozetle(entry.title, body)
                    
                    mesaj = (
                        f"📰 <b>{entry.title}</b>\n\n"
                        f"🤖 <b>AI ÖZETİ:</b>\n{ozet}\n\n"
                        f"🔗 <a href='{link}'>Haberin Tamamı</a>"
                    )

                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                    gonderilenler.add(link)
                    print(f"✅ Başarıyla paylaşıldı: {entry.title[:30]}...")
                    await asyncio.sleep(5) 
        except Exception as e:
            print(f"⚠️ Akış hatası: {e}")

async def main():
    if not KANAL_ID or not TOKEN:
        print("❌ HATA: KANAL_ID veya TOKEN eksik!")
        return
    
    print("🚀 Bot ve Yeni AI Motoru Başlatıldı...")
    while True:
        await haberleri_kontrol_et()
        await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(main())
