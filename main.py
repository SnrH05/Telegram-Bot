import feedparser
import asyncio
import os
from telegram import Bot
from telegram.constants import ParseMode

# Değişkenleri Railway'den alırken sağındaki solundaki boşlukları temizleyelim
TOKEN = os.getenv("BOT_TOKEN", "").strip()
# Kanal ID string gelirse hata vermesin diye int'e çeviriyoruz
KANAL_ID_RAW = os.getenv("KANAL_ID", "").strip()
KANAL_ID = int(KANAL_ID_RAW) if KANAL_ID_RAW else None

RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://feeds.bbci.co.uk/turkce/rss.xml",
    "https://www.ntv.com.tr/ekonomi.rss"
]

# RSS listesini de tertemiz yapalım
RSS_LIST = [url.strip() for url in RSS_LIST]

bot = Bot(token=TOKEN)
gonderilenler = set()

async def haberleri_kontrol_et():
    for rss in RSS_LIST:
        try:
            # feedparser bazen bozuk URL'de hata vermez ama sonuç boş döner, kontrol edelim
            feed = feedparser.parse(rss)
            if not feed.feed:
                print(f"⚠️ Kaynak çekilemedi veya boş: {rss}")
                continue

            kaynak = feed.feed.get("title", "Haber Kaynağı")

            for entry in feed.entries[:3]:
                # Linki temizleyip kontrol edelim
                clean_link = entry.link.strip()
                
                if clean_link not in gonderilenler:
                    mesaj = (
                        f"📰 <b>{entry.title}</b>\n"
                        f"📌 {kaynak}\n\n"
                        f"🔗 {clean_link}"
                    )

                    await bot.send_message(
                        chat_id=KANAL_ID,
                        text=mesaj,
                        parse_mode=ParseMode.HTML
                    )
                    gonderilenler.add(clean_link)
                    print(f"✅ Gönderildi: {entry.title}")
                    await asyncio.sleep(2)

        except Exception as e:
            print(f"❌ Hata oluştu ({rss}): {e}")

async def main():
    if not TOKEN or not KANAL_ID:
        print("❌ HATA: BOT_TOKEN veya KANAL_ID eksik! Railway Variables kısmını kontrol et.")
        return

    print("🤖 Bot aktif, haberler taranıyor...")
    while True:
        await haberleri_kontrol_et()
        # Çok sık kontrol edip IP ban yemeyelim, 5 dakika (300sn) ideal
        await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(main())
