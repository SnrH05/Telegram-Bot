import feedparser
import asyncio
import os
from telegram import Bot
from telegram.constants import ParseMode

TOKEN = os.getenv("BOT_TOKEN")
KANAL_ID = int(os.getenv("KANAL_ID"))

RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://feeds.bbci.co.uk/turkce/rss.xml",
    "https://www.ntv.com.tr/ekonomi.rss"
]

bot = Bot(token=TOKEN)
gonderilenler = set()

async def haberleri_kontrol_et():
    for rss in RSS_LIST:
        feed = feedparser.parse(rss)
        kaynak = feed.feed.get("title", "Haber Kaynağı")

        for entry in feed.entries[:3]:
            if entry.link not in gonderilenler:
                mesaj = (
                    f"📰 <b>{entry.title}</b>\n"
                    f"📌 {kaynak}\n\n"
                    f"🔗 {entry.link}"
                )

                try:
                    await bot.send_message(
                        chat_id=KANAL_ID,
                        text=mesaj,
                        parse_mode=ParseMode.HTML
                    )
                    gonderilenler.add(entry.link)
                    await asyncio.sleep(2)

                except Exception as e:
                    print("Hata:", e)

async def main():
    print("🤖 Bot çalışıyor...")
    while True:
        await haberleri_kontrol_et()
        await asyncio.sleep(300)

asyncio.run(main())
