import feedparser
import asyncio
import os
import sys
import sqlite3
import time
import re
import threading
from datetime import datetime, timedelta
from flask import Flask, request
from google import genai
from telegram import Bot
from telegram.constants import ParseMode

print("⚙️ Premium Hibrit Bot (Haber + Sinyal) Başlatılıyor...")

# --- ENV ---
TOKEN = os.getenv("BOT_TOKEN", "").strip()
KANAL_ID = int(os.getenv("KANAL_ID", "0"))
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()
WEBHOOK_PORT = int(os.getenv("PORT", 8080)) # Sunucu portu (Genelde 80 veya 8080)

if not TOKEN or not GEMINI_KEY or not KANAL_ID:
    print("❌ ENV eksik")
    sys.exit(1)

# --- GLOBAL DEĞİŞKENLER ---
client = genai.Client(api_key=GEMINI_KEY, http_options={"api_version": "v1"})
bot = Bot(token=TOKEN)
app = Flask(__name__)
main_loop = None # Ana döngüye threadlerden erişmek için

# --- RSS LISTESI ---
RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://decrypt.co/feed",
]

# --- COIN EVRENİ ---
COIN_LIST = [
    "BTC","ETH","SOL","XRP","BNB","ADA","AVAX","DOGE",
    "TON","LINK","DOT","MATIC","ARB","OP","LTC","BCH"
]

# ==========================================
# 🚀 1. MODÜL: TRADINGVIEW WEBHOOK (SİNYAL)
# ==========================================

@app.route('/webhook', methods=['POST'])
def webhook():
    """TradingView'dan gelen POST isteğini karşılar"""
    try:
        data = request.json
        # TradingView'dan beklediğimiz JSON formatı:
        # {"symbol": "BTCUSDT", "side": "BUY", "price": 65000, "desc": "RSI Breakout"}
        
        symbol = data.get('symbol', 'UNKNOWN')
        side = data.get('side', 'SİNYAL')
        price = data.get('price', '0')
        desc = data.get('desc', 'Teknik Analiz')

        emoji = "🟢" if side.upper() == "BUY" or side.upper() == "LONG" else "🔴"
        
        mesaj = f"""
🚨 <b>YENİ SİNYAL GELDİ!</b>

🪙 <b>{symbol}</b>
{emoji} <b>Yön:</b> {side.upper()}
💰 <b>Fiyat:</b> {price}
📉 <b>Strateji:</b> {desc}

🤖 <i>TradingView Bot</i>
"""
        # Flask (Thread) içinden Async (Ana Döngü) fonksiyon çağırma:
        if main_loop:
            asyncio.run_coroutine_threadsafe(
                bot.send_message(
                    chat_id=KANAL_ID, 
                    text=mesaj, 
                    parse_mode=ParseMode.HTML
                ),
                main_loop
            )
        return "Sinyal Alindi", 200
    except Exception as e:
        print(f"Webhook Hatası: {e}")
        return "Hata", 500

def run_flask_server():
    """Flask sunucusunu ayrı thread'de başlatır"""
    print(f"📡 Webhook Dinleniyor: Port {WEBHOOK_PORT}")
    # '0.0.0.0' dışarıdan erişime açar.
    app.run(host='0.0.0.0', port=WEBHOOK_PORT, debug=False, use_reloader=False)

# ==========================================
# 📰 2. MODÜL: HABER VE AI (ESKİ KODUN)
# ==========================================

def db_baslat():
    conn = sqlite3.connect("haber_hafizasi.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS gonderilenler (link TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()

def link_var_mi(link):
    conn = sqlite3.connect("haber_hafizasi.db")
    c = conn.cursor()
    c.execute("SELECT 1 FROM gonderilenler WHERE link=?", (link,))
    r = c.fetchone()
    conn.close()
    return r is not None

def link_kaydet(link):
    conn = sqlite3.connect("haber_hafizasi.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO gonderilenler VALUES (?)", (link,))
        conn.commit()
    except:
        pass
    conn.close()

def haber_yeni_mi(entry):
    try:
        if entry.published_parsed:
            t = datetime.fromtimestamp(time.mktime(entry.published_parsed))
            return (datetime.now() - t) < timedelta(minutes=15)
    except:
        pass
    return True

def coinleri_bul(text):
    bulunanlar = []
    for coin in COIN_LIST:
        if re.search(rf"\b{coin}\b", text, re.IGNORECASE):
            bulunanlar.append(coin)
    return bulunanlar[:5]

def skor_etiketi(s):
    if s >= 2: return "🟢 Güçlü Bullish 🚀"
    if s == 1: return "🟢 Bullish"
    if s == 0: return "⚖️ Nötr"
    if s == -1: return "🔴 Bearish"
    return "🔴 Güçlü Bearish 🔻"

async def ai_analiz(baslik, ozet, coinler):
    coin_text = ", ".join(coinler) if coinler else "Genel Piyasa"
    prompt = f"""
Sen elit bir kripto hedge-fund analistisin.
HABER: {baslik}\n{ozet}
COINLER: {coin_text}
FORMAT DIŞINA ÇIKMA!
🔥 Özet: (max 12 kelime)
💡 Kritik Nokta: (tek cümle)
🪙 Coin Etkisi:
- Coin: Bullish/Bearish/Nötr (max 6 kelime)
🎯 Skor Analizi:
Skor: -2,-1,0,1,2
Yorum: Bullish 🚀 / Bearish 🔻 / Nötr ⚖️
Gerekçe: max 6 kelime
"""
    try:
        r = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        text = r.text.strip()
        return "\n".join(text.splitlines()[:12])
    except Exception as e:
        return f"AI Hatası: {str(e)}"

async def haberleri_kontrol_et():
    for rss in RSS_LIST:
        try:
            feed = feedparser.parse(rss)
            for entry in feed.entries[:3]:
                link = entry.link.strip()
                if link_var_mi(link): 
                    continue
                if not haber_yeni_mi(entry):
                    link_kaydet(link)
                    continue

                link_kaydet(link)
                ozet = entry.get("summary", "")[:400]
                metin = entry.title + " " + ozet
                coinler = coinleri_bul(metin)
                ai_text = await ai_analiz(entry.title, ozet, coinler)
                
                skor_match = re.search(r"Skor:\s*(-?\d)", ai_text)
                skor = int(skor_match.group(1)) if skor_match else 0

                mesaj = f"""
📰 <b>{entry.title}</b>

🧠 <b>PİYASA ANALİZİ</b>
<b>Skor:</b> {skor} | {skor_etiketi(skor)}

{ai_text}

🔗 <a href="{link}">Kaynak</a>
"""
                await bot.send_message(
                    chat_id=KANAL_ID,
                    text=mesaj,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True
                )
                print("✅ Haber Paylaşıldı:", entry.title[:60])
                await asyncio.sleep(8)
        except Exception as e:
            print(f"RSS Hatası: {e}")

# ==========================================
# 🏁 MAIN
# ==========================================
async def main():
    global main_loop
    main_loop = asyncio.get_running_loop() # Thread'lerin erişmesi için loop'u kaydet
    
    db_baslat()
    
    # Flask sunucusunu ayrı bir thread'de başlat
    flask_thread = threading.Thread(target=run_flask_server)
    flask_thread.daemon = True # Ana program kapanınca bu da kapansın
    flask_thread.start()
    
    print("🚀 Premium Bot Aktif (Haber + Sinyal)")

    while True:
        print(f"🔄 ({datetime.now().strftime('%H:%M:%S')}) RSS Taraması...")
        await haberleri_kontrol_et()
        print("💤 Döngü tamamlandı | 60 sn bekleniyor\n")
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
