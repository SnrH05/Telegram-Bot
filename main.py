import feedparser
import asyncio
import os
import sys
import sqlite3
import time
import re
import ccxt
import numpy as np
import pandas as pd
import mplfinance as mpf
import io
from datetime import datetime, timedelta
from google import genai
from telegram import Bot
from telegram.constants import ParseMode

print("⚙️ ULTRA QUANT BOT BAŞLATILIYOR...")

# ==========================================
# 🔧 AYARLAR VE GÜVENLİK
# ==========================================
TOKEN = os.getenv("BOT_TOKEN", "").strip()
KANAL_ID = int(os.getenv("KANAL_ID", "0"))
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

if not TOKEN or not GEMINI_KEY or not KANAL_ID:
    print("❌ HATA: ENV bilgileri eksik! Railway Variables kısmını kontrol et.")
    sys.exit(1)

client = genai.Client(api_key=GEMINI_KEY, http_options={"api_version": "v1"})
bot = Bot(token=TOKEN)

# KuCoin Spot (Veri çekmek için stabil)
exchange = ccxt.kucoin({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'} 
})

# Takip Edilecek Coinler
COIN_LIST = [
    "BTC","ETH","SOL","XRP","BNB","ADA","AVAX","DOGE",
    "TON","LINK","DOT","MATIC","LTC","BCH","PEPE","FET",
    "SUI","APT","ARB","OP", "TIA", "INJ", "RNDR"
]

# RSS Haber Kaynakları
RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed"
]

# ==========================================
# 🧮 BÖLÜM 1: FİNANSAL MATEMATİK MOTORU
# ==========================================

def calculate_ema(series, span):
    """Üstel Hareketli Ortalama (Trend Tespiti için)"""
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, period=14):
    """Göreceli Güç Endeksi (Momentum için)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series):
    """MACD (Trend Değişimi ve Kesişimler)"""
    exp1 = calculate_ema(series, 12)
    exp2 = calculate_ema(series, 26)
    macd_line = exp1 - exp2
    signal_line = calculate_ema(macd_line, 9)
    return macd_line, signal_line

def calculate_adx(df, period=14):
    """ADX (Trendin Gücünü Ölçer - Yatay piyasayı eler)"""
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(df['high'] - df['low'])
    tr2 = pd.DataFrame(abs(df['high'] - df['close'].shift(1)))
    tr3 = pd.DataFrame(abs(df['low'] - df['close'].shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(period).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = 100 * (abs(minus_dm).ewm(alpha=1/period).mean() / atr)
    
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/period).mean()
    return adx

def calculate_atr(df, period=14):
    """ATR (Stop Loss mesafesini hesaplar)"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

# ==========================================
# 🎨 BÖLÜM 2: GRAFİK OLUŞTURUCU (Dark Theme)
# ==========================================

def grafik_olustur(coin, df_gelen, tp_price, sl_price):
    """Profesyonel TradingView Tarzı Grafik Çizer"""
    try:
        df = df_gelen.copy()
        
        # Grafik Verilerini Hazırla
        apds = [
            mpf.make_addplot(df['macd'], panel=1, color='#2962FF', title="MACD", width=1.0),
            mpf.make_addplot(df['signal'], panel=1, color='#FF6D00', width=1.0),
            mpf.make_addplot(df['ema200'], panel=0, color='white', width=0.8, linestyle='--') # Trend Referansı
        ]

        buf = io.BytesIO()
        
        # Tema Ayarları (Koyu Lacivert/Gri)
        theme_color = '#131722'
        grid_color = '#363c4e'
        text_color = '#b2b5be'
        
        my_style = mpf.make_mpf_style(
            base_mpf_style='binance',
            facecolor=theme_color,
            figcolor=theme_color,
            edgecolor=theme_color,
            gridcolor=grid_color,
            gridstyle=':',
            rc={'axes.labelcolor': text_color, 'xtick.color': text_color, 'ytick.color': text_color, 'text.color': text_color}
        )

        # TP ve SL Çizgileri (Kesikli Çizgi)
        h_lines = dict(hlines=[tp_price, sl_price], colors=['#089981', '#F23645'], linewidths=[1.5, 1.5], alpha=0.9, linestyle='-.')

        mpf.plot(
            df,
            type='candle',
            style=my_style,
            title=f"\n{coin}/USDT - Quant Stratejisi",
            ylabel='Fiyat ($)',
            ylabel_lower='MACD',
            addplot=apds,
            hlines=h_lines,
            volume=False,
            panel_ratios=(3, 1),
            savefig=dict(fname=buf, dpi=120, bbox_inches='tight', facecolor=theme_color)
        )
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Grafik Hatası: {e}")
        return None

# ==========================================
# 🧠 BÖLÜM 3: YAPAY ZEKA VE HABERLER
# ==========================================

def db_baslat():
    conn = sqlite3.connect("haber_hafizasi.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS gonderilenler (link TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()

def link_kontrol(link):
    conn = sqlite3.connect("haber_hafizasi.db")
    c = conn.cursor()
    # Kaydetmeye çalış, hata verirse zaten vardır
    try:
        c.execute("INSERT INTO gonderilenler VALUES (?)", (link,))
        conn.commit()
        yeni_mi = True
    except sqlite3.IntegrityError:
        yeni_mi = False
    conn.close()
    return yeni_mi

async def ai_analiz(baslik, ozet):
    prompt = f"""
    Bir kripto analistisin. Haberi analiz et.
    HABER: {baslik}
    {ozet}
    
    Çıktı Formatı:
    🔥 Özet: [Tek cümle]
    💡 Kritik: [Yatırımcı notu]
    🎯 Skor: [ -2 (Çok Kötü) ile 2 (Çok İyi) arası tam sayı]
    """
    try:
        r = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        text = r.text.strip()
        # Skoru regex ile çek
        skor_match = re.search(r"Skor:\s*(-?\d)", text)
        skor = int(skor_match.group(1)) if skor_match else 0
        return text, skor
    except:
        return "🔥 Özet: Analiz edilemedi.", 0

async def haberleri_kontrol_et():
    print("📰 Haberler taranıyor...")
    for rss in RSS_LIST:
        try:
            feed = feedparser.parse(rss)
            for entry in feed.entries[:2]:
                if not link_kontrol(entry.link): continue # Zaten veritabanında varsa atla
                
                # Çok eski haberleri atla (30 dk)
                if entry.published_parsed:
                    t = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    if (datetime.now() - t) > timedelta(minutes=30): continue

                ai_text, skor = await ai_analiz(entry.title, entry.get("summary", "")[:300])
                
                # Skor emojisi
                skor_icon = "🟢" if skor > 0 else "🔴" if skor < 0 else "⚖️"
                
                mesaj = f"""
📰 <b>{entry.title}</b>

{ai_text}

📊 <b>Etki Skoru:</b> {skor} {skor_icon}
🔗 <a href="{entry.link}">Haberi Oku</a>
"""
                await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                await asyncio.sleep(5)
        except Exception as e:
            print(f"RSS Hatası: {e}")

# ==========================================
# 🚀 BÖLÜM 4: ANA STRATEJİ DÖNGÜSÜ
# ==========================================

async def piyasayi_tarama():
    print(f"🔍 ({datetime.now().strftime('%H:%M')}) PİYASA ANALİZİ BAŞLIYOR...")
    
    for coin in COIN_LIST:
        symbol = f"{coin}/USDT"
        try:
            # 1. VERİ ÇEKME (EMA 200 için en az 300 mum)
            bars = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=300) #TEST İÇİN 1 DAKİKDA KONTROL EDİYOR(NORMALİ 1 SAAT)
            if not bars or len(bars) < 250: continue

            df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'], unit='ms')
            df.set_index('date', inplace=True)

            # 2. İNDİKATÖRLERİ HESAPLA
            df['ema200'] = calculate_ema(df['close'], 200) # Ana Trend
            df['rsi'] = calculate_rsi(df['close'])         # Momentum
            df['macd'], df['signal'] = calculate_macd(df['close']) # Kesişim
            df['adx'] = calculate_adx(df)                  # Trend Gücü
            df['atr'] = calculate_atr(df)                  # Volatilite (Stop için)

            # Son Veriler
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            fiyat = curr['close']
            atr = curr['atr']

            # 3. QUANT SİNYAL MANTIĞI
            sinyal = None
            risk_reward = 1.5 # 1 Risk al, 1.5 Kazan
            setup_reason = ""

            # --- LONG KURALLARI ---
            # 1. Trend Yukarı (Fiyat > EMA200)
            # 2. Trend Güçlü (ADX > 20) - Testere piyasasını eler
            if fiyat > curr['ema200'] and curr['adx'] > 20:
                # Sinyal: MACD Golden Cross VEYA RSI Oversold Dönüşü
                macd_cross = (prev['macd'] < prev['signal']) and (curr['macd'] > curr['signal'])
                rsi_bounce = (prev['rsi'] < 40) and (curr['rsi'] > 40)
                
                if macd_cross or rsi_bounce:
                    sinyal = "LONG 🟢"
                    stop_loss = fiyat - (atr * 2.0)
                    take_profit = fiyat + (atr * 2.0 * risk_reward)
                    setup_reason = "EMA200 Üstü Trend + Momentum Girişi"

            # --- SHORT KURALLARI ---
            # 1. Trend Aşağı (Fiyat < EMA200)
            # 2. Trend Güçlü (ADX > 20)
            elif fiyat < curr['ema200'] and curr['adx'] > 20:
                # Sinyal: MACD Death Cross VEYA RSI Overbought Dönüşü
                macd_cross = (prev['macd'] > prev['signal']) and (curr['macd'] < curr['signal'])
                rsi_dump = (prev['rsi'] > 60) and (curr['rsi'] < 60)
                
                if macd_cross or rsi_dump:
                    sinyal = "SHORT 🔴"
                    stop_loss = fiyat + (atr * 2.0)
                    take_profit = fiyat - (atr * 2.0 * risk_reward)
                    setup_reason = "EMA200 Altı Baskı + Momentum Kaybı"

            # 4. SİNYAL VARSA GÖNDER
            if sinyal:
                print(f"🎯 Sinyal Bulundu: {coin} -> {sinyal}")
                
                resim = grafik_olustur(coin, df.tail(80), take_profit, stop_loss)
                
                mesaj = f"""
⚡ <b>QUANT SİNYAL</b>

🪙 <b>#{coin}</b>
📊 <b>Yön:</b> {sinyal}
📉 <b>Setup:</b> {setup_reason}

💰 <b>Giriş:</b> ${fiyat}
🎯 <b>Hedef:</b> ${take_profit:.4f}
🛑 <b>Stop:</b> ${stop_loss:.4f}

🧠 <i>AI Notu: ADX filtresi {curr['adx']:.1f} puanla trendin güçlü olduğunu teyit etti.</i>
"""
                if resim:
                    await bot.send_photo(chat_id=KANAL_ID, photo=resim, caption=mesaj, parse_mode=ParseMode.HTML)
                else:
                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                
                # Aynı coine peş peşe sinyal atmaması için kısa bekleme
                await asyncio.sleep(2)

        except Exception as e:
            print(f"Hata ({coin}): {e}")
            continue

# ==========================================
# 🏁 MAIN
# ==========================================
async def main():
    db_baslat()
    print("🚀 Bot Tamamen Aktif! (Haber + Teknik Analiz)")
    
    sayac = 0
    while True:
        # Her döngüde haber kontrolü (Hızlı aksiyon)
        await haberleri_kontrol_et()
        
        # Her 15 dakikada bir Teknik Analiz (15 x 60sn = 900sn)
        # Bu, grafiklerin oturmasını bekler ve spamı önler
        # Test için her dakika çalışsın: (NORMALİ 1 SAAT)
        if True: 
            await piyasayi_tarama()
        
        sayac += 1
        print(f"💤 Bekleme... (Döngü: {sayac})")
        await asyncio.sleep(60) # 1 dakika bekle

if __name__ == "__main__":
    asyncio.run(main())
