import feedparser
import asyncio
import os
import sys
import sqlite3
import time
import re
import ccxt
import numpy as np
from datetime import datetime, timedelta
from google import genai
from telegram import Bot
from telegram.constants import ParseMode
import pandas as pd
import mplfinance as mpf
import io

def atr_hesapla(df, periyot=14):
    """ATR (Ortalama Gerçek Aralık) Hesaplar"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    atr = true_range.rolling(periyot).mean()
    return atr.iloc[-1]

print("⚙️ Premium Skorlu Analist Botu Başlatılıyor...")

# --- ENV ---
TOKEN = os.getenv("BOT_TOKEN", "").strip()
KANAL_ID = int(os.getenv("KANAL_ID", "0"))
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

if not TOKEN or not GEMINI_KEY or not KANAL_ID:
    print("❌ ENV eksik! Railway Variables kısmını kontrol et.")
    sys.exit(1)

# --- AYARLAR ---
client = genai.Client(api_key=GEMINI_KEY, http_options={"api_version": "v1"})
bot = Bot(token=TOKEN)

# KuCoin Bağlantısı (Amerika sunucularına izin verir)
exchange = ccxt.kucoin({
    'enableRateLimit': True,
    # KuCoin'de vadeli yerine spot fiyatlarına bakmak daha stabildir ve RSI için yeterlidir
})

# --- RSS LİSTESİ ---
RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed"
]

# --- COIN LİSTESİ ---
COIN_LIST = [
    "BTC","ETH","SOL","XRP","BNB","ADA","AVAX","DOGE",
    "TON","LINK","DOT","MATIC","LTC","BCH","PEPE","FET"
]

# ==========================================
# 🎨 BÖLÜM 1: GÖRSEL FORMAT VE AI (HABER)
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
    except: pass
    conn.close()

def haber_yeni_mi(entry):
    try:
        if entry.published_parsed:
            t = datetime.fromtimestamp(time.mktime(entry.published_parsed))
            return (datetime.now() - t) < timedelta(minutes=20)
    except: pass
    return True

def coinleri_bul(text):
    bulunanlar = []
    for coin in COIN_LIST:
        if re.search(rf"\b{coin}\b", text, re.IGNORECASE):
            bulunanlar.append(coin)
    return bulunanlar[:5]

def skor_etiketi(s):
    if s >= 2: return "🟢 Güçlü Bullish"
    if s == 1: return "🟢 Bullish"
    if s == 0: return "⚖️ Nötr"
    if s == -1: return "🔴 Bearish"
    return "🔴 Güçlü Bearish"

async def ai_analiz(baslik, ozet, coinler):
    coin_text = ", ".join(coinler) if coinler else "Genel Piyasa"
    
    # SENİN İSTEDİĞİN GÖRSEL FORMATI BURADA OLUŞTURUYORUZ
    prompt = f"""
Sen bir kripto uzmanısın. Şu haberi analiz et:
HABER: {baslik}
{ozet}
COINLER: {coin_text}

Çıktıyı TAM OLARAK aşağıdaki şablonla ver ve Tamamdır işte analizin, Elit bir kripto analisti olarak yorumum gibi ibareleri kullanma (Emojileri kullan):

🔥 Özet: [Haberin tek cümlelik vurucu özeti]

💡 Kritik Nokta: [Yatırımcı için en önemli detay]

🪙 Coin Etkisi:
- [Coin Sembolü]: [Etki]

🎯 Skor Analizi:
Skor: [ -2 ile 2 arası bir tam sayı]
Yorum: [Kısa gerekçe]
"""
    try:
        r = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return r.text.strip()
    except: return "🔥 Özet: Analiz yapılamadı.\n🎯 Skor Analizi:\nSkor: 0"

async def haberleri_kontrol_et():
    for rss in RSS_LIST:
        try:
            feed = feedparser.parse(rss)
            for entry in feed.entries[:2]:
                link = entry.link.strip()
                if link_var_mi(link): continue
                
                # Tarih kontrolü (Eski haberleri atla)
                if not haber_yeni_mi(entry): 
                    link_kaydet(link)
                    continue

                link_kaydet(link)
                
                # Veri hazırlığı
                ozet = entry.get("summary", "")[:300]
                metin = entry.title + " " + ozet
                coinler = coinleri_bul(metin)
                
                # AI Analizi
                ai_text = await ai_analiz(entry.title, ozet, coinler)
                
                # Skoru metnin içinden çekip başlığa koymak için regex
                skor_match = re.search(r"Skor:\s*(-?\d)", ai_text)
                skor = int(skor_match.group(1)) if skor_match else 0

                # 📨 TELEGRAM MESAJ TASLAĞI (GÖRSELDEKİ FORMAT)
                mesaj = f"""
📰 <b>{entry.title}</b>

🧠 PİYASA ANALİZİ
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
                print(f"✅ Haber Atıldı: {entry.title[:30]}")
                await asyncio.sleep(5)
        except Exception as e:
            print(f"RSS Hatası: {e}")

# ==========================================
# 📈 BÖLÜM 2: TEKNİK SİNYAL (TRADINGVIEW'SIZ)
# ==========================================

def rsi_hesapla(fiyatlar, periyot=14):
    deltalar = np.diff(fiyatlar)
    seed = deltalar[:periyot+1]
    up = seed[seed >= 0].sum()/periyot
    down = -seed[seed < 0].sum()/periyot
    rs = up/down
    rsi = np.zeros_like(fiyatlar)
    rsi[:periyot] = 100. - 100./(1. + rs)

    for i in range(periyot, len(fiyatlar)):
        delta = deltalar[i-1]
        if delta > 0: upval = delta; downval = 0.
        else: upval = 0.; downval = -delta
        
        up = (up * (periyot - 1) + upval) / periyot
        down = (down * (periyot - 1) + downval) / periyot
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)
        
    return rsi[-1]

async def piyasayi_tarama():
    print(f"🔍 ({datetime.now().strftime('%H:%M')}) Teknik Tarama...")
    for coin in COIN_LIST:
        symbol = f"{coin}/USDT"
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=20)
            closes = np.array([x[4] for x in bars])
            guncel_rsi = rsi_hesapla(closes)
            fiyat = closes[-1]
            
            sinyal = None
            if guncel_rsi < 30:
                sinyal = "LONG (AL) 🟢"
                yorum = f"RSI Dipte ({guncel_rsi:.1f})"
            elif guncel_rsi > 70:
                sinyal = "SHORT (SAT) 🔴"
                yorum = f"RSI Tepede ({guncel_rsi:.1f})"

            if sinyal:
                mesaj = f"""
🚨 <b>SİNYAL ALINDI</b>

🪙 <b>#{coin}</b>
📊 <b>Yön:</b> {sinyal}
💰 <b>Fiyat:</b> ${fiyat}
📉 <b>Sebep:</b> {yorum}
"""
                await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                await asyncio.sleep(1)
        except: continue

def ema_hesapla(values, window):
    """Basit Numpy tabanlı EMA hesaplaması"""
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    
    # Convolution yerine daha geleneksel EMA formülü (daha hassas)
    alpha = 2 / (window + 1)
    ema = np.zeros_like(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = (values[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def macd_hesapla(fiyatlar):
    """MACD (12, 26, 9) hesaplar"""
    # 12 ve 26 periyotluk EMA
    ema12 = ema_hesapla(fiyatlar, 12)
    ema26 = ema_hesapla(fiyatlar, 26)
    
    # MACD Hattı
    macd_line = ema12 - ema26
    
    # Sinyal Hattı (MACD hattının 9 periyotluk EMA'sı)
    signal_line = ema_hesapla(macd_line, 9)
    
    return macd_line, signal_line

def rsi_hesapla(fiyatlar, periyot=14):
    deltalar = np.diff(fiyatlar)
    seed = deltalar[:periyot+1]
    up = seed[seed >= 0].sum()/periyot
    down = -seed[seed < 0].sum()/periyot
    
    if down == 0: return 100
    
    rs = up/down
    rsi = np.zeros_like(fiyatlar)
    rsi[:periyot] = 100. - 100./(1. + rs)

    for i in range(periyot, len(fiyatlar)):
        delta = deltalar[i-1]
        if delta > 0: upval = delta; downval = 0.
        else: upval = 0.; downval = -delta
        
        up = (up * (periyot - 1) + upval) / periyot
        down = (down * (periyot - 1) + downval) / periyot
        
        if down == 0: rsi[i] = 100
        else:
            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)
            
    return rsi[-1]

async def piyasayi_tarama():
    print(f"🔍 ({datetime.now().strftime('%H:%M')}) Teknik Tarama (RSI + MACD)...")
    for coin in COIN_LIST:
        symbol = f"{coin}/USDT"
        try:
            # MACD için en az 26+9 bar gerekli, garanti olsun diye 100 çekiyoruz
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
            if not bars or len(bars) < 50: continue

            closes = np.array([x[4] for x in bars])
            fiyat = closes[-1]
            
            # --- GÖSTERGE HESAPLAMALARI ---
            guncel_rsi = rsi_hesapla(closes)
            macd_line, signal_line = macd_hesapla(closes)
            
            # Son iki değer (Kesişim kontrolü için)
            macd_now = macd_line[-1]
            signal_now = signal_line[-1]
            macd_prev = macd_line[-2]
            signal_prev = signal_line[-2]

            # --- SİNYAL MANTIĞI ---
            sinyal = None
            sebep = []
            skor = 0 # Güç ölçer

            # 1. RSI Kontrolü
            if guncel_rsi < 30:
                sebep.append(f"RSI Dipte ({guncel_rsi:.1f})")
                skor += 1
            elif guncel_rsi > 70:
                sebep.append(f"RSI Tepede ({guncel_rsi:.1f})")
                skor -= 1

            # 2. MACD Kesişim (Crossover) Kontrolü
            # Bullish Crossover (Alttan yukarı kesişim)
            if macd_prev < signal_prev and macd_now > signal_now:
                sebep.append("MACD Al Kesişimi (Golden Cross)")
                skor += 2 # MACD kesişimi güçlü sinyaldir
            
            # Bearish Crossover (Üstten aşağı kesişim)
            elif macd_prev > signal_prev and macd_now < signal_now:
                sebep.append("MACD Sat Kesişimi (Death Cross)")
                skor -= 2

            # --- KARAR MEKANİZMASI ---
            if skor >= 2:
                sinyal = "🚀 GÜÇLÜ LONG (AL)"
            elif skor == 1: # Sadece RSI dipteyse veya zayıf sinyal
                sinyal = "🟢 LONG (AL)"
            elif skor <= -2:
                sinyal = "🩸 GÜÇLÜ SHORT (SAT)"
            elif skor == -1:
                sinyal = "🔴 SHORT (SAT)"

            # Sadece bir sinyal varsa gönder
            if sinyal:
                yorum_metni = " + ".join(sebep)
                mesaj = f"""
🚨 <b>SİNYAL ALINDI</b>

🪙 <b>#{coin}</b>
📊 <b>Yön:</b> {sinyal}
💰 <b>Fiyat:</b> ${fiyat}
📉 <b>Analiz:</b> {yorum_metni}
"""
                await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                print(f"🔔 Sinyal: {coin} -> {sinyal}")
                await asyncio.sleep(1) # Spam engellemek için kısa bekleme
                
        except Exception as e:
            print(f"Hata ({coin}): {e}")
            continue

# ==========================================
# 📈 BÖLÜM 2: TEKNİK SİNYAL VE GRAFİK (RSI + MACD)
# ==========================================

# ==========================================
# 📈 BÖLÜM 2: TEKNİK SİNYAL VE GRAFİK (RSI + MACD)
# ==========================================

def grafik_olustur(coin, df_gelen, macd, signal, tp_price, sl_price):
    """Verilen verilerden TP ve SL çizgili grafik oluşturur"""
    try:
        # Verinin kopyasını al
        df = df_gelen.copy()
        
        # MACD verilerini ekle
        df['MACD'] = macd
        df['Signal'] = signal
        
        # MACD Çizgileri
        apds = [
            mpf.make_addplot(df['MACD'], panel=1, color='#ff00ff', title="MACD", width=1.0),
            mpf.make_addplot(df['Signal'], panel=1, color='#00ffff', width=1.0)
        ]

        buf = io.BytesIO()

        # Tema Ayarları
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
            rc={'axes.labelcolor': text_color, 'xtick.color': text_color, 'ytick.color': text_color, 'text.color': 'white', 'axes.edgecolor': grid_color}
        )

        # TP ve SL Çizgileri (Yatay Çizgiler)
        # hlines: Fiyat seviyeleri, colors: Renkler (Yeşil/Kırmızı), linewidths: Kalınlık
        h_lines = dict(hlines=[tp_price, sl_price], colors=['#00FF00', '#FF0000'], linewidths=[1.5, 1.5], alpha=0.7)

        # Grafik Çizimi
        mpf.plot(
            df,
            type='candle',
            style=my_style, 
            title=f"\n{coin}/USDT - 1H Analiz",
            ylabel='Fiyat ($)',
            ylabel_lower='MACD',
            addplot=apds,
            hlines=h_lines, # <-- Yeni eklenen kısım (TP/SL Çizgileri)
            volume=False,
            panel_ratios=(3, 1),
            savefig=dict(fname=buf, dpi=100, bbox_inches='tight', facecolor=theme_color)
        )
        
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Grafik Hatası: {e}")
        return None
    
    try:
        # MACD verilerini DataFrame'e ekle (Çizim için)
        df['MACD'] = macd
        df['Signal'] = signal
        
        # Ekstra Grafikler (AddPlots) - MACD ve Sinyal çizgisi
        apds = [
            mpf.make_addplot(df['MACD'], panel=1, color='fuchsia', title="MACD"),
            mpf.make_addplot(df['Signal'], panel=1, color='b')
        ]

        # Resmi belleğe kaydetmek için buffer
        buf = io.BytesIO()

        # Grafik Stili ve Çizim
        mpf.plot(
            df,
            type='candle',
            style='binance', # Koyu tema
            title=f"\n{coin}/USDT - 1H Analiz",
            ylabel='Fiyat ($)',
            ylabel_lower='MACD',
            addplot=apds,
            volume=False,
            panel_ratios=(3, 1), # Fiyat grafiği büyük, MACD küçük olsun
            savefig=dict(fname=buf, dpi=100, bbox_inches='tight')
        )
        
        buf.seek(0) # Dosyanın başına dön
        return buf
    except Exception as e:
        print(f"Grafik Hatası: {e}")
        return None

def ema_hesapla(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    alpha = 2 / (window + 1)
    ema = np.zeros_like(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = (values[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def macd_hesapla(fiyatlar):
    ema12 = ema_hesapla(fiyatlar, 12)
    ema26 = ema_hesapla(fiyatlar, 26)
    macd_line = ema12 - ema26
    signal_line = ema_hesapla(macd_line, 9)
    return macd_line, signal_line

def rsi_hesapla(fiyatlar, periyot=14):
    deltalar = np.diff(fiyatlar)
    seed = deltalar[:periyot+1]
    up = seed[seed >= 0].sum()/periyot
    down = -seed[seed < 0].sum()/periyot
    if down == 0: return 100
    rs = up/down
    rsi = np.zeros_like(fiyatlar)
    rsi[:periyot] = 100. - 100./(1. + rs)
    for i in range(periyot, len(fiyatlar)):
        delta = deltalar[i-1]
        if delta > 0: upval = delta; downval = 0.
        else: upval = 0.; downval = -delta
        up = (up * (periyot - 1) + upval) / periyot
        down = (down * (periyot - 1) + downval) / periyot
        if down == 0: rsi[i] = 100
        else:
            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)
    return rsi[-1]

async def piyasayi_tarama():
    print(f"🔍 ({datetime.now().strftime('%H:%M')}) Teknik Tarama (ATR Destekli)...")
    for coin in COIN_LIST:
        symbol = f"{coin}/USDT"
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
            if not bars or len(bars) < 50: continue

            # Numpy array (Hız için)
            closes = np.array([x[4] for x in bars])
            fiyat = closes[-1]
            
            # Pandas DataFrame (Grafik ve ATR için)
            df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'], unit='ms')
            df.set_index('date', inplace=True)

            # --- GÖSTERGELER ---
            guncel_rsi = rsi_hesapla(closes)
            macd_line, signal_line = macd_hesapla(closes)
            atr_degeri = atr_hesapla(df) # <-- ATR Hesapladık
            
            macd_now = macd_line[-1]
            signal_now = signal_line[-1]
            macd_prev = macd_line[-2]
            signal_prev = signal_line[-2]

            # --- SİNYAL MANTIĞI ---
            sinyal = None
            sebep = []
            skor = 0
            tp_fiyat = 0
            sl_fiyat = 0

            # RSI ve MACD Kontrolleri
            if guncel_rsi < 30:
                sebep.append(f"RSI Dipte ({guncel_rsi:.1f})")
                skor += 1
            elif guncel_rsi > 70:
                sebep.append(f"RSI Tepede ({guncel_rsi:.1f})")
                skor -= 1

            if macd_prev < signal_prev and macd_now > signal_now:
                sebep.append("MACD Al Kesişimi")
                skor += 2
            elif macd_prev > signal_prev and macd_now < signal_now:
                sebep.append("MACD Sat Kesişimi")
                skor -= 2

            # KARAR ve TP/SL Hesaplama
            # Strateji: SL = 2 ATR, TP = 3 ATR (1.5 Risk/Reward Oranı)
            
            if skor >= 2: 
                sinyal = "🚀 GÜÇLÜ LONG"
                sl_fiyat = fiyat - (atr_degeri * 2.0)
                tp_fiyat = fiyat + (atr_degeri * 3.0)
                
            elif skor == 1: 
                sinyal = "🟢 LONG"
                sl_fiyat = fiyat - (atr_degeri * 1.5) # Zayıf sinyalde stop daha yakın
                tp_fiyat = fiyat + (atr_degeri * 2.5)

            elif skor <= -2: 
                sinyal = "🩸 GÜÇLÜ SHORT"
                sl_fiyat = fiyat + (atr_degeri * 2.0)
                tp_fiyat = fiyat - (atr_degeri * 3.0)

            elif skor == -1: 
                sinyal = "🔴 SHORT"
                sl_fiyat = fiyat + (atr_degeri * 1.5)
                tp_fiyat = fiyat - (atr_degeri * 2.5)

            if sinyal:
                print(f"📸 Sinyal ve Grafik: {coin}...")
                
                # Grafiği Çiz (TP ve SL gönderiyoruz)
                resim = grafik_olustur(coin, df.tail(60), macd_line[-60:], signal_line[-60:], tp_fiyat, sl_fiyat)
                
                yorum_metni = " + ".join(sebep)
                
                # Mesaj Formatı
                mesaj = f"""
🚨 <b>SİNYAL ALINDI</b>

🪙 <b>#{coin}</b>
📊 <b>Yön:</b> {sinyal}
💰 <b>Giriş:</b> ${fiyat}

🎯 <b>Hedef (TP):</b> ${tp_fiyat:.4f}
🛑 <b>Stop (SL):</b> ${sl_fiyat:.4f}

📉 <b>Analiz:</b> {yorum_metni}
"""
                if resim:
                    await bot.send_photo(chat_id=KANAL_ID, photo=resim, caption=mesaj, parse_mode=ParseMode.HTML)
                else:
                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                
                await asyncio.sleep(2)

        except Exception as e:
            print(f"Hata ({coin}): {e}")
            continue

# ==========================================
# 🏁 MAIN LOOP
# ==========================================
async def main():
    db_baslat()
    print("🚀 Bot Aktif!")
    
    sayac = 0
    while True:
        # Her dakika haber bak
        await haberleri_kontrol_et()
        
        # Her 15 dakikada bir (15. döngüde) teknik analiz yap
        if sayac % 15 == 0:
            await piyasayi_tarama()
        
        sayac += 1
        print("💤 Bekleme (60sn)...")
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
