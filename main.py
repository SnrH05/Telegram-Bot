import feedparser
import asyncio
import os
import sys
import sqlite3
import time
import re
# DİKKAT: ccxt'nin asenkron modülünü çağırıyoruz
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
import mplfinance as mpf
import io
from datetime import datetime, timedelta
from google import genai
from telegram import Bot
from telegram.constants import ParseMode

print("⚙️ ULTRA QUANT PIVOT MASTER BOT (TURBO MOD v2) BAŞLATILIYOR...")

# ==========================================
# 🔧 AYARLAR
# ==========================================
TOKEN = os.getenv("BOT_TOKEN", "").strip()
KANAL_ID = int(os.getenv("KANAL_ID", "0"))
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

if not TOKEN or not GEMINI_KEY or not KANAL_ID:
    print("❌ HATA: ENV bilgileri eksik!")
    sys.exit(1)

# Gemini Client (Thread içinde çağıracağız)
client = genai.Client(api_key=GEMINI_KEY, http_options={"api_version": "v1"})
bot = Bot(token=TOKEN)

exchange_config = {
'enableRateLimit': True,
'options': {'defaultType': 'spot'} 
}

COIN_LIST = [
    "BTC","ETH","SOL","XRP","BNB","ADA","AVAX","DOGE",
    "TON","LINK","DOT","POL","LTC","BCH","PEPE","FET",
    "SUI","APT","ARB","OP", "TIA", "INJ", "RENDER"
]

RSS_LIST = [
    "https://cryptonews.com/news/feed/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed"
]

SON_SINYAL_ZAMANI = {}

# ==========================================
# 🧮 BÖLÜM 1: İNDİKATÖRLER
# ==========================================
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series):
    exp1 = calculate_ema(series, 12)
    exp2 = calculate_ema(series, 26)
    macd_line = exp1 - exp2
    signal_line = calculate_ema(macd_line, 9)
    return macd_line, signal_line

def calculate_adx(df, period=14):
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm = plus_dm.where(plus_dm > 0, 0)
    minus_dm = minus_dm.where(minus_dm < 0, 0)
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
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def calculate_pivots(df_hourly):
    try:
        df_daily = df_hourly.resample('D').agg({'high': 'max','low': 'min','close': 'last'})
        last_day = df_daily.iloc[-2]
        pivot = (last_day['high'] + last_day['low'] + last_day['close']) / 3
        r1 = (2 * pivot) - last_day['low']
        s1 = (2 * pivot) - last_day['high']
        return pivot, r1, s1
    except:
        return 0, 0, 0

# ==========================================
# 🎨 BÖLÜM 2: GRAFİK (THREAD İLE OPTİMİZE)
# ==========================================
def _grafik_olustur_sync(coin, df_gelen, tp1, tp2, tp3, sl_price, pivot, r1, s1):
    try:
        df = df_gelen.copy()
        apds = [
            mpf.make_addplot(df['macd'], panel=1, color='#2962FF', title="MACD", width=1.0),
            mpf.make_addplot(df['signal'], panel=1, color='#FF6D00', width=1.0),
            mpf.make_addplot(df['ema200'], panel=0, color='white', width=0.8, linestyle='--')
        ]
        buf = io.BytesIO()
        theme_color = '#131722'
        grid_color = '#363c4e'
        text_color = '#b2b5be'
        my_style = mpf.make_mpf_style(
            base_mpf_style='binance', facecolor=theme_color, figcolor=theme_color, edgecolor=theme_color,
            gridcolor=grid_color, gridstyle=':', rc={'axes.labelcolor': text_color, 'xtick.color': text_color, 'ytick.color': text_color, 'text.color': text_color}
        )
        h_lines = dict(
            hlines=[tp1, tp2, tp3, sl_price, pivot, r1, s1], 
            colors=['#98FB98', '#32CD32', '#006400', '#FF0000', '#FFFF00', '#FF4500', '#00BFFF'],
            linewidths=[1.0, 1.2, 1.5, 1.5, 0.8, 0.8, 0.8], alpha=0.8, linestyle='-.'
        )
        mpf.plot(
            df, type='candle', style=my_style, title=f"\n{coin}/USDT - Pivot & TP Analiz",
            ylabel='Fiyat ($)', ylabel_lower='MACD', addplot=apds, hlines=h_lines, volume=False,
            panel_ratios=(3, 1), savefig=dict(fname=buf, dpi=120, bbox_inches='tight', facecolor=theme_color)
        )
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Grafik Hatası: {e}")
        return None

async def grafik_olustur_async(coin, df, tp1, tp2, tp3, sl, pivot, r1, s1):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _grafik_olustur_sync, coin, df, tp1, tp2, tp3, sl, pivot, r1, s1)

# ==========================================
# 🧠 BÖLÜM 3: YAPAY ZEKA (Strict Mode & Temiz Format)
# ==========================================

def db_baslat():
    conn = sqlite3.connect("haber_hafizasi.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS gonderilenler (link TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()

def link_kontrol(link):
    with sqlite3.connect("haber_hafizasi.db") as conn:
        c = conn.cursor()
        try:
            c.execute("INSERT INTO gonderilenler VALUES (?)", (link,))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

# 🚀 GÜNCELLEME: Promptu "Katı Kurallı" moda geçirdik, gevezelik yapamaz.
def _ai_analiz_sync(prompt):
    try:
        r = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        text = r.text.strip()
        
        # Regex ile sadece istenen kısımları çekiyoruz
        ozet_match = re.search(r"ÖZET:(.*)", text, re.DOTALL)
        skor_match = re.search(r"SKOR:\s*(-?\d)", text)
        
        temiz_ozet = ozet_match.group(1).strip() if ozet_match else "Özet oluşturulamadı."
        skor = int(skor_match.group(1)) if skor_match else 0
        return temiz_ozet, skor
    except:
        return "Analiz yapılamadı.", 0

async def ai_analiz(baslik, ozet):
    # Katı Prompt
    prompt = f"""
    GÖREV: Aşağıdaki kripto haberini analiz et.
    HABER BAŞLIĞI: {baslik}
    HABER ÖZETİ: {ozet}
    
    KURALLAR:
    1. Asla "Tamam", "Anlaşıldı", "Analiz ediyorum" gibi giriş cümleleri kurma.
    2. Asla "Varsayımlar", "Ek Notlar" gibi başlıklar ekleme.
    3. Çıktı formatına %100 sadık kal.
    4. Skor -2 (Çok Kötü) ile +2 (Çok İyi) arasında tam sayı olsun.

    İSTENEN ÇIKTI FORMATI:
    ÖZET:[Tek bir emoji ile başlayan maksimum 2 cümlelik özet]
    SKOR:[Sadece Sayı]
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _ai_analiz_sync, prompt)

# 🚀 GÜNCELLEME: Haber mesaj tasarımı sadeleştirildi
async def haberleri_kontrol_et():
    print("📰 Haberler taranıyor...")
    for rss in RSS_LIST:
        try:
            feed = feedparser.parse(rss)
            for entry in feed.entries[:2]:
                if not link_kontrol(entry.link): continue 
                if entry.published_parsed:
                    t = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    if (datetime.now() - t) > timedelta(minutes=45): continue
                
                # HTML temizliği
                raw_summary = entry.get("summary", entry.get("description", ""))
                clean_text = re.sub('<[^<]+?>', '', raw_summary)
                
                ai_text, skor = await ai_analiz(entry.title, clean_text[:500])
                if abs(skor) < 2: continue 
                
                skor_icon = "🟢" if skor > 0 else "🔴"
                
                mesaj = f"""
<b>{entry.title}</b>

{ai_text}

🎯 <b>Piyasa Etkisi:</b> {skor_icon} <b>({skor})</b>
🔗 <a href='{entry.link}'>Kaynağa Git</a>
"""
                await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                await asyncio.sleep(2)
        except Exception as e:
            print(f"RSS Hatası: {e}")

# ==========================================
# 📊 BÖLÜM 4: RAPORLAMA VE DB (DETAYLI BİLDİRİM)
# ==========================================
RAPOR_ZAMANI = datetime.now()

def pnl_db_baslat():
    with sqlite3.connect("trade_pnl.db") as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS islemler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT, yon TEXT, giris_fiyat REAL, tp1 REAL, sl REAL,
            durum TEXT DEFAULT 'ACIK', pnl_yuzde REAL DEFAULT 0,
            kapanis_zamani DATETIME
        )""")

def islem_kaydet(coin, yon, giris, tp1, sl):
    with sqlite3.connect("trade_pnl.db") as conn:
        conn.execute("INSERT INTO islemler (coin, yon, giris_fiyat, tp1, sl) VALUES (?, ?, ?, ?, ?)", 
                  (coin, yon, giris, tp1, sl))

def detayli_performans_analizi():
    try:
        with sqlite3.connect("trade_pnl.db") as conn:
            df = pd.read_sql_query("SELECT * FROM islemler", conn)
        
        if df.empty:
            print("\n📭 Veritabanı boş, henüz işlem açılmadı.\n")
            return

        print("\n" + "="*60)
        print("📋 DETAYLI İŞLEM GEÇMİŞİ")
        print("="*60)
        ozet_df = df[['coin', 'yon', 'giris_fiyat', 'durum', 'pnl_yuzde', 'kapanis_zamani']]
        print(ozet_df.to_string(index=False))
        print("-" * 60)
        
        biten_islemler = df[df['durum'] != 'ACIK']
        if len(biten_islemler) > 0:
            kazanan = len(biten_islemler[biten_islemler['durum'] == 'KAZANDI'])
            win_rate = (kazanan / len(biten_islemler)) * 100
            toplam_pnl = biten_islemler['pnl_yuzde'].sum()
            print(f"📊 İSTATİSTİKLER: Win Rate: %{win_rate:.2f} | Net PnL: %{toplam_pnl:.2f}")
        else:
            print("📊 Henüz sonuçlanmış işlem yok.")
        print("="*60 + "\n")
    except Exception as e:
        print(f"Rapor Hatası: {e}")

# 🚀 GÜNCELLEME: İşlem kapandığında detaylı rapor atan fonksiyon
async def islemleri_kontrol_et(exchange):
    with sqlite3.connect("trade_pnl.db") as conn:
        c = conn.cursor()
        c.execute("SELECT id, coin, yon, giris_fiyat, tp1, sl FROM islemler WHERE durum='ACIK'")
        acik_islemler = c.fetchall()
    
    if not acik_islemler: return

    for islem in acik_islemler:
        id, coin, yon, giris, tp1, sl = islem
        try:
            ticker = await exchange.fetch_ticker(f"{coin}/USDT") 
            fiyat = ticker['last']
            sonuc, pnl = None, 0
            sebep = ""

            if yon == "LONG":
                if fiyat >= tp1: 
                    sonuc, pnl = "KAZANDI", ((tp1-giris)/giris)*100
                    sebep = "TP1 Hedefi 🎯"
                elif fiyat <= sl: 
                    sonuc, pnl = "KAYBETTI", ((sl-giris)/giris)*100
                    sebep = "Stop Loss 🛑"
            elif yon == "SHORT":
                if fiyat <= tp1: 
                    sonuc, pnl = "KAZANDI", ((giris-tp1)/giris)*100
                    sebep = "TP1 Hedefi 🎯"
                elif fiyat >= sl: 
                    sonuc, pnl = "KAYBETTI", ((giris-sl)/giris)*100
                    sebep = "Stop Loss 🛑"

            if sonuc:
                with sqlite3.connect("trade_pnl.db") as conn:
                    conn.execute("UPDATE islemler SET durum=?, pnl_yuzde=?, kapanis_zamani=? WHERE id=?", 
                              (sonuc, pnl, datetime.now(), id))
                
                # Şık Bildirim Tasarımı
                ikon = "✅" if sonuc == "KAZANDI" else "❌"
                renk = "🟢" if sonuc == "KAZANDI" else "🔴"
                p_fmt = ".8f" if fiyat < 0.01 else ".4f"

                mesaj = f"""
🏁 <b>POZİSYON KAPANDI</b> {ikon}

🪙 <b>Coin:</b> #{coin}
📊 <b>Yön:</b> {yon} {renk}
🏷️ <b>Durum:</b> {sonuc} ({sebep})

💰 <b>Giriş:</b> ${giris:{p_fmt}}
🚪 <b>Çıkış:</b> ${fiyat:{p_fmt}}
📉 <b>Kâr/Zarar:</b> %{pnl:.2f}

🤖 <i>Otomatik Takip Sistemi</i>
"""
                await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                detayli_performans_analizi()
        except: continue

# ==========================================
# 🚀 BÖLÜM 5: TEKNİK ANALİZ (ASENKRON & PARALEL)
# ==========================================

async def get_ohlcv_safe(exchange, symbol):
    try:
        return symbol, await exchange.fetch_ohlcv(symbol, timeframe='1h', limit=300)
    except Exception as e:
        print(f"Veri çekme hatası ({symbol}): {e}")
        return symbol, None

async def piyasayi_tarama(exchange):
    print(f"🔍 ({datetime.now().strftime('%H:%M')}) TEKNİK TARAMA (PARALEL)...")
    su_an = datetime.now()

    # 1. BTC Verisini Çek
    btc_trend = "NEUTRAL"
    try:
        btc_bars = await exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=250)
        btc_df = pd.DataFrame(btc_bars, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        btc_ema200 = calculate_ema(btc_df['close'], 200).iloc[-1]
        btc_price = btc_df['close'].iloc[-1]
        btc_trend = "BULL" if btc_price > btc_ema200 else "BEAR"
        print(f"🦁 BTC YÖN: {btc_trend} (Fiyat: {btc_price:.0f})")
    except Exception as e:
        print(f"⚠️ BTC Analiz Hatası: {e}")

    # 2. Tüm Coinleri Çek
    tasks = [get_ohlcv_safe(exchange, f"{coin}/USDT") for coin in COIN_LIST]
    results = await asyncio.gather(*tasks)

    # 3. Sonuçları İşle
    for symbol_pair, bars in results:
        coin = symbol_pair.split('/')[0]
        if coin in SON_SINYAL_ZAMANI:
            if (su_an - SON_SINYAL_ZAMANI[coin]) < timedelta(hours=2): continue 
        
        if not bars or len(bars) < 250: continue

        try:
            df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'], unit='ms')
            df.set_index('date', inplace=True)

            df['ema200'] = calculate_ema(df['close'], 200) 
            df['rsi'] = calculate_rsi(df['close'])          
            df['macd'], df['signal'] = calculate_macd(df['close']) 
            df['adx'] = calculate_adx(df)                   
            df['atr'] = calculate_atr(df)
            df['vol_ma'] = df['volume'].rolling(window=20).mean()
            pivot, r1, s1 = calculate_pivots(df)

            curr = df.iloc[-1]
            prev = df.iloc[-2]
            fiyat = curr['close']
            atr = curr['atr']
            
            sinyal = None
            setup_reason = ""
            hacim_teyidi = curr['volume'] > curr['vol_ma']
            dirence_yakinlik = (r1 - fiyat) / fiyat
            destege_yakinlik = (fiyat - s1) / fiyat
            tp1, tp2, tp3, stop_loss = 0,0,0,0

            # Strateji
            if (fiyat > curr['ema200']) and (curr['adx'] > 20):
                if dirence_yakinlik > 0.005: 
                    macd_cross = (prev['macd'] < prev['signal']) and (curr['macd'] > curr['signal'])
                    rsi_bounce = (prev['rsi'] < 40) and (curr['rsi'] > 40)
                    if (macd_cross or rsi_bounce) and hacim_teyidi:
                        sinyal = "LONG 🟢"
                        setup_reason = "Trend + Hacim + Pivot Onayı"
                        stop_loss = fiyat - (atr * 2.0)
                        tp1 = fiyat + (atr * 1.5)
                        tp2 = fiyat + (atr * 3.0)
                        tp3 = fiyat + (atr * 6.0)

            elif (fiyat < curr['ema200']) and (curr['adx'] > 15):
                if destege_yakinlik > 0.005:
                    macd_cross = (prev['macd'] > prev['signal']) and (curr['macd'] < curr['signal'])
                    rsi_dump = (curr['rsi'] < 55) and (curr['rsi'] < prev['rsi'])
                    if (macd_cross or rsi_dump) and hacim_teyidi:
                        sinyal = "SHORT 🔴"
                        setup_reason = "Baskı + Hacim + Pivot Onayı"
                        stop_loss = fiyat + (atr * 2.0)
                        tp1 = fiyat - (atr * 1.5)
                        tp2 = fiyat - (atr * 3.0)
                        tp3 = fiyat - (atr * 6.0)

            # BTC Filtresi
            if sinyal:
                if "LONG" in sinyal and btc_trend == "BEAR":
                    print(f"🚫 {coin} LONG iptal (BTC Bear)")
                    sinyal = None
                elif "SHORT" in sinyal and btc_trend == "BULL":
                    print(f"⚠️ {coin} SHORT (BTC Bull kuralı es geçildi)")

            if sinyal:
                SON_SINYAL_ZAMANI[coin] = su_an
                yon_str = "LONG" if "LONG" in sinyal else "SHORT"
                islem_kaydet(coin, yon_str, fiyat, tp1, stop_loss)
                print(f"🎯 Sinyal: {coin} -> {sinyal}")
                
                resim = await grafik_olustur_async(coin, df.tail(80), tp1, tp2, tp3, stop_loss, pivot, r1, s1)
                p_fmt = ".8f" if fiyat < 0.01 else ".4f"
                mesaj = f"""
⚡ <b>QUANT VIP SİNYAL</b>
🪙 <b>#{coin}</b>
📊 <b>Yön:</b> {sinyal}
📉 <b>Sebep:</b> {setup_reason}

💰 <b>Giriş:</b> ${fiyat:{p_fmt}}
🎯 <b>HEDEFLER</b>
1️⃣ <b>TP1:</b> ${tp1:{p_fmt}}
2️⃣ <b>TP2:</b> ${tp2:{p_fmt}}
3️⃣ <b>TP3:</b> ${tp3:{p_fmt}}
🛑 <b>Stop Loss:</b> ${stop_loss:{p_fmt}}

🦁 <b>Piyasa:</b> BTC {btc_trend}
🧱 <b>Pivot:</b> R1: ${r1:{p_fmt}} | S1: ${s1:{p_fmt}}
"""
                if resim:
                    await bot.send_photo(chat_id=KANAL_ID, photo=resim, caption=mesaj, parse_mode=ParseMode.HTML)
                else:
                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                await asyncio.sleep(1)

        except Exception as e:
            print(f"İşlem Hatası ({coin}): {e}")
            continue

# ==========================================
# 🏁 MAIN
# ==========================================
async def main():
    db_baslat()
    pnl_db_baslat()
    global RAPOR_ZAMANI
    
    exchange = ccxt.kucoin(exchange_config)
    print("🚀 Bot Tamamen Aktif! (TURBO ASYNC MOD v2)")
    detayli_performans_analizi()
    
    sayac = 0
    try:
        while True:
            await haberleri_kontrol_et()
            await piyasayi_tarama(exchange)
            await islemleri_kontrol_et(exchange)
            
            if (datetime.now() - RAPOR_ZAMANI) > timedelta(hours=24):
                detayli_performans_analizi()
                RAPOR_ZAMANI = datetime.now()
            
            sayac += 1
            print(f"💤 Bekleme... (Döngü: {sayac})")
            await asyncio.sleep(180)
    except KeyboardInterrupt:
        print("\n🛑 Bot Durduruluyor...")
    finally:
        await exchange.close()
        print("🔌 Bağlantılar kapatıldı.")

if __name__ == "__main__":
    asyncio.run(main()) 
