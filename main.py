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

print("⚙️ ULTRA QUANT PIVOT MASTER BOT BAŞLATILIYOR...")

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

exchange = ccxt.kucoin({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'} 
})

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

# 🕒 HAFIZA
SON_SINYAL_ZAMANI = {}

# ==========================================
# 🧮 BÖLÜM 1: İNDİKATÖRLER VE PIVOTLAR
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
    
    # Güvenli atama
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
        df_daily = df_hourly.resample('D').agg({
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        last_day = df_daily.iloc[-2]
        high = last_day['high']
        low = last_day['low']
        close = last_day['close']
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low  
        s1 = (2 * pivot) - high 
        return pivot, r1, s1
    except:
        return 0, 0, 0

# ==========================================
# 🎨 BÖLÜM 2: GRAFİK OLUŞTURUCU
# ==========================================

def grafik_olustur(coin, df_gelen, tp1, tp2, tp3, sl_price, pivot, r1, s1):
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
            base_mpf_style='binance',
            facecolor=theme_color,
            figcolor=theme_color,
            edgecolor=theme_color,
            gridcolor=grid_color,
            gridstyle=':',
            rc={'axes.labelcolor': text_color, 'xtick.color': text_color, 'ytick.color': text_color, 'text.color': text_color}
        )
        h_lines = dict(
            hlines=[tp1, tp2, tp3, sl_price, pivot, r1, s1], 
            colors=['#98FB98', '#32CD32', '#006400', '#FF0000', '#FFFF00', '#FF4500', '#00BFFF'],
            linewidths=[1.0, 1.2, 1.5, 1.5, 0.8, 0.8, 0.8], 
            alpha=0.8, 
            linestyle='-.'
        )
        mpf.plot(
            df, type='candle', style=my_style,
            title=f"\n{coin}/USDT - Pivot & TP Analiz",
            ylabel='Fiyat ($)', ylabel_lower='MACD',
            addplot=apds, hlines=h_lines, volume=False,
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
    try:
        c.execute("INSERT INTO gonderilenler VALUES (?)", (link,))
        conn.commit()
        yeni_mi = True
    except sqlite3.IntegrityError:
        yeni_mi = False
    conn.close()
    return yeni_mi

async def ai_analiz(baslik, ozet):
    prompt = f"Analist sensin. Haberi yorumla.\nHABER: {baslik}\n{ozet}\nÇıktı Formatı:\n🔥 Özet: [Kısa cümle]\n🎯 Skor: [ -2 (Çok Kötü) ile 2 (Çok İyi) arası tam sayı]"
    try:
        r = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        text = r.text.strip()
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
                if not link_kontrol(entry.link): continue 
                if entry.published_parsed:
                    t = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    if (datetime.now() - t) > timedelta(minutes=30): continue
                ai_text, skor = await ai_analiz(entry.title, entry.get("summary", "")[:300])
                if abs(skor) < 2: continue 
                skor_icon = "🟢" if skor > 0 else "🔴" if skor < 0 else "⚖️"
                mesaj = f"📰 <b>{entry.title}</b>\n\n{ai_text}\n\n📊 <b>Etki:</b> {skor} {skor_icon}\n🔗 <a href='{entry.link}'>Haberi Oku</a>"
                await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                await asyncio.sleep(5)
        except Exception as e:
            print(f"RSS Hatası: {e}")

# ==========================================
# 📊 BÖLÜM 4: VERİTABANI VE RAPORLAMA MODÜLÜ
# ==========================================
RAPOR_ZAMANI = datetime.now()

def pnl_db_baslat():
    conn = sqlite3.connect("trade_pnl.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS islemler (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        coin TEXT, yon TEXT, giris_fiyat REAL, tp1 REAL, sl REAL,
        durum TEXT DEFAULT 'ACIK', pnl_yuzde REAL DEFAULT 0,
        kapanis_zamani DATETIME
    )""")
    conn.commit()
    conn.close()

def islem_kaydet(coin, yon, giris, tp1, sl):
    conn = sqlite3.connect("trade_pnl.db")
    c = conn.cursor()
    c.execute("INSERT INTO islemler (coin, yon, giris_fiyat, tp1, sl) VALUES (?, ?, ?, ?, ?)", 
              (coin, yon, giris, tp1, sl))
    conn.commit()
    conn.close()

def detayli_performans_analizi():
    conn = sqlite3.connect("trade_pnl.db")
    try:
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
        kazanan = len(biten_islemler[biten_islemler['durum'] == 'KAZANDI'])
        kaybeden = len(biten_islemler[biten_islemler['durum'] == 'KAYBETTI'])
        
        win_rate = (kazanan / len(biten_islemler)) * 100 if len(biten_islemler) > 0 else 0.0
        toplam_pnl = biten_islemler['pnl_yuzde'].sum()
        
        print(f"📊 İSTATİSTİKLER: Win Rate: %{win_rate:.2f} | Net PnL: %{toplam_pnl:.2f}")
        print("="*60 + "\n")
    except Exception as e:
        print(f"Raporlama hatası: {e}")
    finally:
        conn.close()

async def islemleri_kontrol_et():
    conn = sqlite3.connect("trade_pnl.db")
    c = conn.cursor()
    c.execute("SELECT id, coin, yon, giris_fiyat, tp1, sl FROM islemler WHERE durum='ACIK'")
    acik_islemler = c.fetchall()
    conn.close()
    
    if not acik_islemler: return

    for islem in acik_islemler:
        id, coin, yon, giris, tp1, sl = islem
        try:
            ticker = exchange.fetch_ticker(f"{coin}/USDT")
            fiyat = ticker['last']
            sonuc, pnl = None, 0

            if yon == "LONG":
                if fiyat >= tp1: sonuc, pnl = "KAZANDI", ((tp1-giris)/giris)*100
                elif fiyat <= sl: sonuc, pnl = "KAYBETTI", ((sl-giris)/giris)*100
            elif yon == "SHORT":
                if fiyat <= tp1: sonuc, pnl = "KAZANDI", ((giris-tp1)/giris)*100
                elif fiyat >= sl: sonuc, pnl = "KAYBETTI", ((giris-sl)/giris)*100

            if sonuc:
                conn = sqlite3.connect("trade_pnl.db")
                c = conn.cursor()
                c.execute("UPDATE islemler SET durum=?, pnl_yuzde=?, kapanis_zamani=? WHERE id=?", 
                          (sonuc, pnl, datetime.now(), id))
                conn.commit()
                conn.close()
                await bot.send_message(chat_id=KANAL_ID, text=f"{'✅' if sonuc=='KAZANDI' else '❌'} <b>İŞLEM SONUCU:</b> #{coin}\n<b>Durum:</b> {sonuc} (%{pnl:.2f})", parse_mode=ParseMode.HTML)
                detayli_performans_analizi()
        except: continue

# ==========================================
# 🚀 BÖLÜM 5: TEKNİK ANALİZ (BTC FİLTRELİ)
# ==========================================

async def piyasayi_tarama():
    print(f"🔍 ({datetime.now().strftime('%H:%M')}) TEKNİK TARAMA (BTC FİLTRELİ)...")
    su_an = datetime.now()

    # --- 🟢 YENİ: BTC GENEL PİYASA ANALİZİ ---
    btc_trend = "NEUTRAL"
    try:
        btc_bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=250)
        btc_df = pd.DataFrame(btc_bars, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        btc_ema200 = calculate_ema(btc_df['close'], 200).iloc[-1]
        btc_price = btc_df['close'].iloc[-1]
        
        # Piyasa Yönünü Belirle
        btc_trend = "BULL" if btc_price > btc_ema200 else "BEAR"
        print(f"🦁 BTC PİYASA YÖNÜ: {btc_trend} (Fiyat: {btc_price:.2f} | EMA200: {btc_ema200:.2f})")
    except Exception as e:
        print(f"⚠️ BTC Piyasa Analizi Başarısız: {e}")
    # ----------------------------------------

    for coin in COIN_LIST:
        symbol = f"{coin}/USDT"
        if coin in SON_SINYAL_ZAMANI:
            if (su_an - SON_SINYAL_ZAMANI[coin]) < timedelta(hours=2): continue 

        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=300)
            if not bars or len(bars) < 250: continue

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

            tp1, tp2, tp3 = 0, 0, 0
            stop_loss = 0

            dirence_yakinlik = (r1 - fiyat) / fiyat
            destege_yakinlik = (fiyat - s1) / fiyat
            
            # --- LONG SİNYAL MANTIĞI ---
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

            # --- SHORT SİNYAL MANTIĞI ---
            elif (fiyat < curr['ema200']) and (curr['adx'] > 20):
                if destege_yakinlik > 0.005:
                    macd_cross = (prev['macd'] > prev['signal']) and (curr['macd'] < curr['signal'])
                    rsi_dump = (prev['rsi'] > 60) and (curr['rsi'] < 60)
                    if (macd_cross or rsi_dump) and hacim_teyidi:
                        sinyal = "SHORT 🔴"
                        setup_reason = "Baskı + Hacim + Pivot Onayı"
                        stop_loss = fiyat + (atr * 2.0)
                        tp1 = fiyat - (atr * 1.5)
                        tp2 = fiyat - (atr * 3.0)
                        tp3 = fiyat - (atr * 6.0)

            # --- 🟢 YENİ: BTC FİLTRESİ UYGULAMA ---
            if sinyal:
                if "LONG" in sinyal and btc_trend == "BEAR":
                    print(f"🚫 {coin} LONG iptal edildi (BTC Düşüş Trendinde)")
                    sinyal = None
                elif "SHORT" in sinyal and btc_trend == "BULL":
                    print(f"🚫 {coin} SHORT iptal edildi (BTC Yükseliş Trendinde)")
                    sinyal = None
            # --------------------------------------

            if sinyal:
                SON_SINYAL_ZAMANI[coin] = su_an
                yon_str = "LONG" if "LONG" in sinyal else "SHORT"
                islem_kaydet(coin, yon_str, fiyat, tp1, stop_loss)
                print(f"🎯 Sinyal Onaylandı: {coin} -> {sinyal}")
                
                resim = grafik_olustur(coin, df.tail(80), tp1, tp2, tp3, stop_loss, pivot, r1, s1)
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

🦁 <b>Piyasa Durumu:</b> BTC {btc_trend}
🧱 <b>Pivot:</b> R1: ${r1:{p_fmt}} | S1: ${s1:{p_fmt}}
🧠 <i>ADX: {curr['adx']:.1f}</i>
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
# 🏁 MAIN
# ==========================================
async def main():
    db_baslat()
    pnl_db_baslat()
    global RAPOR_ZAMANI
    
    print("🚀 Bot Tamamen Aktif! (Haber + Multi-TP + Pivot + BTC Filtreli)")
    detayli_performans_analizi()
    
    sayac = 0
    while True:
        await haberleri_kontrol_et()
        await piyasayi_tarama()
        await islemleri_kontrol_et()
        
        if (datetime.now() - RAPOR_ZAMANI) > timedelta(hours=24):
             detayli_performans_analizi()
             RAPOR_ZAMANI = datetime.now()
        
        sayac += 1
        print(f"💤 Bekleme... (Döngü: {sayac})")
        await asyncio.sleep(180)

if __name__ == "__main__":
    asyncio.run(main())
