import feedparser
import asyncio
import os
import sys
import sqlite3
import time
import re
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
import mplfinance as mpf
import io
from datetime import datetime, timedelta
from google import genai
from telegram import Bot
from telegram.constants import ParseMode

print("⚙️ TITANIUM PREMIUM BOT (V3: TREND + AI + MACRO) BAŞLATILIYOR...")

# ==========================================
# 🔧 AYARLAR
# ==========================================
TOKEN = os.getenv("BOT_TOKEN", "").strip()
KANAL_ID = int(os.getenv("KANAL_ID", "0"))
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

if not TOKEN or not GEMINI_KEY or not KANAL_ID:
    print("❌ HATA: ENV bilgileri eksik! (BOT_TOKEN, KANAL_ID, GEMINI_KEY)")
    # sys.exit(1) 

# Gemini Client
try:
    client = genai.Client(api_key=GEMINI_KEY, http_options={"api_version": "v1"})
except:
    print("⚠️ Gemini Client başlatılamadı.")

bot = Bot(token=TOKEN)

exchange_config = {
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'} 
}

# TITANIUM COIN LISTESI
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
SON_RAPOR_TARIHI = None 

# ==========================================
# 🧮 BÖLÜM 1: İNDİKATÖRLER (GÜNCELLENDİ)
# ==========================================
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_sma(series, window):
    return series.rolling(window=window).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- YENİ EKLENEN: ADX (TREND GÜCÜ) ---
def calculate_adx(df, period=14):
    """
    ADX Hesaplar. 
    ADX > 20-25 ise Trend var demektir.
    ADX < 20 ise Piyasa yataydır.
    """
    df = df.copy()
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    
    df['atr'] = df['tr'].rolling(period).mean()
    
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    df['plus_di'] = 100 * (df['plus_dm'].ewm(alpha=1/period).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].ewm(alpha=1/period).mean() / df['atr'])
    
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].ewm(alpha=1/period).mean()
    
    return df['adx']

# ==========================================
# 🎨 BÖLÜM 2: GRAFİK
# ==========================================
def _grafik_olustur_sync(coin, df_gelen, tp, sl, yon):
    try:
        df = df_gelen.copy()
        apds = [
            mpf.make_addplot(df['sma50'], panel=0, color='cyan', width=0.8),
            mpf.make_addplot(df['sma200'], panel=0, color='white', width=1.0),
            mpf.make_addplot(df['rsi'], panel=1, color='#FF6D00', width=1.0, title="RSI")
        ]
        
        # RSI Referans Cizgileri
        h_lines_rsi = dict(hlines=[30, 80], colors=['green', 'red'], linewidths=[0.5, 0.5], linestyle='--')
        
        buf = io.BytesIO()
        theme_color = '#131722'
        grid_color = '#363c4e'
        text_color = '#b2b5be'
        my_style = mpf.make_mpf_style(
            base_mpf_style='binance', facecolor=theme_color, figcolor=theme_color, edgecolor=theme_color,
            gridcolor=grid_color, gridstyle=':', rc={'axes.labelcolor': text_color, 'xtick.color': text_color, 'ytick.color': text_color, 'text.color': text_color}
        )
        
        h_lines = dict(
            hlines=[tp, sl], 
            colors=['#00FF00', '#FF0000'],
            linewidths=[1.5, 1.5], alpha=0.9, linestyle='-.'
        )
        
        title_color = "#00FF00" if yon == "LONG" else "#FF0000"
        
        mpf.plot(
            df, type='candle', style=my_style, title=f"\n{coin}/USDT - TITANIUM {yon}",
            ylabel='Fiyat ($)', ylabel_lower='RSI', addplot=apds, hlines=h_lines, volume=False,
            panel_ratios=(3, 1), savefig=dict(fname=buf, dpi=120, bbox_inches='tight', facecolor=theme_color)
        )
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Grafik Hatası: {e}")
        return None

async def grafik_olustur_async(coin, df, tp, sl, yon):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _grafik_olustur_sync, coin, df, tp, sl, yon)

# ==========================================
# 📊 BÖLÜM 3: DB & SİNYAL YÖNETİMİ
# ==========================================
def db_ilk_kurulum():
    with sqlite3.connect("titanium_live.db") as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS islemler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT, yon TEXT, giris_fiyat REAL, tp REAL, sl REAL,
            durum TEXT DEFAULT 'ACIK', pnl_yuzde REAL DEFAULT 0,
            acilis_zamani DATETIME, kapanis_zamani DATETIME
        )""")
        conn.execute("CREATE TABLE IF NOT EXISTS haberler (link TEXT PRIMARY KEY)")

def short_var_mi(coin):
    with sqlite3.connect("titanium_live.db") as conn:
        c = conn.cursor()
        c.execute("SELECT count(*) FROM islemler WHERE coin=? AND yon='SHORT' AND durum='ACIK'", (coin,))
        count = c.fetchone()[0]
        return count > 0

def islem_kaydet(coin, yon, giris, tp, sl):
    with sqlite3.connect("titanium_live.db") as conn:
        conn.execute("INSERT INTO islemler (coin, yon, giris_fiyat, tp, sl, acilis_zamani) VALUES (?, ?, ?, ?, ?, ?)", 
                  (coin, yon, giris, tp, sl, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

async def gunluk_rapor_gonder():
    try:
        bugun = datetime.now().strftime("%Y-%m-%d")
        print(f"📊 {bugun} Günlük Rapor Hazırlanıyor...")

        with sqlite3.connect("titanium_live.db") as conn:
            query = """
            SELECT coin, yon, durum, pnl_yuzde, kapanis_zamani 
            FROM islemler 
            WHERE durum IN ('KAZANDI', 'KAYBETTI') 
            AND date(kapanis_zamani) = ?
            """
            df_rapor = pd.read_sql_query(query, conn, params=(bugun,))

        if df_rapor.empty:
            print("ℹ️ Bugün kapanan işlem yok.")
            return

        toplam_pnl = df_rapor['pnl_yuzde'].sum()
        win_count = len(df_rapor[df_rapor['durum'] == 'KAZANDI'])
        total_count = len(df_rapor)
        win_rate = (win_count / total_count) * 100 if total_count > 0 else 0
        
        pnl_ikon = "✅" if toplam_pnl > 0 else "🔻"
        
        mesaj = f"📅 <b>GÜNLÜK PERFORMANS RAPORU ({bugun})</b>\n\n"
        mesaj += "<code>Coin   | Yön   | Sonuç   | PNL</code>\n"
        mesaj += "<code>-------|-------|---------|------</code>\n"
        
        for index, row in df_rapor.iterrows():
            coin_kisa = row['coin'][:4] 
            durum_ikon = "W" if row['durum'] == 'KAZANDI' else "L"
            pnl_val = row['pnl_yuzde']
            mesaj += f"<code>{coin_kisa:<6} | {row['yon']:<5} | {durum_ikon:<7} | %{pnl_val:.1f}</code>\n"
            
        mesaj += "\n" + "-"*25 + "\n"
        mesaj += f"🔢 <b>Toplam İşlem:</b> {total_count}\n"
        mesaj += f"🎯 <b>Başarı Oranı:</b> %{win_rate:.1f}\n"
        mesaj += f"💰 <b>GÜNLÜK NET PNL:</b> {pnl_ikon} <b>%{toplam_pnl:.2f}</b>\n"
        mesaj += "\n🤖 <i>Titanium Premium Bot</i>"

        await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
        print("✅ Günlük rapor Telegram'a iletildi.")

    except Exception as e:
        print(f"❌ Günlük Rapor Hatası: {e}")

# ==========================================
# 🧠 BÖLÜM 4: AI HABER ANALİZİ
# ==========================================
def link_kontrol(link):
    with sqlite3.connect("titanium_live.db") as conn:
        c = conn.cursor()
        try:
            c.execute("INSERT INTO haberler VALUES (?)", (link,))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

def _ai_analiz_sync(prompt):
    try:
        r = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        text = r.text.strip()
        ozet_match = re.search(r"ÖZET:(.*)", text, re.DOTALL)
        skor_match = re.search(r"SKOR:\s*(-?\d)", text)
        temiz_ozet = ozet_match.group(1).strip() if ozet_match else "Özet oluşturulamadı."
        skor = int(skor_match.group(1)) if skor_match else 0
        return temiz_ozet, skor
    except:
        return "Analiz yapılamadı.", 0

async def ai_analiz(baslik, ozet):
    prompt = f"""
    GÖREV: Aşağıdaki kripto haberini analiz et.
    HABER BAŞLIĞI: {baslik}
    HABER ÖZETİ: {ozet}
    
    KURALLAR:
    1. Asla "Tamam", "Anlaşıldı" deme.
    2. Çıktı formatına %100 sadık kal.
    3. Skor -2 (Çok Kötü) ile +2 (Çok İyi) arasında tam sayı olsun.

    İSTENEN ÇIKTI FORMATI:
    ÖZET:[Tek bir emoji ile başlayan maksimum 2 cümlelik özet]
    SKOR:[Sadece Sayı]
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _ai_analiz_sync, prompt)

async def haberleri_kontrol_et():
    print("📰 Haberler taranıyor (AI Analiz)...")
    for rss in RSS_LIST:
        try:
            feed = feedparser.parse(rss)
            for entry in feed.entries[:2]:
                if not link_kontrol(entry.link): continue 
                if entry.published_parsed:
                    t = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    if (datetime.now() - t) > timedelta(minutes=45): continue
                
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
                try:
                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                except Exception as e:
                    print(f"Haber Telegram Hatası: {e}")
                await asyncio.sleep(2)
        except Exception as e:
            print(f"RSS Hatası: {e}")

# ==========================================
# 🚀 BÖLÜM 5: STRATEJİ MOTORU (GÜNCELLENDİ)
# ==========================================

# --- YENİ EKLENEN: BTC TREND KONTROLÜ ---
async def btc_trend_kontrol(exchange):
    """
    Genel Piyasa Yönünü (BTC) belirler.
    BTC SMA50 > SMA200 ise BULL, değilse BEAR döner.
    """
    try:
        # BTC verisini çek
        ohlcv = await exchange.fetch_ohlcv("BTC/USDT", '1h', limit=210)
        if not ohlcv: return "NEUTRAL"
        
        df = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        sma50 = df['close'].rolling(50).mean().iloc[-1]
        sma200 = df['close'].rolling(200).mean().iloc[-1]
        
        return "BULL" if sma50 > sma200 else "BEAR"
    except Exception as e:
        print(f"⚠️ BTC Trend Kontrol Hatası: {e}")
        return "NEUTRAL"

async def piyasayi_tarama(exchange):
    print(f"🔍 ({datetime.now().strftime('%H:%M')}) TITANIUM HYBRID V3 SCAN...")
    
    # 1. ÖNCE GENEL PİYASA YÖNÜNE BAK (BTC MASTER FILTER)
    genel_trend = await btc_trend_kontrol(exchange)
    print(f"🌍 GENEL PİYASA (BTC): {genel_trend}")
    
    # 2. COIN VERILERINI CEK
    async def fetch_candle(s):
        try:
            ohlcv = await exchange.fetch_ohlcv(f"{s}/USDT", '1h', limit=300)
            return s, ohlcv
        except Exception as e:
            print(f"❌ Veri Hatası ({s}): {e}")
            return s, None

    tasks = [fetch_candle(c) for c in COIN_LIST]
    results = await asyncio.gather(*tasks)
    
    for coin, bars in results:
        if not bars: continue
        
        df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df.set_index('date', inplace=True)
        
        # --- İNDİKATÖRLER ---
        df['sma50'] = calculate_sma(df['close'], 50)
        df['sma200'] = calculate_sma(df['close'], 200)
        df['rsi'] = calculate_rsi(df['close'])
        
        # 🔥 YENİ: ADX Hesapla (Trend Gücü)
        df['adx'] = calculate_adx(df)
        
        curr = df.iloc[-1]
        price = curr['close']
        rsi_val = curr['rsi']
        adx_val = curr['adx'] # Yeni ADX değeri
        
        # --- STRATEJİ MANTIKLARI ---
        
        # 1. Coin Kendi Trendi
        coin_bullish = curr['sma50'] > curr['sma200']
        coin_bearish = curr['sma50'] < curr['sma200']
        rally_mode = price > curr['sma50']
        
        # 2. Trend Güçlü Mü? (ADX Filtresi)
        # ADX 25'in altındaysa piyasa yataydır, işlem açma.
        trend_guclu = adx_val > 25
        
        # Debug Logu
        # print(f"👀 {coin}: RSI={rsi_val:.1f} | ADX={adx_val:.1f} | BTC={genel_trend}")
        
        sinyal = None
        setup = ""
        tp_rate = 0.0
        sl_rate = 0.0
        
        # --- STRATEJİ 1: LONG (MACRO DESTEKLİ) ---
        # Şartlar:
        # 1. BTC Bullish OLACAK (Piyasa arkanda)
        # 2. Coin Bullish OLACAK (SMA50 > SMA200)
        # 3. Trend Güçlü OLACAK (ADX > 25)
        # 4. Pullback Fırsatı (Fiyat > SMA200 ama RSI < 35 DİPTE)
        
        if (genel_trend == "BULL") and coin_bullish and trend_guclu and (price > curr['sma200']) and (rsi_val < 35):
            
            if coin in SON_SINYAL_ZAMANI and (datetime.now() - SON_SINYAL_ZAMANI[coin]) < timedelta(hours=2):
                pass
            else:
                sinyal = "LONG"
                setup = "Trend Pullback (BTC & ADX Onaylı)"
                tp_rate = 0.030 
                sl_rate = 0.060 
        
        # --- STRATEJİ 2: SHORT (MACRO DESTEKLİ) ---
        # Şartlar:
        # 1. BTC Bearish OLACAK (Piyasa düşüyor)
        # 2. Coin Bearish OLACAK
        # 3. Trend Güçlü OLACAK (ADX > 25)
        # 4. RSI > 75 (Tepede, Şişkin)
        
        elif (genel_trend == "BEAR") and coin_bearish and rally_mode and trend_guclu and (rsi_val > 75):
            
            if short_var_mi(coin):
                pass # Zaten short varsa açma
            else:
                 if coin in SON_SINYAL_ZAMANI and (datetime.now() - SON_SINYAL_ZAMANI[coin]) < timedelta(hours=2):
                    pass
                 else:
                    sinyal = "SHORT"
                    setup = "Trend Reversal (BTC & ADX Onaylı)"
                    tp_rate = 0.035 
                    sl_rate = 0.060 
        
        # SINYAL GÖNDERİMİ
        if sinyal:
            tp_price = price * (1 + tp_rate) if sinyal == "LONG" else price * (1 - tp_rate)
            sl_price = price * (1 - sl_rate) if sinyal == "LONG" else price * (1 + sl_rate)
            
            p_fmt = ".8f" if price < 0.01 else ".4f"
            
            islem_kaydet(coin, sinyal, price, tp_price, sl_price)
            SON_SINYAL_ZAMANI[coin] = datetime.now()
            
            print(f"🎯 {sinyal} SINYALI: {coin} (RSI: {rsi_val:.1f} | ADX: {adx_val:.1f})")
            
            resim = await grafik_olustur_async(coin, df.tail(100), tp_price, sl_price, sinyal)
            
            ikon = "🟢" if sinyal == "LONG" else "🔴"
            mesaj = f"""
{ikon} <b>TITANIUM SİNYAL ({sinyal})</b> #V3

🪙 <b>Coin:</b> #{coin}
📉 <b>Setup:</b> {setup}
📊 <b>RSI:</b> {rsi_val:.1f} | <b>ADX:</b> {adx_val:.1f}

💰 <b>Giriş:</b> ${price:{p_fmt}}
🎯 <b>HEDEF (TP):</b> ${tp_price:{p_fmt}}
🛑 <b>STOP (SL):</b> ${sl_price:{p_fmt}}

🌍 <b>Piyasa Yönü:</b> {genel_trend}
⚠️ <i>Limit Emir Kullanın!</i>
""" 
            try:
                if resim:
                    await bot.send_photo(chat_id=KANAL_ID, photo=resim, caption=mesaj, parse_mode=ParseMode.HTML)
                else:
                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
            except Exception as e:
                print(f"Telegram Hatasi: {e}")

# ==========================================
# 🛡️ BÖLÜM 6: POZİSYON TAKİBİ
# ==========================================
async def pozisyonlari_yokla(exchange):
    with sqlite3.connect("titanium_live.db") as conn:
        c = conn.cursor()
        c.execute("SELECT id, coin, yon, giris_fiyat, tp, sl FROM islemler WHERE durum='ACIK'")
        acik_islemler = c.fetchall()
        
    if not acik_islemler: return

    for islem in acik_islemler:
        id, coin, yon, giris, tp, sl = islem
        try:
            ticker = await exchange.fetch_ticker(f"{coin}/USDT")
            fiyat = ticker['last']
            sonuc = None
            pnl_yuzde = 0.0
            
            if yon == "LONG":
                if fiyat >= tp:
                    sonuc = "KAZANDI"
                    pnl_yuzde = 3.0
                elif fiyat <= sl:
                    sonuc = "KAYBETTI"
                    pnl_yuzde = -6.0
            elif yon == "SHORT":
                if fiyat <= tp:
                    sonuc = "KAZANDI"
                    pnl_yuzde = 3.5
                elif fiyat >= sl:
                    sonuc = "KAYBETTI"
                    pnl_yuzde = -6.0
            
            if sonuc:
                with sqlite3.connect("titanium_live.db") as conn:
                    conn.execute("UPDATE islemler SET durum=?, pnl_yuzde=?, kapanis_zamani=? WHERE id=?", 
                              (sonuc, pnl_yuzde, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), id))
                
                ikon = "✅" if sonuc == "KAZANDI" else "❌"
                p_fmt = ".8f" if fiyat < 0.01 else ".4f"
                
                mesaj = f"""
🏁 <b>POZİSYON KAPANDI</b> {ikon}

🪙 <b>#{coin}</b> ({yon})
🏷️ <b>Sonuç:</b> {sonuc}

💰 <b>Giriş:</b> ${giris:{p_fmt}}
🚪 <b>Çıkış:</b> ${fiyat:{p_fmt}}
📉 <b>Kâr/Zarar:</b> %{pnl_yuzde}

🤖 <i>Titanium Bot</i>
"""
                await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
        except: continue

# ==========================================
# 🏁 MAIN LOOP
# ==========================================
async def main():
    global SON_RAPOR_TARIHI
    db_ilk_kurulum()
    print("🚀 Titanium PREMIUM Bot Aktif! (Telegram: Sinyal + Haber + Macro)")
    
    exchange = ccxt.kucoin(exchange_config)
    
    try:
        await bot.send_message(chat_id=KANAL_ID, text="🚀 **TITANIUM BOT V3 BAŞLATILDI!**\n\n✅ Sistem: Aktif\n✅ Filtreler: BTC Trend + ADX (25)\n✅ Borsa: KuCoin\n📊 Raporlama: Aktif", parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        print(f"❌ Telegram Test Mesajı Hatası: {e}")

    if "ETH" in COIN_LIST:
        COIN_LIST.remove("ETH")

    try:
        while True:
            simdi = datetime.now()
            bugun_str = simdi.strftime("%Y-%m-%d")
            
            if simdi.hour == 23 and simdi.minute >= 55 and SON_RAPOR_TARIHI != bugun_str:
                await gunluk_rapor_gonder()
                SON_RAPOR_TARIHI = bugun_str
            
            await haberleri_kontrol_et()
            await piyasayi_tarama(exchange)
            await pozisyonlari_yokla(exchange)
            
            print("💤 Bekleme (1dk)...")
            await asyncio.sleep(60) 
    except KeyboardInterrupt:
        print("🛑 Bot Durduruluyor...")
    finally:
        await exchange.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
