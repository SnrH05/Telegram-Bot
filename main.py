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

print("⚙️ TITANIUM STRATEGY BOT (V1 - LIVE - KUCOIN EDITION) BAŞLATILIYOR...")

# ==========================================
# 🔧 AYARLAR
# ==========================================
TOKEN = os.getenv("BOT_TOKEN", "").strip()
KANAL_ID = int(os.getenv("KANAL_ID", "0"))
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()

# Hata kontrolü (Opsiyonel: Eğer lokalde test ediyorsan burayı yorum satırı yapabilirsin)
if not TOKEN or not GEMINI_KEY or not KANAL_ID:
    print("⚠️ UYARI: ENV bilgileri eksik olabilir! (BOT_TOKEN, KANAL_ID, GEMINI_KEY)")

# Gemini Client
try:
    client = genai.Client(api_key=GEMINI_KEY, http_options={"api_version": "v1"})
except:
    client = None
    print("⚠️ Gemini Client başlatılamadı.")

bot = Bot(token=TOKEN)

exchange_config = {
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
}

# BACKTEST DOSYASINDAN GELEN LISTE
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

# TITANIUM V1 AYARLARI
KAR_HEDEFI_ORAN = 0.025  # %2.5
ZARAR_DURDUR_ORAN = 0.06 # %6.0

SON_SINYAL_ZAMANI = {}

# ==========================================
# 🧮 BÖLÜM 1: İNDİKATÖRLER
# ==========================================
def calculate_sma(series, window):
    return series.rolling(window=window).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ==========================================
# 🎨 BÖLÜM 2: GRAFİK
# ==========================================
def _grafik_olustur_sync(coin, df_gelen, tp_price, sl_price, signal_type):
    try:
        df = df_gelen.copy()
        
        apds = [
            mpf.make_addplot(df['sma50'], panel=0, color='#2962FF', width=1.0, linestyle='-'),
            mpf.make_addplot(df['sma200'], panel=0, color='#FF6D00', width=1.5, linestyle='--'),
            mpf.make_addplot(df['rsi'], panel=1, color='purple', width=1.0, ylabel='RSI')
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
            hlines=[tp_price, sl_price], 
            colors=['#00FF00', '#FF0000'],
            linewidths=[1.5, 1.5], alpha=0.9, linestyle='--'
        )
        
        title_str = f"\n{coin}/USDT - TITANIUM {signal_type}"
        
        mpf.plot(
            df, type='candle', style=my_style, title=title_str,
            ylabel='Fiyat ($)', ylabel_lower='RSI', addplot=apds, hlines=h_lines, volume=False,
            panel_ratios=(3, 1), savefig=dict(fname=buf, dpi=120, bbox_inches='tight', facecolor=theme_color)
        )
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Grafik Hatası: {e}")
        return None

async def grafik_olustur_async(coin, df, tp, sl, signal_type):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _grafik_olustur_sync, coin, df, tp, sl, signal_type)

# ==========================================
# 🧠 BÖLÜM 3: YAPAY ZEKA
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
    KURALLAR: 1. Çıktı formatına %100 sadık kal. 2. Skor -2 ile +2 arasında tam sayı olsun.
    İSTENEN ÇIKTI FORMATI:
    ÖZET:[Tek bir emoji ile başlayan maksimum 2 cümlelik özet]
    SKOR:[Sadece Sayı]
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _ai_analiz_sync, prompt)

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
                
                raw_summary = entry.get("summary", entry.get("description", ""))
                clean_text = re.sub('<[^<]+?>', '', raw_summary)
                
                ai_text, skor = await ai_analiz(entry.title, clean_text[:500])
                if abs(skor) < 2: continue 
                
                skor_icon = "🟢" if skor > 0 else "🔴"
                mesaj = f"""<b>{entry.title}</b>\n{ai_text}\n🎯 <b>Piyasa Etkisi:</b> {skor_icon} <b>({skor})</b>\n🔗 <a href='{entry.link}'>Kaynağa Git</a>"""
                
                try:
                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                except Exception as e:
                    print(f"❌ HABER TELEGRAM HATASI: {e}")
                await asyncio.sleep(2)
        except Exception as e:
            print(f"RSS Hatası: {e}")

# ==========================================
# 📊 BÖLÜM 4: RAPORLAMA VE DB
# ==========================================
RAPOR_ZAMANI = datetime.now()

def pnl_db_baslat():
    with sqlite3.connect("titanium_trades.db") as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS islemler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT, yon TEXT, giris_fiyat REAL, tp_price REAL, sl_price REAL,
            durum TEXT DEFAULT 'ACIK', pnl_yuzde REAL DEFAULT 0,
            acilis_zamani DATETIME DEFAULT CURRENT_TIMESTAMP,
            kapanis_zamani DATETIME
        )""")

def islem_kaydet(coin, yon, giris, tp, sl):
    with sqlite3.connect("titanium_trades.db") as conn:
        conn.execute("INSERT INTO islemler (coin, yon, giris_fiyat, tp_price, sl_price) VALUES (?, ?, ?, ?, ?)", 
                  (coin, yon, giris, tp, sl))

def detayli_performans_analizi():
    try:
        with sqlite3.connect("titanium_trades.db") as conn:
            df = pd.read_sql_query("SELECT * FROM islemler", conn)
        
        if df.empty:
            print("\n📭 Veritabanı boş, henüz işlem açılmadı.\n")
            return

        print("\n" + "="*60)
        print("📋 TITANIUM İŞLEM GEÇMİŞİ")
        print("="*60)
        ozet_df = df[['coin', 'yon', 'giris_fiyat', 'durum', 'pnl_yuzde']]
        print(ozet_df.tail(10).to_string(index=False)) 
        print("-" * 60)
        
        biten_islemler = df[df['durum'] != 'ACIK']
        if len(biten_islemler) > 0:
            kazanan = len(biten_islemler[biten_islemler['durum'] == 'KAZANDI'])
            win_rate = (kazanan / len(biten_islemler)) * 100
            toplam_pnl = biten_islemler['pnl_yuzde'].sum()
            print(f"📊 İSTATİSTİKLER: Win Rate: %{win_rate:.2f} | Net PnL (Kümülatif %): %{toplam_pnl:.2f}")
        else:
            print("📊 Henüz sonuçlanmış işlem yok.")
        print("="*60 + "\n")
    except Exception as e:
        print(f"Rapor Hatası: {e}")

async def islemleri_kontrol_et(exchange):
    with sqlite3.connect("titanium_trades.db") as conn:
        c = conn.cursor()
        c.execute("SELECT id, coin, yon, giris_fiyat, tp_price, sl_price FROM islemler WHERE durum='ACIK'")
        acik_islemler = c.fetchall()
    
    if not acik_islemler: return

    for islem in acik_islemler:
        id, coin, yon, giris, tp, sl = islem
        try:
            # KuCoin sembol formatı aynıdır (COIN/USDT)
            ticker = await exchange.fetch_ticker(f"{coin}/USDT") 
            fiyat = ticker['last']
            sonuc, pnl = None, 0
            sebep = ""

            if yon == "LONG":
                if fiyat >= tp: 
                    sonuc, pnl = "KAZANDI", ((tp-giris)/giris)*100
                    sebep = "TP Hedefi 🎯"
                elif fiyat <= sl: 
                    sonuc, pnl = "KAYBETTI", ((sl-giris)/giris)*100
                    sebep = "Stop Loss 🛑"
            elif yon == "SHORT":
                if fiyat <= tp: 
                    sonuc, pnl = "KAZANDI", ((giris-tp)/giris)*100
                    sebep = "TP Hedefi 🎯"
                elif fiyat >= sl: 
                    sonuc, pnl = "KAYBETTI", ((giris-sl)/giris)*100
                    sebep = "Stop Loss 🛑"

            if sonuc:
                with sqlite3.connect("titanium_trades.db") as conn:
                    conn.execute("UPDATE islemler SET durum=?, pnl_yuzde=?, kapanis_zamani=? WHERE id=?", 
                              (sonuc, pnl, datetime.now(), id))
                
                ikon = "✅" if sonuc == "KAZANDI" else "❌"
                renk = "🟢" if sonuc == "KAZANDI" else "🔴"
                p_fmt = ".8f" if fiyat < 0.01 else ".4f"

                mesaj = f"""
🏁 <b>TITANIUM POZİSYON KAPANDI</b> {ikon}

🪙 <b>Coin:</b> #{coin}
📊 <b>Yön:</b> {yon} {renk}
🏷️ <b>Durum:</b> {sonuc} ({sebep})

💰 <b>Giriş:</b> ${giris:{p_fmt}}
🚪 <b>Çıkış:</b> ${fiyat:{p_fmt}}
📉 <b>Kâr/Zarar:</b> %{pnl:.2f}
"""
                try:
                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                except Exception as e:
                    print(f"❌ KAPANIŞ MESAJI TELEGRAM HATASI: {e}")
                
                detayli_performans_analizi()
        except Exception as e: 
            print(f"İşlem Kontrol Hatası ({coin}): {e}")
            continue

# ==========================================
# 🚀 BÖLÜM 5: TEKNİK ANALİZ (GÜNCELLENDİ)
# ==========================================

async def get_ohlcv_safe(exchange, symbol):
    try:
        return symbol, await exchange.fetch_ohlcv(symbol, timeframe='1h', limit=250)
    except Exception as e:
        if '451' in str(e) or 'restricted' in str(e).lower(): 
            print(f"⚠️ {symbol} için erişim engeli (451 Restricted).")
        else: 
            print(f"⚠️ Veri çekme hatası ({symbol}): {e}")
        return symbol, None

async def piyasayi_tarama(exchange):
    print(f"🔍 ({datetime.now().strftime('%H:%M')}) TITANIUM TARAMA (KUCOIN - PARALEL)...")
    su_an = datetime.now()

    tasks = [get_ohlcv_safe(exchange, f"{coin}/USDT") for coin in COIN_LIST]
    results = await asyncio.gather(*tasks)

    for symbol_pair, bars in results:
        if not symbol_pair or not bars: 
            continue
        
        coin = symbol_pair.split('/')[0]
        
        # Spam kontrolü (4 saat)
        if coin in SON_SINYAL_ZAMANI:
            if (su_an - SON_SINYAL_ZAMANI[coin]) < timedelta(hours=4): continue 
        
        if len(bars) < 210: 
            # print(f"⚠️ {coin} için yetersiz veri: {len(bars)}")
            continue

        try:
            df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'], unit='ms')
            df.set_index('date', inplace=True)

            df['sma50'] = calculate_sma(df['close'], 50)
            df['sma200'] = calculate_sma(df['close'], 200)
            df['rsi'] = calculate_rsi(df['close'])
            
            curr = df.iloc[-1]
            price = curr['close']
            rsi_val = curr['rsi']
            
            trend_bullish = curr['sma50'] > curr['sma200']
            trend_bearish = curr['sma50'] < curr['sma200']
            
            # --- DEBUG LOGU ---
            # Bunu konsolda görüyorsan veri çekiliyor demektir
            yon_debug = "BULL" if trend_bullish else "BEAR"
            print(f"👀 {coin}: Fiyat={price:.2f} | RSI={rsi_val:.1f} | Trend={yon_debug}")
            
            # --- STRATEJİ (ESNETİLDİ) ---
            # 30/70 -> 35/65
            oversold = rsi_val < 35
            overbought = rsi_val > 65
            
            above_trend = price > curr['sma200']
            below_trend = price < curr['sma200']
            
            yon = None
            setup_reason = ""
            
            if trend_bullish and oversold and above_trend:
                yon = "LONG"
                setup_reason = "Bull Trend + RSI < 35 + Price > SMA200"
            elif trend_bearish and overbought and below_trend:
                yon = "SHORT"
                setup_reason = "Bear Trend + RSI > 65 + Price < SMA200"
            
            if yon:
                if yon == "LONG":
                    tp_price = price * (1 + KAR_HEDEFI_ORAN)
                    sl_price = price * (1 - ZARAR_DURDUR_ORAN)
                else: 
                    tp_price = price * (1 - KAR_HEDEFI_ORAN)
                    sl_price = price * (1 + ZARAR_DURDUR_ORAN)
                
                SON_SINYAL_ZAMANI[coin] = su_an
                islem_kaydet(coin, yon, price, tp_price, sl_price)
                print(f"🎯 Sinyal YAKALANDI: {coin} -> {yon}")
                
                resim = await grafik_olustur_async(coin, df.tail(100), tp_price, sl_price, yon)
                p_fmt = ".8f" if price < 0.01 else ".4f"
                
                mesaj = f"""
⚡ <b>TITANIUM V1 SİNYAL</b> (Live)
🪙 <b>#{coin}</b>
📊 <b>Yön:</b> {yon}
📉 <b>Sebep:</b> {setup_reason}

💰 <b>Fiyat:</b> ${price:{p_fmt}}
🎯 <b>Hedef:</b> ${tp_price:{p_fmt}} (%2.5)
🛑 <b>Stop:</b> ${sl_price:{p_fmt}} (%6.0)
ℹ️ <b>RSI:</b> {rsi_val:.1f}

🤖 <i>Auto-Trade System</i>
"""
                try:
                    if resim:
                        await bot.send_photo(chat_id=KANAL_ID, photo=resim, caption=mesaj, parse_mode=ParseMode.HTML)
                    else:
                        await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                except Exception as e:
                    print(f"❌ TELEGRAM SİNYAL HATASI: {e}")

        except Exception as e:
            print(f"⚠️ Analiz Hatası ({coin}): {e}")
            continue

# ==========================================
# 🏁 MAIN (KUCOIN AKTİF)
# ==========================================
async def main():
    db_baslat()
    pnl_db_baslat()
    global RAPOR_ZAMANI
    
    # ----------------------------------------
    # 🛠️ BORSA DEĞİŞİMİ YAPILDI: KuCoin
    # ----------------------------------------
    exchange = ccxt.kucoin(exchange_config)
    print("🚀 TITANIUM BOT Aktif! (Live Monitor - KuCoin)")
    detayli_performans_analizi()
    
    sayac = 0
    try:
        while True:
            # Haberleri şimdilik kapalı tutuyoruz, odak teknik analizde.
            # await haberleri_kontrol_et()
            
            await piyasayi_tarama(exchange)
            await islemleri_kontrol_et(exchange)
            
            if (datetime.now() - RAPOR_ZAMANI) > timedelta(hours=24):
                detayli_performans_analizi()
                RAPOR_ZAMANI = datetime.now()
            
            sayac += 1
            print(f"💤 Bekleme... (Döngü: {sayac})")
            await asyncio.sleep(60) 
    except KeyboardInterrupt:
        print("\n🛑 Bot Durduruluyor...")
    finally:
        await exchange.close()
        print("🔌 Bağlantılar kapatıldı.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
