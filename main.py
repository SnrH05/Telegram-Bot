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

print("⚙️ TITANIUM PREMIUM BOT (V5.4: SCORE+VOLUME+TRAILING) BAŞLATILIYOR...")

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
# 🧮 BÖLÜM 1: İNDİKATÖRLER
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

def calculate_atr(df, period=14):
    """Calculate Average True Range for dynamic stops"""
    df = df.copy()
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    return df['tr'].rolling(period).mean()

# ADX İndikatörü
def calculate_adx(df, period=14):
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
        # Islemler Tablosu (Multi-TP Support)
        conn.execute("""CREATE TABLE IF NOT EXISTS islemler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT, yon TEXT, giris_fiyat REAL, 
            tp1 REAL, tp2 REAL, tp3 REAL, sl REAL,
            tp1_hit INTEGER DEFAULT 0, tp2_hit INTEGER DEFAULT 0,
            durum TEXT DEFAULT 'ACIK', pnl_yuzde REAL DEFAULT 0,
            acilis_zamani DATETIME, kapanis_zamani DATETIME
        )""")
        
        # Migrate old table if tp2/tp3 columns missing
        try:
            conn.execute("ALTER TABLE islemler ADD COLUMN tp2 REAL")
            conn.execute("ALTER TABLE islemler ADD COLUMN tp3 REAL")
            conn.execute("ALTER TABLE islemler ADD COLUMN tp1_hit INTEGER DEFAULT 0")
            conn.execute("ALTER TABLE islemler ADD COLUMN tp2_hit INTEGER DEFAULT 0")
            # Rename old tp column to tp1
            conn.execute("UPDATE islemler SET tp1 = tp, tp2 = tp, tp3 = tp WHERE tp1 IS NULL") # Assuming 'tp' was the old column name
            conn.execute("ALTER TABLE islemler DROP COLUMN tp") # Drop the old single TP column
        except sqlite3.OperationalError as e:
            # This error occurs if the column already exists or if 'tp' column doesn't exist to drop
            if "duplicate column name" not in str(e) and "no such column" not in str(e):
                print(f"DB Migration Error: {e}")
        except Exception as e:
            print(f"Unexpected DB Migration Error: {e}")
        
        # Haberler Tablosu (Haber Hafızası)
        conn.execute("CREATE TABLE IF NOT EXISTS haberler (link TEXT PRIMARY KEY)")

def short_var_mi(coin):
    """Check if there's an open SHORT position for a coin"""
    with sqlite3.connect("titanium_live.db") as conn:
        c = conn.cursor()
        c.execute("SELECT count(*) FROM islemler WHERE coin=? AND yon='SHORT' AND durum='ACIK'", (coin,))
        count = c.fetchone()[0]
        return count > 0

def pozisyon_acik_mi(coin):
    """Check if there's ANY open position (LONG or SHORT) for a coin - Anti-Spam"""
    with sqlite3.connect("titanium_live.db") as conn:
        c = conn.cursor()
        c.execute("SELECT count(*) FROM islemler WHERE coin=? AND durum='ACIK'", (coin,))
        count = c.fetchone()[0]
        return count > 0

def islem_kaydet(coin, yon, giris, tp1, tp2, tp3, sl):
    """Save trade with multiple take profit levels"""
    with sqlite3.connect("titanium_live.db") as conn:
        conn.execute("INSERT INTO islemler (coin, yon, giris_fiyat, tp1, tp2, tp3, sl, acilis_zamani) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
                  (coin, yon, giris, tp1, tp2, tp3, sl, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

async def gunluk_rapor_gonder():
    try:
        bugun = datetime.now().strftime("%Y-%m-%d")
        print(f"📊 {bugun} Günlük Rapor Hazırlanıyor...")

        with sqlite3.connect("titanium_live.db") as conn:
            # TP hit bilgilerini de çek
            query = """
            SELECT coin, yon, durum, pnl_yuzde, tp1_hit, tp2_hit, kapanis_zamani 
            FROM islemler 
            WHERE durum IN ('KAZANDI', 'KAYBETTI', 'PARTIAL') 
            AND date(kapanis_zamani) = ?
            """
            df_rapor = pd.read_sql_query(query, conn, params=(bugun,))

        if df_rapor.empty:
            return

        toplam_pnl = df_rapor['pnl_yuzde'].sum()
        
        # Detaylı istatistikler
        full_win = len(df_rapor[df_rapor['durum'] == 'KAZANDI'])  # TP3 tam kazanç
        partial_win = len(df_rapor[df_rapor['durum'] == 'PARTIAL'])  # Kısmi kazanç (TP1/TP2 hit + SL)
        loss_count = len(df_rapor[df_rapor['durum'] == 'KAYBETTI'])  # Tam kayıp
        total_count = len(df_rapor)
        
        # Kazanç oranı (tam + kısmi kazanç)
        win_count = full_win + partial_win
        win_rate = (win_count / total_count) * 100 if total_count > 0 else 0
        pnl_ikon = "✅" if toplam_pnl > 0 else "🔻"
        
        mesaj = f"📅 <b>GÜNLÜK RAPOR ({bugun})</b>\n\n"
        
        for index, row in df_rapor.iterrows():
            # W veya L
            is_win = row['durum'] in ['KAZANDI', 'PARTIAL']
            wl = "W" if is_win else "L"
            
            # Hangi TP'ler vuruldu
            tp_list = []
            if row.get('tp1_hit', 0):
                tp_list.append("TP1")
            if row.get('tp2_hit', 0):
                tp_list.append("TP2")
            if row['durum'] == 'KAZANDI':
                tp_list.append("TP3")
            
            tp_str = ",".join(tp_list) if tp_list else "-"
            
            # PnL
            pnl_val = row['pnl_yuzde']
            pnl_str = f"+{pnl_val:.1f}" if pnl_val >= 0 else f"{pnl_val:.1f}"
            
            mesaj += f"<code>{row['coin'][:4]:<5}|{row['yon'][0]}|{wl}|{tp_str}|{pnl_str}%</code>\n"
        
        mesaj += f"\n📊 <b>İSTATİSTİKLER</b>\n"
        mesaj += f"🏆 <b>Tam Kazanç:</b> {full_win} | ⚡ <b>Kısmi:</b> {partial_win} | ❌ <b>Kayıp:</b> {loss_count}\n"
        mesaj += f"🔢 <b>Toplam:</b> {total_count} | 🎯 <b>WR:</b> %{win_rate:.0f}\n"
        mesaj += f"💰 <b>NET PNL:</b> {pnl_ikon} <b>%{toplam_pnl:.2f}</b>\n"
        
        # 💵 $100 SİMÜLASYONU
        yatirim_per_sinyal = 100  # Her sinyale $100
        toplam_yatirim = total_count * yatirim_per_sinyal
        toplam_kar = sum([(row['pnl_yuzde'] / 100) * yatirim_per_sinyal for _, row in df_rapor.iterrows()])
        final_bakiye = toplam_yatirim + toplam_kar
        kar_ikon = "📈" if toplam_kar >= 0 else "📉"
        
        mesaj += f"\n💵 <b>$100 SİMÜLASYONU</b>\n"
        mesaj += f"🏦 <b>Yatırım:</b> ${toplam_yatirim} ({total_count} x $100)\n"
        mesaj += f"{kar_ikon} <b>Kâr/Zarar:</b> ${toplam_kar:+.2f}\n"
        mesaj += f"💎 <b>Final:</b> <b>${final_bakiye:.2f}</b>"

        await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)

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
    prompt = f"GÖREV: Haber analizi.\nBAŞLIK: {baslik}\nÖZET: {ozet}\nFORMAT:\nÖZET:[Kısa özet]\nSKOR:[-2 ile +2 arası tamsayı]"
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
                
                clean_text = re.sub('<[^<]+?>', '', entry.get("summary", ""))
                ai_text, skor = await ai_analiz(entry.title, clean_text[:500])
                if abs(skor) < 2: continue
                
                skor_icon = "🟢" if skor > 0 else "🔴"
                mesaj = f"<b>{entry.title}</b>\n\n{ai_text}\n\n🎯 <b>Etki:</b> {skor_icon} <b>({skor})</b>\n🔗 <a href='{entry.link}'>Link</a>"
                try:
                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                except: pass
                await asyncio.sleep(2)
        except: pass

# ==========================================
# 🧠 BÖLÜM 4.5: AKILLI TRAILING - TREND GÜCÜ ANALİZİ
# ==========================================
async def trend_gucunu_analiz_et(exchange, coin, yon, mevcut_fiyat):
    """
    TP hit sonrası trend gücünü analiz et
    
    Kriterler (her biri 25 puan):
    1. RSI: LONG için >55, SHORT için <45
    2. Fiyat vs SMA20: Trendle uyumlu mu?
    3. ADX: >25 ise trend güçlü
    4. Hacim: Ortalama üstü mü?
    
    Returns:
        trend_gucu (str): 'GUCLU', 'ORTA', 'ZAYIF'
        sl_multiplier (float): ATR çarpanı
        analiz_detay (dict): Detaylı analiz bilgisi
    """
    try:
        ohlcv = await exchange.fetch_ohlcv(f"{coin}/USDT", '1h', limit=50)
        df = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        # İndikatörleri hesapla
        rsi_now = calculate_rsi(df['close']).iloc[-1]
        sma20 = df['close'].rolling(20).mean().iloc[-1]
        adx_now = calculate_adx(df).iloc[-1]
        atr_now = calculate_atr(df, 14).iloc[-1]
        
        # Hacim analizi
        vol_sma = df['volume'].rolling(20).mean().iloc[-1]
        curr_vol = df['volume'].iloc[-1]
        vol_ratio = curr_vol / vol_sma if vol_sma > 0 else 1
        
        puan = 0
        detay = {
            'rsi': rsi_now,
            'rsi_ok': False,
            'sma20_ok': False,
            'adx': adx_now,
            'adx_ok': False,
            'vol_ratio': vol_ratio,
            'vol_ok': False,
            'atr': atr_now
        }
        
        # 1. RSI Kontrolü (25 puan)
        if yon == "LONG":
            if rsi_now > 55:
                puan += 25
                detay['rsi_ok'] = True
            elif rsi_now > 45:
                puan += 15  # Nötr bölge
        else:  # SHORT
            if rsi_now < 45:
                puan += 25
                detay['rsi_ok'] = True
            elif rsi_now < 55:
                puan += 15  # Nötr bölge
        
        # 2. Fiyat vs SMA20 (25 puan)
        if yon == "LONG":
            if mevcut_fiyat > sma20:
                puan += 25
                detay['sma20_ok'] = True
        else:  # SHORT
            if mevcut_fiyat < sma20:
                puan += 25
                detay['sma20_ok'] = True
        
        # 3. ADX Kontrolü (25 puan)
        if adx_now > 30:
            puan += 25
            detay['adx_ok'] = True
        elif adx_now > 25:
            puan += 18
            detay['adx_ok'] = True
        elif adx_now > 20:
            puan += 10
        
        # 4. Hacim Kontrolü (25 puan)
        if vol_ratio > 1.5:
            puan += 25
            detay['vol_ok'] = True
        elif vol_ratio > 1.2:
            puan += 18
            detay['vol_ok'] = True
        elif vol_ratio > 1.0:
            puan += 10
        
        # Trend gücü ve SL çarpanı belirleme
        if puan >= 75:
            trend_gucu = "GUCLU"
            sl_multiplier = 1.5  # Geniş SL - Gideni tutma!
        elif puan >= 40:
            trend_gucu = "ORTA"
            sl_multiplier = 1.0  # Dengeli SL
        else:
            trend_gucu = "ZAYIF"
            sl_multiplier = 0.0  # Buffer moduna geç
        
        detay['puan'] = puan
        
        print(f"📊 TREND ANALİZİ: {coin} ({yon})")
        print(f"   RSI: {rsi_now:.1f} {'✅' if detay['rsi_ok'] else '❌'}")
        print(f"   SMA20: {'✅' if detay['sma20_ok'] else '❌'}")
        print(f"   ADX: {adx_now:.1f} {'✅' if detay['adx_ok'] else '❌'}")
        print(f"   Hacim: {vol_ratio:.2f}x {'✅' if detay['vol_ok'] else '❌'}")
        print(f"   TOPLAM: {puan}/100 → {trend_gucu}")
        
        return trend_gucu, sl_multiplier, detay
        
    except Exception as e:
        print(f"⚠️ Trend Analiz Hatası ({coin}): {e}")
        # Hata durumunda güvenli mod - orta seviye
        return "ORTA", 1.0, {'puan': 50, 'atr': 0}

# ==========================================
# 🚀 BÖLÜM 5: STRATEJİ MOTORU (VOLUME + SCORING)
# ==========================================

async def btc_piyasa_puani_hesapla(exchange):
    """
    BTC için Piyasa Puanı (-2.5 ile +2.5 arası)
    
    Kriterler:
    1. SMA 200 (Ana Trend): +/- 1.0
    2. SMA 50 (Kısa Trend): +/- 0.5
    3. RSI 50 (Momentum):   +/- 0.5
    4. HACİM (Volume):      +/- 0.5 (Teyitçi)
    """
    try:
        # BTC verisini çek
        ohlcv = await exchange.fetch_ohlcv("BTC/USDT", '1h', limit=210)
        if not ohlcv: return 0
        
        df = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        # Son değerleri al
        price = df['close'].iloc[-1]
        open_price = df['open'].iloc[-1]
        sma50 = df['close'].rolling(50).mean().iloc[-1]
        sma200 = df['close'].rolling(200).mean().iloc[-1]
        rsi = calculate_rsi(df['close']).iloc[-1]
        
        # --- HACİM ANALİZİ ---
        vol_sma = df['volume'].rolling(20).mean().iloc[-1] # Son 20 mum ortalaması
        curr_vol = df['volume'].iloc[-1] # Şu anki hacim
        vol_ratio = curr_vol / vol_sma
        
        score = 0.0
        
        # 1. ANA TREND
        if price > sma200: score += 1.0
        else: score -= 1.0
            
        # 2. KISA TREND
        if price > sma50: score += 0.5
        else: score -= 0.5
            
        # 3. MOMENTUM
        if rsi > 50: score += 0.5
        else: score -= 0.5
        
        # 4. HACİM TEYİDİ (YENİ EKLENEN KISIM)
        # Hacim ortalamanın %25 üzerindeyse güçlü kabul et
        if vol_ratio > 1.25:
            # Yeşil Mum + Yüksek Hacim = Güçlü Alış (+0.5)
            if price > open_price: 
                score += 0.5
            # Kırmızı Mum + Yüksek Hacim = Güçlü Satış (-0.5)
            else: 
                score -= 0.5
                
        return score
    except Exception as e:
        print(f"⚠️ BTC Puan Hatası: {e}")
        return 0

async def piyasayi_tarama(exchange):
    print(f"🔍 ({datetime.now().strftime('%H:%M')}) TITANIUM V5.4 SCORING+TRAILING...")
    
    # 1. BTC PUANINI HESAPLA (Volume Destekli)
    btc_score = await btc_piyasa_puani_hesapla(exchange)
    
    # İkon Belirleme
    if btc_score >= 1.5: btc_ikon = "🟢🟢 (Güçlü Bull)"
    elif btc_score >= 0.5: btc_ikon = "🟢 (Bull)"
    elif btc_score <= -1.5: btc_ikon = "🔴🔴 (Güçlü Bear)"
    elif btc_score <= -0.5: btc_ikon = "🔴 (Bear)"
    else: btc_ikon = "⚪ (Nötr)"

    print(f"🌍 BTC SKORU: {btc_score} -> {btc_ikon}")
    
    # 2. COIN VERILERINI CEK
    async def fetch_candle(s):
        try:
            ohlcv = await exchange.fetch_ohlcv(f"{s}/USDT", '1h', limit=300)
            return s, ohlcv
        except: return s, None

    tasks = [fetch_candle(c) for c in COIN_LIST]
    results = await asyncio.gather(*tasks)
    
    # ========== HTF (4H) TREND DATA ==========
    async def fetch_htf_candle(s):
        """4H mum verisi çek - HTF trend teyidi için"""
        try:
            ohlcv_4h = await exchange.fetch_ohlcv(f"{s}/USDT", '4h', limit=60)
            return s, ohlcv_4h
        except: 
            return s, None
    
    htf_tasks = [fetch_htf_candle(c) for c in COIN_LIST]
    htf_results = await asyncio.gather(*htf_tasks)
    htf_data = {coin: data for coin, data in htf_results}
    
    for coin, bars in results:
        if not bars: continue
        
        df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df.set_index('date', inplace=True)
        
        # İndikatörler
        df['sma50'] = calculate_sma(df['close'], 50)
        df['sma200'] = calculate_sma(df['close'], 200)
        df['rsi'] = calculate_rsi(df['close'])
        df['adx'] = calculate_adx(df)
        df['atr'] = calculate_atr(df)  # ATR for dynamic stops
        
        curr = df.iloc[-1]
        price = curr['close']
        rsi_val = curr['rsi']
        adx_val = curr['adx']
        atr_val = curr['atr']  # Current ATR value
        
        trend_guclu = adx_val > 25
        
        # ========== HTF (4H) TREND TEYİDİ ==========
        htf_bullish = False
        htf_bearish = False
        
        if coin in htf_data and htf_data[coin]:
            df_4h = pd.DataFrame(htf_data[coin], columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df_4h['sma50'] = calculate_sma(df_4h['close'], 50)
            df_4h['sma20'] = calculate_sma(df_4h['close'], 20)
            
            curr_4h = df_4h.iloc[-1]
            htf_price = curr_4h['close']
            htf_sma50 = curr_4h['sma50']
            htf_sma20 = curr_4h['sma20']
            
            # HTF Bullish: Fiyat > SMA50 VE SMA20 > SMA50 (uptrend)
            htf_bullish = (htf_price > htf_sma50) and (htf_sma20 > htf_sma50)
            # HTF Bearish: Fiyat < SMA50 VE SMA20 < SMA50 (downtrend)
            htf_bearish = (htf_price < htf_sma50) and (htf_sma20 < htf_sma50)
        
        sinyal = None
        setup = ""
        
        # 🚫 ANTI-SPAM: Skip if there's already an open position for this coin
        if pozisyon_acik_mi(coin):
            continue  # Wait until current position closes
        
        # ========== 📊 PUANLIK SKORLAMA SİSTEMİ V5.4 ==========
        # Ağırlıklar:
        # - BTC Skoru: 30 puan (piyasa yönü - en önemli)
        # - 4H HTF Trend: 25 puan (yüksek zaman dilimi teyidi)
        # - SMA200 Trend: 20 puan (ana fiyat trendi)
        # - RSI Seviye: 15 puan (momentum) [ARTIRILDI]
        # - Hacim: 10 puan (coin bazlı volume) [YENİ]
        # - ADX: 10 puan (trend gücü) [DÜŞÜRÜLDÜ]
        # TOPLAM: ~110 puan max, %70+ = Sinyal
        
        long_score = 0
        short_score = 0
        long_breakdown = []
        short_breakdown = []
        
        # 1️⃣ BTC SKORU (30 puan)
        if btc_score >= 1.5:
            long_score += 30
            long_breakdown.append("BTC:30")
        elif btc_score >= 1.0:
            long_score += 25
            long_breakdown.append("BTC:25")
        elif btc_score >= 0.5:
            long_score += 15
            long_breakdown.append("BTC:15")
            
        if btc_score <= -1.5:
            short_score += 30
            short_breakdown.append("BTC:30")
        elif btc_score <= -1.0:
            short_score += 25
            short_breakdown.append("BTC:25")
        elif btc_score <= -0.5:
            short_score += 15
            short_breakdown.append("BTC:15")
        
        # 2️⃣ 4H HTF TREND (25 puan)
        if htf_bullish:
            long_score += 25
            long_breakdown.append("HTF:25")
        if htf_bearish:
            short_score += 25
            short_breakdown.append("HTF:25")
        
        # 3️⃣ SMA200 TREND (20 puan)
        if price > curr['sma200']:
            long_score += 20
            long_breakdown.append("SMA200:20")
        if price < curr['sma200']:
            short_score += 20
            short_breakdown.append("SMA200:20")
        
        # 4️⃣ ADX GÜÇ (10 puan) [DÜŞÜRÜLDÜ]
        if adx_val > 30:
            long_score += 10
            short_score += 10
            long_breakdown.append("ADX:10")
            short_breakdown.append("ADX:10")
        elif adx_val > 25:
            long_score += 7
            short_score += 7
            long_breakdown.append("ADX:7")
            short_breakdown.append("ADX:7")
        elif adx_val > 20:
            long_score += 4
            short_score += 4
            long_breakdown.append("ADX:4")
            short_breakdown.append("ADX:4")
        
        # 5️⃣ RSI SEVİYE (15 puan) [ARTIRILDI]
        if rsi_val < 30:
            long_score += 15
            long_breakdown.append("RSI:15")
        elif rsi_val < 35:
            long_score += 12
            long_breakdown.append("RSI:12")
        elif rsi_val < 40:
            long_score += 8
            long_breakdown.append("RSI:8")
        elif rsi_val < 45:
            long_score += 4
            long_breakdown.append("RSI:4")
            
        if rsi_val > 70:
            short_score += 15
            short_breakdown.append("RSI:15")
        elif rsi_val > 65:
            short_score += 12
            short_breakdown.append("RSI:12")
        elif rsi_val > 60:
            short_score += 8
            short_breakdown.append("RSI:8")
        elif rsi_val > 55:
            short_score += 4
            short_breakdown.append("RSI:4")
        
        # 6️⃣ HACİM ANALİZİ (10 puan) [YENİ]
        vol_sma20 = df['volume'].rolling(20).mean().iloc[-1]
        curr_vol = df['volume'].iloc[-1]
        vol_ratio = curr_vol / vol_sma20 if vol_sma20 > 0 else 1
        
        # Yüksek hacim = Güçlü sinyal teyidi
        if vol_ratio > 1.5:  # %50 üzeri hacim
            long_score += 10
            short_score += 10
            long_breakdown.append("VOL:10")
            short_breakdown.append("VOL:10")
        elif vol_ratio > 1.25:  # %25 üzeri hacim
            long_score += 7
            short_score += 7
            long_breakdown.append("VOL:7")
            short_breakdown.append("VOL:7")
        elif vol_ratio > 1.0:  # Ortalama üzeri
            long_score += 3
            short_score += 3
            long_breakdown.append("VOL:3")
            short_breakdown.append("VOL:3")
        
        # ========== SİNYAL KARARI (%70 EŞİĞİ) ==========
        ESIK = 70  # Minimum skor eşiği
        
        if long_score >= ESIK and long_score > short_score:
            sinyal = "LONG"
            setup = f"Score: {long_score}/100 ({'+'.join(long_breakdown)})"
        elif short_score >= ESIK and short_score > long_score:
            sinyal = "SHORT"
            setup = f"Score: {short_score}/100 ({'+'.join(short_breakdown)})"
        
        if sinyal:
            # ========== ATR-BASED TP/SL CALCULATION ==========
            # ATR Multipliers: SL=2x, TP1=2.5x, TP2=4.5x, TP3=7x (YÜKSEK KÂR)
            atr_sl = atr_val * 2.0    # Stop Loss: 2x ATR
            atr_tp1 = atr_val * 2.5   # TP1: 2.5x ATR (artırıldı)
            atr_tp2 = atr_val * 4.5   # TP2: 4.5x ATR (artırıldı)
            atr_tp3 = atr_val * 7.0   # TP3: 7x ATR (artırıldı)
            
            if sinyal == "LONG":
                tp1_price = price + atr_tp1
                tp2_price = price + atr_tp2
                tp3_price = price + atr_tp3
                sl_price = price - atr_sl
            else:  # SHORT
                tp1_price = price - atr_tp1
                tp2_price = price - atr_tp2
                tp3_price = price - atr_tp3
                sl_price = price + atr_sl
            
            # Calculate percentages for display
            tp1_pct = abs(tp1_price - price) / price * 100
            tp2_pct = abs(tp2_price - price) / price * 100
            tp3_pct = abs(tp3_price - price) / price * 100
            sl_pct = abs(sl_price - price) / price * 100
            atr_pct = (atr_val / price) * 100  # ATR as % of price
            
            p_fmt = ".8f" if price < 0.01 else ".4f"
            
            # Save with all TP levels
            islem_kaydet(coin, sinyal, price, tp1_price, tp2_price, tp3_price, sl_price)
            SON_SINYAL_ZAMANI[coin] = datetime.now()
            
            print(f"🎯 {sinyal}: {coin} (Score: {long_score if sinyal == 'LONG' else short_score}/100, ATR: {atr_pct:.2f}%)")
            
            resim = await grafik_olustur_async(coin, df.tail(100), tp1_price, sl_price, sinyal)
            ikon = "🟢" if sinyal == "LONG" else "🔴"
            
            # Skor bilgisi
            skor_deger = long_score if sinyal == "LONG" else short_score
            skor_breakdown = '+'.join(long_breakdown) if sinyal == "LONG" else '+'.join(short_breakdown)
            
            mesaj = f"""
{ikon} <b>TITANIUM SİNYAL ({sinyal})</b> #V5.3-SCORE

🪙 <b>Coin:</b> #{coin}
� <b>Skor:</b> {skor_deger}/100 ({skor_breakdown})
� <b>RSI:</b> {rsi_val:.1f} | <b>ADX:</b> {adx_val:.1f} | <b>ATR:</b> {atr_pct:.2f}%
🌍 <b>BTC Skoru:</b> {btc_score} {btc_ikon}
⏰ <b>4H Trend:</b> {'✅ Bullish' if htf_bullish else '🔴 Bearish' if htf_bearish else '⚪ Nötr'}

💰 <b>Giriş:</b> ${price:{p_fmt}}

🎯 <b>TP1 (33%):</b> ${tp1_price:{p_fmt}} (+{tp1_pct:.1f}%) [2.5x ATR]
🎯 <b>TP2 (33%):</b> ${tp2_price:{p_fmt}} (+{tp2_pct:.1f}%) [4.5x ATR]
🎯 <b>TP3 (34%):</b> ${tp3_price:{p_fmt}} (+{tp3_pct:.1f}%) [7x ATR]
🛑 <b>STOP (SL):</b> ${sl_price:{p_fmt}} (-{sl_pct:.1f}%) [2x ATR]

📌 <i>%{skor_deger} Güven Skoru ile Sinyal</i>
"""
            try:
                if resim:
                    await bot.send_photo(chat_id=KANAL_ID, photo=resim, caption=mesaj, parse_mode=ParseMode.HTML)
                else:
                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
            except Exception as e:
                print(f"Telegram Hatasi: {e}")

# ==========================================
# 🛡️ BÖLÜM 6: POZİSYON TAKİBİ (MULTI-TP)
# ==========================================
async def pozisyonlari_yokla(exchange):
    """Track open positions with multi-level TP support"""
    with sqlite3.connect("titanium_live.db") as conn:
        c = conn.cursor()
        # Updated query for multi-TP columns
        c.execute("""SELECT id, coin, yon, giris_fiyat, tp1, tp2, tp3, sl, tp1_hit, tp2_hit 
                     FROM islemler WHERE durum='ACIK'""")
        acik_islemler = c.fetchall()
        
    if not acik_islemler: return

    for islem in acik_islemler:
        id, coin, yon, giris, tp1, tp2, tp3, sl, tp1_hit, tp2_hit = islem
        try:
            ticker = await exchange.fetch_ticker(f"{coin}/USDT")
            fiyat = ticker['last']
            p_fmt = ".8f" if fiyat < 0.01 else ".4f"
            
            # --- TP1 CHECK (10% TOLERANS + AKILLI TRAILING) ---
            if not tp1_hit:
                # %10 tolerans: TP1'in %90'ına ulaşınca da sayılır
                tp1_tolerance = abs(tp1 - giris) * 0.90
                tp1_target = giris + tp1_tolerance if yon == "LONG" else giris - tp1_tolerance
                tp1_reached = (fiyat >= tp1_target) if yon == "LONG" else (fiyat <= tp1_target)
                if tp1_reached:
                    # 🧠 AKILLI TRAILING: Trend gücünü analiz et
                    trend_gucu, sl_multiplier, analiz = await trend_gucunu_analiz_et(exchange, coin, yon, fiyat)
                    atr_now = analiz.get('atr', abs(tp1 - giris) * 0.5)  # Fallback ATR
                    trend_puan = analiz.get('puan', 50)
                    
                    # SL'i trend gücüne göre belirle
                    if trend_gucu == "GUCLU" or trend_gucu == "ORTA":
                        # ATR bazlı geniş SL - Gideni tutma!
                        if yon == "LONG":
                            new_sl = fiyat - (atr_now * sl_multiplier)
                        else:  # SHORT
                            new_sl = fiyat + (atr_now * sl_multiplier)
                    else:
                        # ZAYIF trend - Sıkı SL, kârı koru!
                        buffer = 0.005  # %0.5 buffer
                        if yon == "LONG":
                            new_sl = tp1 * (1 - buffer)
                        else:  # SHORT
                            new_sl = tp1 * (1 + buffer)
                    
                    with sqlite3.connect("titanium_live.db") as conn:
                        conn.execute("UPDATE islemler SET tp1_hit=1, sl=? WHERE id=?", (new_sl, id))
                    
                    pnl1 = ((tp1 - giris) / giris * 100) if yon == "LONG" else ((giris - tp1) / giris * 100)
                    
                    # Trend göstergeleri
                    trend_ikon = "🟢" if trend_gucu == "GUCLU" else ("🟡" if trend_gucu == "ORTA" else "🔴")
                    sl_tipi = f"ATR×{sl_multiplier}" if sl_multiplier > 0 else "TP1±0.5%"
                    gideni_tutma = "🚀 Gideni Tutmuyoruz!" if trend_gucu == "GUCLU" else ("📊 Dengeli" if trend_gucu == "ORTA" else "🔒 Kârı Koruyoruz!")
                    
                    mesaj = f"""
🎯 <b>TP1 ULAŞILDI!</b> ✅

🪙 <b>#{coin}</b> ({yon})
💰 <b>Giriş:</b> ${giris:{p_fmt}}
🎯 <b>TP1:</b> ${tp1:{p_fmt}}
📈 <b>Kâr:</b> +{pnl1:.2f}%

{trend_ikon} <b>Trend Gücü:</b> {trend_gucu} ({trend_puan}/100)
   ├─ RSI: {'✅' if analiz.get('rsi_ok') else '❌'}
   ├─ SMA20: {'✅' if analiz.get('sma20_ok') else '❌'}
   ├─ ADX: {'✅' if analiz.get('adx_ok') else '❌'}
   └─ Hacim: {'✅' if analiz.get('vol_ok') else '❌'}

{gideni_tutma}
🔒 <b>YENİ SL:</b> ${new_sl:{p_fmt}} ({sl_tipi})
📌 <i>%33 Kapatıldı - Kalan %67 Devam!</i>
"""
                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                    continue  # Check other TPs next cycle
            
            # --- MOMENTUM KONTROLÜ (TP1 SONRASI) ---
            momentum_strong = False
            if tp1_hit:
                try:
                    ohlcv_mom = await exchange.fetch_ohlcv(f"{coin}/USDT", '1h', limit=50)
                    df_mom = pd.DataFrame(ohlcv_mom, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                    rsi_now = calculate_rsi(df_mom['close']).iloc[-1]
                    sma20 = df_mom['close'].rolling(20).mean().iloc[-1]
                    sma50 = df_mom['close'].rolling(50).mean().iloc[-1]
                    
                    if yon == "LONG":
                        # LONG için: RSI > 45 VE fiyat > SMA20 = momentum güçlü
                        momentum_strong = (rsi_now > 45) and (fiyat > sma20)
                    else:  # SHORT
                        # SHORT için: RSI < 55 VE fiyat < SMA20 = momentum güçlü
                        momentum_strong = (rsi_now < 55) and (fiyat < sma20)
                    
                    print(f"📊 MOMENTUM {coin}: RSI={rsi_now:.1f}, SMA20={'✅' if (fiyat > sma20 if yon == 'LONG' else fiyat < sma20) else '❌'} -> {'GÜÇLÜ' if momentum_strong else 'ZAYIF'}")
                except Exception as mom_err:
                    print(f"⚠️ Momentum Hatası ({coin}): {mom_err}")
                    momentum_strong = True  # Hata olursa default olarak devam et
            
            # --- TP2 CHECK (10% TOLERANS + AKILLI TRAILING) ---
            if tp1_hit and not tp2_hit:
                # %10 tolerans: TP2'nin %90'ına ulaşınca da sayılır
                tp2_tolerance = abs(tp2 - giris) * 0.90
                tp2_target = giris + tp2_tolerance if yon == "LONG" else giris - tp2_tolerance
                tp2_reached = (fiyat >= tp2_target) if yon == "LONG" else (fiyat <= tp2_target)
                if tp2_reached:
                    # 🧠 AKILLI TRAILING: Trend gücünü analiz et
                    trend_gucu, sl_multiplier, analiz = await trend_gucunu_analiz_et(exchange, coin, yon, fiyat)
                    atr_now = analiz.get('atr', abs(tp2 - giris) * 0.3)  # Fallback ATR
                    trend_puan = analiz.get('puan', 50)
                    
                    # TP2'de daha agresif trailing - çarpanı artır
                    sl_multiplier_tp2 = sl_multiplier * 0.8  # TP2'de biraz daha sıkı
                    
                    # SL'i trend gücüne göre belirle
                    if trend_gucu == "GUCLU" or trend_gucu == "ORTA":
                        # ATR bazlı SL
                        if yon == "LONG":
                            new_sl = fiyat - (atr_now * sl_multiplier_tp2)
                        else:  # SHORT
                            new_sl = fiyat + (atr_now * sl_multiplier_tp2)
                    else:
                        # ZAYIF trend - TP2 + buffer
                        buffer = 0.003  # %0.3 buffer (TP2'de daha sıkı)
                        if yon == "LONG":
                            new_sl = tp2 * (1 - buffer)
                        else:  # SHORT
                            new_sl = tp2 * (1 + buffer)
                    
                    with sqlite3.connect("titanium_live.db") as conn:
                        conn.execute("UPDATE islemler SET tp2_hit=1, sl=? WHERE id=?", (new_sl, id))
                    
                    pnl2 = ((tp2 - giris) / giris * 100) if yon == "LONG" else ((giris - tp2) / giris * 100)
                    
                    # Trend göstergeleri
                    trend_ikon = "🟢" if trend_gucu == "GUCLU" else ("🟡" if trend_gucu == "ORTA" else "🔴")
                    devam_msg = "🚀 TREND GÜÇLÜ - TP3'e Devam!" if trend_gucu == "GUCLU" else ("� TP3'e Bırakıldı" if trend_gucu == "ORTA" else "🔒 %66 Kilitlendi")
                    
                    mesaj = f"""
🎯🎯 <b>TP2 ULAŞILDI!</b> ✅✅

🪙 <b>#{coin}</b> ({yon})
💰 <b>Giriş:</b> ${giris:{p_fmt}}
🎯 <b>TP2:</b> ${tp2:{p_fmt}}
📈 <b>Kâr:</b> +{pnl2:.2f}%

{trend_ikon} <b>Trend:</b> {trend_gucu} ({trend_puan}/100)
🔒 <b>YENİ SL:</b> ${new_sl:{p_fmt}}
{devam_msg}
"""
                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                    continue
            
            # --- TP2 SONRASI TRAILING STOP (YENİ!) ---
            if tp1_hit and tp2_hit:
                # ATR hesapla - trailing için
                try:
                    ohlcv = await exchange.fetch_ohlcv(f"{coin}/USDT", '1h', limit=20)
                    df_trail = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                    atr_trail = calculate_atr(df_trail, 14).iloc[-1]
                    trailing_distance = atr_trail * 1.5  # 1.5x ATR trailing mesafesi
                    
                    if yon == "LONG":
                        # Yeni potansiyel SL = Mevcut fiyat - trailing mesafe
                        new_trail_sl = fiyat - trailing_distance
                        # Sadece yukarı doğru güncelle (sl'den yüksekse)
                        if new_trail_sl > sl:
                            with sqlite3.connect("titanium_live.db") as conn:
                                conn.execute("UPDATE islemler SET sl=? WHERE id=?", (new_trail_sl, id))
                            print(f"📈 TRAILING: {coin} SL güncellendi: ${sl:{p_fmt}} -> ${new_trail_sl:{p_fmt}}")
                            sl = new_trail_sl  # Güncel SL'i kullan
                    else:  # SHORT
                        new_trail_sl = fiyat + trailing_distance
                        if new_trail_sl < sl:
                            with sqlite3.connect("titanium_live.db") as conn:
                                conn.execute("UPDATE islemler SET sl=? WHERE id=?", (new_trail_sl, id))
                            print(f"📉 TRAILING: {coin} SL güncellendi: ${sl:{p_fmt}} -> ${new_trail_sl:{p_fmt}}")
                            sl = new_trail_sl
                except Exception as trail_err:
                    print(f"⚠️ Trailing Hesaplama Hatası ({coin}): {trail_err}")
            
            # --- TP3 CHECK (FULL EXIT - 10% TOLERANS) ---
            if tp1_hit and tp2_hit:
                # %10 tolerans: TP3'ün %90'ına ulaşınca da sayılır
                tp3_tolerance = abs(tp3 - giris) * 0.90
                tp3_target = giris + tp3_tolerance if yon == "LONG" else giris - tp3_tolerance
                tp3_reached = (fiyat >= tp3_target) if yon == "LONG" else (fiyat <= tp3_target)
                if tp3_reached:
                    pnl3 = ((tp3 - giris) / giris * 100) if yon == "LONG" else ((giris - tp3) / giris * 100)
                    # Calculate weighted average PnL (33% + 33% + 34%)
                    pnl1 = ((tp1 - giris) / giris * 100) if yon == "LONG" else ((giris - tp1) / giris * 100)
                    pnl2 = ((tp2 - giris) / giris * 100) if yon == "LONG" else ((giris - tp2) / giris * 100)
                    total_pnl = (pnl1 * 0.33) + (pnl2 * 0.33) + (pnl3 * 0.34)
                    
                    with sqlite3.connect("titanium_live.db") as conn:
                        conn.execute("""UPDATE islemler SET durum='KAZANDI', pnl_yuzde=?, 
                                      kapanis_zamani=? WHERE id=?""", 
                                  (total_pnl, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), id))
                    
                    mesaj = f"""
🏆🏆🏆 <b>TÜM HEDEFLER TAMAMLANDI!</b> 🎉

🪙 <b>#{coin}</b> ({yon})
💰 <b>Giriş:</b> ${giris:{p_fmt}}
🎯 <b>TP1:</b> ${tp1:{p_fmt}} ✅
🎯 <b>TP2:</b> ${tp2:{p_fmt}} ✅
🎯 <b>TP3:</b> ${tp3:{p_fmt}} ✅
📈 <b>Toplam Kâr:</b> +{total_pnl:.2f}%

🤖 <i>Titanium V5.4 - Trailing TP Success!</i>
"""
                    await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                    continue
            
            # --- STOP LOSS CHECK ---
            sl_hit = (fiyat <= sl) if yon == "LONG" else (fiyat >= sl)
            if sl_hit:
                # Calculate actual PnL based on which TPs were hit
                partial_profit = 0.0
                if tp1_hit:
                    pnl1 = ((tp1 - giris) / giris * 100) if yon == "LONG" else ((giris - tp1) / giris * 100)
                    partial_profit += pnl1 * 0.33
                if tp2_hit:
                    pnl2 = ((tp2 - giris) / giris * 100) if yon == "LONG" else ((giris - tp2) / giris * 100)
                    partial_profit += pnl2 * 0.33
                
                # Remaining position hit SL
                remaining_pct = 1.0 - (0.33 if tp1_hit else 0) - (0.33 if tp2_hit else 0)
                sl_pnl = ((sl - giris) / giris * 100) if yon == "LONG" else ((giris - sl) / giris * 100)
                total_pnl = partial_profit + (sl_pnl * remaining_pct)
                
                durum = "PARTIAL" if (tp1_hit or tp2_hit) else "KAYBETTI"
                
                with sqlite3.connect("titanium_live.db") as conn:
                    conn.execute("""UPDATE islemler SET durum=?, pnl_yuzde=?, 
                                  kapanis_zamani=? WHERE id=?""", 
                              (durum, total_pnl, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), id))
                
                ikon = "⚠️" if durum == "PARTIAL" else "❌"
                tp_status = f"TP1: {'✅' if tp1_hit else '❌'} | TP2: {'✅' if tp2_hit else '❌'}"
                
                mesaj = f"""
🏁 <b>POZİSYON KAPANDI</b> {ikon}

🪙 <b>#{coin}</b> ({yon})
🏷️ <b>Sonuç:</b> {durum}
{tp_status}

💰 <b>Giriş:</b> ${giris:{p_fmt}}
🚪 <b>SL Çıkış:</b> ${fiyat:{p_fmt}}
📉 <b>Net Kâr/Zarar:</b> {total_pnl:+.2f}%

🤖 <i>Titanium Bot</i>
"""
                await bot.send_message(chat_id=KANAL_ID, text=mesaj, parse_mode=ParseMode.HTML)
                
        except Exception as e:
            print(f"Pozisyon Takip Hatası ({coin}): {e}")
            continue

# ==========================================
# 🏁 MAIN LOOP
# ==========================================
async def main():
    global SON_RAPOR_TARIHI
    db_ilk_kurulum()
    print("🚀 Titanium PREMIUM Bot Aktif! (Telegram: Sinyal + Haber + Macro)")
    
    exchange = ccxt.kucoin(exchange_config)
    
    try:
        await bot.send_message(chat_id=KANAL_ID, text="🚀 **TITANIUM BOT V5.4 BAŞLATILDI!**\n\n✅ Sistem: Aktif\n✅ Filtre: BTC Puanlama + Hacim\n✅ Borsa: KuCoin\n📊 Raporlama: Aktif", parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        print(f"❌ Telegram Test Mesajı Hatası: {e}")

    if "ETH" in COIN_LIST:
        COIN_LIST.remove("ETH")

    try:
        while True:
            # İstanbul saati (UTC+3)
            from datetime import timezone, timedelta
            istanbul_tz = timezone(timedelta(hours=3))
            simdi = datetime.now(istanbul_tz)
            bugun_str = simdi.strftime("%Y-%m-%d")
            
            # Gün sonu raporu - İstanbul saati 23:55
            if simdi.hour == 23 and simdi.minute >= 55 and SON_RAPOR_TARIHI != bugun_str:
                print(f"📊 Gün sonu raporu gönderiliyor... (İstanbul: {simdi.strftime('%H:%M')})")
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
