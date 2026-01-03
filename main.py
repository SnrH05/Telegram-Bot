import feedparser, asyncio, os, sys, sqlite3, time, re, io
import ccxt, numpy as np, pandas as pd, mplfinance as mpf
from datetime import datetime, timedelta
import google.genai as genai
from telegram.ext import Application
from telegram.constants import ParseMode

print("⚙️ ULTRA QUANT PIVOT MASTER BOT BAŞLATILIYOR...")

# ===================== ENV =====================
TOKEN = os.getenv("BOT_TOKEN", "")
KANAL_ID = int(os.getenv("KANAL_ID", "0"))
GEMINI_KEY = os.getenv("GEMINI_KEY", "")

if not TOKEN or not GEMINI_KEY or not KANAL_ID:
    sys.exit("❌ ENV eksik")

# ✅ GOOGLE GENAI CLIENT (DOĞRU YOL)
genai_client = genai.Client(api_key=GEMINI_KEY)

application = Application.builder().token(TOKEN).build()
exchange = ccxt.kucoin({'enableRateLimit': True})

COIN_LIST = ["BTC","ETH","SOL","XRP","BNB","ADA","AVAX","DOGE","TON","LINK","DOT","POL","LTC","BCH","PEPE","FET","SUI","APT","ARB","OP","TIA","INJ","RENDER"]
RSS_LIST = ["https://cryptonews.com/news/feed/","https://cointelegraph.com/rss","https://decrypt.co/feed"]

SON_SINYAL = {}

# ===================== DB =====================
def db_init():
    conn = sqlite3.connect("trade.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS trades(
        id INTEGER PRIMARY KEY,
        coin TEXT, yon TEXT,
        giris REAL, tp1 REAL, tp2 REAL, tp3 REAL, sl REAL,
        kalan REAL DEFAULT 1.0,
        durum TEXT DEFAULT 'ACIK',
        pnl REAL DEFAULT 0,
        zaman DATETIME
    )""")
    conn.commit(); conn.close()

# ===================== INDICATORS =====================
def ema(s, n): return s.ewm(span=n).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = -d.clip(upper=0).rolling(n).mean()
    rs = g / l
    return 100 - (100 / (1 + rs))

def macd(s):
    m = ema(s,12) - ema(s,26)
    return m, ema(m,9)

def adx(df, n=14):
    up = df['high'].diff()
    dn = -df['low'].diff()
    up = up.where((up > dn) & (up > 0), 0)
    dn = dn.where((dn > up) & (dn > 0), 0)
    tr = pd.concat([
        df['high']-df['low'],
        (df['high']-df['close'].shift()).abs(),
        (df['low']-df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    pdi = 100 * (up.ewm(alpha=1/n).mean() / atr)
    mdi = 100 * (dn.ewm(alpha=1/n).mean() / atr)
    dx = (abs(pdi - mdi) / (pdi + mdi)) * 100
    return dx.ewm(alpha=1/n).mean()

# ===================== GRAPH =====================
def grafik(df, coin, tp1, tp2, tp3, sl):
    buf = io.BytesIO()
    df = df.tail(80)
    ap = [mpf.make_addplot(df['ema200'], color='white')]
    mpf.plot(
        df,
        type='candle',
        addplot=ap,
        hlines=dict(hlines=[tp1, tp2, tp3, sl]),
        savefig=buf
    )
    buf.seek(0)
    return buf.read()

# ===================== SIGNAL =====================
async def scan():
    for coin in COIN_LIST:
        if coin in SON_SINYAL and datetime.now() - SON_SINYAL[coin] < timedelta(hours=2):
            continue
        try:
            bars = await asyncio.to_thread(exchange.fetch_ohlcv, f"{coin}/USDT", '1h', 300)
            df = pd.DataFrame(bars, columns=['t','open','high','low','close','vol'])

            df['ema200'] = ema(df['close'],200)
            df['rsi'] = rsi(df['close'])
            df['macd'], df['sig'] = macd(df['close'])
            df['adx'] = adx(df)

            c, p = df.iloc[-1], df.iloc[-2]
            if c['adx'] < 25:
                continue

            if c['close'] > c['ema200'] and p['macd'] < p['sig'] and c['macd'] > c['sig']:
                atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
                tp1, tp2, tp3 = c['close'] + atr*1.5, c['close'] + atr*3, c['close'] + atr*6
                sl = c['close'] - atr*2
                SON_SINYAL[coin] = datetime.now()

                conn = sqlite3.connect("trade.db")
                conn.execute(
                    "INSERT INTO trades VALUES (NULL,?,?,?,?,?,?,?,1,'ACIK',0,?)",
                    (coin,"LONG",c['close'],tp1,tp2,tp3,sl,datetime.now())
                )
                conn.commit(); conn.close()

                img = grafik(df, coin, tp1, tp2, tp3, sl)
                await application.bot.send_photo(
                    chat_id=KANAL_ID,
                    photo=img,
                    caption=f"⚡ <b>VIP SİNYAL</b>\n#{coin}\n📈 LONG\n📊 WinRate: %{winrate():.1f}",
                    parse_mode=ParseMode.HTML
                )
        except Exception as e:
            print("SCAN ERROR:", e)

# ===================== TRADE CONTROL =====================
def winrate():
    conn = sqlite3.connect("trade.db")
    df = pd.read_sql("SELECT * FROM trades WHERE durum!='ACIK'", conn)
    conn.close()
    if len(df) == 0:
        return 0
    return (df[df['pnl'] > 0].shape[0] / len(df)) * 100

async def kontrol():
    conn = sqlite3.connect("trade.db")
    df = pd.read_sql("SELECT * FROM trades WHERE durum='ACIK'", conn)

    for _, r in df.iterrows():
        fiyat = (await asyncio.to_thread(exchange.fetch_ticker, f"{r.coin}/USDT"))['last']
        pnl, kalan = 0, r.kalan

        if fiyat >= r.tp1 and r.kalan == 1:
            pnl += 0.5 * ((r.tp1 - r.giris) / r.giris) * 100
            kalan = 0.5
        elif fiyat >= r.tp2 and r.kalan == 0.5:
            pnl += 0.25 * ((r.tp2 - r.giris) / r.giris) * 100
            kalan = 0.25
        elif fiyat <= r.sl:
            pnl += (fiyat - r.giris) / r.giris * 100 * r.kalan
            kalan = 0

        else:
            continue

        conn.execute(
            "UPDATE trades SET kalan=?, pnl=pnl+?, durum=? WHERE id=?",
            (kalan, pnl, "KAPANDI" if kalan == 0 else "ACIK", r.id)
        )
        conn.commit()

    conn.close()

# ===================== MAIN =====================
async def main():
    db_init()
    await application.initialize()
    await application.start()
    while True:
        await scan()
        await kontrol()
        await asyncio.sleep(180)

asyncio.run(main())
