"""
TITANIUM Bot - Grafik Modülü
============================
Mum grafikleri oluşturma fonksiyonları.
"""

import asyncio
import io
import mplfinance as mpf
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def _grafik_olustur_sync(coin, df_gelen, tp, sl, yon):
    """
    Senkron grafik oluşturma (thread pool'da çalıştırılacak)
    
    Args:
        coin: Coin sembolü
        df_gelen: OHLCV DataFrame
        tp: Take profit fiyatı (veya liste)
        sl: Stop loss fiyatı
        yon: İşlem yönü (LONG/SHORT)
    
    Returns:
        BytesIO: PNG grafik verisi
    """
    try:
        df = df_gelen.copy()
        df.index = pd.to_datetime(df.index)
        
        # TP değeri liste gelebilir (multi-TP), orta değeri al
        if isinstance(tp, (list, tuple)):
            tp_display = tp[1] if len(tp) > 1 else tp[0]  # TP2'yi göster
        else:
            tp_display = tp
        
        # Yatay çizgiler
        hlines = dict(
            hlines=[tp_display, sl],
            colors=['green', 'red'],
            linestyle='--',
            linewidths=1.5
        )
        
        # Stil
        mc = mpf.make_marketcolors(
            up='#26a69a', down='#ef5350',
            edge='inherit',
            wick='inherit',
            volume='in'
        )
        style = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle=':',
            gridcolor='#2a2a2a',
            facecolor='#1a1a2e',
            edgecolor='#1a1a2e'
        )
        
        # Başlık
        title = f"{coin}/USDT ({yon}) | TP: ${tp_display:.4f} | SL: ${sl:.4f}"
        
        # Grafik oluştur
        buf = io.BytesIO()
        mpf.plot(
            df,
            type='candle',
            style=style,
            title=title,
            ylabel='Price (USDT)',
            volume=True,
            hlines=hlines,
            figsize=(12, 8),
            savefig=dict(fname=buf, format='png', dpi=100, bbox_inches='tight')
        )
        buf.seek(0)
        return buf
        
    except Exception as e:
        logger.error(f"❌ Grafik oluşturma hatası ({coin}): {e}")
        return None


async def grafik_olustur_async(coin, df, tp, sl, yon):
    """
    Asenkron grafik oluşturma (main loop'u bloklamaz)
    
    Args:
        coin: Coin sembolü
        df: OHLCV DataFrame
        tp: Take profit fiyatı
        sl: Stop loss fiyatı
        yon: İşlem yönü
    
    Returns:
        BytesIO: PNG grafik verisi
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _grafik_olustur_sync, coin, df, tp, sl, yon)
