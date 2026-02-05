"""
TITANIUM Bot - Database Manager
================================
SQLite ve PostgreSQL arasında geçiş yapılabilir veritabanı abstraction layer.

Kullanım:
    from db_manager import get_db, execute_query, execute_many

    # Tekil sorgu
    results = execute_query("SELECT * FROM trades WHERE coin = ?", ("BTC",))
    
    # Insert/Update
    execute_query("INSERT INTO trades (coin, side) VALUES (?, ?)", ("ETH", "LONG"))
    
    # Batch insert
    execute_many("INSERT INTO trades (coin, side) VALUES (?, ?)", [("BTC", "LONG"), ("ETH", "SHORT")])

Konfigürasyon:
    DATABASE_URL env variable'ı ile kontrol edilir:
    - Boş veya yok: SQLite (titanium_live.db)
    - postgres://...: PostgreSQL
"""

import os
import logging
from typing import Any, List, Tuple, Optional, Union
from contextlib import contextmanager
import sqlite3

logger = logging.getLogger(__name__)

# PostgreSQL bağlantısı için psycopg2 (opsiyonel import)
try:
    import psycopg2
    import psycopg2.extras
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.debug("psycopg2 yüklü değil, sadece SQLite kullanılabilir")


# Veritabanı türü
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

# Railway/Heroku formatını düzelt (postgres:// -> postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Veritabanı türünü belirle
USE_POSTGRES = bool(DATABASE_URL and DATABASE_URL.startswith("postgresql://"))
SQLITE_PATH = "titanium_live.db"

if USE_POSTGRES and not POSTGRES_AVAILABLE:
    logger.warning("⚠️ DATABASE_URL ayarlı ama psycopg2 yüklü değil. SQLite kullanılacak.")
    USE_POSTGRES = False

logger.info(f"📦 Veritabanı: {'PostgreSQL' if USE_POSTGRES else 'SQLite'}")


def _convert_placeholders(query: str) -> str:
    """
    SQLite placeholder'larını (?) PostgreSQL formatına (%s) çevir.
    """
    if USE_POSTGRES:
        return query.replace("?", "%s")
    return query


@contextmanager
def get_db():
    """
    Veritabanı bağlantısı context manager.
    
    Usage:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades")
    """
    conn = None
    try:
        if USE_POSTGRES:
            conn = psycopg2.connect(DATABASE_URL)
            conn.autocommit = False
        else:
            conn = sqlite3.connect(SQLITE_PATH)
            conn.row_factory = sqlite3.Row  # Dict-like erişim
        
        yield conn
        conn.commit()
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"❌ Veritabanı hatası: {e}")
        raise
    finally:
        if conn:
            conn.close()


def execute_query(
    query: str, 
    params: Optional[Tuple] = None,
    fetch: bool = True
) -> Union[List[Tuple], int]:
    """
    Tekil SQL sorgusu çalıştır.
    
    Args:
        query: SQL sorgusu (? placeholder'ları ile)
        params: Parametre tuple'ı
        fetch: True ise sonuçları döndür, False ise etkilenen satır sayısı
    
    Returns:
        fetch=True: Liste of tuples (sorgu sonuçları)
        fetch=False: Etkilenen satır sayısı
    """
    query = _convert_placeholders(query)
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        if fetch:
            if USE_POSTGRES:
                return cursor.fetchall()
            else:
                # SQLite Row'ları tuple'a çevir
                return [tuple(row) for row in cursor.fetchall()]
        else:
            return cursor.rowcount


def execute_many(query: str, params_list: List[Tuple]) -> int:
    """
    Batch SQL işlemi çalıştır (INSERT, UPDATE için).
    
    Args:
        query: SQL sorgusu
        params_list: Parametre listesi
    
    Returns:
        Etkilenen toplam satır sayısı
    """
    query = _convert_placeholders(query)
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        if USE_POSTGRES:
            # PostgreSQL batch insert optimize
            psycopg2.extras.execute_batch(cursor, query, params_list)
        else:
            cursor.executemany(query, params_list)
        
        return cursor.rowcount


def init_database():
    """
    Veritabanı tablolarını oluştur (yoksa).
    """
    # Trades tablosu
    trades_table = """
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY,
        coin TEXT NOT NULL,
        side TEXT NOT NULL,
        entry_price REAL,
        current_price REAL,
        tp1 REAL,
        tp2 REAL,
        tp3 REAL,
        sl REAL,
        tp1_hit INTEGER DEFAULT 0,
        tp2_hit INTEGER DEFAULT 0,
        tp3_hit INTEGER DEFAULT 0,
        durum TEXT DEFAULT 'OPEN',
        strateji TEXT,
        skor REAL,
        entry_time TEXT,
        exit_time TEXT,
        exit_price REAL,
        pnl_pct REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    # SQLite vs PostgreSQL syntax farkı
    if USE_POSTGRES:
        trades_table = trades_table.replace("INTEGER PRIMARY KEY", "SERIAL PRIMARY KEY")
    
    execute_query(trades_table, fetch=False)
    logger.info("✅ Veritabanı tabloları hazır")


def get_open_trades() -> List[dict]:
    """Açık pozisyonları getir"""
    query = "SELECT * FROM trades WHERE durum = ?"
    rows = execute_query(query, ("OPEN",))
    
    # Tuple'ları dict'e çevir
    columns = ['id', 'coin', 'side', 'entry_price', 'current_price', 
               'tp1', 'tp2', 'tp3', 'sl', 'tp1_hit', 'tp2_hit', 'tp3_hit',
               'durum', 'strateji', 'skor', 'entry_time', 'exit_time', 
               'exit_price', 'pnl_pct', 'created_at']
    
    return [dict(zip(columns, row)) for row in rows]


def insert_trade(
    coin: str,
    side: str,
    entry_price: float,
    tp1: float,
    tp2: float,
    tp3: float,
    sl: float,
    strateji: str = "TREND",
    skor: float = 0
) -> int:
    """Yeni trade ekle"""
    from datetime import datetime
    
    query = """
    INSERT INTO trades (coin, side, entry_price, current_price, tp1, tp2, tp3, sl, strateji, skor, entry_time)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    params = (coin, side, entry_price, entry_price, tp1, tp2, tp3, sl, strateji, skor, datetime.now().isoformat())
    return execute_query(query, params, fetch=False)


def update_trade(trade_id: int, **kwargs) -> int:
    """Trade güncelle"""
    if not kwargs:
        return 0
    
    set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
    query = f"UPDATE trades SET {set_clause} WHERE id = ?"
    params = tuple(kwargs.values()) + (trade_id,)
    
    return execute_query(query, params, fetch=False)


def close_trade(
    trade_id: int,
    exit_price: float,
    durum: str,
    pnl_pct: float
) -> int:
    """Trade kapat"""
    from datetime import datetime
    
    return update_trade(
        trade_id,
        exit_price=exit_price,
        exit_time=datetime.now().isoformat(),
        durum=durum,
        pnl_pct=pnl_pct
    )


def get_trade_stats(days: int = 30) -> dict:
    """Trade istatistiklerini getir"""
    query = """
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN durum = 'WIN' THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN durum = 'LOSS' THEN 1 ELSE 0 END) as losses,
        AVG(pnl_pct) as avg_pnl,
        SUM(pnl_pct) as total_pnl
    FROM trades
    WHERE durum != 'OPEN'
    """
    
    rows = execute_query(query)
    if rows and rows[0]:
        total, wins, losses, avg_pnl, total_pnl = rows[0]
        return {
            "total": total or 0,
            "wins": wins or 0,
            "losses": losses or 0,
            "win_rate": (wins / total * 100) if total else 0,
            "avg_pnl": avg_pnl or 0,
            "total_pnl": total_pnl or 0
        }
    return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0}
# ==========================================
# 📊 ISLEMLER TABLOSU (main.py uyumlu)
# ==========================================

def init_islemler_table():
    """main.py ile uyumlu islemler tablosunu oluştur"""
    islemler_table = """
    CREATE TABLE IF NOT EXISTS islemler (
        id INTEGER PRIMARY KEY,
        coin TEXT, 
        yon TEXT, 
        giris_fiyat REAL,
        tp1 REAL, 
        tp2 REAL, 
        tp3 REAL, 
        sl REAL,
        tp1_hit INTEGER DEFAULT 0, 
        tp2_hit INTEGER DEFAULT 0,
        durum TEXT DEFAULT 'ACIK', 
        pnl_yuzde REAL DEFAULT 0,
        acilis_zamani DATETIME, 
        kapanis_zamani DATETIME
    )
    """
    
    if USE_POSTGRES:
        islemler_table = islemler_table.replace("INTEGER PRIMARY KEY", "SERIAL PRIMARY KEY")
    
    execute_query(islemler_table, fetch=False)
    
    # Haberler tablosu
    haberler_table = "CREATE TABLE IF NOT EXISTS haberler (link TEXT PRIMARY KEY)"
    execute_query(haberler_table, fetch=False)


def islem_kaydet(coin: str, yon: str, giris: float, tp1: float, tp2: float, tp3: float, sl: float) -> int:
    """Trade kaydet (main.py uyumlu)"""
    from datetime import datetime
    
    query = """
    INSERT INTO islemler (coin, yon, giris_fiyat, tp1, tp2, tp3, sl, acilis_zamani) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (coin, yon, giris, tp1, tp2, tp3, sl, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return execute_query(query, params, fetch=False)


def pozisyon_acik_mi(coin: str) -> bool:
    """Coin için açık pozisyon var mı?"""
    query = "SELECT count(*) FROM islemler WHERE coin=? AND durum='ACIK'"
    rows = execute_query(query, (coin,))
    return rows[0][0] > 0 if rows else False


def short_var_mi(coin: str) -> bool:
    """Coin için açık SHORT pozisyon var mı?"""
    query = "SELECT count(*) FROM islemler WHERE coin=? AND yon='SHORT' AND durum='ACIK'"
    rows = execute_query(query, (coin,))
    return rows[0][0] > 0 if rows else False


def get_acik_islemler() -> list:
    """Tüm açık işlemleri getir"""
    query = """
    SELECT id, coin, yon, giris_fiyat, tp1, tp2, tp3, sl, tp1_hit, tp2_hit, durum
    FROM islemler 
    WHERE durum = 'ACIK'
    """
    return execute_query(query)


def islem_guncelle(islem_id: int, **kwargs) -> int:
    """İşlem güncelle"""
    if not kwargs:
        return 0
    
    set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
    query = f"UPDATE islemler SET {set_clause} WHERE id = ?"
    params = tuple(kwargs.values()) + (islem_id,)
    
    return execute_query(query, params, fetch=False)


def islem_kapat(islem_id: int, durum: str, pnl_yuzde: float) -> int:
    """İşlemi kapat"""
    from datetime import datetime
    
    return islem_guncelle(
        islem_id,
        durum=durum,
        pnl_yuzde=pnl_yuzde,
        kapanis_zamani=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


# Modül yüklendiğinde tabloları oluştur
try:
    init_database()
    init_islemler_table()
except Exception as e:
    logger.warning(f"⚠️ Veritabanı init hatası: {e}")


if __name__ == "__main__":
    # Test
    print(f"Database: {'PostgreSQL' if USE_POSTGRES else 'SQLite'}")
    
    # Test insert
    insert_trade("TEST", "LONG", 100.0, 105.0, 110.0, 115.0, 95.0, "TEST", 75)
    
    # Test query
    trades = get_open_trades()
    print(f"Open trades: {len(trades)}")
    
    # Test stats
    stats = get_trade_stats()
    print(f"Stats: {stats}")
