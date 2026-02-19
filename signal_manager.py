"""
TITANIUM Bot - Signal Manager
=============================
Bu modül, sinyal yönetimini, skorlamaya göre sıralamayı ve günlük limitleri kontrol eder.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from config import MIN_SCORE_THRESHOLD, GUNLUK_SINYAL_LIMIT

logger = logging.getLogger(__name__)

class SignalManager:
    def __init__(self):
        self.daily_signals = [] # [{"coin": "BTC", "score": 85, "time": datetime}]
        self.failed_signals_log = [] # Debug için, neden gönderilmedi?
        self.last_reset_date = datetime.now().date()

    def _reset_daily_counters_if_needed(self):
        """Gün değiştiyse sayaçları sıfırla"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            logger.info("📅 Yeni gün: Sinyal sayaçları sıfırlanıyor...")
            self.daily_signals = []
            self.failed_signals_log = []
            self.last_reset_date = current_date

    def can_send_signal(self, coin: str, score: float) -> bool:
        """
        Sinyal gönderilebilir mi kontrol et.
        
        Kriterler:
        1. Skor > MIN_SCORE_THRESHOLD
        2. Günlük limit aşılmamış olmalı
        3. Aynı coin için bugün zaten sinyal gönderilmemiş olmalı (Opsiyonel)
        """
        self._reset_daily_counters_if_needed()
        
        # 1. Skor Kontrolü
        if score < MIN_SCORE_THRESHOLD:
            # logger.info(f"🚫 Score too low: {score} < {MIN_SCORE_THRESHOLD} ({coin})")
            return False
            
        # 2. Limit Kontrolü
        if len(self.daily_signals) >= GUNLUK_SINYAL_LIMIT:
            logger.warning(f"🚫 Günlük sinyal limiti doldu! ({len(self.daily_signals)}/{GUNLUK_SINYAL_LIMIT})")
            return False
            
        # 3. Mükerrer Kontrolü (Aynı gün aynı coin)
        for sig in self.daily_signals:
            if sig['coin'] == coin:
                # logger.info(f"🚫 Mükerrer sinyal: {coin} bugün zaten gönderildi.")
                return False
                
        return True

    def record_signal(self, coin: str, score: float, strategy_name: str):
        """Gönderilen sinyali kaydet"""
        self.daily_signals.append({
            "coin": coin,
            "score": score,
            "strategy": strategy_name,
            "time": datetime.now()
        })
        logger.info(f"✅ Sinyal Kaydedildi: {coin} | Skor: {score} | Strateji: {strategy_name}")
        logger.info(f"📊 Günlük Sinyal Durumu: {len(self.daily_signals)}/{GUNLUK_SINYAL_LIMIT}")

    def get_status_report(self) -> str:
        """Kullanıcıya gösterilecek durum raporu"""
        self._reset_daily_counters_if_needed()
        return (
            f"📊 **Sinyal Yöneticisi Durumu**\n"
            f"• Tarih: {self.last_reset_date}\n"
            f"• Gönderilen: {len(self.daily_signals)} / {GUNLUK_SINYAL_LIMIT}\n"
            f"• Eşik Skor: {MIN_SCORE_THRESHOLD}"
        )

# Global Manager Instance
signal_manager = SignalManager()
