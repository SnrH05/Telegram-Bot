"""
TITANIUM Bot - State Manager
============================
Bot durumunu kaydetme ve kurtarma mekanizması.

Özellikler:
- Graceful shutdown handler
- Açık pozisyonları kaydetme/kurtarma
- Periyodik state backup
"""

import json
import os
import signal
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

STATE_FILE = "bot_state.json"


class StateManager:
    """Bot durumunu yöneten sınıf"""
    
    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        self.state: Dict[str, Any] = {
            "last_update": None,
            "open_positions": [],
            "daily_signals": [],
            "cooldown_coins": {},
            "is_running": False,
            "start_time": None,
        }
        self._shutdown_handlers_registered = False
    
    def save_state(self) -> bool:
        """
        Mevcut durumu dosyaya kaydet.
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            self.state["last_update"] = datetime.now().isoformat()
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False, default=str)
            
            logger.debug(f"💾 State kaydedildi: {self.state_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ State kaydetme hatası: {e}")
            return False
    
    def load_state(self) -> bool:
        """
        Kayıtlı durumu yükle.
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            if not os.path.exists(self.state_file):
                logger.info("📁 State dosyası bulunamadı, yeni state oluşturuluyor")
                return False
            
            with open(self.state_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            
            # Mevcut state ile birleştir
            self.state.update(loaded)
            
            last_update = self.state.get("last_update", "Bilinmiyor")
            logger.info(f"✅ State yüklendi (Son güncelleme: {last_update})")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ State dosyası bozuk: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ State yükleme hatası: {e}")
            return False
    
    def update_positions(self, positions: List[Dict]) -> None:
        """Açık pozisyonları güncelle"""
        self.state["open_positions"] = positions
    
    def get_positions(self) -> List[Dict]:
        """Kayıtlı pozisyonları döndür"""
        return self.state.get("open_positions", [])
    
    def update_daily_signals(self, signals: List) -> None:
        """Günlük sinyalleri güncelle"""
        self.state["daily_signals"] = signals
    
    def get_daily_signals(self) -> List:
        """Günlük sinyalleri döndür"""
        return self.state.get("daily_signals", [])
    
    def update_cooldowns(self, cooldowns: Dict) -> None:
        """Coin cooldown'larını güncelle"""
        # Datetime'ları string'e çevir
        serializable = {}
        for coin, timestamp in cooldowns.items():
            if hasattr(timestamp, 'isoformat'):
                serializable[coin] = timestamp.isoformat()
            else:
                serializable[coin] = str(timestamp)
        self.state["cooldown_coins"] = serializable
    
    def get_cooldowns(self) -> Dict:
        """Coin cooldown'larını döndür"""
        cooldowns = self.state.get("cooldown_coins", {})
        # String'leri datetime'a çevir
        result = {}
        for coin, timestamp_str in cooldowns.items():
            try:
                result[coin] = datetime.fromisoformat(timestamp_str)
            except (ValueError, TypeError):
                pass
        return result
    
    def set_running(self, is_running: bool) -> None:
        """Bot çalışma durumunu ayarla"""
        self.state["is_running"] = is_running
        if is_running:
            self.state["start_time"] = datetime.now().isoformat()
    
    def register_shutdown_handlers(self, cleanup_callback=None) -> None:
        """
        Graceful shutdown handler'larını kaydet.
        
        Args:
            cleanup_callback: Shutdown öncesi çağrılacak fonksiyon
        """
        if self._shutdown_handlers_registered:
            return
        
        def shutdown_handler(signum, frame):
            logger.info(f"🛑 Shutdown sinyali alındı (signal={signum})")
            
            # State kaydet
            self.state["is_running"] = False
            self.save_state()
            
            # Custom cleanup
            if cleanup_callback:
                try:
                    cleanup_callback()
                except Exception as e:
                    logger.error(f"Cleanup hatası: {e}")
            
            logger.info("👋 Bot güvenli şekilde kapatıldı")
            sys.exit(0)
        
        # Windows ve Unix için farklı sinyaller
        signal.signal(signal.SIGINT, shutdown_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, shutdown_handler)  # Kill
        
        # Windows'ta SIGBREAK de var
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, shutdown_handler)
        
        self._shutdown_handlers_registered = True
        logger.info("✅ Shutdown handler'lar kaydedildi")
    
    def recover_from_crash(self) -> Dict[str, Any]:
        """
        Crash sonrası kurtarma bilgilerini döndür.
        
        Returns:
            Dict: Kurtarma bilgileri
        """
        if not self.load_state():
            return {"recovered": False, "reason": "No state file"}
        
        # Son güncelleme ne kadar eski?
        last_update_str = self.state.get("last_update")
        if not last_update_str:
            return {"recovered": False, "reason": "No last_update"}
        
        try:
            last_update = datetime.fromisoformat(last_update_str)
            age_hours = (datetime.now() - last_update).total_seconds() / 3600
        except (ValueError, TypeError):
            age_hours = float('inf')
        
        # 24 saatten eski ise kurtarmaya değmez
        if age_hours > 24:
            return {
                "recovered": False, 
                "reason": f"State too old ({age_hours:.1f}h)"
            }
        
        positions = self.get_positions()
        signals = self.get_daily_signals()
        cooldowns = self.get_cooldowns()
        
        return {
            "recovered": True,
            "last_update": last_update_str,
            "age_hours": age_hours,
            "open_positions": len(positions),
            "daily_signals": len(signals),
            "cooldown_coins": len(cooldowns),
            "positions": positions,
            "signals": signals,
            "cooldowns": cooldowns
        }


# Global instance
state_manager = StateManager()


def periodic_save(positions=None, signals=None, cooldowns=None):
    """
    Periyodik state kaydetme helper fonksiyonu.
    Her 5 dakikada bir çağrılmalı.
    """
    if positions is not None:
        state_manager.update_positions(positions)
    if signals is not None:
        state_manager.update_daily_signals(signals)
    if cooldowns is not None:
        state_manager.update_cooldowns(cooldowns)
    
    state_manager.save_state()


if __name__ == "__main__":
    # Test
    sm = StateManager("test_state.json")
    sm.set_running(True)
    sm.update_positions([{"coin": "BTC", "side": "LONG", "entry": 50000}])
    sm.update_daily_signals([("2024-01-01", "SOL", "LONG")])
    sm.update_cooldowns({"ETH": datetime.now()})
    sm.save_state()
    
    # Yükle
    sm2 = StateManager("test_state.json")
    recovery = sm2.recover_from_crash()
    print(f"Recovery: {recovery}")
    
    # Temizle
    os.remove("test_state.json")
