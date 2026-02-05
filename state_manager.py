"""
TITANIUM Bot - State Manager
=============================
Bot çökmelerinde state'i kurtarmak için persistence layer.

Kaydedilen State:
- son_sinyal_zamani: Her coin için son sinyal zamanı
- bugunun_sinyalleri: Bugün üretilen sinyaller
- son_rapor_tarihi: Son günlük rapor tarihi
- kill_switch_state: Risk manager durumu

Kullanım:
    from state_manager import StateManager
    
    state_mgr = StateManager()
    state_mgr.load()  # Startup'ta
    state_mgr.save()  # Değişiklik sonrası
"""

import os
import json
import logging
import atexit
import signal
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# State dosyası yolu
STATE_FILE = "titanium_state.json"
BACKUP_FILE = "titanium_state.backup.json"


class StateManager:
    """
    Bot state'ini JSON dosyasında sakla ve geri yükle.
    
    Thread-safe değil - tek process için tasarlandı.
    """
    
    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        self.backup_file = BACKUP_FILE
        
        # Default state
        self._state: Dict[str, Any] = {
            "son_sinyal_zamani": {},
            "bugunun_sinyalleri": [],
            "son_rapor_tarihi": None,
            "kill_switch_active": False,
            "kill_switch_time": None,
            "kill_switch_reason": "",
            "last_save_time": None,
            "version": "1.0"
        }
        
        self._dirty = False  # Değişiklik var mı?
        
        # Graceful shutdown handler
        self._register_shutdown_handlers()
        
    def _register_shutdown_handlers(self):
        """SIGINT ve SIGTERM için handler kaydet."""
        # atexit - normal çıkışlarda
        atexit.register(self._on_exit)
        
        # Signal handlers - CTRL+C ve kill için
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (ValueError, OSError):
            # Windows'ta bazı signal'lar çalışmayabilir
            pass
            
    def _signal_handler(self, signum, frame):
        """Signal yakalandığında state kaydet."""
        logger.info(f"🛑 Signal {signum} alındı, state kaydediliyor...")
        self.save()
        raise SystemExit(0)
        
    def _on_exit(self):
        """Program çıkışında state kaydet."""
        if self._dirty:
            self.save()
            
    def load(self) -> bool:
        """
        State dosyasından yükle.
        
        Returns:
            True: Başarılı yükleme
            False: Dosya yok veya hata (default state kullanılır)
        """
        try:
            if not os.path.exists(self.state_file):
                logger.info("📂 State dosyası yok, yeni state oluşturuluyor")
                return False
                
            with open(self.state_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                
            # son_sinyal_zamani'daki string tarihleri datetime'a çevir
            if "son_sinyal_zamani" in loaded:
                for coin, time_str in loaded["son_sinyal_zamani"].items():
                    try:
                        loaded["son_sinyal_zamani"][coin] = datetime.fromisoformat(time_str)
                    except (ValueError, TypeError):
                        pass  # Geçersiz format, atla
                        
            # Mevcut state'e merge et (eksik keyler için default değerler korunur)
            self._state.update(loaded)
            self._state["last_load_time"] = datetime.now().isoformat()
            
            logger.info(f"✅ State yüklendi: {len(self._state.get('son_sinyal_zamani', {}))} coin, "
                       f"{len(self._state.get('bugunun_sinyalleri', []))} sinyal")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ State JSON hatası: {e}")
            self._try_restore_backup()
            return False
        except Exception as e:
            logger.error(f"❌ State yükleme hatası: {e}")
            return False
            
    def _try_restore_backup(self):
        """Backup dosyasından geri yükle."""
        if os.path.exists(self.backup_file):
            try:
                import shutil
                shutil.copy(self.backup_file, self.state_file)
                logger.info("🔄 Backup'tan geri yüklendi")
                self.load()  # Tekrar dene
            except Exception as e:
                logger.error(f"❌ Backup geri yükleme hatası: {e}")
                
    def save(self) -> bool:
        """
        State'i dosyaya kaydet.
        
        Returns:
            True: Başarılı kayıt
            False: Hata
        """
        try:
            # Önce mevcut dosyayı backup'la
            if os.path.exists(self.state_file):
                import shutil
                shutil.copy(self.state_file, self.backup_file)
                
            # datetime objelerini string'e çevir
            state_to_save = self._prepare_for_save()
            
            # Atomik yazma - önce temp dosyaya yaz, sonra rename
            temp_file = f"{self.state_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(state_to_save, f, indent=2, ensure_ascii=False)
                
            # Rename (atomik işlem)
            os.replace(temp_file, self.state_file)
            
            self._dirty = False
            logger.debug(f"💾 State kaydedildi: {self.state_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ State kaydetme hatası: {e}")
            return False
            
    def _prepare_for_save(self) -> dict:
        """State'i JSON-serializable hale getir."""
        state_copy = self._state.copy()
        
        # datetime'ları string'e çevir
        if "son_sinyal_zamani" in state_copy:
            state_copy["son_sinyal_zamani"] = {
                coin: (dt.isoformat() if isinstance(dt, datetime) else dt)
                for coin, dt in state_copy["son_sinyal_zamani"].items()
            }
            
        state_copy["last_save_time"] = datetime.now().isoformat()
        return state_copy
        
    # ==========================================
    # Property Accessors
    # ==========================================
    
    @property
    def son_sinyal_zamani(self) -> Dict[str, datetime]:
        """Her coin için son sinyal zamanı."""
        return self._state.get("son_sinyal_zamani", {})
        
    @son_sinyal_zamani.setter
    def son_sinyal_zamani(self, value: Dict[str, datetime]):
        self._state["son_sinyal_zamani"] = value
        self._dirty = True
        
    def set_sinyal_zamani(self, coin: str, zaman: datetime):
        """Tek bir coin için sinyal zamanı güncelle."""
        self._state["son_sinyal_zamani"][coin] = zaman
        self._dirty = True
        
    @property
    def bugunun_sinyalleri(self) -> List[tuple]:
        """Bugün üretilen sinyaller listesi."""
        return self._state.get("bugunun_sinyalleri", [])
        
    @bugunun_sinyalleri.setter
    def bugunun_sinyalleri(self, value: List[tuple]):
        self._state["bugunun_sinyalleri"] = value
        self._dirty = True
        
    def add_sinyal(self, tarih: str, coin: str, yon: str):
        """Bugünün sinyallerine ekle."""
        self._state["bugunun_sinyalleri"].append((tarih, coin, yon))
        self._dirty = True
        
    @property
    def son_rapor_tarihi(self) -> Optional[str]:
        """Son günlük rapor tarihi."""
        return self._state.get("son_rapor_tarihi")
        
    @son_rapor_tarihi.setter
    def son_rapor_tarihi(self, value: str):
        self._state["son_rapor_tarihi"] = value
        self._dirty = True
        
    # ==========================================
    # Kill Switch State
    # ==========================================
    
    def set_kill_switch(self, active: bool, reason: str = ""):
        """Kill switch durumunu güncelle."""
        self._state["kill_switch_active"] = active
        self._state["kill_switch_reason"] = reason
        self._state["kill_switch_time"] = datetime.now().isoformat() if active else None
        self._dirty = True
        
    def get_kill_switch_state(self) -> tuple:
        """Kill switch durumunu al."""
        return (
            self._state.get("kill_switch_active", False),
            self._state.get("kill_switch_reason", ""),
            self._state.get("kill_switch_time")
        )
        
    # ==========================================
    # Günlük Reset
    # ==========================================
    
    def daily_reset(self):
        """Yeni gün için sinyalleri sıfırla."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Sadece bugüne ait olmayan sinyalleri temizle
        self._state["bugunun_sinyalleri"] = [
            s for s in self._state.get("bugunun_sinyalleri", [])
            if s[0] == today
        ]
        self._dirty = True
        self.save()
        logger.info("🔄 Günlük sinyal listesi sıfırlandı")
        
    # ==========================================
    # Utility
    # ==========================================
    
    def get_summary(self) -> str:
        """State özeti."""
        return (
            f"Coins: {len(self.son_sinyal_zamani)} | "
            f"Sinyaller: {len(self.bugunun_sinyalleri)} | "
            f"KillSwitch: {'🔴' if self._state.get('kill_switch_active') else '🟢'}"
        )


# Singleton instance
_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Global StateManager instance'ı al."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
        _state_manager.load()
    return _state_manager



# Convenience exports
def save_state():
    """State'i kaydet."""
    get_state_manager().save()
    
def load_state():
    """State'i yükle."""
    get_state_manager().load()

def periodic_save(positions=None, signals=None, cooldowns=None):
    """
    Periyodik state kaydetme helper fonksiyonu.
    
    Args:
        positions: (Kullanılmıyor, DB'de tutuluyor)
        signals: Bugünün sinyalleri listesi
        cooldowns: Coin cooldown (son sinyal zamanı) sözlüğü
    """
    mgr = get_state_manager()
    
    if signals is not None:
        mgr.bugunun_sinyalleri = signals
        
    if cooldowns is not None:
        mgr.son_sinyal_zamani = cooldowns
        
    mgr.save()

# Global export
state_manager = get_state_manager()


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.DEBUG)
    
    mgr = StateManager("test_state.json")
    
    # Test data
    mgr.set_sinyal_zamani("BTC", datetime.now())
    mgr.add_sinyal("2026-02-05", "ETH", "LONG")
    mgr.son_rapor_tarihi = "2026-02-04"
    
    # Save
    mgr.save()
    print(f"State saved: {mgr.get_summary()}")
    
    # Load fresh
    mgr2 = StateManager("test_state.json")
    mgr2.load()
    print(f"State loaded: {mgr2.get_summary()}")
    print(f"BTC last signal: {mgr2.son_sinyal_zamani.get('BTC')}")
    
    # Cleanup
    os.remove("test_state.json")
    if os.path.exists("test_state.backup.json"):
        os.remove("test_state.backup.json")
