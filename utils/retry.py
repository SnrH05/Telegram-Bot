"""
TITANIUM Bot - Retry Utilities
==============================
API çağrıları için exponential backoff retry mekanizması.
"""

import asyncio
import functools
import logging
from typing import Callable, Any, TypeVar, Optional
import ccxt

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_async(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
    rate_limit_delay: float = 60.0
):
    """
    Async fonksiyonlar için exponential backoff retry decorator.
    
    Args:
        max_retries: Maksimum deneme sayısı
        base_delay: İlk bekleme süresi (saniye)
        max_delay: Maksimum bekleme süresi (saniye)
        exceptions: Yakalanacak exception türleri
        rate_limit_delay: Rate limit hatalarında bekleme süresi
    
    Usage:
        @retry_async(max_retries=3)
        async def fetch_data():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                    
                except ccxt.RateLimitExceeded as e:
                    # Rate limit - uzun bekleme
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"⚠️ Rate limit! Bekleme: {rate_limit_delay}s (deneme {attempt+1}/{max_retries})")
                        await asyncio.sleep(rate_limit_delay)
                    else:
                        raise
                        
                except ccxt.NetworkError as e:
                    # Network hatası - exponential backoff
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(f"⚠️ Network hatası: {e} - Bekleme: {delay:.1f}s (deneme {attempt+1}/{max_retries})")
                        await asyncio.sleep(delay)
                    else:
                        raise
                        
                except ccxt.ExchangeError as e:
                    # Exchange hatası - genellikle retry'a değmez
                    logger.error(f"❌ Exchange hatası: {e}")
                    raise
                    
                except exceptions as e:
                    # Genel hatalar
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(f"⚠️ Hata: {e} - Bekleme: {delay:.1f}s (deneme {attempt+1}/{max_retries})")
                        await asyncio.sleep(delay)
                    else:
                        raise
            
            # Bu noktaya ulaşılmamalı ama yine de
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


def retry_sync(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,)
):
    """
    Senkron fonksiyonlar için exponential backoff retry decorator.
    """
    import time
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(f"⚠️ Retry: {e} - Bekleme: {delay:.1f}s ({attempt+1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        raise
            
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


class RateLimitedClient:
    """
    Rate limiting ile sarılmış API client helper.
    
    Usage:
        client = RateLimitedClient(exchange, calls_per_minute=60)
        await client.call(exchange.fetch_ohlcv, "BTC/USDT", "1h")
    """
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Rate limited API çağrısı"""
        import time
        
        # Son çağrıdan bu yana geçen süre
        elapsed = time.time() - self.last_call_time
        
        # Gerekirse bekle
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        
        self.last_call_time = time.time()
        return await func(*args, **kwargs)


if __name__ == "__main__":
    # Test
    @retry_async(max_retries=3, base_delay=1.0)
    async def test_func():
        import random
        if random.random() < 0.5:
            raise ValueError("Random error")
        return "Success"
    
    async def main():
        try:
            result = await test_func()
            print(f"Result: {result}")
        except ValueError as e:
            print(f"Failed: {e}")
    
    asyncio.run(main())
