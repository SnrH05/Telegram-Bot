"""
TITANIUM Bot - Rate Limiter
============================
Exchange API rate limiting ve ban önleme.

Özellikler:
- Request/dakika limiti (KuCoin: 12 req/s)
- Exponential backoff retry
- 429 error handling
- Endpoint bazlı throttle

Kullanım:
    from rate_limiter import RateLimiter, rate_limited
    
    limiter = RateLimiter()
    
    @rate_limited(limiter)
    async def fetch_data():
        ...
"""

import asyncio
import time
import logging
from typing import Callable, Any, Optional
from functools import wraps
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter with exponential backoff.
    
    KuCoin limits:
    - REST API: 1800 requests per minute (30/s average)
    - Spot: Stricter limits per endpoint
    - Safe limit: 12 requests per second
    """
    
    def __init__(
        self,
        requests_per_second: float = 10.0,  # Safe default
        burst_size: int = 20,  # Max burst
        max_retries: int = 3,
        base_delay: float = 1.0
    ):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.max_retries = max_retries
        self.base_delay = base_delay
        
        # Token bucket
        self.tokens = burst_size
        self.last_update = time.monotonic()
        
        # Request history (for monitoring)
        self.request_times: deque = deque(maxlen=1000)
        
        # Error tracking
        self.consecutive_errors = 0
        self.last_429_time: Optional[float] = None
        self.cooldown_until: Optional[float] = None
        
        # Stats
        self.total_requests = 0
        self.total_retries = 0
        self.total_429s = 0
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
    def _add_tokens(self):
        """Token bucket'a token ekle."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.last_update = now
        
        # Elapsed süreye göre token ekle
        new_tokens = elapsed * self.requests_per_second
        self.tokens = min(self.burst_size, self.tokens + new_tokens)
        
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Token al, gerekirse bekle.
        
        Returns:
            True: Token alındı
            False: Cooldown'dayız, işlem yapılmamalı
        """
        async with self._lock:
            # Cooldown kontrolü
            if self.cooldown_until:
                now = time.monotonic()
                if now < self.cooldown_until:
                    wait_time = self.cooldown_until - now
                    logger.warning(f"⏳ Rate limit cooldown: {wait_time:.1f}s kaldı")
                    await asyncio.sleep(wait_time)
                else:
                    self.cooldown_until = None
                    
            # Token ekle
            self._add_tokens()
            
            # Yeterli token var mı?
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.request_times.append(time.monotonic())
                self.total_requests += 1
                return True
                
            # Token yok, bekle
            wait_time = (tokens - self.tokens) / self.requests_per_second
            logger.debug(f"⏳ Rate limit: {wait_time:.2f}s bekleniyor")
            await asyncio.sleep(wait_time)
            
            # Tekrar dene
            self._add_tokens()
            self.tokens -= tokens
            self.request_times.append(time.monotonic())
            self.total_requests += 1
            return True
            
    def on_success(self):
        """Başarılı request sonrası."""
        self.consecutive_errors = 0
        
    def on_error(self, status_code: Optional[int] = None):
        """
        Hata sonrası backoff uygula.
        
        Args:
            status_code: HTTP status kodu (429 = rate limit)
        """
        self.consecutive_errors += 1
        
        if status_code == 429:
            self.total_429s += 1
            self.last_429_time = time.monotonic()
            
            # 429 alınca agresif cooldown
            cooldown = min(60, 5 * (2 ** self.consecutive_errors))
            self.cooldown_until = time.monotonic() + cooldown
            logger.warning(f"🚨 429 Rate Limit! {cooldown}s cooldown başlatıldı")
            
            # Token'ları sıfırla
            self.tokens = 0
            
    def get_backoff_delay(self) -> float:
        """Exponential backoff delay hesapla."""
        delay = self.base_delay * (2 ** self.consecutive_errors)
        return min(delay, 30.0)  # Max 30s
        
    def get_stats(self) -> dict:
        """Rate limiter istatistikleri."""
        # Son 60 saniyedeki request sayısı
        now = time.monotonic()
        recent_requests = sum(1 for t in self.request_times if now - t < 60)
        
        return {
            "total_requests": self.total_requests,
            "total_retries": self.total_retries,
            "total_429s": self.total_429s,
            "requests_per_minute": recent_requests,
            "current_tokens": round(self.tokens, 2),
            "consecutive_errors": self.consecutive_errors,
            "cooldown_active": self.cooldown_until is not None and time.monotonic() < self.cooldown_until
        }
        
    def get_rate_info(self) -> str:
        """Rate limit bilgisi (log için)."""
        stats = self.get_stats()
        return f"Rate: {stats['requests_per_minute']}/min | 429s: {stats['total_429s']} | Tokens: {stats['current_tokens']:.1f}"


def rate_limited(limiter: RateLimiter, tokens: int = 1):
    """
    Async fonksiyonu rate limit ile wrap et.
    
    Usage:
        @rate_limited(my_limiter)
        async def my_api_call():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            retries = 0
            last_error = None
            
            while retries <= limiter.max_retries:
                # Token al
                await limiter.acquire(tokens)
                
                try:
                    result = await func(*args, **kwargs)
                    limiter.on_success()
                    return result
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    
                    # Rate limit hatası mı?
                    is_rate_limit = (
                        "429" in error_str or 
                        "rate" in error_str or
                        "too many" in error_str
                    )
                    
                    if is_rate_limit:
                        limiter.on_error(429)
                    else:
                        limiter.on_error()
                        
                    retries += 1
                    limiter.total_retries += 1
                    
                    if retries <= limiter.max_retries:
                        delay = limiter.get_backoff_delay()
                        logger.warning(f"⚠️ API Hatası, {delay:.1f}s sonra retry ({retries}/{limiter.max_retries}): {e}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"❌ API Hatası, max retry aşıldı: {e}")
                        raise
                        
            raise last_error
            
        return wrapper
    return decorator


class EndpointRateLimiter:
    """
    Endpoint bazlı farklı rate limit.
    
    Bazı endpoint'ler daha sıkı limite sahip.
    """
    
    # Endpoint kategorileri ve limitleri
    LIMITS = {
        "fetch_ohlcv": 6.0,       # Yüksek veri - daha yavaş
        "fetch_ticker": 15.0,     # Hafif endpoint - daha hızlı
        "fetch_balance": 3.0,     # Hesap endpoint - çok yavaş
        "create_order": 5.0,      # Trade endpoint
        "default": 10.0
    }
    
    def __init__(self):
        self.limiters: dict[str, RateLimiter] = {}
        
    def get_limiter(self, endpoint: str) -> RateLimiter:
        """Endpoint için limiter al veya oluştur."""
        if endpoint not in self.limiters:
            rate = self.LIMITS.get(endpoint, self.LIMITS["default"])
            self.limiters[endpoint] = RateLimiter(requests_per_second=rate)
        return self.limiters[endpoint]
        
    async def acquire(self, endpoint: str, tokens: int = 1) -> bool:
        """Endpoint için token al."""
        limiter = self.get_limiter(endpoint)
        return await limiter.acquire(tokens)


# Global instance
_rate_limiter: Optional[RateLimiter] = None
_endpoint_limiter: Optional[EndpointRateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
    

def get_endpoint_limiter() -> EndpointRateLimiter:
    """Global endpoint limiter instance."""
    global _endpoint_limiter
    if _endpoint_limiter is None:
        _endpoint_limiter = EndpointRateLimiter()
    return _endpoint_limiter


# ==========================================
# CCXT Wrapper
# ==========================================

class RateLimitedExchange:
    """
    CCXT exchange wrapper with rate limiting.
    
    Usage:
        exchange = ccxt.kucoin(config)
        rl_exchange = RateLimitedExchange(exchange)
        
        ohlcv = await rl_exchange.fetch_ohlcv("BTC/USDT", "1h")
    """
    
    def __init__(self, exchange, limiter: Optional[RateLimiter] = None):
        self.exchange = exchange
        self.limiter = limiter or get_rate_limiter()
        self.endpoint_limiter = get_endpoint_limiter()
        
    async def _call(self, method: str, *args, **kwargs):
        """Rate limited API call."""
        await self.endpoint_limiter.acquire(method)
        
        func = getattr(self.exchange, method)
        retries = 0
        
        while retries <= self.limiter.max_retries:
            try:
                result = await func(*args, **kwargs)
                self.limiter.on_success()
                return result
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in error_str or "rate" in error_str
                
                if is_rate_limit:
                    self.limiter.on_error(429)
                else:
                    self.limiter.on_error()
                    
                retries += 1
                
                if retries <= self.limiter.max_retries:
                    delay = self.limiter.get_backoff_delay()
                    logger.warning(f"⚠️ {method} hatası, {delay:.1f}s retry: {e}")
                    await asyncio.sleep(delay)
                else:
                    raise
                    
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100):
        return await self._call("fetch_ohlcv", symbol, timeframe, limit=limit)
        
    async def fetch_ticker(self, symbol: str):
        return await self._call("fetch_ticker", symbol)
        
    async def fetch_balance(self):
        return await self._call("fetch_balance")
        
    async def close(self):
        return await self.exchange.close()
        
    def __getattr__(self, name):
        """Diğer metodlar için fallback."""
        return getattr(self.exchange, name)


if __name__ == "__main__":
    # Test
    import asyncio
    
    logging.basicConfig(level=logging.DEBUG)
    
    async def test():
        limiter = RateLimiter(requests_per_second=2.0)
        
        @rate_limited(limiter)
        async def fake_api_call(n: int):
            print(f"Request {n} at {time.time():.2f}")
            return n * 2
            
        # 10 rapid request
        tasks = [fake_api_call(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        print(f"Results: {results}")
        print(f"Stats: {limiter.get_stats()}")
        
    asyncio.run(test())
