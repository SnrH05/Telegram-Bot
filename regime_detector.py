"""
TITANIUM Regime Detector Module
================================
Market regime classification for adaptive trading.

Regimes:
- TREND: Strong directional movement (ADX > 30)
- RANGE: Sideways consolidation (ADX < 20)
- NO_TRADE: Extreme volatility/chaos (stay out)
- MIXED: Unclear conditions (conservative mode)

Author: TITANIUM Bot V5.9
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from enum import Enum


class MarketRegime(Enum):
    """Market regime classifications."""
    TREND = "TREND"
    RANGE = "RANGE"
    NO_TRADE = "NO_TRADE"
    MIXED = "MIXED"


class RegimeDetector:
    """
    Detect current market regime and adjust trading parameters.
    
    Uses 4H timeframe for regime detection (less noise than 1H).
    """
    
    def __init__(self):
        # Thresholds
        self.ADX_TREND_THRESHOLD = 30
        self.ADX_RANGE_THRESHOLD = 20
        self.VOL_SPIKE_THRESHOLD = 2.0
        self.VOL_LOW_THRESHOLD = 0.8
        
        # Position size multipliers by regime
        self.REGIME_SIZE_MULT = {
            MarketRegime.TREND: 1.0,
            MarketRegime.RANGE: 0.7,
            MarketRegime.MIXED: 0.5,
            MarketRegime.NO_TRADE: 0.0
        }
        
        # Signal threshold adjustments by regime
        self.REGIME_THRESHOLD_ADJ = {
            MarketRegime.TREND: 0,      # Use base threshold
            MarketRegime.RANGE: 5,      # +5 to threshold (stricter)
            MarketRegime.MIXED: 10,     # +10 to threshold (more strict)
            MarketRegime.NO_TRADE: 999  # Impossible to reach
        }
        
        # Cache
        self._current_regime = MarketRegime.MIXED
        self._last_update = None
        
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX indicator."""
        df = df.copy()
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = abs(df['high'] - df['close'].shift(1))
        df['l-pc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        df['atr'] = df['tr'].rolling(period).mean()
        
        # Prevent division by zero
        df['atr'] = df['atr'].replace(0, 1e-10)
        
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        df['plus_di'] = 100 * (df['plus_dm'].ewm(alpha=1/period).mean() / df['atr'])
        df['minus_di'] = 100 * (df['minus_dm'].ewm(alpha=1/period).mean() / df['atr'])
        
        di_sum = df['plus_di'] + df['minus_di']
        di_sum = di_sum.replace(0, 1e-10)
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / di_sum
        df['adx'] = df['dx'].ewm(alpha=1/period).mean()
        
        return df['adx'].fillna(0)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        df = df.copy()
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = abs(df['high'] - df['close'].shift(1))
        df['l-pc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        return df['tr'].rolling(period).mean()
    
    def calculate_bollinger_width(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Bollinger Band width as percentage."""
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        
        # Width as percentage of price
        width = ((upper - lower) / sma) * 100
        return width.fillna(0)
    
    def detect_regime(self, df_4h: pd.DataFrame) -> Tuple[MarketRegime, dict]:
        """
        Detect current market regime from 4H data.
        
        Args:
            df_4h: 4H OHLCV DataFrame (minimum 60 candles recommended)
            
        Returns:
            (regime, details_dict)
        """
        if len(df_4h) < 30:
            return MarketRegime.MIXED, {"error": "Insufficient data"}
        
        # Calculate indicators
        adx = self.calculate_adx(df_4h).iloc[-1]
        atr = self.calculate_atr(df_4h)
        atr_current = atr.iloc[-1]
        atr_sma = atr.rolling(20).mean().iloc[-1]
        bb_width = self.calculate_bollinger_width(df_4h).iloc[-1]
        bb_width_sma = self.calculate_bollinger_width(df_4h).rolling(20).mean().iloc[-1]
        
        # Volatility ratio
        vol_ratio = atr_current / atr_sma if atr_sma > 0 else 1.0
        bb_ratio = bb_width / bb_width_sma if bb_width_sma > 0 else 1.0
        
        # Price trend (simple check)
        sma_20 = df_4h['close'].rolling(20).mean().iloc[-1]
        sma_50 = df_4h['close'].rolling(50).mean().iloc[-1]
        price = df_4h['close'].iloc[-1]
        
        trend_aligned = (price > sma_20 > sma_50) or (price < sma_20 < sma_50)
        
        details = {
            "adx": round(adx, 2),
            "vol_ratio": round(vol_ratio, 2),
            "bb_ratio": round(bb_ratio, 2),
            "trend_aligned": trend_aligned,
            "atr_current": round(atr_current, 6),
            "bb_width": round(bb_width, 2)
        }
        
        # Regime classification logic
        
        # 1. NO_TRADE: Extreme volatility spike
        if vol_ratio > self.VOL_SPIKE_THRESHOLD:
            self._current_regime = MarketRegime.NO_TRADE
            details["reason"] = f"Volatility spike: {vol_ratio:.1f}x normal"
            return MarketRegime.NO_TRADE, details
        
        # 2. TREND: Strong directional movement
        if adx > self.ADX_TREND_THRESHOLD and vol_ratio > 1.0:
            self._current_regime = MarketRegime.TREND
            details["reason"] = f"Strong trend: ADX={adx:.0f}"
            return MarketRegime.TREND, details
        
        # 3. RANGE: Low volatility consolidation
        if adx < self.ADX_RANGE_THRESHOLD and vol_ratio < self.VOL_LOW_THRESHOLD:
            self._current_regime = MarketRegime.RANGE
            details["reason"] = f"Range-bound: ADX={adx:.0f}, vol={vol_ratio:.1f}x"
            return MarketRegime.RANGE, details
        
        # 4. MIXED: Everything else
        self._current_regime = MarketRegime.MIXED
        details["reason"] = "Mixed conditions"
        return MarketRegime.MIXED, details
    
    def get_position_size_mult(self, regime: MarketRegime) -> float:
        """Get position size multiplier for regime."""
        return self.REGIME_SIZE_MULT.get(regime, 0.5)
    
    def get_threshold_adjustment(self, regime: MarketRegime) -> int:
        """Get signal threshold adjustment for regime."""
        return self.REGIME_THRESHOLD_ADJ.get(regime, 10)
    
    def should_use_strategy(self, regime: MarketRegime, strategy: str) -> bool:
        """
        Check if a strategy should be used in current regime.
        
        Args:
            regime: Current market regime
            strategy: "TREND", "RANGE", or "RAPID"
            
        Returns:
            True if strategy should be active
        """
        strategy_map = {
            MarketRegime.TREND: ["TREND", "RAPID"],
            MarketRegime.RANGE: ["RANGE"],
            MarketRegime.MIXED: ["TREND"],  # Conservative - only trend
            MarketRegime.NO_TRADE: []        # No strategies
        }
        
        allowed = strategy_map.get(regime, [])
        return strategy.upper() in allowed


class PositionSizer:
    """
    Kelly-criterion inspired position sizing with safety caps.
    """
    
    def __init__(self, account_balance: float = 1000.0):
        self.account_balance = account_balance
        self.MAX_POSITION_PCT = 0.05  # 5% max per trade
        self.MIN_POSITION_PCT = 0.01  # 1% min per trade
        
    def update_balance(self, new_balance: float):
        """Update account balance."""
        self.account_balance = max(0, new_balance)
    
    def calculate_kelly(self, win_rate: float, avg_win_pct: float, avg_loss_pct: float) -> float:
        """
        Calculate Kelly fraction.
        
        f* = (bp - q) / b
        where:
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
        """
        if avg_loss_pct == 0 or win_rate <= 0 or win_rate >= 1:
            return self.MIN_POSITION_PCT
        
        b = abs(avg_win_pct / avg_loss_pct)
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Safety: cap at 25% and use half-Kelly
        kelly = max(0, min(kelly, 0.25)) * 0.5
        
        return kelly
    
    def calculate_position_size(
        self, 
        win_rate: float = 0.5, 
        avg_win_pct: float = 3.0,
        avg_loss_pct: float = 2.0,
        regime_mult: float = 1.0,
        drawdown_mult: float = 1.0
    ) -> float:
        """
        Calculate position size in USD.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win_pct: Average winning trade %
            avg_loss_pct: Average losing trade %
            regime_mult: Market regime multiplier
            drawdown_mult: Drawdown-based reduction multiplier
            
        Returns:
            Position size in USD
        """
        # Base Kelly sizing
        kelly_fraction = self.calculate_kelly(win_rate, avg_win_pct, avg_loss_pct)
        
        # Apply multipliers
        final_fraction = kelly_fraction * regime_mult * drawdown_mult
        
        # Clamp to limits
        final_fraction = max(self.MIN_POSITION_PCT, min(final_fraction, self.MAX_POSITION_PCT))
        
        position_size = self.account_balance * final_fraction
        
        return round(position_size, 2)


class SlippageModel:
    """
    Model expected slippage based on order size, volatility, and liquidity.
    """
    
    # Tier-1: Most liquid
    LIQUID_COINS = {"BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE"}
    # Tier-2: Medium liquidity
    MEDIUM_COINS = {"AVAX", "LINK", "DOT", "LTC", "BCH", "TON"}
    # Tier-3: Lower liquidity (everything else)
    
    def __init__(self):
        self.BASE_SLIPPAGE = 0.0005  # 0.05% base
        
    def estimate_slippage(
        self, 
        coin: str, 
        position_usd: float, 
        volatility_pct: float
    ) -> float:
        """
        Estimate expected slippage percentage.
        
        Args:
            coin: Coin symbol (e.g., "BTC")
            position_usd: Order size in USD
            volatility_pct: Current ATR as % of price
            
        Returns:
            Expected slippage as decimal (e.g., 0.001 = 0.1%)
        """
        # Size impact: larger orders = more slippage
        size_factor = 1 + (position_usd / 50000) * 0.1  # +10% per $50k
        
        # Volatility impact
        vol_factor = 1 + (volatility_pct / 2) * 0.2  # +20% per 2% volatility
        
        # Liquidity factor
        if coin in self.LIQUID_COINS:
            liquidity_factor = 1.0
        elif coin in self.MEDIUM_COINS:
            liquidity_factor = 1.3
        else:
            liquidity_factor = 1.8
        
        slippage = self.BASE_SLIPPAGE * size_factor * vol_factor * liquidity_factor
        
        # Cap at 1%
        return min(slippage, 0.01)
    
    def adjust_tp_sl(
        self,
        tp_price: float,
        sl_price: float,
        entry_price: float,
        direction: str,
        slippage_pct: float
    ) -> Tuple[float, float]:
        """
        Adjust TP/SL prices to account for slippage.
        
        Returns:
            (adjusted_tp, adjusted_sl)
        """
        if direction == "LONG":
            # TP slightly lower (harder to reach), SL wider (more protection)
            adjusted_tp = tp_price * (1 - slippage_pct)
            adjusted_sl = sl_price * (1 - slippage_pct * 2)
        else:  # SHORT
            adjusted_tp = tp_price * (1 + slippage_pct)
            adjusted_sl = sl_price * (1 + slippage_pct * 2)
        
        return adjusted_tp, adjusted_sl
