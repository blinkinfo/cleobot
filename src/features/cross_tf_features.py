"""Cross-timeframe feature calculations for CleoBot.

Implements all 10 cross-timeframe features from Section 5.4:
1.  15m candle direction (up/down based on current forming candle)
2.  1h candle direction
3.  Alignment score: does 5m direction align with 15m and 1h?
4.  15m RSI
5.  1h RSI
6.  Volatility ratio: 5m ATR / 1h ATR (normalised)
7.  15m trend strength (EMA9 vs EMA21 distance)
8.  1h trend strength
9.  Higher timeframe support/resistance proximity (distance to 1h swing high/low)
10. Multi-timeframe momentum alignment (all TFs trending same direction?)
"""

import numpy as np
import pandas as pd
from typing import Dict

from src.utils.logger import get_logger

logger = get_logger("features.cross_tf")


def _safe_divide(a, b, fill: float = 0.0):
    """Safe division."""
    if isinstance(b, pd.Series):
        return a.div(b.replace(0, np.nan)).fillna(fill)
    return a / b if b != 0 else fill


def _rsi(series: pd.Series, period: int) -> pd.Series:
    """Exponential-smoothed RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain.div(loss.replace(0, np.nan)).fillna(0)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range."""
    prev_c = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_c).abs(),
        (df["low"] - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_cross_tf_features(
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
) -> Dict[str, float]:
    """Compute cross-timeframe features for the current (latest) candle.

    Args:
        df_5m:  DataFrame of 5m candles (columns: open, high, low, close, volume).
        df_15m: DataFrame of 15m candles (same schema).
        df_1h:  DataFrame of 1h candles (same schema).

    Returns:
        Dict of feature_name -> float for the most recent bar.
    """
    feats: Dict[str, float] = {}

    # Helpers for safe tail access
    def _last(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
        if df is None or len(df) == 0:
            return default
        return float(df[col].iloc[-1])

    # ------------------------------------------------------------------ #
    # 1. 15m candle direction
    # ------------------------------------------------------------------ #
    if df_15m is not None and len(df_15m) >= 1:
        last_15m_open = _last(df_15m, "open")
        last_15m_close = _last(df_15m, "close")
        feats["tf15m_direction"] = 1.0 if last_15m_close > last_15m_open else -1.0
    else:
        feats["tf15m_direction"] = 0.0

    # ------------------------------------------------------------------ #
    # 2. 1h candle direction
    # ------------------------------------------------------------------ #
    if df_1h is not None and len(df_1h) >= 1:
        last_1h_open = _last(df_1h, "open")
        last_1h_close = _last(df_1h, "close")
        feats["tf1h_direction"] = 1.0 if last_1h_close > last_1h_open else -1.0
    else:
        feats["tf1h_direction"] = 0.0

    # ------------------------------------------------------------------ #
    # 3. Alignment score: 5m vs 15m vs 1h
    # ------------------------------------------------------------------ #
    if df_5m is not None and len(df_5m) >= 1:
        last_5m_open = _last(df_5m, "open")
        last_5m_close = _last(df_5m, "close")
        dir_5m = 1.0 if last_5m_close > last_5m_open else -1.0
    else:
        dir_5m = 0.0

    alignment = 0
    if dir_5m != 0:
        if feats["tf15m_direction"] == dir_5m:
            alignment += 1
        if feats["tf1h_direction"] == dir_5m:
            alignment += 1
    feats["tf_alignment_score"] = float(alignment)

    # ------------------------------------------------------------------ #
    # 4. 15m RSI (14)
    # ------------------------------------------------------------------ #
    if df_15m is not None and len(df_15m) >= 15:
        feats["tf15m_rsi"] = float(_rsi(df_15m["close"], 14).iloc[-1])
    else:
        feats["tf15m_rsi"] = 50.0

    # ------------------------------------------------------------------ #
    # 5. 1h RSI (14)
    # ------------------------------------------------------------------ #
    if df_1h is not None and len(df_1h) >= 15:
        feats["tf1h_rsi"] = float(_rsi(df_1h["close"], 14).iloc[-1])
    else:
        feats["tf1h_rsi"] = 50.0

    # ------------------------------------------------------------------ #
    # 6. Volatility ratio: 5m ATR(14) / 1h ATR(14) normalised
    # ------------------------------------------------------------------ #
    atr_5m, atr_1h = 0.0, 0.0
    if df_5m is not None and len(df_5m) >= 15:
        atr_5m = float(_atr(df_5m, 14).iloc[-1])
    if df_1h is not None and len(df_1h) >= 15:
        atr_1h = float(_atr(df_1h, 14).iloc[-1])
    feats["tf_vol_ratio"] = _safe_divide(atr_5m, atr_1h, fill=1.0)

    # ------------------------------------------------------------------ #
    # 7. 15m trend strength (EMA9 vs EMA21 distance, normalised by price)
    # ------------------------------------------------------------------ #
    if df_15m is not None and len(df_15m) >= 22:
        ema9_15m = df_15m["close"].ewm(span=9, adjust=False).mean().iloc[-1]
        ema21_15m = df_15m["close"].ewm(span=21, adjust=False).mean().iloc[-1]
        price_15m = _last(df_15m, "close", 1.0)
        feats["tf15m_trend_strength"] = _safe_divide(
            float(ema9_15m - ema21_15m), price_15m, fill=0.0
        )
    else:
        feats["tf15m_trend_strength"] = 0.0

    # ------------------------------------------------------------------ #
    # 8. 1h trend strength
    # ------------------------------------------------------------------ #
    if df_1h is not None and len(df_1h) >= 22:
        ema9_1h = df_1h["close"].ewm(span=9, adjust=False).mean().iloc[-1]
        ema21_1h = df_1h["close"].ewm(span=21, adjust=False).mean().iloc[-1]
        price_1h = _last(df_1h, "close", 1.0)
        feats["tf1h_trend_strength"] = _safe_divide(
            float(ema9_1h - ema21_1h), price_1h, fill=0.0
        )
    else:
        feats["tf1h_trend_strength"] = 0.0

    # ------------------------------------------------------------------ #
    # 9. Proximity to 1h swing high/low (distance normalised by ATR)
    # ------------------------------------------------------------------ #
    if df_1h is not None and len(df_1h) >= 24 and atr_1h > 0:
        swing_high = float(df_1h["high"].rolling(24).max().iloc[-1])
        swing_low = float(df_1h["low"].rolling(24).min().iloc[-1])
        current_price = _last(df_1h, "close", 0.0)
        if current_price > 0:
            dist_to_high = (swing_high - current_price) / atr_1h
            dist_to_low = (current_price - swing_low) / atr_1h
            # Encode as position between 0 (at low) and 1 (at high)
            span = dist_to_high + dist_to_low
            feats["tf1h_sr_proximity"] = _safe_divide(dist_to_low, span, fill=0.5)
        else:
            feats["tf1h_sr_proximity"] = 0.5
    else:
        feats["tf1h_sr_proximity"] = 0.5

    # ------------------------------------------------------------------ #
    # 10. Multi-timeframe momentum alignment
    #     +1 if all agree (up), -1 if all agree (down), 0 otherwise
    # ------------------------------------------------------------------ #
    dirs = [dir_5m, feats["tf15m_direction"], feats["tf1h_direction"]]
    nonzero = [d for d in dirs if d != 0]
    if len(nonzero) == 3 and len(set(nonzero)) == 1:
        feats["tf_momentum_alignment"] = float(nonzero[0])
    else:
        feats["tf_momentum_alignment"] = 0.0

    logger.debug(f"Cross-TF features computed: {len(feats)} features")
    return feats
