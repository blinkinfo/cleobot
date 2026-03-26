"""Candle-based feature calculations for CleoBot.

Implements all ~30 candle features from Section 5.1 of the master plan:
- Returns (multi-lookback)
- Volatility (rolling std, Garman-Klass, Parkinson, ATR)
- Momentum indicators (RSI, MACD, Stochastic, Williams %R, ROC)
- Trend (EMA crossovers, ADX, Aroon)
- Candle patterns (body ratios, wicks, doji, streaks, position)
- Volume features (delta, trend, VWAP deviation)
"""

import numpy as np
import pandas as pd
from typing import Dict

from src.utils.logger import get_logger

logger = get_logger("features.candle")


def _safe_divide(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """Safe division avoiding div-by-zero."""
    return a.div(b.replace(0, np.nan)).fillna(fill)


def compute_candle_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Compute all candle-based features.

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
            sorted ascending by timestamp. Must have at least 50 rows.

    Returns:
        Dict mapping feature name -> pd.Series (same index as df).
    """
    feats: Dict[str, pd.Series] = {}
    idx = df.index

    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    v = df["volume"]

    # ------------------------------------------------------------------ #
    # RETURNS
    # ------------------------------------------------------------------ #
    log_ret = np.log(c / c.shift(1))
    feats["ret_1"] = log_ret
    feats["ret_3"] = np.log(c / c.shift(3))
    feats["ret_6"] = np.log(c / c.shift(6))
    feats["ret_12"] = np.log(c / c.shift(12))
    feats["ret_24"] = np.log(c / c.shift(24))

    # ------------------------------------------------------------------ #
    # VOLATILITY
    # ------------------------------------------------------------------ #
    for window in (6, 12, 24):
        feats[f"vol_std_{window}"] = log_ret.rolling(window).std()

    # Garman-Klass volatility: 0.5*(ln(H/L))^2 - (2*ln2-1)*(ln(C/O))^2
    hl_log = np.log(h / l)
    co_log = np.log(c / o)
    for window in (6, 12, 24):
        gk = 0.5 * hl_log.pow(2) - (2 * np.log(2) - 1) * co_log.pow(2)
        feats[f"vol_gk_{window}"] = gk.rolling(window).mean().apply(lambda x: np.sqrt(max(x, 0)))

    # Parkinson volatility: (1/(4*ln2)) * (ln(H/L))^2
    pk_factor = 1.0 / (4.0 * np.log(2))
    for window in (6, 12, 24):
        pk = pk_factor * hl_log.pow(2)
        feats[f"vol_pk_{window}"] = pk.rolling(window).mean().apply(lambda x: np.sqrt(max(x, 0)))

    # ATR (Average True Range)
    prev_c = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    for window in (6, 12, 24):
        feats[f"atr_{window}"] = tr.ewm(span=window, adjust=False).mean()

    # ------------------------------------------------------------------ #
    # MOMENTUM INDICATORS
    # ------------------------------------------------------------------ #

    # RSI
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs = _safe_divide(gain, loss, fill=0.0)
        return 100 - 100 / (1 + rs)

    feats["rsi_6"] = _rsi(c, 6)
    feats["rsi_14"] = _rsi(c, 14)

    # MACD (12, 26, 9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    feats["macd_line"] = macd_line
    feats["macd_signal"] = signal_line
    feats["macd_hist"] = macd_line - signal_line

    # Stochastic K and D (14, 3)
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    stoch_k = _safe_divide(c - low14, high14 - low14) * 100
    feats["stoch_k"] = stoch_k
    feats["stoch_d"] = stoch_k.rolling(3).mean()

    # Williams %R (14)
    feats["williams_r"] = _safe_divide(high14 - c, high14 - low14) * -100

    # Rate of Change
    feats["roc_6"] = _safe_divide(c - c.shift(6), c.shift(6)) * 100
    feats["roc_12"] = _safe_divide(c - c.shift(12), c.shift(12)) * 100

    # ------------------------------------------------------------------ #
    # TREND
    # ------------------------------------------------------------------ #
    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()

    # EMA crossovers: distance normalised by price
    feats["ema9_21_dist"] = _safe_divide(ema9 - ema21, c)
    feats["ema21_50_dist"] = _safe_divide(ema21 - ema50, c)

    # ADX (14)
    dm_plus = (h - h.shift(1)).clip(lower=0)
    dm_minus = (l.shift(1) - l).clip(lower=0)
    # Directional movement: only count if respective DM is dominant
    pos_dm = dm_plus.where(dm_plus > dm_minus, 0.0)
    neg_dm = dm_minus.where(dm_minus > dm_plus, 0.0)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    di_plus = _safe_divide(pos_dm.ewm(span=14, adjust=False).mean(), atr14) * 100
    di_minus = _safe_divide(neg_dm.ewm(span=14, adjust=False).mean(), atr14) * 100
    dx = _safe_divide((di_plus - di_minus).abs(), di_plus + di_minus) * 100
    feats["adx_14"] = dx.ewm(span=14, adjust=False).mean()

    # Aroon Up and Down (25)
    aroon_high = h.rolling(26).apply(lambda x: x.argmax() / 25 * 100, raw=True)
    aroon_low = l.rolling(26).apply(lambda x: x.argmin() / 25 * 100, raw=True)
    feats["aroon_up"] = aroon_high
    feats["aroon_down"] = aroon_low

    # ------------------------------------------------------------------ #
    # CANDLE PATTERNS
    # ------------------------------------------------------------------ #
    total_range = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    feats["body_ratio"] = _safe_divide(body, total_range)
    feats["upper_wick_ratio"] = _safe_divide(h - pd.concat([c, o], axis=1).max(axis=1), total_range)
    feats["lower_wick_ratio"] = _safe_divide(pd.concat([c, o], axis=1).min(axis=1) - l, total_range)
    feats["is_doji"] = (body < 0.1 * total_range.fillna(0)).astype(float)

    # Consecutive same-direction candles
    direction = np.sign(c - o)  # +1 up, -1 down, 0 neutral

    def _consec_streak(directions: np.ndarray) -> np.ndarray:
        streak = np.zeros(len(directions))
        for i in range(1, len(directions)):
            if directions[i] == directions[i - 1] and directions[i] != 0:
                streak[i] = streak[i - 1] + 1
            else:
                streak[i] = 0
        return streak

    feats["consec_candles"] = pd.Series(
        _consec_streak(direction.values), index=idx
    )

    # Current candle position within recent range (percentile)
    rolling_high = h.rolling(24).max()
    rolling_low = l.rolling(24).min()
    feats["price_position"] = _safe_divide(c - rolling_low, rolling_high - rolling_low)

    # ------------------------------------------------------------------ #
    # VOLUME
    # ------------------------------------------------------------------ #
    vol_avg12 = v.rolling(12).mean()
    feats["volume_delta"] = _safe_divide(v - vol_avg12, vol_avg12)

    # Volume trend (slope of last 6 candles, normalised by mean)
    vol_mean6 = v.rolling(6).mean()
    feats["volume_trend"] = _safe_divide(
        v.rolling(6).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 6 else 0.0,
            raw=True,
        ),
        vol_mean6.replace(0, np.nan),
    )

    # VWAP (cumulative within the last 12 candles as a rolling approx)
    # VWAP = sum(typical_price * volume) / sum(volume)
    typical_price = (h + l + c) / 3
    vwap = (
        (typical_price * v).rolling(12).sum()
        / v.rolling(12).sum().replace(0, np.nan)
    )
    feats["vwap_dev"] = _safe_divide(c - vwap, vwap)

    # ------------------------------------------------------------------ #
    # Fill NaN with 0 for startup periods and return
    # ------------------------------------------------------------------ #
    result = {k: v.fillna(0.0) for k, v in feats.items()}
    logger.debug(f"Candle features computed: {len(result)} features")
    return result
