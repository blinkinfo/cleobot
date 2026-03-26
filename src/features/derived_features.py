"""Derived and interaction feature calculations for CleoBot.

Implements all ~15 derived/interaction features from Section 5.7:
1.  Orderbook imbalance * volatility regime (interaction)
2.  RSI divergence from price direction
3.  Volume-weighted momentum (momentum * relative volume)
4.  Trend alignment * confidence (cross-feature)
5.  Mean reversion signal (VWAP distance * low-vol regime indicator)
6.  Momentum exhaustion (RSI extreme + declining volume + long streak)
7.  Breakout signal (volatility compression + orderbook imbalance surge)
8-17. Feature z-scores for top 10 features (normalised values)

These features are computed from other already-computed feature groups
and require a combined feature dict as input.
"""

import numpy as np
from typing import Dict, Any

from src.utils.logger import get_logger

logger = get_logger("features.derived")

# Names of the top-10 features to z-score (selected by expected LightGBM importance)
# These are the features most likely to carry highest signal per the master plan.
TOP_FEATURES_FOR_ZSCORE = [
    "ob_imbalance_5",
    "ob_net_pressure",
    "ob_imbalance_change_30s",
    "funding_rate",
    "rsi_14",
    "atr_12",
    "macd_hist",
    "ret_1",
    "tf_alignment_score",
    "ob_spread_bps",
]

# Rolling z-score window (number of candles)
ZSCORE_WINDOW = 50


def _safe_get(features: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Safely get a float feature value."""
    val = features.get(key, default)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)


def _zscore(value: float, history: list, window: int = ZSCORE_WINDOW) -> float:
    """Compute z-score of value given a rolling history list."""
    recent = history[-window:] if len(history) >= window else history
    if len(recent) < 2:
        return 0.0
    arr = np.array(recent, dtype=float)
    std = float(np.std(arr))
    mean = float(np.mean(arr))
    if std < 1e-10:
        return 0.0
    return float(np.clip((value - mean) / std, -5.0, 5.0))


def compute_derived_features(
    features: Dict[str, float],
    feature_history: Dict[str, list] = None,
) -> Dict[str, float]:
    """Compute all derived and interaction features.

    Args:
        features: Combined dict of all previously computed features
                  (candle + orderbook + funding + cross_tf + time + polymarket).
        feature_history: Dict mapping feature_name -> list of historical values
                         (most recent last). Used for z-score computation.
                         If None or insufficient, z-scores default to 0.

    Returns:
        Dict of derived feature_name -> float.
    """
    feats: Dict[str, float] = {}
    h = feature_history or {}

    # ------------------------------------------------------------------ #
    # 1. Orderbook imbalance * volatility regime interaction
    # Regime: vol_std_12 percentile relative to its own history
    # ------------------------------------------------------------------ #
    ob_imb = _safe_get(features, "ob_imbalance_5")
    vol_std12 = _safe_get(features, "vol_std_12")
    # Normalise vol_std to [-1, 1] using z-score if history available
    vol_history = h.get("vol_std_12", [])
    if len(vol_history) >= 10:
        vol_z = _zscore(vol_std12, vol_history)
        vol_regime = float(np.tanh(vol_z))  # maps to (-1, 1)
    else:
        vol_regime = 0.0
    feats["derived_ob_vol_interaction"] = float(ob_imb * (1.0 + abs(vol_regime)))

    # ------------------------------------------------------------------ #
    # 2. RSI divergence from price direction
    # Bullish divergence: price falling (ret_1 < 0) but RSI rising vs recent
    # Bearish divergence: price rising (ret_1 > 0) but RSI falling vs recent
    # ------------------------------------------------------------------ #
    ret1 = _safe_get(features, "ret_1")
    rsi14 = _safe_get(features, "rsi_14", 50.0)
    rsi_history = h.get("rsi_14", [])
    if len(rsi_history) >= 3:
        rsi_change = rsi14 - float(np.mean(rsi_history[-3:]))
        # Divergence: price and RSI moving in opposite directions
        feats["derived_rsi_divergence"] = float(
            np.sign(rsi_change) * np.sign(-ret1) * min(abs(rsi_change) / 10.0, 1.0)
        )
    else:
        feats["derived_rsi_divergence"] = 0.0

    # ------------------------------------------------------------------ #
    # 3. Volume-weighted momentum (momentum * relative volume)
    # ------------------------------------------------------------------ #
    roc6 = _safe_get(features, "roc_6")
    vol_delta = _safe_get(features, "volume_delta")  # relative volume vs avg
    # vol_delta > 0 means above-average volume -- amplifies momentum
    # Use (1 + vol_delta) so normal volume = neutral multiplier
    vol_multiplier = 1.0 + float(np.clip(vol_delta, -0.9, 3.0))
    feats["derived_vol_weighted_momentum"] = float(roc6 * vol_multiplier)

    # ------------------------------------------------------------------ #
    # 4. Trend alignment * regime confidence
    # Cross-feature: alignment score scaled by ADX
    # ------------------------------------------------------------------ #
    alignment = _safe_get(features, "tf_alignment_score")  # 0, 1, or 2
    adx = _safe_get(features, "adx_14", 20.0)
    # ADX > 25 = trending, normalise to 0-1 scale
    adx_norm = float(np.clip(adx / 50.0, 0.0, 1.0))
    feats["derived_trend_alignment_strength"] = float(
        (alignment / 2.0) * adx_norm
    )  # 0 when no alignment or flat, up to 1 when perfect alignment + strong trend

    # ------------------------------------------------------------------ #
    # 5. Mean reversion signal
    # Strength = VWAP distance * low-vol regime indicator
    # ------------------------------------------------------------------ #
    vwap_dev = _safe_get(features, "vwap_dev")
    # Low vol regime: vol_std_12 in bottom quartile of history
    if len(vol_history) >= 10:
        vol_pctile = float(
            np.searchsorted(np.sort(vol_history[-50:]), vol_std12)
            / min(len(vol_history), 50)
        )
        low_vol_indicator = float(np.clip(1.0 - vol_pctile * 2, 0.0, 1.0))  # 1 if low vol
    else:
        low_vol_indicator = 0.5
    # Mean reversion: price far from VWAP + low vol -> expect reversion
    feats["derived_mean_reversion"] = float(-vwap_dev * low_vol_indicator)

    # ------------------------------------------------------------------ #
    # 6. Momentum exhaustion
    # RSI extreme (>70 or <30) + declining volume + long streak
    # ------------------------------------------------------------------ #
    is_rsi_extreme = 1.0 if (rsi14 > 70 or rsi14 < 30) else 0.0
    is_vol_declining = 1.0 if vol_delta < -0.2 else 0.0
    consec = abs(_safe_get(features, "consec_candles"))
    is_long_streak = 1.0 if consec >= 3 else 0.0
    exhaustion_score = (is_rsi_extreme + is_vol_declining + is_long_streak) / 3.0
    # Direction: if RSI > 70 (overbought) and streak is up -> bearish exhaustion
    if rsi14 > 70:
        feats["derived_momentum_exhaustion"] = float(-exhaustion_score)
    elif rsi14 < 30:
        feats["derived_momentum_exhaustion"] = float(exhaustion_score)
    else:
        feats["derived_momentum_exhaustion"] = 0.0

    # ------------------------------------------------------------------ #
    # 7. Breakout signal
    # Volatility compression followed by orderbook imbalance surge
    # ------------------------------------------------------------------ #
    # Compression: recent ATR below rolling average
    atr12 = _safe_get(features, "atr_12")
    atr_history = h.get("atr_12", [])
    if len(atr_history) >= 10:
        atr_avg = float(np.mean(atr_history[-24:]))
        is_compressed = 1.0 if (atr_avg > 0 and atr12 < 0.8 * atr_avg) else 0.0
    else:
        is_compressed = 0.0
    # Imbalance surge: ob_imbalance_5 > recent average
    ob_history = h.get("ob_imbalance_5", [])
    if len(ob_history) >= 5:
        ob_avg = float(np.mean(ob_history[-12:]))
        ob_surge = float(abs(ob_imb) - abs(ob_avg))
        is_surge = 1.0 if ob_surge > 0.1 else 0.0
    else:
        is_surge = 0.0
    # Breakout = compression + surge, direction from imbalance
    feats["derived_breakout_signal"] = float(
        np.sign(ob_imb) * is_compressed * is_surge
    )

    # ------------------------------------------------------------------ #
    # 8-17. Feature z-scores for top 10 features
    # ------------------------------------------------------------------ #
    for feat_name in TOP_FEATURES_FOR_ZSCORE:
        value = _safe_get(features, feat_name, 0.0)
        history = h.get(feat_name, [])
        z = _zscore(value, history)
        feats[f"z_{feat_name}"] = z

    logger.debug(f"Derived features computed: {len(feats)} features")
    return feats
