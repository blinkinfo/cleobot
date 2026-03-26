"""Funding rate feature calculations for CleoBot.

Implements all 8 funding rate features from Section 5.3 of the master plan:
1. Current funding rate
2. Funding rate momentum (change over last 3 periods)
3. Distance to next funding settlement (time-based)
4. Funding rate vs 24h average
5. Funding rate vs 7d average
6. Funding rate percentile (vs 7d range)
7. Funding rate direction (positive/negative/neutral)
8. Funding rate acceleration
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from src.utils.logger import get_logger

logger = get_logger("features.funding")

# MEXC perpetual funding interval is 8 hours
FUNDING_INTERVAL_HOURS = 8
NEUTRAL_THRESHOLD = 1e-5  # rates within ±0.001% are "neutral"


def compute_funding_features(
    funding_records: List[Dict[str, Any]],
    current_ts_ms: Optional[int] = None,
) -> Dict[str, float]:
    """Compute all funding rate features.

    Args:
        funding_records: List of funding rate dicts (ascending timestamp),
                         each with keys: 'timestamp' (ms), 'rate' (float),
                         'next_settlement' (ms, optional).
        current_ts_ms: Current timestamp in milliseconds. Defaults to now.

    Returns:
        Dict of feature_name -> float value.
    """
    feats: Dict[str, float] = {}

    if current_ts_ms is None:
        current_ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Default zero features when no data is available
    _defaults = {
        "funding_rate": 0.0,
        "funding_momentum": 0.0,
        "funding_time_to_settlement": 0.5,
        "funding_vs_24h_avg": 0.0,
        "funding_vs_7d_avg": 0.0,
        "funding_pctile_7d": 0.5,
        "funding_direction": 0.0,
        "funding_acceleration": 0.0,
    }

    if not funding_records:
        logger.debug("No funding rate data available, using defaults.")
        return _defaults.copy()

    # Sort ascending by timestamp (should already be, but ensure)
    records = sorted(funding_records, key=lambda r: r["timestamp"])
    rates = [r["rate"] for r in records]
    rates_arr = np.array(rates, dtype=float)

    # ------------------------------------------------------------------ #
    # 1. Current funding rate
    # ------------------------------------------------------------------ #
    current_rate = rates_arr[-1]
    feats["funding_rate"] = float(current_rate)

    # ------------------------------------------------------------------ #
    # 2. Funding rate momentum (change over last 3 periods)
    # ------------------------------------------------------------------ #
    if len(rates_arr) >= 4:
        feats["funding_momentum"] = float(rates_arr[-1] - rates_arr[-4])
    elif len(rates_arr) >= 2:
        feats["funding_momentum"] = float(rates_arr[-1] - rates_arr[0])
    else:
        feats["funding_momentum"] = 0.0

    # ------------------------------------------------------------------ #
    # 3. Distance to next funding settlement (normalised 0-1)
    # MEXC settles every 8 hours at 00:00, 08:00, 16:00 UTC
    # ------------------------------------------------------------------ #
    latest_record = records[-1]
    next_settlement = latest_record.get("next_settlement")
    if next_settlement and next_settlement > current_ts_ms:
        time_remaining_ms = next_settlement - current_ts_ms
        interval_ms = FUNDING_INTERVAL_HOURS * 3600 * 1000
        feats["funding_time_to_settlement"] = float(
            np.clip(time_remaining_ms / interval_ms, 0.0, 1.0)
        )
    else:
        # Estimate from current time: find seconds until next 8h mark
        now_dt = datetime.fromtimestamp(current_ts_ms / 1000, tz=timezone.utc)
        hour = now_dt.hour
        minute = now_dt.minute
        second = now_dt.second
        current_secs = hour * 3600 + minute * 60 + second
        interval_secs = FUNDING_INTERVAL_HOURS * 3600
        elapsed_in_interval = current_secs % interval_secs
        remaining_secs = interval_secs - elapsed_in_interval
        feats["funding_time_to_settlement"] = float(remaining_secs / interval_secs)

    # ------------------------------------------------------------------ #
    # 4. Funding rate vs 24h average
    # 24h = 3 funding periods (at 8h each)
    # ------------------------------------------------------------------ #
    periods_24h = 3
    window_24h = rates_arr[-periods_24h:] if len(rates_arr) >= periods_24h else rates_arr
    avg_24h = float(np.mean(window_24h))
    feats["funding_vs_24h_avg"] = float(current_rate - avg_24h)

    # ------------------------------------------------------------------ #
    # 5. Funding rate vs 7d average
    # 7d = 21 funding periods
    # ------------------------------------------------------------------ #
    periods_7d = 21
    window_7d = rates_arr[-periods_7d:] if len(rates_arr) >= periods_7d else rates_arr
    avg_7d = float(np.mean(window_7d))
    feats["funding_vs_7d_avg"] = float(current_rate - avg_7d)

    # ------------------------------------------------------------------ #
    # 6. Funding rate percentile (vs 7d range)
    # ------------------------------------------------------------------ #
    if len(window_7d) > 1:
        sorted_7d = np.sort(window_7d)
        pctile = float(
            np.searchsorted(sorted_7d, current_rate) / len(sorted_7d)
        )
        feats["funding_pctile_7d"] = np.clip(pctile, 0.0, 1.0)
    else:
        feats["funding_pctile_7d"] = 0.5

    # ------------------------------------------------------------------ #
    # 7. Funding rate direction: +1 positive, -1 negative, 0 neutral
    # ------------------------------------------------------------------ #
    if current_rate > NEUTRAL_THRESHOLD:
        feats["funding_direction"] = 1.0
    elif current_rate < -NEUTRAL_THRESHOLD:
        feats["funding_direction"] = -1.0
    else:
        feats["funding_direction"] = 0.0

    # ------------------------------------------------------------------ #
    # 8. Funding rate acceleration (2nd derivative)
    # ------------------------------------------------------------------ #
    if len(rates_arr) >= 3:
        # First differences
        d1 = np.diff(rates_arr[-4:]) if len(rates_arr) >= 4 else np.diff(rates_arr[-3:])
        if len(d1) >= 2:
            feats["funding_acceleration"] = float(d1[-1] - d1[-2])
        else:
            feats["funding_acceleration"] = 0.0
    else:
        feats["funding_acceleration"] = 0.0

    logger.debug(f"Funding features computed: {len(feats)} features")
    return feats
