"""Time-based feature calculations for CleoBot.

Implements all 8 time features from Section 5.5:
1. Hour of day (sin encoding)
2. Hour of day (cos encoding)
3. Day of week (sin encoding)
4. Day of week (cos encoding)
5. Minutes since last >0.5% move
6. Minutes since last >1% move
7. Time to next funding settlement
8. Is within first/last 30 min of a major session (US, EU, Asia)?
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timezone

from src.utils.logger import get_logger

logger = get_logger("features.time")

# Major session open/close times in UTC hours
# US:   13:30 - 20:00 UTC (NYSE)
# EU:   07:00 - 15:30 UTC (Frankfurt/London)
# Asia: 00:00 - 09:00 UTC (Tokyo/Shanghai)
SESSION_WINDOWS = [
    (7 * 60, 7 * 60 + 30),       # EU open (07:00-07:30)
    (15 * 60, 15 * 60 + 30),      # EU close / US open (15:00-15:30)
    (13 * 60 + 30, 14 * 60),      # US open (13:30-14:00)
    (19 * 60 + 30, 20 * 60),      # US close (19:30-20:00)
    (0 * 60, 0 * 60 + 30),        # Asia open (00:00-00:30)
    (8 * 60 + 30, 9 * 60),        # Asia close (08:30-09:00)
]

# Funding settlement times: 00:00, 08:00, 16:00 UTC
FUNDING_HOURS = [0, 8, 16]
FUNDING_INTERVAL_SECS = 8 * 3600


def compute_time_features(
    current_ts_ms: Optional[int] = None,
    df_5m: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Compute all time-based features.

    Args:
        current_ts_ms: Current UTC timestamp in milliseconds.
                       Defaults to now().
        df_5m: DataFrame of 5m candles with 'close' column (ascending).
               Used to compute time-since-last-big-move features.

    Returns:
        Dict of feature_name -> float.
    """
    feats: Dict[str, float] = {}

    if current_ts_ms is None:
        current_ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    now_dt = datetime.fromtimestamp(current_ts_ms / 1000, tz=timezone.utc)
    hour = now_dt.hour
    minute = now_dt.minute
    dow = now_dt.weekday()  # Monday=0, Sunday=6
    current_mins = hour * 60 + minute  # minutes since midnight UTC
    current_secs = hour * 3600 + minute * 60 + now_dt.second

    # ------------------------------------------------------------------ #
    # 1 & 2. Hour of day (sin/cos encoding)
    # ------------------------------------------------------------------ #
    hour_frac = (hour + minute / 60.0) / 24.0  # 0-1
    feats["time_hour_sin"] = float(np.sin(2 * np.pi * hour_frac))
    feats["time_hour_cos"] = float(np.cos(2 * np.pi * hour_frac))

    # ------------------------------------------------------------------ #
    # 3 & 4. Day of week (sin/cos encoding)
    # ------------------------------------------------------------------ #
    dow_frac = dow / 7.0
    feats["time_dow_sin"] = float(np.sin(2 * np.pi * dow_frac))
    feats["time_dow_cos"] = float(np.cos(2 * np.pi * dow_frac))

    # ------------------------------------------------------------------ #
    # 5 & 6. Minutes since last >0.5% and >1% move
    # ------------------------------------------------------------------ #
    if df_5m is not None and len(df_5m) >= 2:
        closes = df_5m["close"].values
        timestamps = df_5m["timestamp"].values if "timestamp" in df_5m.columns else None

        # Compute absolute returns
        returns = np.abs(np.diff(closes) / closes[:-1])

        def _mins_since_threshold(threshold: float) -> float:
            # Find last index where return exceeded threshold
            indices = np.where(returns >= threshold)[0]
            if len(indices) == 0:
                return 120.0  # cap at 2 hours if never exceeded
            last_idx = int(indices[-1])  # index in returns array (= candle index + 1)
            # Each 5m candle = 5 minutes
            if timestamps is not None and len(timestamps) > last_idx + 1:
                elapsed_ms = (current_ts_ms - int(timestamps[last_idx + 1]))
                return float(np.clip(elapsed_ms / 60000, 0, 120))
            else:
                candles_ago = len(returns) - 1 - last_idx
                return float(np.clip(candles_ago * 5, 0, 120))

        feats["time_since_05pct_move"] = _mins_since_threshold(0.005)
        feats["time_since_1pct_move"] = _mins_since_threshold(0.010)
    else:
        feats["time_since_05pct_move"] = 60.0
        feats["time_since_1pct_move"] = 60.0

    # ------------------------------------------------------------------ #
    # 7. Time to next funding settlement (normalised 0-1)
    # ------------------------------------------------------------------ #
    elapsed_in_interval = current_secs % FUNDING_INTERVAL_SECS
    remaining_secs = FUNDING_INTERVAL_SECS - elapsed_in_interval
    feats["time_to_funding"] = float(remaining_secs / FUNDING_INTERVAL_SECS)

    # ------------------------------------------------------------------ #
    # 8. Is within first/last 30 min of a major session? (0 or 1)
    # ------------------------------------------------------------------ #
    in_session_window = any(
        start <= current_mins < end
        for start, end in SESSION_WINDOWS
    )
    feats["time_session_window"] = 1.0 if in_session_window else 0.0

    logger.debug(f"Time features computed: {len(feats)} features")
    return feats
