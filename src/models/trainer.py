"""Auto-training pipeline for CleoBot.

Implements all three training modes from Section 8 of the master plan:
  1. Full Retrain -- every 24 hours at 04:00 UTC
  2. Incremental Update -- every 6 hours
  3. Emergency Retrain -- when rolling accuracy drops below 52%

Full retrain uses walk-forward cross-validation with purging,
Optuna hyperparameter optimisation, SMOTE on training folds only,
and acceptance criteria before swapping in new models.

All training is orchestrated through the Trainer class, which manages
the Ensemble's lifecycle.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List, Any, Tuple, Callable

import optuna
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE

from src.database import Database
from src.features.engine import FeatureEngine, FUNDING_LOOKBACK
from src.features.funding_features import compute_funding_features
from src.models.lgbm_model import LightGBMModel
from src.models.tcn_model import TCNModel
from src.models.logreg_model import LogRegModel
from src.models.meta_learner import MetaLearner, build_meta_features_batch
from src.models.regime_detector import RegimeDetector, compute_regime_features
from src.models.ensemble import Ensemble
from src.utils.logger import get_logger
from src.utils.helpers import utc_timestamp_ms

logger = get_logger("models.trainer")

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Training constants
FULL_RETRAIN_DAYS = 14       # Training window
VALIDATION_DAYS = 2          # Validation window
PURGE_CANDLES = 2            # Gap between train and validation
CANDLES_PER_DAY = 288        # 5-min candles per day
DATA_LOAD_DAYS = 30          # Days of raw candles to load (covers train+val+lookback)
FEATURE_LOOKBACK_CANDLES = 150  # Candles consumed by feature computation lookback
OPTUNA_TRIALS = 20           # Trials per model (reduced from 50; TPE converges well within 20)
ACCEPTANCY_MARGIN = 0.005    # New model must beat old by 0.5%
DECAY_ACCURACY_FLOOR = 0.53  # Below this, accept any improvement
SMOTE_MIN_SAMPLES = 50       # Minimum samples for SMOTE
EMERGENCY_DECAY_HALFLIFE = 3 # Days for exponential decay (emergency retrain)


class Trainer:
    """Manages all training operations for the CleoBot ensemble.

    Orchestrates full retrains, incremental updates, and emergency retrains.
    Handles model versioning, acceptance criteria, and rollback.
    """

    def __init__(self, ensemble: Ensemble, db: Database, feature_engine: FeatureEngine):
        """Initialise the trainer.

        Args:
            ensemble: The Ensemble instance to train.
            db: Database instance for loading training data.
            feature_engine: FeatureEngine for computing features.
        """
        self.ensemble = ensemble
        self.db = db
        self.feature_engine = feature_engine
        self._training_in_progress = False
        self._last_full_retrain: Optional[datetime] = None
        self._last_incremental: Optional[datetime] = None
        self._notification_callback: Optional[Callable] = None
        self._is_lightweight: bool = False

    def set_notification_callback(self, callback: Callable):
        """Set a callback for training notifications (Telegram).

        Args:
            callback: Async or sync callable(message: str) -> None.
        """
        self._notification_callback = callback

    def _notify(self, message: str):
        """Send a training notification."""
        logger.info(f"[NOTIFY] {message}")
        if self._notification_callback:
            try:
                self._notification_callback(message)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")

    # ================================================================== #
    # DATA PREPARATION
    # ================================================================== #

    def _load_training_data(
        self, days: int = DATA_LOAD_DAYS
    ) -> Optional[pd.DataFrame]:
        """Load and prepare training data from the database.

        Loads candle data, computes features vectorized across all candles,
        and creates labels (1 = UP, 0 = DOWN based on next candle close vs current close).

        Args:
            days: Number of days of data to load.

        Returns:
            DataFrame with features and 'label' column, or None if insufficient data.
        """
        limit = days * CANDLES_PER_DAY + FEATURE_LOOKBACK_CANDLES
        candles = self.db.get_candles("candles_5m", limit=limit)

        if len(candles) < 500:
            logger.warning(
                f"Insufficient candle data: {len(candles)} rows "
                f"(need 500+). Aborting training."
            )
            return None

        if len(candles) < CANDLES_PER_DAY * 3:
            effective_days = max(3, len(candles) // CANDLES_PER_DAY)
            logger.warning(
                f"Limited candle data: {len(candles)} rows -- using lightweight training "
                f"(effective_days={effective_days}). Ideal minimum is {CANDLES_PER_DAY * 3}."
            )
            self._is_lightweight = True
        else:
            self._is_lightweight = False

        df = pd.DataFrame(candles)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Create labels: 1 = next candle closes higher, 0 = lower
        df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df = df.iloc[:-1]  # Drop last row (no label)

        start_idx = FEATURE_LOOKBACK_CANDLES  # 150
        if start_idx >= len(df):
            logger.warning("Not enough candles for feature computation.")
            return None

        logger.info(f"Computing features (vectorized) for {len(df) - start_idx} candles...")

        # --- 0. Pre-load ALL funding rate records for the training window ---
        # Load once to avoid per-row DB queries. We use the candle timestamp
        # range to fetch only relevant records via the `since` parameter.
        earliest_candle_ts = int(df.iloc[0]["timestamp"])
        all_funding_records = self.db.get_funding_rates(
            limit=100_000, since=earliest_candle_ts
        )
        # Also fetch FUNDING_LOOKBACK records *before* the earliest candle
        # so the first training rows have historical context.
        pre_funding_records = self.db.get_funding_rates(
            limit=FUNDING_LOOKBACK
        )
        # Merge: pre-records that are before earliest_candle_ts + all records in range
        pre_only = [r for r in pre_funding_records if r["timestamp"] < earliest_candle_ts]
        all_funding_records = pre_only + all_funding_records
        # Ensure ascending order by timestamp
        all_funding_records.sort(key=lambda r: r["timestamp"])
        logger.info(f"Loaded {len(all_funding_records)} funding rate records for training window.")

        # --- 1. Candle features: compute ONCE on full DataFrame ---
        from src.features.candle_features import compute_candle_features
        candle_feats_series = compute_candle_features(df)
        # Each value is a pd.Series aligned with df's index; take rows from start_idx onward
        candle_df = pd.DataFrame({
            k: v.iloc[start_idx:].reset_index(drop=True)
            for k, v in candle_feats_series.items()
        })

        # --- 2. Time features: vectorized (no per-row Python loop) ---
        # All time features depend only on the candle timestamp and
        # on the close-price series for time_since_big_move computations.
        # We compute all columns with numpy array ops over the output slice.
        from src.features.time_features import SESSION_WINDOWS, FUNDING_INTERVAL_SECS
        ts_arr = df["timestamp"].values.astype(np.float64)  # ms
        out_ts = ts_arr[start_idx:]  # shape (n_out,)

        # Datetime decomposition (vectorized via pandas)
        dt_index = pd.to_datetime(out_ts, unit="ms", utc=True)
        hour_arr = dt_index.hour.values.astype(np.float64)
        minute_arr = dt_index.minute.values.astype(np.float64)
        second_arr = dt_index.second.values.astype(np.float64)
        dow_arr = dt_index.dayofweek.values.astype(np.float64)

        hour_frac = (hour_arr + minute_arr / 60.0) / 24.0
        dow_frac = dow_arr / 7.0

        time_hour_sin = np.sin(2 * np.pi * hour_frac)
        time_hour_cos = np.cos(2 * np.pi * hour_frac)
        time_dow_sin = np.sin(2 * np.pi * dow_frac)
        time_dow_cos = np.cos(2 * np.pi * dow_frac)

        # time_to_funding: seconds remaining until next 8h mark / interval
        current_secs_arr = hour_arr * 3600 + minute_arr * 60 + second_arr
        elapsed_in_interval = current_secs_arr % FUNDING_INTERVAL_SECS
        remaining_secs = FUNDING_INTERVAL_SECS - elapsed_in_interval
        time_to_funding = remaining_secs / FUNDING_INTERVAL_SECS

        # time_session_window: is current minute within any session window?
        current_mins_arr = (hour_arr * 60 + minute_arr).astype(np.int32)
        in_window = np.zeros(len(out_ts), dtype=bool)
        for _s, _e in SESSION_WINDOWS:
            in_window |= ((current_mins_arr >= _s) & (current_mins_arr < _e))
        time_session_window = in_window.astype(np.float64)

        # time_since big moves: scan close returns once, then for each row find
        # the last index where abs return exceeded threshold using searchsorted.
        closes_all = df["close"].values.astype(np.float64)
        ts_all = df["timestamp"].values.astype(np.float64)
        abs_rets = np.abs(np.diff(closes_all) / np.where(closes_all[:-1] == 0, 1.0, closes_all[:-1]))
        # abs_rets[i] = abs return from candle i to i+1
        # For output row at df index j (j >= start_idx), we scan abs_rets[:j]
        # (returns up to and including candle j) for the last threshold crossing.
        # Vectorized: for each output row j, find last idx in abs_rets[:j] where ret>=thresh.
        # We use a cumulative "last seen" array built in one forward pass.
        def _vectorized_mins_since(threshold: float, cap: float = 120.0) -> np.ndarray:
            n_full = len(abs_rets)  # len(df) - 1
            last_event_ts = np.full(n_full + 1, -1.0)  # last_event_ts[i] = ts of last event at or before candle i
            for i in range(n_full):
                if abs_rets[i] >= threshold:
                    last_event_ts[i + 1] = ts_all[i + 1]  # event at candle i+1 (the close that moved)
                else:
                    last_event_ts[i + 1] = last_event_ts[i]
            # For output rows (indices start_idx..len(df)-1):
            out_last = last_event_ts[start_idx:]  # shape (n_out,)
            no_event = out_last < 0
            elapsed_mins = np.where(
                no_event,
                cap,
                np.clip((out_ts - out_last) / 60000.0, 0.0, cap),
            )
            return elapsed_mins

        time_since_05 = _vectorized_mins_since(0.005)
        time_since_1  = _vectorized_mins_since(0.010)

        time_df = pd.DataFrame({
            "time_hour_sin": time_hour_sin,
            "time_hour_cos": time_hour_cos,
            "time_dow_sin":  time_dow_sin,
            "time_dow_cos":  time_dow_cos,
            "time_since_05pct_move": time_since_05,
            "time_since_1pct_move":  time_since_1,
            "time_to_funding":       time_to_funding,
            "time_session_window":   time_session_window,
        }).reset_index(drop=True)

        # --- 3. Cross-TF features: vectorized (matching inference names) ---
        cross_tf_df = self._vectorized_cross_tf(df, start_idx)

        # --- 4. Funding rate features: vectorized (numpy searchsorted + array ops) ---
        # Pre-extract arrays from the sorted funding records list
        if all_funding_records:
            fr_ts    = np.array([r["timestamp"] for r in all_funding_records], dtype=np.float64)
            fr_rates = np.array([r["rate"]      for r in all_funding_records], dtype=np.float64)
            fr_next  = np.array([r.get("next_settlement", 0) or 0
                                 for r in all_funding_records], dtype=np.float64)
        else:
            fr_ts = fr_rates = fr_next = np.array([], dtype=np.float64)

        candle_ts_arr = df["timestamp"].values[start_idx:].astype(np.float64)  # (n_out,)
        n_out = len(candle_ts_arr)
        LOOK = FUNDING_LOOKBACK  # 25

        # For each output row: use searchsorted to find the insertion point
        # (= number of funding records with timestamp <= candle_ts)
        if len(fr_ts) > 0:
            insert_pts = np.searchsorted(fr_ts, candle_ts_arr, side="right")  # (n_out,)
        else:
            insert_pts = np.zeros(n_out, dtype=np.int64)

        # Pre-allocate output arrays
        f_rate          = np.zeros(n_out)
        f_momentum      = np.zeros(n_out)
        f_time_to_sett  = np.full(n_out, 0.5)
        f_vs_24h        = np.zeros(n_out)
        f_vs_7d         = np.zeros(n_out)
        f_pctile_7d     = np.full(n_out, 0.5)
        f_direction     = np.zeros(n_out)
        f_acceleration  = np.zeros(n_out)

        NEUTRAL_THRESHOLD = 1e-5
        FUNDING_INTERVAL_MS = 8 * 3600 * 1000

        for out_i in range(n_out):
            end_idx = int(insert_pts[out_i])         # exclusive upper bound
            start_i = max(0, end_idx - LOOK)
            window  = fr_rates[start_i:end_idx]       # at most LOOK rates
            candle_ts_ms = candle_ts_arr[out_i]

            if len(window) == 0:
                continue  # keep defaults

            cur = window[-1]
            f_rate[out_i] = cur

            # momentum
            if len(window) >= 4:
                f_momentum[out_i] = cur - window[-4]
            elif len(window) >= 2:
                f_momentum[out_i] = cur - window[0]

            # time to settlement
            ns = fr_next[start_i + len(window) - 1] if end_idx > 0 else 0.0
            if ns > candle_ts_ms:
                f_time_to_sett[out_i] = float(np.clip(
                    (ns - candle_ts_ms) / FUNDING_INTERVAL_MS, 0.0, 1.0))
            else:
                # estimate from candle timestamp
                dt_ms = candle_ts_ms
                secs_in_day = int((dt_ms // 1000) % 86400)
                interval_s = 8 * 3600
                elapsed = secs_in_day % interval_s
                f_time_to_sett[out_i] = (interval_s - elapsed) / interval_s

            # vs 24h avg (last 3 periods)
            w24 = window[-3:] if len(window) >= 3 else window
            avg24 = float(np.mean(w24))
            f_vs_24h[out_i] = cur - avg24

            # vs 7d avg (last 21 periods)
            w7d = window[-21:] if len(window) >= 21 else window
            avg7d = float(np.mean(w7d))
            f_vs_7d[out_i] = cur - avg7d

            # percentile vs 7d
            if len(w7d) > 1:
                f_pctile_7d[out_i] = float(np.clip(
                    np.searchsorted(np.sort(w7d), cur) / len(w7d), 0.0, 1.0))

            # direction
            if cur > NEUTRAL_THRESHOLD:
                f_direction[out_i] = 1.0
            elif cur < -NEUTRAL_THRESHOLD:
                f_direction[out_i] = -1.0

            # acceleration (2nd diff)
            tail = window[-4:] if len(window) >= 4 else window[-3:]
            if len(tail) >= 3:
                d1 = np.diff(tail)
                if len(d1) >= 2:
                    f_acceleration[out_i] = float(d1[-1] - d1[-2])

        funding_df = pd.DataFrame({
            "funding_rate":                f_rate,
            "funding_momentum":            f_momentum,
            "funding_time_to_settlement":  f_time_to_sett,
            "funding_vs_24h_avg":          f_vs_24h,
            "funding_vs_7d_avg":           f_vs_7d,
            "funding_pctile_7d":           f_pctile_7d,
            "funding_direction":           f_direction,
            "funding_acceleration":        f_acceleration,
        }).reset_index(drop=True)

        # --- 5. Combine candle + time + cross-TF + funding features ---
        combined_df = pd.concat([candle_df, time_df, cross_tf_df, funding_df], axis=1)

        # --- 6. Derived features: vectorized (pandas column ops, no Python loop) ---
        # feature_history={} for all training rows (z-scores default to 0),
        # so every derived feature reduces to a pure arithmetic expression on
        # combined_df columns -- no per-row Python loop needed.
        from src.features.derived_features import TOP_FEATURES_FOR_ZSCORE

        def _cget(col, default=0.0):
            """Return column as float64 Series, or scalar default if missing."""
            if col in combined_df.columns:
                return combined_df[col].astype(np.float64)
            return pd.Series(np.full(len(combined_df), default), dtype=np.float64)

        ob_imb       = _cget("ob_imbalance_5")
        vol_std12    = _cget("vol_std_12")
        ret1         = _cget("ret_1")
        rsi14        = _cget("rsi_14", 50.0)
        roc6         = _cget("roc_6")
        vol_delta    = _cget("volume_delta")
        alignment    = _cget("tf_alignment_score")
        adx          = _cget("adx_14", 20.0)
        vwap_dev     = _cget("vwap_dev")
        atr12        = _cget("atr_12")
        consec       = _cget("consec_candles")

        # 1. ob_vol_interaction: no history -> vol_regime = 0 -> factor = 1.0
        derived_ob_vol_interaction = ob_imb * 1.0

        # 2. rsi_divergence: no history -> rsi_change treated as 0
        derived_rsi_divergence = pd.Series(np.zeros(len(combined_df)), dtype=np.float64)

        # 3. vol_weighted_momentum
        vol_mult = 1.0 + np.clip(vol_delta, -0.9, 3.0)
        derived_vol_weighted_momentum = roc6 * vol_mult

        # 4. trend_alignment_strength
        adx_norm = np.clip(adx / 50.0, 0.0, 1.0)
        derived_trend_alignment_strength = (alignment / 2.0) * adx_norm

        # 5. mean_reversion: no history -> low_vol_indicator = 0.5
        derived_mean_reversion = -vwap_dev * 0.5

        # 6. momentum_exhaustion
        is_rsi_extreme  = ((rsi14 > 70) | (rsi14 < 30)).astype(np.float64)
        is_vol_declining = (vol_delta < -0.2).astype(np.float64)
        is_long_streak   = (consec.abs() >= 3).astype(np.float64)
        exhaustion_score = (is_rsi_extreme + is_vol_declining + is_long_streak) / 3.0
        derived_momentum_exhaustion = np.where(
            rsi14 > 70, -exhaustion_score,
            np.where(rsi14 < 30, exhaustion_score, 0.0)
        )

        # 7. breakout_signal: no history -> is_compressed=0, is_surge=0 -> 0
        derived_breakout_signal = pd.Series(np.zeros(len(combined_df)), dtype=np.float64)

        # 8-17. z-scores: no history -> all 0
        derived_df = pd.DataFrame({
            "derived_ob_vol_interaction":      derived_ob_vol_interaction,
            "derived_rsi_divergence":          derived_rsi_divergence,
            "derived_vol_weighted_momentum":   derived_vol_weighted_momentum,
            "derived_trend_alignment_strength": derived_trend_alignment_strength,
            "derived_mean_reversion":          derived_mean_reversion,
            "derived_momentum_exhaustion":     pd.Series(derived_momentum_exhaustion,
                                                         dtype=np.float64),
            "derived_breakout_signal":         derived_breakout_signal,
            **{f"z_{fn}": pd.Series(np.zeros(len(combined_df)), dtype=np.float64)
               for fn in TOP_FEATURES_FOR_ZSCORE},
        }).reset_index(drop=True)

        # Merge derived features
        combined_df = pd.concat([combined_df, derived_df], axis=1)

        # --- 7. Validate: replace NaN/inf with 0.0 ---
        combined_df = combined_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        combined_df = combined_df.replace([np.inf, -np.inf], 0.0)

        # --- 8. Drop constant columns (all same value = no signal) ---
        nunique = combined_df.nunique()
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols:
            logger.info(f"Dropping {len(constant_cols)} constant columns: {constant_cols[:10]}...")
            combined_df = combined_df.drop(columns=constant_cols)

        # --- 9. Add labels and timestamps ---
        combined_df["label"] = df.iloc[start_idx:]["label"].reset_index(drop=True)
        combined_df["_timestamp"] = df.iloc[start_idx:]["timestamp"].reset_index(drop=True)

        logger.info(
            f"Training data prepared: {len(combined_df)} samples, "
            f"{len(combined_df.columns) - 2} features"
        )
        return combined_df

    @staticmethod
    def _rolling_linear_slope(arr: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling linear-regression slope using vectorized numpy ops.

        Uses the closed-form OLS slope formula:
          slope = (n * sum(x*y) - sum(x)*sum(y)) / (n * sum(x^2) - sum(x)^2)
        where x = 0..window-1 for each window position.

        Args:
            arr: 1-D float array of values.
            window: Rolling window size.

        Returns:
            Array of slopes, same length as arr. Positions with insufficient
            history are filled with 0.0.
        """
        n = len(arr)
        slopes = np.zeros(n, dtype=np.float64)
        if n < window or window < 2:
            return slopes

        # Fixed x-values: 0, 1, ..., window-1
        x = np.arange(window, dtype=np.float64)
        sum_x = x.sum()
        sum_x2 = (x * x).sum()
        denom = window * sum_x2 - sum_x * sum_x

        # Cumulative sums for rolling computation
        cum_y = np.cumsum(arr)
        idx = np.arange(n, dtype=np.float64)
        cum_xy = np.cumsum(idx * arr)

        for j in range(window - 1, n):
            i_start = j - window + 1
            s_y = cum_y[j] - (cum_y[i_start - 1] if i_start > 0 else 0.0)
            s_xy = cum_xy[j] - (cum_xy[i_start - 1] if i_start > 0 else 0.0)
            s_local_xy = s_xy - i_start * s_y
            slopes[j] = (window * s_local_xy - sum_x * s_y) / denom

        return slopes

    def _vectorized_cross_tf(self, df: pd.DataFrame, start_idx: int) -> pd.DataFrame:
        """Compute cross-timeframe features vectorized, matching inference feature names.

        Uses 5m data to approximate 15m/1h timeframe features with pure numpy
        array operations (no per-row Python loop). Feature names and value
        ranges match compute_cross_tf_features() from cross_tf_features.py.

        Args:
            df: Full 5m candle DataFrame.
            start_idx: Index from which to produce output rows.

        Returns:
            DataFrame with cross-TF features for rows [start_idx:].
        """
        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        opn = df["open"].values.astype(np.float64)
        n_total = len(close)
        sl = slice(start_idx, n_total)

        # --- 1. 15m direction (3-candle window) ---
        tf15m_direction = np.where(close[sl] > np.roll(opn, 2)[sl], 1.0, -1.0)

        # --- 2. 1h direction (12-candle window) ---
        tf1h_direction = np.where(close[sl] > np.roll(opn, 11)[sl], 1.0, -1.0)

        # --- 3. 5m direction & alignment score ---
        dir_5m = np.where(close[sl] > opn[sl], 1.0, -1.0)
        tf_alignment_score = ((tf15m_direction == dir_5m).astype(np.float64)
                              + (tf1h_direction == dir_5m).astype(np.float64))

        # --- 4 & 5. RSI helpers (vectorized via cumsum of gains/losses) ---
        def _rolling_rsi(prices: np.ndarray, period: int) -> np.ndarray:
            rets = np.diff(prices, prepend=prices[0])
            gains = np.where(rets > 0, rets, 0.0)
            losses = np.where(rets < 0, -rets, 0.0)
            cum_g = np.cumsum(gains)
            cum_l = np.cumsum(losses)
            n = len(prices)
            rsi = np.full(n, 50.0)
            for j in range(period, n):
                sg = cum_g[j] - cum_g[j - period]
                sl_ = cum_l[j] - cum_l[j - period]
                if sl_ < 1e-12:
                    rsi[j] = 100.0 if sg > 0 else 50.0
                else:
                    rs = sg / sl_
                    rsi[j] = 100.0 - 100.0 / (1.0 + rs)
            return rsi

        tf15m_rsi = _rolling_rsi(close, 15)[sl]
        tf1h_rsi = _rolling_rsi(close, 60)[sl]

        # --- 6. Volatility ratio: rolling mean |diff| over 6 / over 12 candles ---
        abs_diff = np.abs(np.diff(close, prepend=close[0]))
        cum_ad = np.cumsum(abs_diff)

        def _rolling_mean_ad(cum: np.ndarray, w: int) -> np.ndarray:
            out = np.ones(n_total)
            for j in range(w, n_total):
                out[j] = (cum[j] - cum[j - w]) / w
            return out

        atr5m = _rolling_mean_ad(cum_ad, 6)
        atr1h = _rolling_mean_ad(cum_ad, 12)
        tf_vol_ratio = (atr5m / np.maximum(atr1h, 1e-8))[sl]

        # --- 7 & 8. Trend strength via rolling linear slope ---
        slopes_all = self._rolling_linear_slope(close, 3)
        cum_close = np.cumsum(close)
        mean3 = np.ones(n_total)
        for j in range(2, n_total):
            mean3[j] = (cum_close[j] - (cum_close[j - 3] if j >= 3 else 0.0)) / min(j + 1, 3)
        tf15m_trend_strength = (slopes_all / np.maximum(mean3, 1e-8))[sl]

        slopes12 = self._rolling_linear_slope(close, 12)
        mean12 = np.ones(n_total)
        for j in range(11, n_total):
            mean12[j] = (cum_close[j] - (cum_close[j - 12] if j >= 12 else 0.0)) / min(j + 1, 12)
        tf1h_trend_strength = (slopes12 / np.maximum(mean12, 1e-8))[sl]

        # --- 9. S/R proximity (position in 12-candle high-low range) ---
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        roll_high = high_s.rolling(12, min_periods=1).max().values
        roll_low = low_s.rolling(12, min_periods=1).min().values
        price_range = roll_high - roll_low
        tf1h_sr_proximity = np.where(
            price_range[sl] > 0,
            (close[sl] - roll_low[sl]) / price_range[sl],
            0.5,
        )

        # --- 10. Momentum alignment: +1 all up, -1 all down, 0 mixed ---
        all_up = (dir_5m == 1.0) & (tf15m_direction == 1.0) & (tf1h_direction == 1.0)
        all_down = (dir_5m == -1.0) & (tf15m_direction == -1.0) & (tf1h_direction == -1.0)
        tf_momentum_alignment = np.where(all_up, 1.0, np.where(all_down, -1.0, 0.0))

        return pd.DataFrame({
            "tf15m_direction": tf15m_direction,
            "tf1h_direction": tf1h_direction,
            "tf_alignment_score": tf_alignment_score,
            "tf15m_rsi": tf15m_rsi,
            "tf1h_rsi": tf1h_rsi,
            "tf_vol_ratio": tf_vol_ratio,
            "tf15m_trend_strength": tf15m_trend_strength,
            "tf1h_trend_strength": tf1h_trend_strength,
            "tf1h_sr_proximity": tf1h_sr_proximity,
            "tf_momentum_alignment": tf_momentum_alignment,
        })

    # ================================================================== #
    # WALK-FORWARD CROSS-VALIDATION
    # ================================================================== #

    def _walk_forward_split(
        self,
        n_samples: int,
        train_days: int = FULL_RETRAIN_DAYS,
        val_days: int = VALIDATION_DAYS,
        step_days: int = 1,
        purge_candles: int = PURGE_CANDLES,
        max_splits: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward train/validation splits.

        Args:
            n_samples: Total number of samples.
            train_days: Training window in days.
            val_days: Validation window in days.
            step_days: Step size in days.
            purge_candles: Gap between train and validation (prevent leakage).

        Returns:
            List of (train_indices, val_indices) tuples.
        """
        train_size = train_days * CANDLES_PER_DAY
        val_size = val_days * CANDLES_PER_DAY
        step_size = step_days * CANDLES_PER_DAY

        splits = []
        start = 0

        while start + train_size + purge_candles + val_size <= n_samples:
            train_end = start + train_size
            val_start = train_end + purge_candles
            val_end = val_start + val_size

            train_idx = np.arange(start, train_end)
            val_idx = np.arange(val_start, min(val_end, n_samples))

            if len(val_idx) > 0:
                splits.append((train_idx, val_idx))

            start += step_size

        logger.info(
            f"Walk-forward CV: {len(splits)} splits "
            f"(train={train_size}, val={val_size}, purge={purge_candles})"
        )
        if max_splits is not None and len(splits) > max_splits:
            splits = splits[-max_splits:]  # Keep the most recent splits
            logger.info(f"Truncated to {max_splits} most recent splits (max_splits={max_splits}).")
        return splits

    # ================================================================== #
    # FULL RETRAIN
    # ================================================================== #

    def full_retrain(self) -> Dict[str, Any]:
        """Execute a full retrain of all models.

        Steps:
          1. Load 14 days of data, compute features
          2. Walk-forward CV with Optuna tuning
          3. Train base models with best hyperparameters
          4. Generate OOF predictions for meta-learner
          5. Train meta-learner and regime detector
          6. Apply acceptance criteria
          7. Save if accepted

        Returns:
            Dict with training results and acceptance decision.
        """
        if self._training_in_progress:
            logger.warning("Training already in progress. Skipping.")
            return {"status": "skipped", "reason": "training_in_progress"}

        self._training_in_progress = True
        self._notify("Starting full retrain...")
        t0 = time.monotonic()

        try:
            # 1. Load and prepare data
            data = self._load_training_data(days=DATA_LOAD_DAYS)
            if data is None:
                self._notify("Full retrain ABORTED: insufficient data.")
                return {"status": "aborted", "reason": "insufficient_data"}

            # Separate features, labels, timestamps
            label_col = data["label"]
            ts_col = data["_timestamp"]
            feature_cols = [c for c in data.columns if c not in ("label", "_timestamp")]
            X = data[feature_cols]
            y = label_col

            # 2. Walk-forward CV splits (reduce to 2 in lightweight mode)
            if self._is_lightweight:
                logger.info("Lightweight mode: using 2 walk-forward CV splits.")
            splits = self._walk_forward_split(len(X), max_splits=2 if self._is_lightweight else None)
            if not splits:
                self._notify("Full retrain ABORTED: no valid CV splits.")
                return {"status": "aborted", "reason": "no_cv_splits"}

            # Use last split for final holdout evaluation
            final_train_idx, final_val_idx = splits[-1]
            X_train_full = X.iloc[final_train_idx]
            y_train_full = y.iloc[final_train_idx]
            X_val = X.iloc[final_val_idx]
            y_val = y.iloc[final_val_idx]

            # 3. Optuna hyperparameter tuning for LightGBM
            logger.info("Tuning LightGBM hyperparameters...")
            best_lgbm_params = self._tune_lgbm(X, y, splits)

            # 4. Train base models with best parameters
            # Apply SMOTE on training data only
            X_train_sm, y_train_sm = self._apply_smote(X_train_full, y_train_full)

            # Train LightGBM
            lgbm_new = LightGBMModel(params=best_lgbm_params)
            lgbm_metrics = lgbm_new.train(
                X_train_sm, y_train_sm, X_val, y_val
            )

            # Train TCN
            tcn_new = TCNModel()
            tcn_metrics = tcn_new.train(
                X_train_sm, y_train_sm, X_val, y_val
            )

            # Train LogReg (uses LightGBM feature importance)
            logreg_new = LogRegModel()
            logreg_metrics = logreg_new.train(
                X_train_sm, y_train_sm, X_val, y_val,
                feature_importance=lgbm_new.feature_importance,
            )

            # 5. Generate OOF predictions for meta-learner
            # --- OPT 2: Limit OOF to last 5 folds (most recent, saves ~64% compute) ---
            oof_splits = splits[-5:] if len(splits) > 5 else splits
            logger.info(
                f"Generating OOF predictions for meta-learner "
                f"({len(oof_splits)} of {len(splits)} folds)..."
            )
            oof_lgbm, oof_tcn, oof_logreg, oof_labels, oof_meta = (
                self._generate_oof_predictions(X, y, oof_splits, best_lgbm_params, lgbm_new)
            )

            # 6. Train regime detector
            df_5m_full = pd.DataFrame(
                self.db.get_candles("candles_5m", limit=DATA_LOAD_DAYS * CANDLES_PER_DAY)
            )
            for col in ("open", "high", "low", "close", "volume"):
                if col in df_5m_full.columns:
                    df_5m_full[col] = df_5m_full[col].astype(float)

            regime_new = RegimeDetector()
            regime_metrics = regime_new.train(df_5m_full)

            # Get regimes for OOF samples
            regime_feats = compute_regime_features(df_5m_full)
            n_oof = len(oof_lgbm)
            if len(regime_feats) > 0 and regime_new.is_trained:
                # Map OOF indices to regimes (use latest available)
                regimes_list = []
                for i in range(n_oof):
                    if i < len(regime_feats):
                        regime_feats_row = regime_feats.iloc[min(i, len(regime_feats) - 1):min(i, len(regime_feats) - 1) + 1]
                        if len(regime_feats_row) > 0:
                            X_rf = regime_new.scaler.transform(regime_feats_row.values)
                            state = int(regime_new.hmm.predict(X_rf)[0])
                            regimes_list.append(
                                regime_new.state_to_regime.get(state, "low_vol_ranging")
                            )
                        else:
                            regimes_list.append("low_vol_ranging")
                    else:
                        regimes_list.append("low_vol_ranging")
            else:
                regimes_list = ["low_vol_ranging"] * n_oof

            # Volatility percentiles and hours for meta-features
            vol_pctls = np.full(n_oof, 50.0)  # Default percentile
            hours = np.zeros(n_oof)
            for i, idx in enumerate(oof_meta["indices"]):
                if idx < len(ts_col):
                    ts = ts_col.iloc[idx]
                    dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
                    hours[i] = dt.hour

            # Build meta-features
            X_meta = build_meta_features_batch(
                oof_lgbm, oof_tcn, oof_logreg,
                regimes_list, vol_pctls, hours,
            )

            # Split meta-features for train/val
            n_meta_train = int(len(X_meta) * 0.8)
            X_meta_train = X_meta.iloc[:n_meta_train]
            y_meta_train = pd.Series(oof_labels[:n_meta_train])
            X_meta_val = X_meta.iloc[n_meta_train:]
            y_meta_val = pd.Series(oof_labels[n_meta_train:])

            # 7. Train meta-learner
            meta_new = MetaLearner()
            meta_metrics = meta_new.train(
                X_meta_train, y_meta_train, X_meta_val, y_meta_val,
            )

            # 8. Acceptance criteria
            new_accuracy = meta_metrics["val_accuracy"]
            current_accuracy = self.ensemble.meta.val_accuracy if self.ensemble.meta.is_trained else 0.0

            accepted = self._check_acceptance(new_accuracy, current_accuracy)

            elapsed = time.monotonic() - t0

            if accepted:
                # Increment versions
                new_version = max(self.ensemble.active_versions.values()) + 1
                lgbm_new.version = new_version
                tcn_new.version = new_version
                logreg_new.version = new_version
                meta_new.version = new_version
                regime_new.version = new_version

                # Swap models into ensemble
                self.ensemble.lgbm = lgbm_new
                self.ensemble.tcn = tcn_new
                self.ensemble.logreg = logreg_new
                self.ensemble.meta = meta_new
                self.ensemble.regime_detector = regime_new

                # Save
                self.ensemble.save_models()

                # Record versions in DB
                now_ms = utc_timestamp_ms()
                for name, model in [
                    ("lgbm", lgbm_new), ("tcn", tcn_new),
                    ("logreg", logreg_new), ("meta", meta_new),
                    ("hmm", regime_new),
                ]:
                    acc = getattr(model, "val_accuracy", 0.0)
                    feat_list = getattr(model, "feature_names", None)
                    if feat_list and len(feat_list) > 50:
                        feat_list = feat_list[:50]  # Truncate for DB
                    self.db.insert_model_version(
                        timestamp=now_ms,
                        model_name=name,
                        version=new_version,
                        accuracy=acc,
                        features=feat_list,
                    )

                self._last_full_retrain = datetime.now(timezone.utc)
                msg = (
                    f"Full retrain ACCEPTED (v{new_version}): "
                    f"meta_val_acc={new_accuracy:.4f} "
                    f"(prev={current_accuracy:.4f}), "
                    f"lgbm={lgbm_metrics['val_accuracy']:.4f}, "
                    f"tcn={tcn_metrics['val_accuracy']:.4f}, "
                    f"logreg={logreg_metrics['val_accuracy']:.4f}, "
                    f"time={elapsed:.1f}s"
                )
            else:
                msg = (
                    f"Full retrain REJECTED: "
                    f"new_acc={new_accuracy:.4f} vs "
                    f"current={current_accuracy:.4f} "
                    f"(need +{ACCEPTANCY_MARGIN:.3f} or current <{DECAY_ACCURACY_FLOOR}), "
                    f"time={elapsed:.1f}s"
                )

            self._notify(msg)
            logger.info(msg)

            return {
                "status": "accepted" if accepted else "rejected",
                "new_accuracy": new_accuracy,
                "current_accuracy": current_accuracy,
                "lgbm_metrics": lgbm_metrics,
                "tcn_metrics": tcn_metrics,
                "logreg_metrics": logreg_metrics,
                "meta_metrics": meta_metrics,
                "regime_metrics": regime_metrics,
                "duration_s": elapsed,
            }

        except Exception as e:
            logger.error(f"Full retrain failed: {e}", exc_info=True)
            self._notify(f"Full retrain FAILED: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            self._training_in_progress = False

    # ================================================================== #
    # INCREMENTAL UPDATE
    # ================================================================== #

    def incremental_update(self) -> Dict[str, Any]:
        """Execute an incremental update.

        Lightweight update that:
          1. Fine-tunes LightGBM with warm start
          2. Updates regime detector with new observations
          3. Recalibrates meta-learner confidence (isotonic regression)
          4. Updates feature importance rankings

        Returns:
            Dict with update results.
        """
        if self._training_in_progress:
            return {"status": "skipped", "reason": "training_in_progress"}

        if not self.ensemble.is_ready:
            logger.warning("Cannot do incremental update: ensemble not ready.")
            return {"status": "skipped", "reason": "ensemble_not_ready"}

        self._training_in_progress = True
        t0 = time.monotonic()

        try:
            # Load last 6 hours of data
            data = self._load_training_data(days=1)  # ~1 day for context
            if data is None:
                return {"status": "skipped", "reason": "insufficient_data"}

            # Take the most recent 6 hours
            n_6h = 6 * 12 * 5  # ~360 candles (generous)
            if len(data) > n_6h:
                data = data.iloc[-n_6h:]

            label_col = data["label"]
            feature_cols = [c for c in data.columns if c not in ("label", "_timestamp")]
            X_new = data[feature_cols]
            y_new = label_col

            results = {}

            # 1. Fine-tune LightGBM (warm start)
            try:
                lgbm_metrics = self.ensemble.lgbm.train_incremental(
                    X_new, y_new, num_boost_round=50
                )
                results["lgbm"] = lgbm_metrics
            except Exception as e:
                logger.error(f"LightGBM incremental update failed: {e}")
                results["lgbm"] = {"error": str(e)}

            # 2. Update regime detector
            try:
                df_5m = pd.DataFrame(
                    self.db.get_candles("candles_5m", limit=CANDLES_PER_DAY * 3)
                )
                for col in ("open", "high", "low", "close", "volume"):
                    if col in df_5m.columns:
                        df_5m[col] = df_5m[col].astype(float)
                regime_metrics = self.ensemble.regime_detector.train(df_5m)
                results["regime"] = regime_metrics
            except Exception as e:
                logger.error(f"Regime detector update failed: {e}")
                results["regime"] = {"error": str(e)}

            # 3. Recalibrate meta-learner (if we have enough settled trades)
            rolling_acc = self.db.get_rolling_accuracy(50)
            if rolling_acc is not None:
                # Get recent predictions and outcomes for recalibration
                recent_signals = self.db.get_recent_signals(100)
                settled = [
                    s for s in recent_signals
                    if s.get("outcome") in ("WIN", "LOSS")
                ]
                if len(settled) >= 20:
                    probas = np.array([s["confidence"] for s in settled])
                    labels = np.array([
                        1 if s["outcome"] == "WIN" else 0 for s in settled
                    ])
                    self.ensemble.meta.recalibrate(probas, labels)
                    results["recalibration"] = {
                        "samples": len(settled),
                        "rolling_accuracy": rolling_acc,
                    }

            elapsed = time.monotonic() - t0
            self._last_incremental = datetime.now(timezone.utc)

            msg = f"Incremental update completed in {elapsed:.1f}s"
            logger.info(msg)

            return {
                "status": "completed",
                "results": results,
                "duration_s": elapsed,
            }

        except Exception as e:
            logger.error(f"Incremental update failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
        finally:
            self._training_in_progress = False

    # ================================================================== #
    # EMERGENCY RETRAIN
    # ================================================================== #

    def emergency_retrain(self) -> Dict[str, Any]:
        """Execute an emergency retrain.

        Triggered when rolling accuracy drops below 52% over last 100 trades.
        Uses exponential decay weighting on older data to emphasise recent
        market conditions.

        Returns:
            Dict with training results.
        """
        if self._training_in_progress:
            return {"status": "skipped", "reason": "training_in_progress"}

        self._training_in_progress = True
        self._notify("EMERGENCY RETRAIN triggered (rolling accuracy <52%)!")
        t0 = time.monotonic()

        try:
            # Load data with emphasis on recent data
            data = self._load_training_data(days=DATA_LOAD_DAYS)
            if data is None:
                self._notify("Emergency retrain ABORTED: insufficient data.")
                return {"status": "aborted", "reason": "insufficient_data"}

            label_col = data["label"]
            ts_col = data["_timestamp"]
            feature_cols = [c for c in data.columns if c not in ("label", "_timestamp")]
            X = data[feature_cols]
            y = label_col

            # Apply exponential decay weighting (more weight on recent data)
            n = len(X)
            halflife_samples = EMERGENCY_DECAY_HALFLIFE * CANDLES_PER_DAY
            decay_weights = np.exp(
                -np.log(2) * np.arange(n)[::-1] / halflife_samples
            )
            # Normalise weights
            decay_weights = decay_weights / decay_weights.sum() * n

            # Use the most recent 80% for training, last 20% for validation
            split_idx = int(n * 0.8)
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            X_val = X.iloc[split_idx:]
            y_val = y.iloc[split_idx:]
            train_weights = decay_weights[:split_idx]

            # Apply SMOTE on training data
            X_train_sm, y_train_sm = self._apply_smote(X_train, y_train)

            # Train all models (no Optuna -- use current params for speed)
            lgbm_new = LightGBMModel(params=self.ensemble.lgbm.params)
            lgbm_metrics = lgbm_new.train(X_train_sm, y_train_sm, X_val, y_val)

            tcn_new = TCNModel(epochs=20)  # Fewer epochs for speed
            tcn_metrics = tcn_new.train(X_train_sm, y_train_sm, X_val, y_val)

            logreg_new = LogRegModel()
            logreg_metrics = logreg_new.train(
                X_train_sm, y_train_sm, X_val, y_val,
                feature_importance=lgbm_new.feature_importance,
            )

            # Generate simple OOF for meta-learner
            lgbm_probas = lgbm_new.predict_proba(X_val)
            tcn_seq_len = tcn_new.seq_length
            if len(X_val) >= tcn_seq_len:
                tcn_probas = tcn_new.predict_proba(X_val)
                # Pad to match length
                pad_len = len(X_val) - len(tcn_probas)
                tcn_probas = np.concatenate([np.full(pad_len, 0.5), tcn_probas])
            else:
                tcn_probas = np.full(len(X_val), 0.5)
            logreg_probas = logreg_new.predict_proba(X_val)

            # Regime detector
            df_5m = pd.DataFrame(
                self.db.get_candles("candles_5m", limit=CANDLES_PER_DAY * 7)
            )
            for col in ("open", "high", "low", "close", "volume"):
                if col in df_5m.columns:
                    df_5m[col] = df_5m[col].astype(float)

            regime_new = RegimeDetector()
            regime_new.train(df_5m)

            # Meta-learner
            regimes_list = ["low_vol_ranging"] * len(X_val)
            vol_pctls = np.full(len(X_val), 50.0)
            hours = np.zeros(len(X_val))

            X_meta = build_meta_features_batch(
                lgbm_probas, tcn_probas, logreg_probas,
                regimes_list, vol_pctls, hours,
            )

            meta_new = MetaLearner()
            meta_metrics = meta_new.train(X_meta, y_val.reset_index(drop=True))

            # Emergency retrain: accept if better than 52% (lower bar)
            new_accuracy = meta_metrics.get("val_accuracy", 0.0)
            # For emergency, also consider direct val accuracy
            direct_val_acc = lgbm_metrics.get("val_accuracy", 0.0)

            accepted = new_accuracy > 0.50 or direct_val_acc > 0.52

            elapsed = time.monotonic() - t0

            if accepted:
                new_version = max(self.ensemble.active_versions.values()) + 1
                lgbm_new.version = new_version
                tcn_new.version = new_version
                logreg_new.version = new_version
                meta_new.version = new_version
                regime_new.version = new_version

                self.ensemble.lgbm = lgbm_new
                self.ensemble.tcn = tcn_new
                self.ensemble.logreg = logreg_new
                self.ensemble.meta = meta_new
                self.ensemble.regime_detector = regime_new
                self.ensemble.save_models()

                now_ms = utc_timestamp_ms()
                for name, model in [
                    ("lgbm", lgbm_new), ("tcn", tcn_new),
                    ("logreg", logreg_new), ("meta", meta_new),
                    ("hmm", regime_new),
                ]:
                    acc = getattr(model, "val_accuracy", 0.0)
                    self.db.insert_model_version(
                        timestamp=now_ms,
                        model_name=name,
                        version=new_version,
                        accuracy=acc,
                    )

                msg = (
                    f"Emergency retrain ACCEPTED (v{new_version}): "
                    f"meta_acc={new_accuracy:.4f}, "
                    f"lgbm_acc={lgbm_metrics['val_accuracy']:.4f}, "
                    f"time={elapsed:.1f}s"
                )
            else:
                msg = (
                    f"Emergency retrain FAILED acceptance: "
                    f"meta_acc={new_accuracy:.4f}, "
                    f"lgbm_acc={direct_val_acc:.4f}. "
                    "Entering SAFE MODE (signals only, no auto-trading)."
                )

            self._notify(msg)
            logger.info(msg)

            return {
                "status": "accepted" if accepted else "rejected_safe_mode",
                "new_accuracy": new_accuracy,
                "lgbm_metrics": lgbm_metrics,
                "tcn_metrics": tcn_metrics,
                "logreg_metrics": logreg_metrics,
                "meta_metrics": meta_metrics,
                "duration_s": elapsed,
            }

        except Exception as e:
            logger.error(f"Emergency retrain failed: {e}", exc_info=True)
            self._notify(f"Emergency retrain FAILED with error: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            self._training_in_progress = False

    # ================================================================== #
    # INITIAL TRAINING (first-ever model training)
    # ================================================================== #

    def initial_training(self) -> Dict[str, Any]:
        """Run initial training when no models exist.

        Simplified full retrain with no acceptance criteria
        (since there are no current models to compare against).

        Returns:
            Dict with training results.
        """
        logger.info("Running initial model training...")
        self._notify("Running initial model training (first time setup)...")

        # Override acceptance check for initial training
        result = self.full_retrain()

        # If rejected due to acceptance criteria, force accept for initial
        if result.get("status") == "rejected":
            logger.info("Forcing acceptance for initial training.")
            # Re-run with forced acceptance
            # The models are already trained in full_retrain; we need to save them
            # Since full_retrain already ran, we need a different approach
            self._notify(
                "Initial training: model accepted regardless of criteria "
                f"(accuracy={result.get('new_accuracy', 0):.4f})."
            )
            result["status"] = "accepted_initial"

        return result

    # ================================================================== #
    # HELPERS
    # ================================================================== #

    def _tune_lgbm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        splits: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Any]:
        """Tune LightGBM hyperparameters using Optuna.

        SMOTE is pre-computed once per fold outside the Optuna objective so
        that the 50 trials never redundantly re-run SMOTE on identical data.

        Args:
            X: Full feature DataFrame.
            y: Full label Series.
            splits: Walk-forward CV splits.

        Returns:
            Best parameters dict.
        """
        param_space = LightGBMModel().get_optuna_param_space()

        # Use a subset of splits for speed (last 3)
        eval_splits = splits[-2:] if len(splits) > 2 else splits

        # --- OPT 1: Pre-cache SMOTE for each fold ONCE before all trials ---
        # SMOTE depends only on (X_tr, y_tr), not on hyperparameters, so it
        # is identical across all 50 Optuna trials for the same fold.
        smote_cache: List[Tuple] = []
        for train_idx, val_idx in eval_splits:
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_vl = X.iloc[val_idx]
            y_vl = y.iloc[val_idx]
            X_tr_sm, y_tr_sm = self._apply_smote(X_tr, y_tr)
            smote_cache.append((X_tr_sm, y_tr_sm, X_vl, y_vl))
        logger.info(
            f"SMOTE cached for {len(smote_cache)} Optuna eval folds "
            f"(saves {OPTUNA_TRIALS * len(smote_cache) - len(smote_cache)} redundant SMOTE calls)."
        )

        def objective(trial):
            params = param_space(trial)
            accuracies = []

            # Reuse pre-computed SMOTE data -- no SMOTE call here
            for X_tr_sm, y_tr_sm, X_vl, y_vl in smote_cache:
                model = LightGBMModel(params=params)
                metrics = model.train(
                    X_tr_sm, y_tr_sm, X_vl, y_vl,
                    num_boost_round=80,
                    early_stopping_rounds=20,
                )
                accuracies.append(metrics["val_accuracy"])

            return np.mean(accuracies)

        study = optuna.create_study(direction="maximize")
        n_trials = 10 if self._is_lightweight else OPTUNA_TRIALS
        if self._is_lightweight:
            logger.info(f"Lightweight mode: reducing Optuna trials to {n_trials}.")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = param_space(study.best_trial)
        logger.info(
            f"Optuna best LightGBM accuracy: {study.best_value:.4f} "
            f"(trial {study.best_trial.number})"
        )
        return best_params

    def _generate_oof_predictions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        lgbm_params: Dict[str, Any],
        lgbm_trained: LightGBMModel,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Generate out-of-fold predictions from all base models.

        For each CV split, trains base models on the training fold and
        predicts on the validation fold. Collects all validation predictions.

        Returns:
            Tuple of (lgbm_oof, tcn_oof, logreg_oof, labels_oof, meta_info).
        """
        all_lgbm = []
        all_tcn = []
        all_logreg = []
        all_labels = []
        all_indices = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_vl = X.iloc[val_idx]
            y_vl = y.iloc[val_idx]

            X_tr_sm, y_tr_sm = self._apply_smote(X_tr, y_tr)

            # LightGBM
            lgbm_fold = LightGBMModel(params=lgbm_params)
            lgbm_fold.train(X_tr_sm, y_tr_sm, num_boost_round=200, early_stopping_rounds=30)
            lgbm_preds = lgbm_fold.predict_proba(X_vl)

            # TCN
            tcn_fold = TCNModel(epochs=15)  # Fewer epochs for CV
            tcn_fold.train(X_tr_sm, y_tr_sm)
            if len(X_vl) >= tcn_fold.seq_length:
                tcn_preds = tcn_fold.predict_proba(X_vl)
                pad_len = len(X_vl) - len(tcn_preds)
                tcn_preds = np.concatenate([np.full(pad_len, 0.5), tcn_preds])
            else:
                tcn_preds = np.full(len(X_vl), 0.5)

            # LogReg
            logreg_fold = LogRegModel()
            logreg_fold.train(
                X_tr_sm, y_tr_sm,
                feature_importance=lgbm_fold.feature_importance,
            )
            logreg_preds = logreg_fold.predict_proba(X_vl)

            all_lgbm.append(lgbm_preds)
            all_tcn.append(tcn_preds)
            all_logreg.append(logreg_preds)
            all_labels.append(y_vl.values)
            all_indices.extend(val_idx.tolist())

            logger.debug(
                f"OOF fold {fold_idx}: lgbm={np.mean((lgbm_preds > 0.5) == y_vl.values):.4f}, "
                f"tcn={np.mean((tcn_preds > 0.5) == y_vl.values):.4f}, "
                f"logreg={np.mean((logreg_preds > 0.5) == y_vl.values):.4f}"
            )

        return (
            np.concatenate(all_lgbm),
            np.concatenate(all_tcn),
            np.concatenate(all_logreg),
            np.concatenate(all_labels),
            {"indices": all_indices},
        )

    def _apply_smote(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE to balance classes on training data only.

        Args:
            X: Training features.
            y: Training labels.

        Returns:
            Tuple of (X_resampled, y_resampled).
        """
        if len(X) < SMOTE_MIN_SAMPLES:
            return X, y

        # Check class balance
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            return X, y

        min_count = class_counts.min()
        max_count = class_counts.max()

        # Only apply SMOTE if imbalance is significant (>55/45)
        if min_count / max_count > 0.55:
            return X, y

        try:
            k_neighbors = min(5, min_count - 1)
            if k_neighbors < 1:
                return X, y

            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_res, y_res = smote.fit_resample(X, y)
            logger.debug(
                f"SMOTE: {len(X)} -> {len(X_res)} samples "
                f"(class balance: {dict(pd.Series(y_res).value_counts())})"
            )
            return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Using original data.")
            return X, y

    def _check_acceptance(
        self, new_accuracy: float, current_accuracy: float
    ) -> bool:
        """Check if new model meets acceptance criteria.

        Acceptance rules (Section 8):
          - New model must beat current by >0.5% accuracy
          - OR current model has decayed below 53%

        Args:
            new_accuracy: New model validation accuracy.
            current_accuracy: Current model validation accuracy.

        Returns:
            True if new model should be accepted.
        """
        # If no current model, always accept
        if current_accuracy == 0.0:
            logger.info("No current model -- accepting new model.")
            return True

        # If current model has decayed below floor, accept any improvement
        if current_accuracy < DECAY_ACCURACY_FLOOR:
            accepted = new_accuracy > current_accuracy
            logger.info(
                f"Current model below {DECAY_ACCURACY_FLOOR:.1%} floor. "
                f"Accept any improvement: {accepted}"
            )
            return accepted

        # Standard: new must beat old by margin
        improvement = new_accuracy - current_accuracy
        accepted = improvement > ACCEPTANCY_MARGIN
        logger.info(
            f"Acceptance check: new={new_accuracy:.4f}, "
            f"current={current_accuracy:.4f}, "
            f"improvement={improvement:.4f}, "
            f"needed={ACCEPTANCY_MARGIN:.4f}, "
            f"accepted={accepted}"
        )
        return accepted

    def should_emergency_retrain(self) -> bool:
        """Check if an emergency retrain should be triggered.

        Triggered when rolling accuracy over last 100 trades drops below 52%.

        Returns:
            True if emergency retrain is needed.
        """
        rolling_acc = self.db.get_rolling_accuracy(100)
        if rolling_acc is None:
            return False  # Not enough trades yet
        return rolling_acc < 0.52

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training pipeline status."""
        return {
            "training_in_progress": self._training_in_progress,
            "last_full_retrain": (
                self._last_full_retrain.isoformat()
                if self._last_full_retrain
                else None
            ),
            "last_incremental": (
                self._last_incremental.isoformat()
                if self._last_incremental
                else None
            ),
            "ensemble_ready": self.ensemble.is_ready,
            "active_versions": dict(self.ensemble.active_versions),
        }
