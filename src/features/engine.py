"""Feature engineering orchestrator for CleoBot.

Orchestrates all feature modules into a single unified pipeline:
  candle + orderbook + funding + cross_tf + time + polymarket + derived

Outputs a validated feature dict / DataFrame with:
  - 80-120 features total
  - No NaN values (all filled with safe defaults)
  - Correct dtypes (float64)
  - Feature history maintained for z-score / rolling computations
  - Runs in <33 seconds (target: <5 seconds in practice)

Designed to be called once per 5-minute cycle by the trading executor.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque

from src.database import Database
from src.features.candle_features import compute_candle_features
from src.features.orderbook_features import compute_orderbook_features
from src.features.funding_features import compute_funding_features
from src.features.cross_tf_features import compute_cross_tf_features
from src.features.time_features import compute_time_features
from src.features.polymarket_features import compute_polymarket_features
from src.features.derived_features import compute_derived_features, TOP_FEATURES_FOR_ZSCORE
from src.utils.logger import get_logger

logger = get_logger("features.engine")

# Minimum candles required to produce a valid feature set
MIN_CANDLES_5M = 50
MIN_CANDLES_15M = 22
MIN_CANDLES_1H = 15

# How many candles to load for feature computation
CANDLE_LOOKBACK_5M = 150    # ~12.5 hours
CANDLE_LOOKBACK_15M = 100   # ~25 hours
CANDLE_LOOKBACK_1H = 72     # 3 days

# Orderbook snapshots window (~3 minutes worth at 5s interval = 36 + buffer)
ORDERBOOK_LOOKBACK = 800    # ~1h 6min at 5s each

# Funding rate lookback (21 periods = 7 days)
FUNDING_LOOKBACK = 25

# Feature history depth for z-scores
HISTORY_DEPTH = 200


class FeatureEngine:
    """Orchestrates all feature calculations for CleoBot.
    
    Maintains in-memory feature history for rolling z-scores and derived
    features that require temporal context beyond what's in the DB.
    """

    def __init__(self, db: Database):
        """Initialise the feature engine.

        Args:
            db: Database instance for loading candle / orderbook / funding data.
        """
        self.db = db

        # Rolling feature history for z-score / temporal derived features
        # key: feature_name, value: deque of recent values (newest last)
        self._feature_history: Dict[str, deque] = {
            name: deque(maxlen=HISTORY_DEPTH)
            for name in TOP_FEATURES_FOR_ZSCORE
            + ["vol_std_12", "atr_12", "ob_imbalance_5"]
        }

        # Polymarket snapshots (in-memory, updated externally)
        self._pm_snapshots: deque = deque(maxlen=100)
        self._latest_pm_data: Optional[Dict[str, Any]] = None
        self._latest_pm_model_pred: Optional[float] = None

        # Timing stats
        self._last_compute_time_s: float = 0.0
        self._compute_count: int = 0

    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #

    def compute(self,
                current_ts_ms: Optional[int] = None,
                current_orderbook: Optional[Dict[str, Any]] = None,
                ) -> Dict[str, float]:
        """Compute the full feature set for the current moment.

        Args:
            current_ts_ms: Current UTC timestamp in milliseconds.
                           Defaults to now.
            current_orderbook: Latest orderbook snapshot from DataCollector
                               (in-memory, most up-to-date).

        Returns:
            Dict mapping feature_name -> float. Always 80-120 features.
            Never contains NaN values.

        Raises:
            RuntimeError: If insufficient candle data is available (startup).
        """
        t0 = time.monotonic()

        if current_ts_ms is None:
            import datetime
            current_ts_ms = int(
                datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000
            )

        # ---- 1. Load data from database ---------------------------------- #
        df_5m = self._load_candles("candles_5m", CANDLE_LOOKBACK_5M)
        df_15m = self._load_candles("candles_15m", CANDLE_LOOKBACK_15M)
        df_1h = self._load_candles("candles_1h", CANDLE_LOOKBACK_1H)

        if df_5m is None or len(df_5m) < MIN_CANDLES_5M:
            n = len(df_5m) if df_5m is not None else 0
            raise RuntimeError(
                f"Insufficient 5m candle data: {n} rows (need {MIN_CANDLES_5M}). "
                "Waiting for backfill to complete."
            )

        # ---- 2. Load orderbook snapshots --------------------------------- #
        since_ts = current_ts_ms - (ORDERBOOK_LOOKBACK * 5 * 1000)  # ~lookback seconds
        ob_snapshots = self.db.get_orderbook_snapshots(
            since=since_ts, limit=ORDERBOOK_LOOKBACK
        )

        # Use live in-memory orderbook as the "current" snapshot if provided
        if current_orderbook is not None:
            snap = current_orderbook
        elif ob_snapshots:
            snap = ob_snapshots[-1]
        else:
            snap = None

        # ---- 3. Load funding rates --------------------------------------- #
        funding_records = self.db.get_funding_rates(limit=FUNDING_LOOKBACK)

        # ---- 4. Compute each feature group ------------------------------- #
        # 4a. Candle features
        candle_feats = compute_candle_features(df_5m)
        # candle_feats is dict of name -> pd.Series; take the last value
        candle_scalar = {k: float(v.iloc[-1]) for k, v in candle_feats.items()}

        # 4b. Orderbook features
        if snap is not None:
            ob_feats = compute_orderbook_features(
                snapshots=ob_snapshots,
                current_snapshot=snap,
            )
        else:
            ob_feats = compute_orderbook_features(snapshots=[], current_snapshot=None)

        # 4c. Funding rate features
        funding_feats = compute_funding_features(
            funding_records=funding_records,
            current_ts_ms=current_ts_ms,
        )

        # 4d. Cross-timeframe features
        cross_feats = compute_cross_tf_features(
            df_5m=df_5m,
            df_15m=df_15m,
            df_1h=df_1h,
        )

        # 4e. Time-based features
        time_feats = compute_time_features(
            current_ts_ms=current_ts_ms,
            df_5m=df_5m,
        )

        # 4f. Polymarket features
        pm_feats = compute_polymarket_features(
            market_data=self._latest_pm_data,
            historical_snapshots=list(self._pm_snapshots),
            model_prediction=self._latest_pm_model_pred,
        )

        # ---- 5. Merge all features into one dict ------------------------- #
        all_feats: Dict[str, float] = {}
        all_feats.update(candle_scalar)
        all_feats.update(ob_feats)
        all_feats.update(funding_feats)
        all_feats.update(cross_feats)
        all_feats.update(time_feats)
        all_feats.update(pm_feats)

        # ---- 6. Derived features (needs combined dict + history) --------- #
        history_snapshot = {k: list(v) for k, v in self._feature_history.items()}
        derived_feats = compute_derived_features(
            features=all_feats,
            feature_history=history_snapshot,
        )
        all_feats.update(derived_feats)

        # ---- 7. Validate: no NaN, correct dtype -------------------------- #
        all_feats = self._validate_features(all_feats)

        # ---- 8. Update rolling history ----------------------------------- #
        self._update_history(all_feats)

        # ---- 9. Timing --------------------------------------------------- #
        elapsed = time.monotonic() - t0
        self._last_compute_time_s = elapsed
        self._compute_count += 1

        if elapsed > 33:
            logger.warning(f"Feature computation took {elapsed:.2f}s (limit 33s)!")
        else:
            logger.debug(
                f"Features computed: {len(all_feats)} features in {elapsed:.3f}s "
                f"(run #{self._compute_count})"
            )

        return all_feats

    def compute_as_dataframe(self,
                             current_ts_ms: Optional[int] = None,
                             current_orderbook: Optional[Dict[str, Any]] = None,
                             ) -> pd.DataFrame:
        """Compute features and return as a single-row DataFrame.

        Useful for direct model inference (sklearn / LightGBM expect DataFrames).

        Returns:
            pd.DataFrame with shape (1, n_features).
        """
        feats = self.compute(current_ts_ms=current_ts_ms,
                             current_orderbook=current_orderbook)
        return pd.DataFrame([feats])

    def update_polymarket_data(
        self,
        market_data: Dict[str, Any],
        model_prediction: Optional[float] = None,
    ):
        """Update Polymarket market data for next feature computation.

        Args:
            market_data: Current Polymarket market state dict.
            model_prediction: Optional model UP probability for divergence feature.
        """
        self._latest_pm_data = market_data
        self._latest_pm_model_pred = model_prediction
        self._pm_snapshots.append(market_data)

    def get_feature_names(self) -> List[str]:
        """Get the list of all feature names produced by the engine.

        Returns:
            Sorted list of feature name strings.
        """
        # Compute once on minimal synthetic data to get the names
        try:
            feats = self.compute()
            return sorted(feats.keys())
        except RuntimeError:
            # Not enough data yet -- return an estimated list
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        return {
            "last_compute_time_s": self._last_compute_time_s,
            "compute_count": self._compute_count,
            "history_depth": {k: len(v) for k, v in self._feature_history.items()},
        }

    # ------------------------------------------------------------------ #
    # PRIVATE HELPERS
    # ------------------------------------------------------------------ #

    def _load_candles(self, table: str, limit: int) -> Optional[pd.DataFrame]:
        """Load candles from DB into a DataFrame with proper dtypes."""
        try:
            rows = self.db.get_candles(table, limit=limit)
            if not rows:
                return None
            df = pd.DataFrame(rows)
            # Ensure correct columns and types
            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    df[col] = df[col].astype(float)
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"Failed to load candles from {table}: {e}")
            return None

    def _validate_features(self, feats: Dict[str, float]) -> Dict[str, float]:
        """Ensure all features are finite float64 values.

        - Replaces NaN / inf with 0.0
        - Converts all values to Python float (not numpy scalars)
        - Logs a warning if any NaN was found
        """
        cleaned: Dict[str, float] = {}
        nan_keys = []
        for k, v in feats.items():
            try:
                fv = float(v)
                if not np.isfinite(fv):
                    fv = 0.0
                    nan_keys.append(k)
            except (TypeError, ValueError):
                fv = 0.0
                nan_keys.append(k)
            cleaned[k] = fv
        if nan_keys:
            logger.warning(
                f"NaN/inf values replaced with 0.0 for features: {nan_keys}"
            )
        return cleaned

    def _update_history(self, feats: Dict[str, float]):
        """Append current feature values to rolling history deques."""
        for name, dq in self._feature_history.items():
            if name in feats:
                dq.append(feats[name])


# ------------------------------------------------------------------ #
# CONVENIENCE FUNCTION (used by tests and scripts)
# ------------------------------------------------------------------ #

def build_feature_engine(db: Database) -> FeatureEngine:
    """Create and return a FeatureEngine instance.

    Args:
        db: Initialised Database instance.

    Returns:
        FeatureEngine ready for use.
    """
    engine = FeatureEngine(db)
    logger.info("FeatureEngine initialised.")
    return engine
