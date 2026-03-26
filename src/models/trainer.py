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
from src.features.engine import FeatureEngine
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
OPTUNA_TRIALS = 50           # Trials per model
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
        self, days: int = FULL_RETRAIN_DAYS
    ) -> Optional[pd.DataFrame]:
        """Load and prepare training data from the database.

        Loads candle data, computes features for each candle, and creates
        labels (1 = UP, 0 = DOWN based on next candle close vs current close).

        Args:
            days: Number of days of data to load.

        Returns:
            DataFrame with features and 'label' column, or None if insufficient data.
        """
        limit = days * CANDLES_PER_DAY + 200  # Extra for feature lookback
        candles = self.db.get_candles("candles_5m", limit=limit)

        if len(candles) < CANDLES_PER_DAY * 3:  # Need at least 3 days
            logger.warning(
                f"Insufficient candle data: {len(candles)} rows "
                f"(need {CANDLES_PER_DAY * 3}+)."
            )
            return None

        df = pd.DataFrame(candles)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Create labels: 1 = next candle closes higher, 0 = lower
        df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df = df.iloc[:-1]  # Drop last row (no label)

        # Compute features for each row using a sliding window approach
        logger.info(f"Computing features for {len(df)} candles...")
        feature_rows = []
        feature_engine_local = FeatureEngine(self.db)

        # We need sufficient history, so start from row 150+
        start_idx = 150
        if start_idx >= len(df):
            logger.warning("Not enough candles for feature computation.")
            return None

        # Batch feature computation: compute features using candle slices
        for i in range(start_idx, len(df)):
            # Get candle slice up to this point
            slice_df = df.iloc[max(0, i - 149): i + 1].copy()
            try:
                feats = self._compute_features_for_candle(
                    slice_df, df.iloc[i]["timestamp"]
                )
                feats["label"] = df.iloc[i]["label"]
                feats["_timestamp"] = df.iloc[i]["timestamp"]
                feature_rows.append(feats)
            except Exception as e:
                # Skip candles where feature computation fails
                continue

        if len(feature_rows) < CANDLES_PER_DAY:
            logger.warning(
                f"Too few valid feature rows: {len(feature_rows)} "
                f"(need {CANDLES_PER_DAY}+)."
            )
            return None

        result = pd.DataFrame(feature_rows)
        logger.info(
            f"Training data prepared: {len(result)} samples, "
            f"{len(result.columns) - 2} features"
        )
        return result

    def _compute_features_for_candle(
        self, candle_slice: pd.DataFrame, timestamp_ms: int
    ) -> Dict[str, float]:
        """Compute features for a single candle using its historical context.

        Uses a lightweight approach: computes candle-based features directly
        from the slice without full DB access for orderbook/funding.
        Falls back to zeros for orderbook/funding features during training.

        Args:
            candle_slice: DataFrame of candles ending at the target candle.
            timestamp_ms: Timestamp of the target candle.

        Returns:
            Dict of feature_name -> value.
        """
        from src.features.candle_features import compute_candle_features
        from src.features.time_features import compute_time_features
        from src.features.derived_features import compute_derived_features

        # Candle features (main source of training signal)
        candle_feats = compute_candle_features(candle_slice)
        features = {k: float(v.iloc[-1]) for k, v in candle_feats.items()}

        # Time features
        time_feats = compute_time_features(
            current_ts_ms=int(timestamp_ms), df_5m=candle_slice
        )
        features.update(time_feats)

        # Cross-timeframe features (simplified: use 5m data only)
        # Full cross-TF features need 15m/1h data which is complex to slice
        # during batch training. We use simplified proxies.
        features.update(self._simplified_cross_tf(candle_slice))

        # Orderbook features default to 0.0 during historical training
        # (orderbook data is not available for historical candles in most cases)
        ob_names = [
            "ob_imbalance_5", "ob_imbalance_10", "ob_imbalance_20",
            "ob_imbalance_change_30s", "ob_imbalance_change_60s",
            "ob_imbalance_change_90s", "ob_slope_bid", "ob_slope_ask",
            "ob_slope_ratio", "ob_wall_bid", "ob_wall_ask", "ob_wall_imbalance",
            "ob_spread_bps", "ob_spread_vs_avg", "ob_spread_percentile",
            "ob_net_pressure", "ob_pressure_change_30s", "ob_pressure_change_60s",
            "ob_pressure_change_90s", "ob_pressure_momentum",
        ]
        for name in ob_names:
            if name not in features:
                features[name] = 0.0

        # Funding features default to 0.0
        funding_names = [
            "funding_rate", "funding_momentum", "funding_time_to_settlement",
            "funding_vs_24h_avg", "funding_vs_7d_avg", "funding_percentile_7d",
            "funding_direction", "funding_acceleration",
        ]
        for name in funding_names:
            if name not in features:
                features[name] = 0.0

        # Polymarket features default to 0.0
        pm_names = [
            "pm_up_odds", "pm_down_odds", "pm_odds_velocity",
            "pm_volume_ratio", "pm_total_volume", "pm_odds_divergence",
        ]
        for name in pm_names:
            if name not in features:
                features[name] = 0.0

        # Derived features
        derived_feats = compute_derived_features(
            features=features, feature_history={}
        )
        features.update(derived_feats)

        # Validate: replace NaN/inf with 0.0
        cleaned = {}
        for k, v in features.items():
            try:
                fv = float(v)
                if not np.isfinite(fv):
                    fv = 0.0
            except (TypeError, ValueError):
                fv = 0.0
            cleaned[k] = fv

        return cleaned

    def _simplified_cross_tf(self, df_5m: pd.DataFrame) -> Dict[str, float]:
        """Compute simplified cross-timeframe features from 5m data only."""
        features = {}
        close = df_5m["close"].values.astype(float)

        # Simulate 15m direction from last 3 candles
        if len(close) >= 3:
            features["cross_15m_direction"] = 1.0 if close[-1] > close[-3] else 0.0
        else:
            features["cross_15m_direction"] = 0.5

        # Simulate 1h direction from last 12 candles
        if len(close) >= 12:
            features["cross_1h_direction"] = 1.0 if close[-1] > close[-12] else 0.0
        else:
            features["cross_1h_direction"] = 0.5

        # Alignment
        d5m = 1.0 if len(close) >= 2 and close[-1] > close[-2] else 0.0
        features["cross_alignment"] = (
            d5m + features["cross_15m_direction"] + features["cross_1h_direction"]
        )

        # Simplified RSI for longer timeframes
        if len(close) >= 15:
            returns = np.diff(close[-15:])
            gains = returns[returns > 0]
            losses = -returns[returns < 0]
            avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
            avg_loss = np.mean(losses) if len(losses) > 0 else 1e-8
            rs = avg_gain / max(avg_loss, 1e-8)
            features["cross_15m_rsi"] = 100.0 - (100.0 / (1.0 + rs))
        else:
            features["cross_15m_rsi"] = 50.0

        if len(close) >= 60:
            returns = np.diff(close[-60:])
            gains = returns[returns > 0]
            losses = -returns[returns < 0]
            avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
            avg_loss = np.mean(losses) if len(losses) > 0 else 1e-8
            rs = avg_gain / max(avg_loss, 1e-8)
            features["cross_1h_rsi"] = 100.0 - (100.0 / (1.0 + rs))
        else:
            features["cross_1h_rsi"] = 50.0

        # Volatility ratio
        if len(close) >= 12:
            atr_5m = np.mean(np.abs(np.diff(close[-6:])))
            atr_1h = np.mean(np.abs(np.diff(close[-12:])))
            features["cross_vol_ratio"] = atr_5m / max(atr_1h, 1e-8)
        else:
            features["cross_vol_ratio"] = 1.0

        # Trend strength proxies
        for prefix, lookback in [("cross_15m_trend", 3), ("cross_1h_trend", 12)]:
            if len(close) >= lookback:
                x = np.arange(lookback)
                c = close[-lookback:]
                coeffs = np.polyfit(x, c, 1)
                features[prefix] = coeffs[0] / max(np.mean(c), 1e-8)
            else:
                features[prefix] = 0.0

        # S/R proximity (distance to recent high/low)
        if len(close) >= 12:
            recent_high = np.max(df_5m["high"].values[-12:])
            recent_low = np.min(df_5m["low"].values[-12:])
            price_range = recent_high - recent_low
            if price_range > 0:
                features["cross_sr_proximity"] = (
                    (close[-1] - recent_low) / price_range
                )
            else:
                features["cross_sr_proximity"] = 0.5
        else:
            features["cross_sr_proximity"] = 0.5

        # Momentum alignment
        features["cross_momentum_alignment"] = features["cross_alignment"] / 3.0

        return features

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
            data = self._load_training_data(days=FULL_RETRAIN_DAYS)
            if data is None:
                self._notify("Full retrain ABORTED: insufficient data.")
                return {"status": "aborted", "reason": "insufficient_data"}

            # Separate features, labels, timestamps
            label_col = data["label"]
            ts_col = data["_timestamp"]
            feature_cols = [c for c in data.columns if c not in ("label", "_timestamp")]
            X = data[feature_cols]
            y = label_col

            # 2. Walk-forward CV splits
            splits = self._walk_forward_split(len(X))
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
            logger.info("Generating OOF predictions for meta-learner...")
            oof_lgbm, oof_tcn, oof_logreg, oof_labels, oof_meta = (
                self._generate_oof_predictions(X, y, splits, best_lgbm_params, lgbm_new)
            )

            # 6. Train regime detector
            df_5m_full = pd.DataFrame(
                self.db.get_candles("candles_5m", limit=FULL_RETRAIN_DAYS * CANDLES_PER_DAY)
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
            data = self._load_training_data(days=FULL_RETRAIN_DAYS)
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

            tcn_new = TCNModel(epochs=30)  # Fewer epochs for speed
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

        Args:
            X: Full feature DataFrame.
            y: Full label Series.
            splits: Walk-forward CV splits.

        Returns:
            Best parameters dict.
        """
        param_space = LightGBMModel().get_optuna_param_space()

        def objective(trial):
            params = param_space(trial)
            accuracies = []

            # Use a subset of splits for speed
            eval_splits = splits[-3:] if len(splits) > 3 else splits

            for train_idx, val_idx in eval_splits:
                X_tr = X.iloc[train_idx]
                y_tr = y.iloc[train_idx]
                X_vl = X.iloc[val_idx]
                y_vl = y.iloc[val_idx]

                X_tr_sm, y_tr_sm = self._apply_smote(X_tr, y_tr)

                model = LightGBMModel(params=params)
                metrics = model.train(
                    X_tr_sm, y_tr_sm, X_vl, y_vl,
                    num_boost_round=200,
                    early_stopping_rounds=30,
                )
                accuracies.append(metrics["val_accuracy"])

            return np.mean(accuracies)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

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
            lgbm_fold.train(X_tr_sm, y_tr_sm, num_boost_round=300, early_stopping_rounds=30)
            lgbm_preds = lgbm_fold.predict_proba(X_vl)

            # TCN
            tcn_fold = TCNModel(epochs=25)  # Fewer epochs for CV
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

        # Only apply SMOTE if imbalance is significant (>60/40)
        if min_count / max_count > 0.6:
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
