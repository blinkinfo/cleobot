"""Ensemble orchestrator for CleoBot.

Orchestrates the full 3-layer ensemble:
  Layer 1: Base models (LightGBM, TCN, LogReg) -> individual predictions
  Layer 2: Meta-learner -> combined prediction with calibrated confidence
  Layer 3: Regime-aware gating -> regime-adjusted thresholds

The ensemble produces a final signal with direction, confidence, regime,
individual model details, and agreement score.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Optional, List, Any

from src.models.lgbm_model import LightGBMModel
from src.models.tcn_model import TCNModel
from src.models.logreg_model import LogRegModel
from src.models.meta_learner import MetaLearner, build_meta_features
from src.models.regime_detector import (
    RegimeDetector,
    REGIME_DISPLAY,
    REGIME_CONFIDENCE_THRESHOLDS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    compute_regime_features,
)
from src.database import Database
from src.utils.logger import get_logger

logger = get_logger("models.ensemble")

def _align_features_to_training(
    features: Dict[str, Any],
    training_feature_names: List[str],
) -> Dict[str, Any]:
    """Align a feature dict to the training feature set.

    - Features not in the training set are dropped (avoids mismatch errors).
    - Features in training set but missing from inference are filled with 0.0.
    This ensures models receive exactly the features they were trained on.
    """
    aligned = {}
    for name in training_feature_names:
        aligned[name] = features.get(name, 0.0)
    return aligned


# Active versions file
ACTIVE_VERSIONS_FILE = "active_versions.json"


class EnsembleSignal:
    """Structured output from the ensemble prediction."""

    def __init__(
        self,
        direction: str,
        confidence: float,
        probability: float,
        regime: str,
        regime_display: str,
        regime_confidence: float,
        lgbm_result: Dict[str, Any],
        tcn_result: Dict[str, Any],
        logreg_result: Dict[str, Any],
        agreement: int,
        regime_threshold: float,
        inference_time_ms: float,
    ):
        self.direction = direction
        self.confidence = confidence
        self.probability = probability
        self.regime = regime
        self.regime_display = regime_display
        self.regime_confidence = regime_confidence
        self.lgbm = lgbm_result
        self.tcn = tcn_result
        self.logreg = logreg_result
        self.agreement = agreement
        self.regime_threshold = regime_threshold
        self.inference_time_ms = inference_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialisation."""
        return {
            "direction": self.direction,
            "confidence": self.confidence,
            "probability": self.probability,
            "regime": self.regime,
            "regime_display": self.regime_display,
            "regime_confidence": self.regime_confidence,
            "models": {
                "lgbm": self.lgbm,
                "tcn": self.tcn,
                "logreg": self.logreg,
            },
            "agreement": self.agreement,
            "regime_threshold": self.regime_threshold,
            "inference_time_ms": self.inference_time_ms,
        }


class Ensemble:
    """Orchestrates the 3-layer model ensemble.

    Coordinates base model predictions, meta-learner combination, and
    regime-aware gating to produce calibrated trading signals.
    """

    def __init__(self, models_dir: str, db: Optional[Database] = None):
        """Initialise the ensemble.

        Args:
            models_dir: Directory for model storage.
            db: Optional database instance (used for loading training metadata).
        """
        self.models_dir = models_dir
        self.db = db
        os.makedirs(models_dir, exist_ok=True)

        # Base models (Layer 1)
        self.lgbm = LightGBMModel()
        self.tcn = TCNModel()
        self.logreg = LogRegModel()

        # Meta-learner (Layer 2)
        self.meta = MetaLearner()

        # Regime detector (Layer 3)
        self.regime_detector = RegimeDetector()

        # Version tracking
        self.active_versions: Dict[str, int] = {
            "lgbm": 0,
            "tcn": 0,
            "logreg": 0,
            "meta": 0,
            "hmm": 0,
        }

        # Stats
        self._prediction_count: int = 0
        self._total_inference_ms: float = 0.0

        # Training feature names for inference-time alignment (set by Trainer after training)
        self._training_feature_names: Optional[List[str]] = None

        # (Rate-limiter removed -- the scheduler already ensures 5-min spacing
        # between cycles, so caching stale signals caused more harm than good.)

    @property
    def is_ready(self) -> bool:
        """Check if all models are trained and ready for prediction."""
        return (
            self.lgbm.is_trained
            and self.tcn.is_trained
            and self.logreg.is_trained
            and self.meta.is_trained
            and self.regime_detector.is_trained
        )

    def predict(
        self,
        features: Dict[str, float],
        df_5m: pd.DataFrame,
        feature_df_history: Optional[pd.DataFrame] = None,
    ) -> EnsembleSignal:
        """Generate a full ensemble prediction.

        Args:
            features: Current feature dict from FeatureEngine.
            df_5m: Recent 5m candle DataFrame for regime detection and TCN.
            feature_df_history: Historical feature DataFrame for TCN
                (at least seq_length rows). If None, TCN uses neutral prediction.

        Returns:
            EnsembleSignal with full prediction details.
        """
        t0 = time.monotonic()

        if not self.is_ready:
            logger.warning("Ensemble not ready -- returning neutral signal.")
            return self._neutral_signal(0.0)

        # --- Layer 1: Base model predictions --- #
        # Align features to training set if training feature names are known
        if self._training_feature_names is not None:
            features = _align_features_to_training(features, self._training_feature_names)
        features_df = pd.DataFrame([features])

        # LightGBM: single-row prediction
        lgbm_result = self.lgbm.predict_single(features)

        # TCN: needs sequence of features
        if feature_df_history is not None and len(feature_df_history) >= self.tcn.seq_length:
            tcn_result = self.tcn.predict_single(feature_df_history)
        else:
            tcn_result = {"direction": "UP", "probability": 0.5, "confidence": 0.0}
            logger.debug("TCN: insufficient history, using neutral prediction.")

        # LogReg: single-row prediction
        logreg_result = self.logreg.predict_single(features)

        # --- Layer 3: Regime detection --- #
        regime_info = self.regime_detector.predict_with_proba(df_5m)
        regime = regime_info["regime"]
        regime_display = regime_info["display"]
        regime_confidence = regime_info["confidence"]

        # --- Agreement score --- #
        directions = [
            lgbm_result["direction"],
            tcn_result["direction"],
            logreg_result["direction"],
        ]
        up_votes = sum(1 for d in directions if d == "UP")
        down_votes = 3 - up_votes
        agreement = max(up_votes, down_votes)  # 2 or 3

        # --- Layer 2: Meta-learner --- #
        # Get volatility percentile from features
        vol_pctl = self._get_volatility_percentile(features)
        now_utc = datetime.now(timezone.utc)
        hour = now_utc.hour

        meta_result = self.meta.predict_single(
            lgbm_proba=lgbm_result["probability"],
            tcn_proba=tcn_result["probability"],
            logreg_proba=logreg_result["probability"],
            regime=regime,
            volatility_percentile=vol_pctl,
            hour_of_day=hour,
            calibrated=True,
        )

        # --- Final signal --- #
        direction = meta_result["direction"]
        probability = meta_result["probability"]
        confidence = meta_result["confidence"]

        # Regime-specific threshold
        regime_threshold = REGIME_CONFIDENCE_THRESHOLDS.get(
            regime, DEFAULT_CONFIDENCE_THRESHOLD
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._prediction_count += 1
        self._total_inference_ms += elapsed_ms

        signal = EnsembleSignal(
            direction=direction,
            confidence=confidence,
            probability=probability,
            regime=regime,
            regime_display=regime_display,
            regime_confidence=regime_confidence,
            lgbm_result=lgbm_result,
            tcn_result=tcn_result,
            logreg_result=logreg_result,
            agreement=agreement,
            regime_threshold=regime_threshold,
            inference_time_ms=elapsed_ms,
        )

        logger.info(
            f"Ensemble prediction: {direction} "
            f"(conf={confidence:.3f}, prob={probability:.3f}, "
            f"regime={regime_display}, agree={agreement}/3, "
            f"time={elapsed_ms:.1f}ms)"
        )

        return signal

    def load_models(self) -> bool:
        """Load all models from disk.

        Returns:
            True if all models loaded successfully, False otherwise.
        """
        versions = self._load_active_versions()
        success = True

        # Load each model
        for name, loader, ext in [
            ("lgbm", self.lgbm, "pkl"),
            ("tcn", self.tcn, "pt"),
            ("logreg", self.logreg, "pkl"),
            ("meta", self.meta, "pkl"),
            ("hmm", self.regime_detector, "pkl"),
        ]:
            version = versions.get(name, 0)
            path = os.path.join(self.models_dir, f"{name}_v{version}.{ext}")
            if os.path.exists(path):
                try:
                    loader.load(path)
                    loader.version = version
                    self.active_versions[name] = version
                    logger.info(f"Loaded {name} v{version}")
                except Exception as e:
                    logger.error(f"Failed to load {name} from {path}: {e}")
                    success = False
            else:
                logger.warning(f"Model file not found: {path}")
                success = False

        # Load training feature names for inference alignment
        feature_names_path = os.path.join(self.models_dir, "training_feature_names.json")
        if os.path.exists(feature_names_path):
            with open(feature_names_path, "r") as f:
                self._training_feature_names = json.load(f)
            logger.info(f"Loaded {len(self._training_feature_names)} training feature names")
        else:
            self._training_feature_names = None

        if self.is_ready:
            logger.info("All ensemble models loaded successfully.")
        else:
            logger.warning("Ensemble not fully loaded -- training required.")

        return success

    def save_models(self):
        """Save all models to disk with versioning."""
        for name, model, ext in [
            ("lgbm", self.lgbm, "pkl"),
            ("tcn", self.tcn, "pt"),
            ("logreg", self.logreg, "pkl"),
            ("meta", self.meta, "pkl"),
            ("hmm", self.regime_detector, "pkl"),
        ]:
            if model.is_trained:
                version = model.version
                path = os.path.join(self.models_dir, f"{name}_v{version}.{ext}")
                model.save(path)
                self.active_versions[name] = version

        self._save_active_versions()

        # Save training feature names for inference alignment
        if self._training_feature_names is not None:
            feature_names_path = os.path.join(self.models_dir, "training_feature_names.json")
            with open(feature_names_path, "w") as f:
                json.dump(self._training_feature_names, f)
            logger.info(f"Saved {len(self._training_feature_names)} training feature names")

        self._cleanup_old_versions()
        logger.info(f"All models saved. Active versions: {self.active_versions}")

    def get_model_health(self) -> Dict[str, Any]:
        """Get health status of all models.

        Returns:
            Dict with model health information.
        """
        return {
            "is_ready": self.is_ready,
            "lgbm": {
                "trained": self.lgbm.is_trained,
                "version": self.lgbm.version,
                "train_accuracy": self.lgbm.train_accuracy,
                "val_accuracy": self.lgbm.val_accuracy,
            },
            "tcn": {
                "trained": self.tcn.is_trained,
                "version": self.tcn.version,
                "train_accuracy": self.tcn.train_accuracy,
                "val_accuracy": self.tcn.val_accuracy,
            },
            "logreg": {
                "trained": self.logreg.is_trained,
                "version": self.logreg.version,
                "train_accuracy": self.logreg.train_accuracy,
                "val_accuracy": self.logreg.val_accuracy,
            },
            "meta": {
                "trained": self.meta.is_trained,
                "version": self.meta.version,
                "train_accuracy": self.meta.train_accuracy,
                "val_accuracy": self.meta.val_accuracy,
            },
            "regime_detector": {
                "trained": self.regime_detector.is_trained,
                "version": self.regime_detector.version,
            },
            "active_versions": dict(self.active_versions),
            "prediction_count": self._prediction_count,
            "avg_inference_ms": (
                self._total_inference_ms / max(self._prediction_count, 1)
            ),
        }

    def get_feature_rankings(self) -> List[Dict[str, Any]]:
        """Get combined feature importance rankings.

        Returns:
            List of dicts with feature name, LightGBM importance, and
            LogReg coefficient magnitude.
        """
        rankings = []

        lgbm_imp = dict(self.lgbm.get_top_features(50)) if self.lgbm.is_trained else {}
        logreg_coefs = (
            dict(self.logreg.get_coefficients()) if self.logreg.is_trained else {}
        )

        all_features = set(lgbm_imp.keys()) | set(logreg_coefs.keys())
        for feat in all_features:
            rankings.append({
                "feature": feat,
                "lgbm_importance": lgbm_imp.get(feat, 0.0),
                "logreg_coefficient": logreg_coefs.get(feat, 0.0),
            })

        # Sort by LightGBM importance
        rankings.sort(key=lambda x: x["lgbm_importance"], reverse=True)
        return rankings

    def _get_volatility_percentile(self, features: Dict[str, float]) -> float:
        """Extract volatility percentile from features.

        Looks for ATR-based features and normalises to 0-100 percentile.
        Falls back to 50 if no relevant feature found.
        """
        # The feature engine produces atr_12 -- use a simple heuristic
        # The actual percentile calculation is done in the feature engine
        # We look for pre-computed percentile features or estimate
        for key in ["atr_percentile", "vol_atr_percentile"]:
            if key in features:
                return features[key] * 100.0

        # Fallback: use normalized ATR value (assume roughly normal distribution)
        atr = features.get("atr_12", 0.0)
        if atr > 0:
            # Rough mapping: use the candle_position_in_range as proxy
            pos = features.get("candle_position", 0.5)
            return pos * 100.0
        return 50.0

    def _neutral_signal(self, elapsed_ms: float) -> EnsembleSignal:
        """Create a neutral signal when ensemble is not ready."""
        neutral_model = {"direction": "UP", "probability": 0.5, "confidence": 0.0}
        return EnsembleSignal(
            direction="UP",
            confidence=0.0,
            probability=0.5,
            regime="low_vol_ranging",
            regime_display="Low-Vol Ranging",
            regime_confidence=0.0,
            lgbm_result=dict(neutral_model),
            tcn_result=dict(neutral_model),
            logreg_result=dict(neutral_model),
            agreement=3,
            regime_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            inference_time_ms=elapsed_ms,
        )

    def _load_active_versions(self) -> Dict[str, int]:
        """Load active model versions from JSON file."""
        path = os.path.join(self.models_dir, ACTIVE_VERSIONS_FILE)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load active versions: {e}")
        return {"lgbm": 0, "tcn": 0, "logreg": 0, "meta": 0, "hmm": 0}

    def _save_active_versions(self):
        """Save active model versions to JSON file."""
        path = os.path.join(self.models_dir, ACTIVE_VERSIONS_FILE)
        try:
            with open(path, "w") as f:
                json.dump(self.active_versions, f, indent=2)
            logger.debug(f"Active versions saved: {self.active_versions}")
        except Exception as e:
            logger.error(f"Failed to save active versions: {e}")

    def _cleanup_old_versions(self, keep: int = 3):
        """Remove old model versions, keeping the last N.

        Args:
            keep: Number of versions to retain per model.
        """
        for name, ext in [
            ("lgbm", "pkl"),
            ("tcn", "pt"),
            ("logreg", "pkl"),
            ("meta", "pkl"),
            ("hmm", "pkl"),
        ]:
            current = self.active_versions.get(name, 0)
            # Find all version files
            versions_to_check = range(max(0, current - keep - 5), current + 1)
            for v in versions_to_check:
                if v < current - keep + 1 and v >= 0:
                    old_path = os.path.join(self.models_dir, f"{name}_v{v}.{ext}")
                    if os.path.exists(old_path):
                        try:
                            os.remove(old_path)
                            logger.debug(f"Removed old model: {old_path}")
                        except OSError:
                            pass
