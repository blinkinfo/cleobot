"""Meta-learner for CleoBot ensemble.

Layer 2 of the 3-layer ensemble. Trained on out-of-fold (OOF) predictions
from all 3 base models to combine their signals optimally.

Input features (per Section 4):
  - Base model probabilities (3 values)
  - Base model confidence scores (3 values)
  - Model agreement score (0, 1, 2, or 3)
  - Current regime label (one-hot, 4 values)
  - Volatility percentile (1 value)
  - Hour-of-day cyclical encoding (2 values: sin, cos)
  Total: ~16 meta-features

Implementation: XGBoost (max_depth=3, n_estimators=50) or Logistic Regression
This is where 56-58%+ accuracy emerges from combining weaker signals.
"""

import os
import pickle
import math
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any, Tuple

import xgboost as xgb
from sklearn.calibration import IsotonicRegression

from src.utils.logger import get_logger

logger = get_logger("models.meta_learner")

# Regime labels for one-hot encoding
REGIME_LABELS = [
    "low_vol_ranging",
    "trending_up",
    "trending_down",
    "high_vol_chaotic",
]


def build_meta_features(
    lgbm_proba: float,
    tcn_proba: float,
    logreg_proba: float,
    regime: str,
    volatility_percentile: float,
    hour_of_day: int,
) -> Dict[str, float]:
    """Build the meta-feature vector from base model outputs and context.

    Args:
        lgbm_proba: LightGBM UP probability.
        tcn_proba: TCN UP probability.
        logreg_proba: Logistic Regression UP probability.
        regime: Current regime label string.
        volatility_percentile: ATR percentile (0-100).
        hour_of_day: Current UTC hour (0-23).

    Returns:
        Dict of meta-feature name -> value.
    """
    # Base probabilities
    features = {
        "lgbm_proba": lgbm_proba,
        "tcn_proba": tcn_proba,
        "logreg_proba": logreg_proba,
    }

    # Confidence scores (distance from 0.5, scaled 0-1)
    features["lgbm_confidence"] = abs(lgbm_proba - 0.5) * 2
    features["tcn_confidence"] = abs(tcn_proba - 0.5) * 2
    features["logreg_confidence"] = abs(logreg_proba - 0.5) * 2

    # Agreement score (how many models agree on UP)
    votes_up = sum(
        1 for p in [lgbm_proba, tcn_proba, logreg_proba] if p > 0.5
    )
    votes_down = 3 - votes_up
    features["agreement"] = float(max(votes_up, votes_down))  # 2 or 3

    # Regime one-hot encoding
    regime_lower = regime.lower().replace(" ", "_").replace("-", "_")
    for label in REGIME_LABELS:
        features[f"regime_{label}"] = 1.0 if regime_lower == label else 0.0

    # Volatility percentile (normalised 0-1)
    features["volatility_pctl"] = volatility_percentile / 100.0

    # Hour-of-day cyclical encoding
    hour_rad = 2.0 * math.pi * hour_of_day / 24.0
    features["hour_sin"] = math.sin(hour_rad)
    features["hour_cos"] = math.cos(hour_rad)

    return features


def build_meta_features_batch(
    lgbm_probas: np.ndarray,
    tcn_probas: np.ndarray,
    logreg_probas: np.ndarray,
    regimes: List[str],
    volatility_pctls: np.ndarray,
    hours: np.ndarray,
) -> pd.DataFrame:
    """Build meta-features for a batch of samples.

    Args:
        lgbm_probas: Array of LightGBM probabilities.
        tcn_probas: Array of TCN probabilities.
        logreg_probas: Array of LogReg probabilities.
        regimes: List of regime labels.
        volatility_pctls: Array of volatility percentiles.
        hours: Array of hour-of-day values.

    Returns:
        DataFrame of meta-features (n_samples, n_meta_features).
    """
    rows = []
    for i in range(len(lgbm_probas)):
        row = build_meta_features(
            lgbm_proba=float(lgbm_probas[i]),
            tcn_proba=float(tcn_probas[i]),
            logreg_proba=float(logreg_probas[i]),
            regime=regimes[i] if i < len(regimes) else "low_vol_ranging",
            volatility_percentile=float(volatility_pctls[i]),
            hour_of_day=int(hours[i]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


class MetaLearner:
    """XGBoost-based meta-learner for combining base model predictions.

    Includes isotonic regression for confidence calibration.
    """

    def __init__(
        self,
        max_depth: int = 3,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
    ):
        """Initialise the meta-learner.

        Args:
            max_depth: Maximum tree depth.
            n_estimators: Number of boosting rounds.
            learning_rate: XGBoost learning rate.
        """
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.model: Optional[xgb.XGBClassifier] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.meta_feature_names: List[str] = []
        self.version: int = 0
        self.train_accuracy: float = 0.0
        self.val_accuracy: float = 0.0
        self._is_trained: bool = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained and self.model is not None

    def train(
        self,
        X_meta_train: pd.DataFrame,
        y_train: pd.Series,
        X_meta_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Train the meta-learner on OOF predictions from base models.

        Args:
            X_meta_train: Meta-features DataFrame (OOF predictions + context).
            y_train: Training labels (0=DOWN, 1=UP).
            X_meta_val: Validation meta-features.
            y_val: Validation labels.

        Returns:
            Dict with training metrics.
        """
        self.meta_feature_names = list(X_meta_train.columns)

        logger.info(
            f"Training MetaLearner: {X_meta_train.shape[0]} samples, "
            f"{X_meta_train.shape[1]} meta-features"
        )

        self.model = xgb.XGBClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )

        eval_set = [(X_meta_train, y_train)]
        if X_meta_val is not None and y_val is not None:
            eval_set.append((X_meta_val, y_val))

        self.model.fit(
            X_meta_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        # Training accuracy
        train_preds = self.model.predict(X_meta_train)
        self.train_accuracy = float(np.mean(train_preds == y_train.values))

        # Validation accuracy
        if X_meta_val is not None and y_val is not None:
            val_preds = self.model.predict(X_meta_val)
            self.val_accuracy = float(np.mean(val_preds == y_val.values))
        else:
            self.val_accuracy = self.train_accuracy

        # Fit calibrator on validation set (or training set as fallback)
        if X_meta_val is not None and y_val is not None:
            cal_probas = self.model.predict_proba(X_meta_val)[:, 1]
            cal_labels = y_val.values
        else:
            cal_probas = self.model.predict_proba(X_meta_train)[:, 1]
            cal_labels = y_train.values

        self.calibrator = IsotonicRegression(
            y_min=0.01, y_max=0.99, out_of_bounds="clip"
        )
        self.calibrator.fit(cal_probas, cal_labels)

        self._is_trained = True

        metrics = {
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "n_meta_features": len(self.meta_feature_names),
        }
        logger.info(
            f"MetaLearner trained: train_acc={self.train_accuracy:.4f}, "
            f"val_acc={self.val_accuracy:.4f}"
        )
        return metrics

    def predict_proba(self, X_meta: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        """Predict UP probability.

        Args:
            X_meta: Meta-features DataFrame.
            calibrated: Whether to apply isotonic calibration.

        Returns:
            Array of UP probabilities.
        """
        if not self.is_trained:
            raise RuntimeError("MetaLearner not trained.")

        X_aligned = self._align_features(X_meta)
        raw_probas = self.model.predict_proba(X_aligned)[:, 1]

        if calibrated and self.calibrator is not None:
            return self.calibrator.predict(raw_probas)
        return raw_probas

    def predict_single(
        self,
        lgbm_proba: float,
        tcn_proba: float,
        logreg_proba: float,
        regime: str,
        volatility_percentile: float,
        hour_of_day: int,
        calibrated: bool = True,
    ) -> Dict[str, Any]:
        """Predict for a single moment.

        Args:
            lgbm_proba: LightGBM UP probability.
            tcn_proba: TCN UP probability.
            logreg_proba: LogReg UP probability.
            regime: Current regime label.
            volatility_percentile: ATR percentile (0-100).
            hour_of_day: Current UTC hour.
            calibrated: Apply isotonic calibration.

        Returns:
            Dict with 'direction', 'probability', 'confidence'.
        """
        meta_feats = build_meta_features(
            lgbm_proba, tcn_proba, logreg_proba,
            regime, volatility_percentile, hour_of_day,
        )
        df = pd.DataFrame([meta_feats])
        proba = float(self.predict_proba(df, calibrated=calibrated)[0])
        direction = "UP" if proba > 0.5 else "DOWN"
        confidence = abs(proba - 0.5) * 2
        return {
            "direction": direction,
            "probability": proba,
            "confidence": confidence,
        }

    def recalibrate(self, probas: np.ndarray, labels: np.ndarray):
        """Recalibrate confidence scores with new data.

        Used during incremental updates (every 6 hours).

        Args:
            probas: Recent predicted probabilities.
            labels: Actual outcomes (0/1).
        """
        if len(probas) < 10:
            logger.warning("Not enough samples for recalibration (need 10+).")
            return

        self.calibrator = IsotonicRegression(
            y_min=0.01, y_max=0.99, out_of_bounds="clip"
        )
        self.calibrator.fit(probas, labels)
        logger.info(f"MetaLearner recalibrated with {len(probas)} samples.")

    def save(self, path: str):
        """Save meta-learner to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "model": self.model,
            "calibrator": self.calibrator,
            "meta_feature_names": self.meta_feature_names,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "version": self.version,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"MetaLearner saved to {path}")

    def load(self, path: str):
        """Load meta-learner from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.model = state["model"]
        self.calibrator = state["calibrator"]
        self.meta_feature_names = state["meta_feature_names"]
        self.max_depth = state["max_depth"]
        self.n_estimators = state["n_estimators"]
        self.learning_rate = state["learning_rate"]
        self.version = state["version"]
        self.train_accuracy = state["train_accuracy"]
        self.val_accuracy = state["val_accuracy"]
        self._is_trained = True
        logger.info(f"MetaLearner loaded from {path} (v{self.version})")

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align meta-features to training order."""
        if not self.meta_feature_names:
            return X
        missing = set(self.meta_feature_names) - set(X.columns)
        if missing:
            X = X.copy()
            for col in missing:
                X[col] = 0.0
        return X[self.meta_feature_names]

    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """Get meta-feature importance."""
        if not self.is_trained:
            return []
        importance = self.model.feature_importances_
        pairs = list(zip(self.meta_feature_names, importance))
        return sorted(pairs, key=lambda x: x[1], reverse=True)
