"""LightGBM model wrapper for CleoBot.

Model A in the 3-layer ensemble. Handles non-linear feature interactions
on the full engineered feature set (80-120 features).

Expected standalone accuracy: 54-57%
Inference time: <100ms
"""

import os
import json
import pickle
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Any

import lightgbm as lgb

from src.utils.logger import get_logger

logger = get_logger("models.lgbm")

# Default hyperparameters (tuned via Optuna during training)
DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "n_jobs": 1,
    "seed": 42,
}


class LightGBMModel:
    """LightGBM binary classifier for BTC direction prediction.

    Wraps LightGBM with train/predict/save/load and Optuna hyperparameter
    tuning support.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialise the LightGBM model.

        Args:
            params: LightGBM parameters. Uses defaults if not provided.
        """
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self.version: int = 0
        self.train_accuracy: float = 0.0
        self.val_accuracy: float = 0.0
        self._is_trained: bool = False

    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained or loaded."""
        return self._is_trained and self.model is not None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 20,
    ) -> Dict[str, float]:
        """Train the LightGBM model.

        Args:
            X_train: Training features DataFrame.
            y_train: Training labels (0=DOWN, 1=UP).
            X_val: Optional validation features.
            y_val: Optional validation labels.
            num_boost_round: Maximum boosting rounds.
            early_stopping_rounds: Stop if no improvement for N rounds.

        Returns:
            Dict with training metrics (train_accuracy, val_accuracy, best_iteration).
        """
        logger.info(
            f"Training LightGBM: {X_train.shape[0]} samples, "
            f"{X_train.shape[1]} features"
        )

        self.feature_names = list(X_train.columns)
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ["train"]
        callbacks = [lgb.log_evaluation(period=100)]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")
            callbacks.append(
                lgb.early_stopping(stopping_rounds=early_stopping_rounds)
            )

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Compute accuracies
        train_preds = (self.model.predict(X_train) > 0.5).astype(int)
        self.train_accuracy = float(np.mean(train_preds == y_train.values))

        if X_val is not None and y_val is not None:
            val_preds = (self.model.predict(X_val) > 0.5).astype(int)
            self.val_accuracy = float(np.mean(val_preds == y_val.values))
        else:
            self.val_accuracy = self.train_accuracy

        # Feature importance
        importance = self.model.feature_importance(importance_type="gain")
        total = importance.sum()
        if total > 0:
            self.feature_importance = {
                name: float(imp / total)
                for name, imp in zip(self.feature_names, importance)
            }
        else:
            self.feature_importance = {
                name: 0.0 for name in self.feature_names
            }

        self._is_trained = True

        metrics = {
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "best_iteration": self.model.best_iteration
            if self.model.best_iteration > 0
            else num_boost_round,
            "num_features": len(self.feature_names),
        }
        logger.info(
            f"LightGBM trained: train_acc={self.train_accuracy:.4f}, "
            f"val_acc={self.val_accuracy:.4f}, "
            f"best_iter={metrics['best_iteration']}"
        )
        return metrics

    def train_incremental(
        self,
        X_new: pd.DataFrame,
        y_new: pd.Series,
        num_boost_round: int = 50,
    ) -> Dict[str, float]:
        """Incrementally train (warm start) with new data.

        Adds additional boosting rounds using new data without retraining
        from scratch. Used for 6-hourly incremental updates.

        Args:
            X_new: New training features.
            y_new: New training labels.
            num_boost_round: Additional boosting rounds.

        Returns:
            Dict with updated metrics.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot incrementally train: no base model loaded.")

        logger.info(
            f"Incremental LightGBM update: {X_new.shape[0]} new samples, "
            f"+{num_boost_round} rounds"
        )

        new_data = lgb.Dataset(X_new, label=y_new)

        self.model = lgb.train(
            self.params,
            new_data,
            num_boost_round=num_boost_round,
            init_model=self.model,
            callbacks=[lgb.log_evaluation(period=50)],
        )

        # Recompute accuracy on new data
        preds = (self.model.predict(X_new) > 0.5).astype(int)
        new_accuracy = float(np.mean(preds == y_new.values))

        # Update feature importance
        importance = self.model.feature_importance(importance_type="gain")
        total = importance.sum()
        if total > 0:
            self.feature_importance = {
                name: float(imp / total)
                for name, imp in zip(self.feature_names, importance)
            }

        metrics = {
            "incremental_accuracy": new_accuracy,
            "additional_rounds": num_boost_round,
        }
        logger.info(f"Incremental update done: accuracy on new data={new_accuracy:.4f}")
        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict UP probability for each sample.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of UP probabilities (shape: [n_samples]).
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        # Ensure feature alignment
        X_aligned = self._align_features(X)
        return self.model.predict(X_aligned)

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict direction and confidence.

        Args:
            X: Feature DataFrame.

        Returns:
            Tuple of (directions, confidences) where:
              - directions: array of 1 (UP) or 0 (DOWN)
              - confidences: array of confidence scores (distance from 0.5)
        """
        probas = self.predict_proba(X)
        directions = (probas > 0.5).astype(int)
        confidences = np.abs(probas - 0.5) * 2  # Scale to 0-1
        return directions, confidences

    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict for a single sample from a feature dict.

        Args:
            features: Dict mapping feature_name -> value.

        Returns:
            Dict with 'direction' ('UP'/'DOWN'), 'probability', 'confidence'.
        """
        df = pd.DataFrame([features])
        proba = float(self.predict_proba(df)[0])
        direction = "UP" if proba > 0.5 else "DOWN"
        confidence = abs(proba - 0.5) * 2
        return {
            "direction": direction,
            "probability": proba,
            "confidence": confidence,
        }

    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get top N features by importance.

        Args:
            n: Number of top features.

        Returns:
            List of (feature_name, importance_score) tuples, sorted descending.
        """
        sorted_feats = sorted(
            self.feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_feats[:n]

    def save(self, path: str):
        """Save model to disk.

        Args:
            path: File path for the pickle file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "model_str": self.model.model_to_string() if self.model else None,
            "params": self.params,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance,
            "version": self.version,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"LightGBM model saved to {path}")

    def load(self, path: str):
        """Load model from disk.

        Args:
            path: File path for the pickle file.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.params = state["params"]
        self.feature_names = state["feature_names"]
        self.feature_importance = state["feature_importance"]
        self.version = state["version"]
        self.train_accuracy = state["train_accuracy"]
        self.val_accuracy = state["val_accuracy"]
        if state["model_str"]:
            self.model = lgb.Booster(model_str=state["model_str"])
        self._is_trained = True
        logger.info(f"LightGBM model loaded from {path} (v{self.version})")

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align input features to training feature order.

        Adds missing columns as zeros and reorders to match training.

        Args:
            X: Input DataFrame.

        Returns:
            Aligned DataFrame with correct column order.
        """
        if not self.feature_names:
            return X

        # Add any missing columns with zeros
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            for col in missing:
                X = X.copy()
                X[col] = 0.0
            logger.debug(f"Added {len(missing)} missing features as zeros")

        # Reorder to match training order
        return X[self.feature_names]

    def get_optuna_param_space(self):
        """Return the Optuna hyperparameter search space.

        Returns:
            Callable that takes an Optuna trial and returns params dict.
        """

        def suggest_params(trial):
            return {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.02, 0.15, log=True
                ),
                "min_child_samples": trial.suggest_int("min_child_samples", 15, 50),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
                "verbose": -1,
                "n_jobs": 1,
                "seed": 42,
            }

        return suggest_params
