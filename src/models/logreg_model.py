"""Logistic Regression model wrapper for CleoBot.

Model C in the 3-layer ensemble. Serves as a robust baseline and
tie-breaker when LightGBM and TCN disagree.

Input: Top 15-20 most important features (selected by LightGBM
       feature importance)
Regularisation: L2 (Ridge)
Expected standalone accuracy: 52-54%

Also acts as a sanity check: if complex models cannot beat LogReg,
something is fundamentally wrong with the pipeline.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Any

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger("models.logreg")

# Default hyperparameters
DEFAULT_TOP_FEATURES = 20
DEFAULT_C = 1.0  # Inverse regularisation strength
DEFAULT_MAX_ITER = 1000


class LogRegModel:
    """Logistic Regression binary classifier for BTC direction prediction.

    Uses a subset of top features selected by LightGBM feature importance.
    Includes built-in StandardScaler for feature normalisation.
    """

    def __init__(
        self,
        top_n_features: int = DEFAULT_TOP_FEATURES,
        C: float = DEFAULT_C,
        max_iter: int = DEFAULT_MAX_ITER,
    ):
        """Initialise the Logistic Regression model.

        Args:
            top_n_features: Number of top features to select.
            C: Inverse regularisation strength (smaller = stronger L2).
            max_iter: Maximum solver iterations.
        """
        self.top_n_features = top_n_features
        self.C = C
        self.max_iter = max_iter

        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.selected_features: List[str] = []
        self.all_feature_names: List[str] = []
        self.version: int = 0
        self.train_accuracy: float = 0.0
        self.val_accuracy: float = 0.0
        self._is_trained: bool = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained and self.model is not None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Train the Logistic Regression model.

        Args:
            X_train: Training features DataFrame.
            y_train: Training labels (0=DOWN, 1=UP).
            X_val: Optional validation features.
            y_val: Optional validation labels.
            feature_importance: Dict of feature_name -> importance from
                LightGBM. If provided, selects the top N features.
                If None, uses all features.

        Returns:
            Dict with training metrics.
        """
        self.all_feature_names = list(X_train.columns)

        # Select top features
        if feature_importance:
            sorted_feats = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            # Only keep features that exist in the training data
            available = set(X_train.columns)
            self.selected_features = [
                name
                for name, _ in sorted_feats
                if name in available
            ][: self.top_n_features]
        else:
            # Use all features if no importance provided
            self.selected_features = list(X_train.columns)[: self.top_n_features]

        if not self.selected_features:
            self.selected_features = list(X_train.columns)[: self.top_n_features]

        logger.info(
            f"Training LogReg: {X_train.shape[0]} samples, "
            f"{len(self.selected_features)} selected features"
        )

        X_sel = X_train[self.selected_features].values

        # Fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_sel)

        # Train model
        self.model = LogisticRegression(
            C=self.C,
            penalty="l2",
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=42,
        )
        self.model.fit(X_scaled, y_train.values)

        # Training accuracy
        train_preds = self.model.predict(X_scaled)
        self.train_accuracy = float(np.mean(train_preds == y_train.values))

        # Validation accuracy
        if X_val is not None and y_val is not None:
            X_val_sel = X_val[self.selected_features].values
            X_val_scaled = self.scaler.transform(X_val_sel)
            val_preds = self.model.predict(X_val_scaled)
            self.val_accuracy = float(np.mean(val_preds == y_val.values))
        else:
            self.val_accuracy = self.train_accuracy

        self._is_trained = True

        metrics = {
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "num_features": len(self.selected_features),
            "C": self.C,
        }
        logger.info(
            f"LogReg trained: train_acc={self.train_accuracy:.4f}, "
            f"val_acc={self.val_accuracy:.4f}, "
            f"features={len(self.selected_features)}"
        )
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

        X_sel = self._align_features(X)
        X_scaled = self.scaler.transform(X_sel)
        # predict_proba returns [[P(0), P(1)], ...]
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict direction and confidence.

        Args:
            X: Feature DataFrame.

        Returns:
            Tuple of (directions, confidences).
        """
        probas = self.predict_proba(X)
        directions = (probas > 0.5).astype(int)
        confidences = np.abs(probas - 0.5) * 2
        return directions, confidences

    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict for a single sample from a feature dict.

        Args:
            features: Dict mapping feature_name -> value.

        Returns:
            Dict with 'direction', 'probability', 'confidence'.
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

    def save(self, path: str):
        """Save model to disk.

        Args:
            path: File path for the pickle file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "model": self.model,
            "scaler": self.scaler,
            "selected_features": self.selected_features,
            "all_feature_names": self.all_feature_names,
            "top_n_features": self.top_n_features,
            "C": self.C,
            "max_iter": self.max_iter,
            "version": self.version,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"LogReg model saved to {path}")

    def load(self, path: str):
        """Load model from disk.

        Args:
            path: File path for the pickle file.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.model = state["model"]
        self.scaler = state["scaler"]
        self.selected_features = state["selected_features"]
        self.all_feature_names = state["all_feature_names"]
        self.top_n_features = state["top_n_features"]
        self.C = state["C"]
        self.max_iter = state["max_iter"]
        self.version = state["version"]
        self.train_accuracy = state["train_accuracy"]
        self.val_accuracy = state["val_accuracy"]
        self._is_trained = True
        logger.info(f"LogReg model loaded from {path} (v{self.version})")

    def _align_features(self, X: pd.DataFrame) -> np.ndarray:
        """Select and align features to training order.

        Args:
            X: Input DataFrame.

        Returns:
            Numpy array with selected features in correct order.
        """
        missing = set(self.selected_features) - set(X.columns)
        if missing:
            X = X.copy()
            for col in missing:
                X[col] = 0.0
        return X[self.selected_features].values

    def get_coefficients(self) -> List[Tuple[str, float]]:
        """Get feature coefficients from the model.

        Returns:
            List of (feature_name, coefficient) tuples sorted by absolute
            value (descending).
        """
        if not self.is_trained:
            return []
        coefs = self.model.coef_[0]
        pairs = list(zip(self.selected_features, coefs))
        return sorted(pairs, key=lambda x: abs(x[1]), reverse=True)

    def get_optuna_param_space(self):
        """Return the Optuna hyperparameter search space."""

        def suggest_params(trial):
            return {
                "C": trial.suggest_float("C", 0.01, 100.0, log=True),
                "top_n_features": trial.suggest_int("top_n_features", 10, 30),
            }

        return suggest_params
