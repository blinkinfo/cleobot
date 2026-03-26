"""ML models for CleoBot.

3-Layer Ensemble Architecture:
  Layer 1 (Base Models): LightGBM, TCN, Logistic Regression
  Layer 2 (Meta-Learner): XGBoost combining OOF predictions + context
  Layer 3 (Regime Gating): HMM regime detector with adaptive thresholds

Entry points:
  - Ensemble: Orchestrates all models for live prediction
  - Trainer: Manages training lifecycle (full, incremental, emergency)
"""

from src.models.lgbm_model import LightGBMModel
from src.models.tcn_model import TCNModel, TCNNetwork
from src.models.logreg_model import LogRegModel
from src.models.meta_learner import MetaLearner, build_meta_features, build_meta_features_batch
from src.models.regime_detector import (
    RegimeDetector,
    compute_regime_features,
    REGIME_LABELS,
    REGIME_CONFIDENCE_THRESHOLDS,
)
from src.models.ensemble import Ensemble, EnsembleSignal
from src.models.trainer import Trainer

__all__ = [
    # Base models
    "LightGBMModel",
    "TCNModel",
    "TCNNetwork",
    "LogRegModel",
    # Meta-learner
    "MetaLearner",
    "build_meta_features",
    "build_meta_features_batch",
    # Regime detection
    "RegimeDetector",
    "compute_regime_features",
    "REGIME_LABELS",
    "REGIME_CONFIDENCE_THRESHOLDS",
    # Ensemble
    "Ensemble",
    "EnsembleSignal",
    # Training
    "Trainer",
]
