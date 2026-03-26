"""HMM-based regime detector for CleoBot.

Layer 3 of the 3-layer ensemble. Classifies the current market into
one of 4 regimes using a Hidden Markov Model.

4 Regimes:
  1. Low-Volatility Ranging -- tight spreads, small candles, mean-reverting
  2. Trending Up -- sustained upward momentum, higher lows
  3. Trending Down -- sustained downward momentum, lower highs
  4. High-Volatility Chaotic -- large wicks, no clear direction

Regime Features (input to HMM):
  - Rolling 1h volatility (std of returns)
  - Rolling 1h trend strength (slope of linear regression on closes)
  - Average candle body-to-wick ratio over last 12 candles
  - ADX value
  - Volume relative to 24h average
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any, Tuple

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger("models.regime")

# Regime label mapping
REGIME_LABELS = {
    0: "low_vol_ranging",
    1: "trending_up",
    2: "trending_down",
    3: "high_vol_chaotic",
}

REGIME_DISPLAY = {
    "low_vol_ranging": "Low-Vol Ranging",
    "trending_up": "Trending Up",
    "trending_down": "Trending Down",
    "high_vol_chaotic": "High-Vol Chaotic",
}

# Regime-specific confidence thresholds (from Section 7.3)
REGIME_CONFIDENCE_THRESHOLDS = {
    "low_vol_ranging": 0.62,
    "trending_up": 0.56,
    "trending_down": 0.56,
    "high_vol_chaotic": 0.65,
}

# Default confidence threshold
DEFAULT_CONFIDENCE_THRESHOLD = 0.58

# Number of HMM states
N_REGIMES = 4


def compute_regime_features(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Compute the 5 regime features from 5-minute candle data.

    Args:
        df_5m: DataFrame with columns: open, high, low, close, volume.
              Must be sorted by timestamp ascending.

    Returns:
        DataFrame with regime features, one row per candle.
    """
    if len(df_5m) < 24:
        return pd.DataFrame()

    close = df_5m["close"].values.astype(float)
    high = df_5m["high"].values.astype(float)
    low = df_5m["low"].values.astype(float)
    opn = df_5m["open"].values.astype(float)
    volume = df_5m["volume"].values.astype(float)

    # Returns
    returns = np.zeros(len(close))
    returns[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-8)

    n = len(close)
    features = {
        "regime_volatility": np.full(n, np.nan),
        "regime_trend": np.full(n, np.nan),
        "regime_body_wick": np.full(n, np.nan),
        "regime_adx": np.full(n, np.nan),
        "regime_rel_volume": np.full(n, np.nan),
    }

    for i in range(23, n):
        # 1. Rolling 1h volatility (12 candles = 1 hour)
        window_ret = returns[max(0, i - 11): i + 1]
        features["regime_volatility"][i] = np.std(window_ret) if len(window_ret) > 1 else 0.0

        # 2. Rolling 1h trend strength (slope of linear regression on closes)
        window_close = close[max(0, i - 11): i + 1]
        if len(window_close) > 1:
            x = np.arange(len(window_close))
            coeffs = np.polyfit(x, window_close, 1)
            # Normalise slope by mean price
            mean_price = np.mean(window_close)
            features["regime_trend"][i] = coeffs[0] / max(mean_price, 1e-8)
        else:
            features["regime_trend"][i] = 0.0

        # 3. Average candle body-to-wick ratio over last 12 candles
        body_wick_ratios = []
        for j in range(max(0, i - 11), i + 1):
            candle_range = high[j] - low[j]
            if candle_range > 0:
                body = abs(close[j] - opn[j])
                body_wick_ratios.append(body / candle_range)
            else:
                body_wick_ratios.append(0.5)
        features["regime_body_wick"][i] = np.mean(body_wick_ratios)

        # 4. ADX (14-period, simplified calculation)
        if i >= 14:
            features["regime_adx"][i] = _compute_adx(
                high[max(0, i - 24): i + 1],
                low[max(0, i - 24): i + 1],
                close[max(0, i - 24): i + 1],
                period=14,
            )
        else:
            features["regime_adx"][i] = 25.0  # Neutral default

        # 5. Volume relative to 24h average (24 * 12 = 288 candles at 5m)
        vol_lookback = min(i + 1, 288)
        vol_window = volume[max(0, i - vol_lookback + 1): i + 1]
        vol_avg = np.mean(vol_window) if len(vol_window) > 0 else 1.0
        features["regime_rel_volume"][i] = (
            volume[i] / max(vol_avg, 1e-8) if vol_avg > 0 else 1.0
        )

    df = pd.DataFrame(features)
    return df.dropna().reset_index(drop=True)


def _compute_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> float:
    """Compute ADX (Average Directional Index) value.

    Simplified Wilder's smoothing implementation.

    Returns:
        ADX value (0-100 scale).
    """
    n = len(high)
    if n < period + 1:
        return 25.0  # Neutral

    # True Range, +DM, -DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        h_l = high[i] - low[i]
        h_pc = abs(high[i] - close[i - 1])
        l_pc = abs(low[i] - close[i - 1])
        tr[i] = max(h_l, h_pc, l_pc)

        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

    # Wilder's smoothing
    atr = np.mean(tr[1: period + 1])
    plus_di_smooth = np.mean(plus_dm[1: period + 1])
    minus_di_smooth = np.mean(minus_dm[1: period + 1])

    for i in range(period + 1, n):
        atr = (atr * (period - 1) + tr[i]) / period
        plus_di_smooth = (plus_di_smooth * (period - 1) + plus_dm[i]) / period
        minus_di_smooth = (minus_di_smooth * (period - 1) + minus_dm[i]) / period

    if atr < 1e-8:
        return 25.0

    plus_di = 100.0 * plus_di_smooth / atr
    minus_di = 100.0 * minus_di_smooth / atr
    di_sum = plus_di + minus_di

    if di_sum < 1e-8:
        return 25.0

    dx = 100.0 * abs(plus_di - minus_di) / di_sum
    return float(dx)


class RegimeDetector:
    """HMM-based market regime detector with 4 states.

    After training, assigns regime labels to HMM states based on their
    emission distributions (volatility and trend characteristics).
    """

    def __init__(self, n_regimes: int = N_REGIMES):
        """Initialise the regime detector.

        Args:
            n_regimes: Number of hidden states (default 4).
        """
        self.n_regimes = n_regimes
        self.hmm: Optional[GaussianHMM] = None
        self.scaler: Optional[StandardScaler] = None
        self.state_to_regime: Dict[int, str] = {}
        self.version: int = 0
        self._is_trained: bool = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained and self.hmm is not None

    def train(self, df_5m: pd.DataFrame) -> Dict[str, Any]:
        """Train the HMM regime detector.

        Args:
            df_5m: 5-minute candle DataFrame (sorted by timestamp asc).

        Returns:
            Dict with training metrics.
        """
        # Compute regime features
        regime_feats = compute_regime_features(df_5m)

        if len(regime_feats) < 100:
            logger.warning(
                f"Insufficient data for HMM training: {len(regime_feats)} rows "
                "(need 100+). Using default regime assignments."
            )
            self._setup_default_regimes()
            return {"status": "default", "samples": len(regime_feats)}

        logger.info(f"Training HMM regime detector: {len(regime_feats)} samples")

        # Scale features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(regime_feats.values)

        # Fit HMM
        self.hmm = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            tol=0.01,
        )
        self.hmm.fit(X)

        # Predict states for the entire dataset to assign labels
        states = self.hmm.predict(X)

        # Map HMM states to regime labels based on characteristics
        self._assign_regime_labels(regime_feats, states)

        self._is_trained = True

        # Compute regime distribution
        regime_counts = {}
        for s in states:
            label = self.state_to_regime.get(s, "unknown")
            regime_counts[label] = regime_counts.get(label, 0) + 1

        total = len(states)
        regime_pcts = {
            k: f"{v / total * 100:.1f}%" for k, v in regime_counts.items()
        }

        metrics = {
            "status": "trained",
            "samples": len(regime_feats),
            "regime_distribution": regime_pcts,
            "log_likelihood": float(self.hmm.score(X)),
        }
        logger.info(f"HMM trained: {regime_pcts}")
        return metrics

    def predict(self, df_5m: pd.DataFrame) -> str:
        """Predict the current market regime.

        Args:
            df_5m: Recent 5-minute candle DataFrame (at least 24 rows).

        Returns:
            Regime label string.
        """
        if not self.is_trained:
            return "low_vol_ranging"

        regime_feats = compute_regime_features(df_5m)
        if len(regime_feats) == 0:
            return "low_vol_ranging"

        # Use the last row
        X = self.scaler.transform(regime_feats.values[-1:].reshape(1, -1))
        state = int(self.hmm.predict(X)[0])
        return self.state_to_regime.get(state, "low_vol_ranging")

    def predict_with_proba(self, df_5m: pd.DataFrame) -> Dict[str, Any]:
        """Predict regime with state probabilities.

        Args:
            df_5m: Recent 5-minute candle DataFrame.

        Returns:
            Dict with 'regime', 'confidence', 'probabilities'.
        """
        if not self.is_trained:
            return {
                "regime": "low_vol_ranging",
                "display": "Low-Vol Ranging",
                "confidence": 0.5,
                "probabilities": {r: 0.25 for r in REGIME_LABELS.values()},
            }

        regime_feats = compute_regime_features(df_5m)
        if len(regime_feats) == 0:
            return {
                "regime": "low_vol_ranging",
                "display": "Low-Vol Ranging",
                "confidence": 0.5,
                "probabilities": {r: 0.25 for r in REGIME_LABELS.values()},
            }

        X = self.scaler.transform(regime_feats.values[-1:].reshape(1, -1))

        # Get state probabilities
        log_prob = self.hmm.score_samples(X)
        state = int(self.hmm.predict(X)[0])
        regime = self.state_to_regime.get(state, "low_vol_ranging")

        # Get posterior probabilities for all states
        _, posteriors = self.hmm.score_samples(X)
        state_probs = posteriors[0]

        regime_probs = {}
        for s, label in self.state_to_regime.items():
            if s < len(state_probs):
                regime_probs[label] = float(state_probs[s])
            else:
                regime_probs[label] = 0.0

        confidence = float(max(state_probs))

        return {
            "regime": regime,
            "display": REGIME_DISPLAY.get(regime, regime),
            "confidence": confidence,
            "probabilities": regime_probs,
        }

    def get_confidence_threshold(self, regime: str) -> float:
        """Get the confidence threshold for a given regime.

        Args:
            regime: Regime label.

        Returns:
            Confidence threshold for the meta-learner.
        """
        return REGIME_CONFIDENCE_THRESHOLDS.get(regime, DEFAULT_CONFIDENCE_THRESHOLD)

    def predict_history(self, df_5m: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict regimes for an entire history of candles.

        Args:
            df_5m: Full candle DataFrame.

        Returns:
            List of dicts with 'index', 'regime', 'display' for each valid row.
        """
        if not self.is_trained:
            return []

        regime_feats = compute_regime_features(df_5m)
        if len(regime_feats) == 0:
            return []

        X = self.scaler.transform(regime_feats.values)
        states = self.hmm.predict(X)

        results = []
        for i, s in enumerate(states):
            regime = self.state_to_regime.get(int(s), "low_vol_ranging")
            results.append({
                "index": i,
                "regime": regime,
                "display": REGIME_DISPLAY.get(regime, regime),
            })
        return results

    def save(self, path: str):
        """Save regime detector to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "hmm": self.hmm,
            "scaler": self.scaler,
            "state_to_regime": self.state_to_regime,
            "n_regimes": self.n_regimes,
            "version": self.version,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"RegimeDetector saved to {path}")

    def load(self, path: str):
        """Load regime detector from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.hmm = state["hmm"]
        self.scaler = state["scaler"]
        self.state_to_regime = state["state_to_regime"]
        self.n_regimes = state["n_regimes"]
        self.version = state["version"]
        self._is_trained = True
        logger.info(f"RegimeDetector loaded from {path} (v{self.version})")

    def _assign_regime_labels(
        self, regime_feats: pd.DataFrame, states: np.ndarray
    ):
        """Assign semantic regime labels to HMM states.

        Uses the mean volatility and trend of each state's observations
        to determine which regime it represents.
        """
        state_stats = {}
        for s in range(self.n_regimes):
            mask = states == s
            if mask.sum() == 0:
                state_stats[s] = {
                    "volatility": 0.0,
                    "trend": 0.0,
                    "count": 0,
                }
                continue
            state_data = regime_feats.iloc[mask]
            state_stats[s] = {
                "volatility": float(state_data["regime_volatility"].mean()),
                "trend": float(state_data["regime_trend"].mean()),
                "count": int(mask.sum()),
            }

        # Sort states by volatility
        sorted_by_vol = sorted(
            state_stats.items(), key=lambda x: x[1]["volatility"]
        )

        # Assignment logic:
        # Lowest volatility -> low_vol_ranging
        # Highest volatility -> high_vol_chaotic
        # Remaining two: positive trend -> trending_up, negative -> trending_down
        assigned = {}
        used_labels = set()

        # Lowest vol state
        low_vol_state = sorted_by_vol[0][0]
        assigned[low_vol_state] = "low_vol_ranging"
        used_labels.add("low_vol_ranging")

        # Highest vol state
        high_vol_state = sorted_by_vol[-1][0]
        assigned[high_vol_state] = "high_vol_chaotic"
        used_labels.add("high_vol_chaotic")

        # Middle states by trend
        middle_states = [
            s for s, _ in sorted_by_vol[1:-1]
        ] if len(sorted_by_vol) > 2 else []

        middle_states_sorted = sorted(
            middle_states, key=lambda s: state_stats[s]["trend"], reverse=True
        )

        if len(middle_states_sorted) >= 1:
            assigned[middle_states_sorted[0]] = "trending_up"
        if len(middle_states_sorted) >= 2:
            assigned[middle_states_sorted[1]] = "trending_down"

        # Handle edge cases: ensure all 4 labels assigned
        all_labels = set(REGIME_LABELS.values())
        remaining_labels = all_labels - set(assigned.values())
        remaining_states = set(range(self.n_regimes)) - set(assigned.keys())

        for state, label in zip(remaining_states, remaining_labels):
            assigned[state] = label

        self.state_to_regime = assigned
        logger.debug(f"Regime assignments: {assigned}")
        for s, label in assigned.items():
            stats = state_stats.get(s, {})
            logger.debug(
                f"  State {s} -> {label}: "
                f"vol={stats.get('volatility', 0):.6f}, "
                f"trend={stats.get('trend', 0):.8f}, "
                f"count={stats.get('count', 0)}"
            )

    def _setup_default_regimes(self):
        """Set up default regime assignments when training data is insufficient."""
        self.state_to_regime = dict(REGIME_LABELS)
        # Create a minimal HMM for prediction
        self.hmm = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=1,
            random_state=42,
        )
        self.scaler = StandardScaler()
        # Fit with dummy data so predict() works
        dummy = np.random.randn(100, 5)
        self.scaler.fit(dummy)
        self.hmm.fit(dummy)
        self._is_trained = True
        logger.info("Using default regime assignments (insufficient training data).")
