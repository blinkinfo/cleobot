"""Polymarket-specific feature calculations for CleoBot.

Implements all 6 Polymarket features from Section 5.6 of the master plan:
1. Current UP odds
2. Current DOWN odds
3. Odds velocity (change in odds over last 60s)
4. Yes volume vs No volume ratio
5. Total market volume (last 5 min)
6. Odds divergence from model prediction

All features have graceful fallback to neutral values when the
Polymarket API is unavailable or returns no data.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from src.utils.logger import get_logger

logger = get_logger("features.polymarket")

# Neutral/fallback values (50/50 market with no volume signal)
_DEFAULTS: Dict[str, float] = {
    "pm_up_odds": 0.5,
    "pm_down_odds": 0.5,
    "pm_odds_velocity": 0.0,
    "pm_volume_ratio": 1.0,   # equal yes/no volume
    "pm_total_volume": 0.0,
    "pm_odds_divergence": 0.0,
}


def compute_polymarket_features(
    market_data: Optional[Dict[str, Any]] = None,
    historical_snapshots: Optional[List[Dict[str, Any]]] = None,
    model_prediction: Optional[float] = None,
) -> Dict[str, float]:
    """Compute Polymarket features for the current cycle.

    Args:
        market_data: Dict with current market state. Expected keys:
            - 'up_odds' (float): Implied probability of UP outcome (0-1).
            - 'down_odds' (float): Implied probability of DOWN outcome (0-1).
            - 'yes_volume' (float): Volume traded on YES side.
            - 'no_volume' (float): Volume traded on NO side.
            - 'total_volume_5m' (float): Total market volume in last 5 minutes.
        historical_snapshots: List of past market_data dicts with 'timestamp' (ms)
            and 'up_odds' fields for velocity calculation.
        model_prediction: Model's UP probability (0-1) for divergence feature.
                          If None, divergence feature returns 0.

    Returns:
        Dict of feature_name -> float. Returns defaults on any failure.
    """
    if market_data is None:
        logger.debug("No Polymarket data available, using defaults.")
        return _DEFAULTS.copy()

    try:
        feats: Dict[str, float] = {}

        # ------------------------------------------------------------------ #
        # 1. Current UP odds
        # ------------------------------------------------------------------ #
        up_odds = float(market_data.get("up_odds", 0.5))
        up_odds = float(np.clip(up_odds, 0.01, 0.99))
        feats["pm_up_odds"] = up_odds

        # ------------------------------------------------------------------ #
        # 2. Current DOWN odds
        # ------------------------------------------------------------------ #
        down_odds = float(market_data.get("down_odds", 0.5))
        down_odds = float(np.clip(down_odds, 0.01, 0.99))
        feats["pm_down_odds"] = down_odds

        # ------------------------------------------------------------------ #
        # 3. Odds velocity (change in UP odds over last 60 seconds)
        # ------------------------------------------------------------------ #
        if historical_snapshots and len(historical_snapshots) >= 2:
            # Find snapshot closest to 60 seconds ago
            current_ts = market_data.get("timestamp", 0)
            target_ts = current_ts - 60_000
            best_snap = min(
                historical_snapshots,
                key=lambda s: abs(s.get("timestamp", 0) - target_ts),
            )
            past_up_odds = float(best_snap.get("up_odds", up_odds))
            feats["pm_odds_velocity"] = float(up_odds - past_up_odds)
        else:
            feats["pm_odds_velocity"] = 0.0

        # ------------------------------------------------------------------ #
        # 4. Yes volume vs No volume ratio
        # Encoded as imbalance: (yes - no) / (yes + no)
        # ------------------------------------------------------------------ #
        yes_vol = float(market_data.get("yes_volume", 0.0))
        no_vol = float(market_data.get("no_volume", 0.0))
        total_vol = yes_vol + no_vol
        if total_vol > 0:
            feats["pm_volume_ratio"] = float((yes_vol - no_vol) / total_vol)
        else:
            feats["pm_volume_ratio"] = 0.0

        # ------------------------------------------------------------------ #
        # 5. Total market volume (last 5 min)
        # ------------------------------------------------------------------ #
        feats["pm_total_volume"] = float(market_data.get("total_volume_5m", 0.0))

        # ------------------------------------------------------------------ #
        # 6. Odds divergence from model prediction
        # Positive = model more bullish than market
        # Negative = model more bearish than market
        # ------------------------------------------------------------------ #
        if model_prediction is not None:
            feats["pm_odds_divergence"] = float(
                float(np.clip(model_prediction, 0.0, 1.0)) - up_odds
            )
        else:
            feats["pm_odds_divergence"] = 0.0

        logger.debug(f"Polymarket features computed: {len(feats)} features")
        return feats

    except Exception as e:
        logger.warning(f"Polymarket feature computation failed: {e}. Using defaults.")
        return _DEFAULTS.copy()
