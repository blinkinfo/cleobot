"""Orderbook feature calculations for CleoBot.

Implements all ~25 orderbook features from Section 5.2 of the master plan:
- Bid-Ask Imbalance (top 5, 10, 20 levels + temporal change)
- Orderbook Shape (bid/ask slope, slope ratio)
- Large Wall Detection (walls within 0.1% of mid)
- Spread Dynamics (current, vs rolling avg, percentile)
- Pressure (net pressure, change, momentum)

Highest alpha feature group per the master plan.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from src.utils.logger import get_logger

logger = get_logger("features.orderbook")


def _parse_levels(levels: List) -> tuple:
    """Parse a list of [price, qty] levels into price/qty arrays."""
    if not levels:
        return np.array([]), np.array([])
    arr = np.array(levels, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.array([]), np.array([])
    return arr[:, 0], arr[:, 1]


def _imbalance(bid_vol: float, ask_vol: float) -> float:
    """Compute (bid - ask) / (bid + ask) imbalance, clipped to [-1, 1]."""
    total = bid_vol + ask_vol
    if total <= 0:
        return 0.0
    return (bid_vol - ask_vol) / total


def _cum_vol_at_levels(prices: np.ndarray, qtys: np.ndarray, n: int) -> float:
    """Sum quantity for the first N levels."""
    if len(qtys) == 0:
        return 0.0
    return float(np.sum(qtys[:n]))


def _orderbook_slope(prices: np.ndarray, qtys: np.ndarray) -> float:
    """Linear regression slope of cumulative volume vs price distance.

    Bids: prices descending from mid, so we work with absolute distance.
    Asks: prices ascending from mid.
    Returns slope coefficient.
    """
    if len(prices) < 3:
        return 0.0
    cum_vol = np.cumsum(qtys)
    try:
        slope = np.polyfit(np.arange(len(prices)), cum_vol, 1)[0]
        return float(slope)
    except Exception:
        return 0.0


def compute_snapshot_features(snapshot: Dict[str, Any]) -> Dict[str, float]:
    """Compute orderbook features from a single snapshot.

    Args:
        snapshot: Dict with keys 'bids', 'asks', 'mid_price', 'spread'.
                  bids/asks are lists of [price, qty].

    Returns:
        Dict of feature_name -> float value.
    """
    feats: Dict[str, float] = {}

    bids = snapshot.get("bids", [])
    asks = snapshot.get("asks", [])
    mid_price = float(snapshot.get("mid_price", 0.0))
    spread = float(snapshot.get("spread", 0.0))

    bid_prices, bid_qtys = _parse_levels(bids)
    ask_prices, ask_qtys = _parse_levels(asks)

    # ------------------------------------------------------------------ #
    # BID-ASK IMBALANCE
    # ------------------------------------------------------------------ #
    bid5 = _cum_vol_at_levels(bid_prices, bid_qtys, 5)
    ask5 = _cum_vol_at_levels(ask_prices, ask_qtys, 5)
    bid10 = _cum_vol_at_levels(bid_prices, bid_qtys, 10)
    ask10 = _cum_vol_at_levels(ask_prices, ask_qtys, 10)
    bid20 = _cum_vol_at_levels(bid_prices, bid_qtys, 20)
    ask20 = _cum_vol_at_levels(ask_prices, ask_qtys, 20)

    feats["ob_imbalance_5"] = _imbalance(bid5, ask5)
    feats["ob_imbalance_10"] = _imbalance(bid10, ask10)
    feats["ob_imbalance_20"] = _imbalance(bid20, ask20)

    # ------------------------------------------------------------------ #
    # ORDERBOOK SHAPE (slope of cumulative volume vs level index)
    # ------------------------------------------------------------------ #
    bid_slope = _orderbook_slope(bid_prices, bid_qtys)
    ask_slope = _orderbook_slope(ask_prices, ask_qtys)
    feats["ob_bid_slope"] = bid_slope
    feats["ob_ask_slope"] = ask_slope
    total_slope = bid_slope + ask_slope
    feats["ob_slope_ratio"] = (
        (bid_slope - ask_slope) / total_slope if total_slope != 0 else 0.0
    )

    # ------------------------------------------------------------------ #
    # LARGE WALL DETECTION (within 0.1% of mid price)
    # ------------------------------------------------------------------ #
    wall_threshold = mid_price * 0.001 if mid_price > 0 else 0.0

    if len(bid_prices) > 0 and mid_price > 0:
        bid_mask = (mid_price - bid_prices) <= wall_threshold
        bid_wall_qtys = bid_qtys[bid_mask] if bid_mask.any() else np.array([0.0])
        feats["ob_bid_wall"] = float(bid_wall_qtys.max()) if len(bid_wall_qtys) > 0 else 0.0
    else:
        feats["ob_bid_wall"] = 0.0

    if len(ask_prices) > 0 and mid_price > 0:
        ask_mask = (ask_prices - mid_price) <= wall_threshold
        ask_wall_qtys = ask_qtys[ask_mask] if ask_mask.any() else np.array([0.0])
        feats["ob_ask_wall"] = float(ask_wall_qtys.max()) if len(ask_wall_qtys) > 0 else 0.0
    else:
        feats["ob_ask_wall"] = 0.0

    feats["ob_wall_imbalance"] = _imbalance(
        feats["ob_bid_wall"], feats["ob_ask_wall"]
    )

    # ------------------------------------------------------------------ #
    # SPREAD DYNAMICS (raw values; rolling averages computed in engine)
    # ------------------------------------------------------------------ #
    # Spread in basis points
    spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0.0
    feats["ob_spread_bps"] = spread_bps

    # ------------------------------------------------------------------ #
    # PRESSURE (net volume within 0.5% of mid)
    # ------------------------------------------------------------------ #
    pressure_threshold = mid_price * 0.005 if mid_price > 0 else 0.0

    if len(bid_prices) > 0 and mid_price > 0:
        bid_pressure_mask = (mid_price - bid_prices) <= pressure_threshold
        bid_pressure = float(bid_qtys[bid_pressure_mask].sum()) if bid_pressure_mask.any() else 0.0
    else:
        bid_pressure = 0.0

    if len(ask_prices) > 0 and mid_price > 0:
        ask_pressure_mask = (ask_prices - mid_price) <= pressure_threshold
        ask_pressure = float(ask_qtys[ask_pressure_mask].sum()) if ask_pressure_mask.any() else 0.0
    else:
        ask_pressure = 0.0

    feats["ob_net_pressure"] = _imbalance(bid_pressure, ask_pressure)

    return feats


def compute_orderbook_features(
    snapshots: List[Dict[str, Any]],
    current_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Compute all orderbook features including temporal changes.

    Args:
        snapshots: List of historical orderbook snapshots (ascending timestamp).
                   Used for rolling averages and temporal changes.
        current_snapshot: Most recent orderbook snapshot. If None, uses
                          the last entry in snapshots.

    Returns:
        Dict of feature_name -> float value for the current moment.
    """
    feats: Dict[str, float] = {}

    if not snapshots and current_snapshot is None:
        # No data at all -- return zero features
        _zero_keys = [
            "ob_imbalance_5", "ob_imbalance_10", "ob_imbalance_20",
            "ob_imbalance_change_30s", "ob_imbalance_change_60s", "ob_imbalance_change_90s",
            "ob_bid_slope", "ob_ask_slope", "ob_slope_ratio",
            "ob_bid_wall", "ob_ask_wall", "ob_wall_imbalance",
            "ob_spread_bps", "ob_spread_vs_avg", "ob_spread_pctile",
            "ob_net_pressure", "ob_pressure_change_30s", "ob_pressure_change_60s",
            "ob_pressure_change_90s", "ob_pressure_momentum",
        ]
        return {k: 0.0 for k in _zero_keys}

    snap = current_snapshot if current_snapshot is not None else snapshots[-1]
    current_ts = snap.get("timestamp", 0)

    # Compute features for current snapshot
    current_feats = compute_snapshot_features(snap)
    feats.update(current_feats)

    # ------------------------------------------------------------------ #
    # TEMPORAL CHANGES IN IMBALANCE AND PRESSURE
    # ------------------------------------------------------------------ #
    # Build a time-indexed lookup for past snapshots
    # We need snapshots ~30s, ~60s, ~90s ago
    def _find_snap_at_offset(offset_ms: int) -> Optional[Dict[str, Any]]:
        """Find the snapshot closest to current_ts - offset_ms."""
        target_ts = current_ts - offset_ms
        best = None
        best_diff = float("inf")
        for s in snapshots:
            diff = abs(s.get("timestamp", 0) - target_ts)
            if diff < best_diff:
                best_diff = diff
                best = s
        # Only use if within 15 seconds of target
        if best is not None and best_diff <= 15000:
            return best
        return None

    for offset_s, key_suffix in [(30, "30s"), (60, "60s"), (90, "90s")]:
        past_snap = _find_snap_at_offset(offset_s * 1000)
        if past_snap is not None:
            past_feats = compute_snapshot_features(past_snap)
            feats[f"ob_imbalance_change_{key_suffix}"] = (
                current_feats["ob_imbalance_5"] - past_feats["ob_imbalance_5"]
            )
            feats[f"ob_pressure_change_{key_suffix}"] = (
                current_feats["ob_net_pressure"] - past_feats["ob_net_pressure"]
            )
        else:
            feats[f"ob_imbalance_change_{key_suffix}"] = 0.0
            feats[f"ob_pressure_change_{key_suffix}"] = 0.0

    # Pressure momentum: acceleration = change_30s - change_60s
    feats["ob_pressure_momentum"] = (
        feats["ob_pressure_change_30s"] - feats["ob_pressure_change_60s"]
    )

    # ------------------------------------------------------------------ #
    # SPREAD DYNAMICS: rolling average and percentile
    # ------------------------------------------------------------------ #
    if len(snapshots) >= 2:
        spread_series = [
            compute_snapshot_features(s)["ob_spread_bps"] for s in snapshots
        ]
        spread_arr = np.array(spread_series)
        feats["ob_spread_vs_avg"] = (
            feats["ob_spread_bps"] - float(np.mean(spread_arr[-60:]))
        )  # vs last ~5 min (60 snapshots @ 5s each)
        # Percentile vs last 1 hour (720 snapshots)
        window = spread_arr[-720:]
        if len(window) > 1:
            feats["ob_spread_pctile"] = float(
                np.searchsorted(np.sort(window), feats["ob_spread_bps"]) / len(window)
            )
        else:
            feats["ob_spread_pctile"] = 0.5
    else:
        feats["ob_spread_vs_avg"] = 0.0
        feats["ob_spread_pctile"] = 0.5

    logger.debug(f"Orderbook features computed: {len(feats)} features")
    return feats
