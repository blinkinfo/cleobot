"""Comprehensive Phase 2 feature engineering tests for CleoBot.

Tests all feature modules: candle, orderbook, funding, cross_tf, time,
polymarket, derived, and the FeatureEngine integration.
"""

import os
import sys
import time
import math
import json
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Helper: generate synthetic OHLCV DataFrames
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int, base_price: float = 67000.0, interval_ms: int = 300_000,
                start_ts: int = None) -> pd.DataFrame:
    """Generate n rows of realistic OHLCV data."""
    rng = np.random.default_rng(42)
    if start_ts is None:
        start_ts = int(time.time() * 1000) - n * interval_ms
    timestamps = [start_ts + i * interval_ms for i in range(n)]
    closes = [base_price]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + rng.normal(0, 0.002)))
    closes = np.array(closes)
    opens = np.roll(closes, 1)
    opens[0] = closes[0] * 0.9995
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.003, n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.003, n))
    volumes = rng.uniform(1.0, 50.0, n)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def _make_ob_snapshot(mid_price: float = 67000.0, n_levels: int = 20,
                      spread: float = 5.0, timestamp: int = None,
                      bid_heavier: bool = True) -> Dict[str, Any]:
    """Generate a synthetic orderbook snapshot."""
    rng = np.random.default_rng(99)
    if timestamp is None:
        timestamp = int(time.time() * 1000)
    half_spread = spread / 2.0
    best_bid = mid_price - half_spread
    best_ask = mid_price + half_spread
    # Bids: descending prices
    bid_prices = [best_bid - i * 0.5 for i in range(n_levels)]
    bid_qtys = rng.uniform(0.1, 2.0, n_levels).tolist()
    if bid_heavier:
        bid_qtys[0] *= 3  # make bids heavier near top
    # Asks: ascending prices
    ask_prices = [best_ask + i * 0.5 for i in range(n_levels)]
    ask_qtys = rng.uniform(0.1, 2.0, n_levels).tolist()
    return {
        "timestamp": timestamp,
        "mid_price": mid_price,
        "spread": spread,
        "bids": [[p, q] for p, q in zip(bid_prices, bid_qtys)],
        "asks": [[p, q] for p, q in zip(ask_prices, ask_qtys)],
    }


def _all_finite(d: Dict) -> bool:
    """Return True if all values in dict are finite floats."""
    for k, v in d.items():
        try:
            fv = float(v)
            if not math.isfinite(fv):
                print(f"  NON-FINITE: {k} = {v}")
                return False
        except (TypeError, ValueError):
            print(f"  NON-FLOAT: {k} = {v} (type={type(v)})")
            return False
    return True


# ===========================================================================
# TEST A: Candle Features
# ===========================================================================

def test_a_candle_features():
    from src.features.candle_features import compute_candle_features

    df = _make_ohlcv(150)
    result = compute_candle_features(df)

    # result is dict of name -> pd.Series; convert last value to scalar for checks
    assert isinstance(result, dict), "Should return a dict"
    assert len(result) >= 30, f"Expected >= 30 features, got {len(result)}"

    # Check that all last values are finite
    bad = []
    for k, v in result.items():
        last = float(v.iloc[-1])
        if not math.isfinite(last):
            bad.append((k, last))
    assert not bad, f"Non-finite values: {bad}"

    print(f"  Feature names ({len(result)}): {sorted(result.keys())}")
    return len(result)


# ===========================================================================
# TEST B: Orderbook Features
# ===========================================================================

def test_b_orderbook_features():
    from src.features.orderbook_features import compute_orderbook_features

    base_ts = int(time.time() * 1000) - 120_000
    snap1 = _make_ob_snapshot(67000.0, timestamp=base_ts)
    snap2 = _make_ob_snapshot(67010.0, timestamp=base_ts + 30_000, bid_heavier=False)
    snap3 = _make_ob_snapshot(67005.0, timestamp=base_ts + 60_000)

    result = compute_orderbook_features(
        snapshots=[snap1, snap2, snap3],
        current_snapshot=snap3,
    )

    assert isinstance(result, dict), "Should return a dict"
    assert len(result) >= 15, f"Expected >= 15 features, got {len(result)}"
    assert _all_finite(result), "All values should be finite"
    assert "ob_imbalance_5" in result, "ob_imbalance_5 should be present"
    assert -1.0 <= result["ob_imbalance_5"] <= 1.0, (
        f"ob_imbalance_5={result['ob_imbalance_5']} out of range [-1, 1]"
    )

    return len(result)


# ===========================================================================
# TEST C: Funding Rate Features
# ===========================================================================

def test_c_funding_features():
    from src.features.funding_features import compute_funding_features

    current_ts_ms = int(time.time() * 1000)
    interval_ms = 8 * 3600 * 1000  # 8 hours
    records = []
    for i in range(25):
        ts = current_ts_ms - (25 - i) * interval_ms
        rate = 0.0001 + np.random.default_rng(i).normal(0, 0.00005)
        records.append({"timestamp": ts, "rate": float(rate)})

    result = compute_funding_features(records, current_ts_ms)

    assert isinstance(result, dict), "Should return a dict"
    assert len(result) == 8, f"Expected exactly 8 features, got {len(result)}: {list(result.keys())}"
    assert _all_finite(result), "All values should be finite"
    assert result["funding_direction"] in {-1.0, 0.0, 1.0}, (
        f"funding_direction={result['funding_direction']} not in {{-1, 0, 1}}"
    )

    return 8


# ===========================================================================
# TEST D: Cross-TF Features
# ===========================================================================

def test_d_cross_tf_features():
    from src.features.cross_tf_features import compute_cross_tf_features

    now_ts = int(time.time() * 1000)
    df_5m  = _make_ohlcv(150, interval_ms=300_000,   start_ts=now_ts - 150 * 300_000)
    df_15m = _make_ohlcv(100, interval_ms=900_000,   start_ts=now_ts - 100 * 900_000)
    df_1h  = _make_ohlcv(72,  interval_ms=3_600_000, start_ts=now_ts - 72  * 3_600_000)

    result = compute_cross_tf_features(df_5m, df_15m, df_1h)

    assert isinstance(result, dict), "Should return a dict"
    assert len(result) == 10, f"Expected exactly 10 features, got {len(result)}: {list(result.keys())}"
    assert _all_finite(result), "All values should be finite"
    assert result["tf_alignment_score"] in {0.0, 1.0, 2.0}, (
        f"tf_alignment_score={result['tf_alignment_score']} not in {{0, 1, 2}}"
    )

    return 10


# ===========================================================================
# TEST E: Time Features
# ===========================================================================

def test_e_time_features():
    from src.features.time_features import compute_time_features

    now_ts = int(time.time() * 1000)
    df_5m = _make_ohlcv(150)

    result = compute_time_features(current_ts_ms=now_ts, df_5m=df_5m)

    assert isinstance(result, dict), "Should return a dict"
    assert len(result) == 8, f"Expected exactly 8 features, got {len(result)}: {list(result.keys())}"
    assert _all_finite(result), "All values should be finite"
    assert -1.0 <= result["time_hour_sin"] <= 1.0, (
        f"time_hour_sin={result['time_hour_sin']} out of [-1, 1]"
    )
    assert -1.0 <= result["time_hour_cos"] <= 1.0, (
        f"time_hour_cos={result['time_hour_cos']} out of [-1, 1]"
    )
    assert 0.0 <= result["time_to_funding"] <= 1.0, (
        f"time_to_funding={result['time_to_funding']} out of [0, 1]"
    )

    return 8


# ===========================================================================
# TEST F: Polymarket Features
# ===========================================================================

def test_f_polymarket_features():
    from src.features.polymarket_features import compute_polymarket_features

    # Case 1: None data -> defaults
    result_none = compute_polymarket_features(market_data=None)
    assert isinstance(result_none, dict), "Should return a dict"
    assert len(result_none) == 6, f"Expected 6 defaults, got {len(result_none)}"
    assert _all_finite(result_none), "Default values should be finite"

    # Case 2: Sample market data
    market_data = {
        "timestamp": int(time.time() * 1000),
        "up_odds": 0.62,
        "down_odds": 0.38,
        "yes_volume": 1000.0,
        "no_volume": 800.0,
        "total_volume_5m": 250.0,
    }
    result_data = compute_polymarket_features(market_data=market_data)
    assert isinstance(result_data, dict), "Should return a dict"
    assert len(result_data) == 6, f"Expected 6 features, got {len(result_data)}"
    assert _all_finite(result_data), "All values should be finite"

    # Case 3: Bad data -> graceful fallback
    bad_data = {"up_odds": "NaN", "down_odds": None}
    result_bad = compute_polymarket_features(market_data=bad_data)
    assert isinstance(result_bad, dict), "Should return a dict on bad data"
    assert len(result_bad) == 6, f"Expected 6 fallback features, got {len(result_bad)}"

    return 6


# ===========================================================================
# TEST G: Derived Features
# ===========================================================================

def test_g_derived_features():
    from src.features.candle_features import compute_candle_features
    from src.features.orderbook_features import compute_orderbook_features
    from src.features.funding_features import compute_funding_features
    from src.features.cross_tf_features import compute_cross_tf_features
    from src.features.time_features import compute_time_features
    from src.features.polymarket_features import compute_polymarket_features
    from src.features.derived_features import compute_derived_features

    now_ts = int(time.time() * 1000)
    df_5m  = _make_ohlcv(150)
    df_15m = _make_ohlcv(100, interval_ms=900_000)
    df_1h  = _make_ohlcv(72,  interval_ms=3_600_000)

    # Compute all component features
    candle_raw = compute_candle_features(df_5m)
    candle_scalar = {k: float(v.iloc[-1]) for k, v in candle_raw.items()}

    snap = _make_ob_snapshot(67000.0)
    ob_feats = compute_orderbook_features(snapshots=[snap], current_snapshot=snap)

    current_ts = int(time.time() * 1000)
    interval_ms = 8 * 3600 * 1000
    funding_records = [
        {"timestamp": current_ts - (10 - i) * interval_ms, "rate": 0.0001}
        for i in range(10)
    ]
    funding_feats = compute_funding_features(funding_records, current_ts)
    cross_feats = compute_cross_tf_features(df_5m, df_15m, df_1h)
    time_feats = compute_time_features(current_ts_ms=now_ts, df_5m=df_5m)
    pm_feats = compute_polymarket_features(market_data=None)

    # Combine all features
    combined = {}
    combined.update(candle_scalar)
    combined.update(ob_feats)
    combined.update(funding_feats)
    combined.update(cross_feats)
    combined.update(time_feats)
    combined.update(pm_feats)

    # Compute derived features
    result = compute_derived_features(features=combined, feature_history={})

    assert isinstance(result, dict), "Should return a dict"
    assert len(result) >= 17, f"Expected >= 17 features, got {len(result)}: {list(result.keys())}"
    assert _all_finite(result), "All values should be finite"

    return len(result)


# ===========================================================================
# TEST H: Feature Engine Integration
# ===========================================================================

def test_h_feature_engine():
    from src.database import Database
    from src.features.engine import FeatureEngine

    db_path = "/tmp/test_cleobot.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    db = Database(db_path)

    now_ts = int(time.time() * 1000)

    # --- Insert 150 5m candles ---
    interval_5m = 300_000
    df_5m = _make_ohlcv(150, interval_ms=interval_5m,
                         start_ts=now_ts - 150 * interval_5m)
    candles_5m = [
        (int(row.timestamp), row.open, row.high, row.low, row.close, row.volume)
        for row in df_5m.itertuples()
    ]
    db.insert_candles_batch("candles_5m", candles_5m)

    # --- Insert 100 15m candles ---
    interval_15m = 900_000
    df_15m = _make_ohlcv(100, interval_ms=interval_15m,
                          start_ts=now_ts - 100 * interval_15m)
    candles_15m = [
        (int(row.timestamp), row.open, row.high, row.low, row.close, row.volume)
        for row in df_15m.itertuples()
    ]
    db.insert_candles_batch("candles_15m", candles_15m)

    # --- Insert 72 1h candles ---
    interval_1h = 3_600_000
    df_1h = _make_ohlcv(72, interval_ms=interval_1h,
                         start_ts=now_ts - 72 * interval_1h)
    candles_1h = [
        (int(row.timestamp), row.open, row.high, row.low, row.close, row.volume)
        for row in df_1h.itertuples()
    ]
    db.insert_candles_batch("candles_1h", candles_1h)

    # --- Insert 20 orderbook snapshots ---
    for i in range(20):
        ts = now_ts - (20 - i) * 30_000  # 30s apart
        snap = _make_ob_snapshot(67000.0 + i * 5, timestamp=ts)
        db.insert_orderbook_snapshot(
            timestamp=ts,
            bids=snap["bids"],
            asks=snap["asks"],
            mid_price=snap["mid_price"],
            spread=snap["spread"],
        )

    # --- Insert 10 funding rate records ---
    interval_fund = 8 * 3600 * 1000
    for i in range(10):
        ts = now_ts - (10 - i) * interval_fund
        db.insert_funding_rate(
            timestamp=ts,
            rate=0.0001 + np.random.default_rng(i).normal(0, 0.00002),
        )

    # --- Run FeatureEngine ---
    engine = FeatureEngine(db)

    t0 = time.monotonic()
    result = engine.compute(current_ts_ms=now_ts)
    elapsed = time.monotonic() - t0

    assert isinstance(result, dict), "Should return a dict"
    n = len(result)
    assert 80 <= n <= 130, f"Expected 80-130 features, got {n}"

    bad = [(k, v) for k, v in result.items() if not math.isfinite(float(v))]
    assert not bad, f"Non-finite features: {bad}"

    assert elapsed < 33.0, f"Computation took {elapsed:.2f}s, exceeding 33s limit"

    print(f"  Engine feature count: {n}, elapsed: {elapsed:.3f}s")

    # Cleanup
    os.remove(db_path)

    return n, elapsed


# ===========================================================================
# TEST I: NaN Handling
# ===========================================================================

def test_i_nan_handling():
    from src.features.candle_features import compute_candle_features
    from src.features.engine import FeatureEngine

    # Create candles with some NaN/zero values in the middle
    df = _make_ohlcv(150)
    # Inject NaN and zeros in the middle
    df.loc[50:60, "close"] = np.nan
    df.loc[70:75, "volume"] = 0.0
    df.loc[80, "high"] = np.nan
    # Forward-fill NaN closes so OHLCV is still somewhat valid
    df["close"] = df["close"].ffill().bfill()
    df["high"] = df[["high", "close"]].max(axis=1)
    df["low"] = df[["low", "close"]].min(axis=1)

    result = compute_candle_features(df)
    assert isinstance(result, dict), "Should return a dict"

    # All last values should be finite
    bad = []
    for k, v in result.items():
        last = float(v.iloc[-1])
        if not math.isfinite(last):
            bad.append((k, last))
    assert not bad, f"NaN not handled in candle features: {bad}"

    # Test engine._validate_features strips NaN directly
    import sqlite3 as _sq
    # Create a minimal engine to call _validate_features
    class _FakeDB:
        pass
    engine = FeatureEngine.__new__(FeatureEngine)
    engine._feature_history = {}
    engine._pm_snapshots = []
    engine._latest_pm_data = None
    engine._latest_pm_model_pred = None
    engine._last_compute_time_s = 0.0
    engine._compute_count = 0
    engine.db = _FakeDB()

    dirty = {
        "feat_nan": float("nan"),
        "feat_inf": float("inf"),
        "feat_neginf": float("-inf"),
        "feat_ok": 1.23,
    }
    cleaned = engine._validate_features(dirty)
    assert cleaned["feat_nan"] == 0.0, "NaN should become 0.0"
    assert cleaned["feat_inf"] == 0.0, "inf should become 0.0"
    assert cleaned["feat_neginf"] == 0.0, "-inf should become 0.0"
    assert cleaned["feat_ok"] == 1.23, "Valid value should remain unchanged"


# ===========================================================================
# MAIN: run all tests and print summary
# ===========================================================================

def main():
    results = {}
    errors = {}

    def run_test(name, fn):
        try:
            r = fn()
            results[name] = r
            print(f"  {name}: OK")
        except Exception as e:
            import traceback
            errors[name] = str(e)
            print(f"  {name}: FAILED -- {e}")
            traceback.print_exc()

    print("Running Test A (Candle Features)...")
    run_test("A", test_a_candle_features)

    print("Running Test B (Orderbook Features)...")
    run_test("B", test_b_orderbook_features)

    print("Running Test C (Funding Rate Features)...")
    run_test("C", test_c_funding_features)

    print("Running Test D (Cross-TF Features)...")
    run_test("D", test_d_cross_tf_features)

    print("Running Test E (Time Features)...")
    run_test("E", test_e_time_features)

    print("Running Test F (Polymarket Features)...")
    run_test("F", test_f_polymarket_features)

    print("Running Test G (Derived Features)...")
    run_test("G", test_g_derived_features)

    print("Running Test H (Feature Engine Integration)...")
    run_test("H", test_h_feature_engine)

    print("Running Test I (NaN Handling)...")
    run_test("I", test_i_nan_handling)

    # Summary
    print()
    print("=== PHASE 2 TEST RESULTS ===")

    def _fmt(key, label, unit="features"):
        if key in errors:
            return f"Test {key} ({label}): FAIL -- {errors[key]}"
        r = results.get(key)
        if isinstance(r, tuple):
            n, elapsed = r
            return f"Test {key} ({label}): PASS - {n} total features in {elapsed:.2f}s"
        elif r is not None:
            return f"Test {key} ({label}): PASS - {r} {unit}"
        else:
            return f"Test {key} ({label}): PASS"

    print(_fmt("A", "Candle"))
    print(_fmt("B", "Orderbook"))
    print(_fmt("C", "Funding"))
    print(_fmt("D", "Cross-TF"))
    print(_fmt("E", "Time"))
    print(_fmt("F", "Polymarket"))
    print(_fmt("G", "Derived"))
    print(_fmt("H", "Engine"))
    print(_fmt("I", "NaN handling"))

    if errors:
        print(f"\n=== {len(errors)} TEST(S) FAILED ===")
        for k, e in errors.items():
            print(f"  Test {k}: {e}")
        sys.exit(1)
    else:
        print("=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
