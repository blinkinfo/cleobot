"""Backtesting engine for CleoBot.

Simulates the full trading pipeline on historical data:
  - Walk-forward backtesting on stored 5m candles
  - Applies all 6 signal filters exactly as live trading does
  - Tracks: accuracy, P&L, drawdown, trade count, filter impact
  - Supports configurable date ranges (days back)
  - Returns structured BacktestResult for report generation

The engine does NOT require a trained ML ensemble. It includes a
lightweight heuristic signal generator (RSI + EMA crossover + momentum)
for backtests when no models are available. When the live Ensemble IS
ready, it can be injected via the ensemble parameter.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("backtest.engine")

WIN_PNL = 0.88
LOSS_PNL = -1.00
BREAKEVEN_WIN_RATE = 1.00 / (1.00 + WIN_PNL)
DEFAULT_CONFIDENCE_THRESHOLD = 0.58
MIN_WARMUP_CANDLES = 60


@dataclass
class TradeRecord:
    candle_index: int
    timestamp: int
    direction: str
    confidence: float
    regime: str
    won: bool
    pnl: float
    filters_applied: Dict[str, bool]
    skip_reason: str = ""
    was_skipped: bool = False


@dataclass
class FilterImpact:
    filter_name: str
    trades_blocked: int
    accuracy_with: float
    accuracy_without: float
    pnl_with: float
    pnl_without: float


@dataclass
class HourlyStats:
    hour: int
    trades: int
    wins: int
    accuracy: float
    pnl: float


@dataclass
class BacktestResult:
    days: int
    start_ts: int
    end_ts: int
    total_candles: int
    total_trades: int
    wins: int
    losses: int
    skips: int
    accuracy: float
    pnl: float
    max_drawdown: float
    max_consecutive_losses: int
    max_consecutive_wins: int
    sharpe_ratio: float
    profit_factor: float
    trades: List[TradeRecord] = field(default_factory=list)
    hourly_stats: List[HourlyStats] = field(default_factory=list)
    filter_impacts: List[FilterImpact] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    duration_seconds: float = 0.0
    model_used: str = "heuristic"


class HeuristicSignalGenerator:
    """Lightweight signal generator (RSI + EMA + momentum) for model-free backtests."""

    def predict(self, df: pd.DataFrame, idx: int) -> Tuple[str, float]:
        window = df.iloc[max(0, idx - 30): idx + 1].copy()
        if len(window) < 14:
            return "UP", 0.50
        closes = window["close"].values.astype(float)
        volumes = window["volume"].values.astype(float)
        rsi = self._rsi(closes, 14)
        ema9 = self._ema(closes, 9)
        ema21 = self._ema(closes, 21)
        ema_cross = (ema9 - ema21) / (ema21 + 1e-9)
        if len(closes) >= 4:
            rets = np.diff(closes[-4:]) / (closes[-4:-1] + 1e-9)
            momentum = float(np.mean(rets))
        else:
            momentum = 0.0
        avg_vol = float(np.mean(volumes[-12:])) if len(volumes) >= 12 else float(np.mean(volumes))
        vol_ratio = float(volumes[-1]) / (avg_vol + 1e-9)
        score = 0.0
        if rsi > 55:
            score += (rsi - 55) / 45.0 * 0.3
        elif rsi < 45:
            score -= (45 - rsi) / 45.0 * 0.3
        score += float(np.clip(ema_cross * 10, -0.3, 0.3))
        score += float(np.clip(momentum * 100, -0.2, 0.2))
        if vol_ratio > 1.5:
            score *= 1.1
        direction = "UP" if score >= 0 else "DOWN"
        confidence = 0.50 + min(abs(score) * 0.25, 0.20)
        return direction, confidence

    @staticmethod
    def _rsi(closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = float(np.mean(gains)) + 1e-9
        avg_loss = float(np.mean(losses)) + 1e-9
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _ema(closes: np.ndarray, period: int) -> float:
        if len(closes) < period:
            return float(closes[-1]) if len(closes) > 0 else 0.0
        k = 2.0 / (period + 1)
        ema = float(closes[-period])
        for price in closes[-period + 1:]:
            ema = float(price) * k + ema * (1 - k)
        return ema


def _compute_atr(df: pd.DataFrame, idx: int, period: int = 14) -> float:
    start = max(0, idx - period)
    window = df.iloc[start: idx + 1]
    if len(window) < 2:
        return 0.0
    highs = window["high"].values.astype(float)
    lows = window["low"].values.astype(float)
    closes = window["close"].values.astype(float)
    trs = []
    for i in range(1, len(window)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    return float(np.mean(trs)) if trs else 0.0


def _simulate_filters(
    direction: str,
    confidence: float,
    regime: str,
    agreement: int,
    atr_percentile: float,
    consecutive_losses: int,
    rolling_accuracy: Optional[float],
    n_settled: int,
    pause_cycles_remaining: int,
    streak_manual_restart: bool,
) -> Tuple[str, str, Dict[str, bool]]:
    verdicts: Dict[str, bool] = {}
    first_fail = ""
    if regime == "low_vol_ranging":
        threshold = 0.62
    elif regime in ("trending_up", "trending_down"):
        threshold = 0.56
    elif regime == "high_vol_chaotic":
        threshold = 0.65
    else:
        threshold = DEFAULT_CONFIDENCE_THRESHOLD
    passed_conf = confidence >= threshold
    verdicts["confidence"] = passed_conf
    if not passed_conf and not first_fail:
        first_fail = "Confidence below threshold"
    passed_vol = 10.0 <= atr_percentile <= 95.0
    verdicts["volatility"] = passed_vol
    if not passed_vol and not first_fail:
        first_fail = "ATR outside acceptable range"
    if regime == "high_vol_chaotic":
        passed_regime = agreement >= 3 and confidence >= 0.65
    elif regime == "low_vol_ranging":
        passed_regime = confidence >= 0.62
    else:
        passed_regime = confidence >= threshold
    verdicts["regime"] = passed_regime
    if not passed_regime and not first_fail:
        first_fail = "Regime filter ({})".format(regime)
    passed_agree = agreement >= 2
    verdicts["agreement"] = passed_agree
    if not passed_agree and not first_fail:
        first_fail = "Insufficient model agreement"
    if streak_manual_restart or consecutive_losses >= 7:
        passed_streak = False
        if not first_fail:
            first_fail = "7-loss streak: manual restart required"
    elif pause_cycles_remaining > 0:
        passed_streak = False
        if not first_fail:
            first_fail = "Streak pause ({} cycles remaining)".format(pause_cycles_remaining)
    else:
        passed_streak = True
    verdicts["streak"] = passed_streak
    if rolling_accuracy is None or n_settled < 50:
        passed_corr = True
    elif rolling_accuracy < 0.50:
        passed_corr = False
        if not first_fail:
            first_fail = "Rolling accuracy below 50%"
    else:
        passed_corr = True
    verdicts["correlation"] = passed_corr
    decision = "SKIP" if first_fail else "TRADE"
    return decision, first_fail, verdicts


def _estimate_regime(df: pd.DataFrame, idx: int) -> str:
    window = df.iloc[max(0, idx - 24): idx + 1]
    if len(window) < 12:
        return "low_vol_ranging"
    closes = window["close"].values.astype(float)
    returns = np.diff(closes) / (closes[:-1] + 1e-9)
    volatility = float(np.std(returns)) if len(returns) > 1 else 0.0
    x = np.arange(len(closes), dtype=float)
    if len(closes) > 2:
        slope = float(np.polyfit(x, closes, 1)[0])
        normalised_slope = slope / (float(closes[-1]) + 1e-9)
    else:
        normalised_slope = 0.0
    if volatility > 0.003:
        return "high_vol_chaotic"
    elif normalised_slope > 0.0001:
        return "trending_up"
    elif normalised_slope < -0.0001:
        return "trending_down"
    else:
        return "low_vol_ranging"


class BacktestEngine:
    """Walk-forward backtesting engine for CleoBot.

    Simulates the full 5-minute trading cycle including all 6 signal filters,
    streak management, and P&L tracking. Works with or without trained models.
    """

    def __init__(self, db, ensemble=None):
        self.db = db
        self.ensemble = ensemble
        self._heuristic = HeuristicSignalGenerator()

    def run(
        self,
        days: int = 7,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> BacktestResult:
        t_start = time.monotonic()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        if end_ts is None:
            end_ts = now_ms
        if start_ts is None:
            start_ts = int(
                (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
            )
        logger.info("Starting {}d backtest: {} -> {}".format(days, start_ts, end_ts))
        candles = self.db.get_candles(
            "candles_5m",
            limit=days * 300,
            since=start_ts - MIN_WARMUP_CANDLES * 5 * 60 * 1000,
        )
        if len(candles) < MIN_WARMUP_CANDLES + 2:
            logger.warning("Insufficient candle data: {} candles".format(len(candles)))
            return self._empty_result(days, start_ts, end_ts)
        df = pd.DataFrame(candles)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        df = df.sort_values("timestamp").reset_index(drop=True)
        in_range = df["timestamp"] >= start_ts
        use_ml = (
            self.ensemble is not None
            and hasattr(self.ensemble, "is_ready")
            and self.ensemble.is_ready
        )
        model_used = "ensemble" if use_ml else "heuristic"
        logger.info("Signal source: {}".format(model_used))
        trades: List[TradeRecord] = []
        equity_curve: List[float] = [0.0]
        cumulative_pnl = 0.0
        consecutive_losses = 0
        pause_cycles_remaining = 0
        streak_manual_restart = False
        atr_history: List[float] = []
        outcome_history: List[int] = []
        for idx in range(MIN_WARMUP_CANDLES, len(df) - 1):
            if not in_range.iloc[idx]:
                atr = _compute_atr(df, idx)
                if atr > 0:
                    atr_history.append(atr)
                continue
            row = df.iloc[idx]
            next_row = df.iloc[idx + 1]
            if use_ml:
                direction, confidence, agreement, regime = self._ml_signal(df, idx)
            else:
                direction, confidence = self._heuristic.predict(df, idx)
                agreement = 2
                regime = _estimate_regime(df, idx)
            atr = _compute_atr(df, idx)
            if atr > 0:
                atr_history.append(atr)
            atr_history = atr_history[-300:]
            atr_percentile = self._atr_percentile(atr, atr_history)
            rolling_acc: Optional[float] = None
            n_settled = len(outcome_history)
            if n_settled >= 50:
                rolling_acc = float(sum(outcome_history[-50:])) / 50.0
            if pause_cycles_remaining > 0:
                pause_cycles_remaining -= 1
            decision, skip_reason, filter_verdicts = _simulate_filters(
                direction=direction,
                confidence=confidence,
                regime=regime,
                agreement=agreement,
                atr_percentile=atr_percentile,
                consecutive_losses=consecutive_losses,
                rolling_accuracy=rolling_acc,
                n_settled=n_settled,
                pause_cycles_remaining=pause_cycles_remaining,
                streak_manual_restart=streak_manual_restart,
            )
            if decision == "SKIP":
                trades.append(TradeRecord(
                    candle_index=idx,
                    timestamp=int(row["timestamp"]),
                    direction=direction,
                    confidence=confidence,
                    regime=regime,
                    won=False,
                    pnl=0.0,
                    filters_applied=filter_verdicts,
                    skip_reason=skip_reason,
                    was_skipped=True,
                ))
                continue
            candle_up = float(next_row["close"]) > float(next_row["open"])
            won = (direction == "UP" and candle_up) or (direction == "DOWN" and not candle_up)
            pnl = WIN_PNL if won else LOSS_PNL
            cumulative_pnl += pnl
            equity_curve.append(round(cumulative_pnl, 4))
            if won:
                consecutive_losses = 0
            else:
                consecutive_losses += 1
                if consecutive_losses >= 7:
                    streak_manual_restart = True
                elif consecutive_losses == 5:
                    pause_cycles_remaining = max(pause_cycles_remaining, 3)
                elif consecutive_losses == 3:
                    pause_cycles_remaining = max(pause_cycles_remaining, 1)
            outcome_history.append(1 if won else 0)
            trades.append(TradeRecord(
                candle_index=idx,
                timestamp=int(row["timestamp"]),
                direction=direction,
                confidence=confidence,
                regime=regime,
                won=won,
                pnl=pnl,
                filters_applied=filter_verdicts,
                skip_reason="",
                was_skipped=False,
            ))
        executed = [t for t in trades if not t.was_skipped]
        skipped = [t for t in trades if t.was_skipped]
        n_trades = len(executed)
        n_wins = sum(1 for t in executed if t.won)
        n_losses = n_trades - n_wins
        n_skips = len(skipped)
        accuracy = n_wins / n_trades if n_trades > 0 else 0.0
        total_pnl = sum(t.pnl for t in executed)
        max_dd = self._max_drawdown(equity_curve)
        max_cl = self._max_consecutive(executed, won=False)
        max_cw = self._max_consecutive(executed, won=True)
        sharpe = self._sharpe_ratio(executed)
        pf = self._profit_factor(n_wins, n_losses)
        hourly = self._hourly_breakdown(executed)
        fi = self._filter_impact_analysis(trades)
        duration = time.monotonic() - t_start
        logger.info(
            "Backtest complete: {} trades, {:.1%} accuracy, ${:+.2f} P&L, {:.1f}s".format(
                n_trades, accuracy, total_pnl, duration
            )
        )
        return BacktestResult(
            days=days,
            start_ts=start_ts,
            end_ts=end_ts,
            total_candles=len(df),
            total_trades=n_trades,
            wins=n_wins,
            losses=n_losses,
            skips=n_skips,
            accuracy=accuracy,
            pnl=total_pnl,
            max_drawdown=max_dd,
            max_consecutive_losses=max_cl,
            max_consecutive_wins=max_cw,
            sharpe_ratio=sharpe,
            profit_factor=pf,
            trades=trades,
            hourly_stats=hourly,
            filter_impacts=fi,
            equity_curve=equity_curve,
            duration_seconds=duration,
            model_used=model_used,
        )

    def run_filter_analysis(self, days: int = 7) -> List[FilterImpact]:
        result = self.run(days=days)
        return result.filter_impacts

    def compare_models(self, days: int = 7) -> Dict[str, Any]:
        saved = self.ensemble
        self.ensemble = None
        heuristic_result = self.run(days=days)
        self.ensemble = saved
        out: Dict[str, Any] = {
            "heuristic": {
                "trades": heuristic_result.total_trades,
                "accuracy": heuristic_result.accuracy,
                "pnl": heuristic_result.pnl,
                "sharpe": heuristic_result.sharpe_ratio,
            }
        }
        if self.ensemble is not None and getattr(self.ensemble, "is_ready", False):
            ensemble_result = self.run(days=days)
            out["ensemble"] = {
                "trades": ensemble_result.total_trades,
                "accuracy": ensemble_result.accuracy,
                "pnl": ensemble_result.pnl,
                "sharpe": ensemble_result.sharpe_ratio,
            }
        else:
            out["ensemble"] = None
        return out

    def _ml_signal(self, df: pd.DataFrame, idx: int) -> Tuple[str, float, int, str]:
        try:
            window = df.iloc[max(0, idx - 100): idx + 1].copy()
            signal = self.ensemble.predict(features={}, df_5m=window)
            return signal.direction, signal.confidence, signal.agreement, signal.regime
        except Exception as e:
            logger.debug("ML signal failed at idx {}: {}, using heuristic".format(idx, e))
            direction, confidence = self._heuristic.predict(df, idx)
            return direction, confidence, 2, _estimate_regime(df, idx)

    @staticmethod
    def _atr_percentile(current_atr: float, history: List[float]) -> float:
        if len(history) < 10 or current_atr <= 0:
            return 50.0
        arr = np.array(history)
        return float(np.sum(arr < current_atr) / len(arr) * 100.0)

    @staticmethod
    def _max_drawdown(equity_curve: List[float]) -> float:
        if len(equity_curve) < 2:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for v in equity_curve:
            if v > peak:
                peak = v
            dd = peak - v
            if dd > max_dd:
                max_dd = dd
        return round(max_dd, 4)

    @staticmethod
    def _max_consecutive(trades: List[TradeRecord], won: bool) -> int:
        max_s = 0
        cur = 0
        for t in trades:
            if t.won == won:
                cur += 1
                max_s = max(max_s, cur)
            else:
                cur = 0
        return max_s

    @staticmethod
    def _sharpe_ratio(trades: List[TradeRecord]) -> float:
        if len(trades) < 5:
            return 0.0
        pnls = [t.pnl for t in trades]
        mean = float(np.mean(pnls))
        std = float(np.std(pnls))
        if std < 1e-9:
            return 0.0
        return round(mean / std * math.sqrt(288 * 365), 3)

    @staticmethod
    def _profit_factor(wins: int, losses: int) -> float:
        gross_wins = wins * WIN_PNL
        gross_losses = losses * abs(LOSS_PNL)
        if gross_losses < 1e-9:
            return float("inf") if gross_wins > 0 else 1.0
        return round(gross_wins / gross_losses, 3)

    @staticmethod
    def _hourly_breakdown(trades: List[TradeRecord]) -> List[HourlyStats]:
        hourly: Dict[int, Dict[str, Any]] = {
            h: {"trades": 0, "wins": 0, "pnl": 0.0} for h in range(24)
        }
        for t in trades:
            dt = datetime.fromtimestamp(t.timestamp / 1000, tz=timezone.utc)
            h = dt.hour
            hourly[h]["trades"] += 1
            if t.won:
                hourly[h]["wins"] += 1
            hourly[h]["pnl"] += t.pnl
        result = []
        for h in range(24):
            n = hourly[h]["trades"]
            w = hourly[h]["wins"]
            result.append(HourlyStats(
                hour=h,
                trades=n,
                wins=w,
                accuracy=w / n if n > 0 else 0.0,
                pnl=round(hourly[h]["pnl"], 4),
            ))
        return result

    @staticmethod
    def _filter_impact_analysis(all_trades: List[TradeRecord]) -> List[FilterImpact]:
        filter_names = ["confidence", "volatility", "regime", "agreement", "streak", "correlation"]
        passed_all = [t for t in all_trades if not t.was_skipped]
        n_passed = len(passed_all)
        acc_with = sum(1 for t in passed_all if t.won) / n_passed if n_passed > 0 else 0.0
        pnl_with = sum(t.pnl for t in passed_all)
        impacts = []
        for fname in filter_names:
            blocked: List[TradeRecord] = []
            for t in all_trades:
                if not t.was_skipped:
                    continue
                verdicts = t.filters_applied
                this_failed = not verdicts.get(fname, True)
                others_ok = all(verdicts.get(fn, True) for fn in filter_names if fn != fname)
                if this_failed and others_ok:
                    blocked.append(t)
            n_without = n_passed + len(blocked)
            if blocked:
                sim_wins = int(len(blocked) * 0.52)
                sim_pnl = sim_wins * WIN_PNL + (len(blocked) - sim_wins) * LOSS_PNL
                acc_without = (
                    (sum(1 for t in passed_all if t.won) + sim_wins) / n_without
                    if n_without > 0 else 0.0
                )
                pnl_without = pnl_with + sim_pnl
            else:
                acc_without = acc_with
                pnl_without = pnl_with
            impacts.append(FilterImpact(
                filter_name=fname,
                trades_blocked=len(blocked),
                accuracy_with=round(acc_with, 4),
                accuracy_without=round(acc_without, 4),
                pnl_with=round(pnl_with, 4),
                pnl_without=round(pnl_without, 4),
            ))
        return impacts

    @staticmethod
    def _empty_result(days: int, start_ts: int, end_ts: int) -> BacktestResult:
        return BacktestResult(
            days=days, start_ts=start_ts, end_ts=end_ts, total_candles=0,
            total_trades=0, wins=0, losses=0, skips=0, accuracy=0.0, pnl=0.0,
            max_drawdown=0.0, max_consecutive_losses=0, max_consecutive_wins=0,
            sharpe_ratio=0.0, profit_factor=0.0,
        )
