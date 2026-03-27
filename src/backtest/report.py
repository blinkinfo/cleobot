"""Backtest report generator for CleoBot Telegram bot.

Generates formatted text reports from BacktestResult objects for display
in Telegram messages. All reports are plain text (no markdown) to ensure
compatibility with all Telegram parse modes.

Reports:
  - summary(): Full backtest result summary
  - hourly_breakdown(): Per-hour accuracy and P&L table
  - filter_analysis(): Per-filter impact on accuracy and trade count
  - model_comparison(): Side-by-side heuristic vs ensemble comparison
  - equity_ascii(): ASCII equity curve for the equity_curve list
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.backtest.engine import BacktestResult, FilterImpact, HourlyStats
from src.utils.logger import get_logger

logger = get_logger("backtest.report")

# Telegram message character limit (4096) -- keep reports well under
MAX_REPORT_CHARS = 3800

# Payout constants (must match engine.py)
WIN_PNL = 0.88
LOSS_PNL = -1.00
BREAKEVEN_WIN_RATE = 53.19


def _pnl_icon(pnl: float) -> str:
    return "+" if pnl >= 0 else ""


def _fmt_ts(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _bar(value: float, max_value: float, width: int = 10, char: str = "#") -> str:
    if max_value <= 0:
        return " " * width
    filled = int(round(value / max_value * width))
    filled = max(0, min(filled, width))
    return char * filled + "." * (width - filled)


class BacktestReport:
    """Generates formatted text reports from a BacktestResult.

    Usage::
        report = BacktestReport(result)
        text = report.summary()
        hourly_text = report.hourly_breakdown()
        filter_text = report.filter_analysis()
    """

    def __init__(self, result: BacktestResult):
        self.result = result

    # ------------------------------------------------------------------
    # Public report methods
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Full backtest result summary card for Telegram."""
        r = self.result
        pnl_sign = _pnl_icon(r.pnl)
        acc_pct = r.accuracy * 100
        be_pct = BREAKEVEN_WIN_RATE

        # Determine edge vs breakeven
        edge = acc_pct - be_pct
        if edge > 0:
            edge_str = "+{:.1f}% above breakeven".format(edge)
        else:
            edge_str = "{:.1f}% below breakeven".format(edge)

        # Model label
        model_label = "ML Ensemble" if r.model_used == "ensemble" else "Heuristic (RSI+EMA)"

        lines = [
            "=" * 32,
            "  BACKTEST RESULTS | {}d".format(r.days),
            "=" * 32,
            "Period:   {} to".format(_fmt_ts(r.start_ts)),
            "          {}".format(_fmt_ts(r.end_ts)),
            "Model:    {}".format(model_label),
            "Candles:  {}".format(r.total_candles),
            "",
            "--- Trade Summary ---",
            "Traded:   {}".format(r.total_trades),
            "Skipped:  {}".format(r.skips),
            "Wins:     {}".format(r.wins),
            "Losses:   {}".format(r.losses),
            "Accuracy: {:.1f}%  ({})".format(acc_pct, edge_str),
            "P&L:      ${}{:.2f}".format(pnl_sign, r.pnl),
            "",
            "--- Risk Metrics ---",
            "Max Drawdown:     ${:.2f}".format(r.max_drawdown),
            "Max Consec Loss:  {}".format(r.max_consecutive_losses),
            "Max Consec Win:   {}".format(r.max_consecutive_wins),
            "Profit Factor:    {:.3f}".format(r.profit_factor),
            "Sharpe Ratio:     {:.3f}".format(r.sharpe_ratio),
            "",
            "--- Daily Rates ---",
        ]

        if r.days > 0:
            trades_per_day = r.total_trades / r.days
            pnl_per_day = r.pnl / r.days
            lines += [
                "Trades/day: {:.1f}".format(trades_per_day),
                "P&L/day:    ${}{:.2f}".format(_pnl_icon(pnl_per_day), pnl_per_day),
            ]

        lines += [
            "",
            "Run time: {:.1f}s".format(r.duration_seconds),
            "=" * 32,
        ]
        return "\n".join(lines)

    def hourly_breakdown(self) -> str:
        """Per-hour accuracy and P&L breakdown table."""
        r = self.result
        if not r.hourly_stats:
            return "No hourly data available."

        # Only show hours with at least 1 trade
        active = [h for h in r.hourly_stats if h.trades > 0]
        if not active:
            return "No trades in this backtest period."

        max_trades = max(h.trades for h in active)

        lines = [
            "=" * 36,
            "  HOURLY BREAKDOWN | {}d Backtest".format(r.days),
            "=" * 36,
            "{:<5} {:>6} {:>7} {:>7}  {}".format("Hour", "Trades", "Acc%", "P&L", "Bar"),
            "-" * 36,
        ]

        for h in active:
            bar = _bar(h.trades, max_trades, width=8)
            acc_pct = h.accuracy * 100
            pnl_sign = _pnl_icon(h.pnl)
            lines.append(
                "{:02d}:00  {:>6}  {:>5.1f}%  ${}{:>5.2f}  {}".format(
                    h.hour, h.trades, acc_pct, pnl_sign, abs(h.pnl), bar
                )
            )

        # Best and worst hours
        best = max(active, key=lambda h: h.accuracy if h.trades >= 3 else 0.0)
        worst = min(active, key=lambda h: h.accuracy if h.trades >= 3 else 1.0)
        lines += [
            "-" * 36,
            "Best:  {:02d}:00  ({:.1f}%, {} trades)".format(
                best.hour, best.accuracy * 100, best.trades
            ),
            "Worst: {:02d}:00  ({:.1f}%, {} trades)".format(
                worst.hour, worst.accuracy * 100, worst.trades
            ),
            "=" * 36,
        ]
        return "\n".join(lines)

    def filter_analysis(self) -> str:
        """Per-filter impact on accuracy and trade count."""
        r = self.result
        if not r.filter_impacts:
            return "No filter analysis available."

        lines = [
            "=" * 44,
            "  FILTER ANALYSIS | {}d Backtest".format(r.days),
            "=" * 44,
            "Filter        Block  AccWith  AccW/o  PnlDiff",
            "-" * 44,
        ]

        for fi in r.filter_impacts:
            acc_with_pct = fi.accuracy_with * 100
            acc_without_pct = fi.accuracy_without * 100
            pnl_diff = fi.pnl_without - fi.pnl_with
            pnl_diff_str = "${}{:.2f}".format(_pnl_icon(pnl_diff), abs(pnl_diff))
            # Indicate whether the filter helps (+) or hurts (-)
            impact = "(+)" if fi.accuracy_with >= fi.accuracy_without else "(-)"
            lines.append(
                "{:<12}  {:>5}  {:>5.1f}%  {:>5.1f}%  {:>8}  {}".format(
                    fi.filter_name[:12],
                    fi.trades_blocked,
                    acc_with_pct,
                    acc_without_pct,
                    pnl_diff_str,
                    impact,
                )
            )

        lines += [
            "-" * 44,
            "Block = trades filtered out by this filter only",
            "AccWith = accuracy WITH filter active (baseline)",
            "AccW/o  = simulated accuracy WITHOUT filter",
            "(+) = filter improves accuracy, (-) = filter hurts",
            "=" * 44,
        ]
        return "\n".join(lines)

    def model_comparison(self, comparison: Dict[str, Any]) -> str:
        """Side-by-side model comparison report.

        Args:
            comparison: Dict from BacktestEngine.compare_models()
        """
        heuristic = comparison.get("heuristic") or {}
        ensemble = comparison.get("ensemble")

        lines = [
            "=" * 38,
            "  MODEL COMPARISON BACKTEST",
            "=" * 38,
            "{:<14} {:>10}  {:>10}".format("Metric", "Heuristic", "Ensemble"),
            "-" * 38,
        ]

        def _fmt_val(d: Optional[Dict], key: str, fmt: str = "{:.1%}") -> str:
            if d is None:
                return "N/A"
            val = d.get(key)
            if val is None:
                return "N/A"
            return fmt.format(val)

        lines += [
            "{:<14} {:>10}  {:>10}".format(
                "Trades",
                _fmt_val(heuristic, "trades", "{}"),
                _fmt_val(ensemble, "trades", "{}"),
            ),
            "{:<14} {:>10}  {:>10}".format(
                "Accuracy",
                _fmt_val(heuristic, "accuracy"),
                _fmt_val(ensemble, "accuracy"),
            ),
            "{:<14} {:>10}  {:>10}".format(
                "P&L",
                _fmt_val(heuristic, "pnl", "${:+.2f}"),
                _fmt_val(ensemble, "pnl", "${:+.2f}"),
            ),
            "{:<14} {:>10}  {:>10}".format(
                "Sharpe",
                _fmt_val(heuristic, "sharpe", "{:.3f}"),
                _fmt_val(ensemble, "sharpe", "{:.3f}"),
            ),
        ]

        if ensemble is None:
            lines += [
                "-" * 38,
                "Ensemble: models not trained yet.",
                "Train models to enable ML comparison.",
            ]

        lines.append("=" * 38)
        return "\n".join(lines)

    def equity_ascii(self, width: int = 28) -> str:
        """ASCII equity curve from the backtest equity_curve list."""
        curve = self.result.equity_curve
        if len(curve) < 2:
            return "Insufficient trades for equity curve."

        min_val = min(curve)
        max_val = max(curve)
        val_range = max_val - min_val

        lines = [
            "Equity Curve ({}d)".format(self.result.days),
            "${:.2f} -> ${:.2f}".format(curve[0], curve[-1]),
            "",
        ]

        # Sample down to `width` points
        step = max(1, len(curve) // width)
        sampled = [curve[i] for i in range(0, len(curve), step)]
        sampled.append(curve[-1])  # always include final value

        # Build ASCII bar chart (vertical: 5 rows)
        rows = 5
        chart_lines = []
        for row in range(rows - 1, -1, -1):
            threshold = min_val + (row / (rows - 1)) * val_range if val_range > 0 else min_val
            label = "${:>6.2f}".format(threshold)
            bar_row = ""
            for v in sampled:
                bar_row += "#" if v >= threshold else " "
            chart_lines.append("{}  {}".format(label, bar_row))

        lines.extend(chart_lines)
        lines.append("         " + "^" * len(sampled))
        lines.append("  Start{}End".format(" " * (len(sampled) - 8)))
        return "\n".join(lines)

    def short_summary(self) -> str:
        """Compact one-line summary for inline display."""
        r = self.result
        pnl_sign = _pnl_icon(r.pnl)
        return (
            "{}d BT | {} trades | {:.1f}% acc | ${}{:.2f} P&L | "
            "MaxDD ${:.2f}".format(
                r.days,
                r.total_trades,
                r.accuracy * 100,
                pnl_sign,
                abs(r.pnl),
                r.max_drawdown,
            )
        )


def format_backtest_result(result: BacktestResult) -> str:
    """Convenience function: generate full summary from a BacktestResult."""
    return BacktestReport(result).summary()


def format_filter_analysis(result: BacktestResult) -> str:
    """Convenience function: generate filter analysis from a BacktestResult."""
    return BacktestReport(result).filter_analysis()


def format_model_comparison(comparison: Dict[str, Any]) -> str:
    """Convenience function: generate model comparison report."""
    # Create a dummy result for the report object
    dummy = BacktestResult(
        days=0, start_ts=0, end_ts=0, total_candles=0,
        total_trades=0, wins=0, losses=0, skips=0, accuracy=0.0, pnl=0.0,
        max_drawdown=0.0, max_consecutive_losses=0, max_consecutive_wins=0,
        sharpe_ratio=0.0, profit_factor=0.0,
    )
    return BacktestReport(dummy).model_comparison(comparison)
