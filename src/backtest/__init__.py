"""CleoBot backtesting package.

Exports:
  BacktestEngine  -- walk-forward backtesting engine
  BacktestResult  -- result dataclass
  BacktestReport  -- text report generator
  TradeRecord     -- individual simulated trade
  FilterImpact    -- per-filter impact stats
  HourlyStats     -- per-hour breakdown
"""

from src.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    TradeRecord,
    FilterImpact,
    HourlyStats,
    HeuristicSignalGenerator,
    WIN_PNL,
    LOSS_PNL,
    BREAKEVEN_WIN_RATE,
)
from src.backtest.report import (
    BacktestReport,
    format_backtest_result,
    format_filter_analysis,
    format_model_comparison,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestReport",
    "TradeRecord",
    "FilterImpact",
    "HourlyStats",
    "HeuristicSignalGenerator",
    "format_backtest_result",
    "format_filter_analysis",
    "format_model_comparison",
    "WIN_PNL",
    "LOSS_PNL",
    "BREAKEVEN_WIN_RATE",
]
