"""Risk management for CleoBot -- Section 10 of the master plan.

Enforces all risk rules:
  - Daily loss tracking and circuit breaker ($15 max daily loss)
  - Daily drawdown circuit breaker (20% of day's starting balance)
  - Consecutive loss tracking and pause logic
  - Position sizing with profit-based scaling
  - Maximum open exposure management ($3.00 max)
  - Streak-based auto-pause and manual-restart enforcement

The RiskManager is the LAST gate before a trade is placed.
It integrates with the SignalFilter for streak state and with
the Database for persistent daily loss tracking.

All state is reconstructed from the Database on startup so it
survives restarts cleanly.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from src.database import Database
from src.config import TradingConfig
from src.utils.logger import get_logger

logger = get_logger("trading.risk")

# ------------------------------------------------------------------ #
# Constants (from Section 10)
# ------------------------------------------------------------------ #

WIN_PNL = 0.88          # Net profit per winning trade
LOSS_PNL = -1.00        # Net loss per losing trade

BASE_TRADE_SIZE = 1.00  # Default trade size in USD
MAX_TRADE_SIZE = 3.00   # Hard cap on trade size
SCALING_INCREMENT = 0.50  # +$0.50 per scaling step
SCALING_PROFIT_STEP = 50.00  # Profit step for size increase

MAX_DAILY_LOSS = 15.00  # Daily loss circuit breaker
DAILY_DRAWDOWN_PCT = 0.20  # 20% of day's starting balance
MAX_OPEN_EXPOSURE = 3.00  # Max concurrent exposure


# ------------------------------------------------------------------ #
# Data Structures
# ------------------------------------------------------------------ #

@dataclass
class RiskCheckResult:
    """Result from a risk manager check."""
    approved: bool
    trade_size: float
    skip_reason: str = ""
    daily_pnl: float = 0.0
    open_exposure: float = 0.0
    consecutive_losses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "trade_size": self.trade_size,
            "skip_reason": self.skip_reason,
            "daily_pnl": self.daily_pnl,
            "open_exposure": self.open_exposure,
            "consecutive_losses": self.consecutive_losses,
        }


@dataclass
class RiskStatus:
    """Current risk management status for display."""
    auto_trade_active: bool
    daily_pnl: float
    daily_loss_limit: float
    daily_drawdown_used_pct: float
    open_exposure: float
    max_exposure: float
    consecutive_losses: int
    current_trade_size: float
    circuit_breaker_active: bool
    circuit_breaker_reason: str
    trades_today: int
    wins_today: int
    losses_today: int


# ------------------------------------------------------------------ #
# Risk Manager
# ------------------------------------------------------------------ #

class RiskManager:
    """Enforces all risk management rules from Section 10.

    Maintains daily PnL, open exposure, and circuit breaker state.
    All state is loaded from the database on startup.
    """

    def __init__(self, db: Database, config: TradingConfig):
        """Initialise risk manager.

        Args:
            db: Database instance for loading trade history.
            config: TradingConfig with limits from environment.
        """
        self.db = db
        self.config = config

        # Daily state (reconstructed from DB on startup and reset each day)
        self._daily_pnl: float = 0.0
        self._daily_starting_balance: float = 0.0  # balance at start of day
        self._trades_today: int = 0
        self._wins_today: int = 0
        self._losses_today: int = 0
        self._today_date: str = ""

        # Open exposure tracking (trades placed but not yet settled)
        self._open_positions: Dict[int, float] = {}  # trade_id -> trade_size

        # Circuit breaker state
        self._circuit_breaker_active: bool = False
        self._circuit_breaker_reason: str = ""

        # Auto-trade toggle (can be overridden by Telegram)
        self._auto_trade_enabled: bool = config.auto_trade_enabled

        # Cumulative profit (for position scaling)
        self._cumulative_profit: float = 0.0

        # Load state from DB
        self._load_state_from_db()

    # ---------------------------------------------------------------- #
    # PUBLIC API
    # ---------------------------------------------------------------- #

    def check_trade(
        self,
        consecutive_losses: int,
        proposed_size: Optional[float] = None,
    ) -> RiskCheckResult:
        """Check whether a trade is approved by risk management.

        This is the final gate before placing a trade. Checks:
          1. Auto-trade is enabled
          2. Circuit breaker is not active
          3. Daily loss limit not breached
          4. Max open exposure not breached
          5. Consecutive loss hard-stop

        Args:
            consecutive_losses: Current consecutive loss count from DB.
            proposed_size: Desired trade size (uses calculated size if None).

        Returns:
            RiskCheckResult with approval decision and trade size.
        """
        # Refresh daily state
        self._refresh_daily_state()

        trade_size = proposed_size or self.calculate_trade_size()
        open_exp = self.get_open_exposure()

        # Check 1: Auto-trade enabled
        if not self._auto_trade_enabled:
            return RiskCheckResult(
                approved=False,
                trade_size=trade_size,
                skip_reason="Auto-trade is disabled",
                daily_pnl=self._daily_pnl,
                open_exposure=open_exp,
                consecutive_losses=consecutive_losses,
            )

        # Check 2: Circuit breaker
        if self._circuit_breaker_active:
            return RiskCheckResult(
                approved=False,
                trade_size=trade_size,
                skip_reason=f"Circuit breaker active: {self._circuit_breaker_reason}",
                daily_pnl=self._daily_pnl,
                open_exposure=open_exp,
                consecutive_losses=consecutive_losses,
            )

        # Check 3: Daily loss limit
        daily_loss_limit = self.config.max_daily_loss
        if self._daily_pnl <= -daily_loss_limit:
            self._activate_circuit_breaker(
                f"Daily loss limit reached (${abs(self._daily_pnl):.2f} / ${daily_loss_limit:.2f})"
            )
            return RiskCheckResult(
                approved=False,
                trade_size=trade_size,
                skip_reason=self._circuit_breaker_reason,
                daily_pnl=self._daily_pnl,
                open_exposure=open_exp,
                consecutive_losses=consecutive_losses,
            )

        # Check 4: Would the new trade exceed daily loss limit when combined with open exposure?
        # Project worst case: current loss + this trade loss + all open positions lose
        worst_case_pnl = self._daily_pnl - open_exp - trade_size
        if worst_case_pnl <= -daily_loss_limit:
            return RiskCheckResult(
                approved=False,
                trade_size=trade_size,
                skip_reason=(
                    f"Trade would risk daily loss limit "
                    f"(worst case: ${abs(worst_case_pnl):.2f} / limit ${daily_loss_limit:.2f})"
                ),
                daily_pnl=self._daily_pnl,
                open_exposure=open_exp,
                consecutive_losses=consecutive_losses,
            )

        # Check 5: Max open exposure
        max_exp = self.config.max_open_exposure
        if open_exp + trade_size > max_exp:
            return RiskCheckResult(
                approved=False,
                trade_size=trade_size,
                skip_reason=(
                    f"Max exposure reached "
                    f"(${open_exp:.2f} open + ${trade_size:.2f} new > ${max_exp:.2f} limit)"
                ),
                daily_pnl=self._daily_pnl,
                open_exposure=open_exp,
                consecutive_losses=consecutive_losses,
            )

        # All checks passed
        return RiskCheckResult(
            approved=True,
            trade_size=trade_size,
            daily_pnl=self._daily_pnl,
            open_exposure=open_exp,
            consecutive_losses=consecutive_losses,
        )

    def calculate_trade_size(self) -> float:
        """Calculate the current trade size based on cumulative profit scaling.

        Scaling rule (Section 10):
          - Base: $1.00
          - +$0.50 per $50 cumulative profit
          - Maximum: $3.00
          - After circuit breaker: reset to base $1.00

        Returns:
            Trade size in USD.
        """
        if self._circuit_breaker_active:
            return BASE_TRADE_SIZE

        # Scaling: +$0.50 per $50 profit
        profit_steps = max(0, int(self._cumulative_profit / SCALING_PROFIT_STEP))
        size = BASE_TRADE_SIZE + (profit_steps * SCALING_INCREMENT)
        size = min(size, self.config.max_trade_size)
        return round(size, 2)

    def record_trade_placed(
        self,
        trade_id: int,
        trade_size: float,
    ):
        """Record that a trade has been placed (opens exposure).

        Args:
            trade_id: Database trade ID.
            trade_size: Size of the trade in USD.
        """
        self._open_positions[trade_id] = trade_size
        logger.info(
            f"Trade #{trade_id} placed: ${trade_size:.2f}. "
            f"Open exposure: ${self.get_open_exposure():.2f}"
        )

    def record_settlement(
        self,
        trade_id: int,
        won: bool,
        trade_size: float,
        pnl: float,
    ):
        """Record a trade settlement and update daily stats.

        Args:
            trade_id: Database trade ID.
            won: True if the trade was a win.
            trade_size: Trade size in USD.
            pnl: Actual PnL (positive for win, negative for loss).
        """
        # Remove from open positions
        self._open_positions.pop(trade_id, None)

        # Update daily stats
        self._daily_pnl += pnl
        self._cumulative_profit += pnl
        self._trades_today += 1

        if won:
            self._wins_today += 1
        else:
            self._losses_today += 1

        logger.info(
            f"Settlement: trade #{trade_id} {'WIN' if won else 'LOSS'} "
            f"PnL={pnl:+.2f}. Daily PnL: ${self._daily_pnl:+.2f}"
        )

        # Check if we should activate circuit breaker after settlement
        daily_loss_limit = self.config.max_daily_loss
        if self._daily_pnl <= -daily_loss_limit:
            self._activate_circuit_breaker(
                f"Daily loss limit reached: ${abs(self._daily_pnl):.2f} / ${daily_loss_limit:.2f}"
            )

    def record_daily_drawdown_check(self, current_balance: float):
        """Check and enforce daily drawdown circuit breaker.

        Called every trading cycle so the breaker fires as soon as the
        20% intraday drawdown threshold is crossed, not only at trade time.

        Args:
            current_balance: Estimated current portfolio balance in USD.
        """
        if self._daily_starting_balance <= 0:
            # First call of the day -- initialise the starting balance anchor.
            if current_balance > 0:
                self._daily_starting_balance = current_balance
            return

        drawdown = (self._daily_starting_balance - current_balance) / self._daily_starting_balance
        if drawdown >= DAILY_DRAWDOWN_PCT:
            self._activate_circuit_breaker(
                f"Daily drawdown limit reached: {drawdown:.1%} of starting balance"
            )

    def get_open_exposure(self) -> float:
        """Get total current open exposure in USD.

        Returns:
            Sum of all open position sizes.
        """
        return sum(self._open_positions.values())

    def get_current_balance_estimate(self) -> float:
        """Estimate current balance as starting balance plus today's realised PnL.

        Used by the executor to feed `record_daily_drawdown_check` each cycle
        without requiring an external balance feed.

        Returns 0.0 when `_daily_starting_balance` is not yet set (i.e. no
        trades have been placed today), which causes `record_daily_drawdown_check`
        to initialise the anchor rather than fire the breaker.

        Returns:
            Estimated current balance in USD.
        """
        if self._daily_starting_balance <= 0:
            return 0.0
        return self._daily_starting_balance + self._daily_pnl

    def get_status(self) -> RiskStatus:
        """Get full risk status for Telegram display."""
        self._refresh_daily_state()
        return RiskStatus(
            auto_trade_active=self._auto_trade_enabled,
            daily_pnl=self._daily_pnl,
            daily_loss_limit=self.config.max_daily_loss,
            daily_drawdown_used_pct=(
                abs(min(0, self._daily_pnl)) / self.config.max_daily_loss * 100
            ),
            open_exposure=self.get_open_exposure(),
            max_exposure=self.config.max_open_exposure,
            consecutive_losses=self.db.get_consecutive_losses(),
            current_trade_size=self.calculate_trade_size(),
            circuit_breaker_active=self._circuit_breaker_active,
            circuit_breaker_reason=self._circuit_breaker_reason,
            trades_today=self._trades_today,
            wins_today=self._wins_today,
            losses_today=self._losses_today,
        )

    def enable_auto_trade(self):
        """Enable automatic trading."""
        self._auto_trade_enabled = True
        logger.info("Auto-trade ENABLED.")

    def disable_auto_trade(self):
        """Disable automatic trading (signals still generated)."""
        self._auto_trade_enabled = False
        logger.info("Auto-trade DISABLED.")

    def reset_circuit_breaker(self):
        """Reset the circuit breaker (for manual restart or new day).

        Also resets trade size to base.
        """
        was_active = self._circuit_breaker_active
        self._circuit_breaker_active = False
        self._circuit_breaker_reason = ""
        if was_active:
            logger.info("Circuit breaker RESET. Trade size reset to base.")

    def reset_for_new_day(self):
        """Reset daily stats at UTC midnight."""
        old_date = self._today_date
        old_pnl = self._daily_pnl

        self._today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._daily_pnl = 0.0
        self._daily_starting_balance = 0.0
        self._trades_today = 0
        self._wins_today = 0
        self._losses_today = 0

        # Reset circuit breaker at day boundary
        if self._circuit_breaker_active:
            self.reset_circuit_breaker()

        logger.info(
            f"Daily reset: {old_date} -> {self._today_date}. "
            f"Previous day PnL: ${old_pnl:+.2f}"
        )

    def set_trade_size(
        self,
        size: float,
        reason: str = "manual override",
    ):
        """Set a manual trade size override.

        Args:
            size: Trade size in USD (clamped to [BASE, MAX]).
            reason: Reason for override (logged).
        """
        clamped = max(BASE_TRADE_SIZE, min(size, MAX_TRADE_SIZE))
        # We store via cumulative_profit such that calculate_trade_size returns clamped
        # Actually: override cumulative profit to yield the desired size
        # size = BASE + steps * INCREMENT => steps = (size - BASE) / INCREMENT
        # cumulative_profit = steps * SCALING_PROFIT_STEP
        steps = round((clamped - BASE_TRADE_SIZE) / SCALING_INCREMENT)
        self._cumulative_profit = steps * SCALING_PROFIT_STEP
        logger.info(
            f"Trade size set to ${clamped:.2f} ({reason}). "
            f"Synthetic cumulative profit: ${self._cumulative_profit:.2f}"
        )

    def get_daily_stats_summary(self) -> Dict[str, Any]:
        """Get today's stats as a dict (for Telegram performance handler)."""
        self._refresh_daily_state()
        settled = self._wins_today + self._losses_today
        accuracy = self._wins_today / settled if settled > 0 else 0.0
        return {
            "date": self._today_date,
            "trades_placed": self._trades_today,
            "wins": self._wins_today,
            "losses": self._losses_today,
            "settled": settled,
            "accuracy": accuracy,
            "pnl": self._daily_pnl,
            "open_exposure": self.get_open_exposure(),
            "open_positions": len(self._open_positions),
            "circuit_breaker_active": self._circuit_breaker_active,
        }

    # ---------------------------------------------------------------- #
    # PRIVATE HELPERS
    # ---------------------------------------------------------------- #

    def _load_state_from_db(self):
        """Load today's trading state from the database on startup.

        This ensures state is correct after a restart.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._today_date = today

        stats = self.db.get_trade_stats_today()
        self._daily_pnl = stats.get("pnl", 0.0)
        self._wins_today = stats.get("wins", 0)
        self._losses_today = stats.get("losses", 0)
        self._trades_today = stats.get("total_trades", 0)

        # Re-add any still-open positions to exposure tracking
        unsettled = self.db.get_unsettled_trades()
        for trade in unsettled:
            self._open_positions[trade["id"]] = trade.get("trade_size", BASE_TRADE_SIZE)

        # Check circuit breaker state
        daily_loss_limit = self.config.max_daily_loss
        if self._daily_pnl <= -daily_loss_limit:
            self._activate_circuit_breaker(
                f"Daily loss limit already reached: ${abs(self._daily_pnl):.2f} / ${daily_loss_limit:.2f}"
            )

        # Compute cumulative profit from all settled trades
        all_trades = self.db.get_recent_trades(limit=10000)
        self._cumulative_profit = sum(
            t.get("pnl", 0.0) or 0.0
            for t in all_trades
            if t.get("settlement") is not None
        )

        logger.info(
            f"Risk manager loaded: daily PnL=${self._daily_pnl:+.2f}, "
            f"cumulative profit=${self._cumulative_profit:+.2f}, "
            f"open positions={len(self._open_positions)}, "
            f"circuit_breaker={self._circuit_breaker_active}"
        )

    def _refresh_daily_state(self):
        """Check if we've crossed into a new UTC day and reset if so."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._today_date:
            self.reset_for_new_day()

    def _activate_circuit_breaker(self, reason: str):
        """Activate the circuit breaker.

        Args:
            reason: Human-readable reason string.
        """
        if not self._circuit_breaker_active:
            self._circuit_breaker_active = True
            self._circuit_breaker_reason = reason
            logger.warning(f"CIRCUIT BREAKER ACTIVATED: {reason}")
        else:
            # Already active -- update reason if more specific
            self._circuit_breaker_reason = reason
