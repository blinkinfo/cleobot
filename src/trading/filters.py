"""Signal filters for CleoBot -- Section 7 of the master plan.

All 6 filters are implemented here:
  1. Confidence Filter   -- meta-learner probability threshold
  2. Volatility Filter   -- ATR percentile range check
  3. Regime Filter       -- regime-specific confidence thresholds
  4. Agreement Filter    -- minimum 2/3 base model agreement
  5. Streak Filter       -- pause after consecutive losses
  6. Correlation Filter  -- rolling accuracy check (emergency brake)

Filters are evaluated in order. The first SKIP terminates evaluation
but ALL filter verdicts are collected and returned for the signal card.

The filter pipeline returns a FilterResult dataclass with:
  - decision: 'TRADE' or 'SKIP'
  - skip_reason: str (populated if SKIP)
  - verdicts: dict of per-filter verdicts (for Telegram signal card)
  - is_premium: bool (3/3 agreement + high confidence)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.models.ensemble import EnsembleSignal
from src.models.regime_detector import REGIME_CONFIDENCE_THRESHOLDS, DEFAULT_CONFIDENCE_THRESHOLD
from src.utils.logger import get_logger

logger = get_logger("trading.filters")

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

# Base confidence threshold (used when regime doesn't override)
BASE_CONFIDENCE_THRESHOLD = 0.58
BASE_CONFIDENCE_THRESHOLD_LOW = 0.42  # DOWN direction

# Volatility filter percentile bounds
VOL_ATR_LOW_PCTL = 10.0
VOL_ATR_HIGH_PCTL = 95.0

# Agreement filter minimum
MIN_AGREEMENT = 2  # at least 2/3 models must agree
PREMIUM_AGREEMENT = 3
PREMIUM_CONFIDENCE = 0.60

# Streak filter pause rules
STREAK_PAUSE_3 = 1   # cycles to pause after 3 consecutive losses
STREAK_PAUSE_5 = 3   # cycles to pause after 5 consecutive losses
STREAK_PAUSE_7 = -1  # -1 means require manual restart

# Correlation filter thresholds (rolling accuracy)
CORR_EMERGENCY_THRESHOLD = 0.50   # below this -> emergency retrain
CORR_REDUCE_THRESHOLD = 0.53      # below this -> raise threshold by 0.03
CORR_CHECK_EVERY = 10             # check every N trades
CORR_WINDOW = 50                  # rolling window size

# Low-vol ranging raised threshold
LOW_VOL_THRESHOLD_OVERRIDE = 0.62

# Chaotic regime high-conviction threshold
CHAOTIC_THRESHOLD = 0.65


# ------------------------------------------------------------------ #
# Data Structures
# ------------------------------------------------------------------ #

@dataclass
class FilterVerdict:
    """Result from a single filter evaluation."""
    filter_name: str
    passed: bool
    value: float        # The measured value
    threshold: float    # The threshold it was compared against
    message: str        # Human-readable description (for Telegram)
    is_warning: bool = False  # WARN state (between pass and fail)

    @property
    def status_str(self) -> str:
        if self.is_warning:
            return "WARN"
        return "PASS" if self.passed else "FAIL"


@dataclass
class FilterResult:
    """Aggregated result from the full filter pipeline."""
    decision: str                           # 'TRADE' or 'SKIP'
    skip_reason: str = ""                   # Populated if SKIP
    verdicts: Dict[str, FilterVerdict] = field(default_factory=dict)
    is_premium: bool = False                # 3/3 agreement + high confidence
    adjusted_threshold: float = BASE_CONFIDENCE_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dict for DB storage."""
        return {
            "decision": self.decision,
            "skip_reason": self.skip_reason,
            "is_premium": self.is_premium,
            "adjusted_threshold": self.adjusted_threshold,
            "verdicts": {
                name: {
                    "passed": v.passed,
                    "value": v.value,
                    "threshold": v.threshold,
                    "message": v.message,
                    "is_warning": v.is_warning,
                }
                for name, v in self.verdicts.items()
            },
        }


# ------------------------------------------------------------------ #
# Signal Filter Class
# ------------------------------------------------------------------ #

class SignalFilter:
    """Evaluates all 6 signal filters and returns a FilterResult.

    Maintains internal state for:
      - Streak tracking (consecutive losses)
      - Calibrator (isotonic regression)
      - Pause counters
      - Rolling accuracy tracking
    """

    def __init__(self):
        # Isotonic regression calibrator for confidence scores
        self._calibrator: Optional[IsotonicRegression] = None
        self._calibrator_fitted: bool = False

        # Streak state
        self._pause_cycles_remaining: int = 0
        self._streak_requires_manual_restart: bool = False

        # Rolling accuracy buffer (list of 1=win, 0=loss for last CORR_WINDOW trades)
        self._rolling_outcomes: List[int] = []

        # Trade count since last correlation check
        self._trades_since_corr_check: int = 0

        # Dynamic confidence threshold adjustment from correlation filter
        self._corr_threshold_boost: float = 0.0

        # ATR history for percentile calculation (last 24h = 288 5m candles)
        self._atr_history: List[float] = []
        self._atr_history_maxlen: int = 300

        # Spread history for percentile calculation
        self._spread_history: List[float] = []
        self._spread_history_maxlen: int = 720  # ~1 hour at 5s intervals

    # ---------------------------------------------------------------- #
    # PUBLIC API
    # ---------------------------------------------------------------- #

    def evaluate(
        self,
        signal: EnsembleSignal,
        current_atr: float,
        consecutive_losses: int,
        rolling_accuracy: Optional[float],
        n_settled_trades: int,
    ) -> FilterResult:
        """Run all 6 filters and return a combined FilterResult.

        Filters are evaluated in order. Evaluation does NOT short-circuit:
        all filters are always run so the Telegram card shows every verdict.
        The first failing filter becomes the skip_reason.

        Args:
            signal: EnsembleSignal from the model ensemble.
            current_atr: Current ATR value from feature engine.
            consecutive_losses: Number of consecutive losses from DB.
            rolling_accuracy: Rolling accuracy (0-1) over last N trades, or None.
            n_settled_trades: Total settled trades (for correlation check timing).

        Returns:
            FilterResult with decision, verdicts, and optional skip reason.
        """
        verdicts: Dict[str, FilterVerdict] = {}
        skip_reason: str = ""
        first_fail: str = ""

        # Determine effective confidence threshold based on regime
        effective_threshold = self._get_effective_threshold(
            signal.regime, signal.confidence
        )

        # Apply correlation boost to threshold
        effective_threshold = min(
            effective_threshold + self._corr_threshold_boost, 0.70
        )

        # ---- Filter 1: Confidence ---- #
        conf_verdict = self._filter_confidence(signal, effective_threshold)
        verdicts["confidence"] = conf_verdict
        if not conf_verdict.passed and not first_fail:
            first_fail = "Confidence below threshold"

        # ---- Filter 2: Volatility ---- #
        vol_verdict = self._filter_volatility(current_atr)
        verdicts["volatility"] = vol_verdict
        if not vol_verdict.passed and not first_fail:
            first_fail = "ATR outside acceptable volatility range"

        # ---- Filter 3: Regime ---- #
        regime_verdict = self._filter_regime(signal, effective_threshold)
        verdicts["regime"] = regime_verdict
        if not regime_verdict.passed and not first_fail:
            first_fail = f"Regime filter: {regime_verdict.message}"

        # ---- Filter 4: Agreement ---- #
        agree_verdict = self._filter_agreement(signal)
        verdicts["agreement"] = agree_verdict
        if not agree_verdict.passed and not first_fail:
            first_fail = "Insufficient model agreement (1/3)"

        # ---- Filter 5: Streak ---- #
        streak_verdict = self._filter_streak(consecutive_losses)
        verdicts["streak"] = streak_verdict
        if not streak_verdict.passed and not first_fail:
            first_fail = streak_verdict.message

        # ---- Filter 6: Correlation ---- #
        corr_verdict = self._filter_correlation(
            rolling_accuracy, n_settled_trades
        )
        verdicts["correlation"] = corr_verdict
        if not corr_verdict.passed and not first_fail:
            first_fail = corr_verdict.message

        # ---- Determine decision ---- #
        if self._streak_requires_manual_restart:
            decision = "SKIP"
            skip_reason = "7-loss streak: manual restart required (/start)"
        elif first_fail:
            decision = "SKIP"
            skip_reason = first_fail
        else:
            decision = "TRADE"

        # ---- Premium signal detection ---- #
        is_premium = (
            signal.agreement == PREMIUM_AGREEMENT
            and signal.confidence >= PREMIUM_CONFIDENCE
            and decision == "TRADE"
        )

        result = FilterResult(
            decision=decision,
            skip_reason=skip_reason,
            verdicts=verdicts,
            is_premium=is_premium,
            adjusted_threshold=effective_threshold,
        )

        logger.info(
            f"Filter result: {decision} "
            f"(conf={signal.confidence:.3f}, "
            f"threshold={effective_threshold:.3f}, "
            f"regime={signal.regime_display}, "
            f"agreement={signal.agreement}/3, "
            f"streak={consecutive_losses}L, "
            f"rolling_acc={(f'{rolling_accuracy:.3f}' if rolling_accuracy is not None else 'N/A')})"
            + (f" SKIP: {skip_reason}" if decision == "SKIP" else "")
        )

        return result

    def record_outcome(self, won: bool):
        """Record a trade outcome for streak and rolling accuracy tracking.

        Must be called after every settled trade.

        Args:
            won: True if the trade was a win.
        """
        self._rolling_outcomes.append(1 if won else 0)
        if len(self._rolling_outcomes) > CORR_WINDOW:
            self._rolling_outcomes.pop(0)

        self._trades_since_corr_check += 1

        # Update correlation threshold boost every CORR_CHECK_EVERY trades
        if self._trades_since_corr_check >= CORR_CHECK_EVERY:
            self._update_correlation_boost()
            self._trades_since_corr_check = 0

    def update_streak_state(self, consecutive_losses: int):
        """Sync streak state with the database's consecutive loss count.

        Called at startup or after a manual reset.

        Args:
            consecutive_losses: Current consecutive losses from DB.
        """
        if consecutive_losses >= 7 and self._pause_cycles_remaining == 0:
            self._streak_requires_manual_restart = True
        elif consecutive_losses < 7:
            self._streak_requires_manual_restart = False

    def manual_restart_streak(self):
        """Clear the manual-restart requirement (called by /start command)."""
        self._streak_requires_manual_restart = False
        self._pause_cycles_remaining = 0
        logger.info("Streak manual restart: trading re-enabled.")

    def update_pause_counter(self):
        """Decrement pause counter by 1. Call once per trading cycle."""
        if self._pause_cycles_remaining > 0:
            self._pause_cycles_remaining -= 1
            logger.debug(
                f"Pause counter decremented: {self._pause_cycles_remaining} cycles remaining."
            )

    def add_atr_observation(self, atr_value: float):
        """Add a new ATR observation for percentile calculation.

        Args:
            atr_value: Current ATR value.
        """
        if atr_value > 0:
            self._atr_history.append(atr_value)
            if len(self._atr_history) > self._atr_history_maxlen:
                self._atr_history.pop(0)

    def recalibrate(
        self,
        probabilities: List[float],
        outcomes: List[int],
    ):
        """Recalibrate confidence scores using isotonic regression.

        Called during incremental updates (every 6 hours).

        Args:
            probabilities: List of raw model probabilities.
            outcomes: List of binary outcomes (1=win, 0=loss).
        """
        if len(probabilities) < 20:
            logger.warning(
                f"Skipping calibration: only {len(probabilities)} samples (need 20+)."
            )
            return

        try:
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(
                np.array(probabilities).reshape(-1, 1).ravel(),
                np.array(outcomes),
            )
            self._calibrator_fitted = True
            logger.info(
                f"Confidence calibrator fitted on {len(probabilities)} samples."
            )
        except Exception as e:
            logger.error(f"Calibration failed: {e}")

    def calibrate_confidence(self, raw_probability: float) -> float:
        """Apply isotonic calibration to a raw probability.

        Args:
            raw_probability: Raw model probability (0-1).

        Returns:
            Calibrated probability (0-1). Returns input if calibrator not fitted.
        """
        if not self._calibrator_fitted or self._calibrator is None:
            return raw_probability
        try:
            calibrated = float(
                self._calibrator.predict([raw_probability])[0]
            )
            return max(0.0, min(1.0, calibrated))
        except Exception:
            return raw_probability

    def get_state(self) -> Dict[str, Any]:
        """Get current filter state for inspection/debugging."""
        return {
            "pause_cycles_remaining": self._pause_cycles_remaining,
            "streak_requires_manual_restart": self._streak_requires_manual_restart,
            "rolling_outcomes_count": len(self._rolling_outcomes),
            "rolling_accuracy": (
                sum(self._rolling_outcomes) / len(self._rolling_outcomes)
                if self._rolling_outcomes else None
            ),
            "corr_threshold_boost": self._corr_threshold_boost,
            "atr_history_count": len(self._atr_history),
            "calibrator_fitted": self._calibrator_fitted,
        }

    # ---------------------------------------------------------------- #
    # INDIVIDUAL FILTER IMPLEMENTATIONS
    # ---------------------------------------------------------------- #

    def _filter_confidence(
        self, signal: EnsembleSignal, threshold: float
    ) -> FilterVerdict:
        """Filter 1: Meta-learner probability must exceed threshold."""
        confidence = signal.confidence
        # For DOWN direction, confidence is distance of probability from 0.5
        # The signal already has confidence as abs(prob - 0.5) * 2
        passed = confidence >= threshold
        return FilterVerdict(
            filter_name="confidence",
            passed=passed,
            value=confidence,
            threshold=threshold,
            message=(
                f"PASS ({confidence:.1%} > {threshold:.1%})"
                if passed
                else f"FAIL ({confidence:.1%} < {threshold:.1%})"
            ),
        )

    def _filter_volatility(self, current_atr: float) -> FilterVerdict:
        """Filter 2: ATR must be within acceptable percentile range."""
        if len(self._atr_history) < 10:
            # Not enough history yet -- pass through with warning
            return FilterVerdict(
                filter_name="volatility",
                passed=True,
                value=current_atr,
                threshold=0.0,
                message=f"PASS (insufficient ATR history, {len(self._atr_history)} obs)",
                is_warning=True,
            )

        atr_arr = np.array(self._atr_history)
        percentile = float(np.sum(atr_arr < current_atr) / len(atr_arr) * 100)

        too_low = percentile < VOL_ATR_LOW_PCTL
        too_high = percentile > VOL_ATR_HIGH_PCTL
        passed = not too_low and not too_high

        if too_low:
            msg = f"FAIL (ATR {current_atr:.4f} at {percentile:.0f}th pctl -- too quiet)"
        elif too_high:
            msg = f"FAIL (ATR {current_atr:.4f} at {percentile:.0f}th pctl -- too chaotic)"
        else:
            msg = f"PASS (ATR: {percentile:.0f}th pctl)"

        return FilterVerdict(
            filter_name="volatility",
            passed=passed,
            value=percentile,
            threshold=VOL_ATR_LOW_PCTL,
            message=msg,
        )

    def _filter_regime(
        self, signal: EnsembleSignal, effective_threshold: float
    ) -> FilterVerdict:
        """Filter 3: Regime-based confidence threshold."""
        regime = signal.regime
        confidence = signal.confidence
        regime_display = signal.regime_display

        if regime == "high_vol_chaotic":
            # Chaotic: need all 3 models to agree AND high confidence
            if signal.agreement < PREMIUM_AGREEMENT or confidence < CHAOTIC_THRESHOLD:
                return FilterVerdict(
                    filter_name="regime",
                    passed=False,
                    value=confidence,
                    threshold=CHAOTIC_THRESHOLD,
                    message=(
                        f"FAIL (Chaotic: need 3/3 agree + conf>{CHAOTIC_THRESHOLD:.0%}, "
                        f"got {signal.agreement}/3 + {confidence:.1%})"
                    ),
                )
            return FilterVerdict(
                filter_name="regime",
                passed=True,
                value=confidence,
                threshold=CHAOTIC_THRESHOLD,
                message=f"PASS (Chaotic: high-conviction met)",
            )

        elif regime == "low_vol_ranging":
            # Low-vol: raised threshold, issue warning
            threshold = LOW_VOL_THRESHOLD_OVERRIDE
            passed = confidence >= threshold
            return FilterVerdict(
                filter_name="regime",
                passed=passed,
                value=confidence,
                threshold=threshold,
                message=(
                    f"{'PASS' if passed else 'FAIL'} "
                    f"(Low-vol, threshold raised to {threshold:.0%})"
                ),
                is_warning=not passed,
            )

        elif regime in ("trending_up", "trending_down"):
            # Trending: use slightly lower threshold (56% instead of 58%)
            threshold = REGIME_CONFIDENCE_THRESHOLDS.get(regime, DEFAULT_CONFIDENCE_THRESHOLD)
            passed = confidence >= threshold
            direction_match = (
                (regime == "trending_up" and signal.direction == "UP")
                or (regime == "trending_down" and signal.direction == "DOWN")
            )
            bonus = " (trend-aligned)" if direction_match else ""
            return FilterVerdict(
                filter_name="regime",
                passed=passed,
                value=confidence,
                threshold=threshold,
                message=(
                    f"{'PASS' if passed else 'FAIL'} "
                    f"(Trending, threshold={threshold:.0%}{bonus})"
                ),
            )

        # Default / unknown regime
        passed = confidence >= effective_threshold
        return FilterVerdict(
            filter_name="regime",
            passed=passed,
            value=confidence,
            threshold=effective_threshold,
            message=f"{'PASS' if passed else 'FAIL'} ({regime_display})",
        )

    def _filter_agreement(self, signal: EnsembleSignal) -> FilterVerdict:
        """Filter 4: At least 2/3 models must agree on direction."""
        agreement = signal.agreement
        passed = agreement >= MIN_AGREEMENT
        return FilterVerdict(
            filter_name="agreement",
            passed=passed,
            value=float(agreement),
            threshold=float(MIN_AGREEMENT),
            message=(
                f"PASS ({agreement}/3)"
                if passed
                else f"FAIL ({agreement}/3 -- too much disagreement)"
            ),
        )

    def _filter_streak(self, consecutive_losses: int) -> FilterVerdict:
        """Filter 5: Pause after consecutive losses."""
        # Hard 7-loss wall
        if consecutive_losses >= 7 or self._streak_requires_manual_restart:
            self._streak_requires_manual_restart = True
            return FilterVerdict(
                filter_name="streak",
                passed=False,
                value=float(consecutive_losses),
                threshold=7.0,
                message="FAIL (7-loss streak: manual /start required)",
            )

        # 5-loss pause (3 cycles)
        if consecutive_losses >= 5:
            if self._pause_cycles_remaining == 0:
                self._pause_cycles_remaining = STREAK_PAUSE_5
                logger.warning(
                    f"5-loss streak -- pausing for {STREAK_PAUSE_5} cycles."
                )

        # 3-loss pause (1 cycle)
        elif consecutive_losses >= 3:
            if self._pause_cycles_remaining == 0:
                self._pause_cycles_remaining = STREAK_PAUSE_3
                logger.warning(
                    f"3-loss streak -- pausing for {STREAK_PAUSE_3} cycle."
                )

        # Check if currently paused
        if self._pause_cycles_remaining > 0:
            paused_msg = (
                f"FAIL (Streak pause: {self._pause_cycles_remaining} cycles remaining, "
                f"{consecutive_losses}L streak)"
            )
            return FilterVerdict(
                filter_name="streak",
                passed=False,
                value=float(consecutive_losses),
                threshold=3.0,
                message=paused_msg,
            )

        # No issue
        if consecutive_losses > 0:
            msg = f"PASS (L{consecutive_losses} streak, below pause threshold)"
        else:
            msg = "PASS (no loss streak)"

        return FilterVerdict(
            filter_name="streak",
            passed=True,
            value=float(consecutive_losses),
            threshold=3.0,
            message=msg,
        )

    def _filter_correlation(
        self,
        rolling_accuracy: Optional[float],
        n_settled_trades: int,
    ) -> FilterVerdict:
        """Filter 6: Rolling accuracy check -- emergency brake."""
        if rolling_accuracy is None or n_settled_trades < CORR_WINDOW:
            return FilterVerdict(
                filter_name="correlation",
                passed=True,
                value=0.0,
                threshold=CORR_EMERGENCY_THRESHOLD,
                message=f"PASS (insufficient history: {n_settled_trades} trades)",
                is_warning=n_settled_trades < 10,
            )

        if rolling_accuracy < CORR_EMERGENCY_THRESHOLD:
            # Emergency: below 50% over last 50 trades
            return FilterVerdict(
                filter_name="correlation",
                passed=False,
                value=rolling_accuracy,
                threshold=CORR_EMERGENCY_THRESHOLD,
                message=(
                    f"FAIL ({rolling_accuracy:.1%} < {CORR_EMERGENCY_THRESHOLD:.0%} "
                    f"over {CORR_WINDOW} trades -- emergency retrain needed)"
                ),
            )

        if rolling_accuracy < CORR_REDUCE_THRESHOLD:
            # Degraded: between 50% and 53% -- warn but don't block
            return FilterVerdict(
                filter_name="correlation",
                passed=True,
                value=rolling_accuracy,
                threshold=CORR_REDUCE_THRESHOLD,
                message=(
                    f"WARN ({rolling_accuracy:.1%} (50-53%) -- threshold boosted +3%)"
                ),
                is_warning=True,
            )

        return FilterVerdict(
            filter_name="correlation",
            passed=True,
            value=rolling_accuracy,
            threshold=CORR_EMERGENCY_THRESHOLD,
            message=f"PASS ({rolling_accuracy:.1%} / {CORR_WINDOW} trades)",
        )

    # ---------------------------------------------------------------- #
    # PRIVATE HELPERS
    # ---------------------------------------------------------------- #

    def _get_effective_threshold(self, regime: str, confidence: float) -> float:
        """Compute the effective confidence threshold for the given regime."""
        base = REGIME_CONFIDENCE_THRESHOLDS.get(regime, DEFAULT_CONFIDENCE_THRESHOLD)
        return base

    def _update_correlation_boost(self):
        """Recalculate the dynamic threshold boost from correlation filter."""
        if len(self._rolling_outcomes) < CORR_WINDOW:
            return

        recent_acc = sum(self._rolling_outcomes) / len(self._rolling_outcomes)

        if recent_acc < CORR_EMERGENCY_THRESHOLD:
            # Will be caught by the filter itself -- no boost needed (it's a hard fail)
            self._corr_threshold_boost = 0.0
        elif recent_acc < CORR_REDUCE_THRESHOLD:
            self._corr_threshold_boost = 0.03
            logger.info(
                f"Correlation filter: accuracy {recent_acc:.1%} below {CORR_REDUCE_THRESHOLD:.0%} "
                "-- threshold boosted by +3%."
            )
        else:
            if self._corr_threshold_boost > 0:
                logger.info(
                    f"Correlation filter: accuracy recovered to {recent_acc:.1%} -- boost removed."
                )
            self._corr_threshold_boost = 0.0
