"""Trading executor for CleoBot -- the full 5-minute cycle orchestrator.

The TradingExecutor is called once per 5-minute candle close and runs:
  1. Compute features (FeatureEngine)
  2. Run ensemble prediction (Ensemble)
  3. Apply signal filters (SignalFilter)
  4. Check risk management (RiskManager)
  5. Place Polymarket order (PolymarketClient)
  6. Settle pending trades
  7. Send Telegram notification
  8. Log everything to the database

Timing budget (seconds):
  Feature computation: < 33s
  Model inference:     < 5s
  Filters + risk:      < 1s
  Order placement:     < 25s
  Total:               < 64s
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

from src.config import Config
from src.database import Database
from src.features.engine import FeatureEngine
from src.models.ensemble import Ensemble, EnsembleSignal
from src.trading.filters import SignalFilter, FilterResult
from src.trading.risk_manager import RiskManager
from src.trading.polymarket import PolymarketClient, OrderResult
from src.utils.logger import get_logger

logger = get_logger("trading.executor")

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

MIN_CYCLES_BETWEEN_TRADES = 1
RETRAIN_HOUR_UTC = 4
RETRAIN_LOOKBACK_ROWS = 10000
INCREMENTAL_UPDATE_CYCLES = 72
SETTLEMENT_SCAN_CYCLES = 1
MAX_UNSETTLED_AGE_MINUTES = 12

DIR_ICON = {"UP": "\U0001F7E2", "DOWN": "\U0001F534"}
FILTER_ICON = {"PASS": "\u2705", "FAIL": "\u274C", "WARN": "\u26A0\uFE0F"}


# ------------------------------------------------------------------ #
# Cycle Result
# ------------------------------------------------------------------ #

class CycleResult:
    """Summary of a single 5-minute trading cycle."""

    def __init__(
        self,
        cycle_ts: datetime,
        signal: Optional[EnsembleSignal] = None,
        filter_result: Optional[FilterResult] = None,
        trade_placed: bool = False,
        order_result: Optional[OrderResult] = None,
        trade_id: Optional[int] = None,
        error: str = "",
        duration_s: float = 0.0,
    ):
        self.cycle_ts = cycle_ts
        self.signal = signal
        self.filter_result = filter_result
        self.trade_placed = trade_placed
        self.order_result = order_result
        self.trade_id = trade_id
        self.error = error
        self.duration_s = duration_s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_ts": self.cycle_ts.isoformat(),
            "trade_placed": self.trade_placed,
            "trade_id": self.trade_id,
            "direction": self.signal.direction if self.signal else None,
            "confidence": self.signal.confidence if self.signal else None,
            "filter_decision": self.filter_result.decision if self.filter_result else None,
            "skip_reason": self.filter_result.skip_reason if self.filter_result else "",
            "error": self.error,
            "duration_s": self.duration_s,
        }


# ------------------------------------------------------------------ #
# Trading Executor
# ------------------------------------------------------------------ #

class TradingExecutor:
    """Orchestrates the full 5-minute trading cycle."""

    def __init__(
        self,
        config: Config,
        db: Database,
        feature_engine: FeatureEngine,
        ensemble: Ensemble,
        polymarket_client: PolymarketClient,
        telegram_bot: Optional[Any] = None,
    ):
        self.config = config
        self.db = db
        self.feature_engine = feature_engine
        self.ensemble = ensemble
        self.polymarket = polymarket_client
        self.telegram = telegram_bot

        self.signal_filter = SignalFilter()
        self.risk_manager = RiskManager(db, config.trading)

        self._cycle_count: int = 0
        self._cycles_since_last_trade: int = MIN_CYCLES_BETWEEN_TRADES
        self._cycles_since_incremental: int = 0
        self._last_retrain_date: str = ""

        self._feature_history: List[Dict[str, float]] = []
        self._feature_history_maxlen: int = 200

        self._pending_settlements: Dict[int, Dict[str, Any]] = {}

        self._total_cycles: int = 0
        self._total_trades: int = 0
        self._total_skips: int = 0
        self._total_errors: int = 0

        logger.info("TradingExecutor initialised.")

    # ---------------------------------------------------------------- #
    # MAIN CYCLE ENTRY POINT
    # ---------------------------------------------------------------- #

    async def run_cycle(
        self,
        candle_ts_ms: int,
        current_orderbook: Optional[Dict[str, Any]] = None,
    ) -> CycleResult:
        """Run a full 5-minute trading cycle."""
        t0 = time.monotonic()
        cycle_ts = datetime.fromtimestamp(candle_ts_ms / 1000, tz=timezone.utc)
        self._cycle_count += 1
        self._total_cycles += 1

        logger.info(
            f"--- Cycle #{self._cycle_count} | {cycle_ts.strftime('%Y-%m-%d %H:%M:%S')} UTC ---"
        )

        result = CycleResult(cycle_ts=cycle_ts)

        try:
            # Step 0: Daily reset check + drawdown circuit breaker
            await self._check_daily_reset(cycle_ts)

            # Step 1: Settle pending trades
            if self._cycle_count % SETTLEMENT_SCAN_CYCLES == 0:
                await self._settle_pending_trades(cycle_ts)

            # Step 2: Compute features
            features = self.feature_engine.compute(
                current_ts_ms=candle_ts_ms,
                current_orderbook=current_orderbook,
            )
            self._update_feature_history(features)

            atr = features.get("atr_12", 0.0)
            self.signal_filter.add_atr_observation(atr)

            # Step 3: Update Polymarket features
            await self._update_polymarket_features(features)

            # Step 4: Run ensemble prediction
            import pandas as pd
            df_5m = self._get_df_5m()
            feature_history_df = self._get_feature_history_df()

            signal = self.ensemble.predict(
                features=features,
                df_5m=df_5m,
                feature_df_history=feature_history_df,
            )
            result.signal = signal

            # Step 5: Apply signal filters
            consecutive_losses = self.db.get_consecutive_losses()
            rolling_accuracy = self.db.get_rolling_accuracy(n_trades=50)
            n_settled = self.db.get_total_settled_trades()

            filter_result = self.signal_filter.evaluate(
                signal=signal,
                current_atr=atr,
                consecutive_losses=consecutive_losses,
                rolling_accuracy=rolling_accuracy,
                n_settled_trades=n_settled,
            )
            result.filter_result = filter_result

            self.signal_filter.update_pause_counter()

            # Step 6: Risk management check
            if filter_result.decision == "TRADE":
                risk_check = self.risk_manager.check_trade(
                    consecutive_losses=consecutive_losses,
                )

                if not risk_check.approved:
                    logger.info(f"Risk manager blocked trade: {risk_check.skip_reason}")
                    filter_result = FilterResult(
                        decision="SKIP",
                        skip_reason=f"Risk: {risk_check.skip_reason}",
                        verdicts=filter_result.verdicts,
                        is_premium=filter_result.is_premium,
                    )
                    result.filter_result = filter_result

            # Step 7: Place trade
            if (
                filter_result.decision == "TRADE"
                and self._cycles_since_last_trade >= MIN_CYCLES_BETWEEN_TRADES
            ):
                order_result, trade_id = await self._place_trade(
                    signal=signal,
                    filter_result=filter_result,
                    features=features,
                )
                result.order_result = order_result
                result.trade_id = trade_id
                result.trade_placed = order_result.success if order_result else False

                if result.trade_placed:
                    self._cycles_since_last_trade = 0
                    self._total_trades += 1
                else:
                    self._total_skips += 1
            else:
                if filter_result.decision == "SKIP":
                    self._total_skips += 1
                self._cycles_since_last_trade += 1

            # Step 8: Send Telegram notification
            await self._send_signal_notification(
                signal=signal,
                filter_result=filter_result,
                order_result=result.order_result,
                trade_id=result.trade_id,
                cycle_ts=cycle_ts,
                features=features,
            )

            # Step 9: Incremental model update check
            self._cycles_since_incremental += 1
            if self._cycles_since_incremental >= INCREMENTAL_UPDATE_CYCLES:
                await self._run_incremental_update()
                self._cycles_since_incremental = 0

            # Step 10: Daily retrain check
            await self._check_retrain_schedule(cycle_ts)

        except RuntimeError as e:
            logger.warning(f"Cycle skipped (startup): {e}")
            result.error = str(e)
        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)
            result.error = str(e)
            self._total_errors += 1

        result.duration_s = time.monotonic() - t0
        logger.info(
            f"Cycle #{self._cycle_count} complete in {result.duration_s:.2f}s | "
            f"trades={self._total_trades} skips={self._total_skips} errors={self._total_errors}"
        )

        return result

    # ---------------------------------------------------------------- #
    # TRADE PLACEMENT
    # ---------------------------------------------------------------- #

    async def _place_trade(
        self,
        signal: EnsembleSignal,
        filter_result: FilterResult,
        features: Dict[str, float],
    ) -> tuple:
        """Place a trade on Polymarket and record it in the database."""
        trade_size = self.risk_manager.calculate_trade_size()
        direction = signal.direction

        logger.info(
            f"Placing {direction} trade: size=${trade_size:.2f}, "
            f"conf={signal.confidence:.3f}, premium={filter_result.is_premium}"
        )

        market = await self.polymarket.find_current_btc_market()

        order_result = await self.polymarket.place_order(
            direction=direction,
            size=trade_size,
            market=market,
        )

        if not order_result.success:
            logger.error(f"Order placement failed: {order_result.error}")
            await self._send_error_notification(f"Order failed: {order_result.error}")
            return order_result, None

        signal_dict = signal.to_dict()
        signal_dict["filter_result"] = filter_result.to_dict()
        signal_dict["features_snapshot"] = {
            k: v for k, v in features.items() if k in [
                "close", "atr_12", "rsi_14", "vol_std_12", "ob_imbalance_5",
                "funding_rate", "pm_up_price", "pm_model_divergence",
            ]
        }

        trade_id = self.db.record_trade(
            direction=direction,
            trade_size=trade_size,
            entry_price=order_result.fill_price,
            order_id=order_result.order_id,
            market_id=order_result.market_id,
            token_id=order_result.token_id,
            signal=signal_dict,
            is_simulated=order_result.is_simulated,
            is_premium=filter_result.is_premium,
        )

        self.risk_manager.record_trade_placed(trade_id, trade_size)

        self._pending_settlements[trade_id] = {
            "direction": direction,
            "trade_size": trade_size,
            "order_id": order_result.order_id,
            "market_id": order_result.market_id,
            "token_id": order_result.token_id,
            "placed_at": datetime.now(timezone.utc),
            "fill_price": order_result.fill_price,
            "is_simulated": order_result.is_simulated,
        }

        sim_label = "SIMULATED" if order_result.is_simulated else "LIVE"
        logger.info(
            f"Trade #{trade_id} placed: {direction} ${trade_size:.2f} "
            f"@ {order_result.fill_price:.3f} ({sim_label})"
        )

        return order_result, trade_id

    # ---------------------------------------------------------------- #
    # SETTLEMENT
    # ---------------------------------------------------------------- #

    async def _settle_pending_trades(self, current_ts: datetime):
        """Scan and settle any pending trades."""
        db_unsettled = self.db.get_unsettled_trades()
        for t in db_unsettled:
            tid = t["id"]
            if tid not in self._pending_settlements:
                self._pending_settlements[tid] = {
                    "direction": t.get("direction", "UP"),
                    "trade_size": t.get("trade_size", 1.0),
                    "order_id": t.get("order_id", ""),
                    "market_id": t.get("market_id", ""),
                    "token_id": t.get("token_id", ""),
                    "placed_at": datetime.fromisoformat(
                        t.get("created_at", current_ts.isoformat())
                    ),
                    "fill_price": t.get("entry_price", 0.5),
                    "is_simulated": t.get("is_simulated", False),
                }

        settled_ids = []
        for trade_id, info in list(self._pending_settlements.items()):
            settlement = await self._try_settle_trade(
                trade_id=trade_id,
                info=info,
                current_ts=current_ts,
            )
            if settlement:
                settled_ids.append(trade_id)
                await self._finalize_settlement(
                    trade_id=trade_id,
                    info=info,
                    settlement=settlement,
                )

        for tid in settled_ids:
            self._pending_settlements.pop(tid, None)

        if settled_ids:
            logger.info(f"Settled {len(settled_ids)} trade(s): {settled_ids}")

    async def _try_settle_trade(
        self,
        trade_id: int,
        info: Dict[str, Any],
        current_ts: datetime,
    ) -> Optional[Dict[str, Any]]:
        """Try to settle a single trade. Returns settlement dict or None."""
        placed_at = info["placed_at"]
        if isinstance(placed_at, datetime):
            age_minutes = (current_ts - placed_at).total_seconds() / 60
        else:
            age_minutes = 999

        # Simulated: settle from candle after 10 minutes
        if info.get("is_simulated") and age_minutes >= 10:
            return await self._settle_from_candle(
                direction=info["direction"],
                placed_at=placed_at,
                trade_size=info["trade_size"],
            )

        # Live: try CLOB settlement
        if not info.get("is_simulated") and self.polymarket.is_connected:
            settlement = await self.polymarket.check_settlement(
                trade_id=trade_id,
                direction=info["direction"],
                token_id=info["token_id"],
                order_id=info["order_id"],
            )
            if settlement and settlement.get("settled"):
                return settlement

        # Fallback: candle settlement after max age
        if age_minutes >= MAX_UNSETTLED_AGE_MINUTES:
            logger.info(
                f"Trade #{trade_id} age={age_minutes:.1f}m -- using candle settlement"
            )
            return await self._settle_from_candle(
                direction=info["direction"],
                placed_at=placed_at,
                trade_size=info["trade_size"],
            )

        return None

    async def _settle_from_candle(
        self,
        direction: str,
        placed_at: datetime,
        trade_size: float,
    ) -> Optional[Dict[str, Any]]:
        """Settle a trade using the candle that closed after placement."""
        try:
            placed_ts_ms = int(placed_at.timestamp() * 1000)
            candles = self.db.get_candles(
                "candles_5m",
                limit=5,
                since=placed_ts_ms,
            )
            if not candles or len(candles) < 2:
                return None

            settlement_candle = candles[1]

            result = await self.polymarket.settle_from_candle(
                direction=direction,
                candle_open=float(settlement_candle["open"]),
                candle_close=float(settlement_candle["close"]),
                trade_size=trade_size,
            )
            result["settled"] = True
            return result

        except Exception as e:
            logger.error(f"Candle settlement error: {e}")
            return None

    async def _finalize_settlement(
        self,
        trade_id: int,
        info: Dict[str, Any],
        settlement: Dict[str, Any],
    ):
        """Record settlement outcome in DB, update risk manager and filters."""
        outcome = settlement.get("outcome", "LOSS")
        pnl = settlement.get("pnl", -1.0)
        won = outcome == "WIN"
        trade_size = info["trade_size"]

        self.db.settle_trade(
            trade_id=trade_id,
            outcome=outcome,
            pnl=pnl,
            settlement_data=settlement,
        )

        self.risk_manager.record_settlement(
            trade_id=trade_id,
            won=won,
            trade_size=trade_size,
            pnl=pnl,
        )

        self.signal_filter.record_outcome(won=won)
        consecutive = self.db.get_consecutive_losses()
        self.signal_filter.update_streak_state(consecutive)

        await self._send_settlement_notification(
            trade_id=trade_id,
            direction=info["direction"],
            outcome=outcome,
            pnl=pnl,
            settlement=settlement,
        )

        actual = settlement.get("actual", "?")
        logger.info(
            f"Trade #{trade_id} settled: {outcome} PnL={pnl:+.2f} "
            f"(direction={info['direction']}, actual={actual})"
        )

    # ---------------------------------------------------------------- #
    # POLYMARKET FEATURES UPDATE
    # ---------------------------------------------------------------- #

    async def _update_polymarket_features(self, features: Dict[str, float]):
        """Fetch Polymarket market data and update feature engine."""
        try:
            market = await self.polymarket.find_current_btc_market()
            if market is None:
                return
            snapshot = await self.polymarket.get_market_snapshot(market)
            if snapshot:
                model_pred = features.get("meta_probability", None)
                self.feature_engine.update_polymarket_data(
                    market_data=snapshot,
                    model_prediction=model_pred,
                )
        except Exception as e:
            logger.debug(f"Polymarket feature update skipped: {e}")

    # ---------------------------------------------------------------- #
    # FEATURE HISTORY (for TCN)
    # ---------------------------------------------------------------- #

    def _update_feature_history(self, features: Dict[str, float]):
        """Maintain rolling feature history for TCN sequence model."""
        self._feature_history.append(dict(features))
        if len(self._feature_history) > self._feature_history_maxlen:
            self._feature_history.pop(0)

    def _get_feature_history_df(self):
        """Get feature history as a DataFrame for TCN."""
        import pandas as pd
        if len(self._feature_history) < 2:
            return None
        return pd.DataFrame(self._feature_history)

    def _get_df_5m(self):
        """Get recent 5m candle DataFrame for regime detection."""
        import pandas as pd
        rows = self.db.get_candles("candles_5m", limit=150)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = df[col].astype(float)
        return df.sort_values("timestamp").reset_index(drop=True)

    # ---------------------------------------------------------------- #
    # DAILY RESET
    # ---------------------------------------------------------------- #

    async def _check_daily_reset(self, cycle_ts: datetime):
        """Check if we have crossed midnight UTC, reset daily state if so,
        and run the 20% daily drawdown circuit breaker check every cycle.

        The drawdown check is performed here (before any trade logic) so it
        fires as soon as the threshold is crossed, not only at trade time.
        """
        # Delegate day-boundary detection and reset to the risk manager.
        # _refresh_daily_state() compares today's UTC date against the stored
        # date and calls reset_for_new_day() if they differ.
        self.risk_manager._refresh_daily_state()

        # Enforce 20% daily drawdown circuit breaker every cycle.
        # get_current_balance_estimate() returns 0.0 when no starting balance
        # is set yet (i.e. first cycle of the day), in which case
        # record_daily_drawdown_check initialises the anchor and returns early.
        current_balance = self.risk_manager.get_current_balance_estimate()
        self.risk_manager.record_daily_drawdown_check(current_balance)

    # ---------------------------------------------------------------- #
    # RETRAINING
    # ---------------------------------------------------------------- #

    async def _check_retrain_schedule(self, cycle_ts: datetime):
        """Check if the daily 4 AM UTC retrain should run."""
        retrain_hour = self.config.system.retrain_hour_utc
        today = cycle_ts.strftime("%Y-%m-%d")

        if (
            cycle_ts.hour == retrain_hour
            and cycle_ts.minute < 5
            and today != self._last_retrain_date
        ):
            logger.info(f"Starting scheduled daily retrain at {cycle_ts}")
            self._last_retrain_date = today
            await self._run_full_retrain()

    async def _run_full_retrain(self):
        """Run the full model retraining pipeline."""
        t0 = time.monotonic()
        logger.info("=== FULL RETRAIN STARTING ===")

        try:
            from src.models.trainer import Trainer

            trainer = Trainer(
                ensemble=self.ensemble,
                db=self.db,
                feature_engine=self.feature_engine,
            )

            df_5m = self._get_df_5m()
            if len(df_5m) < 500:
                logger.warning(
                    f"Insufficient training data: {len(df_5m)} candles (need 500+). "
                    "Skipping retrain."
                )
                return

            results = trainer.full_retrain()

            self.ensemble.load_models()

            elapsed = time.monotonic() - t0
            logger.info(
                f"=== FULL RETRAIN COMPLETE in {elapsed:.1f}s === Results: {results}"
            )

            if self.telegram:
                status = results.get('status', 'unknown')
                new_acc = results.get('new_accuracy', 0)
                lgbm_m = results.get('lgbm_metrics', {})
                tcn_m = results.get('tcn_metrics', {})
                logreg_m = results.get('logreg_metrics', {})
                await self.telegram.send_message(
                    f"\U0001F504 Models retrained in {elapsed:.0f}s\n"
                    f"Status: {status}\n"
                    f"Meta val acc: {new_acc:.4f}\n"
                    f"LGBM val acc: {lgbm_m.get('val_accuracy', 'N/A')}\n"
                    f"TCN val acc: {tcn_m.get('val_accuracy', 'N/A')}\n"
                    f"LogReg val acc: {logreg_m.get('val_accuracy', 'N/A')}"
                )

        except ImportError as e:
            logger.error(f"Trainer import failed: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Retrain failed: {e}", exc_info=True)
            await self._send_error_notification(f"Retrain failed: {e}")

    async def _run_incremental_update(self):
        """Run incremental model update (every 6 hours)."""
        logger.info("Running incremental model update...")
        try:
            recent_trades = self.db.get_recent_settled_trades(limit=200)
            if len(recent_trades) < 20:
                logger.debug(
                    f"Skipping incremental update: only {len(recent_trades)} settled trades."
                )
                return

            probabilities = [
                t.get("signal_confidence", 0.5)
                for t in recent_trades
                if t.get("signal_confidence") is not None
            ]
            outcomes = [
                1 if t.get("outcome") == "WIN" else 0
                for t in recent_trades
                if t.get("signal_confidence") is not None
            ]

            if len(probabilities) >= 20:
                self.signal_filter.recalibrate(probabilities, outcomes)
                logger.info(f"Incremental calibration update: {len(probabilities)} samples.")

        except Exception as e:
            logger.error(f"Incremental update error: {e}")

    # ---------------------------------------------------------------- #
    # TELEGRAM NOTIFICATIONS
    # ---------------------------------------------------------------- #

    @staticmethod
    def _make_conf_bar(confidence: float, width: int = 10) -> str:
        """Make a visual confidence bar string like '[=======   ]'."""
        filled = int(confidence * width)
        empty = width - filled
        return "[" + "=" * filled + " " * empty + "]"

    async def _send_signal_notification(
        self,
        signal: EnsembleSignal,
        filter_result: FilterResult,
        order_result: Optional[OrderResult],
        trade_id: Optional[int],
        cycle_ts: datetime,
        features: Dict[str, float],
    ):
        """Send signal card to Telegram (Section 8 format)."""
        if not self.telegram:
            return

        try:
            direction = signal.direction
            confidence = signal.confidence
            regime = signal.regime_display
            agreement = signal.agreement
            is_premium = filter_result.is_premium
            decision = filter_result.decision

            dir_icon = DIR_ICON.get(direction, "\u26AA")

            if is_premium:
                header = "\u2B50 PREMIUM SIGNAL \u2B50"
            elif decision == "TRADE":
                header = f"{dir_icon} SIGNAL: {direction}"
            else:
                header = f"\u23ED\uFE0F SKIP: {direction}"

            time_str = cycle_ts.strftime("%H:%M UTC")
            conf_pct = int(confidence * 100)
            conf_bar = self._make_conf_bar(confidence)

            verdicts_str = ""
            for name, verdict in filter_result.verdicts.items():
                icon = FILTER_ICON.get(verdict.status_str, "\u26AA")
                verdicts_str += f"  {icon} {name.title()}: {verdict.message}\n"

            lines = [
                header,
                f"\U0001F551 {time_str}",
                f"\U0001F4CA Conf: {conf_pct}% {conf_bar}",
                f"\U0001F3AF Regime: {regime}",
                f"\U0001F91D Agreement: {agreement}/3",
                "",
                "\U0001F6E1\uFE0F Filters:",
                verdicts_str.rstrip(),
            ]

            if decision == "TRADE" and order_result and order_result.success:
                sim_tag = " (SIM)" if order_result.is_simulated else ""
                lines += [
                    "",
                    f"\U0001F4B0 TRADE PLACED{sim_tag}",
                    f"  Size: ${order_result.size:.2f}",
                    f"  Price: {order_result.fill_price:.3f}",
                    f"  Trade ID: #{trade_id}",
                ]
            elif decision == "SKIP":
                lines += [
                    "",
                    f"\u23ED\uFE0F Skipped: {filter_result.skip_reason}",
                ]

            lgbm_dir = signal.lgbm.get("direction", "?")
            tcn_dir = signal.tcn.get("direction", "?")
            lr_dir = signal.logreg.get("direction", "?")
            lgbm_conf = signal.lgbm.get("confidence", 0.0)
            tcn_conf = signal.tcn.get("confidence", 0.0)
            lr_conf = signal.logreg.get("confidence", 0.0)

            lines += [
                "",
                "\U0001F916 Models:",
                f"  LGBM: {lgbm_dir} ({lgbm_conf:.0%})",
                f"  TCN:  {tcn_dir} ({tcn_conf:.0%})",
                f"  LR:   {lr_dir} ({lr_conf:.0%})",
            ]

            risk_status = self.risk_manager.get_status()
            lines += [
                "",
                "\U0001F4B3 Risk:",
                f"  Daily PnL: ${risk_status.daily_pnl:+.2f} / -${risk_status.daily_loss_limit:.0f}",
                f"  Open: ${risk_status.open_exposure:.2f} / ${risk_status.max_exposure:.0f}",
                f"  W/L today: {risk_status.wins_today}W / {risk_status.losses_today}L",
            ]

            message = "\n".join(lines)
            await self.telegram.send_message(message)

        except Exception as e:
            logger.error(f"Failed to send signal notification: {e}")

    async def _send_settlement_notification(
        self,
        trade_id: int,
        direction: str,
        outcome: str,
        pnl: float,
        settlement: Dict[str, Any],
    ):
        """Send settlement result to Telegram."""
        if not self.telegram:
            return

        try:
            is_win = outcome == "WIN"
            icon = "\U0001F7E2" if is_win else "\U0001F534"
            actual = settlement.get("actual", "?")
            move_pct = settlement.get("candle_move_pct", None)
            move_str = f" ({move_pct:+.2f}%)" if move_pct is not None else ""

            risk_status = self.risk_manager.get_status()

            msg = (
                f"{icon} SETTLED: Trade #{trade_id}\n"
                f"  Prediction: {direction}\n"
                f"  Actual: {actual}{move_str}\n"
                f"  Outcome: {outcome}\n"
                f"  PnL: ${pnl:+.2f}\n"
                f"  Daily PnL: ${risk_status.daily_pnl:+.2f}\n"
                f"  W/L: {risk_status.wins_today}W/{risk_status.losses_today}L today"
            )
            await self.telegram.send_message(msg)

        except Exception as e:
            logger.error(f"Failed to send settlement notification: {e}")

    async def _send_error_notification(self, message: str):
        """Send error alert to Telegram."""
        if not self.telegram:
            return
        try:
            await self.telegram.send_message(
                f"\u26A0\uFE0F CleoBot Error:\n{message}"
            )
        except Exception:
            pass

    # ---------------------------------------------------------------- #
    # STATS
    # ---------------------------------------------------------------- #

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            "total_cycles": self._total_cycles,
            "total_trades": self._total_trades,
            "total_skips": self._total_skips,
            "total_errors": self._total_errors,
            "pending_settlements": len(self._pending_settlements),
            "feature_history_depth": len(self._feature_history),
            "filter_state": self.signal_filter.get_state(),
            "risk_status": self.risk_manager.get_daily_stats_summary(),
            "polymarket_stats": self.polymarket.get_stats(),
        }


# ------------------------------------------------------------------ #
# FACTORY FUNCTION
# ------------------------------------------------------------------ #

def build_executor(
    config: Config,
    db: Database,
    feature_engine: FeatureEngine,
    ensemble: Ensemble,
    polymarket_client: PolymarketClient,
    telegram_bot: Optional[Any] = None,
) -> TradingExecutor:
    """Create and return a configured TradingExecutor."""
    executor = TradingExecutor(
        config=config,
        db=db,
        feature_engine=feature_engine,
        ensemble=ensemble,
        polymarket_client=polymarket_client,
        telegram_bot=telegram_bot,
    )
    logger.info("TradingExecutor built and ready.")
    return executor
