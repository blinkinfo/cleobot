"""CleoBot main entry point -- Phase 7: Full production orchestrator.

Startup sequence (Section 6):
1. Load environment variables
2. Initialize SQLite database
3. Connect to MEXC WebSocket streams
4. Backfill candle history if database has gaps
5. Load latest trained models (or trigger initial training)
6. Initialize Telegram bot
7. Initialize APScheduler with all jobs
8. Send startup notification
9. Enter main loop

Also handles:
- Graceful shutdown (SIGTERM from Railway)
- Restart recovery (detect last processed candle, resume)
- Signal-only mode (no auto-trading, just signals to Telegram)
- Health check HTTP endpoint on port 8080
"""

import asyncio
import signal
import sys
import os
import json
import time
from datetime import datetime, timezone
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, Config
from src.database import Database
from src.data.mexc_ws import MEXCWebSocketClient
from src.data.mexc_rest import MEXCRestClient
from src.data.collector import DataCollector
from src.data.backfill import DataBackfill
from src.features.engine import FeatureEngine
from src.models.ensemble import Ensemble
from src.trading.polymarket import PolymarketClient
from src.trading.executor import build_executor
from src.telegram_bot.bot import CleoBotTelegram
from src.utils.logger import setup_logger, get_logger
from src.utils.scheduler import (
    create_scheduler,
    add_trading_cycle_job,
    add_settlement_check_job,
    add_funding_rate_job,
    add_retrain_job,
    add_incremental_update_job,
    add_daily_summary_job,
)

logger = get_logger("main")

HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8080"))
LOG_PATH = os.getenv("LOG_PATH", "/data/cleobot.log")


# ------------------------------------------------------------------ #
# Health Check HTTP Server
# ------------------------------------------------------------------ #

async def start_health_server(app: "CleoBot") -> None:
    """Start a lightweight HTTP health check server on HEALTH_PORT."""
    try:
        from aiohttp import web

        async def health_handler(request):
            uptime = time.time() - app._start_ts
            status = {
                "status": "ok" if app._is_running else "starting",
                "uptime_s": round(uptime, 1),
                "auto_trade": app.auto_trade_enabled,
                "candles_5m": app.db.get_candle_count("candles_5m") if app.db else 0,
                "models_ready": app.ensemble.is_ready if app.ensemble else False,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            return web.Response(
                text=json.dumps(status),
                content_type="application/json",
            )

        async def ready_handler(request):
            return web.Response(text="OK")

        server_app = web.Application()
        server_app.router.add_get("/health", health_handler)
        server_app.router.add_get("/ready", ready_handler)
        server_app.router.add_get("/", ready_handler)

        runner = web.AppRunner(server_app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", HEALTH_PORT)
        await site.start()
        logger.info(f"Health check server running on port {HEALTH_PORT}")
    except ImportError:
        logger.warning("aiohttp not available -- health check server disabled.")
    except Exception as e:
        logger.warning(f"Health check server failed to start: {e}")


# ------------------------------------------------------------------ #
# CleoBot Application
# ------------------------------------------------------------------ #

class CleoBot:
    """Main CleoBot application orchestrator.

    Wires all components: data collection, feature engineering,
    ML models, signal generation, trading execution, and Telegram bot.

    Attributes exposed to Telegram handlers via bot_data["cleobot"]:
        - executor:       TradingExecutor
        - db:             Database
        - ensemble:       Ensemble
        - feature_engine: FeatureEngine
        - auto_trade_enabled: bool
        - config:         Config
    """

    def __init__(self):
        self.config: Optional[Config] = None
        self.db: Optional[Database] = None
        self.ws_client: Optional[MEXCWebSocketClient] = None
        self.rest_client: Optional[MEXCRestClient] = None
        self.collector: Optional[DataCollector] = None
        self.backfill: Optional[DataBackfill] = None
        self.feature_engine: Optional[FeatureEngine] = None
        self.ensemble: Optional[Ensemble] = None
        self.polymarket: Optional[PolymarketClient] = None
        self.executor = None
        self.telegram: Optional[CleoBotTelegram] = None
        self.scheduler = None

        self._shutdown_event = asyncio.Event()
        self._is_running = False
        self._start_ts = time.time()
        self._cycle_count = 0
        self._last_candle_ts: Optional[int] = None
        self._health_task: Optional[asyncio.Task] = None

    @property
    def auto_trade_enabled(self) -> bool:
        if self.executor:
            return getattr(self.executor, "_auto_trade_enabled",
                           self.config.trading.auto_trade_enabled if self.config else False)
        return self.config.trading.auto_trade_enabled if self.config else False

    # ---------------------------------------------------------------- #
    # STARTUP
    # ---------------------------------------------------------------- #

    async def start(self):
        """Execute the full 9-step startup sequence."""
        try:
            await self._startup()
            self._is_running = True
            await self._shutdown_event.wait()
        except Exception as e:
            logger.error(f"Fatal error during startup: {e}", exc_info=True)
            raise
        finally:
            await self.shutdown()

    async def _startup(self):
        logger.info("=" * 60)
        logger.info("  CleoBot Starting Up")
        logger.info("=" * 60)

        # Step 1: Load configuration
        self.config = load_config()
        setup_logger("cleobot", self.config.system.log_level)
        logger.info(
            f"Step 1/9: Config loaded | "
            f"auto_trade={self.config.trading.auto_trade_enabled} | "
            f"data_dir={self.config.system.data_dir}"
        )

        # Step 2: Database
        self.db = Database(self.config.system.db_path)
        db_stats = self.db.get_db_stats()
        n_5m = db_stats.get("candles_5m", 0)
        n_trades = db_stats.get("trades", 0)
        logger.info(f"Step 2/9: Database ready | 5m={n_5m} candles | trades={n_trades}")

        # Start health check server early so Railway doesn't kill us
        self._health_task = asyncio.create_task(start_health_server(self))

        # Step 3: MEXC data collection
        self.ws_client = MEXCWebSocketClient(symbol=self.config.mexc.symbol)
        self.rest_client = MEXCRestClient(
            symbol=self.config.mexc.symbol,
            api_key=self.config.mexc.api_key,
            secret_key=self.config.mexc.secret_key,
        )
        self.collector = DataCollector(self.db, self.ws_client, self.rest_client)
        await self.collector.start()
        logger.info("Step 3/9: MEXC data collection started")

        # Step 4: Backfill
        self.backfill = DataBackfill(self.db, self.rest_client)
        health = await self.backfill.check_data_health()
        needs_backfill = any(not info.get("sufficient") for info in health.values())
        if needs_backfill:
            logger.info("Step 4/9: Running data backfill...")
            results = await self.backfill.run_backfill()
            for interval, result in results.items():
                logger.info(f"  {interval}: {result}")
        else:
            logger.info(f"Step 4/9: Data sufficient ({n_5m} 5m candles) -- no backfill needed")

        # Ensure minimum candle count after backfill (safety net)
        ok = await self.backfill.ensure_minimum_candles(min_5m=1000)
        if not ok:
            logger.warning(
                "Step 4/9: WARNING -- could not reach 1000 5m candles after backfill. "
                "Continuing with degraded data; training may use lightweight mode."
            )

        # Restart recovery: track last processed candle
        self._last_candle_ts = self.db.get_latest_candle_timestamp("candles_5m")
        if self._last_candle_ts:
            logger.info(
                f"  Restart recovery: last 5m candle at "
                f"{datetime.fromtimestamp(self._last_candle_ts / 1000, tz=timezone.utc)}"
            )

        # Step 5: Load models
        self.feature_engine = FeatureEngine(self.db)
        self.ensemble = Ensemble(
            models_dir=self.config.system.models_dir,
            db=self.db,
        )
        self.ensemble.load_models()

        if not self.ensemble.is_ready:
            logger.info("Step 5/9: No trained models found -- running initial training...")
            await self._run_initial_training()
        else:
            logger.info("Step 5/9: Models loaded successfully")

        # Step 6: Polymarket client
        self.polymarket = PolymarketClient(self.config.polymarket, self.db)
        logger.info(
            f"Step 6/9: Polymarket client initialised "
            f"(configured={self.config.polymarket.is_configured})"
        )

        # Step 7: Build executor
        self.telegram = CleoBotTelegram(self.config.telegram)
        self.executor = build_executor(
            config=self.config,
            db=self.db,
            feature_engine=self.feature_engine,
            ensemble=self.ensemble,
            polymarket_client=self.polymarket,
            telegram_bot=self.telegram,
        )
        # Wire auto_trade flag from config
        self.executor._auto_trade_enabled = self.config.trading.auto_trade_enabled
        logger.info(
            f"Step 7/9: Trading executor built | "
            f"auto_trade={self.config.trading.auto_trade_enabled}"
        )

        # Initialize executor: load feature history from DB + pre-populate ATR
        await self.executor.initialize()
        logger.info("Step 7/9: Executor initialized (feature history + ATR pre-populated from DB)")

        # Step 8: Telegram bot
        await self.telegram.start(
            cleobot_app=self,
            db_path=self.config.system.db_path,
            log_path=LOG_PATH,
        )
        logger.info(
            f"Step 8/9: Telegram bot started "
            f"(configured={self.config.telegram.is_configured})"
        )

        # Step 9: APScheduler
        self.scheduler = create_scheduler()
        add_trading_cycle_job(self.scheduler, self._trading_cycle)
        add_settlement_check_job(self.scheduler, self._settlement_check)
        add_funding_rate_job(self.scheduler, self.collector.fetch_funding_rate)
        add_retrain_job(
            self.scheduler,
            self._full_retrain,
            hour_utc=self.config.system.retrain_hour_utc,
        )
        add_incremental_update_job(self.scheduler, self._incremental_update)
        add_daily_summary_job(self.scheduler, self._daily_summary)
        self.scheduler.start()
        logger.info("Step 9/9: APScheduler started with all 6 jobs")

        # Send startup notification
        mode = "AUTO-TRADE" if self.config.trading.auto_trade_enabled else "SIGNALS ONLY"
        startup_msg = (
            f"\U0001F916 CleoBot Online\n"
            f"Mode: {mode}\n"
            f"5m candles: {self.db.get_candle_count('candles_5m')}\n"
            f"Models ready: {self.ensemble.is_ready}\n"
            f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        await self.telegram.send_message(startup_msg)
        logger.info("=" * 60)
        logger.info("  CleoBot is RUNNING")
        logger.info("=" * 60)

    # ---------------------------------------------------------------- #
    # TRADING CYCLE
    # ---------------------------------------------------------------- #

    async def _trading_cycle(self):
        """Execute one full 5-minute trading cycle.

        Called at :02, :07, :12, ... of every 5-minute slot.
        Delegates to TradingExecutor which handles features -> signal -> filters -> trade.
        """
        if not self.executor:
            logger.warning("Trading cycle called before executor is ready -- skipping.")
            return

        self._cycle_count += 1
        now = datetime.now(timezone.utc)
        candle_ts_ms = int(now.timestamp() * 1000)

        try:
            # Get latest orderbook snapshot for the executor
            current_orderbook = self.collector.get_latest_orderbook() if self.collector else None

            # Honour signal-only mode: override auto-trade flag
            if not self.config.trading.auto_trade_enabled:
                if hasattr(self.executor, "_auto_trade_enabled"):
                    self.executor._auto_trade_enabled = False

            result = await self.executor.run_cycle(
                candle_ts_ms=candle_ts_ms,
                current_orderbook=current_orderbook,
            )

            # Cache latest signal for Telegram handlers
            if result.signal and self.telegram:
                self.telegram.cache_signal(result.signal.to_dict())

            logger.debug(
                f"Cycle #{self._cycle_count} done | "
                f"trade={result.trade_placed} | "
                f"dur={result.duration_s:.2f}s"
            )

        except Exception as e:
            logger.error(f"Trading cycle #{self._cycle_count} error: {e}", exc_info=True)
            if self.telegram:
                await self.telegram.send_message(f"\u26A0\uFE0F Cycle error: {e}")

    async def _settlement_check(self):
        """Check and settle pending trades.

        Called at :00:05, :05:05, :10:05, ... (5s after candle close).
        The executor's _settle_pending_trades handles the actual logic.
        """
        if not self.executor:
            return
        try:
            now = datetime.now(timezone.utc)
            await self.executor._settle_pending_trades(now)
        except Exception as e:
            logger.error(f"Settlement check error: {e}", exc_info=True)

    # ---------------------------------------------------------------- #
    # RETRAINING
    # ---------------------------------------------------------------- #

    async def _run_initial_training(self):
        """Run initial model training when no models exist."""
        try:
            from src.models.trainer import Trainer

            logger.info("Running initial model training...")
            candles = self.db.get_candles("candles_5m", limit=5000)
            if len(candles) < 200:
                logger.warning(
                    f"Only {len(candles)} candles available for initial training. "
                    "Need 200+. Training deferred."
                )
                return

            trainer = Trainer(
                ensemble=self.ensemble,
                db=self.db,
                feature_engine=self.feature_engine,
            )
            results = trainer.initial_training()
            self.ensemble.load_models()
            logger.info(f"Initial training complete: {results}")
            if self.telegram:
                await self.telegram.send_message(
                    f"\u2705 Initial training complete\n"
                    f"Models: {list(results.keys()) if isinstance(results, dict) else 'done'}"
                )
        except ImportError as e:
            logger.error(f"Trainer import failed: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Initial training failed: {e}", exc_info=True)

    async def _full_retrain(self):
        """Run daily full model retrain (04:00 UTC)."""
        if not self.executor:
            return
        try:
            logger.info("=== Scheduled daily retrain starting ===")
            if self.telegram:
                await self.telegram.send_message("\U0001F504 Daily retrain starting...")
            await self.executor._run_full_retrain()
            ts = datetime.now(timezone.utc).isoformat()
            if self.telegram:
                self.telegram.record_retrain_ts(ts)
        except Exception as e:
            logger.error(f"Full retrain scheduler error: {e}", exc_info=True)

    async def _incremental_update(self):
        """Run 6-hourly incremental model update."""
        if not self.executor:
            return
        try:
            await self.executor._run_incremental_update()
        except Exception as e:
            logger.error(f"Incremental update error: {e}", exc_info=True)

    # ---------------------------------------------------------------- #
    # DAILY SUMMARY
    # ---------------------------------------------------------------- #

    async def _daily_summary(self):
        """Send daily summary at 00:00:30 UTC."""
        if not self.telegram:
            return
        try:
            stats = self.db.get_trade_stats_today()
            total = stats.get("total_trades", 0)
            wins = stats.get("wins", 0)
            losses = stats.get("losses", 0)
            pnl = stats.get("pnl", 0.0)
            accuracy = stats.get("accuracy", 0.0)

            # Save session stats
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self.db.update_session_stats(
                date=today,
                trades_count=stats.get("total_trades", 0),
                wins=wins,
                losses=losses,
                skips=0,
                pnl=pnl,
                accuracy=accuracy,
            )

            # Cleanup old data
            self.db.cleanup_old_data(candle_days=30, orderbook_days=7)

            icon = "\U0001F4CA"
            msg = (
                f"{icon} Daily Summary -- {today}\n"
                f"Trades: {total} | W{wins}/L{losses}\n"
                f"Accuracy: {accuracy:.1%}\n"
                f"P&L: ${pnl:+.2f}\n"
            )
            if self.executor:
                risk_stats = self.executor.risk_manager.get_daily_stats_summary()
                msg += f"Daily PnL: ${risk_stats.get('daily_pnl', 0):+.2f}\n"

            await self.telegram.send_message(msg)
            logger.info(f"Daily summary sent: {total} trades, {accuracy:.1%} accuracy, ${pnl:+.2f} P&L")
        except Exception as e:
            logger.error(f"Daily summary error: {e}", exc_info=True)

    # ---------------------------------------------------------------- #
    # SHUTDOWN
    # ---------------------------------------------------------------- #

    async def shutdown(self):
        """Graceful shutdown sequence."""
        if not self._is_running and self._cycle_count == 0:
            # Called before fully started
            logger.info("Shutdown called before fully started -- cleaning up.")

        logger.info("=" * 60)
        logger.info("  CleoBot Shutting Down")
        logger.info("=" * 60)
        self._is_running = False

        # Stop scheduler (no new jobs fire)
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped.")

        # Send shutdown notification
        if self.telegram:
            try:
                uptime_s = time.time() - self._start_ts
                h, m = divmod(int(uptime_s // 60), 60)
                await self.telegram.send_message(
                    f"\U0001F6D1 CleoBot shutting down\n"
                    f"Uptime: {h}h {m}m\n"
                    f"Cycles: {self._cycle_count}"
                )
            except Exception:
                pass
            await self.telegram.stop()
            logger.info("Telegram bot stopped.")

        # Stop data collection
        if self.collector:
            await self.collector.stop()
            logger.info("Data collector stopped.")

        # Close database
        if self.db:
            self.db.close()
            logger.info("Database closed.")

        logger.info("Shutdown complete.")

    def _handle_signal(self, sig):
        """Handle OS signal for graceful shutdown."""
        logger.info(f"Received {sig.name} -- initiating shutdown...")
        self._shutdown_event.set()


# ------------------------------------------------------------------ #
# ENTRY POINT
# ------------------------------------------------------------------ #

def main():
    """Main entry point for CleoBot."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    bot = CleoBot()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, bot._handle_signal, sig)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda s, f: bot._handle_signal(signal.Signals(s)))

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if not loop.is_closed():
            loop.close()


if __name__ == "__main__":
    main()
