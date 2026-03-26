"""CleoBot main entry point.

Implements the startup sequence from Section 6:
1. Load environment variables
2. Initialize SQLite database (create tables if not exist)
3. Connect to MEXC WebSocket streams
4. Backfill candle history (REST API) if database has gaps
5. Load latest trained models from disk (or trigger initial training if none exist)
6. Initialize Telegram bot
7. Initialize APScheduler with 5-minute cycle jobs
8. Send startup notification to Telegram
9. Enter main loop

Also handles:
- Graceful shutdown (SIGTERM from Railway)
- Restart recovery (detect last processed candle, resume from next slot)
"""

import asyncio
import signal
import sys
import os
from datetime import datetime, timezone

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, Config
from src.database import Database
from src.data.mexc_ws import MEXCWebSocketClient
from src.data.mexc_rest import MEXCRestClient
from src.data.collector import DataCollector
from src.data.backfill import DataBackfill
from src.utils.logger import setup_logger, get_logger
from src.utils.scheduler import (
    create_scheduler,
    add_trading_cycle_job,
    add_settlement_check_job,
    add_funding_rate_job,
)

logger = get_logger("main")


class CleoBot:
    """Main CleoBot application.
    
    Orchestrates all components: data collection, feature engineering,
    ML models, signal generation, trading, and Telegram bot.
    """

    def __init__(self):
        """Initialize CleoBot (components created in start())."""
        self.config: Config = None
        self.db: Database = None
        self.ws_client: MEXCWebSocketClient = None
        self.rest_client: MEXCRestClient = None
        self.collector: DataCollector = None
        self.backfill: DataBackfill = None
        self.scheduler = None
        self._shutdown_event = asyncio.Event()
        self._is_running = False

    async def start(self):
        """Execute the full startup sequence."""
        try:
            # Step 1: Load environment variables
            logger.info("=" * 60)
            logger.info("  CleoBot Starting Up")
            logger.info("=" * 60)
            
            self.config = load_config()
            root_logger = setup_logger("cleobot", self.config.system.log_level)
            logger.info(f"Step 1/9: Configuration loaded (log_level={self.config.system.log_level})")
            logger.info(f"  Data directory: {self.config.system.data_dir}")
            logger.info(f"  MEXC configured: {self.config.mexc.is_configured}")
            logger.info(f"  Telegram configured: {self.config.telegram.is_configured}")
            logger.info(f"  Polymarket configured: {self.config.polymarket.is_configured}")
            logger.info(f"  Auto-trade enabled: {self.config.trading.auto_trade_enabled}")

            # Step 2: Initialize SQLite database
            self.db = Database(self.config.system.db_path)
            logger.info(f"Step 2/9: Database initialized at {self.config.system.db_path}")
            db_stats = self.db.get_db_stats()
            for table, count in db_stats.items():
                if count > 0:
                    logger.info(f"  {table}: {count} records")

            # Step 3: Connect to MEXC WebSocket streams
            self.ws_client = MEXCWebSocketClient(symbol=self.config.mexc.symbol)
            self.rest_client = MEXCRestClient(
                symbol=self.config.mexc.symbol,
                api_key=self.config.mexc.api_key,
                secret_key=self.config.mexc.secret_key,
            )
            self.collector = DataCollector(self.db, self.ws_client, self.rest_client)
            await self.collector.start()
            logger.info("Step 3/9: MEXC data collection started (WebSocket + REST)")

            # Step 4: Backfill candle history if database has gaps
            self.backfill = DataBackfill(self.db, self.rest_client)
            logger.info("Step 4/9: Checking data and running backfill if needed...")
            health_before = await self.backfill.check_data_health()
            for interval, info in health_before.items():
                logger.info(
                    f"  {interval}: {info['count']} candles, "
                    f"sufficient={info['sufficient']}"
                )

            # Run backfill for intervals that need it
            needs_backfill = any(
                not info["sufficient"] for info in health_before.values()
            )
            if needs_backfill:
                logger.info("  Running backfill...")
                backfill_results = await self.backfill.run_backfill()
                for interval, result in backfill_results.items():
                    logger.info(f"  {interval} backfill: {result}")
            else:
                logger.info("  All candle data is sufficient. No backfill needed.")

            # Step 5: Load latest trained models (Phase 3 -- placeholder for now)
            logger.info("Step 5/9: Model loading (will be implemented in Phase 3)")
            # Models will be loaded here once Phase 3 is complete.
            # For now, the bot runs in signal-collection mode.

            # Step 6: Initialize Telegram bot (Phase 5 -- placeholder for now)
            logger.info("Step 6/9: Telegram bot initialization (will be implemented in Phase 5)")
            # Telegram bot will be initialized here once Phase 5 is complete.

            # Step 7: Initialize APScheduler with cycle jobs
            self.scheduler = create_scheduler()

            # Add trading cycle job (runs at :02, :07, :12, etc.)
            add_trading_cycle_job(self.scheduler, self._trading_cycle)

            # Add settlement check job (runs at :00, :05, :10, etc.)
            add_settlement_check_job(self.scheduler, self._settlement_check)

            # Add funding rate polling (every 60 seconds)
            add_funding_rate_job(self.scheduler, self.collector.fetch_funding_rate)

            self.scheduler.start()
            logger.info("Step 7/9: APScheduler started with trading cycle and funding rate jobs")

            # Step 8: Send startup notification (Telegram placeholder)
            startup_msg = (
                f"CleoBot started at "
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                f"Mode: {'AUTO-TRADE' if self.config.trading.auto_trade_enabled else 'SIGNALS ONLY'}\n"
                f"Data: {self.db.get_candle_count('candles_5m')} 5m candles loaded"
            )
            logger.info(f"Step 8/9: Startup notification: {startup_msg}")
            # Will send via Telegram once Phase 5 is complete.

            # Step 9: Enter main loop
            self._is_running = True
            logger.info("Step 9/9: Entering main loop. CleoBot is running!")
            logger.info("=" * 60)

            # Keep running until shutdown signal
            await self._shutdown_event.wait()

        except Exception as e:
            logger.error(f"Fatal error during startup: {e}", exc_info=True)
            raise
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown sequence.
        
        On shutdown signal (SIGTERM from Railway):
        - Close WebSocket connections
        - Flush pending DB writes
        - Stop scheduler
        - Send Telegram notification
        """
        if not self._is_running:
            return

        logger.info("=" * 60)
        logger.info("  CleoBot Shutting Down")
        logger.info("=" * 60)

        self._is_running = False

        # Stop scheduler first
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped.")

        # Stop data collection (closes WebSocket and REST)
        if self.collector:
            await self.collector.stop()
            logger.info("Data collector stopped.")

        # Close database
        if self.db:
            self.db.close()
            logger.info("Database closed.")

        # Send shutdown notification (Telegram placeholder)
        logger.info(
            f"CleoBot shut down at "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        logger.info("Shutdown complete.")

    async def _trading_cycle(self):
        """Execute one trading cycle.
        
        Called every 5 minutes at :02 of each slot.
        Full implementation comes in Phase 4 (filters, execution).
        For now, logs the cycle trigger.
        """
        now = datetime.now(timezone.utc)
        logger.info(
            f"Trading cycle triggered at {now.strftime('%H:%M:%S')} UTC. "
            f"(Full pipeline coming in Phase 4)"
        )

        # Log current data status
        stats = self.collector.get_stats()
        logger.debug(
            f"Data status: 5m={stats['candles_5m_received']}, "
            f"15m={stats['candles_15m_received']}, "
            f"1h={stats['candles_1h_received']}, "
            f"ob_saved={stats['orderbook_snapshots_saved']}, "
            f"price={stats['latest_price']:.2f}"
        )

    async def _settlement_check(self):
        """Check settlement for pending trades.
        
        Called every 5 minutes at :00 of each slot.
        Full implementation comes in Phase 4.
        """
        # Check for unsettled trades
        unsettled = self.db.get_unsettled_trades()
        if unsettled:
            logger.debug(f"Settlement check: {len(unsettled)} unsettled trades pending.")

    def _handle_signal(self, sig):
        """Handle OS signals for graceful shutdown."""
        logger.info(f"Received signal {sig.name}. Initiating shutdown...")
        self._shutdown_event.set()


def main():
    """Main entry point for CleoBot."""
    # Set up event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    bot = CleoBot()

    # Register signal handlers for graceful shutdown
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, bot._handle_signal, sig)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f: bot._handle_signal(signal.Signals(s)))

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
    finally:
        # Ensure cleanup
        if bot._is_running:
            loop.run_until_complete(bot.shutdown())
        loop.close()


if __name__ == "__main__":
    main()
