"""CleoBot data collection orchestrator.

Coordinates data collection from MEXC WebSocket and REST API,
stores data to SQLite database. Manages:
- Real-time candle data (5m, 15m, 1h) via WebSocket
- Orderbook snapshots via WebSocket
- Trade data via WebSocket
- Funding rate polling via REST (every 60s)
- Periodic REST fallback for orderbook snapshots
"""

import asyncio
import time
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from src.data.mexc_ws import MEXCWebSocketClient
from src.data.mexc_rest import MEXCRestClient
from src.database import Database
from src.utils.logger import get_logger
from src.utils.helpers import utc_now, utc_timestamp_ms

logger = get_logger("data.collector")


class DataCollector:
    """Orchestrates all data collection for CleoBot.
    
    Connects WebSocket callbacks to database storage and manages
    the real-time data pipeline.
    """

    def __init__(self, db: Database, ws_client: MEXCWebSocketClient,
                 rest_client: MEXCRestClient):
        """Initialize the data collector.
        
        Args:
            db: Database instance for storage.
            ws_client: MEXC WebSocket client for real-time data.
            rest_client: MEXC REST client for polling and fallback.
        """
        self.db = db
        self.ws = ws_client
        self.rest = rest_client
        self._running = False
        self._ws_task: Optional[asyncio.Task] = None
        self._orderbook_poll_task: Optional[asyncio.Task] = None

        # Track latest data for feature engine access
        self._latest_orderbook: Optional[Dict[str, Any]] = None
        self._latest_ticker_price: float = 0.0
        self._last_orderbook_save_time: float = 0.0
        self._orderbook_save_interval: float = 5.0  # Save orderbook every 5 seconds

        # Deduplication: track last *closed* candle timestamp per timeframe
        # to prevent reconnection backlogs from re-inserting the same candle.
        self._last_closed_ts: Dict[str, int] = {
            "candles_5m": 0,
            "candles_15m": 0,
            "candles_1h": 0,
        }

        # Stats
        self.stats = {
            "candles_5m_received": 0,
            "candles_15m_received": 0,
            "candles_1h_received": 0,
            "orderbook_snapshots_saved": 0,
            "trades_received": 0,
            "funding_rates_fetched": 0,
            "errors": 0,
        }

        # Register WebSocket callbacks
        self._register_callbacks()

    def _register_callbacks(self):
        """Register all WebSocket callbacks for data storage."""
        self.ws.on_kline_5m(self._on_kline_5m)
        self.ws.on_kline_15m(self._on_kline_15m)
        self.ws.on_kline_1h(self._on_kline_1h)
        self.ws.on_depth(self._on_depth)
        self.ws.on_trade(self._on_trade)
        logger.info("WebSocket callbacks registered.")

    async def start(self):
        """Start all data collection processes."""
        logger.info("Starting data collector...")
        self._running = True

        # Start WebSocket connection in background
        self._ws_task = asyncio.create_task(self._run_websocket())

        # Start periodic orderbook REST polling as fallback
        self._orderbook_poll_task = asyncio.create_task(self._orderbook_poll_loop())

        logger.info("Data collector started.")

    async def stop(self):
        """Stop all data collection and close connections."""
        logger.info("Stopping data collector...")
        self._running = False

        # Disconnect WebSocket
        await self.ws.disconnect()

        # Cancel tasks
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self._orderbook_poll_task:
            self._orderbook_poll_task.cancel()
            try:
                await self._orderbook_poll_task
            except asyncio.CancelledError:
                pass

        # Close REST session
        await self.rest.close()

        logger.info("Data collector stopped.")

    async def _run_websocket(self):
        """Run the WebSocket connection with auto-reconnect."""
        try:
            await self.ws.connect()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"WebSocket task error: {e}")
            self.stats["errors"] += 1

    async def _on_kline_5m(self, timestamp_ms: int, open_: float, high: float,
                            low: float, close: float, volume: float, is_closed: bool):
        """Handle 5-minute kline updates.
        
        Only persist finalized (closed) candles to the database to prevent
        the feature engine from reading stale intermediate ticks.
        In-progress ticks still update the live price tracker.
        """
        try:
            # Always track latest price from every tick
            self._latest_ticker_price = close
            self.stats["candles_5m_received"] += 1

            if is_closed:
                # Dedup guard: skip if we already saved this candle timestamp
                if timestamp_ms <= self._last_closed_ts["candles_5m"]:
                    return
                self.db.insert_candle("candles_5m", timestamp_ms, open_, high, low, close, volume)
                self._last_closed_ts["candles_5m"] = timestamp_ms
                logger.debug(
                    f"5m candle closed: {datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime('%H:%M')} "
                    f"O={open_:.2f} H={high:.2f} L={low:.2f} C={close:.2f} V={volume:.4f}"
                )
        except Exception as e:
            logger.error(f"Error saving 5m kline: {e}")
            self.stats["errors"] += 1

    async def _on_kline_15m(self, timestamp_ms: int, open_: float, high: float,
                             low: float, close: float, volume: float, is_closed: bool):
        """Handle 15-minute kline updates. Only persist closed candles."""
        try:
            self.stats["candles_15m_received"] += 1
            if is_closed:
                if timestamp_ms <= self._last_closed_ts["candles_15m"]:
                    return
                self.db.insert_candle("candles_15m", timestamp_ms, open_, high, low, close, volume)
                self._last_closed_ts["candles_15m"] = timestamp_ms
        except Exception as e:
            logger.error(f"Error saving 15m kline: {e}")
            self.stats["errors"] += 1

    async def _on_kline_1h(self, timestamp_ms: int, open_: float, high: float,
                            low: float, close: float, volume: float, is_closed: bool):
        """Handle 1-hour kline updates. Only persist closed candles."""
        try:
            self.stats["candles_1h_received"] += 1
            if is_closed:
                if timestamp_ms <= self._last_closed_ts["candles_1h"]:
                    return
                self.db.insert_candle("candles_1h", timestamp_ms, open_, high, low, close, volume)
                self._last_closed_ts["candles_1h"] = timestamp_ms
        except Exception as e:
            logger.error(f"Error saving 1h kline: {e}")
            self.stats["errors"] += 1

    async def _on_depth(self, timestamp_ms: int, bids: list, asks: list,
                        mid_price: float, spread: float):
        """Handle orderbook depth updates.
        
        Saves orderbook snapshots at a controlled interval (every 5 seconds)
        to avoid overwhelming the database while still capturing granular data.
        """
        try:
            # Always update in-memory latest orderbook
            self._latest_orderbook = {
                "timestamp": timestamp_ms,
                "bids": bids,
                "asks": asks,
                "mid_price": mid_price,
                "spread": spread,
            }
            self._latest_ticker_price = mid_price

            # Save to database at controlled interval
            now = time.time()
            if now - self._last_orderbook_save_time >= self._orderbook_save_interval:
                self.db.insert_orderbook_snapshot(
                    timestamp_ms, bids[:20], asks[:20], mid_price, spread
                )
                self._last_orderbook_save_time = now
                self.stats["orderbook_snapshots_saved"] += 1

        except Exception as e:
            logger.error(f"Error handling depth update: {e}")
            self.stats["errors"] += 1

    async def _on_trade(self, timestamp_ms: int, price: float, quantity: float,
                        is_buyer_maker: bool):
        """Handle individual trade updates."""
        try:
            self._latest_ticker_price = price
            self.stats["trades_received"] += 1
        except Exception as e:
            logger.error(f"Error handling trade: {e}")
            self.stats["errors"] += 1

    async def _orderbook_poll_loop(self):
        """Periodically fetch orderbook via REST as a fallback.
        
        Runs every 10 seconds. If WebSocket orderbook is stale (>30s old),
        uses REST data instead.
        """
        while self._running:
            try:
                await asyncio.sleep(10)
                if not self._running:
                    break

                # Check if WebSocket orderbook is fresh
                if self._latest_orderbook:
                    age = time.time() - (self._latest_orderbook["timestamp"] / 1000)
                    if age < 30:
                        continue  # WebSocket data is fresh, no need for REST

                # Fetch via REST
                ob = await self.rest.get_orderbook(limit=20)
                if ob and ob.get("bids") and ob.get("asks"):
                    self._latest_orderbook = ob
                    self.db.insert_orderbook_snapshot(
                        ob["timestamp"], ob["bids"], ob["asks"],
                        ob["mid_price"], ob["spread"],
                    )
                    self.stats["orderbook_snapshots_saved"] += 1
                    logger.debug("Orderbook updated via REST fallback.")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Orderbook REST poll error: {e}")
                self.stats["errors"] += 1

    async def fetch_funding_rate(self):
        """Fetch and store the latest funding rate.
        
        Called by the scheduler every 60 seconds.
        """
        try:
            data = await self.rest.get_funding_rate()
            if data:
                self.db.insert_funding_rate(
                    timestamp=data["timestamp"],
                    rate=data["rate"],
                    next_settlement=data.get("next_settlement"),
                )
                self.stats["funding_rates_fetched"] += 1
                logger.debug(f"Funding rate saved: {data['rate']:.6f}")
        except Exception as e:
            logger.error(f"Error fetching funding rate: {e}")
            self.stats["errors"] += 1

    def get_latest_orderbook(self) -> Optional[Dict[str, Any]]:
        """Get the most recent orderbook data (in-memory).
        
        Returns:
            Latest orderbook dict or None.
        """
        return self._latest_orderbook

    def get_latest_price(self) -> float:
        """Get the most recent BTC price.
        
        Returns:
            Latest price from any data source.
        """
        return self._latest_ticker_price

    def get_stats(self) -> Dict[str, Any]:
        """Get data collection statistics."""
        return {
            **self.stats,
            "ws_stats": self.ws.get_stats(),
            "rest_stats": self.rest.get_stats(),
            "latest_price": self._latest_ticker_price,
            "has_orderbook": self._latest_orderbook is not None,
        }
