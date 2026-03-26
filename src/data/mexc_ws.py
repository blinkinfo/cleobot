"""MEXC WebSocket client for real-time market data streams.

Connects to MEXC WebSocket API for:
- kline_5m: 5-minute candles (OHLCV)
- kline_15m: 15-minute candles
- kline_1h: 1-hour candles
- depth_20: Orderbook snapshots (top 20 levels)
- trade: Individual trades
"""

import asyncio
import json
import time
from typing import Callable, Dict, Optional, Any, List
from datetime import datetime, timezone

import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

from src.utils.logger import get_logger

logger = get_logger("data.mexc_ws")

# MEXC WebSocket endpoints
MEXC_WS_URL = "wss://wbs.mexc.com/ws"
MEXC_FUTURES_WS_URL = "wss://contract.mexc.com/edge"

# Ping interval to keep connection alive
PING_INTERVAL = 20  # seconds
RECONNECT_DELAY_BASE = 1  # seconds
RECONNECT_DELAY_MAX = 60  # seconds


class MEXCWebSocketClient:
    """Async WebSocket client for MEXC market data.
    
    Handles connection management, automatic reconnection,
    and dispatching messages to registered callbacks.
    """

    def __init__(self, symbol: str = "BTCUSDT"):
        """Initialize the WebSocket client.
        
        Args:
            symbol: Trading pair symbol (default BTCUSDT).
        """
        self.symbol = symbol
        self.symbol_lower = symbol.lower()
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_count = 0
        self._last_message_time = 0.0
        self._tasks: List[asyncio.Task] = []

        # Callbacks for different data types
        self._callbacks: Dict[str, List[Callable]] = {
            "kline_5m": [],
            "kline_15m": [],
            "kline_1h": [],
            "depth": [],
            "trade": [],
        }

        # Connection stats
        self.stats = {
            "messages_received": 0,
            "reconnections": 0,
            "errors": 0,
            "connected_since": None,
        }

    def on_kline_5m(self, callback: Callable):
        """Register callback for 5-minute kline updates.
        
        Callback receives: (timestamp_ms, open, high, low, close, volume, is_closed)
        """
        self._callbacks["kline_5m"].append(callback)

    def on_kline_15m(self, callback: Callable):
        """Register callback for 15-minute kline updates."""
        self._callbacks["kline_15m"].append(callback)

    def on_kline_1h(self, callback: Callable):
        """Register callback for 1-hour kline updates."""
        self._callbacks["kline_1h"].append(callback)

    def on_depth(self, callback: Callable):
        """Register callback for orderbook depth updates.
        
        Callback receives: (timestamp_ms, bids, asks, mid_price, spread)
        """
        self._callbacks["depth"].append(callback)

    def on_trade(self, callback: Callable):
        """Register callback for individual trade updates.
        
        Callback receives: (timestamp_ms, price, quantity, is_buyer_maker)
        """
        self._callbacks["trade"].append(callback)

    async def connect(self):
        """Connect to MEXC WebSocket and start receiving data."""
        self._running = True
        self._reconnect_count = 0

        while self._running:
            try:
                await self._connect_and_listen()
            except (ConnectionClosed, ConnectionClosedError) as e:
                if not self._running:
                    break
                self.stats["errors"] += 1
                delay = min(
                    RECONNECT_DELAY_BASE * (2 ** self._reconnect_count),
                    RECONNECT_DELAY_MAX,
                )
                logger.warning(
                    f"WebSocket connection closed: {e}. "
                    f"Reconnecting in {delay}s (attempt {self._reconnect_count + 1})..."
                )
                await asyncio.sleep(delay)
                self._reconnect_count += 1
                self.stats["reconnections"] += 1
            except Exception as e:
                if not self._running:
                    break
                self.stats["errors"] += 1
                delay = min(
                    RECONNECT_DELAY_BASE * (2 ** self._reconnect_count),
                    RECONNECT_DELAY_MAX,
                )
                logger.error(
                    f"WebSocket error: {e}. "
                    f"Reconnecting in {delay}s (attempt {self._reconnect_count + 1})..."
                )
                await asyncio.sleep(delay)
                self._reconnect_count += 1
                self.stats["reconnections"] += 1

    async def _connect_and_listen(self):
        """Establish connection, subscribe to streams, and listen for messages."""
        logger.info(f"Connecting to MEXC WebSocket at {MEXC_WS_URL}...")

        async with websockets.connect(
            MEXC_WS_URL,
            ping_interval=PING_INTERVAL,
            ping_timeout=10,
            close_timeout=5,
            max_size=10 * 1024 * 1024,  # 10MB max message size
        ) as ws:
            self.ws = ws
            self.stats["connected_since"] = datetime.now(timezone.utc).isoformat()
            self._reconnect_count = 0  # Reset on successful connection
            logger.info("WebSocket connected successfully.")

            # Subscribe to all required streams
            await self._subscribe(ws)

            # Start ping task
            ping_task = asyncio.create_task(self._ping_loop(ws))
            self._tasks.append(ping_task)

            try:
                async for message in ws:
                    if not self._running:
                        break
                    self._last_message_time = time.time()
                    self.stats["messages_received"] += 1
                    await self._handle_message(message)
            finally:
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass

    async def _subscribe(self, ws):
        """Subscribe to all required data streams.
        
        Streams per Section 6:
        - kline_5m, kline_15m, kline_1h
        - depth_20 (orderbook)
        - trade
        """
        subscriptions = [
            f"spot@public.kline.v3.api@{self.symbol}@Min5",
            f"spot@public.kline.v3.api@{self.symbol}@Min15",
            f"spot@public.kline.v3.api@{self.symbol}@Min60",
            f"spot@public.bookTicker.v3.api@{self.symbol}",
            f"spot@public.deals.v3.api@{self.symbol}",
        ]

        for sub in subscriptions:
            msg = {
                "method": "SUBSCRIPTION",
                "params": [sub],
            }
            await ws.send(json.dumps(msg))
            logger.info(f"Subscribed to: {sub}")

        # Also subscribe to partial depth for full orderbook snapshots
        depth_sub = {
            "method": "SUBSCRIPTION",
            "params": [f"spot@public.increase.depth.v3.api@{self.symbol}"],
        }
        await ws.send(json.dumps(depth_sub))
        logger.info(f"Subscribed to: depth stream for {self.symbol}")

    async def _ping_loop(self, ws):
        """Send periodic pings to keep the connection alive."""
        while self._running:
            try:
                await asyncio.sleep(PING_INTERVAL)
                if ws.open:
                    await ws.send(json.dumps({"method": "PING"}))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Ping failed: {e}")

    async def _handle_message(self, raw_message: str):
        """Parse and dispatch a WebSocket message to appropriate callbacks."""
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.debug(f"Non-JSON message received: {raw_message[:100]}")
            return

        # Handle PONG responses
        if data.get("msg") == "PONG" or data.get("code") == 0:
            return

        # Handle subscription confirmations
        if "code" in data and data.get("code") == 0:
            return

        channel = data.get("c", "")
        d = data.get("d", {})

        if not channel or not d:
            return

        try:
            # Route to appropriate handler based on channel
            if "kline" in channel and "Min5" in channel:
                await self._handle_kline(d, "kline_5m")
            elif "kline" in channel and "Min15" in channel:
                await self._handle_kline(d, "kline_15m")
            elif "kline" in channel and "Min60" in channel:
                await self._handle_kline(d, "kline_1h")
            elif "bookTicker" in channel or "depth" in channel:
                await self._handle_depth(d)
            elif "deals" in channel:
                await self._handle_trade(d)
        except Exception as e:
            logger.error(f"Error handling message on channel '{channel}': {e}")
            self.stats["errors"] += 1

    async def _handle_kline(self, data: Dict[str, Any], kline_type: str):
        """Handle kline (candle) update messages.
        
        Args:
            data: Kline data from WebSocket.
            kline_type: Type key ('kline_5m', 'kline_15m', 'kline_1h').
        """
        try:
            k = data.get("k", data)
            timestamp_ms = int(k.get("t", k.get("T", 0)))
            open_price = float(k.get("o", 0))
            high_price = float(k.get("h", 0))
            low_price = float(k.get("l", 0))
            close_price = float(k.get("c", 0))
            volume = float(k.get("v", k.get("a", 0)))
            
            # MEXC uses different field for candle close status
            is_closed = bool(k.get("T", 0))  # T = close time, present when candle is closed

            for callback in self._callbacks[kline_type]:
                if asyncio.iscoroutinefunction(callback):
                    await callback(timestamp_ms, open_price, high_price, low_price,
                                   close_price, volume, is_closed)
                else:
                    callback(timestamp_ms, open_price, high_price, low_price,
                             close_price, volume, is_closed)

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing kline data ({kline_type}): {e} | data: {data}")

    async def _handle_depth(self, data: Dict[str, Any]):
        """Handle orderbook depth update messages.
        
        Args:
            data: Depth data from WebSocket.
        """
        try:
            bids = data.get("bids", data.get("b", []))
            asks = data.get("asks", data.get("a", []))

            if not bids and not asks:
                return

            # Parse price/quantity pairs
            parsed_bids = []
            for bid in bids[:20]:  # Top 20 levels
                if isinstance(bid, dict):
                    parsed_bids.append([float(bid.get("p", 0)), float(bid.get("v", 0))])
                elif isinstance(bid, (list, tuple)):
                    parsed_bids.append([float(bid[0]), float(bid[1])])

            parsed_asks = []
            for ask in asks[:20]:  # Top 20 levels
                if isinstance(ask, dict):
                    parsed_asks.append([float(ask.get("p", 0)), float(ask.get("v", 0))])
                elif isinstance(ask, (list, tuple)):
                    parsed_asks.append([float(ask[0]), float(ask[1])])

            if parsed_bids and parsed_asks:
                best_bid = parsed_bids[0][0]
                best_ask = parsed_asks[0][0]
                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid

                timestamp_ms = int(data.get("t", data.get("E", time.time() * 1000)))

                for callback in self._callbacks["depth"]:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(timestamp_ms, parsed_bids, parsed_asks, mid_price, spread)
                    else:
                        callback(timestamp_ms, parsed_bids, parsed_asks, mid_price, spread)

        except (KeyError, ValueError, TypeError, IndexError) as e:
            logger.error(f"Error parsing depth data: {e}")

    async def _handle_trade(self, data: Dict[str, Any]):
        """Handle individual trade messages.
        
        Args:
            data: Trade data from WebSocket.
        """
        try:
            trades = data.get("deals", [data]) if isinstance(data, dict) else [data]
            for trade in trades:
                timestamp_ms = int(trade.get("t", trade.get("T", time.time() * 1000)))
                price = float(trade.get("p", 0))
                quantity = float(trade.get("v", trade.get("q", 0)))
                # MEXC: S=1 is sell (maker is buyer), S=2 is buy
                is_buyer_maker = trade.get("S", 0) == 1

                for callback in self._callbacks["trade"]:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(timestamp_ms, price, quantity, is_buyer_maker)
                    else:
                        callback(timestamp_ms, price, quantity, is_buyer_maker)

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing trade data: {e}")

    async def disconnect(self):
        """Gracefully disconnect from WebSocket."""
        logger.info("Disconnecting WebSocket client...")
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

        if self.ws and not self.ws.closed:
            await self.ws.close()
            logger.info("WebSocket connection closed.")

        self.ws = None

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is currently connected."""
        return self.ws is not None and not self.ws.closed

    @property
    def seconds_since_last_message(self) -> float:
        """Get seconds since last message was received."""
        if self._last_message_time == 0:
            return float("inf")
        return time.time() - self._last_message_time

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            **self.stats,
            "is_connected": self.is_connected,
            "seconds_since_last_message": round(self.seconds_since_last_message, 1),
            "reconnect_count": self._reconnect_count,
        }
