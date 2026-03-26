"""Polymarket CLOB client for CleoBot.

Handles all Polymarket interactions:
  - Finding the current/next 5-min BTC Up/Down market
  - Placing limit orders (UP or DOWN)
  - Checking fill status
  - Monitoring settlement
  - Recording outcomes to the database

Polymarket BTC 5-min markets are structured as binary options:
  - Each market has a YES token (BTC goes UP) and NO token (BTC goes DOWN)
  - We buy the YES token when predicting UP, NO token when predicting DOWN
  - Markets settle at 1.0 (win) or 0.0 (loss)

The client uses py-clob-client for CLOB interactions.
Graceful fallback: if Polymarket is not configured or unreachable,
all methods return safe defaults and log warnings.
"""

import time
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple

from src.config import PolymarketConfig
from src.database import Database
from src.utils.logger import get_logger

logger = get_logger("trading.polymarket")

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

# Polymarket CLOB API endpoints
CLOB_HOST = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"

# BTC 5-min market search terms
BTC_MARKET_KEYWORDS = ["BTC", "bitcoin", "5 minute", "5-minute", "5min"]

# Order timeout: cancel and retry after this many seconds
ORDER_TIMEOUT_SECONDS = 25

# Fill poll interval
FILL_POLL_INTERVAL = 1.0

# Maximum slippage allowed (in probability units)
MAX_SLIPPAGE = 0.05  # 5 cents on a $1 bet

# Settlement poll interval for pending trades
SETTLEMENT_POLL_INTERVAL = 30  # seconds

# Max retries for API calls
MAX_RETRIES = 3
RETRY_DELAY = 2.0


# ------------------------------------------------------------------ #
# Data Structures
# ------------------------------------------------------------------ #

class OrderResult:
    """Result from placing a Polymarket order."""

    def __init__(
        self,
        success: bool,
        order_id: str = "",
        direction: str = "",
        size: float = 0.0,
        price: float = 0.0,
        fill_price: float = 0.0,
        fill_time_s: float = 0.0,
        market_id: str = "",
        token_id: str = "",
        error: str = "",
        is_simulated: bool = False,
    ):
        self.success = success
        self.order_id = order_id
        self.direction = direction
        self.size = size
        self.price = price           # Requested price
        self.fill_price = fill_price # Actual fill price
        self.fill_time_s = fill_time_s
        self.market_id = market_id
        self.token_id = token_id
        self.error = error
        self.is_simulated = is_simulated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "order_id": self.order_id,
            "direction": self.direction,
            "size": self.size,
            "price": self.price,
            "fill_price": self.fill_price,
            "fill_time_s": self.fill_time_s,
            "market_id": self.market_id,
            "token_id": self.token_id,
            "error": self.error,
            "is_simulated": self.is_simulated,
        }


class MarketInfo:
    """Information about a Polymarket BTC 5-min market."""

    def __init__(
        self,
        condition_id: str,
        question: str,
        start_time: datetime,
        end_time: datetime,
        yes_token_id: str,
        no_token_id: str,
        yes_price: float,
        no_price: float,
        volume: float = 0.0,
        active: bool = True,
    ):
        self.condition_id = condition_id
        self.question = question
        self.start_time = start_time
        self.end_time = end_time
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        self.yes_price = yes_price  # Current UP odds (0-1)
        self.no_price = no_price    # Current DOWN odds (0-1)
        self.volume = volume
        self.active = active

    def get_token_for_direction(self, direction: str) -> Tuple[str, float]:
        """Get the token ID and current price for a direction.

        Args:
            direction: 'UP' or 'DOWN'.

        Returns:
            Tuple of (token_id, current_price).
        """
        if direction == "UP":
            return self.yes_token_id, self.yes_price
        else:
            return self.no_token_id, self.no_price

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "question": self.question,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "yes_token_id": self.yes_token_id,
            "no_token_id": self.no_token_id,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "volume": self.volume,
            "active": self.active,
        }


# ------------------------------------------------------------------ #
# Polymarket Client
# ------------------------------------------------------------------ #

class PolymarketClient:
    """Polymarket CLOB client for CleoBot.

    Wraps py-clob-client with:
      - BTC 5-min market discovery
      - Order placement with fill confirmation
      - Settlement monitoring
      - Graceful fallback when not configured
    """

    def __init__(self, config: PolymarketConfig, db: Database):
        """Initialise the Polymarket client.

        Args:
            config: Polymarket API credentials.
            db: Database for recording outcomes.
        """
        self.config = config
        self.db = db
        self._client = None          # py-clob-client ClobClient instance
        self._is_connected = False
        self._last_market: Optional[MarketInfo] = None
        self._market_cache_ts: float = 0.0
        self._market_cache_ttl: float = 30.0  # seconds

        # Stats
        self._orders_placed: int = 0
        self._orders_filled: int = 0
        self._orders_failed: int = 0

    # ---------------------------------------------------------------- #
    # CONNECTION
    # ---------------------------------------------------------------- #

    async def connect(self) -> bool:
        """Connect to Polymarket CLOB API.

        Returns:
            True if connected successfully, False otherwise.
        """
        if not self.config.is_configured:
            logger.warning(
                "Polymarket not configured (missing API credentials). "
                "Running in simulation mode."
            )
            return False

        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            creds = ApiCreds(
                api_key=self.config.api_key,
                api_secret=self.config.api_secret,
                api_passphrase=self.config.api_passphrase,
            )
            self._client = ClobClient(
                host=CLOB_HOST,
                creds=creds,
            )
            # Verify connection with a lightweight call
            _ = self._client.get_ok()
            self._is_connected = True
            logger.info("Polymarket CLOB connected successfully.")
            return True

        except ImportError:
            logger.warning(
                "py-clob-client not installed. Running in simulation mode."
            )
            return False
        except Exception as e:
            logger.error(f"Polymarket connection failed: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._client is not None

    # ---------------------------------------------------------------- #
    # MARKET DISCOVERY
    # ---------------------------------------------------------------- #

    async def find_current_btc_market(
        self,
        target_candle_start: Optional[datetime] = None,
    ) -> Optional[MarketInfo]:
        """Find the current or next 5-min BTC Up/Down market.

        Searches for an active Polymarket market that covers the
        target_candle_start time. If not connected, returns None.

        Args:
            target_candle_start: The candle we are predicting (default: next slot).

        Returns:
            MarketInfo if found, None otherwise.
        """
        if not self.is_connected:
            logger.debug("Polymarket not connected: cannot find market.")
            return None

        # Use cache if recent
        now = time.monotonic()
        if (
            self._last_market is not None
            and now - self._market_cache_ts < self._market_cache_ttl
        ):
            return self._last_market

        try:
            market = await asyncio.get_event_loop().run_in_executor(
                None, self._find_market_sync, target_candle_start
            )
            if market:
                self._last_market = market
                self._market_cache_ts = now
            return market

        except Exception as e:
            logger.error(f"Failed to find BTC market: {e}")
            return None

    def _find_market_sync(self, target_time: Optional[datetime]) -> Optional[MarketInfo]:
        """Synchronous market search (runs in executor)."""
        try:
            # Query gamma API for active markets
            import httpx

            params = {
                "active": "true",
                "closed": "false",
                "limit": "100",
                "tag": "crypto",
            }
            resp = httpx.get(
                f"{GAMMA_HOST}/markets",
                params=params,
                timeout=10.0,
            )
            resp.raise_for_status()
            markets = resp.json()

            if isinstance(markets, dict):
                markets = markets.get("data", markets.get("markets", []))

            # Filter for BTC 5-min markets
            for m in markets:
                question = m.get("question", "").lower()
                if not (
                    "btc" in question or "bitcoin" in question
                ):
                    continue
                if not (
                    "5 min" in question
                    or "5min" in question
                    or "5-min" in question
                ):
                    continue

                # Parse token IDs
                tokens = m.get("tokens", m.get("outcomes", []))
                yes_token = ""
                no_token = ""
                yes_price = 0.5
                no_price = 0.5

                for t in tokens:
                    outcome = t.get("outcome", "").lower()
                    tid = t.get("token_id", t.get("id", ""))
                    price = float(t.get("price", 0.5))
                    if outcome in ("yes", "up"):
                        yes_token = tid
                        yes_price = price
                    elif outcome in ("no", "down"):
                        no_token = tid
                        no_price = price

                # Parse times
                start_str = m.get("startDate", m.get("start_date", ""))
                end_str = m.get("endDate", m.get("end_date", ""))
                try:
                    start_time = datetime.fromisoformat(
                        start_str.replace("Z", "+00:00")
                    )
                    end_time = datetime.fromisoformat(
                        end_str.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    continue

                return MarketInfo(
                    condition_id=m.get("conditionId", m.get("condition_id", "")),
                    question=m.get("question", ""),
                    start_time=start_time,
                    end_time=end_time,
                    yes_token_id=yes_token,
                    no_token_id=no_token,
                    yes_price=yes_price,
                    no_price=no_price,
                    volume=float(m.get("volume", 0.0)),
                    active=True,
                )

        except Exception as e:
            logger.error(f"Market search error: {e}")
        return None

    async def get_market_odds(self, market: MarketInfo) -> Tuple[float, float]:
        """Get current YES/NO prices for a market.

        Args:
            market: MarketInfo instance.

        Returns:
            Tuple of (yes_price, no_price).
        """
        if not self.is_connected:
            return 0.5, 0.5

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._get_orderbook_mid_sync,
                market.yes_token_id,
                market.no_token_id,
            )
            return result
        except Exception as e:
            logger.error(f"Failed to get market odds: {e}")
            return market.yes_price, market.no_price

    def _get_orderbook_mid_sync(
        self,
        yes_token_id: str,
        no_token_id: str,
    ) -> Tuple[float, float]:
        """Get mid prices from the order books."""
        try:
            yes_book = self._client.get_order_book(yes_token_id)
            yes_price = self._extract_mid_price(yes_book)

            no_book = self._client.get_order_book(no_token_id)
            no_price = self._extract_mid_price(no_book)

            return yes_price, no_price
        except Exception as e:
            logger.error(f"Orderbook fetch error: {e}")
            return 0.5, 0.5

    def _extract_mid_price(self, order_book: Any) -> float:
        """Extract mid price from a Polymarket order book object."""
        try:
            if hasattr(order_book, "bids") and hasattr(order_book, "asks"):
                bids = order_book.bids
                asks = order_book.asks
                if bids and asks:
                    best_bid = float(bids[0].price) if hasattr(bids[0], "price") else float(bids[0][0])
                    best_ask = float(asks[0].price) if hasattr(asks[0], "price") else float(asks[0][0])
                    return (best_bid + best_ask) / 2.0
            return 0.5
        except Exception:
            return 0.5

    # ---------------------------------------------------------------- #
    # ORDER PLACEMENT
    # ---------------------------------------------------------------- #

    async def place_order(
        self,
        direction: str,
        size: float,
        market: MarketInfo,
    ) -> OrderResult:
        """Place a limit order on Polymarket.

        Places a limit order at the current best price and waits for fill.
        If not filled within ORDER_TIMEOUT_SECONDS, cancels and returns failure.

        Args:
            direction: 'UP' or 'DOWN'.
            size: Trade size in USD.
            market: MarketInfo for the target market.

        Returns:
            OrderResult with fill details.
        """
        if not self.is_connected:
            logger.warning("Polymarket not connected: simulating order.")
            return self._simulate_order(direction, size, market)

        token_id, current_price = market.get_token_for_direction(direction)

        if not token_id:
            return OrderResult(
                success=False,
                direction=direction,
                size=size,
                error="No token ID found for direction",
            )

        # Use current price as limit price (add 1 tick buffer to ensure fill)
        # In Polymarket, prices range 0-1 with 0.01 tick size
        limit_price = min(0.99, current_price + 0.01)

        logger.info(
            f"Placing {direction} order: size=${size:.2f}, "
            f"price={limit_price:.3f}, market={market.condition_id[:16]}..."
        )

        t_start = time.monotonic()

        try:
            order_result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._place_order_sync,
                token_id,
                size,
                limit_price,
            )

            if order_result.get("error"):
                self._orders_failed += 1
                return OrderResult(
                    success=False,
                    direction=direction,
                    size=size,
                    price=limit_price,
                    market_id=market.condition_id,
                    token_id=token_id,
                    error=str(order_result.get("error", "Unknown error")),
                )

            order_id = order_result.get("orderID", order_result.get("id", ""))
            self._orders_placed += 1

            # Poll for fill
            fill_price, fill_time = await self._wait_for_fill(
                order_id, limit_price, t_start
            )

            elapsed = time.monotonic() - t_start
            self._orders_filled += 1

            logger.info(
                f"{direction} order filled: price={fill_price:.3f}, "
                f"fill_time={elapsed:.1f}s, order_id={order_id}"
            )

            return OrderResult(
                success=True,
                order_id=order_id,
                direction=direction,
                size=size,
                price=limit_price,
                fill_price=fill_price,
                fill_time_s=elapsed,
                market_id=market.condition_id,
                token_id=token_id,
            )

        except asyncio.TimeoutError:
            logger.warning(f"{direction} order timed out after {ORDER_TIMEOUT_SECONDS}s")
            self._orders_failed += 1
            await self._cancel_order_safe(token_id)
            return OrderResult(
                success=False,
                direction=direction,
                size=size,
                price=limit_price,
                market_id=market.condition_id,
                token_id=token_id,
                error=f"Order timed out after {ORDER_TIMEOUT_SECONDS}s",
            )
        except Exception as e:
            logger.error(f"Order placement error: {e}", exc_info=True)
            self._orders_failed += 1
            return OrderResult(
                success=False,
                direction=direction,
                size=size,
                error=str(e),
            )

    def _place_order_sync(
        self,
        token_id: str,
        size: float,
        price: float,
    ) -> Dict[str, Any]:
        """Synchronous order placement via py-clob-client."""
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType

            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side="BUY",
            )
            resp = self._client.create_and_post_order(order_args)

            # Normalise response
            if hasattr(resp, "orderID"):
                return {"orderID": resp.orderID, "status": resp.status}
            elif isinstance(resp, dict):
                return resp
            else:
                return {"orderID": str(resp), "status": "LIVE"}

        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return {"error": str(e)}

    async def _wait_for_fill(
        self,
        order_id: str,
        limit_price: float,
        t_start: float,
    ) -> Tuple[float, float]:
        """Poll for order fill until timeout.

        Args:
            order_id: Order ID to poll.
            limit_price: Expected price (used as default if fill price unavailable).
            t_start: Time order was placed (monotonic).

        Returns:
            Tuple of (fill_price, fill_time_s).

        Raises:
            asyncio.TimeoutError if not filled within ORDER_TIMEOUT_SECONDS.
        """
        deadline = t_start + ORDER_TIMEOUT_SECONDS

        while time.monotonic() < deadline:
            await asyncio.sleep(FILL_POLL_INTERVAL)

            try:
                status = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._get_order_status_sync,
                    order_id,
                )

                order_status = status.get("status", "").upper()

                if order_status in ("MATCHED", "FILLED"):
                    fill_price = float(status.get("price", limit_price))
                    fill_time = time.monotonic() - t_start
                    return fill_price, fill_time

                elif order_status in ("CANCELLED", "CANCELED", "EXPIRED"):
                    logger.warning(f"Order {order_id} was cancelled/expired")
                    return limit_price, time.monotonic() - t_start

                # Still LIVE -- keep polling

            except Exception as e:
                logger.warning(f"Error polling order {order_id}: {e}")
                await asyncio.sleep(FILL_POLL_INTERVAL)

        raise asyncio.TimeoutError(f"Order {order_id} not filled within {ORDER_TIMEOUT_SECONDS}s")

    def _get_order_status_sync(self, order_id: str) -> Dict[str, Any]:
        """Get order status from CLOB."""
        try:
            resp = self._client.get_order(order_id)
            if hasattr(resp, "status"):
                return {
                    "status": resp.status,
                    "price": getattr(resp, "price", 0.5),
                }
            elif isinstance(resp, dict):
                return resp
            return {"status": "LIVE"}
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {"status": "LIVE"}

    async def _cancel_order_safe(self, order_id: str):
        """Cancel an order, ignoring errors."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._client.cancel,
                order_id,
            )
            logger.debug(f"Cancelled order {order_id}")
        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id}: {e}")

    # ---------------------------------------------------------------- #
    # SETTLEMENT MONITORING
    # ---------------------------------------------------------------- #

    async def check_settlement(
        self,
        trade_id: int,
        direction: str,
        token_id: str,
        order_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Check if a trade has settled.

        Args:
            trade_id: Database trade ID.
            direction: 'UP' or 'DOWN'.
            token_id: The token we bought.
            order_id: The order ID.

        Returns:
            Dict with 'settled', 'outcome' ('WIN'/'LOSS'), 'pnl' if settled.
            None if not yet settled.
        """
        if not self.is_connected:
            return None

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._check_settlement_sync,
                token_id,
                order_id,
            )
            return result
        except Exception as e:
            logger.error(f"Settlement check error for trade #{trade_id}: {e}")
            return None

    def _check_settlement_sync(
        self,
        token_id: str,
        order_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Synchronous settlement check."""
        try:
            # Get trade history to see if position is settled
            trades = self._client.get_trades(
                params={"maker_order_id": order_id}
            )

            if not trades:
                return None

            # Check each trade for settlement
            for trade in trades:
                status = getattr(trade, "status", "") or ""
                if status.upper() in ("CONFIRMED", "SETTLED"):
                    # Determine win/loss from final price
                    final_price = float(getattr(trade, "price", 0.0))
                    if final_price >= 0.99:
                        outcome = "WIN"
                        pnl = WIN_PNL
                    elif final_price <= 0.01:
                        outcome = "LOSS"
                        pnl = LOSS_PNL
                    else:
                        return None  # Still settling

                    return {
                        "settled": True,
                        "outcome": outcome,
                        "pnl": pnl,
                        "final_price": final_price,
                    }

            return None

        except Exception as e:
            logger.error(f"Settlement sync error: {e}")
            return None

    async def settle_from_candle(
        self,
        direction: str,
        candle_open: float,
        candle_close: float,
        trade_size: float = 1.0,
    ) -> Dict[str, Any]:
        """Determine settlement outcome from candle prices.

        Used as a fallback when CLOB settlement check fails or when
        Polymarket API is unavailable. The candle close vs open determines
        whether the prediction was correct.

        Args:
            direction: 'UP' or 'DOWN' (our prediction).
            candle_open: Open price of the target candle.
            candle_close: Close price of the target candle.
            trade_size: Trade size for PnL calculation.

        Returns:
            Dict with 'outcome', 'pnl', 'candle_move_pct'.
        """
        candle_up = candle_close > candle_open
        predicted_correctly = (
            (direction == "UP" and candle_up)
            or (direction == "DOWN" and not candle_up)
        )

        outcome = "WIN" if predicted_correctly else "LOSS"
        # Scale PnL by trade size (base is $1)
        pnl = (WIN_PNL if predicted_correctly else LOSS_PNL) * trade_size

        move_pct = (candle_close - candle_open) / max(candle_open, 1e-8) * 100

        return {
            "outcome": outcome,
            "pnl": pnl,
            "candle_open": candle_open,
            "candle_close": candle_close,
            "candle_move_pct": move_pct,
            "predicted": direction,
            "actual": "UP" if candle_up else "DOWN",
        }

    # ---------------------------------------------------------------- #
    # SIMULATION (no credentials)
    # ---------------------------------------------------------------- #

    def _simulate_order(
        self,
        direction: str,
        size: float,
        market: Optional[MarketInfo],
    ) -> OrderResult:
        """Simulate an order for paper-trading mode."""
        simulated_price = 0.52 if direction == "UP" else 0.48
        simulated_fill_time = 1.2

        logger.info(
            f"[SIMULATED] {direction} order: size=${size:.2f}, "
            f"price={simulated_price:.3f}"
        )

        return OrderResult(
            success=True,
            order_id=f"sim_{int(time.time())}",
            direction=direction,
            size=size,
            price=simulated_price,
            fill_price=simulated_price,
            fill_time_s=simulated_fill_time,
            market_id=market.condition_id if market else "simulated",
            token_id="simulated",
            is_simulated=True,
        )

    # ---------------------------------------------------------------- #
    # MARKET DATA (for Polymarket features)
    # ---------------------------------------------------------------- #

    async def get_market_snapshot(
        self,
        market: Optional[MarketInfo] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get a snapshot of current market data for feature engineering.

        Returns data in the format expected by PolymarketFeatures:
          up_price, down_price, volume, odds_velocity, etc.

        Args:
            market: MarketInfo (fetches current market if None).

        Returns:
            Market snapshot dict or None.
        """
        if market is None:
            market = await self.find_current_btc_market()

        if market is None:
            return None

        yes_price, no_price = await self.get_market_odds(market)

        return {
            "up_price": yes_price,
            "down_price": no_price,
            "volume": market.volume,
            "condition_id": market.condition_id,
            "end_time": market.end_time.isoformat(),
        }

    # ---------------------------------------------------------------- #
    # STATS
    # ---------------------------------------------------------------- #

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "is_connected": self.is_connected,
            "orders_placed": self._orders_placed,
            "orders_filled": self._orders_filled,
            "orders_failed": self._orders_failed,
            "fill_rate": (
                self._orders_filled / max(self._orders_placed, 1) * 100
            ),
        }


# ------------------------------------------------------------------ #
# Constants (referenced by RiskManager for PnL)
# ------------------------------------------------------------------ #

WIN_PNL = 0.88
LOSS_PNL = -1.00
