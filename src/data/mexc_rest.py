"""MEXC REST API client for historical data and funding rates.

Used for:
- Historical candle backfill on startup (Section 6)
- Funding rate polling every 60 seconds (Section 6)
- Fallback data source when WebSocket has gaps
"""

import asyncio
import time
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta

import aiohttp

from src.utils.logger import get_logger

logger = get_logger("data.mexc_rest")

# MEXC API base URL
MEXC_BASE_URL = "https://api.mexc.com"

# Rate limiting
REQUEST_DELAY = 0.2  # 200ms between requests to avoid rate limits
MAX_RETRIES = 3

# Kline interval mappings
INTERVAL_MAP = {
    "5m": "5m",
    "15m": "15m",
    "1h": "60m",
    "1H": "60m",
    "60m": "60m",
}

# Max candles per request
MAX_CANDLES_PER_REQUEST = 1000


class MEXCRestClient:
    """Async REST client for MEXC API.
    
    Handles historical candle data retrieval and funding rate polling.
    Includes rate limiting, retry logic, and error handling.
    """

    def __init__(self, symbol: str = "BTCUSDT", api_key: str = "", secret_key: str = ""):
        """Initialize the REST client.
        
        Args:
            symbol: Trading pair symbol.
            api_key: MEXC API key (optional for public endpoints).
            secret_key: MEXC API secret (optional for public endpoints).
        """
        self.symbol = symbol
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = MEXC_BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0.0

        # Stats
        self.stats = {
            "requests_made": 0,
            "errors": 0,
            "candles_fetched": 0,
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {}
            if self.api_key:
                headers["X-MEXC-APIKEY"] = self.api_key
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            await asyncio.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    async def _request(self, method: str, endpoint: str, 
                       params: Optional[Dict] = None) -> Optional[Any]:
        """Make an API request with retries and error handling.
        
        Args:
            method: HTTP method ('GET', 'POST').
            endpoint: API endpoint path.
            params: Query parameters.
        
        Returns:
            Parsed JSON response, or None on failure.
        """
        url = f"{self.base_url}{endpoint}"
        session = await self._get_session()

        for attempt in range(MAX_RETRIES):
            try:
                await self._rate_limit()
                self.stats["requests_made"] += 1

                async with session.request(method, url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        # Rate limited -- back off
                        retry_after = int(response.headers.get("Retry-After", 5))
                        logger.warning(
                            f"Rate limited on {endpoint}. Waiting {retry_after}s..."
                        )
                        await asyncio.sleep(retry_after)
                        continue
                    else:
                        text = await response.text()
                        logger.error(
                            f"API error {response.status} on {endpoint}: {text}"
                        )
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(1 * (attempt + 1))
                        continue

            except aiohttp.ClientError as e:
                self.stats["errors"] += 1
                logger.error(f"Request error on {endpoint} (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                continue
            except Exception as e:
                self.stats["errors"] += 1
                logger.error(f"Unexpected error on {endpoint}: {e}")
                break

        return None

    async def get_klines(self, interval: str = "5m", limit: int = 500,
                         start_time: Optional[int] = None,
                         end_time: Optional[int] = None) -> List[Tuple]:
        """Fetch historical kline (candle) data.
        
        Args:
            interval: Candle interval ('5m', '15m', '1h').
            limit: Number of candles to fetch (max 1000).
            start_time: Start timestamp in milliseconds.
            end_time: End timestamp in milliseconds.
        
        Returns:
            List of (timestamp_ms, open, high, low, close, volume) tuples.
        """
        mexc_interval = INTERVAL_MAP.get(interval, interval)
        params = {
            "symbol": self.symbol,
            "interval": mexc_interval,
            "limit": min(limit, MAX_CANDLES_PER_REQUEST),
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = await self._request("GET", "/api/v3/klines", params=params)

        if not data or not isinstance(data, list):
            logger.error(f"Failed to fetch klines for {interval}")
            return []

        candles = []
        for item in data:
            try:
                candles.append((
                    int(item[0]),        # Open time (ms)
                    float(item[1]),      # Open
                    float(item[2]),      # High
                    float(item[3]),      # Low
                    float(item[4]),      # Close
                    float(item[5]),      # Volume
                ))
            except (IndexError, ValueError, TypeError) as e:
                logger.debug(f"Skipping malformed kline entry: {e}")
                continue

        self.stats["candles_fetched"] += len(candles)
        return candles

    async def get_klines_range(self, interval: str, start_time: int,
                               end_time: Optional[int] = None) -> List[Tuple]:
        """Fetch all candles within a time range, handling pagination.
        
        Makes multiple API calls if the range exceeds the per-request limit.
        
        Args:
            interval: Candle interval ('5m', '15m', '1h').
            start_time: Start timestamp in milliseconds.
            end_time: End timestamp in milliseconds (default: now).
        
        Returns:
            List of (timestamp_ms, open, high, low, close, volume) tuples.
        """
        if end_time is None:
            end_time = int(time.time() * 1000)

        # Calculate interval in ms for pagination
        interval_ms_map = {
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "60m": 60 * 60 * 1000,
        }
        interval_ms = interval_ms_map.get(interval, 5 * 60 * 1000)

        all_candles = []
        current_start = start_time

        while current_start < end_time:
            batch = await self.get_klines(
                interval=interval,
                limit=MAX_CANDLES_PER_REQUEST,
                start_time=current_start,
                end_time=end_time,
            )

            if not batch:
                break

            all_candles.extend(batch)

            # Move start time to after the last candle
            last_timestamp = batch[-1][0]
            if last_timestamp <= current_start:
                break  # Prevent infinite loop
            current_start = last_timestamp + interval_ms

            logger.debug(
                f"Fetched {len(batch)} candles ({interval}). "
                f"Total so far: {len(all_candles)}"
            )

        # Deduplicate by timestamp
        seen = set()
        unique_candles = []
        for c in all_candles:
            if c[0] not in seen:
                seen.add(c[0])
                unique_candles.append(c)

        logger.info(
            f"Fetched {len(unique_candles)} total candles ({interval}) "
            f"from {datetime.fromtimestamp(start_time / 1000, tz=timezone.utc).isoformat()} "
            f"to {datetime.fromtimestamp(end_time / 1000, tz=timezone.utc).isoformat()}"
        )
        return sorted(unique_candles, key=lambda x: x[0])

    async def get_funding_rate(self) -> Optional[Dict[str, Any]]:
        """Fetch the current funding rate for the symbol.
        
        Returns:
            Dict with 'rate', 'timestamp', and 'next_settlement', or None on failure.
        """
        # MEXC futures funding rate endpoint
        params = {"symbol": f"{self.symbol}_USDT"}

        data = await self._request(
            "GET", "/api/v3/ticker/price", params={"symbol": self.symbol}
        )

        # Try the futures API for funding rate
        funding_data = None
        try:
            session = await self._get_session()
            await self._rate_limit()
            async with session.get(
                f"https://contract.mexc.com/api/v1/contract/funding_rate/{self.symbol}",
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success") and result.get("data"):
                        fd = result["data"]
                        funding_data = {
                            "rate": float(fd.get("fundingRate", 0)),
                            "timestamp": int(time.time() * 1000),
                            "next_settlement": int(fd.get("nextSettleTime", 0)),
                        }
        except Exception as e:
            logger.debug(f"Funding rate fetch from futures API failed: {e}")

        # Fallback: return zero rate if futures API unavailable
        if funding_data is None:
            funding_data = {
                "rate": 0.0,
                "timestamp": int(time.time() * 1000),
                "next_settlement": None,
            }
            logger.debug("Using fallback zero funding rate.")

        return funding_data

    async def get_ticker(self) -> Optional[Dict[str, Any]]:
        """Fetch current ticker data.
        
        Returns:
            Dict with price, volume, and change data, or None.
        """
        data = await self._request(
            "GET", "/api/v3/ticker/24hr", params={"symbol": self.symbol}
        )
        if data:
            return {
                "price": float(data.get("lastPrice", 0)),
                "volume_24h": float(data.get("volume", 0)),
                "change_24h": float(data.get("priceChangePercent", 0)),
                "high_24h": float(data.get("highPrice", 0)),
                "low_24h": float(data.get("lowPrice", 0)),
            }
        return None

    async def get_orderbook(self, limit: int = 20) -> Optional[Dict[str, Any]]:
        """Fetch current orderbook snapshot via REST.
        
        Fallback for when WebSocket orderbook data is unavailable.
        
        Args:
            limit: Number of orderbook levels (5, 10, 20, or 50).
        
        Returns:
            Dict with bids, asks, mid_price, spread, or None.
        """
        data = await self._request(
            "GET", "/api/v3/depth",
            params={"symbol": self.symbol, "limit": limit},
        )

        if not data:
            return None

        try:
            bids = [[float(b[0]), float(b[1])] for b in data.get("bids", [])]
            asks = [[float(a[0]), float(a[1])] for a in data.get("asks", [])]

            if bids and asks:
                mid_price = (bids[0][0] + asks[0][0]) / 2
                spread = asks[0][0] - bids[0][0]
            else:
                mid_price = 0.0
                spread = 0.0

            return {
                "bids": bids,
                "asks": asks,
                "mid_price": mid_price,
                "spread": spread,
                "timestamp": int(time.time() * 1000),
            }
        except (IndexError, ValueError, TypeError) as e:
            logger.error(f"Error parsing orderbook: {e}")
            return None

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("MEXC REST client session closed.")

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {**self.stats}
