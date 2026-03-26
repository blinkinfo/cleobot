"""CleoBot historical data backfill.

Backfills candle history from MEXC REST API on startup
when the database has gaps or insufficient data.

Per Section 6: retain at minimum 30 days of candle data.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.data.mexc_rest import MEXCRestClient
from src.database import Database
from src.utils.logger import get_logger
from src.utils.helpers import utc_now, utc_timestamp_ms

logger = get_logger("data.backfill")

# Minimum data requirements
MIN_CANDLE_DAYS = 14  # Minimum 14 days for training (Section 8)
TARGET_CANDLE_DAYS = 30  # Target 30 days per Section 6
CANDLE_INTERVALS = {
    "5m": ("candles_5m", 5 * 60 * 1000),
    "15m": ("candles_15m", 15 * 60 * 1000),
    "1h": ("candles_1h", 60 * 60 * 1000),
}


class DataBackfill:
    """Handles historical data backfill from MEXC REST API.
    
    On startup, checks each candle table for data gaps and
    fetches missing history.
    """

    def __init__(self, db: Database, rest_client: MEXCRestClient):
        """Initialize the backfill module.
        
        Args:
            db: Database instance.
            rest_client: MEXC REST client for fetching historical data.
        """
        self.db = db
        self.rest = rest_client

    async def run_backfill(self, days: int = TARGET_CANDLE_DAYS) -> dict:
        """Run the full backfill process for all candle intervals.
        
        Args:
            days: Number of days of history to ensure we have.
        
        Returns:
            Dict with backfill results per interval.
        """
        logger.info(f"Starting data backfill (target: {days} days of history)...")
        results = {}

        for interval, (table, interval_ms) in CANDLE_INTERVALS.items():
            try:
                result = await self._backfill_interval(interval, table, interval_ms, days)
                results[interval] = result
            except Exception as e:
                logger.error(f"Error backfilling {interval}: {e}")
                results[interval] = {"status": "error", "error": str(e)}

        logger.info(f"Backfill complete. Results: {results}")
        return results

    async def _backfill_interval(self, interval: str, table: str,
                                  interval_ms: int, days: int) -> dict:
        """Backfill a single candle interval.
        
        Args:
            interval: Candle interval string ('5m', '15m', '1h').
            table: Database table name.
            interval_ms: Interval duration in milliseconds.
            days: Number of days to backfill.
        
        Returns:
            Dict with backfill status and counts.
        """
        now_ms = utc_timestamp_ms()
        target_start_ms = now_ms - (days * 24 * 60 * 60 * 1000)

        # Check what we already have
        latest_ts = self.db.get_latest_candle_timestamp(table)
        count_before = self.db.get_candle_count(table)

        if latest_ts and latest_ts > target_start_ms:
            # We have some data -- check if we have enough
            expected_candles = (days * 24 * 60 * 60 * 1000) // interval_ms
            if count_before >= expected_candles * 0.9:  # 90% threshold
                logger.info(
                    f"  {interval}: Already have {count_before} candles "
                    f"(>= 90% of {expected_candles} expected). Filling gaps only."
                )
                # Just fill gaps from latest to now
                gap_start = latest_ts + interval_ms
                if gap_start < now_ms:
                    candles = await self.rest.get_klines_range(
                        interval=interval,
                        start_time=gap_start,
                        end_time=now_ms,
                    )
                    if candles:
                        self.db.insert_candles_batch(table, candles)
                        logger.info(f"  {interval}: Filled {len(candles)} gap candles.")
                    return {
                        "status": "gap_fill",
                        "existing": count_before,
                        "added": len(candles) if candles else 0,
                    }
                return {"status": "up_to_date", "existing": count_before, "added": 0}

        # Need full or partial backfill
        fetch_start = target_start_ms
        if latest_ts:
            # We have some old data -- decide whether to fill from beginning or from latest
            oldest_we_need = target_start_ms
            fetch_start = oldest_we_need
            logger.info(
                f"  {interval}: Have {count_before} candles. "
                f"Backfilling from {datetime.fromtimestamp(fetch_start / 1000, tz=timezone.utc).isoformat()}"
            )
        else:
            logger.info(
                f"  {interval}: No existing data. "
                f"Full backfill from {datetime.fromtimestamp(fetch_start / 1000, tz=timezone.utc).isoformat()}"
            )

        # Fetch all candles
        candles = await self.rest.get_klines_range(
            interval=interval,
            start_time=fetch_start,
            end_time=now_ms,
        )

        if candles:
            self.db.insert_candles_batch(table, candles)
            count_after = self.db.get_candle_count(table)
            added = count_after - count_before
            logger.info(
                f"  {interval}: Backfill complete. "
                f"Added {added} candles (total: {count_after})."
            )
            return {
                "status": "backfilled",
                "existing": count_before,
                "fetched": len(candles),
                "added": added,
                "total": count_after,
            }
        else:
            logger.warning(f"  {interval}: No candles returned from API.")
            return {"status": "no_data", "existing": count_before, "added": 0}

    async def check_data_health(self) -> dict:
        """Check the health of stored candle data.
        
        Returns:
            Dict with health status for each interval.
        """
        health = {}
        now_ms = utc_timestamp_ms()

        for interval, (table, interval_ms) in CANDLE_INTERVALS.items():
            count = self.db.get_candle_count(table)
            latest_ts = self.db.get_latest_candle_timestamp(table)

            if latest_ts:
                age_seconds = (now_ms - latest_ts) / 1000
                latest_dt = datetime.fromtimestamp(latest_ts / 1000, tz=timezone.utc)
                expected_min = (MIN_CANDLE_DAYS * 24 * 60 * 60 * 1000) // interval_ms
                
                health[interval] = {
                    "count": count,
                    "latest": latest_dt.isoformat(),
                    "age_seconds": round(age_seconds, 1),
                    "has_minimum_data": count >= expected_min,
                    "sufficient": count >= expected_min and age_seconds < interval_ms / 1000 * 3,
                }
            else:
                health[interval] = {
                    "count": 0,
                    "latest": None,
                    "age_seconds": None,
                    "has_minimum_data": False,
                    "sufficient": False,
                }

        return health
