"""CleoBot SQLite database manager with all table schemas."""

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger("database")


class Database:
    """Thread-safe SQLite database manager for CleoBot.

    Manages all persistent data: candles, orderbook snapshots, funding rates,
    signals, trades, model versions, and session stats.
    """

    def __init__(self, db_path: str):
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path,
                timeout=30,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            )
            self._local.connection.row_factory = sqlite3.Row
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA busy_timeout=5000")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
        return self._local.connection

    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor with auto-commit/rollback."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_db(self):
        """Create all tables if they don't exist."""
        logger.info(f"Initializing database at {self.db_path}")
        with self.get_cursor() as cursor:
            # 5-minute candles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles_5m (
                    timestamp INTEGER PRIMARY KEY,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL
                )
            """)

            # 15-minute candles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles_15m (
                    timestamp INTEGER PRIMARY KEY,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL
                )
            """)

            # 1-hour candles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles_1h (
                    timestamp INTEGER PRIMARY KEY,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL
                )
            """)

            # Orderbook snapshots
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    bids_json TEXT NOT NULL,
                    asks_json TEXT NOT NULL,
                    mid_price REAL NOT NULL,
                    spread REAL NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_orderbook_timestamp
                ON orderbook_snapshots(timestamp)
            """)

            # Funding rates
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS funding_rates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL UNIQUE,
                    rate REAL NOT NULL,
                    next_settlement INTEGER
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_funding_timestamp
                ON funding_rates(timestamp)
            """)

            # Signals
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    models_json TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    filters_json TEXT NOT NULL,
                    traded INTEGER NOT NULL DEFAULT 0,
                    outcome TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_timestamp
                ON signals(timestamp)
            """)

            # Trades -- extended schema with Polymarket order metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    signal_id INTEGER NOT NULL DEFAULT 0,
                    direction TEXT NOT NULL,
                    entry_price REAL,
                    polymarket_odds REAL,
                    fill_time REAL,
                    settlement TEXT,
                    pnl REAL,
                    trade_size REAL NOT NULL DEFAULT 1.0,
                    order_id TEXT,
                    market_id TEXT,
                    token_id TEXT,
                    is_simulated INTEGER NOT NULL DEFAULT 1,
                    is_premium INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT,
                    signal_json TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp
                ON trades(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_signal_id
                ON trades(signal_id)
            """)

            # Model versions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    accuracy REAL,
                    features_json TEXT
                )
            """)

            # Session stats (daily aggregates)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_stats (
                    date TEXT PRIMARY KEY,
                    trades_count INTEGER NOT NULL DEFAULT 0,
                    wins INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,
                    skips INTEGER NOT NULL DEFAULT 0,
                    pnl REAL NOT NULL DEFAULT 0.0,
                    accuracy REAL NOT NULL DEFAULT 0.0
                )
            """)

            # Feature snapshots for restart recovery
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feature_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_ms INTEGER NOT NULL,
                    features_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feature_snapshots_ts ON feature_snapshots (timestamp_ms)
            """)

        logger.info("Database initialized -- all tables created.")

    # ==================== CANDLE OPERATIONS ====================

    def insert_candle(self, table: str, timestamp: int, open_: float, high: float,
                      low: float, close: float, volume: float):
        """Insert or replace a candle record.

        Args:
            table: Table name ('candles_5m', 'candles_15m', 'candles_1h').
            timestamp: Candle open timestamp in milliseconds.
            open_: Open price.
            high: High price.
            low: Low price.
            close: Close price.
            volume: Volume.
        """
        if table not in ("candles_5m", "candles_15m", "candles_1h"):
            raise ValueError(f"Invalid candle table: {table}")
        with self.get_cursor() as cursor:
            cursor.execute(
                f"INSERT OR REPLACE INTO {table} (timestamp, open, high, low, close, volume) "
                f"VALUES (?, ?, ?, ?, ?, ?)",
                (timestamp, open_, high, low, close, volume),
            )

    def insert_candles_batch(self, table: str, candles: List[Tuple]):
        """Insert multiple candle records at once.

        Args:
            table: Table name.
            candles: List of (timestamp, open, high, low, close, volume) tuples.
        """
        if table not in ("candles_5m", "candles_15m", "candles_1h"):
            raise ValueError(f"Invalid candle table: {table}")
        with self.get_cursor() as cursor:
            cursor.executemany(
                f"INSERT OR REPLACE INTO {table} (timestamp, open, high, low, close, volume) "
                f"VALUES (?, ?, ?, ?, ?, ?)",
                candles,
            )
        logger.debug(f"Inserted {len(candles)} candles into {table}.")

    def get_candles(self, table: str, limit: int = 100,
                    since: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent candles from a table.

        Args:
            table: Table name.
            limit: Maximum number of candles to return.
            since: Only return candles after this timestamp (ms).

        Returns:
            List of candle dicts sorted by timestamp ascending.
        """
        if table not in ("candles_5m", "candles_15m", "candles_1h"):
            raise ValueError(f"Invalid candle table: {table}")
        with self.get_cursor() as cursor:
            if since:
                cursor.execute(
                    f"SELECT * FROM {table} WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT ?",
                    (since, limit),
                )
            else:
                cursor.execute(
                    f"SELECT * FROM {table} ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
                rows = cursor.fetchall()
                return [dict(r) for r in reversed(rows)]
            return [dict(r) for r in cursor.fetchall()]

    def get_latest_candle_timestamp(self, table: str) -> Optional[int]:
        """Get the timestamp of the most recent candle.

        Args:
            table: Table name.

        Returns:
            Timestamp in milliseconds, or None if table is empty.
        """
        if table not in ("candles_5m", "candles_15m", "candles_1h"):
            raise ValueError(f"Invalid candle table: {table}")
        with self.get_cursor() as cursor:
            cursor.execute(f"SELECT MAX(timestamp) FROM {table}")
            result = cursor.fetchone()
            return result[0] if result and result[0] is not None else None

    def get_candle_count(self, table: str) -> int:
        """Get the number of candles in a table.

        Args:
            table: Table name.

        Returns:
            Number of candle records.
        """
        if table not in ("candles_5m", "candles_15m", "candles_1h"):
            raise ValueError(f"Invalid candle table: {table}")
        with self.get_cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            return cursor.fetchone()[0]

    # ==================== ORDERBOOK OPERATIONS ====================

    def insert_orderbook_snapshot(self, timestamp: int, bids: List, asks: List,
                                  mid_price: float, spread: float):
        """Insert an orderbook snapshot.

        Args:
            timestamp: Snapshot timestamp in milliseconds.
            bids: List of [price, quantity] bid levels.
            asks: List of [price, quantity] ask levels.
            mid_price: Mid price at snapshot time.
            spread: Spread at snapshot time.
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO orderbook_snapshots "
                "(timestamp, bids_json, asks_json, mid_price, spread) "
                "VALUES (?, ?, ?, ?, ?)",
                (timestamp, json.dumps(bids), json.dumps(asks), mid_price, spread),
            )

    def get_orderbook_snapshots(self, since: int,
                                limit: int = 100) -> List[Dict[str, Any]]:
        """Get orderbook snapshots since a timestamp.

        Args:
            since: Timestamp in milliseconds.
            limit: Maximum number of snapshots.

        Returns:
            List of snapshot dicts with parsed bids/asks.
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM orderbook_snapshots WHERE timestamp >= ? "
                "ORDER BY timestamp ASC LIMIT ?",
                (since, limit),
            )
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                d["bids"] = json.loads(d.pop("bids_json"))
                d["asks"] = json.loads(d.pop("asks_json"))
                results.append(d)
            return results

    def get_latest_orderbook(self) -> Optional[Dict[str, Any]]:
        """Get the most recent orderbook snapshot.

        Returns:
            Snapshot dict or None.
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM orderbook_snapshots ORDER BY timestamp DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["bids"] = json.loads(d.pop("bids_json"))
                d["asks"] = json.loads(d.pop("asks_json"))
                return d
            return None

    def cleanup_old_orderbook(self, days: int = 7):
        """Delete orderbook snapshots older than N days.

        Args:
            days: Number of days of data to retain.
        """
        cutoff = int(
            (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
        )
        with self.get_cursor() as cursor:
            cursor.execute(
                "DELETE FROM orderbook_snapshots WHERE timestamp < ?", (cutoff,)
            )
            deleted = cursor.rowcount
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old orderbook snapshots (>{days} days).")

    # ==================== FUNDING RATE OPERATIONS ====================

    def insert_funding_rate(self, timestamp: int, rate: float,
                            next_settlement: Optional[int] = None):
        """Insert a funding rate record.

        Args:
            timestamp: Timestamp in milliseconds.
            rate: Funding rate value.
            next_settlement: Next settlement timestamp in milliseconds.
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                "INSERT OR IGNORE INTO funding_rates (timestamp, rate, next_settlement) "
                "VALUES (?, ?, ?)",
                (timestamp, rate, next_settlement),
            )

    def get_funding_rates(self, limit: int = 100,
                          since: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent funding rates.

        Args:
            limit: Maximum number of records.
            since: Only return records after this timestamp (ms).

        Returns:
            List of funding rate dicts.
        """
        with self.get_cursor() as cursor:
            if since:
                cursor.execute(
                    "SELECT * FROM funding_rates WHERE timestamp >= ? "
                    "ORDER BY timestamp ASC LIMIT ?",
                    (since, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM funding_rates ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
                rows = cursor.fetchall()
                return [dict(r) for r in reversed(rows)]
            return [dict(r) for r in cursor.fetchall()]

    def get_latest_funding_rate(self) -> Optional[Dict[str, Any]]:
        """Get the most recent funding rate.

        Returns:
            Funding rate dict or None.
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM funding_rates ORDER BY timestamp DESC LIMIT 1"
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    # ==================== SIGNAL OPERATIONS ====================

    def insert_signal(self, timestamp: int, direction: str, confidence: float,
                      models: Dict, regime: str, filters: Dict,
                      traded: bool = False) -> int:
        """Insert a signal record and return its ID.

        Args:
            timestamp: Signal timestamp in milliseconds.
            direction: 'UP' or 'DOWN'.
            confidence: Meta-learner confidence (0-1).
            models: Dict of individual model predictions.
            regime: Current regime classification.
            filters: Dict of filter verdicts.
            traded: Whether a trade was placed.

        Returns:
            The signal ID (primary key).
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO signals (timestamp, direction, confidence, models_json, "
                "regime, filters_json, traded) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    timestamp, direction, confidence,
                    json.dumps(models), regime, json.dumps(filters), int(traded),
                ),
            )
            return cursor.lastrowid

    def update_signal_outcome(self, signal_id: int, outcome: str):
        """Update a signal's outcome after settlement.

        Args:
            signal_id: Signal ID.
            outcome: 'WIN', 'LOSS', or 'PUSH'.
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                "UPDATE signals SET outcome = ? WHERE id = ?",
                (outcome, signal_id),
            )

    def get_recent_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent signals.

        Args:
            limit: Maximum number of signals.

        Returns:
            List of signal dicts with parsed JSON fields.
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                d["models"] = json.loads(d.pop("models_json"))
                d["filters"] = json.loads(d.pop("filters_json"))
                d["traded"] = bool(d["traded"])
                results.append(d)
            return list(reversed(results))

    def get_signals_since(self, since: int) -> List[Dict[str, Any]]:
        """Get all signals since a timestamp.

        Args:
            since: Timestamp in milliseconds.

        Returns:
            List of signal dicts.
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM signals WHERE timestamp >= ? ORDER BY timestamp ASC",
                (since,),
            )
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                d["models"] = json.loads(d.pop("models_json"))
                d["filters"] = json.loads(d.pop("filters_json"))
                d["traded"] = bool(d["traded"])
                results.append(d)
            return results

    def get_signal_count(self) -> int:
        """Get total number of signals."""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM signals")
            return cursor.fetchone()[0]

    # ==================== TRADE OPERATIONS ====================

    def insert_trade(
        self,
        timestamp: int,
        signal_id: int,
        direction: str,
        entry_price: Optional[float] = None,
        polymarket_odds: Optional[float] = None,
        fill_time: Optional[float] = None,
        trade_size: float = 1.0,
        order_id: Optional[str] = None,
        market_id: Optional[str] = None,
        token_id: Optional[str] = None,
        is_simulated: bool = True,
        is_premium: bool = False,
        created_at: Optional[str] = None,
        signal_json: Optional[str] = None,
    ) -> int:
        """Insert a trade record and return its ID.

        Args:
            timestamp: Trade timestamp in milliseconds.
            signal_id: Associated signal ID.
            direction: 'UP' or 'DOWN'.
            entry_price: BTC price at trade time.
            polymarket_odds: Polymarket odds at fill.
            fill_time: Time to fill in seconds.
            trade_size: Trade size in USD.
            order_id: Polymarket order ID.
            market_id: Polymarket market ID.
            token_id: Polymarket token ID.
            is_simulated: True if this is a paper trade.
            is_premium: True if signal met premium criteria.
            created_at: ISO timestamp string of when the trade was placed.
            signal_json: JSON-serialised signal dict.

        Returns:
            The trade ID.
        """
        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat()
        with self.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO trades "
                "(timestamp, signal_id, direction, entry_price, polymarket_odds, "
                "fill_time, trade_size, order_id, market_id, token_id, "
                "is_simulated, is_premium, created_at, signal_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    timestamp, signal_id, direction, entry_price, polymarket_odds,
                    fill_time, trade_size, order_id, market_id, token_id,
                    int(is_simulated), int(is_premium), created_at, signal_json,
                ),
            )
            return cursor.lastrowid

    def record_trade(
        self,
        direction: str,
        trade_size: float,
        entry_price: Optional[float] = None,
        order_id: Optional[str] = None,
        market_id: Optional[str] = None,
        token_id: Optional[str] = None,
        signal: Optional[Dict[str, Any]] = None,
        is_simulated: bool = True,
        is_premium: bool = False,
    ) -> int:
        """High-level method: record a newly placed trade and return its DB trade ID.

        Inserts a signal record first (to satisfy FK), then inserts the trade row
        with all Polymarket order metadata.

        Args:
            direction: 'UP' or 'DOWN'.
            trade_size: Trade size in USD.
            entry_price: Polymarket fill price (0-1 odds).
            order_id: Polymarket order ID string.
            market_id: Polymarket market ID string.
            token_id: Polymarket token ID string.
            signal: Full signal dict (from EnsembleSignal.to_dict() + extras).
            is_simulated: True if paper trading.
            is_premium: True if signal met premium criteria.

        Returns:
            The new trade's database ID.
        """
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        now_iso = datetime.now(timezone.utc).isoformat()

        sig = signal or {}
        confidence = float(sig.get("confidence", 0.0))
        models = sig.get("models", {})
        regime = str(sig.get("regime", "unknown"))
        filters = sig.get("filter_result", {})

        # Insert signal record first
        signal_id = self.insert_signal(
            timestamp=now_ms,
            direction=direction,
            confidence=confidence,
            models=models,
            regime=regime,
            filters=filters,
            traded=True,
        )

        # Insert trade record
        trade_id = self.insert_trade(
            timestamp=now_ms,
            signal_id=signal_id,
            direction=direction,
            entry_price=entry_price,
            trade_size=trade_size,
            order_id=order_id,
            market_id=market_id,
            token_id=token_id,
            is_simulated=is_simulated,
            is_premium=is_premium,
            created_at=now_iso,
            signal_json=json.dumps(sig) if sig else None,
        )

        logger.info(
            "record_trade: trade #%d (signal #%d) | %s $%.2f @ %s (%s)",
            trade_id, signal_id, direction, trade_size, entry_price,
            "SIM" if is_simulated else "LIVE",
        )
        return trade_id

    def settle_trade(
        self,
        trade_id: int,
        outcome: str,
        pnl: float,
        settlement_data: Optional[Dict[str, Any]] = None,
    ):
        """High-level method: record a trade settlement result.

        Updates the trade row with settlement outcome and PnL, and also
        updates the linked signal's outcome field.

        Args:
            trade_id: Trade DB ID.
            outcome: 'WIN' or 'LOSS'.
            pnl: Actual PnL amount (positive for win, negative for loss).
            settlement_data: Full settlement dict (unused, reserved for future).
        """
        self.update_trade_settlement(trade_id=trade_id, settlement=outcome, pnl=pnl)

        # Also update the linked signal's outcome
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT signal_id FROM trades WHERE id = ?", (trade_id,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                self.update_signal_outcome(signal_id=row[0], outcome=outcome)

        logger.info("settle_trade: trade #%d -> %s PnL=%+.2f", trade_id, outcome, pnl)

    def update_trade_settlement(self, trade_id: int, settlement: str, pnl: float):
        """Update a trade with settlement result.

        Args:
            trade_id: Trade ID.
            settlement: 'WIN' or 'LOSS'.
            pnl: Profit/loss amount.
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                "UPDATE trades SET settlement = ?, pnl = ? WHERE id = ?",
                (settlement, pnl, trade_id),
            )

    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trades.

        Args:
            limit: Maximum number of trades.

        Returns:
            List of trade dicts sorted by timestamp ascending.
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in reversed(cursor.fetchall())]

    def get_unsettled_trades(self) -> List[Dict[str, Any]]:
        """Get trades that haven't been settled yet.

        Returns:
            List of unsettled trade dicts.
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM trades WHERE settlement IS NULL ORDER BY timestamp ASC"
            )
            return [dict(r) for r in cursor.fetchall()]

    def get_trades_today(self) -> List[Dict[str, Any]]:
        """Get all trades from today (UTC).

        Returns:
            List of today's trade dicts.
        """
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        today_start_ms = int(today_start.timestamp() * 1000)
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp ASC",
                (today_start_ms,),
            )
            return [dict(r) for r in cursor.fetchall()]

    def get_trade_stats_today(self) -> Dict[str, Any]:
        """Get aggregated trading stats for today.

        Returns:
            Dict with wins, losses, total, accuracy, pnl.
        """
        trades = self.get_trades_today()
        settled = [t for t in trades if t["settlement"] is not None]
        wins = sum(1 for t in settled if t["settlement"] == "WIN")
        losses = sum(1 for t in settled if t["settlement"] == "LOSS")
        total = wins + losses
        pnl = sum(t["pnl"] for t in settled if t["pnl"] is not None)
        accuracy = wins / total if total > 0 else 0.0
        return {
            "total_trades": len(trades),
            "settled": total,
            "wins": wins,
            "losses": losses,
            "unsettled": len(trades) - total,
            "accuracy": accuracy,
            "pnl": pnl,
        }

    def get_consecutive_losses(self) -> int:
        """Get the current consecutive loss streak.

        Returns:
            Number of consecutive losses (0 if last trade was a win).
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT settlement FROM trades WHERE settlement IS NOT NULL "
                "ORDER BY timestamp DESC LIMIT 20"
            )
            streak = 0
            for row in cursor.fetchall():
                if row["settlement"] == "LOSS":
                    streak += 1
                else:
                    break
            return streak

    def get_rolling_accuracy(self, n_trades: int = 50) -> Optional[float]:
        """Get accuracy over the last N settled trades.

        Args:
            n_trades: Number of trades to consider.

        Returns:
            Accuracy as a ratio (0-1), or None if insufficient trades.
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT settlement FROM trades WHERE settlement IS NOT NULL "
                "ORDER BY timestamp DESC LIMIT ?",
                (n_trades,),
            )
            rows = cursor.fetchall()
            if len(rows) < 3:  # Need minimum sample
                return None
            wins = sum(1 for r in rows if r["settlement"] == "WIN")
            return wins / len(rows)

    def get_total_settled_trades(self) -> int:
        """Return total count of all settled trades."""
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM trades WHERE settlement IS NOT NULL"
            )
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def get_recent_settled_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent settled trades ordered by timestamp desc."""
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM trades WHERE settlement IS NOT NULL "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    # ==================== MODEL VERSION OPERATIONS ====================

    def insert_model_version(self, timestamp: int, model_name: str, version: int,
                             accuracy: Optional[float] = None,
                             features: Optional[List[str]] = None):
        """Record a model version.

        Args:
            timestamp: Training timestamp in milliseconds.
            model_name: Model name ('lgbm', 'tcn', 'logreg', 'meta', 'hmm').
            version: Version number.
            accuracy: Validation accuracy.
            features: List of feature names used.
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO model_versions "
                "(timestamp, model_name, version, accuracy, features_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    timestamp, model_name, version, accuracy,
                    json.dumps(features) if features else None,
                ),
            )

    def get_latest_model_version(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get the latest version record for a model.

        Args:
            model_name: Model name.

        Returns:
            Model version dict or None.
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM model_versions WHERE model_name = ? "
                "ORDER BY version DESC LIMIT 1",
                (model_name,),
            )
            row = cursor.fetchone()
            if row:
                d = dict(row)
                if d.get("features_json"):
                    d["features"] = json.loads(d.pop("features_json"))
                else:
                    d.pop("features_json", None)
                    d["features"] = None
                return d
            return None

    # ==================== SESSION STATS OPERATIONS ====================

    def update_session_stats(self, date: str, trades_count: int, wins: int,
                             losses: int, skips: int, pnl: float,
                             accuracy: float):
        """Insert or update daily session stats.

        Args:
            date: Date string 'YYYY-MM-DD'.
            trades_count: Total trades placed.
            wins: Number of wins.
            losses: Number of losses.
            skips: Number of skipped signals.
            pnl: Total P&L for the day.
            accuracy: Win rate (0-1).
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                "INSERT OR REPLACE INTO session_stats "
                "(date, trades_count, wins, losses, skips, pnl, accuracy) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (date, trades_count, wins, losses, skips, pnl, accuracy),
            )

    def get_session_stats(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get session stats for the last N days.

        Args:
            days: Number of days to retrieve.

        Returns:
            List of daily stats dicts.
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM session_stats ORDER BY date DESC LIMIT ?",
                (days,),
            )
            return [dict(r) for r in reversed(cursor.fetchall())]

    # ==================== UTILITY OPERATIONS ====================

    def get_db_stats(self) -> Dict[str, int]:
        """Get record counts for all tables.

        Returns:
            Dict mapping table name to record count.
        """
        tables = [
            "candles_5m", "candles_15m", "candles_1h",
            "orderbook_snapshots", "funding_rates",
            "signals", "trades", "model_versions", "session_stats",
        ]
        stats = {}
        with self.get_cursor() as cursor:
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
        return stats

    def get_db_size_mb(self) -> float:
        """Get database file size in megabytes.

        Returns:
            File size in MB.
        """
        try:
            size_bytes = os.path.getsize(self.db_path)
            return size_bytes / (1024 * 1024)
        except OSError:
            return 0.0

    def cleanup_old_data(self, candle_days: int = 30, orderbook_days: int = 7):
        """Clean up old data to manage storage.

        Args:
            candle_days: Days of candle data to retain.
            orderbook_days: Days of orderbook data to retain.
        """
        candle_cutoff = int(
            (datetime.now(timezone.utc) - timedelta(days=candle_days)).timestamp() * 1000
        )
        ob_cutoff = int(
            (datetime.now(timezone.utc) - timedelta(days=orderbook_days)).timestamp() * 1000
        )

        with self.get_cursor() as cursor:
            for table in ("candles_5m", "candles_15m", "candles_1h"):
                cursor.execute(
                    f"DELETE FROM {table} WHERE timestamp < ?", (candle_cutoff,)
                )
                if cursor.rowcount > 0:
                    logger.info(f"Cleaned {cursor.rowcount} old records from {table}.")

            cursor.execute(
                "DELETE FROM orderbook_snapshots WHERE timestamp < ?", (ob_cutoff,)
            )
            if cursor.rowcount > 0:
                logger.info(f"Cleaned {cursor.rowcount} old orderbook snapshots.")

        # Clean up old feature snapshots (keep 7 days)
        cutoff_features_ms = int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp() * 1000)
        with self.get_cursor() as cursor:
            cursor.execute("DELETE FROM feature_snapshots WHERE timestamp_ms < ?", (cutoff_features_ms,))
            deleted = cursor.rowcount
        logger.info(f"Cleaned up {deleted} old feature snapshots")

    def save_feature_snapshot(self, timestamp_ms: int, features_json: str) -> None:
        """Save a feature snapshot to the database for restart recovery."""
        with self.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO feature_snapshots (timestamp_ms, features_json) VALUES (?, ?)",
                (timestamp_ms, features_json),
            )
        logger.debug(f"Saved feature snapshot at ts={timestamp_ms}")

    def get_feature_snapshots(self, limit: int = 200) -> List[Dict]:
        """Get recent feature snapshots for restart recovery, oldest first."""
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT timestamp_ms, features_json FROM feature_snapshots ORDER BY timestamp_ms DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
        # Reverse to get oldest-first order for proper history reconstruction
        result = []
        for row in reversed(rows):
            try:
                features = json.loads(row["features_json"])
                features["_snapshot_ts"] = row["timestamp_ms"]
                result.append(features)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse feature snapshot: {e}")
        return result

    def close(self):
        """Close the database connection."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
            logger.info("Database connection closed.")
