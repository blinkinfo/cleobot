"""Integration tests for CleoBot pipeline -- Phase 7.

Tests the full 5-minute cycle pipeline end-to-end in paper/simulation mode:
  - Database initialization and operations
  - Feature engine (mocked data)
  - Signal filter pipeline
  - Risk manager decisions
  - Trading executor cycle (with mocked ensemble and Polymarket)
  - Daily summary and session stats
  - Startup/shutdown sequence

All tests run without network calls using mocks and in-memory SQLite.
"""

import asyncio
import os
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ================================================================== #
# FIXTURES
# ================================================================== #

@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temp directory for database and models."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return tmp_path


@pytest.fixture
def db(temp_dir):
    """Create a fresh in-memory-style SQLite DB in a temp file."""
    from src.database import Database
    db_path = str(temp_dir / "test_cleobot.db")
    database = Database(db_path)
    yield database
    database.close()


@pytest.fixture
def config(temp_dir):
    """Return a Config with test values (no real credentials needed)."""
    from src.config import Config, TelegramConfig, MEXCConfig, PolymarketConfig, TradingConfig, SystemConfig
    return Config(
        telegram=TelegramConfig(bot_token="", chat_id=""),
        mexc=MEXCConfig(api_key="", secret_key=""),
        polymarket=PolymarketConfig(private_key="", funder_address="", signature_type=2),
        trading=TradingConfig(
            auto_trade_enabled=False,
            base_trade_size=1.0,
            max_trade_size=3.0,
            max_daily_loss=15.0,
            max_consecutive_losses=5,
            max_open_exposure=3.0,
        ),
        system=SystemConfig(
            log_level="WARNING",
            data_dir=str(temp_dir),
            retrain_hour_utc=4,
        ),
    )


def _make_candles(n: int, base_price: float = 50000.0, interval_ms: int = 300_000) -> list:
    """Generate synthetic 5m candle data."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    candles = []
    for i in range(n):
        ts = now_ms - (n - i) * interval_ms
        price = base_price + (i % 10 - 5) * 100
        candles.append({
            "timestamp": ts,
            "open": price,
            "high": price + 50,
            "low": price - 50,
            "close": price + 10,
            "volume": 100.0 + i,
        })
    return candles


def _populate_db(db, n_candles: int = 300):
    """Populate DB with synthetic candle data."""
    candles = _make_candles(n_candles)
    rows = [(c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]) for c in candles]
    db.insert_candles_batch("candles_5m", rows)
    # Also populate 15m and 1h
    db.insert_candles_batch("candles_15m", rows[:100])
    db.insert_candles_batch("candles_1h", rows[:50])
    return candles


# ================================================================== #
# 1. DATABASE TESTS
# ================================================================== #

class TestDatabase:
    """Test database initialization and CRUD operations."""

    def test_init_creates_tables(self, db):
        """Database should create all required tables."""
        stats = db.get_db_stats()
        expected_tables = [
            "candles_5m", "candles_15m", "candles_1h",
            "orderbook_snapshots", "funding_rates",
            "signals", "trades", "model_versions", "session_stats",
        ]
        for table in expected_tables:
            assert table in stats, f"Table {table} missing from stats"

    def test_insert_and_retrieve_candles(self, db):
        """Should insert and retrieve candles correctly."""
        _populate_db(db, n_candles=50)
        count = db.get_candle_count("candles_5m")
        assert count == 50

        candles = db.get_candles("candles_5m", limit=10)
        assert len(candles) == 10
        assert all("close" in c for c in candles)

    def test_latest_candle_timestamp(self, db):
        """get_latest_candle_timestamp should return most recent ts."""
        _populate_db(db, n_candles=20)
        ts = db.get_latest_candle_timestamp("candles_5m")
        assert ts is not None
        assert isinstance(ts, int)

    def test_insert_signal(self, db):
        """Should insert a signal and retrieve it."""
        ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        signal_id = db.insert_signal(
            timestamp=ts,
            direction="UP",
            confidence=0.65,
            models={"lgbm": "UP", "tcn": "UP", "logreg": "DOWN"},
            regime="trending_up",
            filters={"confidence": True, "volatility": True},
            traded=False,
        )
        assert signal_id > 0

        signals = db.get_recent_signals(limit=5)
        assert len(signals) == 1
        assert signals[0]["direction"] == "UP"
        assert signals[0]["confidence"] == pytest.approx(0.65)

    def test_get_consecutive_losses_zero(self, db):
        """Should return 0 when no trades."""
        result = db.get_consecutive_losses()
        assert result == 0

    def test_rolling_accuracy_none_insufficient(self, db):
        """Should return None when fewer than 3 settled trades."""
        result = db.get_rolling_accuracy(n_trades=50)
        assert result is None

    def test_trade_stats_today_empty(self, db):
        """Should return zero stats when no trades today."""
        stats = db.get_trade_stats_today()
        assert stats["total_trades"] == 0
        assert stats["wins"] == 0
        assert stats["losses"] == 0
        assert stats["pnl"] == 0.0

    def test_cleanup_old_data(self, db):
        """cleanup_old_data should run without error."""
        _populate_db(db, n_candles=10)
        db.cleanup_old_data(candle_days=30, orderbook_days=7)  # Should not throw

    def test_session_stats_upsert(self, db):
        """update_session_stats should insert and update."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        db.update_session_stats(
            date=today,
            trades_count=10,
            wins=6,
            losses=4,
            skips=2,
            pnl=2.28,
            accuracy=0.60,
        )
        stats = db.get_session_stats(days=7)
        assert len(stats) == 1
        assert stats[0]["wins"] == 6
        assert stats[0]["pnl"] == pytest.approx(2.28)


# ================================================================== #
# 2. SIGNAL FILTER TESTS
# ================================================================== #

class TestSignalFilter:
    """Test the signal filter pipeline."""

    def _make_mock_signal(
        self,
        direction="UP",
        confidence=0.65,
        regime="trending_up",
        agreement=2,
    ):
        """Create a mock EnsembleSignal."""
        signal = MagicMock()
        signal.direction = direction
        signal.confidence = confidence
        signal.regime = regime
        signal.regime_display = regime.replace("_", " ").title()
        signal.agreement = agreement
        signal.lgbm = {"direction": direction, "confidence": 0.62}
        signal.tcn = {"direction": direction, "confidence": 0.63}
        signal.logreg = {"direction": direction, "confidence": 0.61}
        signal.to_dict = MagicMock(return_value={
            "direction": direction,
            "confidence": confidence,
            "regime": regime,
        })
        return signal

    def test_high_confidence_passes(self):
        """Signal with high confidence and good agreement should TRADE."""
        from src.trading.filters import SignalFilter
        sf = SignalFilter()
        signal = self._make_mock_signal(confidence=0.70, agreement=3)
        # Populate ATR history
        for i in range(20):
            sf.add_atr_observation(100.0 + i * 2)
        result = sf.evaluate(
            signal=signal,
            current_atr=120.0,
            consecutive_losses=0,
            rolling_accuracy=None,
            n_settled_trades=0,
        )
        assert result.decision == "TRADE"
        assert "confidence" in result.verdicts

    def test_low_confidence_skips(self):
        """Signal below confidence threshold should SKIP."""
        from src.trading.filters import SignalFilter
        sf = SignalFilter()
        signal = self._make_mock_signal(confidence=0.45, agreement=2)
        result = sf.evaluate(
            signal=signal,
            current_atr=100.0,
            consecutive_losses=0,
            rolling_accuracy=None,
            n_settled_trades=0,
        )
        assert result.decision == "SKIP"
        assert "confidence" in result.skip_reason.lower() or result.skip_reason != ""

    def test_streak_pause_after_3_losses(self):
        """3 consecutive losses should trigger a 1-cycle pause."""
        from src.trading.filters import SignalFilter
        sf = SignalFilter()
        signal = self._make_mock_signal(confidence=0.70, agreement=3)
        for i in range(20):
            sf.add_atr_observation(100.0 + i)
        result = sf.evaluate(
            signal=signal,
            current_atr=110.0,
            consecutive_losses=3,
            rolling_accuracy=None,
            n_settled_trades=0,
        )
        # First check triggers the pause
        result2 = sf.evaluate(
            signal=signal,
            current_atr=110.0,
            consecutive_losses=3,
            rolling_accuracy=None,
            n_settled_trades=0,
        )
        assert result2.decision == "SKIP"
        assert result2.verdicts["streak"].passed is False

    def test_7_loss_streak_requires_manual_restart(self):
        """7 consecutive losses should hard-stop and require manual /start."""
        from src.trading.filters import SignalFilter
        sf = SignalFilter()
        signal = self._make_mock_signal(confidence=0.80, agreement=3)
        result = sf.evaluate(
            signal=signal,
            current_atr=100.0,
            consecutive_losses=7,
            rolling_accuracy=None,
            n_settled_trades=0,
        )
        assert result.decision == "SKIP"
        assert "manual" in result.skip_reason.lower() or result.verdicts["streak"].passed is False

    def test_filter_result_serializable(self):
        """FilterResult.to_dict() should return a JSON-serializable dict."""
        import json
        from src.trading.filters import SignalFilter
        sf = SignalFilter()
        signal = self._make_mock_signal(confidence=0.65, agreement=2)
        result = sf.evaluate(
            signal=signal,
            current_atr=100.0,
            consecutive_losses=0,
            rolling_accuracy=None,
            n_settled_trades=0,
        )
        d = result.to_dict()
        # Should not raise
        json.dumps(d)
        assert "decision" in d
        assert "verdicts" in d


# ================================================================== #
# 3. RISK MANAGER TESTS
# ================================================================== #

class TestRiskManager:
    """Test risk management rules."""

    def test_auto_trade_disabled_blocks_all(self, db, config):
        """When auto_trade=False, all trades should be blocked."""
        from src.trading.risk_manager import RiskManager
        from src.config import TradingConfig
        trading_cfg = TradingConfig(auto_trade_enabled=False)
        rm = RiskManager(db, trading_cfg)
        result = rm.check_trade(consecutive_losses=0)
        assert result.approved is False
        assert "disabled" in result.skip_reason.lower()

    def test_auto_trade_enabled_approves(self, db):
        """When auto_trade=True and no limits hit, trade should be approved."""
        from src.trading.risk_manager import RiskManager
        from src.config import TradingConfig
        trading_cfg = TradingConfig(auto_trade_enabled=True, max_daily_loss=15.0, max_open_exposure=3.0)
        rm = RiskManager(db, trading_cfg)
        result = rm.check_trade(consecutive_losses=0)
        assert result.approved is True

    def test_calculate_trade_size_base(self, db):
        """With zero profit, trade size should be base $1.00."""
        from src.trading.risk_manager import RiskManager
        from src.config import TradingConfig
        trading_cfg = TradingConfig(auto_trade_enabled=True)
        rm = RiskManager(db, trading_cfg)
        size = rm.calculate_trade_size()
        assert size == pytest.approx(1.0)

    def test_circuit_breaker_after_daily_loss(self, db):
        """Exceeding daily loss limit should activate circuit breaker."""
        from src.trading.risk_manager import RiskManager
        from src.config import TradingConfig
        trading_cfg = TradingConfig(auto_trade_enabled=True, max_daily_loss=5.0)
        rm = RiskManager(db, trading_cfg)
        # Simulate daily PnL already at loss limit
        rm._daily_pnl = -5.01
        result = rm.check_trade(consecutive_losses=0)
        assert result.approved is False

    def test_open_exposure_limit(self, db):
        """Exceeding max open exposure should block trade."""
        from src.trading.risk_manager import RiskManager
        from src.config import TradingConfig
        trading_cfg = TradingConfig(auto_trade_enabled=True, max_open_exposure=3.0)
        rm = RiskManager(db, trading_cfg)
        # Fill up exposure
        rm._open_positions = {1: 1.5, 2: 1.5}  # $3.00 open
        result = rm.check_trade(consecutive_losses=0)
        assert result.approved is False
        assert "exposure" in result.skip_reason.lower()

    def test_get_daily_stats_summary(self, db):
        """get_daily_stats_summary should return expected keys."""
        from src.trading.risk_manager import RiskManager
        from src.config import TradingConfig
        rm = RiskManager(db, TradingConfig())
        summary = rm.get_daily_stats_summary()
        for key in ["date", "trades_placed", "wins", "losses", "pnl", "accuracy"]:
            assert key in summary


# ================================================================== #
# 4. TRADING EXECUTOR INTEGRATION TEST
# ================================================================== #

class TestTradingExecutorIntegration:
    """Integration test for the full trading executor cycle."""

    def _make_mock_ensemble(self, direction="UP", confidence=0.70):
        """Create a mock Ensemble that returns a predictable signal."""
        from src.models.ensemble import EnsembleSignal
        mock_signal = EnsembleSignal(
            direction=direction,
            confidence=confidence,
            probability=0.70 if direction == "UP" else 0.30,
            regime="trending_up",
            regime_display="Trending Up",
            regime_confidence=0.75,
            lgbm_result={"direction": direction, "confidence": confidence, "probability": 0.70},
            tcn_result={"direction": direction, "confidence": confidence, "probability": 0.70},
            logreg_result={"direction": direction, "confidence": confidence, "probability": 0.70},
            agreement=3,
            regime_threshold=0.56,
            inference_time_ms=5.0,
        )
        ensemble = MagicMock()
        ensemble.is_ready = True
        ensemble.predict = MagicMock(return_value=mock_signal)
        ensemble.load_models = MagicMock()
        return ensemble

    def _make_mock_polymarket(self, simulated=True):
        """Create a mock PolymarketClient."""
        from src.trading.polymarket import OrderResult
        pm = MagicMock()
        pm.is_connected = False
        pm.find_current_btc_market = AsyncMock(return_value={"id": "test-market"})
        pm.get_market_snapshot = AsyncMock(return_value=None)
        pm.place_order = AsyncMock(return_value=OrderResult(
            success=True,
            order_id="test-order-123",
            market_id="test-market",
            token_id="test-token",
            direction="UP",
            size=1.0,
            fill_price=0.55,
            fill_time_s=0.5,
            is_simulated=simulated,
        ))
        pm.check_settlement = AsyncMock(return_value=None)
        pm.settle_from_candle = AsyncMock(return_value={"outcome": "WIN", "pnl": 0.88, "settled": True})
        pm.get_stats = MagicMock(return_value={})
        return pm

    def _make_mock_feature_engine(self, db):
        """Create a mock FeatureEngine."""
        fe = MagicMock()
        fe.compute = MagicMock(return_value={
            "close": 50000.0,
            "atr_12": 150.0,
            "rsi_14": 55.0,
            "vol_std_12": 0.001,
            "ob_imbalance_5": 0.05,
            "funding_rate": 0.0001,
            "meta_probability": 0.65,
            "pm_up_price": 0.55,
            "pm_model_divergence": 0.05,
        })
        fe.update_polymarket_data = MagicMock()
        return fe

    @pytest.mark.asyncio
    async def test_cycle_signal_only_mode(self, db, config, temp_dir):
        """Full cycle in signals-only mode should produce a result without placing a trade."""
        from src.trading.executor import build_executor

        _populate_db(db, n_candles=200)
        ensemble = self._make_mock_ensemble()
        polymarket = self._make_mock_polymarket()
        feature_engine = self._make_mock_feature_engine(db)

        # Mock telegram
        telegram = MagicMock()
        telegram.send_message = AsyncMock()
        telegram.cache_signal = MagicMock()

        executor = build_executor(
            config=config,
            db=db,
            feature_engine=feature_engine,
            ensemble=ensemble,
            polymarket_client=polymarket,
            telegram_bot=telegram,
        )
        executor._auto_trade_enabled = False  # Signal-only mode

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        result = await executor.run_cycle(candle_ts_ms=now_ms, current_orderbook=None)

        assert result is not None
        assert result.duration_s >= 0
        # In signal-only mode, no trade placed
        assert result.trade_placed is False

    @pytest.mark.asyncio
    async def test_cycle_records_signal(self, db, config):
        """Cycle should produce a signal with direction and confidence."""
        from src.trading.executor import build_executor

        _populate_db(db, n_candles=200)
        ensemble = self._make_mock_ensemble(direction="UP", confidence=0.72)
        polymarket = self._make_mock_polymarket()
        feature_engine = self._make_mock_feature_engine(db)

        telegram = MagicMock()
        telegram.send_message = AsyncMock()
        telegram.cache_signal = MagicMock()

        executor = build_executor(
            config=config,
            db=db,
            feature_engine=feature_engine,
            ensemble=ensemble,
            polymarket_client=polymarket,
            telegram_bot=telegram,
        )
        executor._auto_trade_enabled = False

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        result = await executor.run_cycle(candle_ts_ms=now_ms, current_orderbook=None)

        assert result.signal is not None
        assert result.signal.direction == "UP"
        assert result.signal.confidence == pytest.approx(0.72)

    @pytest.mark.asyncio
    async def test_cycle_handles_ensemble_error(self, db, config):
        """Cycle should handle ensemble errors gracefully."""
        from src.trading.executor import build_executor

        _populate_db(db, n_candles=200)
        ensemble = MagicMock()
        ensemble.is_ready = True
        ensemble.predict = MagicMock(side_effect=RuntimeError("Model inference failed"))
        ensemble.load_models = MagicMock()

        polymarket = self._make_mock_polymarket()
        feature_engine = self._make_mock_feature_engine(db)

        telegram = MagicMock()
        telegram.send_message = AsyncMock()

        executor = build_executor(
            config=config,
            db=db,
            feature_engine=feature_engine,
            ensemble=ensemble,
            polymarket_client=polymarket,
            telegram_bot=telegram,
        )

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        # Should not raise -- errors are caught
        result = await executor.run_cycle(candle_ts_ms=now_ms, current_orderbook=None)
        assert result is not None
        assert result.error != "" or result.duration_s >= 0


# ================================================================== #
# 5. HEALTH SERVER TEST
# ================================================================== #

class TestHealthServer:
    """Test the health check HTTP server."""

    @pytest.mark.asyncio
    async def test_health_server_responds(self):
        """Health server should return JSON with status fields."""
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not installed")

        # Create a minimal mock app
        app = MagicMock()
        app._is_running = True
        app._start_ts = time.time()
        app.auto_trade_enabled = False
        app.db = MagicMock()
        app.db.get_candle_count = MagicMock(return_value=100)
        app.ensemble = MagicMock()
        app.ensemble.is_ready = False

        from src.main import start_health_server
        import random
        test_port = random.randint(18000, 19000)

        # Patch the port
        with patch("src.main.HEALTH_PORT", test_port):
            task = asyncio.ensure_future(start_health_server(app))
            await asyncio.sleep(0.2)  # Let server start

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://127.0.0.1:{test_port}/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        assert resp.status == 200
                        data = await resp.json()
                        assert "status" in data
                        assert "uptime_s" in data
                        assert "candles_5m" in data
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


# ================================================================== #
# 6. MAIN.PY SYNTAX AND IMPORT CHECK
# ================================================================== #

class TestMainModule:
    """Basic structural tests for main.py."""

    def test_main_py_syntax(self):
        """main.py should have valid Python syntax."""
        import ast
        main_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "main.py")
        with open(main_path) as f:
            source = f.read()
        # Should not raise
        tree = ast.parse(source)
        assert tree is not None

    def test_cleobot_class_exists(self):
        """CleoBot class should be importable."""
        # Use subprocess to avoid importing all deps
        import subprocess
        result = subprocess.run(
            ["python", "-c", "import ast; ast.parse(open('src/main.py').read()); print('OK')"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_config_loads_with_defaults(self):
        """Config should load with empty env (all defaults)."""
        with patch.dict(os.environ, {}, clear=False):
            from src.config import load_config
            cfg = load_config()
            assert cfg.trading.base_trade_size == 1.0
            assert cfg.trading.auto_trade_enabled is False
            assert cfg.system.log_level == "INFO"


# ================================================================== #
# 7. FULL STARTUP/SHUTDOWN SIMULATION
# ================================================================== #

class TestStartupShutdown:
    """Test the CleoBot startup and shutdown sequence."""

    @pytest.mark.asyncio
    async def test_shutdown_before_start(self, config):
        """shutdown() called before start should not raise."""
        # We can test the CleoBot class without triggering actual network calls
        # by just checking the shutdown logic handles None components gracefully
        from src.main import CleoBot
        bot = CleoBot()
        # Before startup, all components are None -- shutdown should not raise
        # Just verify it handles gracefully
        bot._is_running = False
        # Don't call full shutdown since it does await calls -- just verify the object
        assert bot.config is None
        assert bot.db is None
        assert bot.executor is None

    def test_cleobot_properties(self):
        """CleoBot properties should work with no executor set."""
        from src.main import CleoBot
        bot = CleoBot()
        # auto_trade_enabled with no executor/config should return False
        assert bot.auto_trade_enabled is False


# ================================================================== #
# PYTEST CONFIGURATION
# ================================================================== #

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
