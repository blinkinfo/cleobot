"""CleoBot Telegram bot -- initialisation, handler registration, error handler.

This module creates and configures the python-telegram-bot Application with:
  - Command handlers (/start, /help, /status, /menu, /setsize)
  - Callback query router for all 8 submenus and action buttons
  - Error handler that logs and notifies via Telegram
  - send_message() helper used by the trading executor for auto-notifications

Usage (from main.py / orchestrator):
    bot = CleoBotTelegram(config)
    await bot.start(cleobot_app=app)      # sets bot_data and sends startup msg
    await bot.stop()                       # sends shutdown msg and stops polling
"""

import time
import asyncio
import traceback
from typing import Optional, Any

from telegram import Update, Bot
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.error import TelegramError

from src.config import TelegramConfig
from src.telegram_bot.keyboards import main_menu_keyboard
from src.telegram_bot.handlers.trading import (
    handle_trading_menu,
    handle_trading_start,
    handle_trading_stop,
    handle_trading_pause_1,
    handle_trading_pause_3,
    handle_trading_status,
    handle_set_size,
    cmd_setsize,
)
from src.telegram_bot.handlers.signals import (
    handle_signals_menu,
    handle_signals_next,
    handle_signals_last5,
    handle_signals_breakdown,
    handle_signals_regime,
    handle_signals_features,
    handle_signal_detail,
    handle_signal_force,
)
from src.telegram_bot.handlers.performance import (
    handle_performance_menu,
    handle_perf_today,
    handle_perf_weekly,
    handle_perf_monthly,
    handle_perf_hourly,
    handle_perf_streaks,
    handle_perf_equity,
)
from src.telegram_bot.handlers.models import (
    handle_models_menu,
    handle_models_health,
    handle_models_retrain,
    handle_models_retrain_confirm,
    handle_models_features,
    handle_models_compare,
    handle_models_regime_history,
)
from src.telegram_bot.handlers.backtest import (
    handle_backtest_menu,
    handle_backtest_7d,
    handle_backtest_30d,
    handle_backtest_compare,
    handle_backtest_filters,
)
from src.telegram_bot.handlers.risk import (
    handle_risk_menu,
    handle_risk_drawdown,
    handle_risk_limits,
    handle_risk_update,
    handle_risk_exposure,
)
from src.telegram_bot.handlers.system import (
    handle_system_menu,
    handle_system_latency,
    handle_system_uptime,
    handle_system_logs,
    handle_system_errors,
    handle_system_db,
)
from src.utils.logger import get_logger

logger = get_logger("telegram_bot.bot")


# ------------------------------------------------------------------ #
# COMMAND HANDLERS
# ------------------------------------------------------------------ #

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command -- show main menu."""
    await update.message.reply_text(
        "\U0001F916 CleoBot Control Panel\n\nSelect a section:",
        reply_markup=main_menu_keyboard(),
    )


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /menu command -- show main menu."""
    await update.message.reply_text(
        "CleoBot Menu",
        reply_markup=main_menu_keyboard(),
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    help_text = (
        "CleoBot Commands\n"
        "=" * 25 + "\n"
        "/start   - Open control panel\n"
        "/menu    - Show main menu\n"
        "/status  - Quick trading status\n"
        "/setsize <n> - Set base trade size\n"
        "/help    - This help message\n\n"
        "Use the inline buttons to navigate all sections.\n\n"
        "Sections: Trading, Signals, Performance,\n"
        "Models, Backtest, Risk, System, Settings"
    )
    await update.message.reply_text(help_text)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command -- quick status summary."""
    bot_app = context.bot_data.get("cleobot")
    lines = ["\U0001F916 CleoBot Status", "=" * 25]
    if bot_app is None:
        lines.append("Bot not initialised.")
    else:
        auto = getattr(bot_app, "auto_trade_enabled", False)
        auto_icon = "\u2705" if auto else "\U0001F534"
        lines.append(f"Auto-Trade: {auto_icon} {'ON' if auto else 'OFF'}")
        if hasattr(bot_app, "executor") and bot_app.executor:
            stats = bot_app.executor.get_stats()
            risk = stats.get("risk_status", {})
            lines += [
                f"Cycles:     {stats.get('total_cycles', 0)}",
                f"Trades:     {stats.get('total_trades', 0)}",
                f"Skips:      {stats.get('total_skips', 0)}",
                f"Daily PnL:  ${risk.get('daily_pnl', 0):+.2f}",
                f"W/L Today:  {risk.get('wins_today', 0)}W / {risk.get('losses_today', 0)}L",
            ]
        if hasattr(bot_app, "ensemble") and bot_app.ensemble:
            ready = bot_app.ensemble.is_ready
            lines.append(f"Models:     {'\u2705 Ready' if ready else '\u26A0\uFE0F Not ready'}")
    await update.message.reply_text("\n".join(lines), reply_markup=main_menu_keyboard())


# ------------------------------------------------------------------ #
# MAIN MENU ROUTER
# ------------------------------------------------------------------ #

async def handle_menu_main(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Return to main menu."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "\U0001F916 CleoBot Control Panel\n\nSelect a section:",
        reply_markup=main_menu_keyboard(),
    )


# ------------------------------------------------------------------ #
# CALLBACK QUERY ROUTER
# ------------------------------------------------------------------ #

async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route all callback queries to the appropriate handler."""
    query = update.callback_query
    data = query.data or ""

    # Main menu
    if data == "menu:main":
        await handle_menu_main(update, context)
    elif data == "menu:trading":
        await handle_trading_menu(update, context)
    elif data == "menu:signals":
        await handle_signals_menu(update, context)
    elif data == "menu:performance":
        await handle_performance_menu(update, context)
    elif data == "menu:models":
        await handle_models_menu(update, context)
    elif data == "menu:backtest":
        await handle_backtest_menu(update, context)
    elif data == "menu:risk":
        await handle_risk_menu(update, context)
    elif data == "menu:system":
        await handle_system_menu(update, context)
    elif data == "menu:settings":
        await query.answer()
        await query.edit_message_text(
            "Settings\n\nConfigure thresholds and notifications via environment variables."
            "\n\nAvailable commands:\n/setsize <amount>  - Set base trade size",
            reply_markup=main_menu_keyboard(),
        )

    # Trading
    elif data == "trading:start":    await handle_trading_start(update, context)
    elif data == "trading:stop":     await handle_trading_stop(update, context)
    elif data == "trading:pause_1":  await handle_trading_pause_1(update, context)
    elif data == "trading:pause_3":  await handle_trading_pause_3(update, context)
    elif data == "trading:status":   await handle_trading_status(update, context)
    elif data == "trading:set_size": await handle_set_size(update, context)

    # Signals
    elif data == "signals:next":      await handle_signals_next(update, context)
    elif data == "signals:last5":     await handle_signals_last5(update, context)
    elif data == "signals:breakdown": await handle_signals_breakdown(update, context)
    elif data == "signals:regime":    await handle_signals_regime(update, context)
    elif data == "signals:features":  await handle_signals_features(update, context)

    # Signal cards (dynamic IDs)
    elif data.startswith("signal:detail:"):
        await handle_signal_detail(update, context)
    elif data.startswith("signal:force:"):
        await handle_signal_force(update, context)

    # Performance
    elif data == "perf:today":   await handle_perf_today(update, context)
    elif data == "perf:weekly":  await handle_perf_weekly(update, context)
    elif data == "perf:monthly": await handle_perf_monthly(update, context)
    elif data == "perf:hourly":  await handle_perf_hourly(update, context)
    elif data == "perf:streaks": await handle_perf_streaks(update, context)
    elif data == "perf:equity":  await handle_perf_equity(update, context)

    # Models
    elif data == "models:health":         await handle_models_health(update, context)
    elif data == "models:retrain":        await handle_models_retrain(update, context)
    elif data == "models:features":       await handle_models_features(update, context)
    elif data == "models:compare":        await handle_models_compare(update, context)
    elif data == "models:regime_history": await handle_models_regime_history(update, context)

    # Backtest
    elif data == "backtest:7d":      await handle_backtest_7d(update, context)
    elif data == "backtest:30d":     await handle_backtest_30d(update, context)
    elif data == "backtest:compare": await handle_backtest_compare(update, context)
    elif data == "backtest:filters": await handle_backtest_filters(update, context)

    # Risk
    elif data == "risk:drawdown": await handle_risk_drawdown(update, context)
    elif data == "risk:limits":   await handle_risk_limits(update, context)
    elif data == "risk:update":   await handle_risk_update(update, context)
    elif data == "risk:exposure": await handle_risk_exposure(update, context)

    # System
    elif data == "system:latency": await handle_system_latency(update, context)
    elif data == "system:uptime":  await handle_system_uptime(update, context)
    elif data == "system:logs":    await handle_system_logs(update, context)
    elif data == "system:errors":  await handle_system_errors(update, context)
    elif data == "system:db":      await handle_system_db(update, context)

    # Confirm dialogs
    elif data == "confirm:retrain:yes":
        await handle_models_retrain_confirm(update, context)
    elif data.startswith("confirm:") and data.endswith(":no"):
        await query.answer("Cancelled.")
        await query.edit_message_text(
            "Action cancelled.",
            reply_markup=main_menu_keyboard(),
        )

    else:
        await query.answer(f"Unknown action: {data}")
        logger.warning(f"Unknown callback data: {data}")


# ------------------------------------------------------------------ #
# ERROR HANDLER
# ------------------------------------------------------------------ #

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors and send a notification to Telegram."""
    error = context.error
    tb = traceback.format_exception(type(error), error, error.__traceback__)
    tb_str = "".join(tb)[-500:]  # last 500 chars to avoid Telegram message limits

    logger.error(f"Telegram bot error: {error}\n{tb_str}")

    # Try to notify via Telegram
    chat_id = context.bot_data.get("chat_id")
    if chat_id:
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"\u26A0\uFE0F Bot Error:\n"
                    f"{type(error).__name__}: {str(error)[:300]}"
                ),
            )
        except Exception:
            pass  # Don't let error handler itself crash


# ------------------------------------------------------------------ #
# CleoBotTelegram -- main interface class
# ------------------------------------------------------------------ #

class CleoBotTelegram:
    """Telegram bot wrapper for CleoBot.

    Wraps python-telegram-bot Application with a clean interface:
      - start() / stop() lifecycle
      - send_message() for async notifications from the trading executor
      - Stores references to cleobot app objects in bot_data for handlers
    """

    def __init__(self, config: TelegramConfig):
        """
        Args:
            config: TelegramConfig with bot_token and chat_id.
        """
        self.config = config
        self._app: Optional[Application] = None
        self._is_running = False

        if not config.is_configured:
            logger.warning(
                "Telegram not configured (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID). "
                "Bot will run in silent mode."
            )

    # ---------------------------------------------------------------- #
    # LIFECYCLE
    # ---------------------------------------------------------------- #

    async def start(
        self,
        cleobot_app: Optional[Any] = None,
        db_path: str = "/data/cleobot.db",
        log_path: str = "/data/cleobot.log",
    ) -> None:
        """Build and start the Telegram bot.

        Args:
            cleobot_app: The main CleoBot application object (stored in bot_data).
            db_path: Path to SQLite database file (for system handler).
            log_path: Path to log file (for system handler).
        """
        if not self.config.is_configured:
            logger.info("Telegram not configured -- skipping bot start.")
            return

        # Build Application
        self._app = (
            ApplicationBuilder()
            .token(self.config.bot_token)
            .build()
        )

        # Store shared references in bot_data for handlers
        self._app.bot_data["cleobot"] = cleobot_app
        self._app.bot_data["chat_id"] = self.config.chat_id
        self._app.bot_data["db_path"] = db_path
        self._app.bot_data["log_path"] = log_path
        self._app.bot_data["start_ts"] = time.time()
        self._app.bot_data["last_signal"] = None
        self._app.bot_data["regime_history"] = []
        self._app.bot_data["last_retrain_ts"] = None

        self._register_handlers()
        self._app.add_error_handler(error_handler)

        # Start polling
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )
        self._is_running = True
        logger.info("Telegram bot started (polling).")

    async def stop(self) -> None:
        """Stop the Telegram bot cleanly."""
        if self._app and self._is_running:
            try:
                await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception as e:
                logger.error(f"Error stopping Telegram bot: {e}")
            self._is_running = False
            logger.info("Telegram bot stopped.")

    # ---------------------------------------------------------------- #
    # HANDLER REGISTRATION
    # ---------------------------------------------------------------- #

    def _register_handlers(self) -> None:
        """Register all command and callback handlers."""
        app = self._app

        # Commands
        app.add_handler(CommandHandler("start",   cmd_start))
        app.add_handler(CommandHandler("menu",    cmd_menu))
        app.add_handler(CommandHandler("help",    cmd_help))
        app.add_handler(CommandHandler("status",  cmd_status))
        app.add_handler(CommandHandler("setsize", cmd_setsize))

        # All callback queries routed through a single dispatcher
        app.add_handler(CallbackQueryHandler(callback_router))

        logger.debug("All handlers registered.")

    # ---------------------------------------------------------------- #
    # SEND MESSAGE (for trading executor notifications)
    # ---------------------------------------------------------------- #

    async def send_message(
        self,
        text: str,
        reply_markup=None,
        max_retries: int = 3,
    ) -> bool:
        """Send a message to the configured chat.

        Called by TradingExecutor and auto-notification functions.

        Args:
            text: Message text.
            reply_markup: Optional InlineKeyboardMarkup.
            max_retries: Number of retry attempts.

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self.config.is_configured:
            logger.debug(f"[TELEGRAM SILENT] {text[:80]}")
            return False

        bot: Bot = self._app.bot if self._app else None
        if bot is None:
            logger.warning("Telegram bot not started -- cannot send message.")
            return False

        for attempt in range(max_retries):
            try:
                await bot.send_message(
                    chat_id=self.config.chat_id,
                    text=text,
                    reply_markup=reply_markup,
                )
                return True
            except TelegramError as e:
                logger.warning(
                    f"Telegram send failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Unexpected Telegram error: {e}")
                break
        return False

    # ---------------------------------------------------------------- #
    # CACHE HELPERS (called by executor to keep bot_data fresh)
    # ---------------------------------------------------------------- #

    def cache_signal(self, signal_dict: dict) -> None:
        """Cache the latest signal dict in bot_data for the breakdown handler."""
        if self._app:
            self._app.bot_data["last_signal"] = signal_dict

    def record_regime_change(
        self,
        old_regime: str,
        new_regime: str,
        new_display: str,
        ts: str,
    ) -> None:
        """Append a regime change entry to bot_data history."""
        if self._app:
            history = self._app.bot_data.get("regime_history", [])
            history.append({"old": old_regime, "new": new_display, "ts": ts})
            if len(history) > 100:
                history.pop(0)
            self._app.bot_data["regime_history"] = history

    def record_retrain_ts(self, ts: str) -> None:
        """Record the last retrain timestamp."""
        if self._app:
            self._app.bot_data["last_retrain_ts"] = ts

    # ---------------------------------------------------------------- #
    # PROPERTIES
    # ---------------------------------------------------------------- #

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def bot(self) -> Optional[Bot]:
        return self._app.bot if self._app else None
