"""Backtesting handlers for CleoBot Telegram bot.

Integrates with BacktestEngine and BacktestReport for full walk-forward
backtesting with filter analysis and model comparison reports.
"""

import asyncio
from datetime import datetime, timezone, timedelta

from telegram import Update
from telegram.ext import ContextTypes

from src.telegram_bot.keyboards import backtest_keyboard
from src.utils.logger import get_logger

logger = get_logger("telegram_bot.handlers.backtest")


async def handle_backtest_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "Backtesting\n\nRun historical simulations on stored candles.",
        reply_markup=backtest_keyboard(),
    )


async def _run_backtest(
    update: Update, context: ContextTypes.DEFAULT_TYPE, days: int
) -> None:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "\u23F3 Running {}-day backtest... This may take a moment.".format(days)
    )
    chat_id = update.effective_chat.id
    bot_app = context.bot_data.get("cleobot")

    if not (bot_app and hasattr(bot_app, "db") and bot_app.db):
        await context.bot.send_message(
            chat_id=chat_id,
            text="Backtest unavailable: database not ready.",
            reply_markup=backtest_keyboard(),
        )
        return

    try:
        from src.backtest.engine import BacktestEngine
        from src.backtest.report import BacktestReport

        ensemble = getattr(bot_app, "ensemble", None)
        engine = BacktestEngine(db=bot_app.db, ensemble=ensemble)

        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: engine.run(days=days))

        report = BacktestReport(result)
        summary_text = report.summary()

        # Truncate if too long for Telegram
        if len(summary_text) > 3800:
            summary_text = summary_text[:3800] + "\n... (truncated)"

        await context.bot.send_message(
            chat_id=chat_id,
            text=summary_text,
            reply_markup=backtest_keyboard(),
        )

        # Cache result for follow-up reports
        context.bot_data["last_backtest_result"] = result

    except Exception as e:
        logger.error("Backtest error: {}".format(e), exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            text="Backtest failed: {}".format(e),
            reply_markup=backtest_keyboard(),
        )


async def handle_backtest_7d(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _run_backtest(update, context, days=7)


async def handle_backtest_30d(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _run_backtest(update, context, days=30)


async def handle_backtest_compare(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("\u23F3 Running model comparison backtest (7d)...")
    chat_id = update.effective_chat.id
    bot_app = context.bot_data.get("cleobot")

    if not (bot_app and hasattr(bot_app, "db") and bot_app.db):
        await context.bot.send_message(
            chat_id=chat_id,
            text="Backtest unavailable: database not ready.",
            reply_markup=backtest_keyboard(),
        )
        return

    try:
        from src.backtest.engine import BacktestEngine
        from src.backtest.report import BacktestReport

        ensemble = getattr(bot_app, "ensemble", None)
        engine = BacktestEngine(db=bot_app.db, ensemble=ensemble)

        loop = asyncio.get_event_loop()
        comparison = await loop.run_in_executor(None, lambda: engine.compare_models(days=7))

        from src.backtest.report import format_model_comparison
        text = format_model_comparison(comparison)

        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=backtest_keyboard(),
        )
    except Exception as e:
        logger.error("Model comparison error: {}".format(e), exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            text="Model comparison failed: {}".format(e),
            reply_markup=backtest_keyboard(),
        )


async def handle_backtest_filters(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id
    bot_app = context.bot_data.get("cleobot")

    # Try to use cached result first (much faster)
    cached = context.bot_data.get("last_backtest_result")
    if cached is not None:
        try:
            from src.backtest.report import BacktestReport
            report = BacktestReport(cached)
            text = report.filter_analysis()
            await query.edit_message_text(text, reply_markup=backtest_keyboard())
            return
        except Exception as e:
            logger.warning("Failed to use cached backtest result: {}".format(e))

    # No cached result -- run a fresh 7d backtest
    if not (bot_app and hasattr(bot_app, "db") and bot_app.db):
        await query.edit_message_text(
            "Filter analysis unavailable: database not ready.",
            reply_markup=backtest_keyboard(),
        )
        return

    await query.edit_message_text("\u23F3 Running filter analysis (7d backtest)...")

    try:
        from src.backtest.engine import BacktestEngine
        from src.backtest.report import BacktestReport

        ensemble = getattr(bot_app, "ensemble", None)
        engine = BacktestEngine(db=bot_app.db, ensemble=ensemble)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: engine.run(days=7))

        context.bot_data["last_backtest_result"] = result
        report = BacktestReport(result)
        text = report.filter_analysis()

        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=backtest_keyboard(),
        )
    except Exception as e:
        logger.error("Filter analysis error: {}".format(e), exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            text="Filter analysis failed: {}".format(e),
            reply_markup=backtest_keyboard(),
        )
