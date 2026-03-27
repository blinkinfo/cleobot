"""Trading control handlers for CleoBot Telegram bot."""

from telegram import Update
from telegram.ext import ContextTypes

from src.telegram_bot.keyboards import trading_keyboard
from src.utils.logger import get_logger

logger = get_logger("telegram_bot.handlers.trading")


async def handle_trading_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        text="Trading Controls\n\nManage auto-trading, pause cycles, and set trade size.",
        reply_markup=trading_keyboard(),
    )


async def handle_trading_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    if bot_app is None:
        await query.edit_message_text("Bot not initialised.", reply_markup=trading_keyboard())
        return
    bot_app.auto_trade_enabled = True
    logger.info("Auto-trading ENABLED via Telegram.")
    await query.edit_message_text(
        "\u2705 Auto-trading ENABLED\n\nThe bot will place trades on the next qualifying signal.",
        reply_markup=trading_keyboard(),
    )


async def handle_trading_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    if bot_app is None:
        await query.edit_message_text("Bot not initialised.", reply_markup=trading_keyboard())
        return
    bot_app.auto_trade_enabled = False
    logger.info("Auto-trading DISABLED via Telegram.")
    await query.edit_message_text(
        "\U0001F6D1 Auto-trading DISABLED\n\nNo new trades will be placed until re-enabled.",
        reply_markup=trading_keyboard(),
    )


async def handle_trading_pause(update: Update, context: ContextTypes.DEFAULT_TYPE, cycles: int) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    if bot_app and hasattr(bot_app, "executor") and bot_app.executor:
        bot_app.executor.signal_filter._pause_cycles_remaining = cycles
        logger.info(f"Trading paused for {cycles} cycle(s) via Telegram.")
        await query.edit_message_text(
            f"\u23F8\uFE0F Trading paused for {cycles} cycle(s).\n\nWill resume automatically.",
            reply_markup=trading_keyboard(),
        )
    else:
        await query.edit_message_text("Executor not available.", reply_markup=trading_keyboard())


async def handle_trading_pause_1(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_trading_pause(update, context, cycles=1)


async def handle_trading_pause_3(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_trading_pause(update, context, cycles=3)


async def handle_trading_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    if bot_app is None:
        await query.edit_message_text("Bot not initialised.", reply_markup=trading_keyboard())
        return
    auto = getattr(bot_app, "auto_trade_enabled", False)
    auto_icon = "\u2705" if auto else "\U0001F534"
    lines = [
        "Trading Status",
        "=" * 25,
        f"Auto-Trade:  {auto_icon} {'ON' if auto else 'OFF'}",
    ]
    if hasattr(bot_app, "executor") and bot_app.executor:
        stats = bot_app.executor.get_stats()
        risk = stats.get("risk_status", {})
        filt = stats.get("filter_state", {})
        lines += [
            f"Cycles:      {stats.get('total_cycles', 0)}",
            f"Trades:      {stats.get('total_trades', 0)}",
            f"Skips:       {stats.get('total_skips', 0)}",
            f"Pending:     {stats.get('pending_settlements', 0)}",
            "",
            "Risk:",
            f"  Daily PnL: ${risk.get('daily_pnl', 0):+.2f}",
            f"  W/L Today: {risk.get('wins_today', 0)}W / {risk.get('losses_today', 0)}L",
            "",
            "Filter State:",
            f"  Paused:    {filt.get('pause_cycles_remaining', 0) > 0}",
            f"  Streak:    {filt.get('streak_requires_manual_restart', False)}",
        ]
    await query.edit_message_text("\n".join(lines), reply_markup=trading_keyboard())


async def handle_set_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    context.user_data["awaiting"] = "trade_size"
    await query.edit_message_text(
        "Enter new base trade size in USD (e.g. 2.5):\n\nReply with /setsize <amount>",
    )


async def cmd_setsize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []
    if not args:
        await update.message.reply_text("Usage: /setsize <amount>  e.g. /setsize 2.5")
        return
    try:
        size = float(args[0])
        if size <= 0 or size > 100:
            raise ValueError("out of range")
    except ValueError:
        await update.message.reply_text("Invalid size. Enter a positive number up to 100.")
        return
    bot_app = context.bot_data.get("cleobot")
    if bot_app and hasattr(bot_app, "executor") and bot_app.executor:
        bot_app.executor.risk_manager.set_trade_size(size)
        logger.info(f"Trade size set to ${size:.2f} via Telegram.")
        await update.message.reply_text(
            f"\u2705 Base trade size set to ${size:.2f}",
            reply_markup=trading_keyboard(),
        )
    else:
        await update.message.reply_text("Executor not available.")
