"""Risk management handlers for CleoBot Telegram bot."""

from telegram import Update
from telegram.ext import ContextTypes

from src.telegram_bot.keyboards import risk_keyboard
from src.utils.logger import get_logger

logger = get_logger("telegram_bot.handlers.risk")


async def handle_risk_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "Risk Management\n\nView drawdown, daily limits, and exposure.",
        reply_markup=risk_keyboard(),
    )


async def handle_risk_drawdown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    lines = ["Drawdown & Streak", "=" * 28]
    if bot_app and hasattr(bot_app, "executor") and bot_app.executor:
        risk = bot_app.executor.risk_manager.get_status()
        lines += [
            f"Daily PnL:         ${risk.daily_pnl:+.2f}",
            f"Daily Loss Limit:  -${risk.daily_loss_limit:.2f}",
            f"Remaining Budget:  ${max(0, risk.daily_loss_limit + risk.daily_pnl):.2f}",
            f"Consec Losses:     {risk.consecutive_losses}",
            f"Max Consec Limit:  {risk.max_consecutive_losses}",
            f"Open Exposure:     ${risk.open_exposure:.2f}",
            f"Max Exposure:      ${risk.max_exposure:.2f}",
            f"Wins Today:        {risk.wins_today}",
            f"Losses Today:      {risk.losses_today}",
        ]
    else:
        lines.append("Risk manager not available.")
    await query.edit_message_text("\n".join(lines), reply_markup=risk_keyboard())


async def handle_risk_limits(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    lines = ["Daily Risk Limits", "=" * 28]
    if bot_app and hasattr(bot_app, "executor") and bot_app.executor:
        risk = bot_app.executor.risk_manager.get_status()
        cfg = getattr(bot_app.executor.risk_manager, "config", None)
        base_size = getattr(bot_app.executor.risk_manager, "_base_trade_size",
                            getattr(cfg, "base_trade_size", 1.0) if cfg else 1.0)
        max_size = getattr(cfg, "max_trade_size", 3.0) if cfg else 3.0
        lines += [
            f"Max Daily Loss:    -${risk.daily_loss_limit:.2f}",
            f"Max Consec Losses: {risk.max_consecutive_losses}",
            f"Max Open Exposure: ${risk.max_exposure:.2f}",
            f"Base Trade Size:   ${base_size:.2f}",
            f"Max Trade Size:    ${max_size:.2f}",
            "",
            "Use /setsize <amount> to change base trade size.",
        ]
    else:
        lines.append("Risk manager not available.")
    await query.edit_message_text("\n".join(lines), reply_markup=risk_keyboard())


async def handle_risk_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "To update risk limits, use these commands:\n\n"
        "/setsize <amount>    Set base trade size\n\n"
        "Advanced limits can be configured via environment variables\n"
        "(MAX_DAILY_LOSS, MAX_CONSECUTIVE_LOSSES) and restarting the bot.",
        reply_markup=risk_keyboard(),
    )


async def handle_risk_exposure(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    lines = ["Current Exposure", "=" * 25]
    if bot_app and hasattr(bot_app, "executor") and bot_app.executor:
        risk = bot_app.executor.risk_manager.get_status()
        pending = len(bot_app.executor._pending_settlements)
        lines += [
            f"Open Exposure:   ${risk.open_exposure:.2f}",
            f"Max Exposure:    ${risk.max_exposure:.2f}",
            f"Utilisation:     {risk.open_exposure / max(risk.max_exposure, 0.01):.1%}",
            f"Pending Trades:  {pending}",
        ]
    else:
        lines.append("Risk manager not available.")
    await query.edit_message_text("\n".join(lines), reply_markup=risk_keyboard())
