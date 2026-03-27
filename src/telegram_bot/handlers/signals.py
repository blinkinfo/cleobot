"""Signal analysis handlers for CleoBot Telegram bot."""

from telegram import Update
from telegram.ext import ContextTypes

from src.telegram_bot.keyboards import signals_keyboard, main_menu_keyboard
from src.utils.logger import get_logger

logger = get_logger("telegram_bot.handlers.signals")


async def handle_signals_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "Signals Analysis\n\nView recent signals, model breakdown, and regime info.",
        reply_markup=signals_keyboard(),
    )


async def handle_signals_next(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    lines = ["Next Signal Preview", "=" * 25]
    if bot_app and hasattr(bot_app, "executor") and bot_app.executor:
        stats = bot_app.executor.get_stats()
        filt = stats.get("filter_state", {})
        lines += [
            "Waiting for next 5-min candle close.",
            "",
            f"Filter state:",
            f"  Paused: {filt.get('pause_cycles_remaining', 0) > 0}",
            f"  Streak (manual restart): {filt.get('streak_requires_manual_restart', False)}",
            f"  Pause remaining: {filt.get('pause_cycles_remaining', 0)} cycle(s)",
        ]
    else:
        lines.append("Executor not running.")
    await query.edit_message_text("\n".join(lines), reply_markup=signals_keyboard())


async def handle_signals_last5(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    lines = ["Last 5 Signals", "=" * 25]
    if bot_app and hasattr(bot_app, "db") and bot_app.db:
        try:
            trades = bot_app.db.get_recent_settled_trades(limit=5)
            unsettled = bot_app.db.get_unsettled_trades()
            all_recent = list(unsettled) + list(trades)
            all_recent = all_recent[:5]
            if not all_recent:
                lines.append("No recent signals found.")
            else:
                for t in all_recent:
                    direction = t.get("direction", "?")
                    outcome = t.get("outcome", "PENDING")
                    pnl = t.get("pnl", 0.0) or 0.0
                    created = t.get("created_at", "")[:16] if t.get("created_at") else "?"
                    dir_icon = "\U0001F7E2" if direction == "UP" else "\U0001F534"
                    out_icon = "\u2705" if outcome == "WIN" else ("\u274C" if outcome == "LOSS" else "\u23F3")
                    lines.append(f"{dir_icon} {direction}  {out_icon} {outcome}  ${pnl:+.2f}  {created}")
        except Exception as e:
            lines.append(f"Error fetching signals: {e}")
    else:
        lines.append("Database not available.")
    await query.edit_message_text("\n".join(lines), reply_markup=signals_keyboard())


async def handle_signals_breakdown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    lines = ["Model Breakdown (Last Signal)", "=" * 30]
    cached = context.bot_data.get("last_signal")
    if cached:
        s = cached
        lines += [
            f"Direction:   {s.get('direction', '?')}",
            f"Confidence:  {s.get('confidence', 0):.1%}",
            f"Probability: {s.get('probability', 0):.3f}",
            "",
            "Models:",
            f"  LightGBM: {s.get('lgbm', {}).get('direction', '?')} ({s.get('lgbm', {}).get('confidence', 0):.1%})",
            f"  TCN:      {s.get('tcn', {}).get('direction', '?')} ({s.get('tcn', {}).get('confidence', 0):.1%})",
            f"  LogReg:   {s.get('logreg', {}).get('direction', '?')} ({s.get('logreg', {}).get('confidence', 0):.1%})",
            f"  Agreement: {s.get('agreement', 0)}/3",
            "",
            f"Regime:    {s.get('regime_display', '?')}",
            f"Threshold: {s.get('regime_threshold', 0):.1%}",
            f"Infer ms:  {s.get('inference_time_ms', 0):.1f}ms",
        ]
    else:
        lines.append("No signal cached yet. Wait for the next cycle.")
    await query.edit_message_text("\n".join(lines), reply_markup=signals_keyboard())


async def handle_signals_regime(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    cached = context.bot_data.get("last_signal")
    lines = ["Current Regime", "=" * 25]
    if cached:
        lines += [
            f"Regime:      {cached.get('regime_display', '?')}",
            f"Confidence:  {cached.get('regime_confidence', 0):.1%}",
            f"Threshold:   {cached.get('regime_threshold', 0):.1%}",
        ]
    else:
        lines.append("No regime data available yet.")
    await query.edit_message_text("\n".join(lines), reply_markup=signals_keyboard())


async def handle_signals_features(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    lines = ["Top Feature Importance", "=" * 28]
    if bot_app and hasattr(bot_app, "ensemble") and bot_app.ensemble:
        try:
            rankings = bot_app.ensemble.get_feature_rankings()
            top = rankings[:10]
            for i, r in enumerate(top, 1):
                name = r.get("feature", "?")[:20]
                imp = r.get("lgbm_importance", 0.0)
                lines.append(f"  {i:>2}. {name:<20} {imp:.4f}")
        except Exception as e:
            lines.append(f"Error: {e}")
    else:
        lines.append("Models not loaded.")
    await query.edit_message_text("\n".join(lines), reply_markup=signals_keyboard())


async def handle_signal_detail(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data  # signal:detail:<id>
    parts = data.split(":")
    signal_id = parts[2] if len(parts) > 2 else "?"
    bot_app = context.bot_data.get("cleobot")
    lines = [f"Signal #{signal_id} Details", "=" * 28]
    if bot_app and hasattr(bot_app, "db") and bot_app.db:
        try:
            trade = bot_app.db.get_trade_by_id(int(signal_id))
            if trade:
                lines += [
                    f"Direction:  {trade.get('direction', '?')}",
                    f"Size:       ${trade.get('trade_size', 0):.2f}",
                    f"Entry:      {trade.get('entry_price', 0):.3f}",
                    f"Outcome:    {trade.get('outcome', 'PENDING')}",
                    f"PnL:        ${trade.get('pnl', 0) or 0:+.2f}",
                    f"Created:    {trade.get('created_at', '?')[:19]}",
                    f"Simulated:  {trade.get('is_simulated', False)}",
                ]
            else:
                lines.append("Trade not found.")
        except Exception as e:
            lines.append(f"Error: {e}")
    else:
        lines.append("Database not available.")
    await query.edit_message_text("\n".join(lines), reply_markup=signals_keyboard())


async def handle_signal_force(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "\u26A0\uFE0F Force-trading is not supported for past signals.\n"
        "Enable auto-trade to allow the bot to trade on future signals.",
        reply_markup=signals_keyboard(),
    )
