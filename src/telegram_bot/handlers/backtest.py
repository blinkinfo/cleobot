"""Backtesting handlers for CleoBot Telegram bot."""

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
        f"\u23F3 Running {days}-day backtest... This may take a moment."
    )
    bot_app = context.bot_data.get("cleobot")
    if not (bot_app and hasattr(bot_app, "db") and bot_app.db and
            hasattr(bot_app, "ensemble") and bot_app.ensemble and
            bot_app.ensemble.is_ready):
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Backtest unavailable: DB or models not ready.",
            reply_markup=backtest_keyboard(),
        )
        return
    try:
        since_ts = int(
            (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
        )
        candles = bot_app.db.get_candles("candles_5m", limit=days * 288, since=since_ts)
        if len(candles) < 50:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Insufficient candle data for {days}-day backtest ({len(candles)} candles).",
                reply_markup=backtest_keyboard(),
            )
            return

        wins = 0
        losses = 0
        total_pnl = 0.0
        skips = 0

        import pandas as pd
        df = pd.DataFrame(candles)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = df[col].astype(float)
        df = df.sort_values("timestamp").reset_index(drop=True)

        WIN_PNL = 0.88
        LOSS_PNL = -1.00

        for i in range(60, len(df) - 1):
            window = df.iloc[max(0, i - 100):i + 1]
            try:
                features = bot_app.feature_engine.compute_from_df(window)
                if not features:
                    skips += 1
                    continue
                signal = bot_app.ensemble.predict(
                    features=features,
                    df_5m=window,
                )
                if signal.confidence < signal.regime_threshold:
                    skips += 1
                    continue
                next_candle = df.iloc[i + 1]
                candle_up = float(next_candle["close"]) > float(next_candle["open"])
                correct = (signal.direction == "UP" and candle_up) or \
                          (signal.direction == "DOWN" and not candle_up)
                if correct:
                    wins += 1
                    total_pnl += WIN_PNL
                else:
                    losses += 1
                    total_pnl += LOSS_PNL
            except Exception:
                skips += 1
                continue

        total = wins + losses
        accuracy = wins / total if total > 0 else 0.0
        pnl_icon = "\U0001F7E2" if total_pnl >= 0 else "\U0001F534"
        lines = [
            f"Backtest Results ({days}d)",
            "=" * 28,
            f"Candles:  {len(df)}",
            f"Traded:   {total}",
            f"Skipped:  {skips}",
            f"Wins:     {wins}",
            f"Losses:   {losses}",
            f"Accuracy: {accuracy:.1%}",
            f"P&L:      {pnl_icon} ${total_pnl:+.2f}",
        ]
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="\n".join(lines),
            reply_markup=backtest_keyboard(),
        )
    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Backtest failed: {e}",
            reply_markup=backtest_keyboard(),
        )


async def handle_backtest_7d(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _run_backtest(update, context, days=7)


async def handle_backtest_30d(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _run_backtest(update, context, days=30)


async def handle_backtest_compare(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "Model comparison backtest is not yet implemented.\n"
        "Run the 7d or 30d backtest for full ensemble results.",
        reply_markup=backtest_keyboard(),
    )


async def handle_backtest_filters(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    lines = ["Filter Analysis", "=" * 25]
    if bot_app and hasattr(bot_app, "executor") and bot_app.executor:
        state = bot_app.executor.signal_filter.get_state()
        lines += [
            f"Paused:           {state.get('paused', False)}",
            f"Pause remaining:  {state.get('pause_remaining', 0)}",
            f"Consec losses:    {state.get('consecutive_losses', 0)}",
            f"ATR window:       {state.get('atr_window_size', 0)}",
            f"Dynamic threshold:{state.get('dynamic_threshold', 0):.3f}",
        ]
    else:
        lines.append("Executor not running.")
    await query.edit_message_text("\n".join(lines), reply_markup=backtest_keyboard())
