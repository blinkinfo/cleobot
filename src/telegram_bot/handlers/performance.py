"""Performance tracking handlers for CleoBot Telegram bot."""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from telegram import Update
from telegram.ext import ContextTypes

from src.telegram_bot.keyboards import performance_keyboard
from src.utils.logger import get_logger

logger = get_logger("telegram_bot.handlers.performance")


def _accuracy(wins: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{wins / total:.1%}"


async def handle_performance_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "Performance Tracking\n\nView stats, equity curve, and hourly breakdown.",
        reply_markup=performance_keyboard(),
    )


async def handle_perf_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    lines = ["Today's Performance", "=" * 28]
    if bot_app and hasattr(bot_app, "db") and bot_app.db:
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            stats = bot_app.db.get_daily_stats(today)
            trades = stats.get("trades", 0)
            wins = stats.get("wins", 0)
            losses = stats.get("losses", 0)
            skips = stats.get("skips", 0)
            pnl = stats.get("pnl", 0.0) or 0.0
            pnl_icon = "\U0001F7E2" if pnl >= 0 else "\U0001F534"
            lines += [
                f"Date:     {today}",
                f"Trades:   {trades}",
                f"Wins:     {wins}",
                f"Losses:   {losses}",
                f"Skips:    {skips}",
                f"Accuracy: {_accuracy(wins, wins + losses)}",
                f"P&L:      {pnl_icon} ${pnl:+.2f}",
            ]
            if bot_app.executor:
                risk = bot_app.executor.risk_manager.get_status()
                lines += [
                    "",
                    "Risk Status:",
                    f"  Consec Losses: {risk.consecutive_losses}",
                    f"  Open Exposure: ${risk.open_exposure:.2f}",
                ]
        except Exception as e:
            lines.append(f"Error: {e}")
    else:
        lines.append("Database not available.")
    await query.edit_message_text("\n".join(lines), reply_markup=performance_keyboard())


async def handle_perf_weekly(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    lines = ["Weekly Performance (Last 7 Days)", "=" * 33]
    if bot_app and hasattr(bot_app, "db") and bot_app.db:
        try:
            total_trades = 0
            total_wins = 0
            total_pnl = 0.0
            for i in range(7):
                d = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
                s = bot_app.db.get_daily_stats(d)
                t = s.get("trades", 0)
                w = s.get("wins", 0)
                p = s.get("pnl", 0.0) or 0.0
                total_trades += t
                total_wins += w
                total_pnl += p
                if t > 0:
                    pnl_icon = "\U0001F7E2" if p >= 0 else "\U0001F534"
                    lines.append(f"  {d}  {w}W/{t-w}L  {pnl_icon} ${p:+.2f}")
            lines += [
                "",
                f"Total:    {total_trades} trades",
                f"Wins:     {total_wins}",
                f"Accuracy: {_accuracy(total_wins, total_trades)}",
                f"P&L:      ${total_pnl:+.2f}",
            ]
        except Exception as e:
            lines.append(f"Error: {e}")
    else:
        lines.append("Database not available.")
    await query.edit_message_text("\n".join(lines), reply_markup=performance_keyboard())


async def handle_perf_monthly(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    lines = ["Monthly Performance (Last 30 Days)", "=" * 35]
    if bot_app and hasattr(bot_app, "db") and bot_app.db:
        try:
            total_trades = 0
            total_wins = 0
            total_pnl = 0.0
            for i in range(30):
                d = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
                s = bot_app.db.get_daily_stats(d)
                total_trades += s.get("trades", 0)
                total_wins += s.get("wins", 0)
                total_pnl += s.get("pnl", 0.0) or 0.0
            pnl_icon = "\U0001F7E2" if total_pnl >= 0 else "\U0001F534"
            lines += [
                f"Trades:   {total_trades}",
                f"Wins:     {total_wins}",
                f"Accuracy: {_accuracy(total_wins, total_trades)}",
                f"P&L:      {pnl_icon} ${total_pnl:+.2f}",
            ]
        except Exception as e:
            lines.append(f"Error: {e}")
    else:
        lines.append("Database not available.")
    await query.edit_message_text("\n".join(lines), reply_markup=performance_keyboard())


async def handle_perf_hourly(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    lines = ["Hourly Performance Heatmap", "=" * 30]
    if bot_app and hasattr(bot_app, "db") and bot_app.db:
        try:
            hourly = bot_app.db.get_hourly_stats()
            if hourly:
                for row in hourly[:12]:
                    h = row.get("hour", 0)
                    w = row.get("wins", 0)
                    t = row.get("total", 0)
                    p = row.get("pnl", 0.0) or 0.0
                    acc = _accuracy(w, t)
                    pnl_icon = "\U0001F7E2" if p >= 0 else "\U0001F534"
                    lines.append(f"  {h:02d}:00  {acc}  {pnl_icon} ${p:+.2f}  ({t} trades)")
            else:
                lines.append("No hourly data yet.")
        except Exception as e:
            lines.append(f"Error: {e}")
    else:
        lines.append("Database not available.")
    await query.edit_message_text("\n".join(lines), reply_markup=performance_keyboard())


async def handle_perf_streaks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    lines = ["Streak History", "=" * 25]
    if bot_app and hasattr(bot_app, "db") and bot_app.db:
        try:
            cons_losses = bot_app.db.get_consecutive_losses()
            rolling50 = bot_app.db.get_rolling_accuracy(window=50)
            rolling20 = bot_app.db.get_rolling_accuracy(window=20)
            lines += [
                f"Current losing streak: {cons_losses}",
                f"Rolling accuracy (20): {rolling20:.1%}",
                f"Rolling accuracy (50): {rolling50:.1%}",
            ]
        except Exception as e:
            lines.append(f"Error: {e}")
    else:
        lines.append("Database not available.")
    await query.edit_message_text("\n".join(lines), reply_markup=performance_keyboard())


async def handle_perf_equity(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    lines = ["Equity Curve (Last 14 Days)", "=" * 30]
    if bot_app and hasattr(bot_app, "db") and bot_app.db:
        try:
            cumulative = 0.0
            for i in range(14, -1, -1):
                d = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
                s = bot_app.db.get_daily_stats(d)
                day_pnl = s.get("pnl", 0.0) or 0.0
                cumulative += day_pnl
                if s.get("trades", 0) > 0:
                    bar_len = max(0, min(20, int(abs(cumulative) / 5)))
                    bar = ("+" if cumulative >= 0 else "-") * bar_len
                    lines.append(f"  {d}  {bar:<20}  ${cumulative:+.2f}")
        except Exception as e:
            lines.append(f"Error: {e}")
    else:
        lines.append("Database not available.")
    await query.edit_message_text("\n".join(lines), reply_markup=performance_keyboard())
