"""System monitoring handlers for CleoBot Telegram bot."""

import os
import time
from datetime import datetime, timezone

from telegram import Update
from telegram.ext import ContextTypes

from src.telegram_bot.keyboards import system_keyboard
from src.utils.logger import get_logger

logger = get_logger("telegram_bot.handlers.system")


async def handle_system_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "System Monitoring\n\nView latency, uptime, logs, and database stats.",
        reply_markup=system_keyboard(),
    )


async def handle_system_latency(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    lines = ["Latency Check", "=" * 25]
    if bot_app and hasattr(bot_app, "db") and bot_app.db:
        try:
            t0 = time.monotonic()
            bot_app.db.get_consecutive_losses()
            db_ms = (time.monotonic() - t0) * 1000
            lines.append(f"  DB round-trip:   {db_ms:.1f}ms")
        except Exception as e:
            lines.append(f"  DB error: {e}")
    if bot_app and hasattr(bot_app, "executor") and bot_app.executor:
        stats = bot_app.executor.get_stats()
        pm_stats = stats.get("polymarket_stats", {})
        lines += [
            f"  Polymarket:      {'Connected' if pm_stats.get('is_connected') else 'Simulation'}",
        ]
    await query.edit_message_text("\n".join(lines), reply_markup=system_keyboard())


async def handle_system_uptime(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    start_ts = context.bot_data.get("start_ts", time.time())
    uptime_s = time.time() - start_ts
    hours = int(uptime_s // 3600)
    minutes = int((uptime_s % 3600) // 60)
    seconds = int(uptime_s % 60)
    bot_app = context.bot_data.get("cleobot")
    lines = [
        "System Uptime",
        "=" * 25,
        f"  Uptime:  {hours}h {minutes}m {seconds}s",
        f"  Started: {datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    ]
    if bot_app and hasattr(bot_app, "executor") and bot_app.executor:
        stats = bot_app.executor.get_stats()
        lines += [
            "",
            f"  Cycles:  {stats.get('total_cycles', 0)}",
            f"  Trades:  {stats.get('total_trades', 0)}",
            f"  Errors:  {stats.get('total_errors', 0)}",
        ]
    await query.edit_message_text("\n".join(lines), reply_markup=system_keyboard())


async def handle_system_logs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    log_path = context.bot_data.get("log_path", "/data/cleobot.log")
    lines = ["Recent Logs (last 10 lines)", "=" * 30]
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                all_lines = f.readlines()
            last_lines = all_lines[-10:]
            for line in last_lines:
                lines.append("  " + line.strip()[:80])
        except Exception as e:
            lines.append(f"Could not read log: {e}")
    else:
        lines.append(f"Log file not found: {log_path}")
    await query.edit_message_text("\n".join(lines), reply_markup=system_keyboard())


async def handle_system_errors(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    log_path = context.bot_data.get("log_path", "/data/cleobot.log")
    lines = ["Error Log", "=" * 25]
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                all_lines = f.readlines()
            error_lines = [l.strip() for l in all_lines if "ERROR" in l or "CRITICAL" in l]
            last_errors = error_lines[-10:]
            if last_errors:
                for line in last_errors:
                    lines.append("  " + line[:80])
            else:
                lines.append("No errors found in log.")
        except Exception as e:
            lines.append(f"Could not read log: {e}")
    else:
        lines.append(f"Log file not found: {log_path}")
    await query.edit_message_text("\n".join(lines), reply_markup=system_keyboard())


async def handle_system_db(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    lines = ["Database Info", "=" * 25]
    db_path = context.bot_data.get("db_path", "/data/cleobot.db")
    if os.path.exists(db_path):
        size_bytes = os.path.getsize(db_path)
        size_mb = size_bytes / (1024 * 1024)
        lines.append(f"  Path:    {db_path}")
        lines.append(f"  Size:    {size_mb:.2f} MB ({size_bytes:,} bytes)")
    else:
        lines.append(f"  DB file not found at {db_path}")
    if bot_app and hasattr(bot_app, "db") and bot_app.db:
        try:
            total = bot_app.db.get_total_settled_trades()
            unsettled = len(bot_app.db.get_unsettled_trades())
            lines += [
                f"  Settled trades:   {total}",
                f"  Unsettled trades: {unsettled}",
            ]
        except Exception as e:
            lines.append(f"  DB query error: {e}")
    await query.edit_message_text("\n".join(lines), reply_markup=system_keyboard())
