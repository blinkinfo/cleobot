"""ML model management handlers for CleoBot Telegram bot."""

from telegram import Update
from telegram.ext import ContextTypes

from src.telegram_bot.keyboards import models_keyboard, confirm_keyboard
from src.utils.logger import get_logger

logger = get_logger("telegram_bot.handlers.models")


async def handle_models_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "ML Model Management\n\nView health, retrain, and compare models.",
        reply_markup=models_keyboard(),
    )


async def handle_models_health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    if bot_app and hasattr(bot_app, "ensemble") and bot_app.ensemble:
        from src.telegram_bot.cards import format_model_health
        health = bot_app.ensemble.get_model_health()
        last_retrain = context.bot_data.get("last_retrain_ts")
        text = format_model_health(health, last_retrain_ts=last_retrain)
    else:
        text = "Ensemble not available."
    await query.edit_message_text(text, reply_markup=models_keyboard())


async def handle_models_retrain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "\u26A0\uFE0F Force retrain will interrupt the current cycle.\n\nAre you sure?",
        reply_markup=confirm_keyboard("retrain"),
    )


async def handle_models_retrain_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    if bot_app and hasattr(bot_app, "executor") and bot_app.executor:
        import asyncio
        asyncio.create_task(bot_app.executor._run_full_retrain())
        await query.edit_message_text(
            "\U0001F504 Retrain scheduled. You will be notified when complete.",
            reply_markup=models_keyboard(),
        )
    else:
        await query.edit_message_text("Executor not available.", reply_markup=models_keyboard())


async def handle_models_features(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    lines = ["Feature Rankings (LightGBM)", "=" * 30]
    if bot_app and hasattr(bot_app, "ensemble") and bot_app.ensemble:
        try:
            rankings = bot_app.ensemble.get_feature_rankings()
            for i, r in enumerate(rankings[:15], 1):
                name = r.get("feature", "?")[:22]
                imp = r.get("lgbm_importance", 0.0)
                lines.append(f"  {i:>2}. {name:<22} {imp:.4f}")
        except Exception as e:
            lines.append(f"Error: {e}")
    else:
        lines.append("Models not loaded.")
    await query.edit_message_text("\n".join(lines), reply_markup=models_keyboard())


async def handle_models_compare(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    bot_app = context.bot_data.get("cleobot")
    lines = ["Model Accuracy Comparison", "=" * 28]
    if bot_app and hasattr(bot_app, "ensemble") and bot_app.ensemble:
        health = bot_app.ensemble.get_model_health()
        for name in ("lgbm", "tcn", "logreg", "meta"):
            info = health.get(name, {})
            trained = info.get("trained", False)
            icon = "\u2705" if trained else "\u274C"
            val = info.get("val_accuracy")
            train = info.get("train_accuracy")
            val_str = f"{val:.1%}" if val is not None else "N/A"
            train_str = f"{train:.1%}" if train is not None else "N/A"
            ver = info.get("version", 0)
            lines.append(f"  {icon} {name.upper():<8} v{ver}  train={train_str}  val={val_str}")
    else:
        lines.append("Models not available.")
    await query.edit_message_text("\n".join(lines), reply_markup=models_keyboard())


async def handle_models_regime_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    history = context.bot_data.get("regime_history", [])
    lines = ["Regime Change History", "=" * 28]
    if history:
        for entry in history[-10:]:
            lines.append(f"  {entry.get('ts', '?')[:16]}  {entry.get('old', '?')} -> {entry.get('new', '?')}")
    else:
        lines.append("No regime changes recorded yet.")
    await query.edit_message_text("\n".join(lines), reply_markup=models_keyboard())
