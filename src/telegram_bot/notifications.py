"""Auto-notification dispatcher for CleoBot.

All 8 push notification types as specified in Section 9 of the master plan:
  1. Traded signal notification
  2. Skipped signal notification
  3. Settlement notification
  4. Daily summary notification
  5. Model health / retrain complete notification
  6. Regime change notification
  7. Circuit breaker / risk alert notification
  8. Error alert notification

Each function sends the appropriate card and keyboard via the Telegram Application.
All functions are async and safe to call concurrently.
"""

import asyncio
from typing import Any, Dict, Optional

from telegram import Bot
from telegram.error import TelegramError
from telegram.constants import ParseMode

from src.telegram_bot.cards import (
    format_traded_signal,
    format_skipped_signal,
    format_settlement,
    format_daily_summary,
    format_model_health,
    format_regime_change,
    format_retrain_start,
    format_retrain_complete,
    format_accuracy_warning,
    format_circuit_breaker,
    format_error_alert,
    format_startup,
    format_shutdown,
)
from src.telegram_bot.keyboards import (
    signal_card_keyboard,
    settlement_keyboard,
)
from src.utils.logger import get_logger

logger = get_logger("telegram_bot.notifications")


# ------------------------------------------------------------------ #
# Internal send helper
# ------------------------------------------------------------------ #

async def _send(
    bot: Bot,
    chat_id: str,
    text: str,
    reply_markup=None,
    max_retries: int = 3,
) -> bool:
    """Send a message with retry logic.

    Args:
        bot: Telegram Bot instance.
        chat_id: Target chat ID.
        text: Message text.
        reply_markup: Optional inline keyboard.
        max_retries: Number of retry attempts on failure.

    Returns:
        True if sent successfully, False otherwise.
    """
    for attempt in range(max_retries):
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode=None,  # plain text -- cards use unicode, not HTML
            )
            return True
        except TelegramError as e:
            logger.warning(
                f"Telegram send failed (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # exponential back-off
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
            break
    return False


# ------------------------------------------------------------------ #
# 1. TRADED SIGNAL
# ------------------------------------------------------------------ #

async def notify_traded_signal(
    bot: Bot,
    chat_id: str,
    signal_id: int,
    direction: str,
    confidence: float,
    regime_display: str,
    lgbm_dir: str,
    lgbm_conf: float,
    tcn_dir: str,
    tcn_conf: float,
    logreg_dir: str,
    logreg_conf: float,
    agreement: int,
    filter_verdicts: Dict[str, Any],
    polymarket_odds: float,
    fill_time_s: float,
    slot_start: str,
    slot_end: str,
    trade_size: float = 1.0,
    is_simulated: bool = False,
    is_premium: bool = False,
) -> bool:
    """Send traded signal notification."""
    text = format_traded_signal(
        signal_id=signal_id,
        direction=direction,
        confidence=confidence,
        regime_display=regime_display,
        lgbm_dir=lgbm_dir,
        lgbm_conf=lgbm_conf,
        tcn_dir=tcn_dir,
        tcn_conf=tcn_conf,
        logreg_dir=logreg_dir,
        logreg_conf=logreg_conf,
        agreement=agreement,
        filter_verdicts=filter_verdicts,
        polymarket_odds=polymarket_odds,
        fill_time_s=fill_time_s,
        slot_start=slot_start,
        slot_end=slot_end,
        trade_size=trade_size,
        is_simulated=is_simulated,
        is_premium=is_premium,
    )
    keyboard = signal_card_keyboard(signal_id, "TRADE")
    return await _send(bot, chat_id, text, reply_markup=keyboard)


# ------------------------------------------------------------------ #
# 2. SKIPPED SIGNAL
# ------------------------------------------------------------------ #

async def notify_skipped_signal(
    bot: Bot,
    chat_id: str,
    signal_id: int,
    direction: str,
    confidence: float,
    regime_display: str,
    lgbm_dir: str,
    lgbm_conf: float,
    tcn_dir: str,
    tcn_conf: float,
    logreg_dir: str,
    logreg_conf: float,
    agreement: int,
    filter_verdicts: Dict[str, Any],
    skip_reason: str,
) -> bool:
    """Send skipped signal notification."""
    text = format_skipped_signal(
        signal_id=signal_id,
        direction=direction,
        confidence=confidence,
        regime_display=regime_display,
        lgbm_dir=lgbm_dir,
        lgbm_conf=lgbm_conf,
        tcn_dir=tcn_dir,
        tcn_conf=tcn_conf,
        logreg_dir=logreg_dir,
        logreg_conf=logreg_conf,
        agreement=agreement,
        filter_verdicts=filter_verdicts,
        skip_reason=skip_reason,
    )
    keyboard = signal_card_keyboard(signal_id, "SKIP")
    return await _send(bot, chat_id, text, reply_markup=keyboard)


# ------------------------------------------------------------------ #
# 3. SETTLEMENT
# ------------------------------------------------------------------ #

async def notify_settlement(
    bot: Bot,
    chat_id: str,
    signal_id: int,
    result: str,
    pnl: float,
    candle_open: float,
    candle_close: float,
    candle_move_pct: float,
    wins_today: int,
    losses_today: int,
    skips_today: int,
    accuracy_today: float,
    pnl_today: float,
) -> bool:
    """Send settlement notification."""
    text = format_settlement(
        signal_id=signal_id,
        result=result,
        pnl=pnl,
        candle_open=candle_open,
        candle_close=candle_close,
        candle_move_pct=candle_move_pct,
        wins_today=wins_today,
        losses_today=losses_today,
        skips_today=skips_today,
        accuracy_today=accuracy_today,
        pnl_today=pnl_today,
    )
    keyboard = settlement_keyboard()
    return await _send(bot, chat_id, text, reply_markup=keyboard)


# ------------------------------------------------------------------ #
# 4. DAILY SUMMARY
# ------------------------------------------------------------------ #

async def notify_daily_summary(
    bot: Bot,
    chat_id: str,
    date: str,
    trades: int,
    wins: int,
    losses: int,
    skips: int,
    accuracy: float,
    pnl: float,
    best_hour: Optional[str] = None,
    worst_hour: Optional[str] = None,
    best_hour_acc: float = 0.0,
    worst_hour_acc: float = 0.0,
) -> bool:
    """Send daily summary notification."""
    text = format_daily_summary(
        date=date,
        trades=trades,
        wins=wins,
        losses=losses,
        skips=skips,
        accuracy=accuracy,
        pnl=pnl,
        best_hour=best_hour,
        worst_hour=worst_hour,
        best_hour_acc=best_hour_acc,
        worst_hour_acc=worst_hour_acc,
    )
    return await _send(bot, chat_id, text)


# ------------------------------------------------------------------ #
# 5. MODEL HEALTH / RETRAIN
# ------------------------------------------------------------------ #

async def notify_retrain_start(
    bot: Bot,
    chat_id: str,
    retrain_type: str = "full",
) -> bool:
    """Send retrain-started notification."""
    text = format_retrain_start(retrain_type)
    return await _send(bot, chat_id, text)


async def notify_retrain_complete(
    bot: Bot,
    chat_id: str,
    retrain_type: str,
    elapsed_s: float,
    accepted: bool,
    lgbm_val_acc: Optional[float] = None,
    tcn_val_acc: Optional[float] = None,
    logreg_val_acc: Optional[float] = None,
    meta_val_acc: Optional[float] = None,
    reject_reason: str = "",
) -> bool:
    """Send retrain-complete notification with accuracy results."""
    text = format_retrain_complete(
        retrain_type=retrain_type,
        elapsed_s=elapsed_s,
        accepted=accepted,
        lgbm_val_acc=lgbm_val_acc,
        tcn_val_acc=tcn_val_acc,
        logreg_val_acc=logreg_val_acc,
        meta_val_acc=meta_val_acc,
        reject_reason=reject_reason,
    )
    return await _send(bot, chat_id, text)


async def notify_model_health(
    bot: Bot,
    chat_id: str,
    health: Dict[str, Any],
    last_retrain_ts: Optional[str] = None,
) -> bool:
    """Send model health summary."""
    text = format_model_health(health, last_retrain_ts=last_retrain_ts)
    return await _send(bot, chat_id, text)


# ------------------------------------------------------------------ #
# 6. REGIME CHANGE
# ------------------------------------------------------------------ #

async def notify_regime_change(
    bot: Bot,
    chat_id: str,
    old_regime: str,
    new_regime: str,
    new_regime_display: str,
    confidence: float,
    timestamp: Optional[str] = None,
) -> bool:
    """Send regime change alert."""
    text = format_regime_change(
        old_regime=old_regime,
        new_regime=new_regime,
        new_regime_display=new_regime_display,
        confidence=confidence,
        timestamp=timestamp,
    )
    return await _send(bot, chat_id, text)


# ------------------------------------------------------------------ #
# 7. CIRCUIT BREAKER / RISK ALERT
# ------------------------------------------------------------------ #

async def notify_accuracy_warning(
    bot: Bot,
    chat_id: str,
    rolling_accuracy: float,
    window: int,
    threshold: float,
) -> bool:
    """Send accuracy degradation warning."""
    text = format_accuracy_warning(rolling_accuracy, window, threshold)
    return await _send(bot, chat_id, text)


async def notify_circuit_breaker(
    bot: Bot,
    chat_id: str,
    reason: str,
    daily_pnl: float,
    daily_limit: float,
) -> bool:
    """Send circuit-breaker / trading-halted notification."""
    text = format_circuit_breaker(reason, daily_pnl, daily_limit)
    return await _send(bot, chat_id, text)


# ------------------------------------------------------------------ #
# 8. ERROR ALERT
# ------------------------------------------------------------------ #

async def notify_error(
    bot: Bot,
    chat_id: str,
    component: str,
    error_msg: str,
) -> bool:
    """Send error alert notification."""
    text = format_error_alert(component, error_msg)
    return await _send(bot, chat_id, text)


# ------------------------------------------------------------------ #
# STARTUP / SHUTDOWN
# ------------------------------------------------------------------ #

async def notify_startup(
    bot: Bot,
    chat_id: str,
    version: str = "1.0",
    models_loaded: bool = False,
    auto_trade: bool = False,
    data_dir: str = "/data",
) -> bool:
    """Send bot startup notification."""
    text = format_startup(
        version=version,
        models_loaded=models_loaded,
        auto_trade=auto_trade,
        data_dir=data_dir,
    )
    return await _send(bot, chat_id, text)


async def notify_shutdown(
    bot: Bot,
    chat_id: str,
) -> bool:
    """Send bot shutdown notification."""
    text = format_shutdown()
    return await _send(bot, chat_id, text)
