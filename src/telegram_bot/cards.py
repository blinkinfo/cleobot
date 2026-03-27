"""Signal and notification card formatters for CleoBot.

Formats all message cards exactly as specified in Section 9 of the master plan:
  - Traded signal card
  - Skipped signal card
  - Settlement notification card
  - Daily summary card
  - Model health card
  - Regime change card

All formatters return plain strings ready to send via Telegram.
Unicode icons are used for visual clarity.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List


# ------------------------------------------------------------------ #
# Icon constants
# ------------------------------------------------------------------ #

ICON_GREEN   = "\U0001F7E2"   # green circle  (UP / WIN)
ICON_RED     = "\U0001F534"   # red circle    (DOWN / LOSS)
ICON_YELLOW  = "\U0001F7E1"   # yellow circle (neutral)
ICON_STAR    = "\u2B50"       # star          (premium)
ICON_SKIP    = "\u23ED\uFE0F" # next-track    (skip)
ICON_PASS    = "\u2705"       # green check   (filter pass)
ICON_FAIL    = "\u274C"       # red X         (filter fail)
ICON_WARN    = "\u26A0\uFE0F" # warning sign  (filter warn)
ICON_ROBOT   = "\U0001F916"   # robot         (models)
ICON_SHIELD  = "\U0001F6E1\uFE0F"  # shield   (filters)
ICON_CHART   = "\U0001F4CA"   # chart         (stats)
ICON_MONEY   = "\U0001F4B0"   # money bag     (trade)
ICON_CLOCK   = "\U0001F551"   # clock         (time)
ICON_TARGET  = "\U0001F3AF"   # target        (regime)
ICON_REFRESH = "\U0001F504"   # arrows        (retrain)
ICON_ALERT   = "\u26A0\uFE0F" # alert         (warning)
ICON_FIRE    = "\U0001F525"   # fire          (circuit breaker)
ICON_BRAIN   = "\U0001F9E0"   # brain         (model)
ICON_ROCKET  = "\U0001F680"   # rocket        (startup)
ICON_STOP    = "\U0001F6D1"   # stop sign     (stop)
ICON_HANDSHAKE = "\U0001F91D" # handshake     (agreement)
ICON_CREDIT  = "\U0001F4B3"   # credit card   (risk)

SEP = "=" * 29
THIN_SEP = "-" * 29


def _dir_icon(direction: str) -> str:
    return ICON_GREEN if direction == "UP" else ICON_RED


def _filter_icon(status: str) -> str:
    if status == "PASS":
        return ICON_PASS
    elif status == "FAIL":
        return ICON_FAIL
    else:
        return ICON_WARN


def _conf_bar(confidence: float, width: int = 10) -> str:
    """Visual confidence bar like '[======    ]'."""
    filled = max(0, min(width, int(confidence * width)))
    empty  = width - filled
    return "[" + "=" * filled + " " * empty + "]"


# ------------------------------------------------------------------ #
# TRADED SIGNAL CARD
# ------------------------------------------------------------------ #

def format_traded_signal(
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
) -> str:
    """Format a traded signal card (exact Section 9 format)."""
    sim_tag = " [SIM]" if is_simulated else ""
    premium_tag = f" {ICON_STAR}PREMIUM{ICON_STAR}" if is_premium else ""
    dir_icon = _dir_icon(direction)

    lines = [
        SEP,
        f"  SIGNAL #{signal_id} | TRADED{sim_tag}{premium_tag}",
        SEP,
        f"Direction:  {dir_icon} {direction}",
        f"Confidence: {confidence:.1%}",
        f"Regime:     {regime_display}",
        "",
        "Models:",
        f"  LightGBM:  {_dir_icon(lgbm_dir)} {lgbm_dir:<4} ({lgbm_conf:.1%})",
        f"  TCN:       {_dir_icon(tcn_dir)} {tcn_dir:<4} ({tcn_conf:.1%})",
        f"  LogReg:    {_dir_icon(logreg_dir)} {logreg_dir:<4} ({logreg_conf:.1%})",
        f"  Agreement: {agreement}/3",
        "",
        "Filters:",
    ]

    for name, verdict in filter_verdicts.items():
        status = verdict.get("status", "PASS" if verdict.get("passed") else "FAIL")
        # verdict may be a FilterVerdict object or a dict
        if hasattr(verdict, "status_str"):
            status = verdict.status_str
            msg    = verdict.message
        else:
            if verdict.get("is_warning"):
                status = "WARN"
            elif verdict.get("passed"):
                status = "PASS"
            else:
                status = "FAIL"
            msg = verdict.get("message", "")
        icon = _filter_icon(status)
        lines.append(f"  {icon} {name.title():<12} {msg}")

    lines += [
        "",
        "Execution:",
        f"  Polymarket Odds: {polymarket_odds:.2f}",
        f"  Trade Size:      ${trade_size:.2f}",
        f"  Fill Time:       {fill_time_s:.1f}s",
        f"  Slot: {slot_start} - {slot_end} UTC",
        SEP,
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# SKIPPED SIGNAL CARD
# ------------------------------------------------------------------ #

def format_skipped_signal(
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
) -> str:
    """Format a skipped signal card (exact Section 9 format)."""
    dir_icon = _dir_icon(direction)

    lines = [
        SEP,
        f"  SIGNAL #{signal_id} | SKIPPED",
        SEP,
        f"Direction:  {dir_icon} {direction}",
        f"Confidence: {confidence:.1%}",
        f"Regime:     {regime_display}",
        "",
        "Models:",
        f"  LightGBM:  {_dir_icon(lgbm_dir)} {lgbm_dir:<4} ({lgbm_conf:.1%})",
        f"  TCN:       {_dir_icon(tcn_dir)} {tcn_dir:<4} ({tcn_conf:.1%})",
        f"  LogReg:    {_dir_icon(logreg_dir)} {logreg_dir:<4} ({logreg_conf:.1%})",
        f"  Agreement: {agreement}/3",
        "",
        "Filters:",
    ]

    for name, verdict in filter_verdicts.items():
        if hasattr(verdict, "status_str"):
            status = verdict.status_str
            msg    = verdict.message
        else:
            if verdict.get("is_warning"):
                status = "WARN"
            elif verdict.get("passed"):
                status = "PASS"
            else:
                status = "FAIL"
            msg = verdict.get("message", "")
        icon = _filter_icon(status)
        lines.append(f"  {icon} {name.title():<12} {msg}")

    lines += [
        "",
        f"Skip Reason: {skip_reason}",
        SEP,
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# SETTLEMENT CARD
# ------------------------------------------------------------------ #

def format_settlement(
    signal_id: int,
    result: str,           # 'WIN' or 'LOSS'
    pnl: float,
    candle_open: float,
    candle_close: float,
    candle_move_pct: float,
    wins_today: int,
    losses_today: int,
    skips_today: int,
    accuracy_today: float,
    pnl_today: float,
) -> str:
    """Format a settlement notification card (exact Section 9 format)."""
    result_icon = ICON_GREEN if result == "WIN" else ICON_RED
    move_icon   = ICON_GREEN if candle_move_pct > 0 else ICON_RED

    lines = [
        SEP,
        f"  SETTLEMENT | SIGNAL #{signal_id}",
        SEP,
        f"  Result:  {result_icon} {result}",
        f"  P&L:     ${pnl:+.2f}",
        "",
        f"  Open:    ${candle_open:,.2f}",
        f"  Close:   ${candle_close:,.2f}",
        f"  Move:    {move_icon} {candle_move_pct:+.3f}%",
        "",
        f"  Today:   {wins_today}W / {losses_today}L / {skips_today} Skip",
        f"  Accuracy: {accuracy_today:.1%}",
        f"  P&L:     ${pnl_today:+.2f}",
        SEP,
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# DAILY SUMMARY CARD
# ------------------------------------------------------------------ #

def format_daily_summary(
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
) -> str:
    """Format the daily summary card sent at 00:00 UTC."""
    pnl_icon = ICON_GREEN if pnl >= 0 else ICON_RED

    lines = [
        SEP,
        f"  DAILY SUMMARY | {date}",
        SEP,
        f"{ICON_CHART} Results:",
        f"  Trades:   {trades}",
        f"  Wins:     {wins}",
        f"  Losses:   {losses}",
        f"  Skips:    {skips}",
        f"  Accuracy: {accuracy:.1%}",
        f"  P&L:      {pnl_icon} ${pnl:+.2f}",
    ]

    if best_hour:
        lines += [
            "",
            f"{ICON_CHART} Hourly:",
            f"  Best:   {best_hour}:00 UTC ({best_hour_acc:.1%})",
            f"  Worst:  {worst_hour}:00 UTC ({worst_hour_acc:.1%})",
        ]

    lines.append(SEP)
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# MODEL HEALTH CARD
# ------------------------------------------------------------------ #

def format_model_health(
    health: Dict[str, Any],
    last_retrain_ts: Optional[str] = None,
) -> str:
    """Format the model health card."""
    is_ready = health.get("is_ready", False)
    ready_icon = ICON_PASS if is_ready else ICON_FAIL
    avg_ms = health.get("avg_inference_ms", 0.0)
    pred_count = health.get("prediction_count", 0)
    active_versions = health.get("active_versions", {})

    def _model_row(name: str, info: Dict[str, Any]) -> str:
        trained = info.get("trained", False)
        icon    = ICON_PASS if trained else ICON_FAIL
        version = info.get("version", 0)
        val_acc = info.get("val_accuracy")
        acc_str = f"{val_acc:.1%}" if val_acc is not None else "N/A"
        return f"  {icon} {name:<10} v{version}  val={acc_str}"

    lines = [
        SEP,
        f"  {ICON_BRAIN} MODEL HEALTH",
        SEP,
        f"Ensemble Ready: {ready_icon} {'Yes' if is_ready else 'No'}",
        f"Predictions:    {pred_count}",
        f"Avg Inference:  {avg_ms:.1f}ms",
        "",
        "Models:",
        _model_row("LightGBM",  health.get("lgbm",   {})),
        _model_row("TCN",       health.get("tcn",    {})),
        _model_row("LogReg",    health.get("logreg", {})),
        _model_row("Meta",      health.get("meta",   {})),
    ]

    regime_info = health.get("regime_detector", {})
    regime_trained = regime_info.get("trained", False)
    regime_icon = ICON_PASS if regime_trained else ICON_FAIL
    regime_ver  = regime_info.get("version", 0)
    lines.append(f"  {regime_icon} HMM        v{regime_ver}  (regime detector)")

    if last_retrain_ts:
        lines += ["", f"Last Retrain: {last_retrain_ts}"]

    lines.append(SEP)
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# REGIME CHANGE CARD
# ------------------------------------------------------------------ #

def format_regime_change(
    old_regime: str,
    new_regime: str,
    new_regime_display: str,
    confidence: float,
    timestamp: Optional[str] = None,
) -> str:
    """Format a regime change alert card."""
    ts = timestamp or datetime.now(timezone.utc).strftime("%H:%M UTC")
    lines = [
        f"{ICON_ALERT} REGIME CHANGE",
        THIN_SEP,
        f"  {old_regime}  ->  {new_regime_display}",
        f"  Confidence: {confidence:.1%}",
        f"  Time:       {ts}",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# RETRAIN NOTIFICATION CARDS
# ------------------------------------------------------------------ #

def format_retrain_start(retrain_type: str = "full") -> str:
    return (
        f"{ICON_REFRESH} RETRAIN STARTING\n"
        f"  Type: {retrain_type.upper()}\n"
        f"  Time: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    )


def format_retrain_complete(
    retrain_type: str,
    elapsed_s: float,
    accepted: bool,
    lgbm_val_acc: Optional[float] = None,
    tcn_val_acc: Optional[float] = None,
    logreg_val_acc: Optional[float] = None,
    meta_val_acc: Optional[float] = None,
    reject_reason: str = "",
) -> str:
    """Format a retrain completion notification."""
    status_icon = ICON_PASS if accepted else ICON_FAIL
    status_str  = "ACCEPTED" if accepted else f"REJECTED ({reject_reason})"

    lines = [
        f"{ICON_REFRESH} RETRAIN COMPLETE",
        THIN_SEP,
        f"  Type:    {retrain_type.upper()}",
        f"  Status:  {status_icon} {status_str}",
        f"  Elapsed: {elapsed_s:.0f}s",
    ]
    if accepted and any(v is not None for v in [lgbm_val_acc, tcn_val_acc, logreg_val_acc]):
        lines.append("  Val Accuracy:")
        if lgbm_val_acc   is not None: lines.append(f"    LGBM:   {lgbm_val_acc:.1%}")
        if tcn_val_acc    is not None: lines.append(f"    TCN:    {tcn_val_acc:.1%}")
        if logreg_val_acc is not None: lines.append(f"    LogReg: {logreg_val_acc:.1%}")
        if meta_val_acc   is not None: lines.append(f"    Meta:   {meta_val_acc:.1%}")
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# ACCURACY WARNING CARD
# ------------------------------------------------------------------ #

def format_accuracy_warning(
    rolling_accuracy: float,
    window: int,
    threshold: float,
) -> str:
    return (
        f"{ICON_ALERT} ACCURACY WARNING\n"
        f"  Rolling accuracy: {rolling_accuracy:.1%}\n"
        f"  Window: {window} trades\n"
        f"  Threshold: {threshold:.1%}\n"
        "  Confidence threshold boosted. Monitor closely."
    )


# ------------------------------------------------------------------ #
# CIRCUIT BREAKER CARD
# ------------------------------------------------------------------ #

def format_circuit_breaker(
    reason: str,
    daily_pnl: float,
    daily_limit: float,
) -> str:
    return (
        f"{ICON_FIRE} CIRCUIT BREAKER ACTIVATED\n"
        f"  Reason:      {reason}\n"
        f"  Daily P&L:   ${daily_pnl:+.2f}\n"
        f"  Daily Limit: -${daily_limit:.2f}\n"
        "  Auto-trading PAUSED for today."
    )


# ------------------------------------------------------------------ #
# ERROR ALERT CARD
# ------------------------------------------------------------------ #

def format_error_alert(component: str, error_msg: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    return (
        f"{ICON_ALERT} ERROR [{component}]\n"
        f"  Time: {ts}\n"
        f"  {error_msg[:400]}"
    )


# ------------------------------------------------------------------ #
# STARTUP / SHUTDOWN CARDS
# ------------------------------------------------------------------ #

def format_startup(
    version: str = "1.0",
    models_loaded: bool = False,
    auto_trade: bool = False,
    data_dir: str = "/data",
) -> str:
    models_icon = ICON_PASS if models_loaded else ICON_WARN
    trade_icon  = ICON_GREEN if auto_trade else ICON_RED
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"{ICON_ROCKET} CleoBot STARTED\n"
        f"  Version:     {version}\n"
        f"  Time:        {ts}\n"
        f"  Models:      {models_icon} {'Loaded' if models_loaded else 'Training required'}\n"
        f"  Auto-trade:  {trade_icon} {'ON' if auto_trade else 'OFF'}\n"
        f"  Data dir:    {data_dir}"
    )


def format_shutdown() -> str:
    ts = datetime.now(timezone.utc).strftime("%H:%M UTC")
    return f"{ICON_STOP} CleoBot shutting down at {ts}. State saved."
