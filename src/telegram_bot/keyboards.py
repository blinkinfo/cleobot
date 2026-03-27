"""Inline keyboard layouts for CleoBot Telegram bot.

All 8 submenus defined as functions returning InlineKeyboardMarkup.
Callback data format: 'section:action' (e.g. 'trading:start', 'perf:today').
"""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


# ------------------------------------------------------------------ #
# MAIN MENU
# ------------------------------------------------------------------ #

def main_menu_keyboard() -> InlineKeyboardMarkup:
    """Main control panel keyboard -- 8 section buttons."""
    buttons = [
        [
            InlineKeyboardButton("Trading",     callback_data="menu:trading"),
            InlineKeyboardButton("Signals",     callback_data="menu:signals"),
        ],
        [
            InlineKeyboardButton("Performance", callback_data="menu:performance"),
            InlineKeyboardButton("Models",      callback_data="menu:models"),
        ],
        [
            InlineKeyboardButton("Backtest",    callback_data="menu:backtest"),
            InlineKeyboardButton("Risk",        callback_data="menu:risk"),
        ],
        [
            InlineKeyboardButton("System",      callback_data="menu:system"),
            InlineKeyboardButton("Settings",    callback_data="menu:settings"),
        ],
    ]
    return InlineKeyboardMarkup(buttons)


# ------------------------------------------------------------------ #
# TRADING SUBMENU
# ------------------------------------------------------------------ #

def trading_keyboard() -> InlineKeyboardMarkup:
    """Trading control submenu."""
    buttons = [
        [
            InlineKeyboardButton("Start Auto-Trade",  callback_data="trading:start"),
            InlineKeyboardButton("Stop Auto-Trade",   callback_data="trading:stop"),
        ],
        [
            InlineKeyboardButton("Pause (1 cycle)",   callback_data="trading:pause_1"),
            InlineKeyboardButton("Pause (3 cycles)",  callback_data="trading:pause_3"),
        ],
        [
            InlineKeyboardButton("Set Trade Size",    callback_data="trading:set_size"),
            InlineKeyboardButton("Current Status",    callback_data="trading:status"),
        ],
        [InlineKeyboardButton("<< Back",              callback_data="menu:main")],
    ]
    return InlineKeyboardMarkup(buttons)


# ------------------------------------------------------------------ #
# SIGNALS SUBMENU
# ------------------------------------------------------------------ #

def signals_keyboard() -> InlineKeyboardMarkup:
    """Signals analysis submenu."""
    buttons = [
        [
            InlineKeyboardButton("Next Signal",        callback_data="signals:next"),
            InlineKeyboardButton("Last 5 Signals",     callback_data="signals:last5"),
        ],
        [
            InlineKeyboardButton("Model Breakdown",    callback_data="signals:breakdown"),
            InlineKeyboardButton("Current Regime",     callback_data="signals:regime"),
        ],
        [
            InlineKeyboardButton("Feature Importance", callback_data="signals:features"),
        ],
        [InlineKeyboardButton("<< Back",               callback_data="menu:main")],
    ]
    return InlineKeyboardMarkup(buttons)


# ------------------------------------------------------------------ #
# PERFORMANCE SUBMENU
# ------------------------------------------------------------------ #

def performance_keyboard() -> InlineKeyboardMarkup:
    """Performance tracking submenu."""
    buttons = [
        [
            InlineKeyboardButton("Today's Stats",    callback_data="perf:today"),
            InlineKeyboardButton("Weekly Report",    callback_data="perf:weekly"),
        ],
        [
            InlineKeyboardButton("Monthly Report",   callback_data="perf:monthly"),
            InlineKeyboardButton("Hourly Heatmap",   callback_data="perf:hourly"),
        ],
        [
            InlineKeyboardButton("Streak History",   callback_data="perf:streaks"),
            InlineKeyboardButton("Equity Curve",     callback_data="perf:equity"),
        ],
        [InlineKeyboardButton("<< Back",             callback_data="menu:main")],
    ]
    return InlineKeyboardMarkup(buttons)


# ------------------------------------------------------------------ #
# MODELS SUBMENU
# ------------------------------------------------------------------ #

def models_keyboard() -> InlineKeyboardMarkup:
    """ML model management submenu."""
    buttons = [
        [
            InlineKeyboardButton("Model Health",       callback_data="models:health"),
            InlineKeyboardButton("Force Retrain",      callback_data="models:retrain"),
        ],
        [
            InlineKeyboardButton("Feature Rankings",   callback_data="models:features"),
            InlineKeyboardButton("Model Comparison",   callback_data="models:compare"),
        ],
        [
            InlineKeyboardButton("Regime History",     callback_data="models:regime_history"),
        ],
        [InlineKeyboardButton("<< Back",               callback_data="menu:main")],
    ]
    return InlineKeyboardMarkup(buttons)


# ------------------------------------------------------------------ #
# BACKTEST SUBMENU
# ------------------------------------------------------------------ #

def backtest_keyboard() -> InlineKeyboardMarkup:
    """Backtesting submenu."""
    buttons = [
        [
            InlineKeyboardButton("Run Backtest (7d)",   callback_data="backtest:7d"),
            InlineKeyboardButton("Run Backtest (30d)",  callback_data="backtest:30d"),
        ],
        [
            InlineKeyboardButton("Compare Models",      callback_data="backtest:compare"),
            InlineKeyboardButton("Filter Analysis",     callback_data="backtest:filters"),
        ],
        [InlineKeyboardButton("<< Back",                callback_data="menu:main")],
    ]
    return InlineKeyboardMarkup(buttons)


# ------------------------------------------------------------------ #
# RISK SUBMENU
# ------------------------------------------------------------------ #

def risk_keyboard() -> InlineKeyboardMarkup:
    """Risk management submenu."""
    buttons = [
        [
            InlineKeyboardButton("Current Drawdown",  callback_data="risk:drawdown"),
            InlineKeyboardButton("Daily Limits",      callback_data="risk:limits"),
        ],
        [
            InlineKeyboardButton("Update Limits",     callback_data="risk:update"),
            InlineKeyboardButton("Exposure",          callback_data="risk:exposure"),
        ],
        [InlineKeyboardButton("<< Back",              callback_data="menu:main")],
    ]
    return InlineKeyboardMarkup(buttons)


# ------------------------------------------------------------------ #
# SYSTEM SUBMENU
# ------------------------------------------------------------------ #

def system_keyboard() -> InlineKeyboardMarkup:
    """System monitoring submenu."""
    buttons = [
        [
            InlineKeyboardButton("Latency Check",   callback_data="system:latency"),
            InlineKeyboardButton("Uptime",          callback_data="system:uptime"),
        ],
        [
            InlineKeyboardButton("Logs (last 10)",  callback_data="system:logs"),
            InlineKeyboardButton("Error Log",       callback_data="system:errors"),
        ],
        [
            InlineKeyboardButton("DB Size",         callback_data="system:db"),
        ],
        [InlineKeyboardButton("<< Back",            callback_data="menu:main")],
    ]
    return InlineKeyboardMarkup(buttons)


# ------------------------------------------------------------------ #
# SETTINGS SUBMENU
# ------------------------------------------------------------------ #

def settings_keyboard() -> InlineKeyboardMarkup:
    """Settings submenu."""
    buttons = [
        [
            InlineKeyboardButton("Confidence Threshold", callback_data="settings:conf_threshold"),
            InlineKeyboardButton("Notifications",        callback_data="settings:notifications"),
        ],
        [InlineKeyboardButton("<< Back",                 callback_data="menu:main")],
    ]
    return InlineKeyboardMarkup(buttons)


# ------------------------------------------------------------------ #
# SIGNAL CARD BUTTONS
# ------------------------------------------------------------------ #

def signal_card_keyboard(signal_id: int, decision: str) -> InlineKeyboardMarkup:
    """Inline buttons shown on signal cards.

    Traded: [View Details] [Today's Stats]
    Skipped: [View Details] [Force Trade] [Today's Stats]
    """
    if decision == "TRADE":
        buttons = [
            [
                InlineKeyboardButton("View Details",  callback_data=f"signal:detail:{signal_id}"),
                InlineKeyboardButton("Today's Stats", callback_data="perf:today"),
            ]
        ]
    else:
        buttons = [
            [
                InlineKeyboardButton("View Details",  callback_data=f"signal:detail:{signal_id}"),
                InlineKeyboardButton("Force Trade",   callback_data=f"signal:force:{signal_id}"),
                InlineKeyboardButton("Today's Stats", callback_data="perf:today"),
            ]
        ]
    return InlineKeyboardMarkup(buttons)


def settlement_keyboard() -> InlineKeyboardMarkup:
    """Inline buttons shown on settlement cards."""
    buttons = [
        [
            InlineKeyboardButton("Dashboard",         callback_data="perf:today"),
            InlineKeyboardButton("Hourly Breakdown",  callback_data="perf:hourly"),
        ]
    ]
    return InlineKeyboardMarkup(buttons)


def confirm_keyboard(action: str) -> InlineKeyboardMarkup:
    """Yes/No confirmation keyboard for destructive actions."""
    buttons = [
        [
            InlineKeyboardButton("Yes, confirm", callback_data=f"confirm:{action}:yes"),
            InlineKeyboardButton("Cancel",       callback_data=f"confirm:{action}:no"),
        ]
    ]
    return InlineKeyboardMarkup(buttons)
