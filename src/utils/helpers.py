"""CleoBot miscellaneous utility functions."""

import time
import math
from datetime import datetime, timezone
from typing import Optional


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def utc_timestamp() -> float:
    """Get current UTC timestamp in seconds."""
    return time.time()


def utc_timestamp_ms() -> int:
    """Get current UTC timestamp in milliseconds."""
    return int(time.time() * 1000)


def ms_to_datetime(ms: int) -> datetime:
    """Convert millisecond timestamp to UTC datetime."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def datetime_to_ms(dt: datetime) -> int:
    """Convert datetime to millisecond timestamp."""
    return int(dt.timestamp() * 1000)


def get_current_candle_start(interval_minutes: int = 5) -> datetime:
    """Get the start time of the current candle slot.
    
    5-minute candles run on fixed slots: :00, :05, :10, :15, etc.
    
    Args:
        interval_minutes: Candle interval in minutes.
    
    Returns:
        Start time of the current candle slot (UTC).
    """
    now = utc_now()
    minutes = now.minute - (now.minute % interval_minutes)
    return now.replace(minute=minutes, second=0, microsecond=0)


def get_next_candle_start(interval_minutes: int = 5) -> datetime:
    """Get the start time of the next candle slot.
    
    Args:
        interval_minutes: Candle interval in minutes.
    
    Returns:
        Start time of the next candle slot (UTC).
    """
    from datetime import timedelta
    current = get_current_candle_start(interval_minutes)
    return current + timedelta(minutes=interval_minutes)


def get_signal_deadline(interval_minutes: int = 5) -> datetime:
    """Get the signal generation deadline for the next candle.
    
    Signal must be generated ~90 seconds BEFORE the target candle opens.
    Per timing spec: signal finalized by minute+3:00 of current candle.
    
    Args:
        interval_minutes: Candle interval in minutes.
    
    Returns:
        Deadline datetime (UTC).
    """
    from datetime import timedelta
    next_candle = get_next_candle_start(interval_minutes)
    return next_candle - timedelta(seconds=90)


def seconds_until(target: datetime) -> float:
    """Get seconds until a target datetime.
    
    Args:
        target: Target datetime (UTC).
    
    Returns:
        Seconds until target (negative if target is in the past).
    """
    return (target - utc_now()).total_seconds()


def format_pnl(pnl: float) -> str:
    """Format P&L value with sign and color indicator.
    
    Args:
        pnl: Profit/loss value.
    
    Returns:
        Formatted string like '+$0.88' or '-$1.00'.
    """
    if pnl >= 0:
        return f"+${pnl:.2f}"
    return f"-${abs(pnl):.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a ratio as percentage.
    
    Args:
        value: Ratio value (e.g., 0.571 for 57.1%).
        decimals: Number of decimal places.
    
    Returns:
        Formatted string like '57.1%'.
    """
    return f"{value * 100:.{decimals}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator.
    
    Args:
        numerator: Numerator.
        denominator: Denominator.
        default: Default value if denominator is zero.
    
    Returns:
        Division result or default.
    """
    if denominator == 0 or math.isnan(denominator):
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max.
    
    Args:
        value: Value to clamp.
        min_val: Minimum value.
        max_val: Maximum value.
    
    Returns:
        Clamped value.
    """
    return max(min_val, min(max_val, value))


def chunk_list(lst: list, chunk_size: int) -> list:
    """Split a list into chunks of given size.
    
    Args:
        lst: Input list.
        chunk_size: Size of each chunk.
    
    Returns:
        List of chunks.
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
