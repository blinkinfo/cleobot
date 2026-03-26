"""CleoBot structured logging setup."""

import logging
import sys
from datetime import datetime, timezone


class UTCFormatter(logging.Formatter):
    """Formatter that uses UTC timestamps."""

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"


def setup_logger(name: str = "cleobot", level: str = "INFO") -> logging.Logger:
    """Set up and return a structured logger.
    
    Args:
        name: Logger name.
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = UTCFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    for noisy_logger in ["websockets", "ccxt", "urllib3", "httpx", "telegram"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Get a child logger for a specific module.
    
    Args:
        module_name: The module name (e.g., 'data.collector').
    
    Returns:
        Child logger instance.
    """
    return logging.getLogger(f"cleobot.{module_name}")
