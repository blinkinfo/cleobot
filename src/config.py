"""CleoBot configuration -- loads all environment variables with defaults and validation."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _get_env(key: str, default: str = None, required: bool = False) -> str:
    """Get environment variable with optional default and required check."""
    value = os.getenv(key, default)
    if required and (value is None or value.strip() == ""):
        raise EnvironmentError(f"Required environment variable '{key}' is not set.")
    return value


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes")


def _get_env_float(key: str, default: float = 0.0) -> float:
    """Get float environment variable."""
    return float(os.getenv(key, str(default)))


def _get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable."""
    return int(os.getenv(key, str(default)))


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str = ""
    chat_id: str = ""

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)


@dataclass(frozen=True)
class MEXCConfig:
    base_url_rest: str = "https://api.mexc.com"
    ws_url: str = "wss://wbs.mexc.com/ws"
    symbol: str = "BTCUSDT"
    api_key: str = ""
    secret_key: str = ""


@dataclass(frozen=True)
class PolymarketConfig:
    """Polymarket credentials using wallet-based authentication.

    Authentication is done via private key + funder address, which is the
    standard py-clob-client auth pattern. The client derives L2 API
    credentials automatically via create_or_derive_api_creds().

    Attributes:
        private_key: Ethereum private key (hex string, with or without 0x prefix).
        funder_address: Polymarket funder/proxy wallet address.
        signature_type: CLOB signature type.
            0 = EOA (standard wallet)
            1 = POLY_PROXY (Polymarket proxy wallet)
            2 = POLY_GNOSIS_SAFE (Polymarket Gnosis Safe, default)
    """
    private_key: str = ""
    funder_address: str = ""
    signature_type: int = 2

    @property
    def is_configured(self) -> bool:
        return bool(self.private_key and self.funder_address)


@dataclass(frozen=True)
class TradingConfig:
    auto_trade_enabled: bool = False
    base_trade_size: float = 1.0
    max_trade_size: float = 3.0
    max_daily_loss: float = 15.0
    max_consecutive_losses: int = 5
    max_open_exposure: float = 3.0
    scaling_increment: float = 0.50
    scaling_profit_step: float = 50.0


@dataclass(frozen=True)
class SystemConfig:
    log_level: str = "INFO"
    data_dir: str = "/data"
    retrain_hour_utc: int = 4
    db_path: str = ""

    def __post_init__(self):
        if not self.db_path:
            object.__setattr__(self, "db_path", os.path.join(self.data_dir, "cleobot.db"))

    @property
    def models_dir(self) -> str:
        return os.path.join(self.data_dir, "models")


@dataclass(frozen=True)
class Config:
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    mexc: MEXCConfig = field(default_factory=MEXCConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)


def load_config() -> Config:
    """Load configuration from environment variables."""
    data_dir = _get_env("DATA_DIR", "./data")

    # Ensure data directories exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(data_dir, "models")).mkdir(parents=True, exist_ok=True)

    return Config(
        telegram=TelegramConfig(
            bot_token=_get_env("TELEGRAM_BOT_TOKEN", ""),
            chat_id=_get_env("TELEGRAM_CHAT_ID", ""),
        ),
        mexc=MEXCConfig(
            api_key=_get_env("MEXC_API_KEY", ""),
            secret_key=_get_env("MEXC_SECRET_KEY", ""),
        ),
        polymarket=PolymarketConfig(
            private_key=_get_env("POLYMARKET_PRIVATE_KEY", ""),
            funder_address=_get_env("POLYMARKET_FUNDER_ADDRESS", ""),
            signature_type=_get_env_int("POLYMARKET_SIGNATURE_TYPE", 2),
        ),
        trading=TradingConfig(
            auto_trade_enabled=_get_env_bool("AUTO_TRADE_ENABLED", False),
            base_trade_size=_get_env_float("BASE_TRADE_SIZE", 1.0),
            max_trade_size=_get_env_float("MAX_TRADE_SIZE", 3.0),
            max_daily_loss=_get_env_float("MAX_DAILY_LOSS", 15.0),
            max_consecutive_losses=_get_env_int("MAX_CONSECUTIVE_LOSSES", 5),
        ),
        system=SystemConfig(
            log_level=_get_env("LOG_LEVEL", "INFO"),
            data_dir=data_dir,
            retrain_hour_utc=_get_env_int("RETRAIN_HOUR_UTC", 4),
        ),
    )
