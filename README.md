# CleoBot — Polymarket 5-min BTC Up/Down Trading Bot

CleoBot is a fully automated trading bot that trades BTC Up/Down markets on Polymarket using a 5-minute candle ML pipeline.

## Architecture

```
MEXC WebSocket (BTC price, orderbook)
         ↓
   DataCollector (5m, 15m, 1h candles + orderbook)
         ↓
   FeatureEngine (80+ features: technical, orderbook, market microstructure)
         ↓
   Ensemble (LightGBM + TCN + LogReg → MetaLearner + RegimeDetector)
         ↓
   SignalFilter (6 filters: confidence, volatility, regime, agreement, streak, correlation)
         ↓
   RiskManager (daily loss limit, open exposure, position sizing)
         ↓
   PolymarketClient → place UP/DOWN order
         ↓
   TelegramBot → signal cards + trade results
```

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/blinkinfo/cleobot.git
cd cleobot
pip install -r requirements.txt
```

### 2. Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Yes | Bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | Yes | Your Telegram chat/channel ID |
| `MEXC_API_KEY` | No | MEXC API key (for WebSocket auth; public data works without) |
| `MEXC_SECRET_KEY` | No | MEXC secret key |
| `POLYMARKET_API_KEY` | Yes (for live) | Polymarket CLOB API key |
| `POLYMARKET_API_SECRET` | Yes (for live) | Polymarket API secret |
| `POLYMARKET_API_PASSPHRASE` | Yes (for live) | Polymarket API passphrase |
| `AUTO_TRADE_ENABLED` | No | `true` to enable live trading (default: `false`) |
| `BASE_TRADE_SIZE` | No | Trade size in USD (default: `1.0`) |
| `MAX_TRADE_SIZE` | No | Maximum trade size (default: `3.0`) |
| `MAX_DAILY_LOSS` | No | Daily loss circuit breaker in USD (default: `15.0`) |
| `MAX_CONSECUTIVE_LOSSES` | No | Pause after N losses (default: `5`) |
| `DATA_DIR` | No | Data directory for DB + models (default: `./data`) |
| `LOG_LEVEL` | No | `DEBUG`, `INFO`, `WARNING` (default: `INFO`) |
| `RETRAIN_HOUR_UTC` | No | Hour for daily retrain (default: `4`) |
| `HEALTH_PORT` | No | Health check HTTP port (default: `8080`) |

### 3. Run Locally

```bash
# Signal-only mode (no trades placed)
AUTO_TRADE_ENABLED=false python -m src.main

# Paper trading (simulated orders)
AUTO_TRADE_ENABLED=true python -m src.main

# With .env file
python -m dotenv run python -m src.main
```

## Railway Deployment

### One-Click Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

### Manual Deploy

1. Install [Railway CLI](https://docs.railway.app/develop/cli): `npm install -g @railway/cli`
2. Login: `railway login`
3. Link project: `railway link`
4. Set environment variables via Railway dashboard or CLI:
   ```bash
   railway variables set TELEGRAM_BOT_TOKEN=your_token
   railway variables set TELEGRAM_CHAT_ID=your_chat_id
   railway variables set AUTO_TRADE_ENABLED=false
   # ... set all other required variables
   ```
5. Deploy: `railway up`

### Persistent Volume

CleoBot stores the SQLite database and trained models in `/data`. Configure a Railway volume:

1. In Railway dashboard → your service → **Volumes**
2. Mount path: `/data`
3. This persists across deploys and restarts

### Health Check

Railway uses the `/health` endpoint on port `8080` (configurable via `HEALTH_PORT`):

```
GET http://your-service.railway.app/health
```

Response:
```json
{
  "status": "ok",
  "uptime_s": 3600.0,
  "auto_trade": false,
  "candles_5m": 1440,
  "models_ready": true,
  "ts": "2024-01-01T12:00:00+00:00"
}
```

The `railway.toml` is pre-configured:
```toml
[deploy]
startCommand = "python -m src.main"
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
```

## Trading Modes

| Mode | `AUTO_TRADE_ENABLED` | Description |
|------|---------------------|-------------|
| Signals Only | `false` | Generates and sends signals to Telegram, no trades placed |
| Paper Trading | `true` + no Polymarket creds | Simulated orders (no real money) |
| Live Trading | `true` + Polymarket creds configured | Real Polymarket orders |

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | Current bot status, models, risk metrics |
| `/signal` | Last generated signal |
| `/performance` | Today's P&L, win rate, trade history |
| `/risk` | Risk manager status (daily loss, exposure) |
| `/start` | Resume trading after 7-loss streak pause |
| `/stop` | Pause auto-trading (signals continue) |
| `/help` | All available commands |

## Signal Card Format

```
⭐ PREMIUM SIGNAL ⭐         (or 🟢 SIGNAL: UP)
🕐 14:02 UTC
📊 Conf: 72% [=======   ]
🎯 Regime: Trending Up
🤝 Agreement: 3/3

🛡️ Filters:
  ✅ Confidence: PASS (72% > 58%)
  ✅ Volatility: PASS (ATR: 45th pctl)
  ✅ Regime: PASS (Trending, threshold=56%)
  ✅ Agreement: PASS (3/3)
  ✅ Streak: PASS (no loss streak)
  ✅ Correlation: PASS (64.0% / 50 trades)

💰 TRADE PLACED
  Size: $1.00
  Price: 0.550
  Trade ID: #42

🤖 Models:
  LGBM: UP (68%)
  TCN:  UP (71%)
  LR:   UP (63%)

💳 Risk:
  Daily PnL: +$2.64 / -$15
  Open: $1.00 / $3.00
  W/L today: 3W / 1L
```

## 5-Minute Cycle Timing

```
:00:00  Candle closes
:00:05  Settlement check (check previous trade result)
:02:00  Trading cycle fires
  ├── Feature computation (< 33s)
  ├── Ensemble inference (< 5s)  
  ├── Filter pipeline (< 1s)
  ├── Risk check (< 1s)
  └── FOK market order placement (instant fill or kill)
```

## Risk Management

- **Daily loss limit**: Circuit breaker at $15 (configurable)
- **Max open exposure**: $3.00 maximum concurrent positions
- **Position scaling**: Base $1.00, +$0.50 per $50 cumulative profit, max $3.00
- **Streak pause**: 1 cycle after 3 losses, 3 cycles after 5 losses
- **Hard stop**: Manual `/start` required after 7 consecutive losses
- **Emergency brake**: Trading paused if rolling accuracy drops below 50%

## Model Training

Models are trained daily at 04:00 UTC on the last 10,000 candles. Incremental updates run every 6 hours.

Initial training is triggered automatically on first startup if no models exist (requires 200+ candles).

```bash
# Manual retrain
python -c "
from src.config import load_config
from src.database import Database
from src.models.trainer import ModelTrainer
import pandas as pd

config = load_config()
db = Database(config.system.db_path)
candles = db.get_candles('candles_5m', limit=5000)
df = pd.DataFrame(candles)
trainer = ModelTrainer(db=db, models_dir=config.system.models_dir)
results = trainer.initial_training(df_5m=df)
print(results)
"
```

## Tests

```bash
# Run all tests
pip install pytest pytest-asyncio
pytest tests/ -v --tb=short

# Run specific test files
pytest tests/test_features.py -v
pytest tests/test_pipeline.py -v
```

## Project Structure

```
cleobot/
├── src/
│   ├── main.py              # Entry point + orchestrator
│   ├── config.py            # Environment config
│   ├── database.py          # SQLite data layer
│   ├── data/
│   │   ├── mexc_ws.py       # MEXC WebSocket client
│   │   ├── mexc_rest.py     # MEXC REST client
│   │   ├── collector.py     # DataCollector (candles, orderbook)
│   │   └── backfill.py      # Historical data backfill
│   ├── features/
│   │   └── engine.py        # Feature computation (80+ features)
│   ├── models/
│   │   ├── ensemble.py      # 3-layer ensemble orchestrator
│   │   ├── lgbm_model.py    # LightGBM model
│   │   ├── tcn_model.py     # Temporal Convolutional Network
│   │   ├── logreg_model.py  # Logistic Regression
│   │   ├── meta_learner.py  # Meta-learner (stacking)
│   │   ├── regime_detector.py # HMM regime detection
│   │   └── trainer.py       # Model training pipeline
│   ├── trading/
│   │   ├── executor.py      # 5-min cycle orchestrator
│   │   ├── filters.py       # 6 signal filters
│   │   ├── risk_manager.py  # Risk management
│   │   └── polymarket.py    # Polymarket CLOB client
│   ├── telegram_bot/
│   │   ├── bot.py           # Telegram bot + handlers
│   │   └── notifications.py # Notification formatters
│   └── utils/
│       ├── logger.py        # Logging setup
│       └── scheduler.py     # APScheduler setup
├── tests/
│   ├── test_features.py     # Feature engine tests
│   └── test_pipeline.py     # Integration tests
├── Dockerfile
├── railway.toml
├── requirements.txt
└── README.md
```

## License

Private — all rights reserved.
