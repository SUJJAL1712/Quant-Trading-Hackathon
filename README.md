# Crypto Trading Bot — Roostoo Hackathon

Autonomous crypto trading bot for the SG vs HK University Web3 Quant Trading Hackathon. Competes on Roostoo's mock exchange via REST API.

## Strategy

Multi-signal alpha engine + regime-aware portfolio optimization:

1. **6 Alpha Signals**: Momentum, mean-reversion, volatility, cross-asset correlation, RSI, HMM regime tilt
2. **3-State HMM Regime Detection**: Bull / Neutral / Bear on BTC hourly returns
3. **Black-Litterman + HRP Optimization**: Regime-dependent blending for robust weight estimation
4. **Risk Management**: Drawdown deleveraging, trailing stops, correlation spike detection, VaR/CVaR
5. **Continuous Operation**: 24/7 autonomous rebalancing every 4 hours

## Architecture

```
Binance API (historical OHLCV) + Roostoo API (live prices & execution)
    |
    v
Signal Engine (6 alpha signals + HMM regime)
    |
    v
Portfolio Optimizer (BL + HRP, regime-dependent blend)
    |
    v
Risk Manager (drawdown, stops, correlation monitoring)
    |
    v
Trade Executor (Roostoo API orders with precision handling)
    |
    v
Logging (CSV trade log, portfolio snapshots, risk metrics)
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | All configuration: pairs, API keys, signal params, risk params |
| `roostoo_client.py` | Roostoo REST API client (auth, orders, balance, ticker) |
| `data_engine.py` | Binance historical data fetch + SQLite caching |
| `signals.py` | 6 alpha signals + HMM regime detection + AlphaEngine |
| `optimizer.py` | Black-Litterman + HRP portfolio optimization |
| `risk_manager.py` | VaR, CVaR, drawdown, stops, Sortino/Sharpe/Calmar |
| `executor.py` | Order generation and Roostoo API execution |
| `main.py` | Main bot loop, CLI, state persistence |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy `.env.example` to `.env` and fill in your Roostoo credentials:

```bash
cp .env.example .env
```

```env
ROOSTOO_API_KEY=your_api_key
ROOSTOO_API_SECRET=your_secret
```

### 3. Test connectivity

```bash
python main.py --test
```

## Usage

### Run bot (continuous 24/7)

```bash
python main.py
```

### Single rebalance cycle

```bash
python main.py --once
```

### Check portfolio status

```bash
python main.py --status
```

## Trading Universe

BTC/USD, ETH/USD, BNB/USD, LINK/USD, LTC/USD, EOS/USD, ETC/USD, TRX/USD, BAT/USD

## Evaluation Metrics

The bot optimizes for the hackathon's composite score:
- **0.4 x Sortino Ratio** (return per unit of downside risk)
- **0.3 x Sharpe Ratio** (return per unit of total volatility)
- **0.3 x Calmar Ratio** (return relative to max drawdown)

## Outputs

- `data/csv/trade_log.csv` — All executed trades with timestamps
- `data/csv/portfolio_log.csv` — NAV snapshots, regime, risk metrics
- `data/bot_state.json` — Persistent bot state (survives restarts)
- `logs/bot_YYYYMMDD.log` — Detailed execution logs

## AWS Deployment

```bash
# On AWS EC2 instance
nohup python main.py > bot_output.log 2>&1 &

# Or with systemd service
sudo systemctl start trading-bot
```

## Data Sources

- **Binance API** (public, no auth): Historical OHLCV candlestick data
- **Roostoo API** (authenticated): Live prices, order execution, balance queries
- **SQLite Cache**: Avoids redundant API calls, survives restarts
