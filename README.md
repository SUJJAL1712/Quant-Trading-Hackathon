# Crypto Trading Bot -- Roostoo Hackathon

Autonomous crypto trading bot for the SG vs HK University Web3 Quant Trading Hackathon. Competes on Roostoo's mock exchange via REST API.

## Strategy

Multi-signal alpha engine + regime-aware portfolio optimization + protective allocation:

1. **6 Alpha Signals**: Momentum, breakout, volume-momentum, mean-reversion, relative strength, residual momentum
2. **Trend-Based 3-State Regime Detection**: Bull / Neutral / Bear using BTC trend + market breadth + volatility
3. **Black-Litterman + HRP Optimization**: Regime-dependent blending for robust weight estimation
4. **Protective Allocation Framework**: Continuous stress-scoring (7 indicators) shifts allocation to cash in bear conditions
5. **Risk Management**: Drawdown deleveraging, trailing stops with cooldown, correlation spike detection, VaR/CVaR
6. **Continuous Operation**: 24/7 autonomous rebalancing every 2 hours

## Architecture

```
Binance API (historical OHLCV) + Roostoo API (live prices & execution)
    |
    v
Signal Engine (6 alpha signals + trend regime detector)
    |
    v
Protective Allocation (7 stress indicators -> cash shield)
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
| `signals.py` | 6 alpha signals + trend regime detection + AlphaEngine |
| `optimizer.py` | Black-Litterman + HRP portfolio optimization |
| `protective_allocation.py` | Protective cash allocation framework |
| `risk_manager.py` | VaR, CVaR, drawdown, stops, Sortino/Sharpe/Calmar |
| `executor.py` | Order generation and Roostoo API execution |
| `main.py` | Main bot loop, CLI, BacktestEngine, state persistence |
| `dashboard.py` | Streamlit dashboard (5 tabs) for backtest visualization |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root with your Roostoo credentials:

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

### Backtest

```bash
python main.py --backtest --start 2024-10-01 --end 2025-03-15
```

### Dashboard

```bash
streamlit run dashboard.py
```

## Trading Universe

45 pairs including BTC/USD, ETH/USD, SOL/USD, BNB/USD, XRP/USD, and 40 more mid/small-cap tokens. Full list in `config.py`.

## Evaluation Metrics

The bot optimizes for the hackathon's composite score:
- **0.4 x Sortino Ratio** (return per unit of downside risk)
- **0.3 x Sharpe Ratio** (return per unit of total volatility)
- **0.3 x Calmar Ratio** (return relative to max drawdown)

## Outputs

- `data/csv/trade_log.csv` -- All executed trades with timestamps
- `data/csv/portfolio_log.csv` -- NAV snapshots, regime, risk metrics
- `data/bot_state.json` -- Persistent bot state (survives restarts)
- `logs/bot_YYYYMMDD.log` -- Detailed execution logs

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
