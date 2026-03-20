# Crypto Trading Bot — Roostoo Hackathon

## Problem Solving Principles

- When encountering warnings, errors, or convergence issues, NEVER just suppress or silence them. Always investigate and fix the root cause first.
- When asked to fix something, do NOT take the lazy path (suppressing, commenting out, or wrapping in try/except). Fix the actual problem.
- Never change working code without isolated testing. Prefer generalization over max returns.
- The user wants code, not plans. Don't spend time planning without action.

## Project Overview

Autonomous crypto trading bot for the SG vs HK University Web3 Quant Trading Hackathon.
- **Roostoo mock exchange** (REST API): BUY/SELL spot crypto pairs
- **5 alpha signals** with **adaptive regime-aware weighting** via z-scores
- **Trend-based 3-state regime detection** on BTC (Bull / Neutral / Bear) with hysteresis
- **Black-Litterman + HRP portfolio optimization** with regime-dependent blending
- **Continuous 24/7 operation** with **2-hour rebalancing** (optimized via frequency comparison)
- **Risk management**: drawdown deleveraging, trailing stops with cooldown, correlation spike detection
- **Allocation smoothing**: 50% EMA blend to prevent abrupt weight jumps

## Key Files

| File | Purpose |
|------|---------|
| `config.py` | All config: trading pairs, API keys, signal weights, risk params |
| `roostoo_client.py` | Roostoo REST API client (auth, orders, balance, ticker) |
| `data_engine.py` | Binance historical OHLCV + SQLite cache |
| `signals.py` | 5 alpha signals, AdaptiveSignalWeighter, TrendRegimeDetector, AlphaEngine |
| `optimizer.py` | Black-Litterman + HRP blend, constraint enforcement, allocation smoothing |
| `risk_manager.py` | VaR, CVaR, drawdown tracking, position stops with cooldown |
| `executor.py` | Order generation and Roostoo API execution |
| `main.py` | TradingBot class, BacktestEngine, continuous loop, CLI |
| `dashboard.py` | Streamlit dashboard (5 tabs) for backtest visualization |
| `frequency_comparison.py` | Backtest different rebalance frequencies (1h-12h) |
| `adaptive_comparison.py` | Compare adaptive vs static signal weighting |
| `pdf_strategies.py` | PDF-recommended strategies: TSMomentum, FundingRateOverlay, PairsStrategy |
| `strategy_comparison.py` | Compare PDF strategies vs existing system across periods/frequencies |

## Roostoo API

- Base URL: `https://mock-api.roostoo.com`
- Auth: HMAC SHA256 with API_KEY and SECRET
- Endpoints: `/v3/serverTime`, `/v3/exchangeInfo`, `/v3/ticker`, `/v3/balance`, `/v3/place_order`, `/v3/query_order`, `/v3/cancel_order`, `/v3/pending_count`
- Order types: MARKET and LIMIT
- Commission: 0.008-0.012% (maker/taker), ~0.01% per side
- Trading pairs: BTC/USD, ETH/USD, SOL/USD, BNB/USD, XRP/USD, DOGE/USD, ADA/USD, AVAX/USD, LINK/USD, DOT/USD, SUI/USD, NEAR/USD, LTC/USD, UNI/USD, AAVE/USD, FIL/USD, HBAR/USD, TRX/USD, APT/USD, SEI/USD

## Signal Stack

5 signals with **adaptive regime-aware weighting**:

1. **MomentumSignal**: Multi-timeframe (12h fast, 72h medium, 168h slow), skip last 2h for noise
2. **BreakoutSignal**: Price vs 72h (3-day) high/low range
3. **VolumeMomentumSignal**: Volume-confirmed momentum
4. **MeanReversionSignal**: 48h lookback with RSI gate for contrarian entries
5. **RelativeStrengthSignal**: Performance vs BTC benchmark

### Adaptive Signal Weights by Regime

| Signal | BULL | NEUTRAL | BEAR |
|--------|------|---------|------|
| Momentum | **40%** | 28% | 16% |
| Breakout | **25%** | 20% | 12% |
| VolumeMomentum | 15% | 16% | 16% |
| MeanReversion | 10% | 20% | **31%** |
| RelativeStrength | 10% | 16% | **25%** |

- **BULL**: Trend-following dominant (momentum + breakout = 65%)
- **BEAR**: Contrarian dominant (mean-reversion + relative strength = 56%)
- IC_IR tracking fine-tunes within ±20% of regime priors (bounded to prevent overfitting)

Trend filter: zero alpha if price < 7-day SMA (168 hours).

## Regime Detection (Trend-Based, NOT HMM)

- **Method**: BTC price vs 7d/3d SMA, SMA slope, drawdown from 14-day high, 24h momentum
- **3 states**: Bull (0), Neutral (1), Bear (2)
- **Hysteresis**: Requires 6 consecutive cycles (24h at 4h frequency) before regime change
  - Crash override: immediate BEAR transition when bear_signals >= 4
  - Step-through: BULL↔BEAR must pass through NEUTRAL (unless crash override)
- **Recalculation**: Every rebalance cycle (every 2 hours)

## Portfolio Optimization

- **Black-Litterman**: Prior = equal-weight, views = alpha scores scaled by view_confidence
- **HRP**: Correlation-distance clustering + recursive inverse-variance bisection
- **Regime blending**:
  - Bull: 80% BL / 20% HRP, view_conf=0.06
  - Neutral: 50% BL / 50% HRP, view_conf=0.04
  - Bear: 25% BL / 75% HRP, view_conf=0.02
- **Allocation smoothing**: 50% EMA blend toward target weights each cycle (prevents jumps)
- Drawdown deleveraging: linear scale 1.0→0.3 between 12%-22% DD
- Vol targeting: 40% annualized (crypto vol), clipped per regime
- Constraints: max 20% single position, max 45% sector, 20-90% invested, 3-8 holdings

## Risk Management

- Drawdown tracking with HWM
- Per-position trailing stops:
  - Default: 12% from position HWM
  - Bear regime: 20% (wider to avoid churn)
- **Stop-loss cooldown**: 6-cycle (24h) re-entry cooldown after stop-loss trigger
  - HWM resets on stop-out so re-entry starts fresh
  - Coins on cooldown get alpha clamped to -1.0 (prevents re-entry)
- Correlation spike detection (avg pairwise > 0.85 = crash regime)
- VaR/CVaR computation (historical method)
- Sortino, Sharpe, Calmar ratio tracking

## Rebalancing

- **Frequency**: Every 2 hours (optimized via backtesting 1h/2h/4h/5h/10h/12h across 3 periods)
- Min trade value: $50 USD
- Min weight change: 3% before trading
- Max turnover: 40% per rebalance cycle
- Sells before buys (free up capital)
- All pending orders cancelled before new cycle

## Execution

- Orders via Roostoo API: MARKET orders by default
- Quantity rounded to exchange precision (pair-specific)

## State Persistence

- `data/bot_state.json`: NAV history, entry prices, rebalance count, commission total
- `data/csv/trade_log.csv`: All executed trades
- `data/csv/portfolio_log.csv`: NAV snapshots with risk metrics
- `data/db/crypto.db`: SQLite cache for historical OHLCV data

## CLI Commands

```bash
python main.py                                              # Run bot (continuous 24/7 loop)
python main.py --once                                       # Single rebalance cycle
python main.py --status                                     # Check portfolio status
python main.py --test                                       # Test API connectivity
python main.py --backtest --start 2024-09-15 --end 2025-03-15  # Backtest
python main.py --backtest --start 2024-10-01 --end 2025-01-01 --adaptive  # Backtest with adaptive weights
python main.py --backtest --start 2025-01-01 --end 2025-03-15 --rebalance-hours 4  # Override frequency
streamlit run dashboard.py                                  # Launch Streamlit dashboard
python frequency_comparison.py                              # Compare rebalancing frequencies
python adaptive_comparison.py                               # Compare adaptive vs static weights
```

## Dashboard (Streamlit)

5-tab layout (`streamlit run dashboard.py`):
1. **Performance**: NAV chart, drawdown, cumulative return
2. **Regime & Allocation**: Regime timeline, invested %, regime distribution pie
3. **Trade Analysis**: Per-coin P&L, trades over time, trade value distribution
4. **Risk & Returns**: Position count, drawdown, hourly/daily return histograms
5. **Compare Runs**: All backtests comparison table, return vs DD scatter, composite bar chart

## Backtest Results (Latest)

| Period | Return | Max DD | Composite | Rebalances | Trades |
|--------|--------|--------|-----------|------------|--------|
| Sep 2024 - Mar 2025 (6mo) | +74.07% | 37.5% | 0.0195 | 2,173 | 2,892 |
| Oct - Dec 2024 (bull) | +86.53% | 22.67% | 0.0362 | - | - |
| Jan - Mar 2025 (bear) | -17.17% | 29.33% | -0.0150 | - | - |

### Frequency Comparison Results (Risk-Adjusted Composite)
- **1h: 0.0019** | **2h: 0.0014** (selected) | 4h: 0.0009 | 5h: -0.0007 | 10h: -0.0076 | 12h: -0.0095

## Hackathon Evaluation Criteria

Composite score: 0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar

## Important Constraints

- NO manual trades allowed — bot must be fully autonomous
- Must run on AWS VM continuously
- At least 8 active trading days with enough trades per day
- Trade log integrity and commit history transparency required
- NEVER RUN any commands that would make manual API calls to Roostoo outside the bot

## Data Storage

- Database: `data/db/crypto.db` (SQLite cache for Binance historical data)
- CSV output: `data/csv/` (trade_log.csv, portfolio_log.csv, backtest results)
- State: `data/bot_state.json`
- Logs: `logs/bot_YYYYMMDD.log`

## Known Limitations / TODO

- ~~Stop-loss cooldown is only in BacktestEngine~~ — DONE: ported to live TradingBot + persisted to bot_state.json
- BULL regime rarely accumulates enough IC observations (regime detector spends most time in NEUTRAL)
- ~~Roostoo API credentials not yet configured~~ — DONE: .env configured, early validation added
- ~~Live bot not tested~~ — DONE: API connectivity verified (exchange running, 50 pairs available)
- Retry logic added to roostoo_client.py (3 retries with 2/5/10s backoff)
- Partial fills now tracked and logged in executor.py
- State saved immediately after trade execution (before CSV logging)
