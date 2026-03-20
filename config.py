"""
Configuration — Crypto Trading Bot for Roostoo Hackathon
=========================================================
Crypto portfolio on Roostoo mock exchange.
- HMM 3-state regime detection on hourly returns (Bull / Neutral / Bear)
- Alpha signals: momentum, mean-reversion, volatility, cross-asset correlation
- Black-Litterman + HRP portfolio optimization
- Continuous 24/7 rebalancing (every 1-4 hours)
- BUY/SELL only (spot, no short selling)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

# ── Roostoo API credentials ──
ROOSTOO_API_KEY = os.getenv("ROOSTOO_API_KEY", "")
ROOSTOO_API_SECRET = os.getenv("ROOSTOO_API_SECRET", "")
ROOSTOO_BASE_URL = "https://mock-api.roostoo.com"

# ── Trading universe ──
# Top-liquidity pairs on Roostoo that also have Binance historical data.
# Dynamically discoverable at startup via exchange_info(), but this static list
# defines which pairs we TRADE (alpha model + optimizer). Others are ignored.
# Orthogonal 15-coin universe: maximally decorrelated price drivers.
# Core 10: each has a genuinely DIFFERENT price driver from every other.
# +5 strategic: accept partial BTC correlation but each has independent dynamics.
TRADING_PAIRS: List[str] = [
    # ── Core orthogonal 10 (lowest cross-correlation) ──
    "BTC/USD",      # Store of Value anchor
    "XRP/USD",      # Payments — lowest BTC correlation (~0.55) among majors
    "BNB/USD",      # Exchange revenue — fee burn economics, not speculation
    "TRX/USD",      # Stablecoin volume driver — USDT ecosystem revenue
    "LINK/USD",     # Oracle monopoly — data feeds, independent catalyst
    "FET/USD",      # AI agent narrative — decouples on AI news cycles
    "TAO/USD",      # AI compute market — orthogonal to financial crypto
    "HBAR/USD",     # Enterprise/gov adoption — different buyer base
    "ONDO/USD",     # RWA tokenization — TradFi bridge, unique catalyst
    "DOGE/USD",     # Social/meme — Elon-driven, decouples on social events
    # ── Strategic 5 (partial BTC corr but independent ecosystem dynamics) ──
    "ETH/USD",      # DeFi base layer — own ecosystem (DeFi TVL, L2 fees)
    "SOL/USD",      # Own ecosystem (Solana memecoins, DePIN, Firedancer)
    "AAVE/USD",     # DeFi lending revenue — fundamental yield, not speculation
    "SUI/USD",      # New-gen L1 — Move-based, different correlation structure
    "AVAX/USD",     # Subnet/institutional — enterprise partnerships
]

# Base coins (without /USD suffix) for data fetching
COINS: List[str] = [p.split("/")[0] for p in TRADING_PAIRS]

# Sector classification — each coin represents a unique price driver
SECTOR_MAP: Dict[str, str] = {
    "BTC":  "Store of Value",
    "XRP":  "Payments",
    "BNB":  "Exchange Token",
    "TRX":  "Exchange Token",
    "LINK": "Infrastructure",
    "FET":  "AI",
    "TAO":  "AI",
    "HBAR": "Infrastructure",
    "ONDO": "RWA",
    "DOGE": "Meme",
    "ETH":  "Smart Contract L1",
    "SOL":  "Smart Contract L1",
    "AAVE": "DeFi",
    "SUI":  "Smart Contract L1",
    "AVAX": "Smart Contract L1",
}

# Benchmark asset for relative strength (BTC is the crypto benchmark)
BENCHMARK = "BTC"

# Safe asset — USD (cash). In crypto, holding USD is the safe haven.
SAFE_ASSET = "USD"


@dataclass
class PortfolioConstraints:
    max_single_position_pct: float = 0.25       # 25% max in one coin (larger with 15-coin universe)
    min_single_position_pct: float = 0.03
    max_sector_pct: float = 0.40                # tighter sector cap with fewer sectors
    min_invested_pct: float = 0.20              # can hold 80% cash in bear
    max_invested_pct: float = 0.90              # max 90% invested, keep 10% buffer
    min_holdings: int = 3                       # at least 3 positions
    max_holdings: int = 8                       # max 8 positions (15-coin universe)


@dataclass
class RiskParams:
    target_volatility: float = 0.25             # tighter vol budget for <6% max DD
    max_drawdown: float = 0.06                  # 6% max drawdown target
    var_confidence: float = 0.95
    lookback_risk_days: int = 90                # 90 days for risk estimation (crypto moves fast)
    vol_scaling: bool = True
    drawdown_deleveraging: bool = True
    dd_deleverage_start: float = 0.03           # start deleveraging at 3% DD (well before 6%)
    dd_deleverage_full: float = 0.08            # full deleverage by 8% DD
    position_stop_loss_pct: float = 0.10        # 10% per-position trailing stop (fallback)
    correlation_spike_threshold: float = 0.80   # lower threshold for earlier crash detection

    # ── Dynamic Risk Aversion (λ) Scaling ──
    lambda_base: float = 2.5                    # baseline BL risk aversion
    lambda_corr_threshold: float = 0.45         # tighter — start scaling earlier
    lambda_corr_sensitivity: float = 25.0       # κ: more aggressive correlation-driven scaling
    lambda_dd_sensitivity: float = 6.0          # δ: more aggressive drawdown-driven scaling
    lambda_dd_start: float = 0.02               # start DD-based λ boost at 2%

    # ── Enhanced Volatility Targeting ──
    vol_scale_cap: float = 1.0                  # NEVER lever up — only delever
    ewma_cov_alpha: float = 0.4                 # higher reactivity to recent vol changes

    # ── ATR-Based Dynamic Stop-Losses ──
    atr_stop_multiplier: float = 2.5            # k × realized_vol = stop distance
    atr_lookback_hours: int = 24                # lookback for realized vol estimate
    atr_stop_floor: float = 0.05                # 5% minimum stop distance
    atr_stop_ceiling: float = 0.20              # 20% maximum stop distance


@dataclass
class SignalParams:
    factor_weights: Dict[str, float] = field(default_factory=lambda: {
        "momentum":          0.25,      # trend-following (strongest in crypto)
        "breakout":          0.20,      # price breakout from range
        "volume_momentum":   0.15,      # volume-confirmed momentum
        "mean_reversion":    0.10,      # short-term with RSI gate
        "relative_strength": 0.15,      # vs BTC benchmark
        "residual_momentum": 0.15,      # idiosyncratic trend after market/beta effects
    })

    # Momentum params
    momentum_fast: int = 12             # 12 hours
    momentum_medium: int = 72           # 3 days
    momentum_slow: int = 168            # 7 days
    momentum_skip: int = 2              # skip last 2 hours (noise)

    # Breakout params
    breakout_lookback: int = 72         # 3-day range for breakout detection

    # Mean reversion params
    mean_reversion_lookback: int = 48   # 48 hours (2 days)

    # Residual momentum params
    residual_momentum_lookback: int = 168   # 7 days

    # Universe quality filters
    min_history_hours: int = 336            # 14 days of data before an asset is tradable
    liquidity_lookback_hours: int = 24      # 24h dollar volume for liquidity filtering
    liquidity_filter_quantile: float = 0.40 # keep top 60% by recent dollar volume

    # Regime detection params (trend-based, not HMM)
    regime_lookback_hours: int = 336    # 14 days lookback for SMA computation
    regime_model: str = "legacy_trend"    # legacy_trend | feature_hmm | feature_gmm | score
    regime_feature_set: str = "market"  # return_only | trend_vol | market
    regime_n_components: int = 4        # allow crash/recovery to stay separate
    regime_train_hours: int = 24 * 90   # 90d rolling fit window
    regime_retrain_hours: int = 24      # refit once per day

    # Trend filter
    trend_filter: bool = True
    trend_filter_ma: int = 168          # 7-day SMA as trend filter


@dataclass
class RebalanceConfig:
    frequency_hours: int = 2            # rebalance every 2 hours (best risk-adj composite)
    min_trade_value_usd: float = 100.0  # minimum trade size (was $50, too much churn)
    min_weight_change: float = 0.04     # don't trade if weight change < 4% (was 3%)
    max_turnover_pct: float = 0.35      # max 35% turnover per rebalance (was 40%)


@dataclass
class CrisisModeConfig:
    """Emergency risk-off for flash crashes."""
    enabled: bool = False
    max_invested_pct: float = 0.30          # cap at 30% invested
    cash_reserve_pct: float = 0.70          # hold 70% USD
    expires_hours: int = 24                 # auto-expire after 24 hours
    activated_at: Optional[str] = None


# ── Binance API for historical data (free, no key needed for public endpoints) ──
BINANCE_BASE_URL = "https://api.binance.com"

# Map Roostoo pairs to Binance symbols for historical data
# Auto-generated: coin + "USDT" works for most Binance pairs
ROOSTOO_TO_BINANCE: Dict[str, str] = {
    f"{coin}/USD": f"{coin}USDT" for coin in COINS
}

# ── Initial capital (set by Roostoo — check exchangeInfo for InitialWallet) ──
INITIAL_CAPITAL_USD = 100_000.0     # placeholder — updated from exchangeInfo at startup

# ── Paths ──
LOG_LEVEL = "INFO"
LOG_DIR = "logs"
DB_PATH = "data/db/crypto.db"
CSV_DIR = "data/csv"
DB_DIR = "data/db"
STATE_FILE = "data/bot_state.json"

# ── Instantiate configs ──
CONSTRAINTS = PortfolioConstraints()
RISK = RiskParams()
SIGNALS = SignalParams()
REBALANCE = RebalanceConfig()
CRISIS_MODE = CrisisModeConfig()
