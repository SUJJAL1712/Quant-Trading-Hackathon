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
# All 67 pairs available on Roostoo (discovered via exchangeInfo)
# We select a high-quality subset: top market-cap + sufficient Binance history
# Skipped: very new tokens (HEMI, PLUME, MIRA, AVNT, TUT, LINEA, SOMI, XPL, ASTER, STO),
#          wrapped/derivative tokens (WLFI, PAXG, 1000CHEEMS),
#          very low liquidity (EDEN, LISTA, BMT, PUMP)
TRADING_PAIRS: List[str] = [
    # Tier 1: Large-cap majors
    "BTC/USD", "ETH/USD", "SOL/USD", "BNB/USD", "XRP/USD",
    # Tier 2: Established large-cap
    "DOGE/USD", "ADA/USD", "AVAX/USD", "LINK/USD", "DOT/USD",
    "SUI/USD", "NEAR/USD", "LTC/USD", "TRX/USD", "TON/USD",
    # Tier 3: Strong mid-cap
    "UNI/USD", "AAVE/USD", "FIL/USD", "HBAR/USD", "APT/USD",
    "SEI/USD", "ARB/USD", "ICP/USD", "FET/USD", "ONDO/USD",
    # Tier 4: High-vol mid-cap (more alpha opportunity)
    "PEPE/USD", "FLOKI/USD", "BONK/USD", "WIF/USD", "SHIB/USD",
    "PENDLE/USD", "ENA/USD", "EIGEN/USD", "CRV/USD", "TAO/USD",
    # Tier 5: Volatile small-cap (high alpha, high risk)
    "WLD/USD", "CFX/USD", "TRUMP/USD", "PENGU/USD", "CAKE/USD",
    "ZEC/USD", "ZEN/USD", "XLM/USD", "POL/USD", "VIRTUAL/USD",
    "BIO/USD", "OPEN/USD", "FORM/USD", "OMNI/USD", "S/USD",
]

# Base coins (without /USD suffix) for data fetching
COINS: List[str] = [p.split("/")[0] for p in TRADING_PAIRS]

# Sector classification for crypto — used for sector cap constraints
SECTOR_MAP: Dict[str, str] = {
    # Store of Value
    "BTC": "Store of Value",
    "LTC": "Store of Value",
    "ZEC": "Store of Value",
    # Smart Contract Platforms / L1
    "ETH": "Smart Contract L1",
    "SOL": "Smart Contract L1",
    "ADA": "Smart Contract L1",
    "AVAX": "Smart Contract L1",
    "NEAR": "Smart Contract L1",
    "SUI": "Smart Contract L1",
    "APT": "Smart Contract L1",
    "SEI": "Smart Contract L1",
    "DOT": "Smart Contract L1",
    "TON": "Smart Contract L1",
    "ICP": "Smart Contract L1",
    # DeFi
    "UNI": "DeFi",
    "AAVE": "DeFi",
    "CRV": "DeFi",
    "PENDLE": "DeFi",
    "CAKE": "DeFi",
    "ENA": "DeFi",
    "EIGEN": "DeFi",
    # Infrastructure / Oracle / AI
    "LINK": "Infrastructure",
    "FIL": "Infrastructure",
    "HBAR": "Infrastructure",
    "FET": "AI",
    "TAO": "AI",
    "ARB": "Infrastructure",
    "ONDO": "RWA",
    # Exchange / Utility
    "BNB": "Exchange Token",
    "TRX": "Exchange Token",
    # Meme / Speculative
    "DOGE": "Meme",
    "PEPE": "Meme",
    "FLOKI": "Meme",
    "BONK": "Meme",
    "WIF": "Meme",
    "SHIB": "Meme",
    "TRUMP": "Meme",
    "PENGU": "Meme",
    # Payments / Other
    "XRP": "Payments",
    "XLM": "Payments",
    "WLD": "Identity",
    "CFX": "Smart Contract L1",
    "ZEN": "Privacy",
    "POL": "Smart Contract L1",
    "VIRTUAL": "AI",
    "BIO": "DeSci",
    "OPEN": "Infrastructure",
    "FORM": "Infrastructure",
    "OMNI": "Infrastructure",
    "S": "Smart Contract L1",
}

# Benchmark asset for relative strength (BTC is the crypto benchmark)
BENCHMARK = "BTC"

# Safe asset — USD (cash). In crypto, holding USD is the safe haven.
SAFE_ASSET = "USD"


@dataclass
class PortfolioConstraints:
    max_single_position_pct: float = 0.20       # max 20% in one coin
    min_single_position_pct: float = 0.03
    max_sector_pct: float = 0.45                # relaxed for crypto sectors
    min_invested_pct: float = 0.05              # can hold 95% cash in bear (was 20%, too high)
    max_invested_pct: float = 0.90              # max 90% invested, keep 10% buffer
    min_holdings: int = 4                       # at least 4 positions for diversification
    max_holdings: int = 15                      # max 15 positions (45-coin universe)


@dataclass
class RiskParams:
    target_volatility: float = 0.40             # crypto vol is 3-5x equity; target ~40% annualized
    max_drawdown: float = 0.25                  # 25% max drawdown trigger
    var_confidence: float = 0.95
    lookback_risk_days: int = 90                # 90 days for risk estimation (crypto moves fast)
    vol_scaling: bool = True
    drawdown_deleveraging: bool = True
    dd_deleverage_start: float = 0.08           # start deleveraging at 8% DD (was 12%)
    dd_deleverage_full: float = 0.18            # full deleverage at 18% DD (was 22%)
    position_stop_loss_pct: float = 0.12        # 12% per-position trailing stop
    correlation_spike_threshold: float = 0.85   # crypto correlations spike in crashes


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
