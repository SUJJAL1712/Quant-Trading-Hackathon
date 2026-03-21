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

# Sector classification — each coin represents a unique price driver
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
    "CFX": "Smart Contract L1",
    "POL": "Smart Contract L1",
    "S": "Smart Contract L1",
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
    "OPEN": "Infrastructure",
    "FORM": "Infrastructure",
    "OMNI": "Infrastructure",
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
    "ZEN": "Privacy",
    "VIRTUAL": "AI",
    "BIO": "DeSci",
}

# ── Asset class classification ──
# Coins grouped by market cap / risk tier for differentiated treatment:
# - Position sizing limits per class
# - Slippage assumptions in backtest
# - Alpha dampening in bear regimes
# - Concentration limits

class AssetClass:
    LARGE_CAP = "large_cap"      # BTC, ETH, SOL, BNB, XRP — deep liquidity, low slippage
    MID_CAP = "mid_cap"          # Established alts — decent liquidity
    SMALL_CAP = "small_cap"      # Emerging tokens — moderate liquidity
    MEME = "meme"                # Meme coins — high vol, speculative
    MICRO_CAP = "micro_cap"      # Very small / new — thin orderbooks

ASSET_CLASS_MAP: Dict[str, str] = {
    # Large-cap majors
    "BTC": AssetClass.LARGE_CAP,
    "ETH": AssetClass.LARGE_CAP,
    "SOL": AssetClass.LARGE_CAP,
    "BNB": AssetClass.LARGE_CAP,
    "XRP": AssetClass.LARGE_CAP,
    # Mid-cap established
    "DOGE": AssetClass.MID_CAP,
    "ADA": AssetClass.MID_CAP,
    "AVAX": AssetClass.MID_CAP,
    "LINK": AssetClass.MID_CAP,
    "DOT": AssetClass.MID_CAP,
    "SUI": AssetClass.MID_CAP,
    "NEAR": AssetClass.MID_CAP,
    "LTC": AssetClass.MID_CAP,
    "TRX": AssetClass.MID_CAP,
    "TON": AssetClass.MID_CAP,
    # Small-cap alts
    "UNI": AssetClass.SMALL_CAP,
    "AAVE": AssetClass.SMALL_CAP,
    "FIL": AssetClass.SMALL_CAP,
    "HBAR": AssetClass.SMALL_CAP,
    "APT": AssetClass.SMALL_CAP,
    "SEI": AssetClass.SMALL_CAP,
    "ARB": AssetClass.SMALL_CAP,
    "ICP": AssetClass.SMALL_CAP,
    "FET": AssetClass.SMALL_CAP,
    "ONDO": AssetClass.SMALL_CAP,
    "PENDLE": AssetClass.SMALL_CAP,
    "ENA": AssetClass.SMALL_CAP,
    "EIGEN": AssetClass.SMALL_CAP,
    "CRV": AssetClass.SMALL_CAP,
    "TAO": AssetClass.SMALL_CAP,
    # Meme coins
    "PEPE": AssetClass.MEME,
    "FLOKI": AssetClass.MEME,
    "BONK": AssetClass.MEME,
    "WIF": AssetClass.MEME,
    "SHIB": AssetClass.MEME,
    "TRUMP": AssetClass.MEME,
    "PENGU": AssetClass.MEME,
    # Micro-cap / thin liquidity
    "WLD": AssetClass.MICRO_CAP,
    "CFX": AssetClass.MICRO_CAP,
    "CAKE": AssetClass.MICRO_CAP,
    "ZEC": AssetClass.MICRO_CAP,
    "ZEN": AssetClass.MICRO_CAP,
    "XLM": AssetClass.MICRO_CAP,
    "POL": AssetClass.MICRO_CAP,
    "VIRTUAL": AssetClass.MICRO_CAP,
    "BIO": AssetClass.MICRO_CAP,
    "OPEN": AssetClass.MICRO_CAP,
    "FORM": AssetClass.MICRO_CAP,
    "OMNI": AssetClass.MICRO_CAP,
    "S": AssetClass.MICRO_CAP,
}

# Per-class constraints and parameters
ASSET_CLASS_PARAMS: Dict[str, Dict] = {
    AssetClass.LARGE_CAP: {
        "max_position_pct": 0.25,    # up to 25% per coin (matches global max)
        "max_class_pct": 0.60,       # up to 60% total in large-cap (anchor)
        "slippage_bps": 1.0,         # tight spreads
        "bear_alpha_mult": 1.0,      # no penalty in bear (flight to quality)
        "min_history_hours": 168,    # 7 days
    },
    AssetClass.MID_CAP: {
        "max_position_pct": 0.15,    # up to 15%
        "max_class_pct": 0.50,       # up to 50% total
        "slippage_bps": 2.0,
        "bear_alpha_mult": 0.8,      # slight penalty in bear
        "min_history_hours": 336,    # 14 days
    },
    AssetClass.SMALL_CAP: {
        "max_position_pct": 0.10,    # up to 10%
        "max_class_pct": 0.35,       # up to 35% total
        "slippage_bps": 3.0,
        "bear_alpha_mult": 0.6,      # moderate penalty in bear
        "min_history_hours": 336,
    },
    AssetClass.MEME: {
        "max_position_pct": 0.08,    # up to 8%
        "max_class_pct": 0.20,       # up to 20% total in meme
        "slippage_bps": 4.0,
        "bear_alpha_mult": 0.3,      # heavy penalty in bear
        "min_history_hours": 336,
    },
    AssetClass.MICRO_CAP: {
        "max_position_pct": 0.06,    # up to 6%
        "max_class_pct": 0.25,       # up to 25% total
        "slippage_bps": 5.0,
        "bear_alpha_mult": 0.4,      # heavy penalty in bear
        "min_history_hours": 504,    # 21 days — need more history
    },
}


def get_asset_class(coin: str) -> str:
    """Get the asset class for a coin, defaulting to micro_cap."""
    return ASSET_CLASS_MAP.get(coin, AssetClass.MICRO_CAP)


def get_class_param(coin: str, param: str, default=None):
    """Get a class-specific parameter for a coin."""
    ac = get_asset_class(coin)
    return ASSET_CLASS_PARAMS.get(ac, {}).get(param, default)


# Benchmark asset for relative strength (BTC is the crypto benchmark)
BENCHMARK = "BTC"

# Safe asset — USD (cash). In crypto, holding USD is the safe haven.
SAFE_ASSET = "USD"


@dataclass
class PortfolioConstraints:
    max_single_position_pct: float = 0.25       # 25% max in one coin
    min_single_position_pct: float = 0.03
    max_sector_pct: float = 0.40                # tighter sector cap with fewer sectors
    min_invested_pct: float = 0.20              # can hold 80% cash in bear
    max_invested_pct: float = 0.90              # max 90% invested, keep 10% buffer
    min_holdings: int = 3                       # at least 3 positions
    max_holdings: int = 8                       # max 8 positions


@dataclass
class RiskParams:
    target_volatility: float = 0.55             # crypto vol is 50-80% annualized; 55% allows adequate exposure
    max_drawdown: float = 0.15                  # 15% max drawdown (realistic for crypto)
    var_confidence: float = 0.95
    lookback_risk_days: int = 90                # 90 days for risk estimation (crypto moves fast)
    vol_scaling: bool = True
    drawdown_deleveraging: bool = True
    dd_deleverage_start: float = 0.08           # start deleveraging at 8% DD (crypto-realistic)
    dd_deleverage_full: float = 0.18            # full deleverage by 18% DD
    position_stop_loss_pct: float = 0.12        # 12% per-position trailing stop
    correlation_spike_threshold: float = 0.85   # crypto always has elevated corr; only panic at 0.85

    # ── Dynamic Risk Aversion (λ) Scaling ──
    lambda_base: float = 2.5                    # baseline BL risk aversion
    lambda_corr_threshold: float = 0.60         # crypto baseline corr ~0.5; only scale above 0.60
    lambda_corr_sensitivity: float = 10.0       # κ: moderate correlation-driven scaling (was 25)
    lambda_dd_sensitivity: float = 3.0          # δ: moderate drawdown-driven scaling (was 6)
    lambda_dd_start: float = 0.05               # start DD-based λ boost at 5% (was 2%)

    # ── Enhanced Volatility Targeting ──
    vol_scale_cap: float = 1.0                  # NEVER lever up — only delever
    ewma_cov_alpha: float = 0.30                # blend recent vol (lower = more stable)

    # ── ATR-Based Dynamic Stop-Losses ──
    atr_stop_multiplier: float = 2.5            # k × realized_vol = stop distance
    atr_lookback_hours: int = 24                # lookback for realized vol estimate
    atr_stop_floor: float = 0.05                # 5% minimum stop distance
    atr_stop_ceiling: float = 0.25              # 25% maximum stop distance


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
    min_trade_value_usd: float = 150.0  # minimum trade size (reduce small churn trades)
    min_weight_change: float = 0.05     # don't trade if weight change < 5% (reduce churn)
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
