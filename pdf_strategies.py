"""
PDF Strategy Implementations — Four-Hour Rebalance Crypto Trading Strategies
=============================================================================
Implements the three strategy layers recommended by the research PDF:
1. TSMomentumStrategy — Vol-managed time-series momentum (core)
2. FundingRateOverlay — Carry/crowdedness signal from funding rates (add-on)
3. PairsStrategy — Distance-based pairs trading on majors (satellite)

All strategies produce pd.Series alpha scores compatible with the existing
PortfolioOptimizer.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

import config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def zscore(series: pd.Series, clip: float = 3.0) -> pd.Series:
    """Cross-sectional z-score with clipping."""
    if series.std() == 0 or series.isna().all():
        return series * 0.0
    z = (series - series.mean()) / series.std()
    return z.clip(-clip, clip)


def ewma_vol(returns: pd.Series, halflife: int = 72) -> pd.Series:
    """EWMA volatility estimate."""
    return returns.ewm(halflife=halflife, min_periods=max(halflife // 2, 1)).std()


# ===========================================================================
# Strategy A: Volatility-Managed Time-Series Momentum
# ===========================================================================

class TSMomentumStrategy:
    """Per-asset time-series momentum with EWMA volatility scaling.

    Key differences from the existing cross-sectional MomentumSignal:
    - Time-series: each asset's signal depends only on its own history
    - Vol-managed: position size inversely proportional to recent volatility
    - Long-only: go to cash when signal is negative
    - Multi-timeframe ensemble: fast/medium/slow lookbacks

    Parameters (PDF recommended ranges for 4h bars):
    - Lookbacks: 18, 72, 168 bars (3d, 12d, 28d)
    - Vol halflife: 24-168 bars
    - Target annualized vol: 20% (middle of PDF's 10-25% range)
    """

    def __init__(self, lookbacks: List[int] = None,
                 vol_halflife: int = 72,
                 target_ann_vol: float = 0.20):
        self.lookbacks = lookbacks or [18, 72, 168]
        self.vol_halflife = vol_halflife
        self.target_ann_vol = target_ann_vol
        # Weights for fast/medium/slow (more weight on slow = more stable)
        self.lookback_weights = [0.20, 0.30, 0.50]

    def generate(self, prices: pd.DataFrame, returns: pd.DataFrame,
                 rebalance_hours: int = 4) -> pd.Series:
        """Compute vol-managed TSMOM alpha scores.

        Args:
            prices: Close price DataFrame (time x coins)
            returns: Log return DataFrame (time x coins)
            rebalance_hours: Current rebalance frequency in hours

        Returns:
            pd.Series of alpha scores (positive = long, 0 = cash)
        """
        coins = prices.columns.tolist()
        alpha = pd.Series(0.0, index=coins)

        if len(prices) < max(self.lookbacks):
            return alpha

        # Annualization factor from hourly returns
        ann_factor = np.sqrt(365 * 24)
        # Per-bar target vol
        target_bar_vol = self.target_ann_vol / ann_factor

        for i, lookback in enumerate(self.lookbacks):
            if len(prices) < lookback:
                continue

            weight = self.lookback_weights[i] if i < len(self.lookback_weights) else 0.2

            for coin in coins:
                # Time-series momentum signal: sign of return over lookback
                ret = prices[coin].iloc[-1] / prices[coin].iloc[-lookback] - 1

                # Signal: positive return = long, negative = cash (0)
                signal = max(ret, 0)  # long-only: clamp negatives to 0

                # EWMA volatility scaling
                if coin in returns.columns and len(returns[coin].dropna()) >= self.vol_halflife:
                    recent_vol = ewma_vol(returns[coin], self.vol_halflife).iloc[-1]
                    if recent_vol > 0 and np.isfinite(recent_vol):
                        vol_scale = target_bar_vol / recent_vol
                        vol_scale = np.clip(vol_scale, 0.1, 3.0)  # cap extreme scaling
                    else:
                        vol_scale = 1.0
                else:
                    vol_scale = 1.0

                alpha[coin] += weight * signal * vol_scale

        # Z-score for cross-asset comparability
        return zscore(alpha)


# ===========================================================================
# Strategy B: Funding Rate Overlay
# ===========================================================================

class FundingRateOverlay:
    """Fetch Binance perpetual funding rates and use as crowdedness indicator.

    Extremely positive funding = crowded longs → reduce alpha (risk of squeeze)
    Extremely negative funding = crowded shorts → reduce alpha for longs

    This is an overlay: it scales existing alpha scores, not a standalone signal.

    Data source: Binance USDT-Margined Futures GET /fapi/v1/fundingRate
    """

    BINANCE_FUTURES_URL = "https://fapi.binance.com"

    def __init__(self, lookback_periods: int = 30, extreme_z: float = 1.5):
        self.lookback_periods = lookback_periods  # number of funding periods for z-score
        self.extreme_z = extreme_z
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: float = 0

    def fetch_funding_rates(self, symbols: List[str],
                            start_time: int = None,
                            end_time: int = None) -> Dict[str, pd.DataFrame]:
        """Fetch historical funding rates from Binance Futures API.

        Returns {symbol: DataFrame with columns [fundingRate, fundingTime]}
        """
        result = {}
        for symbol in symbols:
            binance_symbol = f"{symbol}USDT"
            try:
                params = {"symbol": binance_symbol, "limit": 100}
                if start_time:
                    params["startTime"] = start_time
                if end_time:
                    params["endTime"] = end_time

                resp = requests.get(
                    f"{self.BINANCE_FUTURES_URL}/fapi/v1/fundingRate",
                    params=params, timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        df = pd.DataFrame(data)
                        df["fundingRate"] = df["fundingRate"].astype(float)
                        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
                        df = df.set_index("fundingTime").sort_index()
                        result[symbol] = df
                time.sleep(0.05)  # rate limit
            except Exception as e:
                logger.debug("Failed to fetch funding for %s: %s", symbol, e)

        return result

    def fetch_funding_for_backtest(self, symbols: List[str],
                                   start: pd.Timestamp,
                                   end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """Fetch funding rates covering a backtest period."""
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        all_funding = {}
        for symbol in symbols:
            binance_symbol = f"{symbol}USDT"
            frames = []
            current_start = start_ms

            while current_start < end_ms:
                try:
                    params = {
                        "symbol": binance_symbol,
                        "startTime": current_start,
                        "endTime": end_ms,
                        "limit": 1000,
                    }
                    resp = requests.get(
                        f"{self.BINANCE_FUTURES_URL}/fapi/v1/fundingRate",
                        params=params, timeout=10,
                    )
                    if resp.status_code != 200 or not resp.json():
                        break
                    data = resp.json()
                    df = pd.DataFrame(data)
                    df["fundingRate"] = df["fundingRate"].astype(float)
                    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
                    frames.append(df)
                    last_ts = int(df["fundingTime"].iloc[-1].timestamp() * 1000)
                    if last_ts <= current_start:
                        break
                    current_start = last_ts + 1
                    time.sleep(0.05)
                except Exception:
                    break

            if frames:
                combined = pd.concat(frames).drop_duplicates(subset=["fundingTime"])
                combined = combined.set_index("fundingTime").sort_index()
                all_funding[symbol] = combined

        return all_funding

    def compute_overlay(self, alpha_scores: pd.Series,
                        funding_data: Dict[str, pd.DataFrame],
                        current_time: pd.Timestamp = None) -> pd.Series:
        """Scale alpha scores based on funding rate extremes.

        Reduces alpha when funding indicates crowded positioning.
        """
        scaled = alpha_scores.copy()

        for coin in alpha_scores.index:
            if coin not in funding_data or funding_data[coin].empty:
                continue

            df = funding_data[coin]
            if current_time is not None:
                df = df[df.index <= current_time]

            if len(df) < 3:
                continue

            # Get recent funding rates
            recent = df["fundingRate"].iloc[-self.lookback_periods:]
            if len(recent) < 3:
                continue

            # Z-score of current funding rate
            current_rate = recent.iloc[-1]
            mean_rate = recent.mean()
            std_rate = recent.std()
            if std_rate < 1e-10:
                continue

            funding_z = (current_rate - mean_rate) / std_rate

            # Scale down alpha when funding is extreme
            if abs(funding_z) > self.extreme_z:
                # Reduce alpha proportionally to how extreme funding is
                # At z=1.5: scale=0.8, at z=3.0: scale=0.4
                dampening = max(0.3, 1.0 - 0.2 * (abs(funding_z) - self.extreme_z))
                scaled[coin] *= dampening

                # Extra penalty if trying to go long with very positive funding (crowded longs)
                if funding_z > self.extreme_z and alpha_scores[coin] > 0:
                    scaled[coin] *= 0.7  # additional 30% reduction

        return scaled


# ===========================================================================
# Strategy C: Distance-Based Pairs Trading
# ===========================================================================

class PairsStrategy:
    """Distance-based pairs trading on major crypto pairs.

    Uses normalized price distance to detect mean-reverting spread deviations.
    Only trades the most liquid pairs to ensure stable cointegration.

    Parameters (PDF recommended for 4h bars):
    - Formation window: 180-720 bars (30-120 days)
    - Entry z-threshold: 1.5
    - Exit z-threshold: 0.5
    - Stop z-threshold: 4.0
    """

    # Only trade the most liquid, stably-correlated pairs
    PAIR_UNIVERSE = [
        ("BTC", "ETH"),
        ("BTC", "SOL"),
        ("ETH", "SOL"),
        ("BTC", "BNB"),
        ("ETH", "BNB"),
        ("BTC", "XRP"),
    ]

    def __init__(self, formation_window: int = 360,
                 entry_z: float = 1.5, exit_z: float = 0.5, stop_z: float = 4.0):
        self.formation_window = formation_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        # Track active pair positions: {(coinA, coinB): direction}
        self._active_pairs: Dict[Tuple[str, str], float] = {}

    def generate(self, prices: pd.DataFrame) -> pd.Series:
        """Compute pairs-trading alpha scores.

        Returns alpha scores where:
        - Positive = expect price to rise (undervalued in pair)
        - Negative = expect price to fall (overvalued in pair)
        """
        coins = prices.columns.tolist()
        alpha = pd.Series(0.0, index=coins)

        if len(prices) < self.formation_window:
            return alpha

        for coin_a, coin_b in self.PAIR_UNIVERSE:
            if coin_a not in coins or coin_b not in coins:
                continue

            # Compute log price ratio (spread)
            pa = prices[coin_a].iloc[-self.formation_window:]
            pb = prices[coin_b].iloc[-self.formation_window:]

            if pa.std() == 0 or pb.std() == 0:
                continue

            # Normalize prices to start at 1.0
            pa_norm = pa / pa.iloc[0]
            pb_norm = pb / pb.iloc[0]

            # Spread = log ratio
            spread = np.log(pa_norm / pb_norm)
            spread_mean = spread.mean()
            spread_std = spread.std()

            if spread_std < 1e-8:
                continue

            # Current z-score of spread
            current_z = (spread.iloc[-1] - spread_mean) / spread_std

            pair_key = (coin_a, coin_b)

            # Check stop-loss on active positions
            if pair_key in self._active_pairs:
                direction = self._active_pairs[pair_key]
                if abs(current_z) > self.stop_z:
                    # Stop out
                    del self._active_pairs[pair_key]
                    continue
                elif abs(current_z) < self.exit_z:
                    # Mean reverted, exit
                    del self._active_pairs[pair_key]
                    continue

            # Entry signals
            if current_z > self.entry_z:
                # Spread too wide: short A, long B (expect A to underperform)
                signal_strength = min(current_z / self.entry_z, 2.0) - 1.0
                alpha[coin_a] -= signal_strength * 0.5
                alpha[coin_b] += signal_strength * 0.5
                self._active_pairs[pair_key] = -1.0

            elif current_z < -self.entry_z:
                # Spread too narrow: long A, short B (expect A to outperform)
                signal_strength = min(abs(current_z) / self.entry_z, 2.0) - 1.0
                alpha[coin_a] += signal_strength * 0.5
                alpha[coin_b] -= signal_strength * 0.5
                self._active_pairs[pair_key] = 1.0

            # Maintain active positions
            elif pair_key in self._active_pairs:
                direction = self._active_pairs[pair_key]
                # Weaker continuation signal
                alpha[coin_a] += direction * 0.2
                alpha[coin_b] -= direction * 0.2

        return zscore(alpha)


# ===========================================================================
# Combined PDF Alpha Engine
# ===========================================================================

class PDFAlphaEngine:
    """Combined alpha engine using PDF strategy stack.

    Configurable strategy combination:
    - tsmom_only: Just TSMomentum
    - tsmom_funding: TSMomentum + FundingRateOverlay
    - tsmom_pairs: TSMomentum + PairsStrategy
    - full_stack: All three combined

    Uses the existing TrendRegimeDetector for regime detection.
    """

    STRATEGY_MODES = ["tsmom_only", "tsmom_funding", "tsmom_pairs", "full_stack"]

    def __init__(self, mode: str = "tsmom_only",
                 rebalance_hours: int = 4,
                 target_ann_vol: float = 0.20):
        from signals import TrendRegimeDetector

        self.mode = mode
        self.rebalance_hours = rebalance_hours
        self.regime_detector = TrendRegimeDetector(rebalance_hours=rebalance_hours)

        # Strategy components
        self.tsmom = TSMomentumStrategy(target_ann_vol=target_ann_vol)
        self.funding_overlay = FundingRateOverlay() if "funding" in mode or mode == "full_stack" else None
        self.pairs = PairsStrategy() if "pairs" in mode or mode == "full_stack" else None

        # Funding data cache (fetched once for backtest)
        self._funding_data: Optional[Dict[str, pd.DataFrame]] = None

    def set_funding_data(self, funding_data: Dict[str, pd.DataFrame]):
        """Pre-load funding data for backtesting (avoid repeated API calls)."""
        self._funding_data = funding_data

    def compute_alpha(self, prices: pd.DataFrame, returns: pd.DataFrame,
                      volumes: pd.DataFrame = None,
                      current_time: pd.Timestamp = None) -> pd.Series:
        """Compute composite alpha using PDF strategy stack."""

        # Infer current_time from prices if not provided
        if current_time is None and not prices.empty:
            current_time = prices.index[-1]

        # Detect regime
        try:
            regime = self.regime_detector.detect(prices)
        except Exception:
            regime = 1

        # Core: Time-series momentum (always present)
        alpha = self.tsmom.generate(prices, returns, self.rebalance_hours)

        # Add-on: Pairs trading signal
        if self.pairs is not None:
            pairs_alpha = self.pairs.generate(prices)
            # Blend: 70% TSMOM + 30% pairs
            alpha = 0.70 * alpha + 0.30 * pairs_alpha

        # Overlay: Funding rate scaling
        if self.funding_overlay is not None and self._funding_data:
            alpha = self.funding_overlay.compute_overlay(
                alpha, self._funding_data, current_time)

        # Apply regime-based scaling (defensive in bear)
        if regime == 2:  # Bear
            alpha = alpha.clip(upper=0.5)  # cap bullish signals
        elif regime == 0:  # Bull
            alpha = alpha.clip(lower=-0.5)  # cap bearish signals

        return zscore(alpha)

    @property
    def current_regime(self) -> int:
        return self.regime_detector.current_regime

    @property
    def regime_name(self) -> str:
        names = {0: "BULL", 1: "NEUTRAL", 2: "BEAR"}
        return names.get(self.current_regime, "UNKNOWN")

    @property
    def adaptive(self) -> bool:
        return False  # PDF strategies don't use adaptive IC tracking

    @property
    def adaptive_diagnostics(self) -> Dict:
        return {}
