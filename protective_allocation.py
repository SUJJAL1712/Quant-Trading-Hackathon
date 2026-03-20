"""
Protective Allocation Framework — Crypto Cash Shield
=====================================================
Analogous to a bond allocation in traditional portfolios, but using USD cash
as the protective asset (crypto has no risk-free yield instrument on Roostoo).

The framework computes a continuous "protection score" from 0.0 (fully risk-on)
to 1.0 (fully risk-off) using multiple market stress indicators. This score
scales down the portfolio's invested allocation, pushing capital to cash.

Trigger parameters (all tunable via config):
1. Portfolio drawdown depth & velocity
2. BTC trend breakdown (price vs multi-timeframe SMAs)
3. Market breadth collapse (fraction of coins in downtrend)
4. Realized volatility spike vs historical norm
5. Cross-asset correlation surge (crash regime)
6. Momentum reversal (recent returns turning negative across timeframes)

The protection score feeds into the optimizer as a multiplier on target_invested,
effectively creating a dynamic cash allocation that scales with market stress.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)


@dataclass
class ProtectiveAllocConfig:
    """Parameters that control when and how aggressively to shift to cash."""

    enabled: bool = True

    # -- Drawdown triggers --
    dd_fast_scale_start: float = 0.03     # start protecting at 3% DD (was 5%)
    dd_fast_scale_full: float = 0.12      # max protection at 12% DD (was 15%)
    dd_weight: float = 0.25              # weight in composite score

    # -- Drawdown velocity (how fast DD is increasing) --
    dd_velocity_lookback: int = 12        # hours to measure DD acceleration
    dd_velocity_threshold: float = 0.025  # 2.5% DD increase in lookback = alarm
    dd_velocity_weight: float = 0.10

    # -- BTC trend breakdown --
    btc_sma_short: int = 72              # 3-day SMA
    btc_sma_long: int = 168             # 7-day SMA
    btc_trend_weight: float = 0.25       # higher weight — BTC trend is strongest signal

    # -- Market breadth --
    breadth_sma_period: int = 72         # 3-day SMA for breadth calc
    breadth_bearish_threshold: float = 0.40   # below this = bearish breadth (was 0.35)
    breadth_crisis_threshold: float = 0.25    # below this = crisis (was 0.20)
    breadth_weight: float = 0.15

    # -- Volatility spike --
    vol_short_window: int = 24           # 24h realized vol
    vol_long_window: int = 168           # 7-day realized vol (baseline)
    vol_spike_threshold: float = 1.3     # short vol > 1.3x long vol = spike (was 1.5)
    vol_crisis_threshold: float = 2.0    # short vol > 2.0x long vol = crisis (was 2.5)
    vol_weight: float = 0.10

    # -- Correlation surge --
    corr_lookback: int = 24              # 24h window for correlation
    corr_threshold: float = 0.70         # avg pairwise > 0.70 = elevated (was 0.75)
    corr_crisis_threshold: float = 0.85  # avg pairwise > 0.85 = crisis
    corr_weight: float = 0.05           # lower weight — already handled by regime

    # -- Multi-timeframe momentum reversal --
    mom_windows: list = field(default_factory=lambda: [24, 72, 168])
    mom_bearish_threshold: float = -0.01  # negative return = bearish (was -0.02)
    mom_weight: float = 0.10

    # -- Protection intensity --
    # The protection score is mapped through this to get the actual cash %
    min_invested_override: float = 0.15   # floor: never below 15% invested
    protection_smoothing_up: float = 0.55   # moderate ramp UP to protect (was 0.65)
    protection_smoothing_down: float = 0.35 # faster ramp DOWN to re-enter (was 0.25)


class ProtectiveAllocationEngine:
    """Computes a continuous protection score and adjusts allocation.

    The protection score aggregates multiple stress indicators into a single
    [0, 1] value. Higher = more stress = more cash allocation.

    Usage in the optimizer:
        protection = engine.compute_protection(prices, returns, drawdown, ...)
        effective_invested = target_invested * (1 - protection.cash_fraction)
    """

    def __init__(self, config: ProtectiveAllocConfig = None):
        self.config = config or ProtectiveAllocConfig()
        self._prev_protection_score = 0.0
        self._dd_history: list = []  # track drawdown over time for velocity

    def compute_protection(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        current_drawdown: float,
        regime: int,
        benchmark: str = None,
    ) -> Dict:
        """Compute the protective allocation adjustment.

        Args:
            prices: Historical close prices (time x coins)
            returns: Historical log returns (time x coins)
            current_drawdown: Current portfolio drawdown (0 to 1)
            regime: Current detected regime (0=Bull, 1=Neutral, 2=Bear)
            benchmark: Benchmark coin (default: BTC)

        Returns:
            Dict with:
                protection_score: float [0, 1] — composite stress score
                cash_fraction: float [0, 1] — how much to shift to cash
                invested_multiplier: float [0, 1] — multiply target_invested by this
                components: dict — individual signal contributions
        """
        if not self.config.enabled:
            return {
                "protection_score": 0.0,
                "cash_fraction": 0.0,
                "invested_multiplier": 1.0,
                "components": {},
            }

        benchmark = benchmark or cfg.BENCHMARK
        components = {}
        scores = []
        weights = []

        # 1. Drawdown depth
        dd_score = self._drawdown_score(current_drawdown)
        components["drawdown"] = dd_score
        scores.append(dd_score)
        weights.append(self.config.dd_weight)

        # 2. Drawdown velocity
        dd_vel_score = self._drawdown_velocity_score(current_drawdown)
        components["dd_velocity"] = dd_vel_score
        scores.append(dd_vel_score)
        weights.append(self.config.dd_velocity_weight)

        # 3. BTC trend breakdown
        btc_score = self._btc_trend_score(prices, benchmark)
        components["btc_trend"] = btc_score
        scores.append(btc_score)
        weights.append(self.config.btc_trend_weight)

        # 4. Market breadth
        breadth_score = self._breadth_score(prices)
        components["breadth"] = breadth_score
        scores.append(breadth_score)
        weights.append(self.config.breadth_weight)

        # 5. Volatility spike
        vol_score = self._volatility_score(prices, benchmark)
        components["volatility"] = vol_score
        scores.append(vol_score)
        weights.append(self.config.vol_weight)

        # 6. Correlation surge
        corr_score = self._correlation_score(returns)
        components["correlation"] = corr_score
        scores.append(corr_score)
        weights.append(self.config.corr_weight)

        # 7. Momentum reversal
        mom_score = self._momentum_reversal_score(prices, benchmark)
        components["momentum_reversal"] = mom_score
        scores.append(mom_score)
        weights.append(self.config.mom_weight)

        # Weighted composite
        total_weight = sum(weights)
        if total_weight > 0:
            raw_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            raw_score = 0.0

        # Regime adjustment: amplify in bear, dampen in bull
        if regime == 2:
            raw_score = min(1.0, raw_score * 1.15)  # 15% boost in bear
        elif regime == 0:
            raw_score = raw_score * 0.7  # 30% dampening in bull

        raw_score = np.clip(raw_score, 0.0, 1.0)

        # Circuit breaker: if 4+ out of 7 signals are above 0.5, skip smoothing
        # and jump directly to high protection. This catches fast crashes.
        high_signals = sum(1 for s in scores if s >= 0.5)
        circuit_breaker = high_signals >= 4

        if circuit_breaker:
            # Immediate high protection — bypass EMA smoothing
            smoothed = max(raw_score, self._prev_protection_score)
            logger.info("CIRCUIT BREAKER: %d/7 signals >= 0.5 -- immediate protection %.3f",
                        high_signals, smoothed)
        else:
            # Asymmetric EMA smoothing: fast UP, slow DOWN
            if raw_score > self._prev_protection_score:
                alpha = self.config.protection_smoothing_up
            else:
                alpha = self.config.protection_smoothing_down
            smoothed = alpha * raw_score + (1 - alpha) * self._prev_protection_score

        self._prev_protection_score = smoothed

        # Map protection score to cash fraction — linear scaled
        # score=0.3 -> cash=21%, score=0.5 -> cash=35%, score=0.8 -> cash=56%
        cash_fraction = float(np.clip(smoothed * 0.7, 0.0, 0.80))

        # Invested multiplier: how much of the regime's target_invested to actually use
        invested_multiplier = 1.0 - cash_fraction

        logger.info(
            "Protection: score=%.3f (raw=%.3f), cash=%.1f%%, mult=%.3f | "
            "DD=%.2f, vel=%.2f, btc=%.2f, breadth=%.2f, vol=%.2f, corr=%.2f, mom=%.2f",
            smoothed, raw_score, cash_fraction * 100, invested_multiplier,
            dd_score, dd_vel_score, btc_score, breadth_score,
            vol_score, corr_score, mom_score,
        )

        return {
            "protection_score": float(smoothed),
            "cash_fraction": float(cash_fraction),
            "invested_multiplier": float(invested_multiplier),
            "components": components,
        }

    def _drawdown_score(self, current_drawdown: float) -> float:
        """Score based on portfolio drawdown depth. 0=no DD, 1=severe DD.

        Piecewise: gentle 3-8% DD, aggressive 8-12% DD.
        3% DD -> 0.0, 8% DD -> 0.3, 10% DD -> 0.65, 12%+ -> 1.0
        """
        start = self.config.dd_fast_scale_start  # 3%
        full = self.config.dd_fast_scale_full     # 12%
        if current_drawdown <= start:
            return 0.0
        if current_drawdown >= full:
            return 1.0
        # Midpoint at 8% DD = score 0.3 (gentle early)
        mid_dd = (start + full) / 2  # ~7.5%
        if current_drawdown <= mid_dd:
            # Gentle ramp: 3% -> 0.0, 7.5% -> 0.3
            return 0.3 * (current_drawdown - start) / (mid_dd - start)
        else:
            # Aggressive ramp: 7.5% -> 0.3, 12% -> 1.0
            return 0.3 + 0.7 * (current_drawdown - mid_dd) / (full - mid_dd)

    def _drawdown_velocity_score(self, current_drawdown: float) -> float:
        """Score based on how fast drawdown is increasing."""
        self._dd_history.append(current_drawdown)
        # Keep bounded history
        max_len = max(self.config.dd_velocity_lookback * 2, 50)
        if len(self._dd_history) > max_len:
            self._dd_history = self._dd_history[-max_len:]

        lookback = self.config.dd_velocity_lookback
        if len(self._dd_history) < lookback:
            return 0.0

        dd_past = self._dd_history[-lookback]
        dd_now = current_drawdown
        velocity = dd_now - dd_past  # positive = DD increasing

        if velocity <= 0:
            return 0.0  # DD improving, no alarm

        threshold = self.config.dd_velocity_threshold
        return min(velocity / threshold, 1.0)

    def _btc_trend_score(self, prices: pd.DataFrame, benchmark: str) -> float:
        """Score based on BTC position relative to SMAs. 0=above both, 1=well below both."""
        if benchmark not in prices.columns:
            return 0.0

        btc = prices[benchmark]
        if len(btc) < self.config.btc_sma_long:
            return 0.0

        price = btc.iloc[-1]
        sma_short = btc.rolling(self.config.btc_sma_short, min_periods=36).mean().iloc[-1]
        sma_long = btc.rolling(self.config.btc_sma_long, min_periods=84).mean().iloc[-1]

        score = 0.0

        # Below short SMA = 0.4 stress
        if price < sma_short:
            gap = (sma_short - price) / sma_short
            score += min(0.4, 0.4 * (gap / 0.05))  # full 0.4 at 5% below

        # Below long SMA = 0.4 stress
        if price < sma_long:
            gap = (sma_long - price) / sma_long
            score += min(0.4, 0.4 * (gap / 0.08))  # full 0.4 at 8% below

        # SMA death cross (short < long) = 0.2 stress
        if sma_short < sma_long:
            score += 0.2

        return min(score, 1.0)

    def _breadth_score(self, prices: pd.DataFrame) -> float:
        """Score based on market breadth. 0=strong breadth, 1=collapsed."""
        if len(prices) < self.config.breadth_sma_period or len(prices.columns) < 5:
            return 0.0

        sma = prices.rolling(self.config.breadth_sma_period, min_periods=36).mean()
        breadth = float((prices.iloc[-1] > sma.iloc[-1]).mean())

        if breadth >= 0.65:
            return 0.0  # strong breadth, no stress

        crisis = self.config.breadth_crisis_threshold
        bearish = self.config.breadth_bearish_threshold

        if breadth <= crisis:
            return 1.0
        if breadth <= bearish:
            return 0.6 + 0.4 * (bearish - breadth) / (bearish - crisis)
        # Between 0.65 and bearish threshold: gradual ramp
        return 0.6 * (0.65 - breadth) / (0.65 - bearish)

    def _volatility_score(self, prices: pd.DataFrame, benchmark: str) -> float:
        """Score based on realized vol spike. 0=normal, 1=crisis vol."""
        if benchmark not in prices.columns:
            return 0.0

        btc = prices[benchmark]
        btc_ret = np.log(btc / btc.shift(1)).dropna()

        short_w = self.config.vol_short_window
        long_w = self.config.vol_long_window

        if len(btc_ret) < long_w:
            return 0.0

        vol_short = float(btc_ret.iloc[-short_w:].std())
        vol_long = float(btc_ret.iloc[-long_w:].std())

        if vol_long <= 0:
            return 0.0

        ratio = vol_short / vol_long

        spike = self.config.vol_spike_threshold
        crisis = self.config.vol_crisis_threshold

        if ratio <= spike:
            return 0.0
        if ratio >= crisis:
            return 1.0
        return (ratio - spike) / (crisis - spike)

    def _correlation_score(self, returns: pd.DataFrame) -> float:
        """Score based on cross-asset correlation. 0=normal, 1=crash correlation."""
        lookback = self.config.corr_lookback
        if len(returns) < lookback or len(returns.columns) < 3:
            return 0.0

        # Use subset of coins for speed
        cols = returns.columns[:min(15, len(returns.columns))]
        recent = returns[cols].iloc[-lookback:].dropna(axis=1, how="all")
        if len(recent.columns) < 3:
            return 0.0

        corr = recent.corr()
        n = len(corr)
        mask = ~np.eye(n, dtype=bool)
        avg_corr = float(corr.values[mask].mean())

        threshold = self.config.corr_threshold
        crisis = self.config.corr_crisis_threshold

        if avg_corr <= threshold:
            return 0.0
        if avg_corr >= crisis:
            return 1.0
        return (avg_corr - threshold) / (crisis - threshold)

    def _momentum_reversal_score(self, prices: pd.DataFrame, benchmark: str) -> float:
        """Score based on multi-timeframe BTC momentum reversal.
        0=positive momentum, 1=all timeframes negative.
        """
        if benchmark not in prices.columns:
            return 0.0

        btc = prices[benchmark]
        negative_count = 0
        total = 0

        for window in self.config.mom_windows:
            if len(btc) < window:
                continue
            total += 1
            ret = btc.iloc[-1] / btc.iloc[-window] - 1
            if ret < self.config.mom_bearish_threshold:
                negative_count += 1
            elif ret < 0:
                negative_count += 0.5

        if total == 0:
            return 0.0
        return negative_count / total
