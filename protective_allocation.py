"""
Protective Allocation Framework — Crypto Cash Shield
=====================================================
Analogous to a bond allocation in traditional portfolios, but using USD cash
as the protective asset (crypto has no risk-free yield instrument on Roostoo).

The framework uses a data-driven 2-signal combo selected via exhaustive
backtest analysis of all pairwise OR combinations from 9 stress indicators:

    BEST COMBO:  BTC 24h Vol  OR  Portfolio 6h Return
    Net saving:  +5.20%  (LossCaught=-12.13%  GainMissed=+6.93%)

When EITHER signal fires, the engine shifts allocation toward cash.
This combo was selected because:
- BTC 24h Vol catches volatility spikes early (3.7x crisis/calm ratio)
- Portfolio 6h Return catches sudden portfolio drops (2.1x ratio)
- Together they cover distinct failure modes with minimal overlap

The protection score feeds into the optimizer as a multiplier on target_invested,
effectively creating a dynamic cash allocation that scales with market stress.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)


@dataclass
class ProtectiveAllocConfig:
    """Parameters that control when and how aggressively to shift to cash."""

    enabled: bool = True

    # -- Signal 1: BTC 24h Realized Volatility Spike --
    # Fires when 24h vol exceeds threshold relative to 7d baseline
    vol_short_window: int = 24           # 24h realized vol
    vol_long_window: int = 168           # 7-day realized vol (baseline)
    vol_spike_threshold: float = 1.3     # short vol > 1.3x long vol = spike
    vol_crisis_threshold: float = 2.0    # short vol > 2.0x long vol = crisis (score=1.0)

    # -- Signal 2: Portfolio Return (dual window: 6h main + 3h fast) --
    # Fires when portfolio drops more than threshold in either window
    portfolio_ret_window: int = 6        # 6-hour return lookback (main)
    portfolio_ret_threshold: float = -0.007  # Optuna (was -0.008): more sensitive to drops
    portfolio_ret_crisis: float = -0.025     # -2.5% in 6h = crisis (score=1.0)
    portfolio_ret_fast_window: int = 3   # 3-hour return (fast detector)
    portfolio_ret_fast_threshold: float = -0.006  # -0.6% in 3h = fire

    # -- Drawdown depth (supplementary, always active) --
    dd_fast_scale_start: float = 0.03     # start protecting at 3% DD
    dd_fast_scale_full: float = 0.12      # max protection at 12% DD

    # -- Protection intensity --
    min_invested_override: float = 0.15   # floor: never below 15% invested
    protection_smoothing_up: float = 0.90   # Optuna (was 0.70): near-instant protection ramp
    protection_smoothing_down: float = 0.25 # Optuna (was 0.30): slightly slower re-entry

    # -- Circuit breaker: if both signals fire simultaneously --
    circuit_breaker_both: bool = True     # skip smoothing if both fire


class ProtectiveAllocationEngine:
    """Computes a continuous protection score using 2-signal OR combo.

    The protection score is based on:
    1. BTC 24h Volatility spike (vs 7d baseline)
    2. Portfolio 6h Return (negative = stress)
    + Drawdown depth as a supplementary scaler

    When EITHER signal fires (OR logic), protection activates.
    When BOTH fire, circuit breaker engages (immediate max protection).

    Usage in the optimizer:
        protection = engine.compute_protection(prices, returns, drawdown, ...)
        effective_invested = target_invested * (1 - protection.cash_fraction)
    """

    def __init__(self, config: ProtectiveAllocConfig = None):
        self.config = config or ProtectiveAllocConfig()
        self._prev_protection_score = 0.0
        self._dd_history: list = []

    def compute_protection(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        current_drawdown: float,
        regime: int,
        benchmark: str = None,
        portfolio_returns: pd.Series = None,
    ) -> Dict:
        """Compute the protective allocation adjustment.

        Args:
            prices: Historical close prices (time x coins)
            returns: Historical log returns (time x coins)
            current_drawdown: Current portfolio drawdown (0 to 1)
            regime: Current detected regime (0=Bull, 1=Neutral, 2=Bear)
            benchmark: Benchmark coin (default: BTC)
            portfolio_returns: Actual portfolio return series (optional;
                              if None, uses BTC returns as proxy)

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

        # Signal 1: BTC 24h Vol spike
        vol_score = self._volatility_score(prices, benchmark)
        components["btc_24h_vol"] = vol_score
        vol_fired = vol_score > 0.0

        # Signal 2: Portfolio 6h Return
        port_ret_score = self._portfolio_return_score(
            prices, returns, benchmark, portfolio_returns)
        components["portfolio_6h_ret"] = port_ret_score
        port_fired = port_ret_score > 0.0

        # Supplementary: Drawdown depth (always contributes, but scaled lower)
        dd_score = self._drawdown_score(current_drawdown)
        components["drawdown"] = dd_score

        # OR combo: take the MAX of the two primary signals
        # This is the core innovation — either signal firing triggers protection
        primary_score = max(vol_score, port_ret_score)

        # Blend with drawdown as a floor/amplifier:
        # - If DD is high but neither signal fires yet, DD provides mild protection
        # - If a signal fires AND DD is high, DD amplifies
        raw_score = max(primary_score, dd_score * 0.5) + dd_score * 0.3
        raw_score = min(raw_score, 1.0)

        # Regime adjustment: amplify in bear, dampen in bull
        if regime == 2:
            raw_score = min(1.0, raw_score * 1.15)  # 15% boost in bear
        elif regime == 0:
            raw_score = raw_score * 0.7  # 30% dampening in bull

        raw_score = np.clip(raw_score, 0.0, 1.0)

        # Circuit breaker: if BOTH signals fire OR raw score surges, skip smoothing
        both_fired = vol_fired and port_fired
        circuit_breaker = both_fired and self.config.circuit_breaker_both
        # Rapid deterioration: raw score jumped >0.3 above smoothed → skip smoothing
        rapid_threshold = getattr(self, '_rapid_deterioration_threshold', 0.30)
        rapid_deterioration = (raw_score - self._prev_protection_score) > rapid_threshold

        if circuit_breaker or rapid_deterioration:
            smoothed = max(raw_score, self._prev_protection_score)
            if circuit_breaker:
                logger.info("CIRCUIT BREAKER: Both signals fired — immediate protection %.3f",
                            smoothed)
            else:
                logger.info("RAPID DETERIORATION: raw=%.3f >> prev=%.3f — immediate protection %.3f",
                            raw_score, self._prev_protection_score, smoothed)
        else:
            # Asymmetric EMA smoothing: fast UP, regime-aware DOWN
            if raw_score > self._prev_protection_score:
                alpha = self.config.protection_smoothing_up
            else:
                # In BULL with no signals firing, clear protection faster
                # so we can re-invest capital into the confirmed trend
                if regime == 0 and not vol_fired and not port_fired:
                    alpha = 0.50  # faster decay in bull
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
            "vol=%.2f(%s) port_ret=%.2f(%s) dd=%.2f | combo=%s",
            smoothed, raw_score, cash_fraction * 100, invested_multiplier,
            vol_score, "FIRE" if vol_fired else "ok",
            port_ret_score, "FIRE" if port_fired else "ok",
            dd_score,
            "BOTH" if both_fired else ("OR" if (vol_fired or port_fired) else "NONE"),
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
        """
        start = self.config.dd_fast_scale_start  # 3%
        full = self.config.dd_fast_scale_full     # 12%
        if current_drawdown <= start:
            return 0.0
        if current_drawdown >= full:
            return 1.0
        mid_dd = (start + full) / 2  # ~7.5%
        if current_drawdown <= mid_dd:
            return 0.3 * (current_drawdown - start) / (mid_dd - start)
        else:
            return 0.3 + 0.7 * (current_drawdown - mid_dd) / (full - mid_dd)

    def _volatility_score(self, prices: pd.DataFrame, benchmark: str) -> float:
        """Score based on BTC 24h realized vol spike vs 7d baseline.
        0=normal, 1=crisis vol.

        This is Signal 1 of the best combo (BTC 24h Vol).
        Crisis/calm firing ratio: 3.7x — highest discriminative power.
        """
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

    def _portfolio_return_score(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        benchmark: str,
        portfolio_returns: pd.Series = None,
    ) -> float:
        """Score based on portfolio return (dual window: 6h + 3h fast).
        0=positive/neutral, 1=crisis drawdown.

        Uses the MAX of the main (6h) and fast (3h) window scores.
        The fast window catches sudden drops before the main window fires.
        """
        # Main window (6h)
        window = self.config.portfolio_ret_window
        score_main = self._compute_ret_score(
            prices, benchmark, portfolio_returns,
            window, self.config.portfolio_ret_threshold, self.config.portfolio_ret_crisis)

        # Fast window (3h) — catches sharp drops earlier
        fast_window = self.config.portfolio_ret_fast_window
        fast_threshold = self.config.portfolio_ret_fast_threshold
        score_fast = self._compute_ret_score(
            prices, benchmark, portfolio_returns,
            fast_window, fast_threshold, fast_threshold * 3)  # crisis = 3x threshold

        return max(score_main, score_fast)

    def _compute_ret_score(self, prices, benchmark, portfolio_returns,
                           window, threshold, crisis) -> float:
        """Helper: compute return score for a single window."""
        if portfolio_returns is not None and len(portfolio_returns) >= window:
            ret = float(portfolio_returns.iloc[-window:].sum())
        elif benchmark in prices.columns and len(prices) >= window + 1:
            btc = prices[benchmark]
            ret = float(btc.iloc[-1] / btc.iloc[-window - 1] - 1)
        else:
            return 0.0

        if ret >= threshold:
            return 0.0
        if ret <= crisis:
            return 1.0
        return (threshold - ret) / (threshold - crisis)
