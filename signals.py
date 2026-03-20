"""
Signal Engine -- Crypto Alpha Signals
======================================
6 alpha signals combined via weighted z-scores.
Trend-based regime detection (replaces HMM -- see test_hmm_regimes.py for why).

Signal stack (research-backed for crypto):
1. Momentum -- strongest factor in crypto. Multi-timeframe.
2. Breakout -- price vs N-period high/low range.
3. VolumeMomentum -- momentum confirmed by volume.
4. MeanReversion -- short-term oversold with RSI confirmation.
5. RelativeStrength -- performance vs BTC benchmark.
6. ResidualMomentum -- idiosyncratic trend after removing BTC beta.

Research-driven portfolio hygiene:
- Prefer liquid names over thin names.
- Require minimum live history before a coin becomes tradable.
- Reduce exposure to market-beta-only momentum by using residual momentum.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

import config as cfg
from regime_models import FeatureScoreRegimeDetector, UnsupervisedFeatureRegimeDetector

logger = logging.getLogger(__name__)


# -- Utility functions --

def zscore(series: pd.Series, clip: float = 3.0) -> pd.Series:
    """Cross-sectional z-score with clipping."""
    if series.std() == 0 or series.isna().all():
        return series * 0.0
    z = (series - series.mean()) / series.std()
    return z.clip(-clip, clip)


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Compute RSI for a single price series. Returns latest RSI value."""
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return val if np.isfinite(val) else 50.0


def compute_sma(prices: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return prices.rolling(period, min_periods=max(1, period // 2)).mean()


# -- Signal Classes --

class MomentumSignal:
    """Multi-timeframe momentum -- the strongest alpha factor in crypto.

    Combines three timeframes:
    - Fast (12h): captures intraday trends
    - Medium (72h): 3-day momentum
    - Slow (168h): 7-day momentum
    Skip last 2 hours to avoid microstructure noise.
    Weights: 20% fast, 30% medium, 50% slow (trend > noise).
    """

    @staticmethod
    def generate(prices: pd.DataFrame, fast: int = 12, medium: int = 72,
                 slow: int = 168, skip: int = 2) -> pd.Series:
        if len(prices) < slow + skip:
            return pd.Series(0.0, index=prices.columns)

        p = prices.iloc[:-skip] if skip > 0 else prices

        scores = pd.Series(0.0, index=prices.columns)

        # Fast momentum
        if len(p) >= fast:
            fast_ret = p.iloc[-1] / p.iloc[-fast] - 1
            scores += 0.20 * zscore(fast_ret)

        # Medium momentum
        if len(p) >= medium:
            med_ret = p.iloc[-1] / p.iloc[-medium] - 1
            scores += 0.30 * zscore(med_ret)

        # Slow momentum
        if len(p) >= slow:
            slow_ret = p.iloc[-1] / p.iloc[-slow] - 1
            scores += 0.50 * zscore(slow_ret)

        return zscore(scores)


class BreakoutSignal:
    """Breakout detection -- price relative to recent range.

    Coins near their N-period high get positive score (breakout).
    Coins near their N-period low get negative score (breakdown).
    Score = (price - low) / (high - low) mapped to [-1, 1].

    This is the second-strongest signal in crypto after momentum.
    """

    @staticmethod
    def generate(prices: pd.DataFrame, lookback: int = 72) -> pd.Series:
        if len(prices) < lookback:
            return pd.Series(0.0, index=prices.columns)

        recent = prices.iloc[-lookback:]
        current = prices.iloc[-1]
        high = recent.max()
        low = recent.min()
        range_width = high - low

        # Position within range: 0 = at low, 1 = at high
        position = (current - low) / range_width.replace(0, np.nan)
        # Map to [-1, 1]: at low = -1, at high = +1
        score = 2 * position - 1
        return zscore(score.fillna(0))


class VolumeMomentumSignal:
    """Volume-confirmed momentum.

    Momentum is more reliable when confirmed by above-average volume.
    Score = momentum * volume_ratio (volume / avg_volume).
    High momentum + high volume = strong signal.
    """

    @staticmethod
    def generate(prices: pd.DataFrame, volumes: pd.DataFrame,
                 mom_lookback: int = 48, vol_lookback: int = 168) -> pd.Series:
        if prices.empty or volumes is None or volumes.empty:
            return pd.Series(0.0, index=prices.columns)

        common = prices.columns.intersection(volumes.columns)
        if len(common) == 0:
            return pd.Series(0.0, index=prices.columns)

        if len(prices) < mom_lookback:
            return pd.Series(0.0, index=prices.columns)

        # Momentum component
        p = prices[common]
        momentum = p.iloc[-1] / p.iloc[-mom_lookback] - 1

        # Volume confirmation
        v = volumes[common]
        if len(v) >= vol_lookback:
            recent_vol = v.iloc[-24:].mean()  # last 24h average volume
            avg_vol = v.iloc[-vol_lookback:].mean()  # longer-term average
            vol_ratio = (recent_vol / avg_vol.replace(0, np.nan)).fillna(1.0)
            # Clip extreme volume ratios
            vol_ratio = vol_ratio.clip(0.5, 3.0)
        else:
            vol_ratio = pd.Series(1.0, index=common)

        # Combine: momentum * volume_confirmation
        score = momentum * vol_ratio
        result = zscore(score).reindex(prices.columns, fill_value=0)
        return result


class MeanReversionSignal:
    """Short-term mean reversion with RSI confirmation.

    Detects oversold conditions using:
    1. Z-score of recent returns (inverted)
    2. RSI confirmation (only trigger if RSI < 35)

    Without RSI gate, mean reversion catches falling knives.
    """

    @staticmethod
    def generate(prices: pd.DataFrame, lookback: int = 48) -> pd.Series:
        if len(prices) < lookback:
            return pd.Series(0.0, index=prices.columns)

        recent = prices.iloc[-lookback:]
        returns = np.log(recent / recent.shift(1)).dropna()

        if returns.empty:
            return pd.Series(0.0, index=prices.columns)

        # Current return deviation from mean (inverted for mean reversion)
        current_ret = returns.iloc[-1]
        mean_ret = returns.mean()
        std_ret = returns.std().replace(0, np.nan)
        deviation = -(current_ret - mean_ret) / std_ret

        # RSI gate: only allow mean reversion signal when RSI is extreme
        scores = {}
        for coin in prices.columns:
            rsi = compute_rsi(prices[coin], period=14)
            dev = deviation.get(coin, 0)
            if not np.isfinite(dev):
                dev = 0

            if rsi < 30:
                # Oversold + mean reversion = strong buy
                scores[coin] = abs(dev) * 1.5
            elif rsi > 70:
                # Overbought + mean reversion pointing down = sell
                scores[coin] = -abs(dev) * 0.5
            elif rsi < 40:
                # Mildly oversold
                scores[coin] = dev * 0.5
            else:
                # Neutral zone -- weak signal
                scores[coin] = dev * 0.2

        return zscore(pd.Series(scores))


class RelativeStrengthSignal:
    """Relative strength vs BTC benchmark.

    Coins outperforming BTC tend to continue outperforming (momentum effect).
    Multi-timeframe: 24h, 72h, 168h relative performance.
    Recency-weighted: more weight on recent outperformance.
    """

    @staticmethod
    def generate(prices: pd.DataFrame, benchmark: str = None) -> pd.Series:
        benchmark = benchmark or cfg.BENCHMARK

        if benchmark not in prices.columns:
            return pd.Series(0.0, index=prices.columns)

        scores = pd.Series(0.0, index=prices.columns)
        windows = [(24, 0.50), (72, 0.30), (168, 0.20)]

        for window, weight in windows:
            if len(prices) < window:
                continue

            coin_ret = prices.iloc[-1] / prices.iloc[-window] - 1
            btc_ret = coin_ret.get(benchmark, 0)

            # Relative performance vs BTC
            relative = coin_ret - btc_ret
            scores += weight * zscore(relative)

        # BTC gets neutral score (it's the benchmark)
        scores[benchmark] = 0.0
        return zscore(scores)


class ResidualMomentumSignal:
    """Residual momentum after removing market beta.

    Inspired by residual-momentum research: rank coins on the persistence of
    idiosyncratic returns, not raw market-beta-heavy returns. In crypto this is
    useful because broad beta swings can dominate short-horizon rankings.
    """

    @staticmethod
    def generate(prices: pd.DataFrame, benchmark: str = None,
                 lookback: int = None, skip: int = None) -> pd.Series:
        benchmark = benchmark or cfg.BENCHMARK
        lookback = lookback or cfg.SIGNALS.residual_momentum_lookback
        skip = cfg.SIGNALS.momentum_skip if skip is None else skip

        if benchmark not in prices.columns or len(prices) < lookback + skip:
            return pd.Series(0.0, index=prices.columns)

        p = prices.iloc[:-skip] if skip > 0 else prices
        rets = np.log(p / p.shift(1)).iloc[-lookback:]
        if rets.empty or benchmark not in rets.columns:
            return pd.Series(0.0, index=prices.columns)

        market = rets[benchmark].dropna()
        if len(market) < max(24, lookback // 4):
            return pd.Series(0.0, index=prices.columns)

        scores = {}
        for coin in prices.columns:
            if coin == benchmark:
                scores[coin] = 0.0
                continue

            coin_ret = rets[coin].dropna()
            common = coin_ret.index.intersection(market.index)
            if len(common) < max(24, lookback // 4):
                scores[coin] = 0.0
                continue

            y = coin_ret.loc[common]
            x = market.loc[common]
            x_var = x.var()
            if not np.isfinite(x_var) or x_var < 1e-12:
                beta = 0.0
            else:
                beta = y.cov(x) / x_var
                if not np.isfinite(beta):
                    beta = 0.0

            alpha_hat = y.mean() - beta * x.mean()
            residual = y - (alpha_hat + beta * x)
            resid_std = residual.std()
            if not np.isfinite(resid_std) or resid_std < 1e-8:
                scores[coin] = 0.0
                continue

            # T-stat-like ranking favors persistent residual trends over noisy spikes.
            scores[coin] = residual.sum() / (resid_std * np.sqrt(len(residual)))

        return zscore(pd.Series(scores).reindex(prices.columns, fill_value=0.0))


# -- Regime Detection --

class TrendRegimeDetector:
    """Enhanced trend-based regime detection on BTC + market breadth.

    Uses BTC price action plus cross-asset breadth signals for
    faster and more accurate regime identification.

    Signals:
    1. BTC price vs 7-day and 3-day SMA
    2. SMA slope (trend direction)
    3. Drawdown from recent high
    4. Short-term momentum (24h)
    5. Medium-term momentum (72h) — catches sustained moves
    6. Market breadth (fraction of coins above their own 3-day SMA)
    7. Realized volatility spike detection

    States:
    - BULL (0): Strong uptrend with positive breadth
    - NEUTRAL (1): Mixed signals
    - BEAR (2): Downtrend with negative breadth or drawdown
    """

    def __init__(self, rebalance_hours: int = None):
        self._current_regime = 1  # default: Neutral
        self._regime_history = []
        self._regime_hold_count = 0
        freq = rebalance_hours or cfg.REBALANCE.frequency_hours
        self._MIN_HOLD = max(2, 16 // freq)  # ~16h hysteresis (was 8h)
        # Re-risk cooldown: after leaving BEAR, require extra hold before BULL
        self._RERISK_COOLDOWN = max(1, 12 // freq)  # 12h cooldown (was 24h)
        self._bear_exit_count = 0  # cycles since last BEAR exit
        self._was_bear = False     # track if we were recently in BEAR

    def detect(
        self,
        prices: pd.DataFrame,
        benchmark: str = None,
        volumes: pd.DataFrame = None,
    ) -> int:
        benchmark = benchmark or cfg.BENCHMARK

        if benchmark not in prices.columns or len(prices) < 168:
            self._current_regime = 1
            return 1

        btc = prices[benchmark]

        # 1. Price vs SMA
        sma_168 = compute_sma(btc, 168)  # 7-day
        sma_72 = compute_sma(btc, 72)    # 3-day
        current_price = btc.iloc[-1]

        above_7d_sma = current_price > sma_168.iloc[-1]
        above_3d_sma = current_price > sma_72.iloc[-1]

        # 2. SMA slope (24h rate of change)
        sma_slope = (sma_168.iloc[-1] / sma_168.iloc[-24] - 1) if len(sma_168) >= 24 else 0

        # 3. Drawdown from 14-day high
        recent_high = btc.iloc[-336:].max() if len(btc) >= 336 else btc.max()
        drawdown = (recent_high - current_price) / recent_high if recent_high > 0 else 0

        # 4. Short-term momentum (24h)
        short_mom = (btc.iloc[-1] / btc.iloc[-24] - 1) if len(btc) >= 24 else 0

        # 5. Medium-term momentum (72h) — catches sustained declines
        med_mom = (btc.iloc[-1] / btc.iloc[-72] - 1) if len(btc) >= 72 else 0

        # 6. Market breadth — fraction of coins above their 3-day SMA
        breadth = 0.5
        if len(prices) >= 72 and len(prices.columns) >= 5:
            sma_72_all = prices.rolling(72, min_periods=36).mean()
            breadth = float((prices.iloc[-1] > sma_72_all.iloc[-1]).mean())

        # 7. Realized vol (24h) — spike detection
        btc_ret = np.log(btc / btc.shift(1))
        vol_24h = float(btc_ret.iloc[-24:].std()) if len(btc_ret) >= 24 else 0.01
        vol_168h = float(btc_ret.iloc[-168:].std()) if len(btc_ret) >= 168 else vol_24h
        vol_spike = vol_24h > 1.5 * vol_168h  # vol 50% above normal

        # Decision logic — weighted signal combination
        bull_signals = 0
        bear_signals = 0

        # SMA position (2 signals)
        if above_7d_sma:
            bull_signals += 1
        else:
            bear_signals += 1
        if above_3d_sma:
            bull_signals += 1
        else:
            bear_signals += 1

        # SMA slope
        if sma_slope > 0.001:
            bull_signals += 1
        elif sma_slope < -0.001:
            bear_signals += 1

        # Drawdown (weighted more heavily)
        if drawdown > 0.12:
            bear_signals += 2  # strong bear
        elif drawdown > 0.06:
            bear_signals += 1

        # Short-term momentum
        if short_mom > 0.02:
            bull_signals += 1
        elif short_mom < -0.03:
            bear_signals += 1

        # Medium-term momentum (72h) — NEW: catches sustained moves
        if med_mom > 0.05:
            bull_signals += 1
        elif med_mom < -0.05:
            bear_signals += 1
        elif med_mom < -0.03:
            bear_signals += 0.5

        # Market breadth — NEW: cross-sectional confirmation
        if breadth > 0.65:
            bull_signals += 1
        elif breadth < 0.35:
            bear_signals += 1
        elif breadth < 0.45:
            bear_signals += 0.5

        # Volatility spike — NEW: high vol → defensive tilt
        if vol_spike and drawdown > 0.03:
            bear_signals += 0.5

        # Classify raw signal — balanced thresholds (was asymmetric, caused bear bias)
        if bear_signals >= 3.5:   # need strong confirmation for bear (was 2.5)
            raw_regime = 2
        elif bull_signals >= 3.0:  # moderate proof for bull (was 3.5)
            raw_regime = 0
        else:
            raw_regime = 1

        # Re-risk gate: after being in BEAR, require stronger proof before BULL
        # Must have: BTC above 7d SMA, breadth > 0.60, positive 72h momentum
        if self._was_bear and self._bear_exit_count < self._RERISK_COOLDOWN:
            self._bear_exit_count += 1
            rerisk_ok = (above_7d_sma and breadth > 0.55 and med_mom > 0.01)
            if raw_regime == 0 and not rerisk_ok:
                raw_regime = 1  # demote to NEUTRAL until conditions met
                logger.info("Re-risk gate: BULL blocked (cooldown %d/%d, breadth=%.2f, med_mom=%.3f)",
                            self._bear_exit_count, self._RERISK_COOLDOWN, breadth, med_mom)

        # Hysteresis with crash-fast override
        self._regime_hold_count += 1
        if raw_regime != self._current_regime:
            # Crash override: immediate BEAR only on very strong signals
            crash_override = (raw_regime == 2 and bear_signals >= 4.5)
            if crash_override or self._regime_hold_count >= self._MIN_HOLD:
                prev = self._current_regime
                # Track BEAR exits for re-risk cooldown
                if prev == 2 and raw_regime != 2:
                    self._was_bear = True
                    self._bear_exit_count = 0
                if raw_regime == 2:
                    self._was_bear = False  # entering bear, reset
                # Step through NEUTRAL unless crash override
                if abs(raw_regime - prev) == 2 and not crash_override:
                    regime = 1
                else:
                    regime = raw_regime
                self._current_regime = regime
                self._regime_hold_count = 0
            else:
                regime = self._current_regime
        else:
            regime = self._current_regime
            # Clear re-risk cooldown once we've been in BULL for enough cycles
            if regime == 0 and self._was_bear:
                self._bear_exit_count += 1
                if self._bear_exit_count >= self._RERISK_COOLDOWN:
                    self._was_bear = False

        self._regime_history.append(regime)

        regime_names = {0: "BULL", 1: "NEUTRAL", 2: "BEAR"}
        logger.info("Regime: %s (raw=%s, bull=%.1f, bear=%.1f, slope=%.4f, DD=%.1f%%, breadth=%.2f)",
                     regime_names[regime], regime_names[raw_regime],
                     bull_signals, bear_signals,
                     sma_slope, drawdown * 100, breadth)

        return regime

    @property
    def current_regime(self) -> int:
        return self._current_regime

    def generate_tilt(self, returns: pd.DataFrame, regime: int,
                      benchmark: str = None) -> pd.Series:
        """Generate regime-based allocation tilt.

        Bull -> overweight high-momentum coins (risk-on)
        Bear -> underweight everything, favor low-vol (risk-off)
        Neutral -> no tilt
        """
        benchmark = benchmark or cfg.BENCHMARK
        scores = {}

        for coin in returns.columns:
            if regime == 0:  # Bull -> reward recent momentum
                if len(returns) >= 72:
                    mom = returns[coin].iloc[-72:].sum()
                    scores[coin] = mom
                else:
                    scores[coin] = 0
            elif regime == 2:  # Bear -> reward low vol (defensive)
                coin_vol = returns[coin].std()
                btc_vol = returns[benchmark].std() if benchmark in returns.columns else coin_vol
                inv_vol = btc_vol / coin_vol if coin_vol > 0 else 0.5
                scores[coin] = inv_vol
            else:  # Neutral -> no tilt
                scores[coin] = 0.0

        return zscore(pd.Series(scores))


# -- Adaptive Signal Weighter --

class AdaptiveSignalWeighter:
    """Track per-signal performance by regime and adapt weights.

    Uses Information Coefficient (IC) -- rank correlation between signal scores
    and subsequent returns -- to measure signal quality per regime.

    Improvements over naive IC tracking:
    1. IC history buffer for IC_IR computation (IC / std(IC)) -- stability matters
    2. Regime-aware priors from empirical crypto research
    3. Gradual adaptation with EMA blending
    4. Bounded adjustments to prevent overfitting

    Base weights (from config) act as a prior. Adaptive adjustment is bounded
    to +/- 60% of base weight to prevent overfitting.
    """

    # How much IC data can tilt weights beyond regime priors
    # 0.20 was too conservative — signals with negative IC stayed heavily weighted
    ADAPTATION_STRENGTH = 0.45

    # Minimum weight floor as fraction of base (allow signals to be nearly zeroed)
    MIN_WEIGHT_FLOOR = 0.10

    # Maximum weight ceiling as multiple of base
    MAX_WEIGHT_CEIL = 2.5

    # Minimum observations before adapting with IC data
    MIN_OBS = 3

    # IC history buffer size per regime (rolling window)
    IC_BUFFER_SIZE = 50

    # Softmax temperature: lower = more aggressive tilting, higher = more uniform
    SOFTMAX_TEMP = 4.0

    SIGNAL_NAMES = ["momentum", "breakout", "volume_momentum",
                    "mean_reversion", "relative_strength", "residual_momentum"]

    # Empirical regime priors: which signals tend to work in which regime
    # Based on crypto factor research (momentum in bull, mean-rev in bear, etc.)
    # Regime priors from empirical crypto research
    REGIME_PRIORS = {
        0: {  # Bull: momentum + breakout dominate
            "momentum": 0.35, "breakout": 0.20, "volume_momentum": 0.10,
            "mean_reversion": 0.05, "relative_strength": 0.10,
            "residual_momentum": 0.20,
        },
        1: {  # Neutral: balanced
            "momentum": 0.25, "breakout": 0.15, "volume_momentum": 0.10,
            "mean_reversion": 0.15, "relative_strength": 0.15,
            "residual_momentum": 0.20,
        },
        2: {  # Bear: mean-reversion + relative strength dominate
            "momentum": 0.10, "breakout": 0.05, "volume_momentum": 0.10,
            "mean_reversion": 0.25, "relative_strength": 0.20,
            "residual_momentum": 0.30,
        },
    }

    def __init__(self, base_weights: dict):
        self.base_weights = base_weights.copy()

        # Per-regime IC history buffer: {regime: {signal_name: [ic_values]}}
        self._ic_history = {
            r: {s: [] for s in self.SIGNAL_NAMES}
            for r in [0, 1, 2]
        }

        # Observation count per regime
        self._obs_count = {0: 0, 1: 0, 2: 0}

        # Current adaptive weights per regime (starts as regime priors)
        self._current_weights = {r: self.REGIME_PRIORS[r].copy() for r in [0, 1, 2]}

        # EMA of adaptive weights for smooth transitions
        self._ema_weights = {r: base_weights.copy() for r in [0, 1, 2]}

    def _compute_ic(self, scores: pd.Series, returns: pd.Series) -> float:
        """Compute rank IC (Spearman correlation) between signal and returns."""
        common = scores.index.intersection(returns.index)
        if len(common) < 3:
            return 0.0
        try:
            s_vals = scores.reindex(common).values
            r_vals = returns.reindex(common).values
            if np.std(s_vals) < 1e-10 or np.std(r_vals) < 1e-10:
                return 0.0
            ic, _ = stats.spearmanr(s_vals, r_vals)
            return ic if np.isfinite(ic) else 0.0
        except Exception:
            return 0.0

    def update(self, signal_scores: Dict[str, pd.Series],
               forward_returns: pd.Series, regime: int):
        """Update IC tracking with new observation.

        Call this AFTER computing signals but BEFORE using weights,
        passing the PREVIOUS cycle's signals and the returns since then.
        """
        if forward_returns.empty or not signal_scores:
            return

        for name, scores in signal_scores.items():
            if name not in self.SIGNAL_NAMES:
                continue

            ic = self._compute_ic(scores, forward_returns)

            # Append to history buffer (bounded size)
            buf = self._ic_history[regime][name]
            buf.append(ic)
            if len(buf) > self.IC_BUFFER_SIZE:
                buf.pop(0)

        self._obs_count[regime] += 1

    def _compute_ic_ir(self, regime: int) -> Dict[str, float]:
        """Compute IC_IR (Information Ratio of IC) per signal for a regime.

        IC_IR = mean(IC) / std(IC) -- rewards signals that are both
        predictive AND consistent. A signal with IC=0.05 and std=0.02
        (IC_IR=2.5) is better than one with IC=0.10 and std=0.15 (IC_IR=0.67).
        """
        ic_ir = {}
        for name in self.SIGNAL_NAMES:
            buf = self._ic_history[regime][name]
            if len(buf) < 3:
                ic_ir[name] = 0.0
                continue
            arr = np.array(buf)
            mean_ic = np.mean(arr)
            std_ic = np.std(arr)
            if std_ic < 1e-6:
                # Constant IC -- use mean directly (could be good or bad)
                ic_ir[name] = mean_ic * 5.0  # scale to comparable range
            else:
                ic_ir[name] = mean_ic / std_ic
        return ic_ir

    def get_weights(self, regime: int) -> Dict[str, float]:
        """Get adaptive weights for the current regime.

        Strategy: ALWAYS use regime priors as the primary anchor.
        IC data only fine-tunes within a narrow band around priors.
        This is because with only ~10 coins, IC estimates are very noisy,
        but regime priors (domain knowledge) are stable.
        """
        priors = self.REGIME_PRIORS.get(regime, self.base_weights)
        obs = self._obs_count.get(regime, 0)

        # Phase 1: Before MIN_OBS, blend base -> regime priors (ramp up prior)
        if obs < self.MIN_OBS:
            # Gradually shift from base weights to regime priors
            prior_alpha = min(obs / max(self.MIN_OBS, 1), 0.8)
            weights = {}
            for name in self.SIGNAL_NAMES:
                base_w = self.base_weights.get(name, 0.15)
                prior_w = priors.get(name, 0.15)
                weights[name] = (1 - prior_alpha) * base_w + prior_alpha * prior_w
            return weights

        # Phase 2: Enough IC data -- use regime prior + small IC_IR tilt
        ic_ir = self._compute_ic_ir(regime)

        # Softmax over IC_IR to get proportional weights
        ir_values = np.array([ic_ir[s] for s in self.SIGNAL_NAMES])

        # Only apply IC tilt if there's meaningful signal differentiation
        ir_range = np.max(ir_values) - np.min(ir_values)
        if ir_range < 0.05:
            # IC_IR values are too similar -- just use regime priors
            self._current_weights[regime] = priors.copy()
            return priors.copy()

        # Softmax with temperature
        ir_shifted = ir_values - np.max(ir_values)
        exp_ir = np.exp(ir_shifted / max(self.SOFTMAX_TEMP, 0.1))
        ic_proportions = exp_ir / exp_ir.sum()

        # Build weights: regime prior (dominant) + IC tilt
        # Bear regime gets stronger adaptation (cut bad signals faster)
        if regime == 2:
            adapt_strength = min((obs - self.MIN_OBS) / 15.0, 0.55)
            floor_mult = 0.05   # allow near-zero for bad signals in bear
        else:
            adapt_strength = min((obs - self.MIN_OBS) / 20.0, self.ADAPTATION_STRENGTH)
            floor_mult = self.MIN_WEIGHT_FLOOR
        data_confidence = adapt_strength
        raw_weights = {}
        for i, name in enumerate(self.SIGNAL_NAMES):
            base_w = self.base_weights.get(name, 0.15)
            prior_w = priors.get(name, 0.15)
            ic_w = ic_proportions[i]

            blended = (1 - data_confidence) * prior_w + data_confidence * ic_w

            # Floor and ceiling relative to base
            blended = max(blended, floor_mult * base_w)
            blended = min(blended, self.MAX_WEIGHT_CEIL * base_w)

            raw_weights[name] = blended

        # Normalize
        base_total = sum(self.base_weights.get(s, 0) for s in self.SIGNAL_NAMES)
        raw_total = sum(raw_weights.values())
        if raw_total > 0:
            for name in raw_weights:
                raw_weights[name] *= base_total / raw_total

        # EMA smooth (prevent jumps between cycles)
        ema_alpha = 0.3
        prev_ema = self._ema_weights[regime]
        for name in self.SIGNAL_NAMES:
            prev_ema[name] = (1 - ema_alpha) * prev_ema.get(name, raw_weights[name]) + ema_alpha * raw_weights[name]

        self._current_weights[regime] = prev_ema.copy()
        return prev_ema.copy()

    def get_diagnostics(self) -> Dict:
        """Get diagnostic info about adaptive weights."""
        ic_ir_all = {}
        for regime in [0, 1, 2]:
            if self._obs_count[regime] >= 3:
                ic_ir_all[regime] = {
                    s: round(v, 4) for s, v in self._compute_ic_ir(regime).items()
                }
            else:
                ic_ir_all[regime] = {s: 0.0 for s in self.SIGNAL_NAMES}

        mean_ic = {}
        for regime in [0, 1, 2]:
            mean_ic[regime] = {}
            for s in self.SIGNAL_NAMES:
                buf = self._ic_history[regime][s]
                mean_ic[regime][s] = round(np.mean(buf), 4) if buf else 0.0

        return {
            "ic_ir": ic_ir_all,
            "mean_ic": mean_ic,
            "obs_count": self._obs_count.copy(),
            "current_weights": {
                regime: {k: round(v, 4) for k, v in w.items()}
                for regime, w in self._current_weights.items()
            },
        }


# -- Alpha Engine --

class AlphaEngine:
    """Combine all signals into a single alpha score per coin.

    Signal weights (research-backed for crypto):
    - Momentum:         35% (strongest factor)
    - Breakout:         20% (trend continuation)
    - VolumeMomentum:   15% (confirmation)
    - MeanReversion:    15% (contrarian with RSI gate)
    - RelativeStrength: 15% (vs BTC)

    Adaptive weighting: tracks per-signal IC (information coefficient) by
    regime and tilts weights toward better-performing signals.

    Trend filter: zero alpha if price < 7-day SMA (168 hours).
    Regime tilt: applied as overlay after signal combination.
    """

    def __init__(self, adaptive: bool = True, rebalance_hours: int = None,
                 regime_detector=None):
        self.regime_detector = regime_detector or self._build_regime_detector(rebalance_hours)
        self.base_weights = cfg.SIGNALS.factor_weights
        self.adaptive = adaptive
        self.signal_weighter = AdaptiveSignalWeighter(self.base_weights)
        self.rebalance_hours = rebalance_hours or cfg.REBALANCE.frequency_hours

        # Store previous cycle's signals for IC computation
        self._prev_signal_scores: Optional[Dict[str, pd.Series]] = None
        self._prev_regime: Optional[int] = None

    @staticmethod
    def _build_regime_detector(rebalance_hours: int = None):
        model = str(getattr(cfg.SIGNALS, "regime_model", "legacy_trend")).lower()
        feature_set = getattr(cfg.SIGNALS, "regime_feature_set", "market")
        n_components = getattr(cfg.SIGNALS, "regime_n_components", 4)
        train_hours = getattr(cfg.SIGNALS, "regime_train_hours", 24 * 90)
        retrain_hours = getattr(cfg.SIGNALS, "regime_retrain_hours", 24)

        if model == "feature_hmm":
            return UnsupervisedFeatureRegimeDetector(
                model_kind="hmm",
                feature_set=feature_set,
                n_components=n_components,
                rebalance_hours=rebalance_hours,
                train_hours=train_hours,
                retrain_hours=retrain_hours,
            )
        if model == "feature_gmm":
            return UnsupervisedFeatureRegimeDetector(
                model_kind="gmm",
                feature_set=feature_set,
                n_components=n_components,
                rebalance_hours=rebalance_hours,
                train_hours=train_hours,
                retrain_hours=retrain_hours,
            )
        if model == "score":
            return FeatureScoreRegimeDetector(
                feature_set=feature_set,
                rebalance_hours=rebalance_hours,
                train_hours=train_hours,
            )
        return TrendRegimeDetector(rebalance_hours=rebalance_hours)

    def _liquidity_mask(self, prices: pd.DataFrame,
                        volumes: Optional[pd.DataFrame]) -> pd.Series:
        """Keep only sufficiently liquid coins eligible for long exposure."""
        liquid = pd.Series(True, index=prices.columns, dtype=bool)
        if volumes is None or volumes.empty or prices.empty:
            return liquid

        common = prices.columns.intersection(volumes.columns)
        if len(common) == 0:
            return liquid

        lookback = min(cfg.SIGNALS.liquidity_lookback_hours, len(prices), len(volumes))
        if lookback < 4:
            return liquid

        recent_prices = prices[common].iloc[-lookback:]
        recent_volumes = volumes[common].iloc[-lookback:]
        dollar_volume = (recent_prices * recent_volumes).mean()
        dollar_volume = dollar_volume.replace([np.inf, -np.inf], np.nan).dropna()
        if dollar_volume.empty:
            return liquid

        cutoff = dollar_volume.quantile(cfg.SIGNALS.liquidity_filter_quantile)
        keep = dollar_volume[dollar_volume >= cutoff].index
        liquid.loc[common] = False
        liquid.loc[keep] = True
        return liquid

    def _history_mask(self, prices: pd.DataFrame) -> pd.Series:
        """Require a minimum number of real observations before trading a coin."""
        history = prices.notna().sum()
        return history >= cfg.SIGNALS.min_history_hours

    def compute_alpha(self, prices: pd.DataFrame, returns: pd.DataFrame,
                      volumes: pd.DataFrame = None) -> pd.Series:
        """Compute composite alpha scores for all coins.

        Args:
            prices: Close price DataFrame (time x coins)
            returns: Log return DataFrame (time x coins)
            volumes: Volume DataFrame (optional, improves volume-momentum signal)

        Returns:
            pd.Series of alpha scores indexed by coin, z-scored
        """
        coins = prices.columns.tolist()

        # Detect regime first (needed for adaptive weights)
        try:
            regime = self.regime_detector.detect(prices, volumes=volumes)
        except Exception as e:
            logger.warning("Regime detection failed: %s", e)
            regime = 1

        # Update adaptive weights with previous cycle's signal performance
        if self.adaptive and self._prev_signal_scores is not None and len(returns) >= self.rebalance_hours:
            # Forward returns = cumulative return over the actual holding period
            # (rebalance_hours rows of hourly returns, not just the last 1h)
            fwd_ret = returns.iloc[-self.rebalance_hours:].sum()
            self.signal_weighter.update(
                self._prev_signal_scores, fwd_ret,
                self._prev_regime if self._prev_regime is not None else 1
            )

        # Get weights: always use regime-aware priors, optionally with IC adaptation
        if self.adaptive:
            weights = self.signal_weighter.get_weights(regime)
        else:
            # Use regime priors even in static mode (better than flat base weights)
            weights = AdaptiveSignalWeighter.REGIME_PRIORS.get(regime, self.base_weights)

        # Compute individual signals
        signal_scores = {}
        alpha = pd.Series(0.0, index=coins)

        # 1. Momentum signal
        try:
            mom = MomentumSignal.generate(prices)
            signal_scores["momentum"] = mom
            alpha += weights.get("momentum", 0.35) * mom.reindex(coins, fill_value=0)
        except Exception as e:
            logger.warning("Momentum signal failed: %s", e)

        # 2. Breakout signal
        try:
            brk = BreakoutSignal.generate(prices)
            signal_scores["breakout"] = brk
            alpha += weights.get("breakout", 0.20) * brk.reindex(coins, fill_value=0)
        except Exception as e:
            logger.warning("Breakout signal failed: %s", e)

        # 3. Volume-Momentum signal
        try:
            vmom = VolumeMomentumSignal.generate(prices, volumes)
            signal_scores["volume_momentum"] = vmom
            alpha += weights.get("volume_momentum", 0.15) * vmom.reindex(coins, fill_value=0)
        except Exception as e:
            logger.warning("Volume-momentum signal failed: %s", e)

        # 4. Mean reversion signal
        try:
            mr = MeanReversionSignal.generate(prices)
            signal_scores["mean_reversion"] = mr
            alpha += weights.get("mean_reversion", 0.15) * mr.reindex(coins, fill_value=0)
        except Exception as e:
            logger.warning("Mean reversion signal failed: %s", e)

        # 5. Relative strength signal
        try:
            rs = RelativeStrengthSignal.generate(prices)
            signal_scores["relative_strength"] = rs
            alpha += weights.get("relative_strength", 0.15) * rs.reindex(coins, fill_value=0)
        except Exception as e:
            logger.warning("Relative strength signal failed: %s", e)

        # 6. Residual momentum signal
        try:
            rmom = ResidualMomentumSignal.generate(prices)
            signal_scores["residual_momentum"] = rmom
            alpha += weights.get("residual_momentum", 0.15) * rmom.reindex(coins, fill_value=0)
        except Exception as e:
            logger.warning("Residual momentum signal failed: %s", e)

        # Store for next cycle's IC computation
        self._prev_signal_scores = signal_scores
        self._prev_regime = regime

        # 6. Apply regime tilt as overlay
        try:
            tilt = self.regime_detector.generate_tilt(returns, regime)
            alpha += 0.15 * tilt.reindex(coins, fill_value=0)
        except Exception as e:
            logger.warning("Regime tilt failed: %s", e)

        # Apply trend filter: dampen alpha for coins below 7-day SMA
        if cfg.SIGNALS.trend_filter and len(prices) >= cfg.SIGNALS.trend_filter_ma:
            sma = compute_sma(prices, cfg.SIGNALS.trend_filter_ma)
            below_trend = prices.iloc[-1] < sma.iloc[-1]
            if regime == 2:  # Bear: strong dampening
                alpha[below_trend] = -1.0
            else:
                alpha[below_trend] *= 0.1

        # Bear quality filter: in BEAR only, softly penalize meme/speculative.
        # Don't hard-cap — just apply a multiplier so they rank lower but
        # can still be held if alpha is strong enough.
        if regime == 2:
            MEME_SECTOR = {"Meme"}

            for coin in coins:
                sector = cfg.SECTOR_MAP.get(coin, "Other")
                if sector in MEME_SECTOR:
                    alpha[coin] = alpha.get(coin, 0) * 0.5  # halve meme alpha in bear
                elif sector in {"DeSci", "Identity", "Privacy"}:
                    alpha[coin] = alpha.get(coin, 0) * 0.7  # reduce speculative alpha

        # Final z-score
        alpha = zscore(alpha)

        # History and liquidity gating: illiquid or too-new names should not rank
        # above established, liquid coins in a long-only portfolio.
        history_ok = self._history_mask(prices)
        liquid_ok = self._liquidity_mask(prices, volumes)
        not_eligible = ~(history_ok & liquid_ok)
        if not_eligible.any():
            alpha.loc[not_eligible] = alpha.loc[not_eligible].clip(upper=0.0)

        return alpha

    @staticmethod
    def _get_breadth(prices: pd.DataFrame, period: int = 72) -> float:
        """Get market breadth (fraction of coins above their SMA)."""
        if len(prices) < period or len(prices.columns) < 5:
            return 0.5
        sma = prices.rolling(period, min_periods=period // 2).mean()
        return float((prices.iloc[-1] > sma.iloc[-1]).mean())

    @property
    def current_regime(self) -> int:
        return self.regime_detector.current_regime

    @property
    def regime_name(self) -> str:
        names = {0: "BULL", 1: "NEUTRAL", 2: "BEAR"}
        return names.get(self.current_regime, "UNKNOWN")

    @property
    def adaptive_diagnostics(self) -> Dict:
        """Get adaptive weight diagnostics."""
        return self.signal_weighter.get_diagnostics()
