"""
Regime detection models for crypto market-state classification.

This module keeps the production detector lightweight enough for live use while
also exposing alternative model families for offline research.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import config as cfg

logger = logging.getLogger(__name__)


def _safe_zscore(series: pd.Series, clip: float = 3.0) -> pd.Series:
    """Robust z-score helper used by research detectors."""
    if series.empty:
        return series
    std = series.std()
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    z = (series - series.mean()) / std
    return z.clip(-clip, clip)


def _latest_scalar(series: pd.Series, default: float = 0.0) -> float:
    """Return the latest finite value of a series or a default."""
    if series.empty:
        return default
    value = series.iloc[-1]
    return float(value) if np.isfinite(value) else default


def build_regime_feature_table(
    prices: pd.DataFrame,
    volumes: Optional[pd.DataFrame] = None,
    benchmark: str = "BTC",
) -> pd.DataFrame:
    """Build a market-state feature table from a price panel.

    The features follow the recurring structure found in the literature:
    multi-horizon trend, drawdown, realized volatility, breadth, dispersion,
    correlation stress, and optionally volume shocks.
    """
    if prices.empty or benchmark not in prices.columns:
        return pd.DataFrame()

    prices = prices.sort_index().ffill()
    returns = np.log(prices / prices.shift(1))
    btc = prices[benchmark]
    btc_ret = returns[benchmark]

    feat = pd.DataFrame(index=prices.index)
    feat["ret_1h"] = btc_ret
    feat["ret_6h"] = np.log(btc / btc.shift(6))
    feat["ret_24h"] = np.log(btc / btc.shift(24))
    feat["ret_72h"] = np.log(btc / btc.shift(72))
    feat["ret_168h"] = np.log(btc / btc.shift(168))

    sma_24 = btc.rolling(24).mean()
    sma_72 = btc.rolling(72).mean()
    sma_168 = btc.rolling(168).mean()
    feat["ma_gap_24"] = np.log(btc / sma_24)
    feat["ma_gap_72"] = np.log(btc / sma_72)
    feat["ma_gap_168"] = np.log(btc / sma_168)

    feat["dd_72h"] = btc / btc.rolling(72).max() - 1.0
    feat["dd_168h"] = btc / btc.rolling(168).max() - 1.0

    feat["vol_24h"] = btc_ret.rolling(24).std()
    feat["vol_72h"] = btc_ret.rolling(72).std()
    feat["vol_168h"] = btc_ret.rolling(168).std()
    feat["downside_72h"] = btc_ret.where(btc_ret < 0, 0.0).rolling(72).std()

    range_72 = (btc.rolling(72).max() - btc.rolling(72).min()).replace(0, np.nan)
    range_168 = (btc.rolling(168).max() - btc.rolling(168).min()).replace(0, np.nan)
    feat["range_pos_72h"] = (btc - btc.rolling(72).min()) / range_72
    feat["range_pos_168h"] = (btc - btc.rolling(168).min()) / range_168

    mom_24 = prices / prices.shift(24) - 1.0
    mom_72 = prices / prices.shift(72) - 1.0
    feat["breadth_pos_24"] = (mom_24 > 0).mean(axis=1)
    feat["breadth_pos_72"] = (mom_72 > 0).mean(axis=1)
    feat["breadth_above_72"] = (prices > prices.rolling(72).mean()).mean(axis=1)
    feat["cs_mom_24"] = mom_24.median(axis=1)
    feat["cs_mom_72"] = mom_72.median(axis=1)
    feat["dispersion_24"] = returns.rolling(24).std().median(axis=1)

    # Correlation computation: use subset of liquid coins for speed
    # (rolling corr across 50 coins is the main bottleneck)
    _CORR_COINS = [c for c in ["ETH", "SOL", "BNB", "XRP", "DOGE", "ADA",
                                "AVAX", "LINK", "DOT", "SUI"] if c in returns.columns]
    if _CORR_COINS:
        corr_subset = returns[_CORR_COINS]
        corr_to_btc = corr_subset.rolling(24, min_periods=12).corr(returns[benchmark])
        feat["avg_corr_24"] = corr_to_btc.mean(axis=1)
    else:
        feat["avg_corr_24"] = 0.5
    feat["down_frac_24"] = (returns < -0.02).rolling(24).mean().mean(axis=1)

    if volumes is not None and not volumes.empty and benchmark in volumes.columns:
        volumes = volumes.sort_index().reindex(prices.index).ffill().fillna(0)
        vol_fast = volumes[benchmark].rolling(24).mean()
        vol_slow = volumes[benchmark].rolling(168).mean().replace(0, np.nan)
        feat["volume_ratio"] = vol_fast / vol_slow
    else:
        feat["volume_ratio"] = 1.0

    return feat.replace([np.inf, -np.inf], np.nan).dropna()


@dataclass(frozen=True)
class FeatureSet:
    columns: Iterable[str]


FEATURE_SETS: Dict[str, FeatureSet] = {
    "return_only": FeatureSet(["ret_1h"]),
    "trend_vol": FeatureSet([
        "ret_6h", "ret_24h", "ret_72h", "ret_168h",
        "ma_gap_72", "ma_gap_168",
        "dd_72h", "dd_168h",
        "vol_24h", "vol_72h", "downside_72h",
        "range_pos_168h",
    ]),
    "market": FeatureSet([
        "ret_6h", "ret_24h", "ret_72h", "ret_168h",
        "ma_gap_72", "ma_gap_168",
        "dd_72h", "dd_168h",
        "vol_24h", "vol_72h", "downside_72h",
        "range_pos_168h",
        "breadth_pos_24", "breadth_pos_72", "breadth_above_72",
        "cs_mom_24", "cs_mom_72",
        "dispersion_24", "avg_corr_24", "down_frac_24",
        "volume_ratio",
    ]),
}


class BaseRegimeDetector:
    """Common interface expected by the signal engine."""

    def __init__(self, rebalance_hours: Optional[int] = None):
        self._current_regime = 1
        self.rebalance_hours = rebalance_hours or cfg.REBALANCE.frequency_hours
        self._feature_cache = pd.DataFrame()
        self._feature_cache_len = 0
        self._feature_cache_last_ts: Optional[pd.Timestamp] = None
        self._feature_cache_columns: tuple = ()

    def _get_feature_table(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
        benchmark: str = "BTC",
        overlap_rows: int = 200,
    ) -> pd.DataFrame:
        """Incrementally update the rolling feature table.

        The detector is called repeatedly on expanding windows during backtests.
        Recomputing all rolling features every time dominates runtime, so this
        method only rebuilds the trailing segment that can affect new values.
        """
        if prices.empty:
            return pd.DataFrame()

        current_len = len(prices)
        current_last_ts = prices.index[-1]
        current_columns = tuple(prices.columns)

        reset_cache = (
            self._feature_cache.empty
            or current_columns != self._feature_cache_columns
            or current_len < self._feature_cache_len
            or self._feature_cache_last_ts is None
            or current_last_ts <= self._feature_cache_last_ts
        )

        if reset_cache:
            feature_table = build_regime_feature_table(prices=prices, volumes=volumes, benchmark=benchmark)
        elif current_len == self._feature_cache_len and current_last_ts == self._feature_cache_last_ts:
            feature_table = self._feature_cache
        else:
            start_idx = max(0, self._feature_cache_len - overlap_rows)
            tail_prices = prices.iloc[start_idx:]
            tail_volumes = volumes.iloc[start_idx:] if volumes is not None and not volumes.empty else None
            tail_features = build_regime_feature_table(prices=tail_prices, volumes=tail_volumes, benchmark=benchmark)
            if tail_features.empty:
                feature_table = self._feature_cache
            else:
                prefix = self._feature_cache.loc[self._feature_cache.index < tail_features.index[0]]
                feature_table = pd.concat([prefix, tail_features]).sort_index()
                feature_table = feature_table[~feature_table.index.duplicated(keep="last")]

        self._feature_cache = feature_table
        self._feature_cache_len = current_len
        self._feature_cache_last_ts = current_last_ts
        self._feature_cache_columns = current_columns
        return feature_table

    @property
    def current_regime(self) -> int:
        return self._current_regime

    def generate_tilt(
        self,
        returns: pd.DataFrame,
        regime: int,
        benchmark: Optional[str] = None,
    ) -> pd.Series:
        """Generate a simple regime-dependent allocation tilt."""
        benchmark = benchmark or cfg.BENCHMARK
        scores = {}
        for coin in returns.columns:
            if regime == 0:
                scores[coin] = returns[coin].iloc[-72:].sum() if len(returns) >= 72 else 0.0
            elif regime == 2:
                coin_vol = returns[coin].std()
                bench_vol = returns[benchmark].std() if benchmark in returns.columns else coin_vol
                scores[coin] = (bench_vol / coin_vol) if coin_vol > 0 else 0.5
            else:
                scores[coin] = 0.0
        return _safe_zscore(pd.Series(scores))


class UnsupervisedFeatureRegimeDetector(BaseRegimeDetector):
    """Latent-state regime detector using market features.

    The model is intentionally simple:
    - price-panel features are computed from hourly OHLCV
    - the detector is retrained on a rolling window
    - four latent states are allowed so "crash/recovery" can stay separate
      from persistent downtrend
    - raw states are collapsed to the three macro regimes used elsewhere
    """

    def __init__(
        self,
        model_kind: str = "hmm",
        feature_set: str = "market",
        n_components: int = 4,
        rebalance_hours: Optional[int] = None,
        train_hours: int = 24 * 90,
        retrain_hours: int = 24,
        min_hold_hours: int = 12,
        random_state: int = 42,
    ):
        super().__init__(rebalance_hours=rebalance_hours)
        self.model_kind = model_kind
        self.feature_set = feature_set
        self.n_components = n_components
        self.train_hours = train_hours
        self.retrain_hours = retrain_hours
        self.min_hold_cycles = max(1, min_hold_hours // self.rebalance_hours)
        self.random_state = random_state

        self._model = None
        self._scaler: Optional[StandardScaler] = None
        self._state_map: Dict[int, int] = {}
        self._last_fit_ts: Optional[pd.Timestamp] = None
        self._hold_count = 0
        self._raw_state: Optional[int] = None
        self._feature_snapshot: Dict[str, float] = {}

    def _select_features(self, feature_table: pd.DataFrame) -> pd.DataFrame:
        selected = FEATURE_SETS.get(self.feature_set, FEATURE_SETS["market"]).columns
        cols = [c for c in selected if c in feature_table.columns]
        return feature_table[cols].dropna()

    def _fit_model(self, values: np.ndarray):
        if self.model_kind == "gmm":
            model = GaussianMixture(
                n_components=self.n_components,
                covariance_type="full",
                reg_covar=1e-5,
                random_state=self.random_state,
            )
            model.fit(values)
            return model

        model = GaussianHMM(
            n_components=self.n_components,
            covariance_type="diag",
            n_iter=200,
            random_state=self.random_state,
        )
        model.fit(values)
        return model

    def _predict_raw_state(self, values: np.ndarray) -> np.ndarray:
        if self.model_kind == "gmm":
            return self._model.predict(values)
        return self._model.predict(values)

    def _map_states(self, feature_table: pd.DataFrame, raw_states: np.ndarray) -> Dict[int, int]:
        train = feature_table.copy()
        train["raw_state"] = raw_states
        means = train.groupby("raw_state").mean(numeric_only=True)

        long_trend_cols = [
            "ret_24h", "ret_72h", "ret_168h",
            "ma_gap_72", "ma_gap_168",
            "breadth_pos_72", "breadth_above_72",
            "cs_mom_72",
        ]
        long_trend_cols = [c for c in long_trend_cols if c in means.columns]
        recovery_cols = [c for c in ["ret_6h", "ret_24h", "breadth_pos_24", "range_pos_168h"] if c in means.columns]
        risk_cols = [c for c in ["vol_24h", "vol_72h", "downside_72h", "avg_corr_24", "down_frac_24"] if c in means.columns]
        drawdown_cols = [c for c in ["dd_72h", "dd_168h"] if c in means.columns]

        long_trend = means[long_trend_cols].mean(axis=1).fillna(0.0) if long_trend_cols else pd.Series(0.0, index=means.index)
        recovery_signal = means[recovery_cols].mean(axis=1).fillna(0.0) if recovery_cols else pd.Series(0.0, index=means.index)
        risk_signal = means[risk_cols].mean(axis=1).fillna(0.0) if risk_cols else pd.Series(0.0, index=means.index)
        drawdown_signal = (-means[drawdown_cols]).mean(axis=1).fillna(0.0) if drawdown_cols else pd.Series(0.0, index=means.index)

        bull_signal = long_trend + 0.25 * recovery_signal - 0.35 * risk_signal
        bear_signal = (-1.5 * long_trend) + 0.45 * risk_signal + 0.60 * drawdown_signal - 0.55 * recovery_signal

        bull_state = int(bull_signal.idxmax())
        bear_candidates = bear_signal.sort_values(ascending=False).index.tolist()
        bear_state = next((int(s) for s in bear_candidates if int(s) != bull_state), bull_state)

        mapping = {int(state): 1 for state in means.index}

        bull_ok = (
            means.loc[bull_state].get("ret_72h", 0.0) > 0.0
            and means.loc[bull_state].get("ma_gap_168", 0.0) > 0.0
            and means.loc[bull_state].get("breadth_above_72", 0.5) > 0.52
        )
        bear_ok = (
            means.loc[bear_state].get("ret_72h", 0.0) < 0.0
            and means.loc[bear_state].get("ma_gap_168", 0.0) < 0.0
            and means.loc[bear_state].get("dd_168h", 0.0) < -0.05
        )

        if bull_ok:
            mapping[bull_state] = 0
        if bear_ok:
            mapping[bear_state] = 2

        # If the window has no clear bull state, keep ambiguous positive/recovery
        # states neutral rather than forcing a risk-on label.
        if not bull_ok and not bear_ok:
            mapping[bear_state] = 2
        return mapping

    def _fit_if_needed(self, selected_features: pd.DataFrame) -> None:
        if selected_features.empty:
            return

        latest_ts = selected_features.index[-1]
        refit_needed = (
            self._model is None
            or self._last_fit_ts is None
            or (latest_ts - self._last_fit_ts).total_seconds() >= self.retrain_hours * 3600
        )
        if not refit_needed:
            return

        train = selected_features.iloc[-self.train_hours:].copy()
        if len(train) < max(500, self.n_components * 100):
            return

        scaler = StandardScaler()
        values = scaler.fit_transform(train.values)
        try:
            model = self._fit_model(values)
            self._model = model
            raw_states = self._predict_raw_state(values)
        except Exception as exc:
            logger.warning("Regime model fit failed (%s/%s): %s", self.model_kind, self.feature_set, exc)
            return

        self._scaler = scaler
        self._state_map = self._map_states(train, raw_states)
        self._last_fit_ts = latest_ts

    def _is_crash_row(self, row: pd.Series) -> bool:
        return (
            row.get("ret_72h", 0.0) < -0.03
            and row.get("dd_72h", 0.0) < -0.05
            and row.get("breadth_pos_24", 0.5) < 0.45
        )

    def detect(
        self,
        prices: pd.DataFrame,
        benchmark: Optional[str] = None,
        volumes: Optional[pd.DataFrame] = None,
    ) -> int:
        feature_table = self._get_feature_table(
            prices=prices,
            volumes=volumes,
            benchmark=benchmark or cfg.BENCHMARK,
        )
        selected = self._select_features(feature_table)
        if selected.empty:
            self._current_regime = 1
            return 1

        self._fit_if_needed(selected)
        if self._model is None or self._scaler is None:
            self._current_regime = 1
            return 1

        latest_row = selected.iloc[[-1]]
        try:
            raw_state = int(self._predict_raw_state(self._scaler.transform(latest_row.values))[0])
        except Exception as exc:
            logger.warning("Regime model predict failed (%s/%s): %s", self.model_kind, self.feature_set, exc)
            self._current_regime = 1
            return 1

        proposed_regime = int(self._state_map.get(raw_state, 1))

        self._hold_count += 1
        allow_switch = proposed_regime == self._current_regime or self._hold_count >= self.min_hold_cycles
        if proposed_regime == 2 and self._is_crash_row(feature_table.iloc[-1]):
            allow_switch = True

        if allow_switch:
            self._current_regime = proposed_regime
            self._hold_count = 0

        self._raw_state = raw_state
        self._feature_snapshot = {
            col: float(latest_row.iloc[0][col]) for col in latest_row.columns if np.isfinite(latest_row.iloc[0][col])
        }

        logger.debug(
            "Regime=%s raw=%s kind=%s features(ret72=%.4f, dd72=%.4f, breadth24=%.2f)",
            self._current_regime,
            raw_state,
            self.model_kind,
            _latest_scalar(feature_table["ret_72h"]),
            _latest_scalar(feature_table["dd_72h"]),
            _latest_scalar(feature_table["breadth_pos_24"], default=0.5),
        )
        return self._current_regime


class FeatureScoreRegimeDetector(BaseRegimeDetector):
    """Heuristic market-state detector based on literature-backed features.

    Improvements over v1:
    - Asymmetric thresholds: faster bear entry (25th pctile) than bull (75th)
    - Crash-fast: immediate BEAR on extreme drawdown + volatility conditions
    - Short-term momentum overlay: recent 24h action gets extra weight
    - Regime confidence score exposed for continuous allocation adjustments
    """

    def __init__(
        self,
        feature_set: str = "market",
        rebalance_hours: Optional[int] = None,
        train_hours: int = 24 * 90,
        min_hold_hours: int = 6,
    ):
        super().__init__(rebalance_hours=rebalance_hours)
        self.feature_set = feature_set
        self.train_hours = train_hours
        self.min_hold_cycles = max(1, min_hold_hours // self.rebalance_hours)
        self._hold_count = 0
        self._regime_score = 0.0  # continuous score for allocation tuning

    @property
    def regime_score(self) -> float:
        """Continuous regime score: positive = bullish, negative = bearish."""
        return self._regime_score

    def detect(
        self,
        prices: pd.DataFrame,
        benchmark: Optional[str] = None,
        volumes: Optional[pd.DataFrame] = None,
    ) -> int:
        feature_table = self._get_feature_table(
            prices=prices,
            volumes=volumes,
            benchmark=benchmark or cfg.BENCHMARK,
        )
        selected = FEATURE_SETS.get(self.feature_set, FEATURE_SETS["market"]).columns
        use = [c for c in selected if c in feature_table.columns]
        feature_table = feature_table[use].dropna()
        if len(feature_table) < 200:
            self._current_regime = 1
            self._regime_score = 0.0
            return 1

        window = feature_table.iloc[-self.train_hours:].copy()
        scaled = window.apply(_safe_zscore).fillna(0.0)

        # Composite regime score — weighted sum of z-scored features
        # Positive = bullish environment, negative = bearish
        score = (
            1.2 * scaled.get("ret_24h", 0.0)
            + 1.8 * scaled.get("ret_72h", 0.0)
            + 2.0 * scaled.get("ret_168h", 0.0)
            + 1.0 * scaled.get("ma_gap_72", 0.0)
            + 1.2 * scaled.get("ma_gap_168", 0.0)
            + 0.8 * scaled.get("range_pos_168h", 0.0)
            + 1.1 * scaled.get("breadth_pos_24", 0.0)
            + 1.0 * scaled.get("breadth_above_72", 0.0)
            + 0.7 * scaled.get("cs_mom_72", 0.0)
            - 0.8 * scaled.get("vol_24h", 0.0)
            - 0.8 * scaled.get("vol_72h", 0.0)
            + 1.2 * scaled.get("dd_72h", 0.0)
            + 1.0 * scaled.get("dd_168h", 0.0)
            - 0.5 * scaled.get("avg_corr_24", 0.0)
            - 0.5 * scaled.get("down_frac_24", 0.0)
        )

        # Asymmetric quantile thresholds:
        # - Bear entry at 25th percentile (faster to go defensive)
        # - Bull entry at 75th percentile (more proof needed to go risk-on)
        bear_q = float(score.quantile(0.25))
        bull_q = float(score.quantile(0.75))
        latest_score = float(score.iloc[-1])
        latest = window.iloc[-1]

        # Store continuous score (z-scored relative to window)
        score_std = float(score.std())
        score_mean = float(score.mean())
        if score_std > 1e-6:
            self._regime_score = (latest_score - score_mean) / score_std
        else:
            self._regime_score = 0.0

        # Crash-fast override: extreme drawdown conditions → immediate BEAR
        crash_conditions = (
            latest.get("ret_72h", 0.0) < -0.03
            and latest.get("dd_72h", 0.0) < -0.05
            and latest.get("breadth_pos_24", 0.5) < 0.45
        )
        severe_crash = (
            latest.get("ret_24h", 0.0) < -0.04
            and latest.get("dd_72h", 0.0) < -0.08
        )

        if severe_crash:
            # Immediate override — no hysteresis
            self._current_regime = 2
            self._hold_count = 0
            self._regime_score = min(self._regime_score, -2.0)
            return 2

        if latest_score >= bull_q:
            proposed = 0
        elif latest_score <= bear_q or crash_conditions:
            proposed = 2
        else:
            proposed = 1

        self._hold_count += 1
        if proposed == self._current_regime or self._hold_count >= self.min_hold_cycles:
            self._current_regime = proposed
            self._hold_count = 0

        return self._current_regime
