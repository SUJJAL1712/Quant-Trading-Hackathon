"""
Risk Manager — Crypto Portfolio Risk Monitoring
=================================================
VaR, CVaR, drawdown tracking, position stop-losses.
Adapted for crypto volatility (3-5x equity vol).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)


class RiskManager:
    """Portfolio risk monitoring and control.

    Tracks:
    - Portfolio NAV and drawdown
    - Per-position P&L and trailing stops
    - Correlation regime (crash detection)
    - VaR/CVaR for risk budgeting
    """

    def __init__(self):
        self.hwm = 0.0          # high water mark
        self.nav_history: List[float] = []
        self.position_hwm: Dict[str, float] = {}  # per-position high water marks

    def update_nav(self, nav: float):
        """Update NAV tracking."""
        self.nav_history.append(nav)
        if nav > self.hwm:
            self.hwm = nav

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from HWM. Returns 0 to 1."""
        if self.hwm <= 0:
            return 0.0
        current = self.nav_history[-1] if self.nav_history else self.hwm
        return max(0, (self.hwm - current) / self.hwm)

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown ever observed."""
        if len(self.nav_history) < 2:
            return 0.0
        navs = np.array(self.nav_history)
        peaks = np.maximum.accumulate(navs)
        drawdowns = (peaks - navs) / np.where(peaks > 0, peaks, 1)
        return float(drawdowns.max())

    def should_deleverage(self) -> Tuple[bool, float]:
        """Check if drawdown-based deleveraging should activate.

        Returns:
            (should_deleverage, scale_factor) tuple
        """
        dd = self.current_drawdown
        if dd < cfg.RISK.dd_deleverage_start:
            return False, 1.0

        # Linear scale from start to full
        dd_range = cfg.RISK.dd_deleverage_full - cfg.RISK.dd_deleverage_start
        if dd_range <= 0:
            return True, 0.3

        progress = (dd - cfg.RISK.dd_deleverage_start) / dd_range
        scale = 1.0 - 0.7 * min(progress, 1.0)  # scale from 1.0 down to 0.3
        return True, max(scale, 0.3)

    def compute_var(self, returns: pd.DataFrame, weights: pd.Series,
                    confidence: float = 0.95, method: str = "historical") -> float:
        """Compute portfolio Value-at-Risk.

        Args:
            returns: Historical return DataFrame
            weights: Current portfolio weights
            confidence: VaR confidence level (e.g., 0.95)
            method: "historical" or "parametric"

        Returns:
            VaR as a positive number (loss amount as fraction of portfolio)
        """
        common = weights.index.intersection(returns.columns)
        if len(common) == 0:
            return 0.0

        w = weights.reindex(common, fill_value=0).values
        port_returns = (returns[common].values @ w)

        if method == "historical":
            var = -np.percentile(port_returns, (1 - confidence) * 100)
        else:
            mu = port_returns.mean()
            sigma = port_returns.std()
            from scipy.stats import norm
            var = -(mu + sigma * norm.ppf(1 - confidence))

        return max(var, 0)

    def compute_cvar(self, returns: pd.DataFrame, weights: pd.Series,
                     confidence: float = 0.95) -> float:
        """Compute Conditional VaR (Expected Shortfall)."""
        common = weights.index.intersection(returns.columns)
        if len(common) == 0:
            return 0.0

        w = weights.reindex(common, fill_value=0).values
        port_returns = (returns[common].values @ w)

        var_threshold = np.percentile(port_returns, (1 - confidence) * 100)
        tail = port_returns[port_returns <= var_threshold]
        if len(tail) == 0:
            return 0.0
        return -float(tail.mean())

    def check_position_stops(self, current_prices: Dict[str, float],
                             entry_prices: Dict[str, float],
                             stop_pct_override: float = None) -> List[str]:
        """Check per-position trailing stop-losses.

        Args:
            current_prices: Current prices per coin
            entry_prices: Entry prices per coin
            stop_pct_override: Override the default stop-loss percentage

        Returns:
            List of coins that triggered stop-loss
        """
        stopped = []
        stop_pct = stop_pct_override or cfg.RISK.position_stop_loss_pct

        for coin, current in current_prices.items():
            if coin not in entry_prices or current <= 0:
                continue

            # Update high water mark for this position
            if coin not in self.position_hwm:
                self.position_hwm[coin] = entry_prices[coin]
            self.position_hwm[coin] = max(self.position_hwm[coin], current)

            # Trailing stop from position HWM
            hwm = self.position_hwm[coin]
            dd = (hwm - current) / hwm
            if dd >= stop_pct:
                logger.warning("Stop-loss triggered for %s: %.1f%% from HWM (%.2f -> %.2f)",
                               coin, dd * 100, hwm, current)
                stopped.append(coin)

        return stopped

    def check_correlation_spike(self, returns: pd.DataFrame,
                                lookback: int = 24) -> bool:
        """Detect correlation spike (crash regime).

        In crypto crashes, all assets correlate to ~1.0.
        """
        if len(returns) < lookback:
            return False

        recent = returns.iloc[-lookback:]
        corr = recent.corr()

        # Average pairwise correlation (excluding diagonal)
        n = len(corr)
        if n < 2:
            return False

        mask = ~np.eye(n, dtype=bool)
        avg_corr = corr.values[mask].mean()

        if avg_corr > cfg.RISK.correlation_spike_threshold:
            logger.warning("Correlation spike detected: avg=%.3f (threshold=%.3f)",
                           avg_corr, cfg.RISK.correlation_spike_threshold)
            return True
        return False

    def compute_sortino_ratio(self, returns: np.ndarray) -> float:
        """Compute Sortino ratio from portfolio returns."""
        if len(returns) < 2:
            return 0.0
        mean_ret = np.mean(returns)
        downside = returns[returns < 0]
        if len(downside) == 0:
            return 10.0 if mean_ret > 0 else 0.0
        downside_std = np.std(downside)
        if downside_std == 0:
            return 0.0
        return float(mean_ret / downside_std)

    def compute_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Compute Sharpe ratio from portfolio returns."""
        if len(returns) < 2:
            return 0.0
        std = np.std(returns)
        if std == 0:
            return 0.0
        return float(np.mean(returns) / std)

    def compute_calmar_ratio(self, returns: np.ndarray) -> float:
        """Compute Calmar ratio from portfolio returns."""
        if len(returns) < 2:
            return 0.0
        # Compute max drawdown from cumulative returns
        cum = np.cumprod(1 + returns)
        peaks = np.maximum.accumulate(cum)
        drawdowns = (peaks - cum) / np.where(peaks > 0, peaks, 1)
        max_dd = drawdowns.max()
        if max_dd == 0:
            return 10.0 if np.mean(returns) > 0 else 0.0
        return float(np.mean(returns) / max_dd)

    def get_risk_summary(self, returns: pd.DataFrame = None,
                         weights: pd.Series = None) -> Dict:
        """Get comprehensive risk summary."""
        summary = {
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "hwm": self.hwm,
            "nav_history_len": len(self.nav_history),
        }

        if returns is not None and weights is not None:
            summary["var_95"] = self.compute_var(returns, weights)
            summary["cvar_95"] = self.compute_cvar(returns, weights)

        if self.nav_history and len(self.nav_history) > 1:
            navs = np.array(self.nav_history)
            nav_returns = np.diff(navs) / navs[:-1]
            summary["sortino"] = self.compute_sortino_ratio(nav_returns)
            summary["sharpe"] = self.compute_sharpe_ratio(nav_returns)
            summary["calmar"] = self.compute_calmar_ratio(nav_returns)

        return summary

    def reset_position_hwm(self, coin: str):
        """Reset position HWM when a position is closed."""
        self.position_hwm.pop(coin, None)
