"""
Portfolio Optimizer — Crypto BL + HRP
======================================
Black-Litterman views (from alpha signals) blended with
Hierarchical Risk Parity for robust weight estimation.
Regime-dependent blending: more BL in bull, more HRP in bear.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

import config as cfg
from protective_allocation import ProtectiveAllocationEngine, ProtectiveAllocConfig

logger = logging.getLogger(__name__)


class BlackLitterman:
    """Black-Litterman model adapted for crypto.

    Prior: market-cap-weighted or equal-weight proxy.
    Views: alpha scores -> expected return views.
    """

    @staticmethod
    def compute(alpha_scores: pd.Series, cov_matrix: pd.DataFrame,
                view_confidence: float = 0.05,
                risk_aversion: float = 2.5) -> pd.Series:
        """Compute BL posterior weights.

        Args:
            alpha_scores: Signal-based views (z-scores)
            cov_matrix: Covariance matrix of returns
            view_confidence: How much to trust views vs prior (higher = more trust)
            risk_aversion: Market risk aversion parameter

        Returns:
            pd.Series of optimal weights
        """
        n = len(alpha_scores)
        if n == 0:
            return pd.Series(dtype=float)

        coins = alpha_scores.index.tolist()
        Sigma = cov_matrix.loc[coins, coins].values

        # Prior: equal-weight equilibrium returns
        w_eq = np.ones(n) / n
        pi = risk_aversion * Sigma @ w_eq  # implied equilibrium returns

        # View matrix: each coin has its own view
        P = np.eye(n)
        Q = alpha_scores.values * view_confidence  # scale views by confidence

        # Omega: uncertainty in views (diagonal, proportional to variance)
        tau = 0.05  # scaling factor for prior uncertainty
        Omega = np.diag(np.diag(tau * P @ Sigma @ P.T)) + 1e-8 * np.eye(n)

        # BL posterior mean
        tau_Sigma = tau * Sigma
        inv_tau_Sigma = np.linalg.inv(tau_Sigma + 1e-8 * np.eye(n))
        inv_Omega = np.linalg.inv(Omega)

        posterior_cov = np.linalg.inv(inv_tau_Sigma + P.T @ inv_Omega @ P)
        posterior_mean = posterior_cov @ (inv_tau_Sigma @ pi + P.T @ inv_Omega @ Q)

        # Optimal weights from posterior
        w = np.linalg.solve(risk_aversion * Sigma + 1e-8 * np.eye(n), posterior_mean)

        # Clip negatives (long-only in crypto spot)
        w = np.maximum(w, 0)
        total = w.sum()
        if total > 0:
            w = w / total

        return pd.Series(w, index=coins)


class HierarchicalRiskParity:
    """HRP: correlation-based clustering + inverse-variance allocation.

    More robust than mean-variance in high dimensions and short data.
    """

    @staticmethod
    def compute(returns: pd.DataFrame) -> pd.Series:
        """Compute HRP weights.

        Args:
            returns: Log return DataFrame (time × coins)

        Returns:
            pd.Series of HRP weights
        """
        coins = returns.columns.tolist()
        n = len(coins)

        if n <= 1:
            return pd.Series(1.0 / max(n, 1), index=coins)

        # Correlation and covariance
        corr = returns.corr().fillna(0)
        cov = returns.cov().fillna(0)

        # Distance matrix from correlation
        dist = np.sqrt(0.5 * (1 - corr.values))
        np.fill_diagonal(dist, 0)
        dist = np.maximum(dist, 0)  # ensure non-negative

        # Hierarchical clustering
        try:
            condensed = squareform(dist, checks=False)
            link = linkage(condensed, method="single")
            sort_ix = leaves_list(link).tolist()
        except Exception as e:
            logger.warning("HRP clustering failed: %s, using equal weight", e)
            return pd.Series(1.0 / n, index=coins)

        # Recursive bisection
        sorted_coins = [coins[i] for i in sort_ix]
        weights = pd.Series(1.0, index=sorted_coins)

        cluster_items = [sorted_coins]
        while cluster_items:
            new_items = []
            for subset in cluster_items:
                if len(subset) <= 1:
                    continue
                mid = len(subset) // 2
                left = subset[:mid]
                right = subset[mid:]

                # Inverse-variance allocation between clusters
                left_var = HierarchicalRiskParity._cluster_var(cov, left)
                right_var = HierarchicalRiskParity._cluster_var(cov, right)

                total_var = left_var + right_var
                if total_var > 0:
                    alloc_left = 1 - left_var / total_var
                else:
                    alloc_left = 0.5

                for coin in left:
                    weights[coin] *= alloc_left
                for coin in right:
                    weights[coin] *= (1 - alloc_left)

                if len(left) > 1:
                    new_items.append(left)
                if len(right) > 1:
                    new_items.append(right)

            cluster_items = new_items

        # Normalize
        total = weights.sum()
        if total > 0:
            weights = weights / total
        return weights.reindex(coins, fill_value=0)

    @staticmethod
    def _cluster_var(cov: pd.DataFrame, subset: list) -> float:
        """Compute inverse-variance weight of a cluster."""
        sub_cov = cov.loc[subset, subset].values
        n = len(subset)
        w = np.ones(n) / n
        var = w @ sub_cov @ w
        return max(var, 1e-10)


class PortfolioOptimizer:
    """Master optimizer: BL + HRP with regime-dependent blending.

    Bull:    80% BL, 20% HRP (trust signals more)
    Neutral: 50% BL, 50% HRP (balanced)
    Bear:    25% BL, 75% HRP (trust diversification more)
    """

    # Regime-dependent blending ratios and invested targets
    REGIME_BLEND = {
        0: {"bl_ratio": 0.80, "view_conf": 0.08, "vol_scale_range": (0.8, 1.8),
            "target_invested": 0.85},   # Bull: maximize gains in uptrend
        1: {"bl_ratio": 0.50, "view_conf": 0.05, "vol_scale_range": (0.5, 1.3),
            "target_invested": 0.65},   # Neutral: moderate
        2: {"bl_ratio": 0.10, "view_conf": 0.02, "vol_scale_range": (0.3, 0.8),
            "target_invested": 0.50},   # Bear: mostly HRP for better diversification
    }

    # Allocation smoothing: blend toward target weights instead of jumping
    SMOOTHING_ALPHA = 0.55  # slightly faster than 0.5

    def __init__(self):
        self._prev_weights = pd.Series(dtype=float)
        self.protection_engine = ProtectiveAllocationEngine()

    def optimize(self, alpha_scores: pd.Series, returns: pd.DataFrame,
                 regime: int = 1, current_drawdown: float = 0.0,
                 prices: pd.DataFrame = None) -> Tuple[pd.Series, Dict]:
        """Run full optimization pipeline.

        Args:
            alpha_scores: Alpha signals per coin
            returns: Historical log returns
            regime: Current market regime (0=Bull, 1=Neutral, 2=Bear)
            current_drawdown: Current portfolio drawdown (0 to 1)

        Returns:
            (weights, diagnostics) tuple
        """
        coins = alpha_scores.index.tolist()
        if not coins:
            return pd.Series(dtype=float), {"error": "no coins"}

        # Filter returns to match alpha coins
        common = [c for c in coins if c in returns.columns]
        if not common:
            return pd.Series(1.0 / len(coins), index=coins), {"error": "no matching returns"}

        # Don't dropna(how='any') — that kills all rows whenever a single
        # coin has a gap.  Instead, keep rows where at least half the coins
        # have data, and let covariance / BL handle per-pair completeness.
        ret = returns[common].dropna(how="all")
        # Further filter: only keep coins with enough non-NaN rows to
        # estimate covariance (at least 48 observations)
        min_obs = min(48, max(len(ret) // 4, 10))
        good_coins = [c for c in common if ret[c].notna().sum() >= min_obs]
        if not good_coins:
            good_coins = common  # fallback: use whatever we have
        ret = ret[good_coins].dropna()
        common = good_coins
        alpha = alpha_scores.reindex(common, fill_value=0)

        # Get regime parameters
        params = self.REGIME_BLEND.get(regime, self.REGIME_BLEND[1])
        bl_ratio = params["bl_ratio"]
        view_conf = params["view_conf"]
        vol_lo, vol_hi = params["vol_scale_range"]
        target_invested = params["target_invested"]

        # Protective allocation: compute stress-based cash shield
        protection_result = {}
        if prices is not None and not prices.empty:
            protection_result = self.protection_engine.compute_protection(
                prices=prices,
                returns=ret,
                current_drawdown=current_drawdown,
                regime=regime,
            )
            # Scale target_invested by the protection multiplier
            invested_mult = protection_result.get("invested_multiplier", 1.0)
            target_invested *= invested_mult

            # Floor: don't go below minimum
            min_invested = self.protection_engine.config.min_invested_override
            target_invested = max(target_invested, min_invested)

        # Concentrate: keep top-N coins by alpha score.
        # Don't filter to only positive alpha — the optimizer weights and
        # target_invested scaling handle exposure. Filtering to positive-only
        # causes extreme concentration (3-5 names) when most alphas are negative.
        max_hold = cfg.CONSTRAINTS.max_holdings
        min_hold = cfg.CONSTRAINTS.min_holdings
        top_coins = alpha.nlargest(max(max_hold, min_hold))
        # Ensure minimum diversification
        if len(top_coins) < min_hold:
            top_coins = alpha.nlargest(min_hold)

        concentrated = [c for c in common if c in top_coins.index]
        if not concentrated:
            concentrated = common[:max_hold]

        ret_c = ret[concentrated]
        alpha_c = alpha.reindex(concentrated, fill_value=0)

        # Covariance estimation (Ledoit-Wolf shrinkage)
        cov = self._estimate_cov(ret_c)

        # Black-Litterman weights
        bl_weights = BlackLitterman.compute(alpha_c, cov, view_confidence=view_conf)

        # HRP weights
        hrp_weights = HierarchicalRiskParity.compute(ret_c)

        # Blend
        combined = bl_ratio * bl_weights.reindex(concentrated, fill_value=0) + \
                   (1 - bl_ratio) * hrp_weights.reindex(concentrated, fill_value=0)

        # Clip negatives (long-only)
        combined = combined.clip(lower=0)

        # Apply regime-dependent invested target
        total = combined.sum()
        if total > 0:
            combined = combined * (target_invested / total)

        # Drawdown deleveraging — skip if protection is already handling DD
        # (they both respond to drawdown, stacking them crushes exposure)
        protection_active = protection_result.get("protection_score", 0.0) > 0.1
        if cfg.RISK.drawdown_deleveraging and current_drawdown > cfg.RISK.dd_deleverage_start and not protection_active:
            dd_range = cfg.RISK.dd_deleverage_full - cfg.RISK.dd_deleverage_start
            if dd_range > 0:
                deleverage = 1 - 0.7 * min((current_drawdown - cfg.RISK.dd_deleverage_start) / dd_range, 1.0)
                deleverage = max(deleverage, 0.3)
                combined *= deleverage
                logger.info("Drawdown deleveraging: %.1f%% DD -> scale=%.2f",
                            current_drawdown * 100, deleverage)

        # Volatility targeting
        if cfg.RISK.vol_scaling and not ret_c.empty:
            port_vol = self._portfolio_vol(combined, cov)
            if port_vol > 0:
                target_vol = cfg.RISK.target_volatility / np.sqrt(365 * 24)  # hourly target
                vol_scale = target_vol / port_vol
                vol_scale = np.clip(vol_scale, vol_lo, vol_hi)
                combined *= vol_scale

        # Enforce constraints
        combined = self._enforce_constraints(combined)

        # Allocation smoothing: blend from previous weights toward new target
        if not self._prev_weights.empty:
            all_coins = combined.index.union(self._prev_weights.index)
            prev = self._prev_weights.reindex(all_coins, fill_value=0)
            curr = combined.reindex(all_coins, fill_value=0)
            smoothed = (1 - self.SMOOTHING_ALPHA) * prev + self.SMOOTHING_ALPHA * curr
            # Drop only truly negligible positions
            smoothed = smoothed[smoothed > 0.002]
            # Re-enforce constraints after smoothing (smoothing can reintroduce
            # old names and violate position/invested limits)
            combined = self._enforce_constraints(smoothed)

        self._prev_weights = combined.copy()

        diagnostics = {
            "regime": regime,
            "bl_ratio": bl_ratio,
            "n_positions": (combined > 0.01).sum(),
            "invested_pct": combined.sum(),
            "max_position": combined.max(),
            "protection_score": protection_result.get("protection_score", 0.0),
            "cash_fraction": protection_result.get("cash_fraction", 0.0),
        }

        return combined, diagnostics

    def _estimate_cov(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate covariance matrix with Ledoit-Wolf shrinkage."""
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(returns.values)
            cov = pd.DataFrame(lw.covariance_,
                               index=returns.columns, columns=returns.columns)
        except (ImportError, Exception) as e:
            logger.debug("LedoitWolf failed (%s), using sample cov", e)
            cov = returns.cov()
        return cov

    def _portfolio_vol(self, weights: pd.Series, cov: pd.DataFrame) -> float:
        """Compute portfolio volatility."""
        common = weights.index.intersection(cov.index)
        if len(common) == 0:
            return 0.0
        w = weights.reindex(common, fill_value=0).values
        C = cov.loc[common, common].values
        var = w @ C @ w
        return np.sqrt(max(var, 0))

    def _enforce_constraints(self, weights: pd.Series) -> pd.Series:
        """Enforce portfolio constraints via iterative capping.

        Respects regime-dependent invested target (weights may sum < 1.0).
        """
        cons = cfg.CONSTRAINTS

        # Remove near-zero weights
        weights = weights[weights > 0.002].copy()

        if weights.empty:
            return weights

        # Enforce max holdings by keeping only top-N
        if len(weights) > cons.max_holdings:
            weights = weights.nlargest(cons.max_holdings)

        # Need minimum number of holdings — scale to max_invested_pct
        # (don't force full investment with a sparse selection)
        if len(weights) < cons.min_holdings:
            n = max(len(weights), 1)
            per_coin = min(1.0 / n, cons.max_single_position_pct)
            total = per_coin * n
            if total > cons.max_invested_pct:
                per_coin = cons.max_invested_pct / n
            return pd.Series(per_coin, index=weights.index)

        # Iterative capping (waterfilling)
        for _ in range(20):
            total = weights.sum()
            if total <= 0:
                break

            # Respect invested bounds
            if total > cons.max_invested_pct:
                weights = weights * (cons.max_invested_pct / total)
            # Don't force minimum investment — if the optimizer wants
            # less than min_invested, let it (the target_invested scaling
            # already set the desired level based on regime)

            # Cap individual positions
            capped = weights > cons.max_single_position_pct
            if capped.any():
                excess = weights[capped] - cons.max_single_position_pct
                weights[capped] = cons.max_single_position_pct
                uncapped = ~capped & (weights > 0)
                if uncapped.any():
                    uncapped_total = weights[uncapped].sum()
                    if uncapped_total > 0:
                        redistrib = excess.sum() * weights[uncapped] / uncapped_total
                        weights[uncapped] += redistrib
                continue

            # Cap sectors
            for sector in set(cfg.SECTOR_MAP.get(coin, "Other") for coin in weights.index):
                sector_coins = [coin for coin in weights.index
                                if cfg.SECTOR_MAP.get(coin, "Other") == sector]
                sector_total = weights[sector_coins].sum()
                if sector_total > cons.max_sector_pct:
                    scale = cons.max_sector_pct / sector_total
                    weights[sector_coins] *= scale

            break  # No constraint violation found

        weights = weights.clip(lower=0)
        return weights
