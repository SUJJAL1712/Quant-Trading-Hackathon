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

    # Regime-dependent blending ratios (invested target now set by trend_strength)
    REGIME_BLEND = {
        0: {"bl_ratio": 0.60, "view_conf": 0.10, "vol_scale_range": (0.65, 1.5)},  # Optuna: more HRP diversification, stronger alpha conviction
        1: {"bl_ratio": 0.50, "view_conf": 0.05, "vol_scale_range": (0.45, 1.2)},
        2: {"bl_ratio": 0.25, "view_conf": 0.02, "vol_scale_range": (0.25, 0.8)},
    }

    # Continuous trend-strength → target_invested mapping
    # Based on Baltas & Kosowski (2020): scale exposure proportionally to trend conviction
    # trend_strength ∈ [-1, 1] maps to [BEAR_FLOOR, BULL_CEILING]
    TREND_INVESTED_FLOOR = 0.18   # minimum invested even in max bear
    TREND_INVESTED_CEILING = 0.85  # Optuna (was 0.80): slightly more invested in strong bull
    TREND_INVESTED_NEUTRAL = 0.55  # Optuna (was 0.45): hold more in neutral regime

    # Asymmetric smoothing: delever fast, re-lever at regime-appropriate speed
    # Research: Moskowitz et al. (2012) "Time Series Momentum" — delayed re-entry
    # destroys more alpha than it saves in whipsaw avoidance
    SMOOTHING_ALPHA_DELEVER = 0.75   # Optuna (was 0.85): deleverage faster
    SMOOTHING_ALPHA_RELEVER = {
        0: 0.70,   # Optuna (was 0.55): re-lever fast in bull
        1: 0.45,   # Optuna (was 0.40): slightly faster in neutral
        2: 0.30,   # Bear: slow re-lever (bear rallies are traps)
    }

    def __init__(self):
        self._prev_weights = pd.Series(dtype=float)
        self.protection_engine = ProtectiveAllocationEngine()

    def optimize(self, alpha_scores: pd.Series, returns: pd.DataFrame,
                 regime: int = 1, current_drawdown: float = 0.0,
                 prices: pd.DataFrame = None,
                 trend_strength: float = 0.0) -> Tuple[pd.Series, Dict]:
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
        vol_lo, _ = params["vol_scale_range"]

        # Continuous trend-strength → target_invested (replaces discrete buckets)
        ts = np.clip(trend_strength, -1.0, 1.0)
        if ts >= 0:
            target_invested = self.TREND_INVESTED_NEUTRAL + \
                ts * (self.TREND_INVESTED_CEILING - self.TREND_INVESTED_NEUTRAL)
        else:
            target_invested = self.TREND_INVESTED_NEUTRAL + \
                ts * (self.TREND_INVESTED_NEUTRAL - self.TREND_INVESTED_FLOOR)

        # Protective allocation: compute stress-based cash shield
        protection_result = {}
        if prices is not None and not prices.empty:
            # Use actual portfolio returns if available (set by backtest engine)
            port_ret = getattr(self.protection_engine, '_portfolio_returns', None)
            protection_result = self.protection_engine.compute_protection(
                prices=prices,
                returns=ret,
                current_drawdown=current_drawdown,
                regime=regime,
                portfolio_returns=port_ret,
            )
            invested_mult = protection_result.get("invested_multiplier", 1.0)
            target_invested *= invested_mult
            min_invested = self.protection_engine.config.min_invested_override
            target_invested = max(target_invested, min_invested)

        # Concentrate: keep top-N coins by alpha score
        # Position hysteresis: coins already held get a persistence bonus
        # to reduce churn (DeMiguel et al. 2014 "Stock Return Serial Dependence
        # and Out-of-Sample Portfolio Performance")
        max_hold = cfg.CONSTRAINTS.max_holdings
        held_coins = set(self._prev_weights[self._prev_weights > 0.01].index) \
            if not self._prev_weights.empty else set()
        alpha_adj = alpha.copy()
        for coin in held_coins:
            if coin in alpha_adj.index and alpha_adj[coin] > -0.5:
                # Held coins with non-terrible alpha get a persistence bonus
                # Reduces churn by making it harder to cycle positions
                alpha_adj[coin] += getattr(self, '_hysteresis_bonus', 0.08)  # Optuna (was 0.10)

        top_coins = alpha_adj.nlargest(max_hold)
        # Only include coins with positive adjusted alpha
        top_coins = top_coins[top_coins > 0]
        if len(top_coins) < cfg.CONSTRAINTS.min_holdings:
            top_coins = alpha_adj.nlargest(cfg.CONSTRAINTS.min_holdings)

        concentrated = [c for c in common if c in top_coins.index]
        if not concentrated:
            concentrated = common[:max_hold]

        ret_c = ret[concentrated]
        alpha_c = alpha.reindex(concentrated, fill_value=0)

        # Covariance estimation (Ledoit-Wolf shrinkage + EWMA blend)
        cov = self._estimate_cov(ret_c)

        # ── Dynamic Risk Aversion (λ) Scaling ──
        # Compute average pairwise correlation from the covariance matrix
        n_assets = len(concentrated)
        lambda_eff = cfg.RISK.lambda_base
        avg_corr = 0.0
        if n_assets >= 2:
            stds = np.sqrt(np.diag(cov.values))
            stds_outer = np.outer(stds, stds)
            stds_outer = np.where(stds_outer > 0, stds_outer, 1e-10)
            corr_matrix = cov.values / stds_outer
            np.fill_diagonal(corr_matrix, 0)
            avg_corr = corr_matrix.sum() / (n_assets * (n_assets - 1))

            # Scale λ with correlation spike
            corr_boost = max(0.0, avg_corr - cfg.RISK.lambda_corr_threshold)
            lambda_eff *= (1.0 + cfg.RISK.lambda_corr_sensitivity * corr_boost)

        # Scale λ with drawdown
        dd_range = cfg.RISK.dd_deleverage_full - cfg.RISK.lambda_dd_start
        if dd_range > 0 and current_drawdown > cfg.RISK.lambda_dd_start:
            dd_boost = min((current_drawdown - cfg.RISK.lambda_dd_start) / dd_range, 1.0)
            lambda_eff *= (1.0 + cfg.RISK.lambda_dd_sensitivity * dd_boost)

        logger.info("Dynamic lambda: base=%.1f -> eff=%.1f (avg_corr=%.3f, DD=%.1f%%)",
                    cfg.RISK.lambda_base, lambda_eff, avg_corr, current_drawdown * 100)

        # Black-Litterman weights (with dynamic risk aversion)
        bl_weights = BlackLitterman.compute(
            alpha_c, cov, view_confidence=view_conf, risk_aversion=lambda_eff)

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
        protection_active = protection_result.get("protection_score", 0.0) > 0.1
        if cfg.RISK.drawdown_deleveraging and current_drawdown > cfg.RISK.dd_deleverage_start and not protection_active:
            dd_range_dl = cfg.RISK.dd_deleverage_full - cfg.RISK.dd_deleverage_start
            if dd_range_dl > 0:
                deleverage = 1 - 0.8 * min((current_drawdown - cfg.RISK.dd_deleverage_start) / dd_range_dl, 1.0)
                deleverage = max(deleverage, 0.10)  # can go as low as 10% invested
                combined *= deleverage
                logger.info("Drawdown deleveraging: %.1f%% DD -> scale=%.2f",
                            current_drawdown * 100, deleverage)

        # ── Enhanced Volatility Targeting ──
        # Cap at vol_scale_cap (1.0) — NEVER lever up, only delever
        if cfg.RISK.vol_scaling and not ret_c.empty:
            port_vol = self._portfolio_vol(combined, cov)
            if port_vol > 0:
                target_vol = cfg.RISK.target_volatility / np.sqrt(365 * 24)  # hourly target
                vol_scale = target_vol / port_vol
                vol_scale = np.clip(vol_scale, vol_lo, cfg.RISK.vol_scale_cap)
                combined *= vol_scale

        # Enforce constraints
        combined = self._enforce_constraints(combined)

        # Asymmetric allocation smoothing: delever fast, re-lever slow
        if not self._prev_weights.empty:
            all_coins = combined.index.union(self._prev_weights.index)
            prev = self._prev_weights.reindex(all_coins, fill_value=0)
            curr = combined.reindex(all_coins, fill_value=0)

            # Choose smoothing alpha based on direction + regime
            prev_total = prev.sum()
            curr_total = curr.sum()
            if curr_total < prev_total:
                # Derisking: move fast toward lower exposure
                smooth_alpha = self.SMOOTHING_ALPHA_DELEVER
            else:
                # Re-levering: speed depends on regime (bull = faster)
                smooth_alpha = self.SMOOTHING_ALPHA_RELEVER.get(regime, 0.40)

            smoothed = (1 - smooth_alpha) * prev + smooth_alpha * curr
            # Drop coins that are being exited (new target = 0 and prev < min threshold)
            smoothed = smoothed[smoothed > 0.005]
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
            "trend_strength": float(trend_strength),
            "target_invested": float(target_invested),
        }

        return combined, diagnostics

    def _estimate_cov(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate covariance matrix with Ledoit-Wolf shrinkage + EWMA blend.

        Blends the full-sample shrunk covariance with a recent-24h covariance
        to make the estimate more reactive to sudden vol regime changes.
        """
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(returns.values)
            cov = pd.DataFrame(lw.covariance_,
                               index=returns.columns, columns=returns.columns)
        except (ImportError, Exception) as e:
            logger.debug("LedoitWolf failed (%s), using sample cov", e)
            cov = returns.cov()

        # EWMA blend: mix in recent 24h covariance for crash responsiveness
        alpha = cfg.RISK.ewma_cov_alpha
        if alpha > 0 and len(returns) >= 24:
            recent_cov = returns.iloc[-24:].cov()
            cov = (1 - alpha) * cov + alpha * recent_cov

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
        Applies both sector caps AND asset-class caps (per-coin and per-class).
        """
        cons = cfg.CONSTRAINTS

        # Remove near-zero weights
        weights = weights[weights > 0.005].copy()

        if weights.empty:
            return weights

        # Enforce max holdings by keeping only top-N
        if len(weights) > cons.max_holdings:
            weights = weights.nlargest(cons.max_holdings)

        # Need minimum number of holdings — scale to max_invested_pct
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
            elif total < cons.min_invested_pct:
                weights = weights * (cons.min_invested_pct / total)

            # Cap individual positions — use per-class max or global max, whichever is lower
            any_capped = False
            for coin in weights.index:
                class_max = cfg.get_class_param(coin, "max_position_pct",
                                                 cons.max_single_position_pct)
                cap = min(class_max, cons.max_single_position_pct)
                if weights[coin] > cap:
                    excess = weights[coin] - cap
                    weights[coin] = cap
                    uncapped = (weights.index != coin) & (weights > 0)
                    if uncapped.any():
                        uncapped_total = weights[uncapped].sum()
                        if uncapped_total > 0:
                            weights[uncapped] += excess * weights[uncapped] / uncapped_total
                    any_capped = True

            if any_capped:
                continue

            # Cap asset classes (aggregate limit per class)
            class_capped = False
            for ac_name in set(cfg.get_asset_class(c) for c in weights.index):
                class_coins = [c for c in weights.index
                               if cfg.get_asset_class(c) == ac_name]
                class_max = cfg.ASSET_CLASS_PARAMS.get(ac_name, {}).get(
                    "max_class_pct", 1.0)
                class_total = weights[class_coins].sum()
                if class_total > class_max:
                    scale = class_max / class_total
                    weights[class_coins] *= scale
                    class_capped = True

            if class_capped:
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
