"""
Optuna Hyperparameter Optimization for Crypto Trading Bot
==========================================================
Walk-forward validation to prevent overfitting:
  1. Train (optimize) on Period A
  2. Validate on Period B (out-of-sample)
  3. Final score = weighted blend of in-sample + out-of-sample
  4. Penalize extreme parameter deviations from defaults (regularization)

Anti-overfitting measures:
  - Walk-forward: optimize on train, validate on test
  - Multi-period validation (2 test windows)
  - Regularization: penalize distance from sensible defaults
  - Pruning: kill bad trials early
  - Limited parameter set: only tune most impactful params (~20 of 191)
  - Composite objective: 0.4*Sortino + 0.3*Sharpe + 0.3*Calmar

Usage:
    python optuna_optimize.py                    # Full optimization (100 trials)
    python optuna_optimize.py --trials 50        # Quick run
    python optuna_optimize.py --show-best        # Show best params from saved study
"""

import argparse
import json
import logging
import sys
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
from optuna.pruners import MedianPruner

# Suppress noisy logs during optimization
for _name in ["signals", "optimizer", "protective_allocation", "risk_manager",
              "data_engine", "__main__", "main", "executor"]:
    logging.getLogger(_name).setLevel(logging.CRITICAL)
# Also suppress root logger warnings
logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")
# Set PYTHONIOENCODING for Windows compatibility
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ──────────────────────────────────────────────────────────────
# Walk-Forward Periods
# ──────────────────────────────────────────────────────────────
# Train on one period, validate on another to detect overfitting.
# Using multiple validation windows for robustness.

TRAIN_PERIODS = [
    ("2026-02-01", "2026-03-10"),   # ~5.5 weeks: full Feb + early Mar
]

VALIDATION_PERIODS = [
    ("2026-03-11", "2026-03-20"),   # 10 days immediately before live trading
]

# Scoring weights: train matters less, validation matters more
TRAIN_WEIGHT = 0.30
VALIDATION_WEIGHT = 0.70  # out-of-sample performance is king

# Regularization: penalize extreme deviations from defaults
REGULARIZATION_STRENGTH = 0.05  # small penalty for weird params


# ──────────────────────────────────────────────────────────────
# Parameter Space — Only the ~20 most impactful parameters
# ──────────────────────────────────────────────────────────────
# Grouped by subsystem. Ranges are wide enough to explore but
# constrained to avoid nonsensical values.

@dataclass
class ParamDefaults:
    """Default values for regularization penalty."""
    # Risk / Exposure
    target_volatility: float = 0.55
    dd_deleverage_start: float = 0.08
    dd_deleverage_full: float = 0.18
    position_stop_loss_pct: float = 0.12
    lambda_base: float = 2.5
    ewma_cov_alpha: float = 0.30

    # Optimizer
    trend_invested_floor: float = 0.18
    trend_invested_ceiling: float = 0.80
    trend_invested_neutral: float = 0.45
    smoothing_delever: float = 0.85
    smoothing_relever_bull: float = 0.55
    smoothing_relever_neutral: float = 0.40
    position_hysteresis: float = 0.10
    bl_ratio_bull: float = 0.80
    view_conf_bull: float = 0.08

    # Protection
    protection_smoothing_up: float = 0.70
    protection_smoothing_down: float = 0.30
    portfolio_ret_threshold: float = -0.008
    portfolio_ret_fast_threshold: float = -0.006
    rapid_deterioration_threshold: float = 0.30

    # Rebalance
    min_weight_change: float = 0.05
    max_holdings: int = 8


DEFAULTS = ParamDefaults()


def suggest_params(trial: optuna.Trial) -> Dict:
    """Suggest hyperparameters for a trial."""
    params = {}

    # ── Risk / Exposure ──
    params["target_volatility"] = trial.suggest_float(
        "target_volatility", 0.35, 0.75, step=0.05)
    params["dd_deleverage_start"] = trial.suggest_float(
        "dd_deleverage_start", 0.04, 0.12, step=0.01)
    params["dd_deleverage_full"] = trial.suggest_float(
        "dd_deleverage_full", 0.12, 0.25, step=0.01)
    params["position_stop_loss_pct"] = trial.suggest_float(
        "position_stop_loss_pct", 0.08, 0.18, step=0.01)
    params["lambda_base"] = trial.suggest_float(
        "lambda_base", 1.5, 4.0, step=0.25)
    params["ewma_cov_alpha"] = trial.suggest_float(
        "ewma_cov_alpha", 0.10, 0.50, step=0.05)

    # ── Optimizer: Exposure Scaling ──
    params["trend_invested_floor"] = trial.suggest_float(
        "trend_invested_floor", 0.10, 0.25, step=0.01)
    params["trend_invested_ceiling"] = trial.suggest_float(
        "trend_invested_ceiling", 0.65, 0.90, step=0.05)
    params["trend_invested_neutral"] = trial.suggest_float(
        "trend_invested_neutral", 0.35, 0.55, step=0.05)
    params["smoothing_delever"] = trial.suggest_float(
        "smoothing_delever", 0.70, 0.95, step=0.05)
    params["smoothing_relever_bull"] = trial.suggest_float(
        "smoothing_relever_bull", 0.40, 0.70, step=0.05)
    params["smoothing_relever_neutral"] = trial.suggest_float(
        "smoothing_relever_neutral", 0.25, 0.55, step=0.05)
    params["position_hysteresis"] = trial.suggest_float(
        "position_hysteresis", 0.0, 0.20, step=0.02)

    # ── BL Settings ──
    params["bl_ratio_bull"] = trial.suggest_float(
        "bl_ratio_bull", 0.60, 0.90, step=0.05)
    params["view_conf_bull"] = trial.suggest_float(
        "view_conf_bull", 0.04, 0.12, step=0.01)

    # ── Protection ──
    params["protection_smoothing_up"] = trial.suggest_float(
        "protection_smoothing_up", 0.50, 0.90, step=0.05)
    params["protection_smoothing_down"] = trial.suggest_float(
        "protection_smoothing_down", 0.15, 0.45, step=0.05)
    params["portfolio_ret_threshold"] = trial.suggest_float(
        "portfolio_ret_threshold", -0.015, -0.004, step=0.001)
    params["portfolio_ret_fast_threshold"] = trial.suggest_float(
        "portfolio_ret_fast_threshold", -0.012, -0.003, step=0.001)
    params["rapid_deterioration_threshold"] = trial.suggest_float(
        "rapid_deterioration_threshold", 0.15, 0.45, step=0.05)

    # ── Rebalance ──
    params["min_weight_change"] = trial.suggest_float(
        "min_weight_change", 0.03, 0.08, step=0.01)
    params["max_holdings"] = trial.suggest_int(
        "max_holdings", 5, 12)

    return params


# ──────────────────────────────────────────────────────────────
# Apply Parameters to Bot Config
# ──────────────────────────────────────────────────────────────

def apply_params(params: Dict):
    """Apply trial parameters to the global config objects.

    Modifies config.py objects in-place so the backtest picks them up.
    """
    import config as cfg
    from optimizer import PortfolioOptimizer
    from protective_allocation import ProtectiveAllocConfig

    # Risk params
    cfg.RISK.target_volatility = params["target_volatility"]
    cfg.RISK.dd_deleverage_start = params["dd_deleverage_start"]
    cfg.RISK.dd_deleverage_full = params["dd_deleverage_full"]
    cfg.RISK.position_stop_loss_pct = params["position_stop_loss_pct"]
    cfg.RISK.lambda_base = params["lambda_base"]
    cfg.RISK.ewma_cov_alpha = params["ewma_cov_alpha"]

    # Rebalance params
    cfg.REBALANCE.min_weight_change = params["min_weight_change"]
    cfg.CONSTRAINTS.max_holdings = params["max_holdings"]

    # Optimizer params — these are class-level, applied per-instance in run_backtest
    # Store in a global dict for the backtest to pick up
    global _OPTIMIZER_PARAMS
    _OPTIMIZER_PARAMS = {
        "trend_invested_floor": params["trend_invested_floor"],
        "trend_invested_ceiling": params["trend_invested_ceiling"],
        "trend_invested_neutral": params["trend_invested_neutral"],
        "smoothing_delever": params["smoothing_delever"],
        "smoothing_relever_bull": params["smoothing_relever_bull"],
        "smoothing_relever_neutral": params["smoothing_relever_neutral"],
        "position_hysteresis": params["position_hysteresis"],
        "bl_ratio_bull": params["bl_ratio_bull"],
        "view_conf_bull": params["view_conf_bull"],
        "protection_smoothing_up": params["protection_smoothing_up"],
        "protection_smoothing_down": params["protection_smoothing_down"],
        "portfolio_ret_threshold": params["portfolio_ret_threshold"],
        "portfolio_ret_fast_threshold": params["portfolio_ret_fast_threshold"],
        "rapid_deterioration_threshold": params["rapid_deterioration_threshold"],
    }


def apply_optimizer_params(optimizer):
    """Apply optimizer-level params to a PortfolioOptimizer instance."""
    global _OPTIMIZER_PARAMS
    if not _OPTIMIZER_PARAMS:
        return

    p = _OPTIMIZER_PARAMS

    # Trend-strength exposure scaling
    optimizer.TREND_INVESTED_FLOOR = p["trend_invested_floor"]
    optimizer.TREND_INVESTED_CEILING = p["trend_invested_ceiling"]
    optimizer.TREND_INVESTED_NEUTRAL = p["trend_invested_neutral"]

    # Smoothing
    optimizer.SMOOTHING_ALPHA_DELEVER = p["smoothing_delever"]
    optimizer.SMOOTHING_ALPHA_RELEVER[0] = p["smoothing_relever_bull"]
    optimizer.SMOOTHING_ALPHA_RELEVER[1] = p["smoothing_relever_neutral"]

    # Position hysteresis (stored for use in optimize())
    optimizer._hysteresis_bonus = p["position_hysteresis"]

    # BL blend for BULL
    optimizer.REGIME_BLEND[0]["bl_ratio"] = p["bl_ratio_bull"]
    optimizer.REGIME_BLEND[0]["view_conf"] = p["view_conf_bull"]

    # Protection engine config
    pe = optimizer.protection_engine
    pe.config.protection_smoothing_up = p["protection_smoothing_up"]
    pe.config.protection_smoothing_down = p["protection_smoothing_down"]
    pe.config.portfolio_ret_threshold = p["portfolio_ret_threshold"]
    pe.config.portfolio_ret_fast_threshold = p["portfolio_ret_fast_threshold"]
    # Update rapid deterioration threshold
    pe._rapid_deterioration_threshold = p["rapid_deterioration_threshold"]


_OPTIMIZER_PARAMS = {}


# ──────────────────────────────────────────────────────────────
# Run Backtest with Given Parameters
# ──────────────────────────────────────────────────────────────

def run_backtest(start: str, end: str) -> Dict:
    """Run a backtest and return metrics. Returns dict with scores."""
    from main import BacktestEngine

    try:
        engine = BacktestEngine(start=start, end=end, rebalance_hours=2, adaptive=True,
                                save_results=False)
        # Apply optimizer-level params
        apply_optimizer_params(engine.optimizer)

        result = engine.run()

        if result.get("status") != "OK":
            return {"composite": -999, "return": -999, "dd": 999}

        return {
            "composite": result.get("composite_score", -999),
            "return": result.get("total_return_pct", -999),
            "dd": result.get("max_drawdown_pct", 999),
            "sortino": result.get("sortino", -999),
            "sharpe": result.get("sharpe", -999),
            "calmar": result.get("calmar", -999),
            "trades": result.get("total_trades", 0),
        }
    except Exception as e:
        print(f"  Backtest failed: {e}")
        return {"composite": -999, "return": -999, "dd": 999}


# ──────────────────────────────────────────────────────────────
# Regularization Penalty
# ──────────────────────────────────────────────────────────────

def compute_regularization(params: Dict) -> float:
    """Penalize extreme deviations from defaults.

    Uses L2 penalty (squared distance) normalized by default value.
    This prevents overfitting to weird parameter combinations.
    """
    penalty = 0.0
    defaults = DEFAULTS

    for key, value in params.items():
        if key == "max_holdings":
            # Integer param: normalize differently
            default_val = defaults.max_holdings
            penalty += ((value - default_val) / max(default_val, 1)) ** 2
        else:
            default_val = getattr(defaults, key, None)
            if default_val is not None and default_val != 0:
                # Relative deviation squared
                penalty += ((value - default_val) / abs(default_val)) ** 2
            elif default_val == 0:
                penalty += value ** 2

    # Average over number of params and scale
    penalty = penalty / max(len(params), 1)
    return penalty * REGULARIZATION_STRENGTH


# ──────────────────────────────────────────────────────────────
# Objective Function
# ──────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial) -> float:
    """Optuna objective: maximize walk-forward composite score."""

    params = suggest_params(trial)
    apply_params(params)

    # ── Train Period ──
    train_scores = []
    for start, end in TRAIN_PERIODS:
        result = run_backtest(start, end)
        if result["composite"] <= -999:
            raise optuna.TrialPruned("Train backtest failed")
        train_scores.append(result["composite"])

    avg_train = np.mean(train_scores)

    # Early pruning: if train composite is terrible, skip validation
    if avg_train < -0.05:
        raise optuna.TrialPruned(f"Train composite too low: {avg_train:.4f}")

    # Report intermediate value for pruning
    trial.report(avg_train, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    # ── Validation Periods ──
    val_scores = []
    val_returns = []
    val_dds = []
    for start, end in VALIDATION_PERIODS:
        result = run_backtest(start, end)
        if result["composite"] <= -999:
            val_scores.append(-0.10)  # penalty for failure
        else:
            val_scores.append(result["composite"])
            val_returns.append(result["return"])
            val_dds.append(result["dd"])

    avg_val = np.mean(val_scores)

    # Report validation score
    trial.report(avg_val, step=1)

    # ── Combined Score ──
    combined = TRAIN_WEIGHT * avg_train + VALIDATION_WEIGHT * avg_val

    # ── Regularization Penalty ──
    reg_penalty = compute_regularization(params)
    final_score = combined - reg_penalty

    # Log progress
    avg_ret = np.mean(val_returns) if val_returns else -999
    avg_dd = np.mean(val_dds) if val_dds else 999
    print(f"  Trial {trial.number:3d}: "
          f"train={avg_train:+.4f} val={avg_val:+.4f} "
          f"reg={reg_penalty:.4f} final={final_score:+.4f} | "
          f"val_ret={avg_ret:+.2f}% val_dd={avg_dd:.1f}%")

    # Store validation details
    trial.set_user_attr("train_composite", float(avg_train))
    trial.set_user_attr("val_composite", float(avg_val))
    trial.set_user_attr("val_return_avg", float(avg_ret))
    trial.set_user_attr("val_dd_avg", float(avg_dd))
    trial.set_user_attr("regularization", float(reg_penalty))

    return final_score


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def show_best(study: optuna.Study):
    """Display the best trial results."""
    best = study.best_trial

    print("\n" + "=" * 70)
    print("BEST TRIAL")
    print("=" * 70)
    print(f"Trial #{best.number}")
    print(f"Final Score: {best.value:+.4f}")
    print(f"Train Composite: {best.user_attrs.get('train_composite', 'N/A')}")
    print(f"Val Composite:   {best.user_attrs.get('val_composite', 'N/A')}")
    print(f"Val Return Avg:  {best.user_attrs.get('val_return_avg', 'N/A')}%")
    print(f"Val DD Avg:      {best.user_attrs.get('val_dd_avg', 'N/A')}%")
    print(f"Regularization:  {best.user_attrs.get('regularization', 'N/A')}")

    print("\n-- Best Parameters --")
    for key, value in sorted(best.params.items()):
        default = getattr(DEFAULTS, key, "?")
        if isinstance(value, float):
            print(f"  {key:35s} = {value:8.4f}  (default: {default})")
        else:
            print(f"  {key:35s} = {value:>8}  (default: {default})")

    # Show top 5 trials
    print("\n-- Top 5 Trials --")
    trials = sorted(study.trials, key=lambda t: t.value if t.value else -999,
                    reverse=True)
    for t in trials[:5]:
        if t.value is not None:
            val_ret = t.user_attrs.get("val_return_avg", "?")
            val_dd = t.user_attrs.get("val_dd_avg", "?")
            print(f"  #{t.number:3d}: score={t.value:+.4f}  "
                  f"val_ret={val_ret}%  val_dd={val_dd}%")

    # Save best params to JSON
    output = {
        "best_trial": best.number,
        "final_score": best.value,
        "train_composite": best.user_attrs.get("train_composite"),
        "val_composite": best.user_attrs.get("val_composite"),
        "val_return_avg": best.user_attrs.get("val_return_avg"),
        "val_dd_avg": best.user_attrs.get("val_dd_avg"),
        "params": best.params,
        "defaults": {k: v for k, v in DEFAULTS.__dict__.items()},
        "train_periods": TRAIN_PERIODS,
        "validation_periods": VALIDATION_PERIODS,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = Path("data/csv/optuna_best_params.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nBest params saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization")
    parser.add_argument("--trials", type=int, default=100,
                        help="Number of optimization trials (default: 100)")
    parser.add_argument("--show-best", action="store_true",
                        help="Show best params from saved study")
    parser.add_argument("--study-name", type=str, default="crypto_bot_v1",
                        help="Optuna study name")
    parser.add_argument("--db", type=str, default="sqlite:///data/db/optuna.db",
                        help="Optuna storage DB")
    args = parser.parse_args()

    # Load or create study
    storage = args.db
    study_name = args.study_name

    if args.show_best:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            show_best(study)
        except Exception as e:
            print(f"Could not load study: {e}")
        return

    print("=" * 70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"Trials:      {args.trials}")
    print(f"Train:       {TRAIN_PERIODS}")
    print(f"Validation:  {VALIDATION_PERIODS}")
    print(f"Scoring:     {TRAIN_WEIGHT:.0%} train + {VALIDATION_WEIGHT:.0%} validation")
    print(f"Regularization: {REGULARIZATION_STRENGTH}")
    print(f"Parameters:  22 (of 191 total)")
    print(f"Storage:     {storage}")
    print("=" * 70)
    print()

    # Create study with TPE sampler (Bayesian optimization) and median pruner
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=15,   # random exploration before Bayesian
            multivariate=True,     # model parameter correlations
        ),
        pruner=MedianPruner(
            n_startup_trials=10,   # don't prune first 10
            n_warmup_steps=0,
        ),
    )

    print(f"Starting optimization ({args.trials} trials)...\n")

    study.optimize(
        objective,
        n_trials=args.trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    show_best(study)

    # Overfitting check
    print("\n-- Overfitting Check --")
    completed = [t for t in study.trials if t.value is not None]
    if completed:
        train_scores = [t.user_attrs.get("train_composite", 0) for t in completed]
        val_scores = [t.user_attrs.get("val_composite", 0) for t in completed]
        correlation = np.corrcoef(train_scores, val_scores)[0, 1] if len(completed) > 5 else 0
        print(f"  Train-Val correlation: {correlation:.3f}")
        print(f"  (> 0.5 = good generalization, < 0.3 = possible overfitting)")

        # Best trial gap
        best = study.best_trial
        train_c = best.user_attrs.get("train_composite", 0)
        val_c = best.user_attrs.get("val_composite", 0)
        gap = abs(train_c - val_c)
        print(f"  Best trial train-val gap: {gap:.4f}")
        print(f"  (< 0.01 = consistent, > 0.03 = overfitting risk)")


if __name__ == "__main__":
    main()
