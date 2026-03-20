"""
Adaptive vs Static Signal Weighting Comparison
===============================================
Compares the improved adaptive signal weighting (regime-specific IC_IR tracking)
against static base weights across multiple time periods.

Usage:
    python adaptive_comparison.py
"""

import json
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("adaptive_compare")

from main import BacktestEngine

# Test periods
PERIODS = {
    "bear_2024Q4":  ("2024-10-01", "2024-12-31"),
    "bull_2025Q1":  ("2025-01-01", "2025-03-15"),
    "full_6mo":     ("2024-09-15", "2025-03-15"),
}

# Use the best frequency from the frequency comparison
REBALANCE_HOURS = 2


def run_comparison():
    """Run adaptive vs static across all periods."""
    results = []

    for period_name, (start, end) in PERIODS.items():
        for adaptive in [False, True]:
            mode = "adaptive" if adaptive else "static"
            logger.info("=" * 60)
            logger.info("Running %s | %s | %dh rebalance", period_name, mode, REBALANCE_HOURS)
            logger.info("=" * 60)

            t0 = time.time()
            try:
                engine = BacktestEngine(
                    start=start, end=end,
                    rebalance_hours=REBALANCE_HOURS,
                    adaptive=adaptive,
                )
                result = engine.run()
                elapsed = time.time() - t0

                if result.get("status") == "OK":
                    row = {
                        "period": period_name,
                        "mode": mode,
                        "total_return_pct": result["total_return_pct"],
                        "max_drawdown_pct": result["max_drawdown_pct"],
                        "sortino": result["sortino"],
                        "sharpe": result["sharpe"],
                        "calmar": result["calmar"],
                        "composite_score": result["composite_score"],
                        "total_trades": result["total_trades"],
                        "total_commission": result["total_commission"],
                        "elapsed_sec": round(elapsed, 1),
                    }

                    # Add adaptive diagnostics if available
                    if adaptive and "adaptive_diagnostics" in result:
                        diag = result["adaptive_diagnostics"]
                        row["adaptive_weights"] = json.dumps(diag.get("current_weights", {}))
                        row["ic_ir"] = json.dumps(diag.get("ic_ir", {}))
                        row["obs_count"] = json.dumps(diag.get("obs_count", {}))

                    results.append(row)
                    logger.info("  -> %s: Return=%.2f%% DD=%.2f%% Composite=%.4f",
                               mode, result["total_return_pct"],
                               result["max_drawdown_pct"],
                               result["composite_score"])
                else:
                    logger.warning("  -> FAILED: %s", result.get("reason"))
            except Exception as e:
                logger.error("  -> ERROR: %s", e)
                import traceback
                traceback.print_exc()

    df = pd.DataFrame(results)

    if df.empty:
        logger.error("All backtest runs failed -- no results to compare.")
        return

    # Print comparison
    logger.info("\n" + "=" * 80)
    logger.info("ADAPTIVE vs STATIC COMPARISON")
    logger.info("=" * 80)

    for period in df["period"].unique():
        pdf = df[df["period"] == period]
        logger.info("\n--- %s ---", period)
        for _, row in pdf.iterrows():
            logger.info("  %-8s | Return: %7.2f%% | DD: %6.2f%% | Sortino: %7.4f | Sharpe: %7.4f | Composite: %7.4f",
                        row["mode"], row["total_return_pct"], row["max_drawdown_pct"],
                        row["sortino"], row["sharpe"], row["composite_score"])

        # Compute improvement
        static_row = pdf[pdf["mode"] == "static"]
        adaptive_row = pdf[pdf["mode"] == "adaptive"]
        if not static_row.empty and not adaptive_row.empty:
            ret_diff = adaptive_row.iloc[0]["total_return_pct"] - static_row.iloc[0]["total_return_pct"]
            dd_diff = adaptive_row.iloc[0]["max_drawdown_pct"] - static_row.iloc[0]["max_drawdown_pct"]
            comp_diff = adaptive_row.iloc[0]["composite_score"] - static_row.iloc[0]["composite_score"]
            logger.info("  DELTA    | Return: %+7.2f%% | DD: %+6.2f%% |                              | Composite: %+7.4f",
                        ret_diff, dd_diff, comp_diff)

    # Aggregate
    logger.info("\n--- AGGREGATE ---")
    agg = df.groupby("mode").agg({
        "total_return_pct": "mean",
        "max_drawdown_pct": "mean",
        "composite_score": "mean",
    }).round(4)
    for mode, row in agg.iterrows():
        logger.info("  %-8s | AvgReturn: %7.2f%% | AvgDD: %6.2f%% | AvgComposite: %7.4f",
                    mode, row["total_return_pct"], row["max_drawdown_pct"],
                    row["composite_score"])

    # Save
    os.makedirs("data/csv", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"data/csv/adaptive_comparison_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("\nResults saved to %s", csv_path)

    # Print adaptive diagnostics
    adaptive_rows = df[df["mode"] == "adaptive"]
    for _, row in adaptive_rows.iterrows():
        if "adaptive_weights" in row and pd.notna(row.get("adaptive_weights")):
            logger.info("\n--- Adaptive Weights for %s ---", row["period"])
            try:
                weights = json.loads(row["adaptive_weights"])
                for regime, w in weights.items():
                    regime_name = {0: "BULL", 1: "NEUTRAL", 2: "BEAR"}.get(int(regime), regime)
                    logger.info("  %s: %s", regime_name, w)
            except Exception:
                pass
            try:
                ic_ir = json.loads(row["ic_ir"])
                logger.info("  IC_IR scores:")
                for regime, scores in ic_ir.items():
                    regime_name = {0: "BULL", 1: "NEUTRAL", 2: "BEAR"}.get(int(regime), regime)
                    logger.info("    %s: %s", regime_name, scores)
            except Exception:
                pass


if __name__ == "__main__":
    run_comparison()
