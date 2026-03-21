"""
Frequency Comparison — Backtest different rebalancing frequencies
================================================================
Tests: 1h, 2h, 4h, 5h, 10h, 12h over multiple time periods
(30min not feasible — we only have 1h candle data from Binance)

Usage:
    python frequency_comparison.py
    python frequency_comparison.py --periods short   # Oct-Dec 2024 only
    python frequency_comparison.py --periods long    # Sep 2024 - Mar 2025
"""

import argparse
import json
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("freq_compare")

from main import BacktestEngine

# ── Test configurations ──

FREQUENCIES = [1, 2, 4, 5, 10, 12]  # hours

# Multiple time periods to test robustness
PERIODS = {
    "bear_2024Q4":  ("2024-10-01", "2024-12-31"),   # ~3 months, mixed/bear
    "bull_2025Q1":  ("2025-01-01", "2025-03-15"),   # ~2.5 months, mixed/bull
    "full_6mo":     ("2024-09-15", "2025-03-15"),   # ~6 months, full cycle
}


def run_single_backtest(start: str, end: str, freq_hours: int) -> Dict:
    """Run a single backtest with given frequency."""
    try:
        engine = BacktestEngine(
            start=start,
            end=end,
            rebalance_hours=freq_hours,
        )
        result = engine.run()
        return result
    except Exception as e:
        logger.error("Backtest failed (freq=%dh, %s->%s): %s", freq_hours, start, end, e)
        return {"status": "FAILED", "reason": str(e)}


def run_comparison(periods: Dict[str, Tuple[str, str]] = None,
                   frequencies: List[int] = None) -> pd.DataFrame:
    """Run all frequency x period combinations and collect results."""
    if periods is None:
        periods = PERIODS
    if frequencies is None:
        frequencies = FREQUENCIES

    results = []
    total_runs = len(periods) * len(frequencies)
    run_num = 0

    for period_name, (start, end) in periods.items():
        for freq in frequencies:
            run_num += 1
            logger.info("=" * 60)
            logger.info("RUN %d/%d: %s @ %dh rebalance", run_num, total_runs, period_name, freq)
            logger.info("=" * 60)

            t0 = time.time()
            result = run_single_backtest(start, end, freq)
            elapsed = time.time() - t0

            if result.get("status") == "OK":
                results.append({
                    "period": period_name,
                    "start": start,
                    "end": end,
                    "freq_hours": freq,
                    "total_return_pct": result["total_return_pct"],
                    "max_drawdown_pct": result["max_drawdown_pct"],
                    "sortino": result["sortino"],
                    "sharpe": result["sharpe"],
                    "calmar": result["calmar"],
                    "composite_score": result["composite_score"],
                    "rebalance_count": result["rebalance_count"],
                    "total_trades": result["total_trades"],
                    "total_commission": result["total_commission"],
                    "final_nav": result["final_nav"],
                    "elapsed_sec": round(elapsed, 1),
                    "output_dir": result.get("output_dir", ""),
                })
                logger.info("  -> Return: %.2f%% | DD: %.2f%% | Composite: %.4f | Time: %.1fs",
                           result["total_return_pct"], result["max_drawdown_pct"],
                           result["composite_score"], elapsed)
            else:
                logger.warning("  -> FAILED: %s", result.get("reason", "unknown"))
                results.append({
                    "period": period_name,
                    "start": start,
                    "end": end,
                    "freq_hours": freq,
                    "total_return_pct": float("nan"),
                    "max_drawdown_pct": float("nan"),
                    "sortino": float("nan"),
                    "sharpe": float("nan"),
                    "calmar": float("nan"),
                    "composite_score": float("nan"),
                    "rebalance_count": 0,
                    "total_trades": 0,
                    "total_commission": 0,
                    "final_nav": float("nan"),
                    "elapsed_sec": round(elapsed, 1),
                    "output_dir": "",
                })

    df = pd.DataFrame(results)
    return df


def analyze_results(df: pd.DataFrame) -> Dict:
    """Analyze results and pick the best frequency."""
    logger.info("\n" + "=" * 80)
    logger.info("FREQUENCY COMPARISON RESULTS")
    logger.info("=" * 80)

    # Per-period results
    for period in df["period"].unique():
        pdf = df[df["period"] == period].sort_values("composite_score", ascending=False)
        logger.info("\n--- %s (%s to %s) ---", period, pdf.iloc[0]["start"], pdf.iloc[0]["end"])
        logger.info("%-6s | %8s | %8s | %8s | %8s | %8s | %10s | %6s | %6s",
                    "Freq", "Return%", "MaxDD%", "Sortino", "Sharpe", "Calmar",
                    "Composite", "Trades", "Comm$")
        for _, row in pdf.iterrows():
            logger.info("%-6s | %8.2f | %8.2f | %8.4f | %8.4f | %8.4f | %10.4f | %6d | %6.1f",
                        f"{int(row['freq_hours'])}h",
                        row["total_return_pct"], row["max_drawdown_pct"],
                        row["sortino"], row["sharpe"], row["calmar"],
                        row["composite_score"], row["total_trades"],
                        row["total_commission"])

    # Aggregate: average composite score across all periods
    logger.info("\n--- AGGREGATE (avg across periods) ---")
    agg = df.groupby("freq_hours").agg({
        "total_return_pct": "mean",
        "max_drawdown_pct": "mean",
        "sortino": "mean",
        "sharpe": "mean",
        "calmar": "mean",
        "composite_score": "mean",
        "total_trades": "mean",
        "total_commission": "mean",
    }).round(4)

    # Also compute consistency: std of composite across periods
    composite_std = df.groupby("freq_hours")["composite_score"].std().round(4)
    agg["composite_std"] = composite_std
    agg["risk_adj_composite"] = agg["composite_score"] - 0.5 * agg["composite_std"].fillna(0)

    agg = agg.sort_values("risk_adj_composite", ascending=False)

    logger.info("%-6s | %8s | %8s | %10s | %10s | %12s",
                "Freq", "AvgRet%", "AvgDD%", "AvgComposite", "StdComposite", "RiskAdjComp")
    for freq, row in agg.iterrows():
        logger.info("%-6s | %8.2f | %8.2f | %10.4f | %10.4f | %12.4f",
                    f"{int(freq)}h",
                    row["total_return_pct"], row["max_drawdown_pct"],
                    row["composite_score"], row["composite_std"],
                    row["risk_adj_composite"])

    best_freq = int(agg.index[0])
    best_composite = agg.iloc[0]["risk_adj_composite"]

    logger.info("\n** BEST FREQUENCY: %dh (risk-adjusted composite: %.4f)", best_freq, best_composite)

    return {
        "best_frequency_hours": best_freq,
        "best_risk_adj_composite": float(best_composite),
        "aggregate": agg.to_dict(),
        "all_results": df.to_dict(orient="records"),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare rebalancing frequencies")
    parser.add_argument("--periods", type=str, default="all",
                       choices=["all", "short", "long", "bear", "bull"],
                       help="Which periods to test")
    parser.add_argument("--freq", type=str, default=None,
                       help="Comma-separated frequencies to test (e.g., '1,4,12')")
    args = parser.parse_args()

    # Select periods
    if args.periods == "all":
        periods = PERIODS
    elif args.periods == "short":
        periods = {"bear_2024Q4": PERIODS["bear_2024Q4"]}
    elif args.periods == "long":
        periods = {"full_6mo": PERIODS["full_6mo"]}
    elif args.periods == "bear":
        periods = {"bear_2024Q4": PERIODS["bear_2024Q4"]}
    elif args.periods == "bull":
        periods = {"bull_2025Q1": PERIODS["bull_2025Q1"]}

    # Select frequencies
    if args.freq:
        frequencies = [int(f) for f in args.freq.split(",")]
    else:
        frequencies = FREQUENCIES

    logger.info("Testing frequencies: %s", frequencies)
    logger.info("Testing periods: %s", list(periods.keys()))
    logger.info("Total runs: %d", len(periods) * len(frequencies))

    # Run comparison
    df = run_comparison(periods, frequencies)

    # Save raw results
    os.makedirs("data/csv", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"data/csv/freq_comparison_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Raw results saved to %s", csv_path)

    # Analyze
    analysis = analyze_results(df)

    # Save analysis
    json_path = f"data/csv/freq_comparison_{timestamp}_analysis.json"
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info("Analysis saved to %s", json_path)

    print(f"\nBest frequency: {analysis['best_frequency_hours']}h")
    print(f"Risk-adjusted composite: {analysis['best_risk_adj_composite']:.4f}")


if __name__ == "__main__":
    main()
