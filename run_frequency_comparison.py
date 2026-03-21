"""
Frequency Comparison — Test different rebalance frequencies.
============================================================
Runs backtests at 1h, 2h, 4h, 5h, 10h, 12h rebalance intervals
over a 5-month window and compares composite scores.

Usage:
    python run_frequency_comparison.py
    python run_frequency_comparison.py --start 2024-10-01 --end 2025-03-15
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd

import config as cfg

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_single_backtest(rebalance_hours: int, start: str, end: str,
                        capital: float = None) -> dict:
    """Run a single backtest with the given rebalance frequency.

    Adjusts regime hysteresis and stop cooldown proportionally.
    """
    from main import BacktestEngine
    from signals import TrendRegimeDetector

    # Scale regime hysteresis: target ~24h hold regardless of frequency
    min_hold = max(2, round(24 / rebalance_hours))

    # Scale stop cooldown: target ~24h cooldown
    stop_cooldown_cycles = max(2, round(24 / rebalance_hours))

    logger.info("=" * 70)
    logger.info("FREQUENCY TEST: rebalance every %dh (MIN_HOLD=%d, COOLDOWN=%d)",
                rebalance_hours, min_hold, stop_cooldown_cycles)
    logger.info("=" * 70)

    engine = BacktestEngine(
        start=start,
        end=end,
        initial_capital=capital or cfg.INITIAL_CAPITAL_USD,
        rebalance_hours=rebalance_hours,
    )

    # Override regime hysteresis for this frequency
    engine.alpha_engine.regime_detector._MIN_HOLD = min_hold

    # Override stop cooldown for this frequency
    engine._STOP_COOLDOWN_CYCLES = stop_cooldown_cycles

    result = engine.run()
    result["rebalance_hours"] = rebalance_hours
    result["min_hold_cycles"] = min_hold
    result["stop_cooldown_cycles"] = stop_cooldown_cycles

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare rebalance frequencies")
    parser.add_argument("--start", type=str, default="2024-10-01",
                        help="Backtest start date")
    parser.add_argument("--end", type=str, default="2025-03-15",
                        help="Backtest end date")
    parser.add_argument("--capital", type=float, default=None,
                        help="Initial capital")
    args = parser.parse_args()

    frequencies = [2, 4, 6, 8, 12]
    results = []

    for freq in frequencies:
        try:
            result = run_single_backtest(freq, args.start, args.end, args.capital)
            results.append(result)
            logger.info("  -> %dh: Return=%.2f%%, DD=%.2f%%, Composite=%.4f",
                        freq,
                        result.get("total_return_pct", 0),
                        result.get("max_drawdown_pct", 0),
                        result.get("composite_score", 0))
        except Exception as e:
            logger.error("Failed for %dh rebalance: %s", freq, e, exc_info=True)
            results.append({
                "status": "FAILED",
                "rebalance_hours": freq,
                "error": str(e),
            })

    # Build comparison table
    logger.info("\n" + "=" * 80)
    logger.info("FREQUENCY COMPARISON RESULTS")
    logger.info("=" * 80)

    rows = []
    for r in results:
        if r.get("status") != "OK":
            continue
        rows.append({
            "Freq (h)": r["rebalance_hours"],
            "Return %": r.get("total_return_pct", 0),
            "Max DD %": r.get("max_drawdown_pct", 0),
            "Sortino": r.get("sortino", 0),
            "Sharpe": r.get("sharpe", 0),
            "Calmar": r.get("calmar", 0),
            "Composite": r.get("composite_score", 0),
            "Trades": r.get("total_trades", 0),
            "Rebalances": r.get("rebalance_count", 0),
            "Commission": r.get("total_commission", 0),
        })

    if rows:
        df = pd.DataFrame(rows).sort_values("Composite", ascending=False)
        print("\n" + df.to_string(index=False))

        best = df.iloc[0]
        print(f"\n*** BEST FREQUENCY: {int(best['Freq (h)'])}h ***")
        print(f"    Composite: {best['Composite']:.4f}")
        print(f"    Return: {best['Return %']:.2f}%")
        print(f"    Max DD: {best['Max DD %']:.2f}%")
        print(f"    Sortino: {best['Sortino']:.4f} | Sharpe: {best['Sharpe']:.4f} | Calmar: {best['Calmar']:.4f}")

        # Save results
        out_dir = cfg.CSV_DIR
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "frequency_comparison.csv")
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")

        # Save full JSON
        json_path = os.path.join(out_dir, "frequency_comparison.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Full results saved to {json_path}")
    else:
        print("No successful backtests!")


if __name__ == "__main__":
    main()
