"""
Strategy Comparison — PDF Strategies vs Existing System
========================================================
Implements backtesting for the PDF-recommended strategies and compares
them against the existing signal stack across multiple periods and
rebalancing frequencies.

Strategies tested:
- baseline:      Existing 5-signal adaptive alpha engine (current system)
- tsmom_only:    Vol-managed time-series momentum (PDF core)
- tsmom_funding: TSMOM + funding rate crowdedness overlay
- tsmom_pairs:   TSMOM + distance-based pairs trading
- full_stack:    All PDF strategies combined

Usage:
    python strategy_comparison.py                    # Full comparison
    python strategy_comparison.py --quick            # Quick test (1 period, 2 strategies)
    python strategy_comparison.py --strategies tsmom_only baseline
    python strategy_comparison.py --frequencies 2 4
    python strategy_comparison.py --periods Q4_2024
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("strategy_compare")

import config as cfg
from main import BacktestEngine
from pdf_strategies import PDFAlphaEngine, FundingRateOverlay
from optimizer import PortfolioOptimizer
from risk_manager import RiskManager

# ── Configuration ──

PERIODS = {
    "Q4_2024":   ("2024-10-01", "2024-12-31"),
    "Q1_2025":   ("2025-01-01", "2025-03-15"),
    "full_6mo":  ("2024-09-15", "2025-03-15"),
}

FREQUENCIES = [1, 2, 4, 6, 12]

STRATEGIES = ["baseline", "tsmom_only", "tsmom_funding", "tsmom_pairs", "full_stack"]

# PDF-recommended risk overrides
PDF_RISK_OVERRIDES = {
    "target_volatility": 0.20,        # 20% ann vol (vs current 40%)
    "dd_deleverage_start": 0.08,      # start at 8% DD (vs 12%)
    "dd_deleverage_full": 0.15,       # full at 15% DD (vs 22%)
    "max_single_position_pct": 0.15,  # 15% max (vs 20%)
    "max_holdings": 8,                # 8 positions (vs 10)
}


# ── PDF Backtest Engine ──

class PDFBacktestEngine(BacktestEngine):
    """BacktestEngine variant that uses PDF strategy alpha engine.

    Overrides the alpha computation while reusing all the existing
    backtest infrastructure (data fetching, execution, risk management).
    """

    def __init__(self, start: str, end: str, strategy_mode: str = "tsmom_only",
                 initial_capital: float = None, rebalance_hours: int = None,
                 target_ann_vol: float = 0.20):
        # Initialize base with adaptive=False (we use our own alpha engine)
        super().__init__(
            start=start, end=end,
            initial_capital=initial_capital,
            rebalance_hours=rebalance_hours,
            adaptive=False,
        )

        # Replace alpha engine with PDF strategy engine
        self.alpha_engine = PDFAlphaEngine(
            mode=strategy_mode,
            rebalance_hours=self.rebalance_hours,
            target_ann_vol=target_ann_vol,
        )
        self.strategy_mode = strategy_mode

        # Apply PDF risk parameter overrides
        self._apply_risk_overrides()

        # Pre-fetch funding data if needed
        self._needs_funding = "funding" in strategy_mode or strategy_mode == "full_stack"

    def _apply_risk_overrides(self):
        """Apply PDF-recommended risk parameters."""
        cfg.RISK.target_volatility = PDF_RISK_OVERRIDES["target_volatility"]
        cfg.RISK.dd_deleverage_start = PDF_RISK_OVERRIDES["dd_deleverage_start"]
        cfg.RISK.dd_deleverage_full = PDF_RISK_OVERRIDES["dd_deleverage_full"]
        cfg.CONSTRAINTS.max_single_position_pct = PDF_RISK_OVERRIDES["max_single_position_pct"]
        cfg.CONSTRAINTS.max_holdings = PDF_RISK_OVERRIDES["max_holdings"]

    def run(self) -> Dict:
        """Run backtest, pre-fetching funding data if needed."""
        # Fetch funding data for the backtest period
        if self._needs_funding:
            logger.info("Fetching funding rate data for backtest period...")
            funding_fetcher = FundingRateOverlay()
            # Only fetch for major coins that have Binance futures
            futures_coins = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE",
                           "ADA", "AVAX", "LINK", "DOT", "SUI", "NEAR"]
            available = [c for c in futures_coins if c in cfg.COINS]
            funding_data = funding_fetcher.fetch_funding_for_backtest(
                available, self.start - timedelta(days=7), self.end)
            self.alpha_engine.set_funding_data(funding_data)
            logger.info("Fetched funding data for %d coins", len(funding_data))

        # Run the base backtest
        return super().run()


def _restore_config():
    """Restore config to defaults after PDF overrides."""
    cfg.RISK.target_volatility = 0.40
    cfg.RISK.dd_deleverage_start = 0.12
    cfg.RISK.dd_deleverage_full = 0.22
    cfg.CONSTRAINTS.max_single_position_pct = 0.20
    cfg.CONSTRAINTS.max_holdings = 10


# ── Comparison Runner ──

def run_single(strategy: str, start: str, end: str,
               rebalance_hours: int) -> Dict:
    """Run a single backtest for a given strategy/period/frequency."""
    try:
        if strategy == "baseline":
            _restore_config()
            engine = BacktestEngine(
                start=start, end=end,
                rebalance_hours=rebalance_hours,
                adaptive=True,
            )
        else:
            engine = PDFBacktestEngine(
                start=start, end=end,
                strategy_mode=strategy,
                rebalance_hours=rebalance_hours,
            )

        result = engine.run()
        _restore_config()  # always restore after
        return result

    except Exception as e:
        _restore_config()
        logger.error("Backtest failed (%s, %dh): %s", strategy, rebalance_hours, e)
        return {"status": "FAILED", "reason": str(e)}


def run_comparison(strategies: List[str] = None,
                   periods: Dict[str, Tuple[str, str]] = None,
                   frequencies: List[int] = None) -> pd.DataFrame:
    """Run full comparison matrix: strategy × period × frequency."""
    if strategies is None:
        strategies = STRATEGIES
    if periods is None:
        periods = PERIODS
    if frequencies is None:
        frequencies = FREQUENCIES

    results = []
    total_runs = len(strategies) * len(periods) * len(frequencies)
    run_num = 0

    for period_name, (start, end) in periods.items():
        for freq in frequencies:
            for strategy in strategies:
                run_num += 1
                logger.info("=" * 60)
                logger.info("RUN %d/%d: %s | %s | %dh rebalance",
                           run_num, total_runs, period_name, strategy, freq)
                logger.info("=" * 60)

                t0 = time.time()
                result = run_single(strategy, start, end, freq)
                elapsed = time.time() - t0

                if result.get("status") == "OK":
                    row = {
                        "period": period_name,
                        "strategy": strategy,
                        "frequency_h": freq,
                        "total_return_pct": result["total_return_pct"],
                        "max_drawdown_pct": result["max_drawdown_pct"],
                        "sortino": result["sortino"],
                        "sharpe": result["sharpe"],
                        "calmar": result["calmar"],
                        "composite_score": result["composite_score"],
                        "total_trades": result["total_trades"],
                        "rebalance_count": result["rebalance_count"],
                        "total_commission": result["total_commission"],
                        "elapsed_sec": round(elapsed, 1),
                    }
                    results.append(row)
                    logger.info("  -> Return: %.2f%% | MaxDD: %.2f%% | Composite: %.4f | %.1fs",
                               row["total_return_pct"], row["max_drawdown_pct"],
                               row["composite_score"], elapsed)
                else:
                    logger.warning("  -> FAILED: %s", result.get("reason", "unknown"))
                    results.append({
                        "period": period_name,
                        "strategy": strategy,
                        "frequency_h": freq,
                        "total_return_pct": np.nan,
                        "max_drawdown_pct": np.nan,
                        "sortino": np.nan,
                        "sharpe": np.nan,
                        "calmar": np.nan,
                        "composite_score": np.nan,
                        "total_trades": 0,
                        "rebalance_count": 0,
                        "total_commission": 0,
                        "elapsed_sec": round(elapsed, 1),
                    })

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame):
    """Print a formatted comparison summary."""
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON RESULTS")
    print("=" * 80)

    # Summary by strategy (averaged across periods and frequencies)
    print("\n--- Average Composite Score by Strategy ---")
    strategy_avg = df.groupby("strategy")[["composite_score", "total_return_pct",
                                            "max_drawdown_pct", "sortino",
                                            "sharpe", "calmar"]].mean()
    strategy_avg = strategy_avg.sort_values("composite_score", ascending=False)
    print(strategy_avg.to_string(float_format="%.4f"))

    # Summary by strategy × frequency
    print("\n--- Composite Score by Strategy × Frequency ---")
    pivot = df.pivot_table(
        values="composite_score",
        index="strategy",
        columns="frequency_h",
        aggfunc="mean",
    )
    pivot = pivot.sort_values(pivot.columns[0], ascending=False)
    print(pivot.to_string(float_format="%.4f"))

    # Summary by strategy × period
    print("\n--- Composite Score by Strategy × Period ---")
    pivot2 = df.pivot_table(
        values="composite_score",
        index="strategy",
        columns="period",
        aggfunc="mean",
    )
    print(pivot2.to_string(float_format="%.4f"))

    # Best configuration
    if not df.empty and df["composite_score"].notna().any():
        best = df.loc[df["composite_score"].idxmax()]
        print(f"\n*** BEST CONFIGURATION ***")
        print(f"  Strategy:   {best['strategy']}")
        print(f"  Period:     {best['period']}")
        print(f"  Frequency:  {best['frequency_h']}h")
        print(f"  Composite:  {best['composite_score']:.4f}")
        print(f"  Return:     {best['total_return_pct']:.2f}%")
        print(f"  Max DD:     {best['max_drawdown_pct']:.2f}%")
        print(f"  Sortino:    {best['sortino']:.4f}")
        print(f"  Sharpe:     {best['sharpe']:.4f}")
        print(f"  Calmar:     {best['calmar']:.4f}")

    print("=" * 80)


def save_results(df: pd.DataFrame) -> str:
    """Save comparison results to CSV."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg.CSV_DIR, f"strategy_comparison_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "comparison_results.csv")
    df.to_csv(csv_path, index=False)

    # Save summary JSON
    summary = {
        "run_id": run_id,
        "strategies": df["strategy"].unique().tolist(),
        "periods": df["period"].unique().tolist(),
        "frequencies": df["frequency_h"].unique().tolist(),
        "total_runs": len(df),
        "successful_runs": df["composite_score"].notna().sum(),
    }

    if df["composite_score"].notna().any():
        best = df.loc[df["composite_score"].idxmax()]
        summary["best"] = {
            "strategy": best["strategy"],
            "period": best["period"],
            "frequency_h": int(best["frequency_h"]),
            "composite_score": round(best["composite_score"], 4),
        }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Results saved to %s", out_dir)
    return out_dir


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="Compare PDF strategies vs existing system")
    parser.add_argument("--strategies", nargs="+", default=None,
                        choices=STRATEGIES + ["all"],
                        help="Strategies to test (default: all)")
    parser.add_argument("--periods", nargs="+", default=None,
                        help="Period names to test (Q4_2024, Q1_2025, full_6mo)")
    parser.add_argument("--frequencies", nargs="+", type=int, default=None,
                        help="Rebalance frequencies in hours (default: 1 2 4 6 12)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 period, baseline + tsmom_only, 2h + 4h")
    args = parser.parse_args()

    if args.quick:
        strategies = ["baseline", "tsmom_only"]
        periods = {"Q4_2024": PERIODS["Q4_2024"]}
        frequencies = [2, 4]
    else:
        strategies = args.strategies if args.strategies and "all" not in args.strategies else STRATEGIES
        if args.periods:
            periods = {k: v for k, v in PERIODS.items() if k in args.periods}
        else:
            periods = PERIODS
        frequencies = args.frequencies or FREQUENCIES

    logger.info("Strategy comparison: %d strategies × %d periods × %d frequencies = %d runs",
               len(strategies), len(periods), len(frequencies),
               len(strategies) * len(periods) * len(frequencies))

    df = run_comparison(strategies=strategies, periods=periods, frequencies=frequencies)

    print_summary(df)
    out_dir = save_results(df)
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
