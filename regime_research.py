"""
Research harness for crypto regime detection.

Runs detector-quality studies and downstream portfolio backtests using the
locally exported monthly OHLCV archive in `data/monthly_ohlcv`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

import config as cfg
from main import BacktestEngine
from regime_models import FeatureScoreRegimeDetector, UnsupervisedFeatureRegimeDetector
from signals import TrendRegimeDetector

logger = logging.getLogger("regime_research")

PERIODS: Dict[str, Tuple[str, str]] = {
    "Q4_2024": ("2024-10-01", "2024-12-31"),
    "Q1_2025": ("2025-01-01", "2025-03-15"),
    "MAR_2026": ("2026-03-01", "2026-03-18"),
}


class MonthlyOHLCVLoader:
    """Load monthly CSV exports into a single hourly panel."""

    def __init__(self, base_dir: str = "data/monthly_ohlcv"):
        self.base_dir = Path(base_dir)

    def discover_stable_coins(self, min_months: int = 24) -> List[str]:
        counts: Counter = Counter()
        for path in self.base_dir.glob("20*/*_USD_1h.csv"):
            coin = path.name.split("_USD_1h.csv")[0]
            counts[coin] += 1
        return sorted([coin for coin, n_months in counts.items() if n_months >= min_months])

    def load_panel(
        self,
        coins: Iterable[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        interval: str = "1h",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")

        close_frames = {}
        volume_frames = {}
        for coin in coins:
            parts = []
            pattern = f"20*/{coin}_USD_{interval}.csv"
            for path in sorted(self.base_dir.glob(pattern)):
                df = pd.read_csv(path, parse_dates=["timestamp"])
                if df.empty:
                    continue
                df = df[["timestamp", "close", "volume"]].set_index("timestamp").sort_index()
                df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
                if not df.empty:
                    parts.append(df)
            if not parts:
                continue
            merged = pd.concat(parts)
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            close_frames[coin] = merged["close"].astype(float)
            volume_frames[coin] = merged["volume"].astype(float)

        # Match production backtest: only fill tiny internal gaps, not full ffill
        prices = pd.DataFrame(close_frames).sort_index().ffill(limit=2)
        volumes = pd.DataFrame(volume_frames).sort_index().fillna(0.0)
        return prices, volumes


class LocalCSVBacktestEngine(BacktestEngine):
    """Backtest engine that uses the local monthly CSV archive."""

    def __init__(self, *args, loader: MonthlyOHLCVLoader | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loader = loader or MonthlyOHLCVLoader()

    def _fetch_historical_range(self, coins, start, end, interval="1h") -> pd.DataFrame:
        prices, volumes = self.loader.load_panel(coins=coins, start=start, end=end, interval=interval)
        self._volume_cache = {coin: volumes[coin] for coin in volumes.columns}
        return prices


DetectorFactory = Callable[[int], object]


def detector_factories() -> Dict[str, DetectorFactory]:
    return {
        "legacy_trend": lambda rebalance_hours: TrendRegimeDetector(rebalance_hours=rebalance_hours),
        "hmm_return": lambda rebalance_hours: UnsupervisedFeatureRegimeDetector(
            model_kind="hmm",
            feature_set="return_only",
            n_components=3,
            rebalance_hours=rebalance_hours,
            train_hours=24 * 60,
            retrain_hours=24,
        ),
        "hmm_trend_vol": lambda rebalance_hours: UnsupervisedFeatureRegimeDetector(
            model_kind="hmm",
            feature_set="trend_vol",
            n_components=4,
            rebalance_hours=rebalance_hours,
            train_hours=24 * 90,
            retrain_hours=24,
        ),
        "hmm_market": lambda rebalance_hours: UnsupervisedFeatureRegimeDetector(
            model_kind="hmm",
            feature_set="market",
            n_components=4,
            rebalance_hours=rebalance_hours,
            train_hours=24 * 90,
            retrain_hours=24,
        ),
        "gmm_market": lambda rebalance_hours: UnsupervisedFeatureRegimeDetector(
            model_kind="gmm",
            feature_set="market",
            n_components=4,
            rebalance_hours=rebalance_hours,
            train_hours=24 * 90,
            retrain_hours=24,
        ),
        "score_market": lambda rebalance_hours: FeatureScoreRegimeDetector(
            feature_set="market",
            rebalance_hours=rebalance_hours,
            train_hours=24 * 90,
        ),
    }


def summarize_regime_series(prices: pd.DataFrame, regimes: pd.Series) -> Dict:
    btc = prices["BTC"].reindex(regimes.index)
    btc_ret = np.log(btc / btc.shift(1))
    df = pd.DataFrame({
        "regime": regimes,
        "fwd_24h": btc.shift(-24) / btc - 1.0,
        "fwd_72h": btc.shift(-72) / btc - 1.0,
        "fwd_vol_24h": btc_ret.rolling(24).std().shift(-24),
    }).dropna()

    rows = {}
    grouped = df.groupby("regime")
    for regime, block in grouped:
        rows[int(regime)] = {
            "count": int(len(block)),
            "fwd_24h_mean": round(float(block["fwd_24h"].mean()), 6),
            "fwd_72h_mean": round(float(block["fwd_72h"].mean()), 6),
            "fwd_24h_pos_rate": round(float((block["fwd_24h"] > 0).mean()), 4),
            "fwd_72h_pos_rate": round(float((block["fwd_72h"] > 0).mean()), 4),
            "future_vol_24h": round(float(block["fwd_vol_24h"].mean()), 6),
        }

    bull_ret = rows.get(0, {}).get("fwd_72h_mean", 0.0)
    bear_ret = rows.get(2, {}).get("fwd_72h_mean", 0.0)
    bull_vol = rows.get(0, {}).get("future_vol_24h", 0.0)
    bear_vol = rows.get(2, {}).get("future_vol_24h", 0.0)

    return {
        "counts": regimes.value_counts().to_dict(),
        "transitions": int((regimes != regimes.shift(1)).sum()),
        "by_regime": rows,
        "quality_score": round((bull_ret - bear_ret) + (bear_vol - bull_vol), 6),
    }


def evaluate_detector_quality(
    detector_factory: DetectorFactory,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    rebalance_hours: int,
    step_hours: int,
) -> Dict:
    detector = detector_factory(rebalance_hours)
    regimes = {}
    step = max(1, step_hours)
    for idx in range(0, len(prices), step):
        ts = prices.index[idx]
        window_prices = prices.iloc[:idx + 1]
        window_volumes = volumes.iloc[:idx + 1] if not volumes.empty else None
        if len(window_prices) < max(cfg.SIGNALS.regime_lookback_hours // 2, 200):
            continue
        regime = detector.detect(window_prices, volumes=window_volumes)
        regimes[ts] = regime

    if not regimes:
        return {"counts": {}, "transitions": 0, "by_regime": {}, "quality_score": -999.0}

    regime_series = pd.Series(regimes).sort_index()
    return summarize_regime_series(prices=prices, regimes=regime_series)


def run_backtest_case(
    detector_name: str,
    detector_factory: DetectorFactory,
    loader: MonthlyOHLCVLoader,
    start: str,
    end: str,
    rebalance_hours: int,
    coins: List[str],
) -> Dict:
    cfg.COINS[:] = coins.copy()
    cfg.ROOSTOO_TO_BINANCE.clear()
    cfg.ROOSTOO_TO_BINANCE.update({f"{coin}/USD": f"{coin}USDT" for coin in cfg.COINS})

    engine = LocalCSVBacktestEngine(
        start=start,
        end=end,
        rebalance_hours=rebalance_hours,
        adaptive=True,
        loader=loader,
    )
    engine.alpha_engine.regime_detector = detector_factory(rebalance_hours)

    result = engine.run()
    return {
        "detector": detector_name,
        "start": start,
        "end": end,
        "rebalance_hours": rebalance_hours,
        "status": result.get("status"),
        "total_return_pct": result.get("total_return_pct"),
        "max_drawdown_pct": result.get("max_drawdown_pct"),
        "sortino": result.get("sortino"),
        "sharpe": result.get("sharpe"),
        "calmar": result.get("calmar"),
        "composite_score": result.get("composite_score"),
        "rebalance_count": result.get("rebalance_count"),
        "total_trades": result.get("total_trades"),
        "output_dir": result.get("output_dir"),
    }


def main():
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.ERROR, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Run regime detector research on local OHLCV exports.")
    parser.add_argument("--detectors", nargs="+", default=["legacy_trend", "hmm_return", "hmm_trend_vol", "hmm_market", "gmm_market", "score_market"])
    parser.add_argument("--periods", nargs="+", default=["Q4_2024", "Q1_2025"])
    parser.add_argument("--rebalance-hours", type=int, default=2)
    parser.add_argument("--quality-step-hours", type=int, default=6)
    parser.add_argument("--min-months", type=int, default=24, help="Minimum monthly files required to include a coin.")
    parser.add_argument("--quality-start", type=str, default="2024-03-15")
    parser.add_argument("--quality-end", type=str, default="2026-03-18")
    parser.add_argument("--quality-only", action="store_true")
    args = parser.parse_args()

    factories = detector_factories()
    unknown = [name for name in args.detectors if name not in factories]
    if unknown:
        raise SystemExit(f"Unknown detector(s): {unknown}")

    loader = MonthlyOHLCVLoader()
    coins = loader.discover_stable_coins(min_months=args.min_months)
    if "BTC" not in coins:
        raise SystemExit("Stable-coin universe does not contain BTC; cannot run regime research.")

    quality_prices, quality_volumes = loader.load_panel(
        coins=coins,
        start=args.quality_start,
        end=args.quality_end,
    )
    if quality_prices.empty:
        raise SystemExit("No local OHLCV data available for the selected quality-study window.")

    quality_rows = []
    logger.info("Evaluating detector quality on %d coins (%s -> %s)", len(coins), args.quality_start, args.quality_end)
    for name in args.detectors:
        logger.info("Quality study: %s", name)
        summary = evaluate_detector_quality(
            detector_factory=factories[name],
            prices=quality_prices,
            volumes=quality_volumes,
            rebalance_hours=args.rebalance_hours,
            step_hours=args.quality_step_hours,
        )
        quality_rows.append({
            "detector": name,
            "quality_score": summary["quality_score"],
            "transitions": summary["transitions"],
            "counts": json.dumps(summary["counts"]),
            "regime_stats": json.dumps(summary["by_regime"]),
        })

    quality_df = pd.DataFrame(quality_rows).sort_values("quality_score", ascending=False)
    print("\nDetector Quality")
    print(quality_df[["detector", "quality_score", "transitions"]].to_string(index=False))

    backtest_rows = []
    if not args.quality_only:
        for period_name in args.periods:
            if period_name not in PERIODS:
                raise SystemExit(f"Unknown period: {period_name}")
            start, end = PERIODS[period_name]
            for name in args.detectors:
                logger.info("Backtest: %s | %s", name, period_name)
                row = run_backtest_case(
                    detector_name=name,
                    detector_factory=factories[name],
                    loader=loader,
                    start=start,
                    end=end,
                    rebalance_hours=args.rebalance_hours,
                    coins=coins,
                )
                row["period"] = period_name
                backtest_rows.append(row)

        backtest_df = pd.DataFrame(backtest_rows)
        if not backtest_df.empty:
            print("\nBacktest Results")
            cols = ["period", "detector", "total_return_pct", "max_drawdown_pct", "composite_score", "total_trades"]
            print(backtest_df[cols].sort_values(["period", "composite_score"], ascending=[True, False]).to_string(index=False))
    else:
        backtest_df = pd.DataFrame()

    out_dir = Path(cfg.CSV_DIR) / f"regime_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    quality_df.to_csv(out_dir / "quality.csv", index=False)
    if not backtest_df.empty:
        backtest_df.to_csv(out_dir / "backtests.csv", index=False)

    summary = {
        "coins": coins,
        "quality_window": {"start": args.quality_start, "end": args.quality_end},
        "rebalance_hours": args.rebalance_hours,
        "quality_leaderboard": quality_df.to_dict(orient="records"),
        "backtests": backtest_df.to_dict(orient="records"),
    }
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nSaved research outputs to {out_dir}")


if __name__ == "__main__":
    main()
