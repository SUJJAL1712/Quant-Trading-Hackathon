"""
Export monthly OHLCV CSVs for all assets in the Roostoo universe.

The script uses Roostoo exchange info to discover trade pairs when available,
then fetches historical candles from Binance because this repo only implements
historical market data on the Binance side.

Output layout:
    data/monthly_ohlcv/
        2025-09/
            BTC_USD_1h.csv
            ETH_USD_1h.csv
        2025-10/
            BTC_USD_1h.csv
            ...
        export_manifest.json
        export_manifest.csv

Usage:
    python export_monthly_ohlcv.py --start 2024-01-01
    python export_monthly_ohlcv.py --start 2024-01-01 --end 2026-03-19
    python export_monthly_ohlcv.py --pairs BTC/USD,ETH/USD,SOL/USD
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import config as cfg
from data_engine import BinanceDataClient
from roostoo_client import RoostooClient

logger = logging.getLogger("monthly_ohlcv_export")

DEFAULT_INTERVALS = ["1h", "4h", "1d"]
DEFAULT_OUT_DIR = Path("data") / "monthly_ohlcv"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def to_utc_ts(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def discover_pairs() -> List[str]:
    """Discover tradable pairs from Roostoo, with config fallback."""
    try:
        client = RoostooClient()
        pairs = sorted(
            pair for pair in client.get_trading_pairs().keys()
            if isinstance(pair, str) and pair.endswith("/USD")
        )
        if pairs:
            logger.info("Discovered %d Roostoo pairs from exchangeInfo", len(pairs))
            return pairs
    except Exception as exc:
        logger.warning("Roostoo pair discovery failed, falling back to config: %s", exc)

    fallback = sorted(cfg.TRADING_PAIRS)
    logger.info("Using %d configured pairs from config.py", len(fallback))
    return fallback


def candidate_symbols(pair: str) -> List[str]:
    """Best-effort Binance symbol candidates for a Roostoo pair."""
    coin = pair.split("/")[0]
    candidates = []
    for symbol in [
        cfg.ROOSTOO_TO_BINANCE.get(pair),
        f"{coin}USDT",
        f"{coin}BUSD",
    ]:
        if symbol and symbol not in candidates:
            candidates.append(symbol)
    return candidates


def fetch_symbol_range(
    client: BinanceDataClient,
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch a precise historical range with pagination."""
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    all_chunks = []
    current_start = start_ms

    while current_start < end_ms:
        df = client.fetch_klines(
            symbol=symbol,
            interval=interval,
            start_time=current_start,
            end_time=end_ms,
            limit=1000,
        )
        if df.empty:
            break

        all_chunks.append(df)
        last_ms = int(df.index[-1].timestamp() * 1000)
        if last_ms <= current_start:
            break
        current_start = last_ms + 1
        time.sleep(0.05)

    if not all_chunks:
        return pd.DataFrame()

    result = pd.concat(all_chunks)
    result = result[~result.index.duplicated(keep="last")]
    result = result.sort_index()
    result = result.loc[(result.index >= start) & (result.index <= end)]
    return result


def fetch_best_history(
    client: BinanceDataClient,
    pair: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    intervals: List[str],
) -> Tuple[Optional[str], Optional[str], pd.DataFrame]:
    """Prefer highest-frequency available data, then fall back."""
    symbols = candidate_symbols(pair)
    for interval in intervals:
        for symbol in symbols:
            df = fetch_symbol_range(client, symbol, interval, start, end)
            if not df.empty:
                return symbol, interval, df
    return None, None, pd.DataFrame()


def save_monthly_csvs(
    df: pd.DataFrame,
    pair: str,
    binance_symbol: str,
    interval: str,
    out_dir: Path,
) -> List[str]:
    """Split a candle DataFrame into one CSV per month."""
    safe_pair = pair.replace("/", "_")
    export = df.reset_index().rename(columns={"index": "timestamp"})
    export["pair"] = pair
    export["binance_symbol"] = binance_symbol
    export["interval"] = interval
    export["source"] = "binance"
    export["month"] = export["timestamp"].dt.strftime("%Y-%m")

    written_files = []
    for month, month_df in export.groupby("month", sort=True):
        month_dir = out_dir / month
        month_dir.mkdir(parents=True, exist_ok=True)
        out_path = month_dir / f"{safe_pair}_{interval}.csv"
        month_df.drop(columns=["month"]).to_csv(out_path, index=False)
        written_files.append(str(out_path))

    return written_files


def parse_pairs_arg(pairs_arg: Optional[str]) -> Optional[List[str]]:
    if not pairs_arg:
        return None
    pairs = [item.strip() for item in pairs_arg.split(",") if item.strip()]
    return sorted(set(pairs)) or None


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Export hourly or best-available OHLCV into monthly CSV files."
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2024-01-01",
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--intervals",
        type=str,
        default="1h,4h,1d",
        help="Comma-separated interval preference order.",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Optional comma-separated pair override, e.g. BTC/USD,ETH/USD",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for monthly CSV exports.",
    )
    args = parser.parse_args()

    start = to_utc_ts(args.start)
    end = to_utc_ts(args.end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    if end <= start:
        raise SystemExit("--end must be after --start")

    intervals = [item.strip() for item in args.intervals.split(",") if item.strip()]
    if not intervals:
        intervals = DEFAULT_INTERVALS

    requested_pairs = parse_pairs_arg(args.pairs)
    pairs = requested_pairs or discover_pairs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting %d pairs from %s to %s with interval preference %s",
        len(pairs),
        start.date(),
        end.date(),
        intervals,
    )

    binance = BinanceDataClient()
    manifest: List[Dict] = []

    for idx, pair in enumerate(pairs, start=1):
        logger.info("[%d/%d] Exporting %s", idx, len(pairs), pair)
        try:
            symbol, interval, df = fetch_best_history(
                client=binance,
                pair=pair,
                start=start,
                end=end,
                intervals=intervals,
            )
            if df.empty or symbol is None or interval is None:
                logger.warning("No historical data found for %s", pair)
                manifest.append({
                    "pair": pair,
                    "status": "no_data",
                    "binance_symbol": None,
                    "interval": None,
                    "rows": 0,
                    "start": None,
                    "end": None,
                    "files": [],
                })
                continue

            files = save_monthly_csvs(
                df=df,
                pair=pair,
                binance_symbol=symbol,
                interval=interval,
                out_dir=out_dir,
            )
            manifest.append({
                "pair": pair,
                "status": "ok",
                "binance_symbol": symbol,
                "interval": interval,
                "rows": int(len(df)),
                "start": str(df.index.min()),
                "end": str(df.index.max()),
                "files": files,
            })
            logger.info(
                "Saved %d rows for %s via %s at %s into %d monthly files",
                len(df),
                pair,
                symbol,
                interval,
                len(files),
            )
        except Exception as exc:
            logger.exception("Export failed for %s: %s", pair, exc)
            manifest.append({
                "pair": pair,
                "status": "failed",
                "binance_symbol": None,
                "interval": None,
                "rows": 0,
                "start": None,
                "end": None,
                "files": [],
                "error": str(exc),
            })

    manifest_json = out_dir / "export_manifest.json"
    manifest_csv = out_dir / "export_manifest.csv"
    with manifest_json.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    pd.DataFrame(manifest).to_csv(manifest_csv, index=False)

    ok_count = sum(1 for row in manifest if row["status"] == "ok")
    no_data_count = sum(1 for row in manifest if row["status"] == "no_data")
    failed_count = sum(1 for row in manifest if row["status"] == "failed")
    logger.info(
        "Export complete: ok=%d no_data=%d failed=%d | manifest=%s",
        ok_count,
        no_data_count,
        failed_count,
        manifest_json,
    )


if __name__ == "__main__":
    main()
