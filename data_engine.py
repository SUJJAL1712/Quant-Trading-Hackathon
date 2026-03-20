"""
Data Engine — Crypto Historical & Live Data
=============================================
Fetches historical OHLCV from Binance (free public API).
Fetches live prices from Roostoo ticker endpoint.
Caches data in SQLite to avoid redundant fetches.
"""

import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import requests

import config as cfg

logger = logging.getLogger(__name__)


class BinanceDataClient:
    """Fetch historical OHLCV data from Binance public API (no auth needed)."""

    BASE_URL = cfg.BINANCE_BASE_URL

    def __init__(self):
        self.session = requests.Session()

    def fetch_klines(self, symbol: str, interval: str = "1h",
                     start_time: int = None, end_time: int = None,
                     limit: int = 1000) -> pd.DataFrame:
        """Fetch klines/candlestick data from Binance.

        Args:
            symbol: Binance symbol (e.g., "BTCUSDT")
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
            start_time: Start time in ms (optional)
            end_time: End time in ms (optional)
            limit: Max candles per request (max 1000)

        Returns:
            DataFrame with columns: open, high, low, close, volume, timestamp
        """
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        try:
            resp = self.session.get(
                f"{self.BASE_URL}/api/v3/klines",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error("Binance klines fetch failed for %s: %s", symbol, e)
            return pd.DataFrame()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ])
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)
        df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]]
        df = df.set_index("timestamp")
        return df

    def fetch_history(self, symbol: str, interval: str = "1h",
                      days: int = 30) -> pd.DataFrame:
        """Fetch multiple pages of historical data.

        Args:
            symbol: Binance symbol
            interval: Candle interval
            days: Number of days of history to fetch

        Returns:
            Full DataFrame of OHLCV data
        """
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)

        all_data = []
        current_start = start_time

        while current_start < end_time:
            df = self.fetch_klines(
                symbol, interval,
                start_time=current_start,
                end_time=end_time,
                limit=1000,
            )
            if df.empty:
                break
            all_data.append(df)
            # Move start to after last candle
            last_ts = int(df.index[-1].timestamp() * 1000)
            if last_ts <= current_start:
                break
            current_start = last_ts + 1
            time.sleep(0.1)  # rate limit courtesy

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data)
        result = result[~result.index.duplicated(keep="last")]
        return result.sort_index()


class DataCache:
    """SQLite cache for OHLCV data to avoid redundant API calls."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or cfg.DB_PATH
        self._ensure_tables()

    def _ensure_tables(self):
        """Create tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL,
                    volume REAL, quote_volume REAL,
                    PRIMARY KEY (symbol, interval, timestamp)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_meta (
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    last_fetched TEXT NOT NULL,
                    PRIMARY KEY (symbol, interval)
                )
            """)

    def get_cached(self, symbol: str, interval: str = "1h",
                   since: datetime = None) -> pd.DataFrame:
        """Retrieve cached OHLCV data."""
        query = "SELECT * FROM ohlcv WHERE symbol=? AND interval=?"
        params: list = [symbol, interval]
        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())
        query += " ORDER BY timestamp"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        df = df.drop(columns=["symbol", "interval"], errors="ignore")
        return df

    def save_ohlcv(self, symbol: str, interval: str, df: pd.DataFrame):
        """Save OHLCV data to cache."""
        if df.empty:
            return
        records = df.reset_index()
        records["symbol"] = symbol
        records["interval"] = interval
        records["timestamp"] = records["timestamp"].astype(str)

        with sqlite3.connect(self.db_path) as conn:
            for _, row in records.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO ohlcv
                    (symbol, interval, timestamp, open, high, low, close, volume, quote_volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row["symbol"], row["interval"], row["timestamp"],
                    row.get("open"), row.get("high"), row.get("low"),
                    row.get("close"), row.get("volume"), row.get("quote_volume"),
                ))
            conn.execute("""
                INSERT OR REPLACE INTO cache_meta (symbol, interval, last_fetched)
                VALUES (?, ?, ?)
            """, (symbol, interval, datetime.now(timezone.utc).isoformat()))

    def is_stale(self, symbol: str, interval: str,
                 max_age_hours: float = 1.0) -> bool:
        """Check if cached data is stale."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT last_fetched FROM cache_meta WHERE symbol=? AND interval=?",
                (symbol, interval),
            ).fetchone()
        if not row:
            return True
        last_fetched = datetime.fromisoformat(row[0])
        if last_fetched.tzinfo is None:
            last_fetched = last_fetched.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - last_fetched
        return age > timedelta(hours=max_age_hours)


class DataEngine:
    """Unified data facade: tries cache first, then Binance API.

    Usage:
        engine = DataEngine()
        prices = engine.get_close_prices(coins=["BTC", "ETH"], interval="1h", days=30)
    """

    def __init__(self, db_path: str = None):
        self.binance = BinanceDataClient()
        self.cache = DataCache(db_path)

    def get_ohlcv(self, coin: str, interval: str = "1h",
                  days: int = 30) -> pd.DataFrame:
        """Get OHLCV data for a single coin. Cache-first, then Binance.

        Args:
            coin: Coin name (e.g., "BTC")
            interval: Candle interval (e.g., "1h", "4h", "1d")
            days: Days of history

        Returns:
            DataFrame indexed by timestamp with open/high/low/close/volume columns
        """
        binance_symbol = cfg.ROOSTOO_TO_BINANCE.get(f"{coin}/USD", f"{coin}USDT")

        # Check cache freshness
        if not self.cache.is_stale(binance_symbol, interval):
            since = datetime.now(timezone.utc) - timedelta(days=days)
            cached = self.cache.get_cached(binance_symbol, interval, since=since)
            if not cached.empty and len(cached) > days * 20:
                logger.debug("Using cached data for %s (%d rows)", coin, len(cached))
                return cached

        # Fetch from Binance
        logger.info("Fetching %s %s data from Binance (%d days)", coin, interval, days)
        df = self.binance.fetch_history(binance_symbol, interval, days)
        if not df.empty:
            self.cache.save_ohlcv(binance_symbol, interval, df)
        else:
            # Fall back to cache even if stale
            since = datetime.now(timezone.utc) - timedelta(days=days)
            df = self.cache.get_cached(binance_symbol, interval, since=since)
            if not df.empty:
                logger.warning("Binance fetch failed for %s, using stale cache (%d rows)",
                               coin, len(df))
        return df

    def get_all_ohlcv(self, coins: List[str] = None, interval: str = "1h",
                      days: int = 30) -> Dict[str, pd.DataFrame]:
        """Get OHLCV data for all coins.

        Returns:
            Dict mapping coin name to its OHLCV DataFrame
        """
        coins = coins or cfg.COINS
        result = {}
        for coin in coins:
            df = self.get_ohlcv(coin, interval, days)
            if not df.empty:
                result[coin] = df
            else:
                logger.warning("No data for %s", coin)
        return result

    def get_close_prices(self, coins: List[str] = None, interval: str = "1h",
                         days: int = 30) -> pd.DataFrame:
        """Get close prices for all coins as a single DataFrame.

        Returns:
            DataFrame indexed by timestamp, columns = coin names
        """
        all_data = self.get_all_ohlcv(coins, interval, days)
        if not all_data:
            return pd.DataFrame()
        close_frames = {}
        for coin, df in all_data.items():
            if "close" in df.columns:
                close_frames[coin] = df["close"]
        if not close_frames:
            return pd.DataFrame()
        prices = pd.DataFrame(close_frames)
        prices = prices.sort_index().ffill()
        return prices

    def get_returns(self, coins: List[str] = None, interval: str = "1h",
                    days: int = 30) -> pd.DataFrame:
        """Get log returns for all coins.

        Returns:
            DataFrame of log returns, indexed by timestamp, columns = coin names
        """
        prices = self.get_close_prices(coins, interval, days)
        if prices.empty:
            return prices
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns

    def get_volumes(self, coins: List[str] = None, interval: str = "1h",
                    days: int = 30) -> pd.DataFrame:
        """Get volume data for all coins."""
        all_data = self.get_all_ohlcv(coins, interval, days)
        vol_frames = {}
        for coin, df in all_data.items():
            if "volume" in df.columns:
                vol_frames[coin] = df["volume"]
        if not vol_frames:
            return pd.DataFrame()
        volumes = pd.DataFrame(vol_frames)
        return volumes.sort_index().fillna(0)
