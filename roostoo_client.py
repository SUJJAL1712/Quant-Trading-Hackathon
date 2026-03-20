"""
Roostoo API Client
==================
Handles all communication with the Roostoo mock exchange.
Authentication via HMAC SHA256 signatures.
"""

import hashlib
import hmac
import time
import logging
import requests
from typing import Dict, List, Optional, Any

import config as cfg

logger = logging.getLogger(__name__)

# Retry configuration for transient network failures
MAX_RETRIES = 3
RETRY_DELAYS = [2, 5, 10]  # seconds between retries (exponential-ish backoff)


class RoostooClient:
    """REST client for Roostoo mock exchange API (v3)."""

    def __init__(self, api_key: str = None, api_secret: str = None,
                 base_url: str = None):
        self.api_key = api_key or cfg.ROOSTOO_API_KEY
        self.api_secret = api_secret or cfg.ROOSTOO_API_SECRET
        self.base_url = base_url or cfg.ROOSTOO_BASE_URL
        if not self.api_key or not self.api_secret:
            logger.warning("Roostoo API credentials not set! "
                           "Create a .env file with ROOSTOO_API_KEY and ROOSTOO_API_SECRET. "
                           "Public endpoints will work, but trading will fail.")
        self.session = requests.Session()
        self.session.headers.update({"RST-API-KEY": self.api_key})
        # Cache exchange info
        self._exchange_info: Optional[Dict] = None

    def _timestamp(self) -> int:
        """Current time in milliseconds."""
        return int(time.time() * 1000)

    def _sign(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature for request params."""
        query_string = "&".join(
            f"{k}={params[k]}" for k in sorted(params.keys())
        )
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _get(self, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """Send GET request with retry logic for transient failures."""
        last_exc = None
        for attempt in range(MAX_RETRIES):
            params_copy = dict(params or {})
            params_copy["timestamp"] = self._timestamp()  # fresh timestamp each attempt
            headers = {}
            if signed:
                headers["MSG-SIGNATURE"] = self._sign(params_copy)
            try:
                resp = self.session.get(
                    f"{self.base_url}{endpoint}",
                    params=params_copy,
                    headers=headers,
                    timeout=15,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_exc = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning("GET %s attempt %d/%d failed: %s — retrying in %ds",
                                   endpoint, attempt + 1, MAX_RETRIES, e, delay)
                    time.sleep(delay)
                else:
                    logger.error("GET %s failed after %d attempts: %s",
                                 endpoint, MAX_RETRIES, e)
        raise last_exc

    def _post(self, endpoint: str, data: Dict = None) -> Dict:
        """Send POST request with retry logic for transient failures."""
        last_exc = None
        for attempt in range(MAX_RETRIES):
            data_copy = dict(data or {})
            data_copy["timestamp"] = self._timestamp()  # fresh timestamp each attempt
            headers = {
                "MSG-SIGNATURE": self._sign(data_copy),
                "Content-Type": "application/x-www-form-urlencoded",
            }
            try:
                resp = self.session.post(
                    f"{self.base_url}{endpoint}",
                    data=data_copy,
                    headers=headers,
                    timeout=15,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_exc = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning("POST %s attempt %d/%d failed: %s — retrying in %ds",
                                   endpoint, attempt + 1, MAX_RETRIES, e, delay)
                    time.sleep(delay)
                else:
                    logger.error("POST %s failed after %d attempts: %s",
                                 endpoint, MAX_RETRIES, e)
        raise last_exc

    # ── Public endpoints ──

    def server_time(self) -> int:
        """Get server time in milliseconds."""
        data = self._get("/v3/serverTime")
        return data.get("ServerTime", self._timestamp())

    def exchange_info(self, force_refresh: bool = False) -> Dict:
        """Get exchange configuration: trading pairs, precision, min orders.

        Returns cached data unless force_refresh=True.
        """
        if self._exchange_info is None or force_refresh:
            self._exchange_info = self._get("/v3/exchangeInfo")
        return self._exchange_info

    def get_trading_pairs(self) -> Dict[str, Dict]:
        """Get available trading pairs with their precision and limits."""
        info = self.exchange_info()
        return info.get("TradePairs", {})

    def get_initial_wallet(self) -> Dict[str, float]:
        """Get initial wallet allocation from exchange info."""
        info = self.exchange_info()
        return info.get("InitialWallet", {})

    def ticker(self, pair: str = None) -> Dict:
        """Get market ticker data.

        Args:
            pair: Trading pair (e.g., "BTC/USD"). None for all pairs.

        Returns:
            Ticker dict with MaxBid, MinAsk, LastPrice, Change, volumes.
        """
        params = {}
        if pair:
            params["pair"] = pair
        return self._get("/v3/ticker", params=params)

    def get_all_prices(self) -> Dict[str, float]:
        """Get latest prices for all trading pairs.

        Returns:
            Dict mapping pair name to last price, e.g., {"BTC/USD": 65000.0}
        """
        data = self.ticker()
        prices = {}
        if isinstance(data, dict):
            for pair_name, ticker_data in data.items():
                if isinstance(ticker_data, dict) and "LastPrice" in ticker_data:
                    prices[pair_name] = float(ticker_data["LastPrice"])
        return prices

    # ── Account endpoints (signed) ──

    def balance(self) -> Dict[str, Dict[str, float]]:
        """Get wallet balances.

        Returns:
            Dict like {"BTC": {"Free": 0.5, "Lock": 0.1}, "USD": {...}}
        """
        data = self._get("/v3/balance", signed=True)
        return data.get("Wallet", {})

    def get_portfolio_value(self) -> float:
        """Compute total portfolio value in USD."""
        wallet = self.balance()
        prices = self.get_all_prices()
        total = 0.0
        for coin, amounts in wallet.items():
            free = float(amounts.get("Free", 0))
            locked = float(amounts.get("Lock", 0))
            qty = free + locked
            if qty <= 0:
                continue
            if coin == "USD":
                total += qty
            else:
                pair = f"{coin}/USD"
                if pair in prices:
                    total += qty * prices[pair]
                else:
                    logger.warning("No price for %s, skipping in NAV", coin)
        return total

    def pending_count(self) -> Dict:
        """Get count of pending orders."""
        return self._get("/v3/pending_count", signed=True)

    # ── Trading endpoints (signed) ──

    def place_order(self, pair: str, side: str, quantity: float,
                    order_type: str = "MARKET", price: float = None) -> Dict:
        """Place a new order.

        Args:
            pair: Trading pair (e.g., "BTC/USD")
            side: "BUY" or "SELL"
            quantity: Order quantity (in base asset units)
            order_type: "MARKET" or "LIMIT"
            price: Required for LIMIT orders

        Returns:
            Order response with OrderID, Status, FilledQuantity, etc.
        """
        data = {
            "pair": pair,
            "side": side,
            "type": order_type,
            "quantity": quantity,
        }
        if order_type == "LIMIT" and price is not None:
            data["price"] = price

        logger.info("Placing %s %s %s qty=%.6f price=%s",
                     order_type, side, pair, quantity, price)
        result = self._post("/v3/place_order", data)
        logger.info("Order result: ID=%s Status=%s Filled=%.6f",
                     result.get("OrderID"), result.get("Status"),
                     float(result.get("FilledQuantity", 0)))
        return result

    def place_market_buy(self, pair: str, quantity: float) -> Dict:
        """Place a market buy order."""
        return self.place_order(pair, "BUY", quantity, "MARKET")

    def place_market_sell(self, pair: str, quantity: float) -> Dict:
        """Place a market sell order."""
        return self.place_order(pair, "SELL", quantity, "MARKET")

    def place_limit_buy(self, pair: str, quantity: float, price: float) -> Dict:
        """Place a limit buy order."""
        return self.place_order(pair, "BUY", quantity, "LIMIT", price)

    def place_limit_sell(self, pair: str, quantity: float, price: float) -> Dict:
        """Place a limit sell order."""
        return self.place_order(pair, "SELL", quantity, "LIMIT", price)

    def query_orders(self, order_id: int = None, pair: str = None,
                     pending_only: bool = False, limit: int = 100) -> List[Dict]:
        """Query order history.

        Args:
            order_id: Specific order ID to query
            pair: Filter by trading pair
            pending_only: Only return pending orders
            limit: Max results (default 100)

        Returns:
            List of order dicts
        """
        data = {"limit": limit}
        if order_id:
            data["order_id"] = order_id
        if pair:
            data["pair"] = pair
        if pending_only:
            data["pending_only"] = "TRUE"
        result = self._post("/v3/query_order", data)
        return result if isinstance(result, list) else result.get("OrderMatched", [])

    def cancel_order(self, order_id: int = None, pair: str = None) -> Dict:
        """Cancel an order or all orders for a pair.

        Args:
            order_id: Specific order to cancel
            pair: Cancel all pending orders for this pair
            (If neither provided, cancels ALL pending orders)

        Returns:
            Dict with CanceledList of order IDs
        """
        data = {}
        if order_id:
            data["order_id"] = order_id
        if pair:
            data["pair"] = pair
        logger.info("Canceling orders: order_id=%s pair=%s", order_id, pair)
        return self._post("/v3/cancel_order", data)

    def cancel_all_pending(self) -> Dict:
        """Cancel all pending orders across all pairs."""
        return self.cancel_order()

    # ── Utility methods ──

    def get_pair_precision(self, pair: str) -> tuple:
        """Get price and amount precision for a pair.

        Returns:
            (price_precision, amount_precision) tuple of ints
        """
        pairs = self.get_trading_pairs()
        if pair in pairs:
            info = pairs[pair]
            return (
                int(info.get("PricePrecision", 2)),
                int(info.get("AmountPrecision", 6)),
            )
        return (2, 6)  # safe defaults

    def get_min_order(self, pair: str) -> float:
        """Get minimum order value (USD) for a pair."""
        pairs = self.get_trading_pairs()
        if pair in pairs:
            return float(pairs[pair].get("MiniOrder", 1.0))
        return 1.0

    def round_quantity(self, pair: str, quantity: float) -> float:
        """Round quantity to pair's amount precision."""
        _, amount_precision = self.get_pair_precision(pair)
        return round(quantity, amount_precision)

    def round_price(self, pair: str, price: float) -> float:
        """Round price to pair's price precision."""
        price_precision, _ = self.get_pair_precision(pair)
        return round(price, price_precision)

    def is_exchange_running(self) -> bool:
        """Check if exchange is currently operational."""
        info = self.exchange_info(force_refresh=True)
        return info.get("IsRunning", False)
