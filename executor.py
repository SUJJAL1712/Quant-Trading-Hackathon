"""
Order Executor — Roostoo Trade Execution
==========================================
Converts target weights -> Roostoo API orders.
Handles order sizing, precision rounding, execution logging,
and cost-aware order type selection (MARKET vs LIMIT).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config as cfg
from roostoo_client import RoostooClient

logger = logging.getLogger(__name__)

# Roostoo commission rates
MAKER_FEE = 0.00008   # 0.008% for limit (maker) orders
TAKER_FEE = 0.00012   # 0.012% for market (taker) orders
AVG_FEE = 0.0001      # 0.01% average (for estimation)


@dataclass
class TradeOrder:
    """Represents a single trade to execute."""
    pair: str               # e.g., "BTC/USD"
    side: str               # "BUY" or "SELL"
    quantity: float         # in base asset units
    order_type: str = "MARKET"
    price: Optional[float] = None  # for LIMIT orders
    reason: str = ""        # why this trade was generated
    urgency: str = "normal" # "urgent" (stop-loss, force exit) or "normal" (rebalance)
    # Filled by execution
    status: str = "PENDING"
    order_id: Optional[int] = None
    fill_price: Optional[float] = None
    fill_quantity: Optional[float] = None
    commission: Optional[float] = None


def estimate_round_trip_cost_bps(order_type: str = "MARKET") -> float:
    """Estimate round-trip cost in basis points (entry + exit).

    Includes commission both ways. MARKET orders also pay spread.
    """
    if order_type == "LIMIT":
        # Maker fee both ways
        return MAKER_FEE * 2 * 10000  # ~1.6 bps
    else:
        # Taker fee both ways + estimated spread cost (~2-5 bps for crypto)
        return TAKER_FEE * 2 * 10000 + 3.0  # ~5.4 bps


class TradeExecutor:
    """Generate and execute orders on Roostoo.

    Pipeline:
    1. Compare current holdings vs target weights
    2. Filter by cost-aware threshold (alpha must exceed round-trip cost)
    3. Choose order type: LIMIT for non-urgent, MARKET for urgent
    4. Generate sell orders first (free up capital)
    5. Generate buy orders (use freed capital)
    6. Execute all orders via Roostoo API
    """

    def __init__(self, client: RoostooClient = None):
        self.client = client or RoostooClient()
        self.trade_log: List[TradeOrder] = []
        self._ticker_cache: Dict[str, Dict] = {}  # pair -> {MaxBid, MinAsk, LastPrice}

    def refresh_ticker_cache(self):
        """Fetch full ticker data with bid/ask spreads for all pairs."""
        try:
            data = self.client.ticker()
            if isinstance(data, dict):
                self._ticker_cache = data
        except Exception as e:
            logger.warning("Failed to refresh ticker cache: %s", e)

    def _get_bid_ask(self, pair: str) -> tuple:
        """Get best bid and ask for a pair.

        Returns:
            (bid, ask) tuple. Falls back to (last_price, last_price) if unavailable.
        """
        ticker = self._ticker_cache.get(pair, {})
        bid = float(ticker.get("MaxBid", 0))
        ask = float(ticker.get("MinAsk", 0))
        last = float(ticker.get("LastPrice", 0))

        # Validate: bid should be < ask, both > 0
        if bid <= 0 or ask <= 0 or bid >= ask:
            return (last, last)
        return (bid, ask)

    def _choose_order_type(self, side: str, pair: str, urgency: str,
                           weight_change: float) -> tuple:
        """Choose between MARKET and LIMIT order.

        Returns:
            (order_type, limit_price_or_None)

        Strategy:
        - Urgent orders (stop-loss, force exit, large rebalance): MARKET
        - Normal rebalances: LIMIT at bid+1tick (buy) or ask-1tick (sell)
          to capture maker rebate
        """
        if urgency == "urgent":
            return ("MARKET", None)

        # For small rebalances, use LIMIT to save on fees
        bid, ask = self._get_bid_ask(pair)

        if bid == 0 or ask == 0 or bid == ask:
            # No spread data available — use MARKET
            return ("MARKET", None)

        # Get price precision for this pair
        price = self.client.round_price(pair, bid)
        tick = 10 ** (-self.client.get_pair_precision(pair)[0])

        if side == "BUY":
            # Post limit buy at bid + 1 tick (aggressive maker)
            limit_price = self.client.round_price(pair, bid + tick)
        else:
            # Post limit sell at ask - 1 tick (aggressive maker)
            limit_price = self.client.round_price(pair, ask - tick)

        if limit_price <= 0:
            return ("MARKET", None)

        return ("LIMIT", limit_price)

    def generate_orders(self, current_holdings: Dict[str, float],
                        target_weights: Dict[str, float],
                        current_prices: Dict[str, float],
                        portfolio_value: float,
                        alpha_scores: Dict[str, float] = None,
                        urgent_coins: set = None) -> List[TradeOrder]:
        """Generate orders to move from current to target allocation.

        Args:
            current_holdings: {coin: quantity} of current positions
            target_weights: {coin: weight} target allocation (0 to 1)
            current_prices: {pair: price} current market prices
            portfolio_value: Total portfolio value in USD
            alpha_scores: {coin: alpha_z_score} for cost-aware filtering
            urgent_coins: set of coins that need urgent execution (stop-loss, etc.)

        Returns:
            List of TradeOrder objects (sells first, then buys)
        """
        sells = []
        buys = []
        urgent_coins = urgent_coins or set()
        alpha_scores = alpha_scores or {}

        min_trade = cfg.REBALANCE.min_trade_value_usd

        for coin in set(list(current_holdings.keys()) + list(target_weights.keys())):
            pair = f"{coin}/USD"
            price = current_prices.get(pair, 0)
            if price <= 0:
                continue

            current_qty = current_holdings.get(coin, 0)
            current_value = current_qty * price
            current_weight = current_value / portfolio_value if portfolio_value > 0 else 0

            target_weight = target_weights.get(coin, 0)
            target_value = target_weight * portfolio_value
            target_qty = target_value / price

            diff_qty = target_qty - current_qty
            diff_value = abs(diff_qty * price)
            force_exit = (target_weight <= 1e-8 and current_qty > 0
                          and current_value >= min_trade)

            # Skip if change is too small
            weight_change = abs(target_weight - current_weight)
            if diff_value < min_trade:
                continue
            if not force_exit and weight_change < cfg.REBALANCE.min_weight_change:
                continue

            # Cost-aware filter: skip trades where expected alpha
            # doesn't justify the round-trip cost
            is_urgent = (coin in urgent_coins or force_exit)
            if not is_urgent and alpha_scores and coin in alpha_scores:
                alpha_z = abs(alpha_scores[coin])
                # Alpha z-score ~ expected excess return in std units
                # Convert to approximate bps: z=1.0 ~ 50bps per rebalance period
                expected_alpha_bps = alpha_z * 50.0
                round_trip_bps = estimate_round_trip_cost_bps("LIMIT")

                # Only trade if expected alpha exceeds cost by at least 2x
                # (to account for estimation error)
                if expected_alpha_bps < round_trip_bps * 2.0 and not force_exit:
                    logger.debug("Skipping %s: alpha %.1f bps < 2x cost %.1f bps",
                                 coin, expected_alpha_bps, round_trip_bps)
                    continue

            # Round quantity to exchange precision
            rounded_qty = self.client.round_quantity(pair, abs(diff_qty))
            if rounded_qty <= 0:
                continue

            # Determine urgency
            urgency = "urgent" if is_urgent else "normal"

            # Choose order type
            side = "SELL" if diff_qty < 0 else "BUY"
            order_type, limit_price = self._choose_order_type(
                side, pair, urgency, weight_change)

            if diff_qty < 0:
                # Need to sell
                sell_qty = min(rounded_qty, current_qty)
                if sell_qty > 0:
                    sells.append(TradeOrder(
                        pair=pair,
                        side="SELL",
                        quantity=sell_qty,
                        order_type=order_type,
                        price=limit_price,
                        urgency=urgency,
                        reason=f"rebalance: {current_weight:.1%} -> {target_weight:.1%}",
                    ))
            else:
                # Need to buy
                buys.append(TradeOrder(
                    pair=pair,
                    side="BUY",
                    quantity=rounded_qty,
                    order_type=order_type,
                    price=limit_price,
                    urgency=urgency,
                    reason=f"rebalance: {current_weight:.1%} -> {target_weight:.1%}",
                ))

        # Sells first (free up capital), then buys
        return sells + buys

    def execute_orders(self, orders: List[TradeOrder]) -> List[TradeOrder]:
        """Execute a list of orders via Roostoo API.

        Args:
            orders: List of TradeOrder objects to execute

        Returns:
            Same list with status/fill fields updated
        """
        for order in orders:
            try:
                result = self.client.place_order(
                    pair=order.pair,
                    side=order.side,
                    quantity=order.quantity,
                    order_type=order.order_type,
                    price=order.price,
                )

                order.order_id = result.get("OrderID")
                order.status = result.get("Status", "UNKNOWN")
                order.fill_price = float(result.get("FilledAverPrice", 0))
                order.fill_quantity = float(result.get("FilledQuantity", 0))
                order.commission = float(result.get("CommissionChargeValue", 0))

                logger.info("Executed %s %s %s: qty=%.6f fill=%.6f @ %.4f commission=%.6f",
                            order.order_type, order.side, order.pair,
                            order.quantity, order.fill_quantity or 0,
                            order.fill_price or 0, order.commission or 0)

            except Exception as e:
                order.status = "FAILED"
                logger.error("Order execution failed for %s %s %s: %s",
                             order.side, order.pair, order.quantity, e)

            self.trade_log.append(order)

        return orders

    def cancel_all_pending(self):
        """Cancel all pending orders."""
        try:
            result = self.client.cancel_all_pending()
            canceled = result.get("CanceledList", [])
            if canceled:
                logger.info("Canceled %d pending orders", len(canceled))
        except Exception as e:
            logger.error("Failed to cancel pending orders: %s", e)

    def get_current_holdings(self) -> Dict[str, float]:
        """Get current coin holdings from Roostoo wallet.

        Returns:
            {coin: free_quantity} dict (excludes USD and locked amounts)
        """
        wallet = self.client.balance()
        holdings = {}
        for coin, amounts in wallet.items():
            if coin == "USD":
                continue
            free = float(amounts.get("Free", 0))
            if free > 0:
                holdings[coin] = free
        return holdings

    def get_usd_balance(self) -> float:
        """Get available USD balance."""
        wallet = self.client.balance()
        usd = wallet.get("USD", {})
        return float(usd.get("Free", 0))

    def compute_turnover(self, orders: List[TradeOrder],
                         current_prices: Dict[str, float],
                         portfolio_value: float) -> Dict:
        """Compute turnover metrics for a set of orders."""
        buy_value = sum(
            o.quantity * current_prices.get(o.pair, 0)
            for o in orders if o.side == "BUY"
        )
        sell_value = sum(
            o.quantity * current_prices.get(o.pair, 0)
            for o in orders if o.side == "SELL"
        )
        total = buy_value + sell_value
        turnover_pct = total / portfolio_value if portfolio_value > 0 else 0

        # Estimate commission savings from limit orders
        n_limit = sum(1 for o in orders if o.order_type == "LIMIT")
        n_market = sum(1 for o in orders if o.order_type == "MARKET")

        return {
            "buy_value_usd": buy_value,
            "sell_value_usd": sell_value,
            "total_value_usd": total,
            "turnover_pct": turnover_pct,
            "n_orders": len(orders),
            "n_limit_orders": n_limit,
            "n_market_orders": n_market,
        }
