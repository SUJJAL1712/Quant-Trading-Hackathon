"""
Order Executor — Roostoo Trade Execution
==========================================
Converts target weights -> Roostoo API orders.
Handles order sizing, precision rounding, and execution logging.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config as cfg
from roostoo_client import RoostooClient

logger = logging.getLogger(__name__)


@dataclass
class TradeOrder:
    """Represents a single trade to execute."""
    pair: str               # e.g., "BTC/USD"
    side: str               # "BUY" or "SELL"
    quantity: float         # in base asset units
    order_type: str = "MARKET"
    price: Optional[float] = None  # for LIMIT orders
    reason: str = ""        # why this trade was generated
    # Filled by execution
    status: str = "PENDING"
    order_id: Optional[int] = None
    fill_price: Optional[float] = None
    fill_quantity: Optional[float] = None
    commission: Optional[float] = None


class TradeExecutor:
    """Generate and execute orders on Roostoo.

    Pipeline:
    1. Compare current holdings vs target weights
    2. Generate sell orders first (free up capital)
    3. Generate buy orders (use freed capital)
    4. Execute all orders via Roostoo API
    """

    def __init__(self, client: RoostooClient = None):
        self.client = client or RoostooClient()
        self.trade_log: List[TradeOrder] = []

    def generate_orders(self, current_holdings: Dict[str, float],
                        target_weights: Dict[str, float],
                        current_prices: Dict[str, float],
                        portfolio_value: float) -> List[TradeOrder]:
        """Generate orders to move from current to target allocation.

        Args:
            current_holdings: {coin: quantity} of current positions
            target_weights: {coin: weight} target allocation (0 to 1)
            current_prices: {pair: price} current market prices
            portfolio_value: Total portfolio value in USD

        Returns:
            List of TradeOrder objects (sells first, then buys)
        """
        orders = []
        sells = []
        buys = []

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

            # Skip if change is too small
            weight_change = abs(target_weight - current_weight)
            if diff_value < min_trade or weight_change < cfg.REBALANCE.min_weight_change:
                continue

            # Round quantity to exchange precision
            rounded_qty = self.client.round_quantity(pair, abs(diff_qty))
            if rounded_qty <= 0:
                continue

            if diff_qty < 0:
                # Need to sell
                # Don't sell more than we have
                sell_qty = min(rounded_qty, current_qty)
                if sell_qty > 0:
                    sells.append(TradeOrder(
                        pair=pair,
                        side="SELL",
                        quantity=sell_qty,
                        reason=f"rebalance: {current_weight:.1%} -> {target_weight:.1%}",
                    ))
            else:
                # Need to buy
                buys.append(TradeOrder(
                    pair=pair,
                    side="BUY",
                    quantity=rounded_qty,
                    reason=f"rebalance: {current_weight:.1%} -> {target_weight:.1%}",
                ))

        # Sells first (free up capital), then buys
        orders = sells + buys
        return orders

    def execute_orders(self, orders: List[TradeOrder]) -> List[TradeOrder]:
        """Execute a list of orders via Roostoo API.

        Args:
            orders: List of TradeOrder objects to execute

        Returns:
            Same list with status/fill fields updated
        """
        for order in orders:
            try:
                if order.order_type == "MARKET":
                    result = self.client.place_order(
                        pair=order.pair,
                        side=order.side,
                        quantity=order.quantity,
                        order_type="MARKET",
                    )
                else:
                    result = self.client.place_order(
                        pair=order.pair,
                        side=order.side,
                        quantity=order.quantity,
                        order_type="LIMIT",
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

        return {
            "buy_value_usd": buy_value,
            "sell_value_usd": sell_value,
            "total_value_usd": total,
            "turnover_pct": turnover_pct,
            "n_orders": len(orders),
        }
