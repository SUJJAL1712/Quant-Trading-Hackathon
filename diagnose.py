"""Diagnose backtest results."""
import pandas as pd
import numpy as np
import glob

dirs = sorted(glob.glob('data/csv/backtest_*/'))
if not dirs:
    print("No backtest results found")
    exit()

latest = dirs[-1]
print(f"Analyzing: {latest}")

plog = pd.read_csv(f'{latest}/portfolio_log.csv')
print('\n=== Portfolio log summary ===')
print(f'Regimes detected: {plog["regime"].value_counts().to_dict()}')
print(f'Avg invested: {plog["invested_pct"].mean()*100:.1f}%')
print(f'Min invested: {plog["invested_pct"].min()*100:.1f}%')
print(f'Max invested: {plog["invested_pct"].max()*100:.1f}%')
print(f'Avg positions: {plog["n_positions"].mean():.1f}')
print(f'Avg trades/rebalance: {plog["n_trades"].mean():.1f}')

tlog = pd.read_csv(f'{latest}/trade_log.csv')
print(f'\n=== Trade log ===')
print(f'Total trades: {len(tlog)}')
buy_count = (tlog["side"] == "BUY").sum()
sell_count = (tlog["side"] == "SELL").sum()
print(f'Buys: {buy_count}, Sells: {sell_count}')
print(f'Total commission: ${tlog["commission"].sum():.2f}')
print(f'Avg trade value: ${tlog["value"].mean():.2f}')

# Per-coin P&L
print('\n=== Per-coin trade P&L ===')
for pair in sorted(tlog['pair'].unique()):
    coin_trades = tlog[tlog['pair'] == pair]
    buys = coin_trades[coin_trades['side'] == 'BUY']
    sells = coin_trades[coin_trades['side'] == 'SELL']
    buy_val = buys['value'].sum()
    sell_val = sells['value'].sum()
    comm = coin_trades['commission'].sum()
    net = sell_val - buy_val - comm
    print(f'  {pair:10s} Buy: ${buy_val:8.0f} Sell: ${sell_val:8.0f} Net: ${net:+8.0f} Comm: ${comm:.0f}')

print('\n=== Regime timeline ===')
for _, row in plog.iloc[::10].iterrows():
    ts = row["timestamp"][:16]
    regime = row["regime"]
    nav = row["nav_usd"]
    inv = row["invested_pct"] * 100
    dd = row["drawdown"] * 100
    n = row["n_positions"]
    print(f'  {ts} | {regime:8s} | NAV: ${nav:>9,.0f} | Inv: {inv:4.0f}% | DD: {dd:5.1f}% | Pos: {n}')
