"""
Backtest Dashboard -- Crypto Trading Bot
=========================================
Streamlit dashboard for visualizing backtest results.

Usage:
    streamlit run dashboard.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ──
st.set_page_config(
    page_title="Crypto Trading Bot -- Backtest Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "csv")


# ── Helpers ──

@st.cache_data
def list_backtests():
    """Find all backtest result directories."""
    pattern = os.path.join(DATA_DIR, "backtest_*/summary.json")
    results = []
    for path in sorted(glob.glob(pattern), reverse=True):
        run_dir = os.path.dirname(path)
        run_id = os.path.basename(run_dir).replace("backtest_", "")
        try:
            with open(path) as f:
                summary = json.load(f)
            summary["run_dir"] = run_dir
            summary["run_id"] = run_id
            results.append(summary)
        except Exception:
            continue
    return results


@st.cache_data
def load_backtest(run_dir: str):
    """Load all data for a single backtest run."""
    data = {}

    nav_path = os.path.join(run_dir, "nav_series.csv")
    if os.path.exists(nav_path):
        df = pd.read_csv(nav_path, parse_dates=["timestamp"])
        data["nav"] = df

    plog_path = os.path.join(run_dir, "portfolio_log.csv")
    if os.path.exists(plog_path):
        df = pd.read_csv(plog_path, parse_dates=["timestamp"])
        data["portfolio"] = df

    tlog_path = os.path.join(run_dir, "trade_log.csv")
    if os.path.exists(tlog_path):
        df = pd.read_csv(tlog_path, parse_dates=["timestamp"])
        data["trades"] = df

    summary_path = os.path.join(run_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data["summary"] = json.load(f)

    return data


def format_pct(val, decimals=2):
    return f"{val:+.{decimals}f}%"


def format_usd(val):
    return f"${val:,.2f}"


# ── Sidebar ──

st.sidebar.title("Backtest Dashboard")
st.sidebar.markdown("---")

backtests = list_backtests()
if not backtests:
    st.error("No backtest results found in data/csv/. Run a backtest first:")
    st.code("python main.py --backtest --start 2024-11-01 --end 2025-03-15")
    st.stop()

# Build dropdown labels
options = {}
for bt in backtests:
    label = (f"{bt.get('start', '?')} to {bt.get('end', '?')} | "
             f"Return: {bt.get('total_return_pct', 0):+.1f}% | "
             f"Score: {bt.get('composite_score', 0):.4f}")
    options[label] = bt

selected_label = st.sidebar.selectbox("Select Backtest Run", list(options.keys()))
selected_bt = options[selected_label]
run_dir = selected_bt["run_dir"]

# Allow comparison
st.sidebar.markdown("---")
compare_enabled = st.sidebar.checkbox("Compare with another run")
compare_bt = None
compare_data = None
if compare_enabled and len(options) > 1:
    other_labels = [l for l in options if l != selected_label]
    compare_label = st.sidebar.selectbox("Compare with", other_labels)
    compare_bt = options[compare_label]
    compare_data = load_backtest(compare_bt["run_dir"])

# Strategy info in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Info")
st.sidebar.markdown("""
**Regime Detection:** Trend-based (not HMM)
- BTC price vs 7d/3d SMA
- SMA slope + drawdown + 24h momentum
- 24h hysteresis (min 6 cycles)
- Recalculated every **4 hours**

**Rebalance Frequency:** Every **4 hours** (24/7)

**Signals (5):**
1. Momentum (35%) -- 12h/72h/168h
2. Breakout (20%) -- 72h range
3. VolumeMomentum (15%)
4. MeanReversion (15%) -- RSI gate
5. RelativeStrength (15%) -- vs BTC

**Optimizer:** BL + HRP blend (regime-dependent)
""")

# ── Load data ──
data = load_backtest(run_dir)
summary = data.get("summary", selected_bt)

# ── Header ──
st.title("Crypto Trading Bot -- Backtest Results")
st.markdown(f"**Period:** {summary.get('start')} to {summary.get('end')} | "
            f"**Universe:** {summary.get('n_coins', '?')} coins | "
            f"**Capital:** {format_usd(summary.get('initial_capital', 100000))}")

# ── KPI Row (always visible above tabs) ──
st.markdown("---")
cols = st.columns(7)

total_ret = summary.get("total_return_pct", 0)
cols[0].metric("Total Return", format_pct(total_ret),
               delta=format_pct(total_ret), delta_color="normal")
cols[1].metric("Final NAV", format_usd(summary.get("final_nav", 0)))
cols[2].metric("Max Drawdown", format_pct(-summary.get("max_drawdown_pct", 0)))
cols[3].metric("Sortino", f"{summary.get('sortino', 0):.4f}")
cols[4].metric("Sharpe", f"{summary.get('sharpe', 0):.4f}")
cols[5].metric("Calmar", f"{summary.get('calmar', 0):.4f}")

composite = summary.get("composite_score", 0)
cols[6].metric("Composite Score",
               f"{composite:.4f}",
               delta="0.4xSortino + 0.3xSharpe + 0.3xCalmar",
               delta_color="off")

# ═══════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════

tab_perf, tab_regime, tab_trades, tab_risk, tab_compare = st.tabs([
    "Performance",
    "Regime & Allocation",
    "Trade Analysis",
    "Risk & Returns",
    "Compare Runs",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: Performance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_perf:
    st.subheader("Portfolio NAV Over Time")

    if "nav" in data:
        nav_df = data["nav"]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=("NAV ($)", "Drawdown (%)")
        )

        fig.add_trace(
            go.Scatter(x=nav_df["timestamp"], y=nav_df["nav_usd"],
                       name="NAV", line=dict(color="#2196F3", width=1.5)),
            row=1, col=1
        )

        initial = summary.get("initial_capital", 100000)
        fig.add_hline(y=initial, line_dash="dash", line_color="gray",
                      annotation_text="Initial Capital", row=1, col=1)

        if compare_data and "nav" in compare_data:
            cmp_nav = compare_data["nav"]
            fig.add_trace(
                go.Scatter(x=cmp_nav["timestamp"], y=cmp_nav["nav_usd"],
                           name=f"Compare: {compare_bt.get('start', '')} to {compare_bt.get('end', '')}",
                           line=dict(color="#FF9800", width=1.5, dash="dot")),
                row=1, col=1
            )

        hwm = nav_df["nav_usd"].cummax()
        dd = (nav_df["nav_usd"] - hwm) / hwm * 100
        fig.add_trace(
            go.Scatter(x=nav_df["timestamp"], y=dd,
                       name="Drawdown", fill="tozeroy",
                       line=dict(color="#F44336", width=1),
                       fillcolor="rgba(244, 67, 54, 0.2)"),
            row=2, col=1
        )

        fig.update_layout(
            height=550,
            margin=dict(l=60, r=30, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="NAV ($)", row=1, col=1)
        fig.update_yaxes(title_text="DD (%)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Cumulative return
        st.subheader("Cumulative Return (%)")
        cum_ret = (nav_df["nav_usd"] / nav_df["nav_usd"].iloc[0] - 1) * 100
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=nav_df["timestamp"], y=cum_ret,
            name="Strategy",
            fill="tozeroy",
            line=dict(color="#2196F3", width=2),
            fillcolor="rgba(33, 150, 243, 0.1)",
        ))
        fig_cum.add_hline(y=0, line_color="gray", line_dash="dash")
        fig_cum.update_layout(
            height=300,
            margin=dict(l=60, r=30, t=30, b=30),
            yaxis_title="Cumulative Return (%)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_cum, use_container_width=True)
    else:
        st.warning("No NAV data available")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: Regime & Allocation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_regime:
    if "portfolio" in data:
        plog = data["portfolio"]

        st.subheader("Regime Timeline")

        fig2 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.5, 0.5],
            subplot_titles=("Market Regime", "Invested %")
        )

        regime_colors = {"BULL": "#4CAF50", "NEUTRAL": "#FFC107", "BEAR": "#F44336"}
        for regime_name, color in regime_colors.items():
            mask = plog["regime"] == regime_name
            if mask.any():
                fig2.add_trace(
                    go.Scatter(
                        x=plog.loc[mask, "timestamp"],
                        y=[regime_name] * mask.sum(),
                        mode="markers",
                        marker=dict(color=color, size=8, symbol="square"),
                        name=regime_name,
                        showlegend=True,
                    ),
                    row=1, col=1
                )

        fig2.add_trace(
            go.Scatter(x=plog["timestamp"], y=plog["invested_pct"] * 100,
                       name="Invested %", fill="tozeroy",
                       line=dict(color="#2196F3", width=1.5),
                       fillcolor="rgba(33, 150, 243, 0.15)"),
            row=2, col=1
        )
        fig2.add_hline(y=85, line_dash="dash", line_color="#4CAF50",
                       annotation_text="Bull target (85%)", row=2, col=1)
        fig2.add_hline(y=60, line_dash="dash", line_color="#FFC107",
                       annotation_text="Neutral target (60%)", row=2, col=1)
        fig2.add_hline(y=35, line_dash="dash", line_color="#F44336",
                       annotation_text="Bear target (35%)", row=2, col=1)

        fig2.update_layout(
            height=500,
            margin=dict(l=60, r=30, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        fig2.update_yaxes(title_text="Regime", row=1, col=1)
        fig2.update_yaxes(title_text="Invested (%)", row=2, col=1)
        st.plotly_chart(fig2, use_container_width=True)

        # Regime distribution
        st.subheader("Regime Distribution")
        col1, col2, col3 = st.columns(3)
        regime_counts = plog["regime"].value_counts()
        total = len(plog)
        for col, regime in zip([col1, col2, col3], ["BULL", "NEUTRAL", "BEAR"]):
            cnt = regime_counts.get(regime, 0)
            pct = cnt / total * 100 if total > 0 else 0
            col.metric(f"{regime} Regime", f"{pct:.1f}%", delta=f"{cnt} cycles")

        # Regime pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=regime_counts.index.tolist(),
            values=regime_counts.values.tolist(),
            marker=dict(colors=[regime_colors.get(r, "#999") for r in regime_counts.index]),
            hole=0.4,
        )])
        fig_pie.update_layout(
            title="Regime Time Distribution",
            height=300,
            margin=dict(l=30, r=30, t=50, b=30),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("No portfolio log data available")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: Trade Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_trades:
    if "trades" in data:
        trades = data["trades"]

        st.subheader("Trade Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", f"{len(trades):,}")
        buys = trades[trades["side"] == "BUY"]
        sells = trades[trades["side"] == "SELL"]
        col2.metric("Buys", f"{len(buys):,}")
        col3.metric("Sells", f"{len(sells):,}")
        col4.metric("Total Commission", format_usd(trades["commission"].sum()))

        # Per-coin P&L
        st.subheader("P&L by Coin")
        coin_stats = []
        for pair in sorted(trades["pair"].unique()):
            pt = trades[trades["pair"] == pair]
            buy_val = pt.loc[pt["side"] == "BUY", "value"].sum()
            sell_val = pt.loc[pt["side"] == "SELL", "value"].sum()
            comm = pt["commission"].sum()
            net = sell_val - buy_val - comm
            coin_stats.append({
                "Pair": pair,
                "Trades": len(pt),
                "Buy Volume ($)": round(buy_val, 0),
                "Sell Volume ($)": round(sell_val, 0),
                "Net P&L ($)": round(net, 0),
                "Commission ($)": round(comm, 2),
            })

        coin_df = pd.DataFrame(coin_stats).sort_values("Net P&L ($)", ascending=False)

        fig3 = go.Figure()
        colors = ["#4CAF50" if v >= 0 else "#F44336" for v in coin_df["Net P&L ($)"]]
        fig3.add_trace(go.Bar(
            x=coin_df["Pair"], y=coin_df["Net P&L ($)"],
            marker_color=colors,
            text=[f"${v:+,.0f}" for v in coin_df["Net P&L ($)"]],
            textposition="outside",
        ))
        fig3.update_layout(
            title="Net P&L by Coin",
            height=400,
            margin=dict(l=60, r=30, t=50, b=30),
            yaxis_title="Net P&L ($)",
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(coin_df, use_container_width=True, hide_index=True)

        # Trades over time
        st.subheader("Trades Over Time")
        trades_copy = trades.copy()
        trades_copy["date"] = trades_copy["timestamp"].dt.date
        daily_trades = trades_copy.groupby(["date", "side"]).size().unstack(fill_value=0)
        fig5 = go.Figure()
        if "BUY" in daily_trades.columns:
            fig5.add_trace(go.Bar(x=daily_trades.index, y=daily_trades["BUY"],
                                  name="BUY", marker_color="#4CAF50"))
        if "SELL" in daily_trades.columns:
            fig5.add_trace(go.Bar(x=daily_trades.index, y=daily_trades["SELL"],
                                  name="SELL", marker_color="#F44336"))
        fig5.update_layout(
            barmode="group",
            height=350,
            margin=dict(l=60, r=30, t=30, b=30),
            xaxis_title="Date",
            yaxis_title="Number of Trades",
        )
        st.plotly_chart(fig5, use_container_width=True)

        # Trade value distribution
        st.subheader("Trade Value Distribution")
        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(
            x=trades["value"], nbinsx=50,
            marker_color="#2196F3",
            name="Trade Value"
        ))
        fig4.update_layout(
            height=300,
            margin=dict(l=60, r=30, t=30, b=30),
            xaxis_title="Trade Value ($)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("No trade data available")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: Risk & Returns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_risk:
    # Position count & drawdown
    if "portfolio" in data:
        plog = data["portfolio"]

        st.subheader("Risk Metrics Over Time")
        col1, col2 = st.columns(2)

        with col1:
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(
                x=plog["timestamp"], y=plog["n_positions"],
                name="# Positions", fill="tozeroy",
                line=dict(color="#9C27B0", width=1.5),
                fillcolor="rgba(156, 39, 176, 0.15)",
            ))
            fig6.update_layout(
                title="Number of Positions",
                height=300,
                margin=dict(l=60, r=30, t=50, b=30),
                yaxis_title="Positions",
            )
            st.plotly_chart(fig6, use_container_width=True)

        with col2:
            fig7 = go.Figure()
            fig7.add_trace(go.Scatter(
                x=plog["timestamp"], y=plog["drawdown"] * 100,
                name="Current DD", fill="tozeroy",
                line=dict(color="#F44336", width=1.5),
                fillcolor="rgba(244, 67, 54, 0.15)",
            ))
            fig7.add_trace(go.Scatter(
                x=plog["timestamp"], y=plog["max_drawdown"] * 100,
                name="Max DD", line=dict(color="#FF9800", width=1, dash="dash"),
            ))
            fig7.update_layout(
                title="Drawdown (%)",
                height=300,
                margin=dict(l=60, r=30, t=50, b=30),
                yaxis_title="Drawdown (%)",
            )
            st.plotly_chart(fig7, use_container_width=True)

    # Returns distribution
    if "nav" in data:
        nav_df = data["nav"]

        st.subheader("Returns Distribution")

        nav_vals = nav_df["nav_usd"].values
        hourly_returns = np.diff(nav_vals) / nav_vals[:-1]
        hourly_returns = hourly_returns[np.isfinite(hourly_returns)]

        daily_nav = nav_df.set_index("timestamp")["nav_usd"].resample("1D").last().dropna()
        daily_returns = daily_nav.pct_change().dropna().values

        col1, col2 = st.columns(2)

        with col1:
            fig8 = go.Figure()
            fig8.add_trace(go.Histogram(
                x=hourly_returns * 100, nbinsx=100,
                marker_color="#2196F3",
                name="Hourly Returns"
            ))
            fig8.add_vline(x=0, line_color="red", line_dash="dash")
            fig8.update_layout(
                title="Hourly Returns Distribution",
                height=300,
                margin=dict(l=60, r=30, t=50, b=30),
                xaxis_title="Return (%)",
                yaxis_title="Count",
            )
            st.plotly_chart(fig8, use_container_width=True)

        with col2:
            fig9 = go.Figure()
            fig9.add_trace(go.Histogram(
                x=daily_returns * 100, nbinsx=50,
                marker_color="#FF9800",
                name="Daily Returns"
            ))
            fig9.add_vline(x=0, line_color="red", line_dash="dash")
            fig9.update_layout(
                title="Daily Returns Distribution",
                height=300,
                margin=dict(l=60, r=30, t=50, b=30),
                xaxis_title="Return (%)",
                yaxis_title="Count",
            )
            st.plotly_chart(fig9, use_container_width=True)

        # Stats row
        st.subheader("Return Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Daily Return", format_pct(daily_returns.mean() * 100, 3))
        col2.metric("Daily Vol", format_pct(daily_returns.std() * 100, 3))
        col3.metric("Best Day", format_pct(daily_returns.max() * 100))
        col4.metric("Worst Day", format_pct(daily_returns.min() * 100))

    if "nav" not in data and "portfolio" not in data:
        st.warning("No risk data available")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5: Compare Runs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_compare:
    st.subheader("All Backtests Comparison")

    compare_rows = []
    for bt in backtests:
        compare_rows.append({
            "Run ID": bt.get("run_id", "?"),
            "Period": f"{bt.get('start', '?')} to {bt.get('end', '?')}",
            "Return (%)": bt.get("total_return_pct", 0),
            "Max DD (%)": bt.get("max_drawdown_pct", 0),
            "Sortino": bt.get("sortino", 0),
            "Sharpe": bt.get("sharpe", 0),
            "Calmar": bt.get("calmar", 0),
            "Composite": bt.get("composite_score", 0),
            "Trades": bt.get("total_trades", 0),
            "Rebalances": bt.get("rebalance_count", 0),
            "Commission ($)": bt.get("total_commission", 0),
        })

    compare_df = pd.DataFrame(compare_rows)
    st.dataframe(
        compare_df.style.format({
            "Return (%)": "{:+.2f}",
            "Max DD (%)": "{:.2f}",
            "Sortino": "{:.4f}",
            "Sharpe": "{:.4f}",
            "Calmar": "{:.4f}",
            "Composite": "{:.4f}",
            "Commission ($)": "{:.2f}",
        }).background_gradient(subset=["Return (%)"], cmap="RdYlGn")
        .background_gradient(subset=["Composite"], cmap="RdYlGn"),
        use_container_width=True,
        hide_index=True,
    )

    # Scatter: Return vs Max DD
    st.subheader("Return vs Max Drawdown")
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=compare_df["Max DD (%)"],
        y=compare_df["Return (%)"],
        mode="markers+text",
        text=compare_df["Period"],
        textposition="top center",
        marker=dict(
            size=12,
            color=compare_df["Composite"],
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="Composite"),
        ),
    ))
    fig_scatter.update_layout(
        height=400,
        margin=dict(l=60, r=30, t=30, b=30),
        xaxis_title="Max Drawdown (%)",
        yaxis_title="Total Return (%)",
    )
    fig_scatter.add_hline(y=0, line_color="gray", line_dash="dash")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Bar: Composite scores
    st.subheader("Composite Scores")
    sorted_df = compare_df.sort_values("Composite", ascending=True)
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in sorted_df["Composite"]]
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=sorted_df["Composite"],
        y=sorted_df["Period"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in sorted_df["Composite"]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        height=max(300, len(sorted_df) * 35),
        margin=dict(l=200, r=60, t=30, b=30),
        xaxis_title="Composite Score",
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ── Footer ──
st.markdown("---")
st.caption(
    "Crypto Trading Bot -- SG vs HK University Web3 Quant Trading Hackathon | "
    "Composite Score = 0.4 x Sortino + 0.3 x Sharpe + 0.3 x Calmar"
)
