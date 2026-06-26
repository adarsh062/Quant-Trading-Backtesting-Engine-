"""
app.py
------
Quant Trading Backtesting Platform
A professional Streamlit dashboard for backtesting algorithmic trading strategies
on historical stock market data (US & Indian exchanges).

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

# Internal modules
from data_loader import (
    fetch_stock_data,
    get_ticker_info,
    compute_technical_indicators,
    validate_date_range,
    export_to_csv,
    POPULAR_STOCKS,
)
from strategies import (
    MovingAverageCrossover,
    RSIStrategy,
    MACDStrategy,
    run_backtest,
    compare_strategies,
    STRATEGY_MAP,
)
from analytics import (
    calculate_metrics,
    metrics_to_dict,
    find_best_strategy,
    build_comparison_table,
    trades_to_dataframe,
)
from charts import (
    plot_price_signals,
    plot_equity_curve,
    plot_drawdown,
    plot_rsi,
    plot_macd,
    plot_comparison_bars,
    plot_monthly_returns_heatmap,
    plot_trade_distribution,
)


# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="QuantLab – Backtesting Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Injected CSS – dark theme overrides
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #0a0e16 100%);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #58a6ff;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(88,166,255,0.15);
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #e6edf3;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-value.positive { color: #3fb950; }
    .metric-value.negative { color: #f85149; }
    .metric-value.neutral  { color: #58a6ff; }

    /* ── Section headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px 0 8px 0;
        border-bottom: 1px solid #21262d;
        margin-bottom: 16px;
    }
    .section-header h2 {
        font-size: 18px;
        font-weight: 600;
        color: #e6edf3;
        margin: 0;
    }

    /* ── Strategy badge ── */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-blue  { background: rgba(88,166,255,0.15); color: #58a6ff; border: 1px solid #388bfd; }
    .badge-green { background: rgba(63,185,80,0.15);  color: #3fb950; border: 1px solid #3fb950; }
    .badge-red   { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid #f85149; }

    /* ── Best strategy highlight ── */
    .best-strategy-banner {
        background: linear-gradient(90deg, rgba(63,185,80,0.1) 0%, rgba(88,166,255,0.05) 100%);
        border: 1px solid #3fb950;
        border-radius: 10px;
        padding: 14px 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
    }
    .best-strategy-banner .trophy { font-size: 28px; }
    .best-strategy-banner .label  { font-size: 12px; color: #8b949e; }
    .best-strategy-banner .name   { font-size: 18px; font-weight: 700; color: #3fb950; }

    /* ── Info panel ── */
    .info-panel {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .info-panel .title { font-size: 14px; font-weight: 600; color: #e6edf3; margin-bottom: 8px; }
    .info-panel .body  { font-size: 13px; color: #8b949e; line-height: 1.6; }

    /* ── Tab styling ── */
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: #8b949e;
        font-weight: 500;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #30363d;
        border-radius: 8px;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Divider ── */
    hr { border-color: #21262d; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# Helper: render a metric card
# ─────────────────────────────────────────────
def metric_card(label: str, value: str, color_class: str = "neutral") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
    </div>
    """


def section_header(icon: str, title: str):
    st.markdown(
        f'<div class="section-header"><h2>{icon} {title}</h2></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("# 📈")
with col_title:
    st.markdown(
        """
        <div style='padding-top: 4px;'>
            <span style='font-size:28px; font-weight:700; color:#e6edf3;'>QuantLab</span>
            <span style='font-size:14px; color:#8b949e; margin-left:12px;'>
                Professional Backtesting Platform
            </span>
        </div>
        <div style='font-size:12px; color:#30363d; font-family: JetBrains Mono, monospace;'>
            v1.0.0 &nbsp;|&nbsp; Python · Streamlit · Plotly
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar – Configuration panel
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # ── Market Data ──────────────────────────
    st.markdown("### 📊 Market Data")

    market = st.selectbox(
        "Market",
        ["US Stocks", "Indian Stocks (NSE)", "Custom Ticker"],
        key="market_sel",
    )

    if market == "Custom Ticker":
        ticker = st.text_input(
            "Ticker Symbol",
            value="AAPL",
            help="Enter any valid Yahoo Finance ticker (e.g. AAPL, RELIANCE.NS)",
            key="custom_ticker",
        ).upper().strip()
    else:
        stock_options = POPULAR_STOCKS[market]
        selected_label = st.selectbox(
            "Select Stock", list(stock_options.keys()), key="stock_label"
        )
        ticker = stock_options[selected_label]
        st.caption(f"Ticker: `{ticker}`")

    # ── Date Range ───────────────────────────
    st.markdown("### 📅 Date Range")
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input(
            "Start", value=date.today() - timedelta(days=3 * 365), key="start_date"
        )
    with col_e:
        end_date = st.date_input("End", value=date.today(), key="end_date")

    # ── Strategy Selection ───────────────────
    st.markdown("### 🎯 Strategy")
    mode = st.radio(
        "Mode",
        ["Single Strategy", "Compare All Strategies"],
        key="mode_radio",
    )

    selected_strategies: list[str] = []
    strategy_kwargs: dict[str, dict] = {}

    if mode == "Single Strategy":
        strat_name = st.selectbox(
            "Strategy", list(STRATEGY_MAP.keys()), key="strat_sel"
        )
        selected_strategies = [strat_name]
    else:
        selected_strategies = list(STRATEGY_MAP.keys())

    # ── Strategy Parameters ──────────────────
    st.markdown("### 🔧 Parameters")

    if "Moving Average Crossover" in selected_strategies:
        with st.expander("MA Crossover", expanded=(mode == "Single Strategy")):
            ma_type = st.selectbox("MA Type", ["SMA", "EMA"], key="ma_type")
            ma_short = st.slider("Short Window", 5, 50, 20, key="ma_short")
            ma_long = st.slider("Long Window", 20, 200, 50, key="ma_long")
            strategy_kwargs["Moving Average Crossover"] = dict(
                short_window=ma_short, long_window=ma_long, ma_type=ma_type
            )

    if "RSI Strategy" in selected_strategies:
        with st.expander("RSI", expanded=(mode == "Single Strategy")):
            rsi_period = st.slider("RSI Period", 7, 21, 14, key="rsi_period")
            oversold = st.slider("Oversold Level", 10, 40, 30, key="oversold")
            overbought = st.slider("Overbought Level", 60, 90, 70, key="overbought")
            strategy_kwargs["RSI Strategy"] = dict(
                rsi_period=rsi_period, oversold=float(oversold), overbought=float(overbought)
            )

    if "MACD Strategy" in selected_strategies:
        with st.expander("MACD", expanded=(mode == "Single Strategy")):
            macd_fast = st.slider("Fast Period", 5, 20, 12, key="macd_fast")
            macd_slow = st.slider("Slow Period", 15, 50, 26, key="macd_slow")
            macd_sig = st.slider("Signal Period", 5, 15, 9, key="macd_sig")
            strategy_kwargs["MACD Strategy"] = dict(
                fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_sig
            )

    # ── Capital & Commission ─────────────────
    st.markdown("### 💼 Portfolio")
    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=1_000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
        key="capital",
    )
    commission = st.slider(
        "Commission (%)", min_value=0.0, max_value=1.0,
        value=0.1, step=0.01, format="%.2f%%", key="commission"
    ) / 100

    st.markdown("---")
    run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
# Main content area
# ─────────────────────────────────────────────
if not run_button:
    # ── Welcome screen ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="info-panel">
                <div class="title">🏦 3 Trading Strategies</div>
                <div class="body">
                    Moving Average Crossover, RSI Mean-Reversion,
                    and MACD Signal-Line strategies with full parameter control.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="info-panel">
                <div class="title">📊 10+ Performance Metrics</div>
                <div class="body">
                    Sharpe, Sortino & Calmar ratios, Max Drawdown,
                    Win Rate, Profit Factor, and more — computed in real time.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="info-panel">
                <div class="title">🌏 US & Indian Markets</div>
                <div class="body">
                    Fetch live historical data for NYSE/NASDAQ stocks
                    and NSE/BSE equities via Yahoo Finance.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center; padding: 40px 0; color: #30363d;">
            <div style="font-size: 48px; margin-bottom: 12px;">📈</div>
            <div style="font-size: 20px; color: #8b949e; font-weight: 600;">
                Configure your backtest in the sidebar and click
                <span style="color: #58a6ff;">Run Backtest</span>
            </div>
            <div style="font-size: 13px; color: #30363d; margin-top: 8px;">
                Supports 20+ US stocks · 10+ Indian NSE stocks · Custom tickers
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ─────────────────────────────────────────────
# Backtest execution
# ─────────────────────────────────────────────
start_str = str(start_date)
end_str = str(end_date)

# Validate date range
is_valid, date_err = validate_date_range(start_str, end_str)
if not is_valid:
    st.error(f"❌ {date_err}")
    st.stop()

# Check strategy parameter conflicts
if "Moving Average Crossover" in selected_strategies:
    p = strategy_kwargs.get("Moving Average Crossover", {})
    if p.get("short_window", 20) >= p.get("long_window", 50):
        st.error("❌ MA Crossover: Short window must be less than the long window.")
        st.stop()
if "MACD Strategy" in selected_strategies:
    p = strategy_kwargs.get("MACD Strategy", {})
    if p.get("fast_period", 12) >= p.get("slow_period", 26):
        st.error("❌ MACD: Fast period must be less than the slow period.")
        st.stop()

with st.spinner(f"🔄 Fetching data for **{ticker}**..."):
    try:
        raw_df = fetch_stock_data(ticker, start_str, end_str)
        df_with_indicators = compute_technical_indicators(raw_df)
        ticker_info = get_ticker_info(ticker)
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

# ─── Ticker info banner ──────────────────────────────────────────────
name = ticker_info.get("name", ticker)
sector = ticker_info.get("sector", "N/A")
exchange = ticker_info.get("exchange", "N/A")
currency = ticker_info.get("currency", "USD")
current_price = ticker_info.get("current_price")

col_i1, col_i2, col_i3, col_i4 = st.columns([3, 2, 2, 2])
with col_i1:
    st.markdown(
        f"""
        <div style='padding: 4px 0;'>
            <div style='font-size:20px; font-weight:700; color:#e6edf3;'>{name}</div>
            <div style='font-size:12px; color:#8b949e;'>
                <span class='badge badge-blue'>{ticker}</span> &nbsp;
                {exchange} &nbsp;·&nbsp; {sector}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_i2:
    price_str = f"{currency} {current_price:,.2f}" if current_price else "N/A"
    st.metric("Current Price", price_str)
with col_i3:
    st.metric("Data Points", f"{len(raw_df):,}")
with col_i4:
    st.metric("Date Range", f"{start_str} → {end_str}")

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Run strategies
# ─────────────────────────────────────────────
strategy_instances = []
for sname in selected_strategies:
    kwargs = strategy_kwargs.get(sname, {})
    try:
        if sname == "Moving Average Crossover":
            strat = MovingAverageCrossover(**kwargs)
        elif sname == "RSI Strategy":
            strat = RSIStrategy(**kwargs)
        elif sname == "MACD Strategy":
            strat = MACDStrategy(**kwargs)
        else:
            continue
        strategy_instances.append(strat)
    except ValueError as e:
        st.error(f"❌ {sname}: {e}")
        st.stop()

with st.spinner("⚙️ Running backtest..."):
    results: dict[str, pd.DataFrame] = compare_strategies(
        raw_df, strategy_instances, initial_capital
    )

# Compute metrics + trades for each strategy
all_metrics = {}
all_trades = {}
for strat in strategy_instances:
    signals_df = strat.generate_signals(raw_df)
    bt_df = results[strat.name]
    trades = strat.extract_trades(signals_df)
    metrics = calculate_metrics(bt_df, trades, initial_capital)
    all_metrics[strat.name] = metrics
    all_trades[strat.name] = trades

best_name = find_best_strategy(all_metrics)

# ── Best Strategy Banner ─────────────────────────────────────────────
if mode == "Compare All Strategies" and best_name:
    st.markdown(
        f"""
        <div class="best-strategy-banner">
            <div class="trophy">🏆</div>
            <div>
                <div class="label">BEST PERFORMING STRATEGY (by Sharpe Ratio)</div>
                <div class="name">{best_name}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Determine active strategy for single-view tabs
# ─────────────────────────────────────────────
active_name = selected_strategies[0] if selected_strategies else best_name
active_df = results[active_name]
active_strat = next(s for s in strategy_instances if s.name == active_name)
active_signals = active_strat.generate_signals(raw_df)

# ─────────────────────────────────────────────
# Key Metrics Cards
# ─────────────────────────────────────────────
section_header("📊", "Performance Metrics")

m = all_metrics[active_name]

def color_class(value: float) -> str:
    return "positive" if value > 0 else ("negative" if value < 0 else "neutral")

row1 = st.columns(4)
row2 = st.columns(4)
row3 = st.columns(4)

cards_row1 = [
    ("Total Return", f"{m.total_return_pct:.2f}%", color_class(m.total_return_pct)),
    ("Ann. Return", f"{m.annualized_return_pct:.2f}%", color_class(m.annualized_return_pct)),
    ("Sharpe Ratio", f"{m.sharpe_ratio:.3f}", color_class(m.sharpe_ratio)),
    ("Sortino Ratio", f"{m.sortino_ratio:.3f}", color_class(m.sortino_ratio)),
]
cards_row2 = [
    ("Max Drawdown", f"{m.max_drawdown_pct:.2f}%", "negative" if m.max_drawdown_pct < 0 else "neutral"),
    ("Win Rate", f"{m.win_rate_pct:.1f}%", color_class(m.win_rate_pct - 50)),
    ("Profit Factor", f"{m.profit_factor:.2f}", color_class(m.profit_factor - 1)),
    ("Total Trades", str(m.total_trades), "neutral"),
]
cards_row3 = [
    ("B&H Return", f"{m.bh_total_return_pct:.2f}%", color_class(m.bh_total_return_pct)),
    ("Ann. Volatility", f"{m.annualized_volatility_pct:.2f}%", "neutral"),
    ("Calmar Ratio", f"{m.calmar_ratio:.3f}", color_class(m.calmar_ratio)),
    ("Final Equity", f"${m.final_equity:,.0f}", color_class(m.final_equity - initial_capital)),
]

for col, (label, value, cls) in zip(row1, cards_row1):
    with col:
        st.markdown(metric_card(label, value, cls), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
for col, (label, value, cls) in zip(row2, cards_row2):
    with col:
        st.markdown(metric_card(label, value, cls), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
for col, (label, value, cls) in zip(row3, cards_row3):
    with col:
        st.markdown(metric_card(label, value, cls), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Chart Tabs
# ─────────────────────────────────────────────
section_header("📈", "Interactive Charts")

tab_labels = ["Price & Signals", "Equity Curve", "Drawdown"]
if "RSI" in str(active_name):
    tab_labels.append("RSI Indicator")
if "MACD" in str(active_name):
    tab_labels.append("MACD Indicator")
tab_labels += ["Monthly Returns", "Trade Distribution"]
if mode == "Compare All Strategies":
    tab_labels.append("Strategy Comparison")

tabs = st.tabs(tab_labels)

with tabs[0]:
    fig_price = plot_price_signals(active_signals, active_name)
    st.plotly_chart(fig_price, use_container_width=True)

with tabs[1]:
    fig_eq = plot_equity_curve(results, initial_capital)
    st.plotly_chart(fig_eq, use_container_width=True)

with tabs[2]:
    fig_dd = plot_drawdown(active_df, active_name)
    st.plotly_chart(fig_dd, use_container_width=True)

tab_idx = 3
if "RSI Indicator" in tab_labels:
    with tabs[tab_idx]:
        fig_rsi = plot_rsi(active_signals)
        st.plotly_chart(fig_rsi, use_container_width=True)
    tab_idx += 1

if "MACD Indicator" in tab_labels:
    with tabs[tab_idx]:
        fig_macd = plot_macd(active_signals)
        st.plotly_chart(fig_macd, use_container_width=True)
    tab_idx += 1

with tabs[tab_idx]:
    fig_monthly = plot_monthly_returns_heatmap(active_df)
    st.plotly_chart(fig_monthly, use_container_width=True)
tab_idx += 1

with tabs[tab_idx]:
    trades_list = all_trades[active_name]
    if trades_list:
        fig_dist = plot_trade_distribution(trades_list)
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("No completed trades to display.")
tab_idx += 1

if "Strategy Comparison" in tab_labels:
    with tabs[tab_idx]:
        comp_df = build_comparison_table(all_metrics)
        fig_comp = plot_comparison_bars(comp_df)
        st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Strategy comparison table
# ─────────────────────────────────────────────
if mode == "Compare All Strategies":
    section_header("🏆", "Strategy Comparison Table")
    comp_df = build_comparison_table(all_metrics)

    # Highlight best strategy row
    def highlight_best(row):
        return [
            "background-color: rgba(63,185,80,0.15); color: #3fb950; font-weight: 600;"
            if row.name == best_name
            else ""
            for _ in row
        ]

    styled = comp_df.style.apply(highlight_best, axis=1).format("{:.2f}")
    st.dataframe(styled, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Trade log
# ─────────────────────────────────────────────
section_header("📋", "Trade Log")

trades_list = all_trades[active_name]
trades_df = trades_to_dataframe(trades_list)

if not trades_df.empty:
    col_tl1, col_tl2 = st.columns([4, 1])
    with col_tl1:
        st.markdown(
            f"<div style='color:#8b949e; font-size:13px; margin-bottom:8px;'>"
            f"Showing {len(trades_df)} trades for <strong style='color:#58a6ff'>{active_name}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )

    def style_pnl(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return "color: #3fb950; font-weight: 600;"
            elif val < 0:
                return "color: #f85149; font-weight: 600;"
        return ""

    styled_trades = trades_df.style.applymap(style_pnl, subset=["P&L (%)"])
    st.dataframe(styled_trades, use_container_width=True, hide_index=True)
else:
    st.info("No trades were executed for this strategy in the selected period.")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Export Section
# ─────────────────────────────────────────────
section_header("💾", "Export Results")
col_e1, col_e2, col_e3 = st.columns(3)

with col_e1:
    csv_backtest = export_to_csv(active_df)
    st.download_button(
        label="📥 Download Backtest Data",
        data=csv_backtest,
        file_name=f"{ticker}_{active_name.replace(' ', '_')}_backtest.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_e2:
    if not trades_df.empty:
        csv_trades = export_to_csv(trades_df.set_index("Trade #"))
        st.download_button(
            label="📥 Download Trade Log",
            data=csv_trades,
            file_name=f"{ticker}_{active_name.replace(' ', '_')}_trades.csv",
            mime="text/csv",
            use_container_width=True,
        )

with col_e3:
    if mode == "Compare All Strategies":
        comp_df = build_comparison_table(all_metrics)
        csv_comp = export_to_csv(comp_df)
        st.download_button(
            label="📥 Download Comparison",
            data=csv_comp,
            file_name=f"{ticker}_strategy_comparison.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Full metrics table (collapsed)
# ─────────────────────────────────────────────
with st.expander("📑 Full Metrics Report", expanded=False):
    full_metrics = metrics_to_dict(m)
    metrics_display_df = pd.DataFrame.from_dict(
        full_metrics, orient="index", columns=["Value"]
    )
    metrics_display_df.index.name = "Metric"
    st.dataframe(metrics_display_df, use_container_width=True)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align:center; padding: 16px 0; color: #30363d; font-size: 12px;'>
        QuantLab &nbsp;·&nbsp; Built with Streamlit, Plotly & yfinance
        &nbsp;·&nbsp; For educational purposes only — not financial advice.
    </div>
    """,
    unsafe_allow_html=True,
)
