"""
charts.py
---------
Plotly chart builders for the Quant Backtesting Dashboard.

All charts use a consistent dark theme with the platform's color palette.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# Design constants
# ─────────────────────────────────────────────
THEME = {
    "bg": "#0d1117",
    "surface": "#161b22",
    "border": "#30363d",
    "text": "#e6edf3",
    "subtext": "#8b949e",
    "accent": "#58a6ff",
    "green": "#3fb950",
    "red": "#f85149",
    "yellow": "#d29922",
    "purple": "#bc8cff",
    "orange": "#ffa657",
    "grid": "rgba(48, 54, 61, 0.5)",
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=THEME["bg"],
    plot_bgcolor=THEME["surface"],
    font=dict(family="Inter, sans-serif", color=THEME["text"], size=12),
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(
        bgcolor="rgba(22,27,34,0.8)",
        bordercolor=THEME["border"],
        borderwidth=1,
    ),
    xaxis=dict(
        gridcolor=THEME["grid"],
        zerolinecolor=THEME["border"],
        showgrid=True,
    ),
    yaxis=dict(
        gridcolor=THEME["grid"],
        zerolinecolor=THEME["border"],
        showgrid=True,
    ),
)


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply the platform dark theme to any Plotly figure."""
    fig.update_layout(**LAYOUT_DEFAULTS, title=dict(
        text=title,
        font=dict(size=16, color=THEME["text"]),
        x=0.02,
    ))
    return fig


# ─────────────────────────────────────────────
# 1. Candlestick + signals chart
# ─────────────────────────────────────────────
def plot_price_signals(
    df: pd.DataFrame,
    strategy_name: str,
    show_ma: bool = True,
) -> go.Figure:
    """
    Plot candlestick chart with buy/sell signal markers and optional MAs.

    Parameters
    ----------
    df : pd.DataFrame
        Backtest result DataFrame with OHLCV, Signal, and MA columns.
    strategy_name : str
        Name of the strategy (used in chart title).
    show_ma : bool
        Whether to overlay moving average lines.
    """
    fig = make_subplots(
        rows=1, cols=1,
        shared_xaxes=True,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color=THEME["green"],
            decreasing_line_color=THEME["red"],
            increasing_fillcolor=THEME["green"],
            decreasing_fillcolor=THEME["red"],
        )
    )

    # Moving average overlays
    if show_ma:
        short_col = df.attrs.get("short_ma_col")
        long_col = df.attrs.get("long_ma_col")

        if short_col and short_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[short_col],
                    name=short_col,
                    line=dict(color=THEME["accent"], width=1.5, dash="solid"),
                )
            )
        if long_col and long_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[long_col],
                    name=long_col,
                    line=dict(color=THEME["orange"], width=1.5, dash="solid"),
                )
            )

        # Bollinger Bands if present
        if "BB_Upper" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["BB_Upper"],
                name="BB Upper", line=dict(color=THEME["purple"], width=1, dash="dot"),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df["BB_Lower"],
                name="BB Lower", line=dict(color=THEME["purple"], width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(188,140,255,0.05)",
                showlegend=False,
            ))

    # Buy signals
    buy_mask = df["Signal"] == 1
    buy_start = buy_mask & (~buy_mask.shift(1).fillna(False))
    buy_df = df[buy_start]
    if not buy_df.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_df.index,
                y=buy_df["Low"] * 0.98,
                mode="markers",
                name="Buy Signal",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color=THEME["green"],
                    line=dict(width=1, color="white"),
                ),
            )
        )

    # Sell signals
    sell_mask = df["Signal"] == -1
    sell_df = df[sell_mask]
    if not sell_df.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_df.index,
                y=sell_df["High"] * 1.02,
                mode="markers",
                name="Sell Signal",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color=THEME["red"],
                    line=dict(width=1, color="white"),
                ),
            )
        )

    # Detect position exits (in_trade → not in_trade) for RSI/MACD
    pos_exit = (df["Position"].shift(1) == 1) & (df["Position"] == 0)
    exit_df = df[pos_exit]
    if not exit_df.empty and sell_df.empty:
        fig.add_trace(
            go.Scatter(
                x=exit_df.index,
                y=exit_df["High"] * 1.02,
                mode="markers",
                name="Exit",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color=THEME["red"],
                    line=dict(width=1, color="white"),
                ),
            )
        )

    fig.update_layout(xaxis_rangeslider_visible=False)
    return _apply_theme(fig, f"📈 {strategy_name} – Price & Signals")


# ─────────────────────────────────────────────
# 2. Equity curve chart
# ─────────────────────────────────────────────
def plot_equity_curve(
    results: dict[str, pd.DataFrame],
    initial_capital: float = 100_000.0,
) -> go.Figure:
    """
    Plot equity curves for one or multiple strategies vs buy-and-hold.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Mapping of strategy_name → backtest DataFrame.
    initial_capital : float
        Starting portfolio value (for reference line).
    """
    fig = go.Figure()

    colours = [THEME["accent"], THEME["green"], THEME["orange"], THEME["purple"]]

    for i, (name, df) in enumerate(results.items()):
        if "Equity_Curve" not in df.columns:
            continue
        colour = colours[i % len(colours)]
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Equity_Curve"],
                name=name,
                line=dict(color=colour, width=2),
                mode="lines",
            )
        )

    # Buy-and-hold baseline (from first strategy)
    first_df = next(iter(results.values()), None)
    if first_df is not None and "BH_Equity" in first_df.columns:
        fig.add_trace(
            go.Scatter(
                x=first_df.index,
                y=first_df["BH_Equity"],
                name="Buy & Hold",
                line=dict(color=THEME["subtext"], width=1.5, dash="dash"),
                mode="lines",
            )
        )

    # Initial capital reference
    if first_df is not None:
        fig.add_hline(
            y=initial_capital,
            line_dash="dot",
            line_color=THEME["yellow"],
            annotation_text="Initial Capital",
            annotation_position="bottom right",
        )

    fig.update_yaxes(title_text="Portfolio Value ($)")
    return _apply_theme(fig, "💰 Equity Curve Comparison")


# ─────────────────────────────────────────────
# 3. Drawdown chart
# ─────────────────────────────────────────────
def plot_drawdown(df: pd.DataFrame, strategy_name: str) -> go.Figure:
    """
    Plot the portfolio drawdown curve (waterfall-style area chart).

    Parameters
    ----------
    df : pd.DataFrame
        Backtest DataFrame with 'Drawdown' column.
    strategy_name : str
        Strategy label for the title.
    """
    fig = go.Figure()

    if "Drawdown" not in df.columns:
        return fig

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Drawdown"],
            name="Drawdown",
            fill="tozeroy",
            line=dict(color=THEME["red"], width=1.5),
            fillcolor="rgba(248,81,73,0.2)",
            mode="lines",
        )
    )

    fig.add_hline(y=0, line_color=THEME["border"], line_width=1)

    fig.update_yaxes(title_text="Drawdown (%)", tickformat=".1f")
    return _apply_theme(fig, f"📉 {strategy_name} – Drawdown")


# ─────────────────────────────────────────────
# 4. RSI panel chart
# ─────────────────────────────────────────────
def plot_rsi(df: pd.DataFrame) -> go.Figure:
    """
    Plot RSI indicator with overbought/oversold threshold bands.
    """
    fig = go.Figure()

    if "RSI" not in df.columns:
        return fig

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["RSI"],
            name="RSI",
            line=dict(color=THEME["purple"], width=2),
        )
    )

    # Threshold bands
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,81,73,0.1)",
                  line_width=0, annotation_text="Overbought (70)")
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(63,185,80,0.1)",
                  line_width=0, annotation_text="Oversold (30)")
    fig.add_hline(y=70, line_dash="dot", line_color=THEME["red"], line_width=1)
    fig.add_hline(y=30, line_dash="dot", line_color=THEME["green"], line_width=1)
    fig.add_hline(y=50, line_dash="dot", line_color=THEME["subtext"], line_width=0.5)

    fig.update_yaxes(range=[0, 100], title_text="RSI")
    return _apply_theme(fig, "📊 RSI Indicator")


# ─────────────────────────────────────────────
# 5. MACD chart
# ─────────────────────────────────────────────
def plot_macd(df: pd.DataFrame) -> go.Figure:
    """
    Plot MACD line, signal line, and histogram.
    """
    fig = make_subplots(rows=1, cols=1)

    if "MACD" not in df.columns:
        return fig

    # Histogram coloring
    hist = df["MACD_Hist"]
    hist_colors = [THEME["green"] if v >= 0 else THEME["red"] for v in hist]

    fig.add_trace(go.Bar(
        x=df.index, y=hist,
        name="Histogram",
        marker_color=hist_colors,
        opacity=0.6,
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"],
        name="MACD",
        line=dict(color=THEME["accent"], width=2),
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_Signal"],
        name="Signal",
        line=dict(color=THEME["orange"], width=1.5, dash="dash"),
    ))

    fig.add_hline(y=0, line_color=THEME["border"], line_width=1)

    return _apply_theme(fig, "📊 MACD Indicator")


# ─────────────────────────────────────────────
# 6. Strategy comparison bar chart
# ─────────────────────────────────────────────
def plot_comparison_bars(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Plot grouped bar chart comparing strategies across key metrics.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame from analytics.build_comparison_table().
    """
    metrics_to_plot = [
        "Total Return (%)", "Annualized Return (%)", "Sharpe Ratio",
        "Max Drawdown (%)", "Win Rate (%)",
    ]
    available = [m for m in metrics_to_plot if m in comparison_df.columns]

    colours = [THEME["accent"], THEME["green"], THEME["orange"]]
    fig = go.Figure()

    for i, strategy in enumerate(comparison_df.index):
        fig.add_trace(go.Bar(
            name=strategy,
            x=available,
            y=[comparison_df.loc[strategy, m] for m in available],
            marker_color=colours[i % len(colours)],
            opacity=0.85,
        ))

    fig.update_layout(
        barmode="group",
        xaxis_title="Metric",
        yaxis_title="Value",
    )
    return _apply_theme(fig, "🏆 Strategy Comparison")


# ─────────────────────────────────────────────
# 7. Monthly returns heatmap
# ─────────────────────────────────────────────
def plot_monthly_returns_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Plot a calendar heatmap of monthly strategy returns.

    Parameters
    ----------
    df : pd.DataFrame
        Backtest DataFrame with 'Strategy_Return' indexed by date.
    """
    if "Strategy_Return" not in df.columns:
        return go.Figure()

    monthly = df["Strategy_Return"].resample("ME").apply(
        lambda x: (1 + x).prod() - 1
    ) * 100

    monthly_df = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values,
    })

    pivot = monthly_df.pivot_table(
        index="Year", columns="Month", values="Return", aggfunc="sum"
    )
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[
                [0.0, THEME["red"]],
                [0.5, THEME["surface"]],
                [1.0, THEME["green"]],
            ],
            zmid=0,
            text=np.round(pivot.values, 1),
            texttemplate="%{text}%",
            textfont=dict(size=10),
            hoverongaps=False,
        )
    )

    fig.update_yaxes(title_text="Year")
    return _apply_theme(fig, "📅 Monthly Returns Heatmap")


# ─────────────────────────────────────────────
# 8. Trade P&L distribution histogram
# ─────────────────────────────────────────────
def plot_trade_distribution(trades: list) -> go.Figure:
    """
    Plot histogram of per-trade P&L percentages.
    """
    if not trades:
        return go.Figure()

    pnls = [t.pnl_pct for t in trades]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pnls,
        nbinsx=30,
        name="Trade P&L",
        marker_color=[
            THEME["green"] if p >= 0 else THEME["red"] for p in pnls
        ],
        opacity=0.8,
    ))

    fig.add_vline(
        x=0, line_dash="dash",
        line_color=THEME["yellow"],
        annotation_text="Break-even",
    )

    avg_pnl = np.mean(pnls)
    fig.add_vline(
        x=avg_pnl, line_dash="dot",
        line_color=THEME["accent"],
        annotation_text=f"Avg: {avg_pnl:.1f}%",
    )

    fig.update_xaxes(title_text="P&L (%)")
    fig.update_yaxes(title_text="Number of Trades")
    return _apply_theme(fig, "📊 Trade P&L Distribution")
