"""
analytics.py
------------
Performance analytics engine for quantitative backtesting.

Computes institutional-grade risk/return metrics and provides
helper functions for chart data preparation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Performance metrics dataclass
# ─────────────────────────────────────────────
@dataclass
class PerformanceMetrics:
    """Container for all computed performance metrics."""

    # Return metrics
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    bh_total_return_pct: float = 0.0       # Buy-and-hold comparison

    # Risk metrics
    annualized_volatility_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration_days: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0

    # Portfolio
    initial_capital: float = 100_000.0
    final_equity: float = 100_000.0
    peak_equity: float = 100_000.0


def calculate_metrics(
    backtest_df: pd.DataFrame,
    trades: list,
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.06,          # 6% annual (Indian T-bill proxy)
    trading_days: int = 252,
) -> PerformanceMetrics:
    """
    Compute a full suite of risk-adjusted performance metrics.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        DataFrame returned by strategies.run_backtest(), must include
        'Strategy_Return', 'Equity_Curve', 'BH_Equity', 'Drawdown'.
    trades : list[Trade]
        List of Trade objects from BaseStrategy.extract_trades().
    initial_capital : float
        Starting capital.
    risk_free_rate : float
        Annual risk-free rate for Sharpe/Sortino calculation.
    trading_days : int
        Number of trading days per year (252 for equities).

    Returns
    -------
    PerformanceMetrics
        Populated metrics dataclass.
    """
    m = PerformanceMetrics(initial_capital=initial_capital)

    returns = backtest_df["Strategy_Return"].dropna()
    equity = backtest_df["Equity_Curve"].dropna()

    if len(equity) == 0:
        return m

    # ── Return Metrics ────────────────────────────────────────────────
    m.final_equity = float(equity.iloc[-1])
    m.peak_equity = float(equity.max())
    m.total_return_pct = (m.final_equity / initial_capital - 1) * 100

    # Annualised return via CAGR formula
    n_years = len(returns) / trading_days
    if n_years > 0 and m.final_equity > 0:
        m.annualized_return_pct = (
            (m.final_equity / initial_capital) ** (1 / n_years) - 1
        ) * 100

    # Buy-and-hold comparison
    bh_equity = backtest_df["BH_Equity"].dropna()
    if len(bh_equity) > 0:
        m.bh_total_return_pct = (bh_equity.iloc[-1] / initial_capital - 1) * 100

    # ── Risk Metrics ──────────────────────────────────────────────────
    daily_rf = risk_free_rate / trading_days
    excess_returns = returns - daily_rf

    m.annualized_volatility_pct = float(returns.std() * np.sqrt(trading_days) * 100)

    # Sharpe Ratio
    if returns.std() > 0:
        m.sharpe_ratio = float(
            excess_returns.mean() / returns.std() * np.sqrt(trading_days)
        )

    # Sortino Ratio (uses downside deviation)
    downside_returns = returns[returns < daily_rf]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        m.sortino_ratio = float(
            excess_returns.mean() / downside_returns.std() * np.sqrt(trading_days)
        )

    # Maximum Drawdown
    drawdown_series = backtest_df["Drawdown"].dropna()
    if len(drawdown_series) > 0:
        m.max_drawdown_pct = float(drawdown_series.min())   # negative value

        # Drawdown duration: longest consecutive drawdown period
        in_drawdown = drawdown_series < 0
        max_dur = 0
        cur_dur = 0
        for val in in_drawdown:
            if val:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0
        m.max_drawdown_duration_days = max_dur

    # Calmar Ratio
    if m.max_drawdown_pct != 0:
        m.calmar_ratio = float(m.annualized_return_pct / abs(m.max_drawdown_pct))

    # ── Trade Statistics ──────────────────────────────────────────────
    m.total_trades = len(trades)

    if trades:
        pnls = [t.pnl_pct for t in trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p <= 0]

        m.winning_trades = len(winning)
        m.losing_trades = len(losing)
        m.win_rate_pct = (m.winning_trades / m.total_trades * 100) if m.total_trades else 0

        m.avg_win_pct = float(np.mean(winning)) if winning else 0.0
        m.avg_loss_pct = float(np.mean(losing)) if losing else 0.0

        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        m.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        m.best_trade_pct = float(max(pnls))
        m.worst_trade_pct = float(min(pnls))

        durations = [
            (t.exit_date - t.entry_date).days for t in trades
            if hasattr(t.exit_date, "days") or isinstance(t.exit_date, pd.Timestamp)
        ]
        if durations:
            m.avg_trade_duration_days = float(np.mean(durations))

    return m


def metrics_to_dict(m: PerformanceMetrics) -> dict:
    """
    Convert a PerformanceMetrics instance to a flat dictionary for display.

    Returns
    -------
    dict
        Human-readable metrics with formatted labels as keys.
    """
    return {
        "Total Return (%)": f"{m.total_return_pct:.2f}%",
        "Annualized Return (%)": f"{m.annualized_return_pct:.2f}%",
        "Buy & Hold Return (%)": f"{m.bh_total_return_pct:.2f}%",
        "Annualized Volatility (%)": f"{m.annualized_volatility_pct:.2f}%",
        "Sharpe Ratio": f"{m.sharpe_ratio:.3f}",
        "Sortino Ratio": f"{m.sortino_ratio:.3f}",
        "Calmar Ratio": f"{m.calmar_ratio:.3f}",
        "Max Drawdown (%)": f"{m.max_drawdown_pct:.2f}%",
        "Max Drawdown Duration (days)": str(m.max_drawdown_duration_days),
        "Total Trades": str(m.total_trades),
        "Winning Trades": str(m.winning_trades),
        "Losing Trades": str(m.losing_trades),
        "Win Rate (%)": f"{m.win_rate_pct:.1f}%",
        "Avg Win (%)": f"{m.avg_win_pct:.2f}%",
        "Avg Loss (%)": f"{m.avg_loss_pct:.2f}%",
        "Profit Factor": f"{m.profit_factor:.2f}",
        "Best Trade (%)": f"{m.best_trade_pct:.2f}%",
        "Worst Trade (%)": f"{m.worst_trade_pct:.2f}%",
        "Final Equity": f"${m.final_equity:,.0f}",
        "Peak Equity": f"${m.peak_equity:,.0f}",
    }


def find_best_strategy(metrics_dict: dict[str, PerformanceMetrics]) -> str:
    """
    Identify the best-performing strategy by Sharpe Ratio.

    Falls back to total return if Sharpe ratios are equal.

    Parameters
    ----------
    metrics_dict : dict[str, PerformanceMetrics]
        Mapping of strategy name → PerformanceMetrics.

    Returns
    -------
    str
        Name of the best-performing strategy.
    """
    if not metrics_dict:
        return ""
    return max(
        metrics_dict.keys(),
        key=lambda k: (
            metrics_dict[k].sharpe_ratio,
            metrics_dict[k].total_return_pct,
        ),
    )


def build_comparison_table(
    metrics_dict: dict[str, PerformanceMetrics]
) -> pd.DataFrame:
    """
    Build a side-by-side comparison DataFrame for multiple strategies.

    Parameters
    ----------
    metrics_dict : dict[str, PerformanceMetrics]
        Mapping of strategy name → PerformanceMetrics.

    Returns
    -------
    pd.DataFrame
        DataFrame with strategies as columns and metrics as rows.
    """
    rows = {}
    for name, m in metrics_dict.items():
        rows[name] = {
            "Total Return (%)": round(m.total_return_pct, 2),
            "Annualized Return (%)": round(m.annualized_return_pct, 2),
            "Sharpe Ratio": round(m.sharpe_ratio, 3),
            "Sortino Ratio": round(m.sortino_ratio, 3),
            "Max Drawdown (%)": round(m.max_drawdown_pct, 2),
            "Win Rate (%)": round(m.win_rate_pct, 1),
            "Profit Factor": round(m.profit_factor, 2),
            "Total Trades": m.total_trades,
        }
    return pd.DataFrame(rows).T


def compute_rolling_metrics(
    backtest_df: pd.DataFrame,
    window: int = 252,
    risk_free_rate: float = 0.06,
) -> pd.DataFrame:
    """
    Compute rolling Sharpe Ratio and rolling volatility.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Backtest result DataFrame with 'Strategy_Return'.
    window : int
        Rolling window in trading days (default: 252 = 1 year).
    risk_free_rate : float
        Annual risk-free rate.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Rolling_Sharpe' and 'Rolling_Vol' columns.
    """
    returns = backtest_df["Strategy_Return"]
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf

    rolling_mean = excess.rolling(window).mean()
    rolling_std = returns.rolling(window).std()

    result = backtest_df.copy()
    result["Rolling_Sharpe"] = (rolling_mean / rolling_std) * np.sqrt(252)
    result["Rolling_Vol"] = rolling_std * np.sqrt(252) * 100

    return result


def trades_to_dataframe(trades: list) -> pd.DataFrame:
    """
    Convert a list of Trade objects into a formatted DataFrame.

    Parameters
    ----------
    trades : list[Trade]
        Trade objects.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame suitable for display/export.
    """
    if not trades:
        return pd.DataFrame()

    records = []
    for i, t in enumerate(trades, start=1):
        records.append(
            {
                "Trade #": i,
                "Direction": t.direction,
                "Entry Date": str(t.entry_date)[:10],
                "Exit Date": str(t.exit_date)[:10],
                "Entry Price": round(t.entry_price, 2),
                "Exit Price": round(t.exit_price, 2),
                "P&L (%)": round(t.pnl_pct, 2),
                "Profitable": "✅" if t.profitable else "❌",
            }
        )
    return pd.DataFrame(records)
