"""
strategies.py
-------------
Implementation of quantitative trading strategies for the backtesting platform.

Strategies implemented
----------------------
1. MovingAverageCrossover  – Classic dual-SMA crossover system
2. RSIStrategy             – Momentum oscillator-based mean-reversion system
3. MACDStrategy            – MACD signal-line crossover system

Each strategy returns a standardised DataFrame with 'Signal' (+1 = Buy, -1 = Sell, 0 = Hold)
and 'Position' columns, plus any strategy-specific indicator columns for chart overlays.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Data class for trade records
# ─────────────────────────────────────────────
@dataclass
class Trade:
    """Represents a single completed trade."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: str          # 'LONG' or 'SHORT'
    pnl_pct: float = field(init=False)
    profitable: bool = field(init=False)

    def __post_init__(self):
        if self.direction == "LONG":
            self.pnl_pct = (self.exit_price - self.entry_price) / self.entry_price * 100
        else:
            self.pnl_pct = (self.entry_price - self.exit_price) / self.entry_price * 100
        self.profitable = self.pnl_pct > 0


# ─────────────────────────────────────────────
# Base strategy class
# ─────────────────────────────────────────────
class BaseStrategy:
    """
    Abstract base class for all trading strategies.

    Subclasses must implement `generate_signals(df)`.
    """

    name: str = "BaseStrategy"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals from OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame (must include at least 'Close').

        Returns
        -------
        pd.DataFrame
            DataFrame with appended 'Signal' and 'Position' columns.
        """
        raise NotImplementedError("Subclasses must implement generate_signals()")

    def extract_trades(self, df: pd.DataFrame) -> list[Trade]:
        """
        Extract a list of Trade objects from the signals DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame returned by generate_signals().

        Returns
        -------
        list[Trade]
            Chronological list of completed round-trip trades.
        """
        trades: list[Trade] = []
        in_trade = False
        entry_date = None
        entry_price = None

        for i in range(1, len(df)):
            prev_signal = df["Signal"].iloc[i - 1]
            curr_signal = df["Signal"].iloc[i]
            price = df["Close"].iloc[i]
            date = df.index[i]

            # Entry: transition from 0/−1 → +1
            if curr_signal == 1 and not in_trade:
                in_trade = True
                entry_date = date
                entry_price = price

            # Exit: transition from +1 → 0/−1
            elif curr_signal != 1 and in_trade:
                in_trade = False
                trades.append(
                    Trade(
                        entry_date=entry_date,
                        exit_date=date,
                        entry_price=entry_price,
                        exit_price=price,
                        direction="LONG",
                    )
                )

        # Close any open trade at end of data
        if in_trade:
            trades.append(
                Trade(
                    entry_date=entry_date,
                    exit_date=df.index[-1],
                    entry_price=entry_price,
                    exit_price=df["Close"].iloc[-1],
                    direction="LONG",
                )
            )

        return trades


# ─────────────────────────────────────────────
# 1. Moving Average Crossover Strategy
# ─────────────────────────────────────────────
class MovingAverageCrossover(BaseStrategy):
    """
    Dual Moving Average Crossover Strategy.

    Entry  : Short-period SMA crosses ABOVE long-period SMA (Golden Cross).
    Exit   : Short-period SMA crosses BELOW long-period SMA (Death Cross).

    Parameters
    ----------
    short_window : int
        Look-back period for the fast SMA (default: 20).
    long_window : int
        Look-back period for the slow SMA (default: 50).
    ma_type : str
        'SMA' for Simple Moving Average or 'EMA' for Exponential MA.
    """

    name = "Moving Average Crossover"

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        ma_type: str = "SMA",
    ):
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window.")
        self.short_window = short_window
        self.long_window = long_window
        self.ma_type = ma_type.upper()

    def _compute_ma(self, series: pd.Series, window: int) -> pd.Series:
        if self.ma_type == "EMA":
            return series.ewm(span=window, adjust=False).mean()
        return series.rolling(window=window).mean()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        close = data["Close"]

        short_label = f"{self.ma_type}_{self.short_window}"
        long_label = f"{self.ma_type}_{self.long_window}"

        data[short_label] = self._compute_ma(close, self.short_window)
        data[long_label] = self._compute_ma(close, self.long_window)

        # +1 when short MA is above long MA, else 0
        data["Signal"] = np.where(
            data[short_label] > data[long_label], 1, 0
        )

        # Detect crossover points for trade entries/exits
        data["Crossover"] = data["Signal"].diff()

        # Position: 1 = in trade, 0 = out of trade
        data["Position"] = data["Signal"]

        # Store indicator labels for chart reference
        data.attrs["short_ma_col"] = short_label
        data.attrs["long_ma_col"] = long_label
        data.attrs["strategy"] = self.name

        return data


# ─────────────────────────────────────────────
# 2. RSI Strategy
# ─────────────────────────────────────────────
class RSIStrategy(BaseStrategy):
    """
    Relative Strength Index (RSI) Mean-Reversion Strategy.

    Entry  : RSI drops below `oversold` threshold (default: 30) → Buy.
    Exit   : RSI rises above `overbought` threshold (default: 70) → Sell.

    Parameters
    ----------
    rsi_period : int
        Look-back window for RSI calculation (default: 14).
    oversold : float
        RSI level below which the asset is considered oversold (default: 30).
    overbought : float
        RSI level above which the asset is considered overbought (default: 70).
    """

    name = "RSI Strategy"

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ):
        if oversold >= overbought:
            raise ValueError("oversold must be strictly less than overbought.")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def _compute_rsi(self, close: pd.Series) -> pd.Series:
        """Compute RSI using exponential moving averages of gains and losses."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=self.rsi_period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=self.rsi_period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["RSI"] = self._compute_rsi(data["Close"])

        # State machine: track whether we are in a trade
        signals = np.zeros(len(data))
        in_trade = False

        for i in range(self.rsi_period, len(data)):
            rsi_val = data["RSI"].iloc[i]

            if not in_trade and rsi_val < self.oversold:
                in_trade = True
                signals[i] = 1          # Buy signal

            elif in_trade and rsi_val > self.overbought:
                in_trade = False
                signals[i] = -1         # Sell signal

            elif in_trade:
                signals[i] = 1          # Maintain long position

        data["Signal"] = signals
        data["Position"] = (data["Signal"] == 1).astype(int)
        data.attrs["strategy"] = self.name
        data.attrs["rsi_col"] = "RSI"

        return data


# ─────────────────────────────────────────────
# 3. MACD Strategy
# ─────────────────────────────────────────────
class MACDStrategy(BaseStrategy):
    """
    Moving Average Convergence Divergence (MACD) Strategy.

    Entry  : MACD line crosses above the Signal line (bullish crossover).
    Exit   : MACD line crosses below the Signal line (bearish crossover).

    Parameters
    ----------
    fast_period : int
        Fast EMA period (default: 12).
    slow_period : int
        Slow EMA period (default: 26).
    signal_period : int
        Signal line EMA period (default: 9).
    """

    name = "MACD Strategy"

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ):
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period.")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        close = data["Close"]

        # Compute MACD components
        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()

        data["MACD"] = ema_fast - ema_slow
        data["MACD_Signal"] = data["MACD"].ewm(
            span=self.signal_period, adjust=False
        ).mean()
        data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

        # Signal: 1 when MACD > Signal line, else 0
        data["Signal"] = np.where(data["MACD"] > data["MACD_Signal"], 1, 0)
        data["Crossover"] = data["Signal"].diff()
        data["Position"] = data["Signal"]

        data.attrs["strategy"] = self.name

        return data


# ─────────────────────────────────────────────
# Strategy factory
# ─────────────────────────────────────────────
STRATEGY_MAP = {
    "Moving Average Crossover": MovingAverageCrossover,
    "RSI Strategy": RSIStrategy,
    "MACD Strategy": MACDStrategy,
}


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """
    Factory function to instantiate a strategy by name.

    Parameters
    ----------
    name : str
        Strategy name (must be a key in STRATEGY_MAP).
    **kwargs
        Keyword arguments forwarded to the strategy constructor.

    Returns
    -------
    BaseStrategy
        An instance of the requested strategy.

    Raises
    ------
    ValueError
        If the strategy name is not recognised.
    """
    if name not in STRATEGY_MAP:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_MAP.keys())}"
        )
    return STRATEGY_MAP[name](**kwargs)


def run_backtest(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    initial_capital: float = 100_000.0,
    commission_pct: float = 0.001,
) -> pd.DataFrame:
    """
    Simulate a simple long-only backtest given a strategy's signals.

    Applies commission on each trade entry and exit.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with 'Signal' and 'Position' columns
        (returned by strategy.generate_signals()).
    strategy : BaseStrategy
        The strategy instance (used to read attrs).
    initial_capital : float
        Starting cash in base currency (default: 100,000).
    commission_pct : float
        Round-trip commission as a fraction of trade value (default: 0.1%).

    Returns
    -------
    pd.DataFrame
        Original DataFrame enriched with:
        - 'Market_Return'  : Daily buy-and-hold return
        - 'Strategy_Return': Daily strategy return (accounting for position & commission)
        - 'Equity_Curve'   : Portfolio value over time
        - 'BH_Equity'      : Buy-and-hold portfolio value over time
        - 'Drawdown'       : Rolling drawdown from peak equity
    """
    data = df.copy()
    close = data["Close"]

    # Daily market returns
    data["Market_Return"] = close.pct_change().fillna(0)

    # Strategy daily returns: position × market return
    data["Raw_Strategy_Return"] = data["Position"].shift(1).fillna(0) * data["Market_Return"]

    # Apply commission on trade transitions
    position_change = data["Position"].diff().abs().fillna(0)
    data["Commission"] = position_change * commission_pct
    data["Strategy_Return"] = data["Raw_Strategy_Return"] - data["Commission"]

    # Equity curves
    data["Equity_Curve"] = initial_capital * (1 + data["Strategy_Return"]).cumprod()
    data["BH_Equity"] = initial_capital * (1 + data["Market_Return"]).cumprod()

    # Drawdown
    rolling_max = data["Equity_Curve"].cummax()
    data["Drawdown"] = (data["Equity_Curve"] - rolling_max) / rolling_max * 100

    return data


def compare_strategies(
    df: pd.DataFrame,
    strategies: list[BaseStrategy],
    initial_capital: float = 100_000.0,
) -> dict[str, pd.DataFrame]:
    """
    Run multiple strategies on the same price data and return results.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    strategies : list[BaseStrategy]
        List of strategy instances to compare.
    initial_capital : float
        Starting portfolio value.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of strategy name → backtest result DataFrame.
    """
    results = {}
    for strat in strategies:
        signals_df = strat.generate_signals(df)
        backtest_df = run_backtest(signals_df, strat, initial_capital)
        results[strat.name] = backtest_df
    return results
