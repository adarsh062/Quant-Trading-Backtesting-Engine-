"""
data_loader.py
--------------
Module for fetching and preprocessing historical stock market data
using the yfinance library. Supports both Indian (NSE/BSE) and US stocks.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Market suffix mappings for Indian exchanges
# ─────────────────────────────────────────────
INDIAN_SUFFIX = {
    "NSE": ".NS",
    "BSE": ".BO",
}

# Popular stock presets for quick selection
POPULAR_STOCKS = {
    "US Stocks": {
        "Apple (AAPL)": "AAPL",
        "Microsoft (MSFT)": "MSFT",
        "Tesla (TSLA)": "TSLA",
        "Amazon (AMZN)": "AMZN",
        "NVIDIA (NVDA)": "NVDA",
        "Alphabet (GOOGL)": "GOOGL",
        "Meta (META)": "META",
        "JPMorgan (JPM)": "JPM",
        "S&P 500 ETF (SPY)": "SPY",
        "NASDAQ ETF (QQQ)": "QQQ",
    },
    "Indian Stocks (NSE)": {
        "Reliance Industries": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "Infosys": "INFY.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "Wipro": "WIPRO.NS",
        "Bajaj Finance": "BAJFINANCE.NS",
        "Axis Bank": "AXISBANK.NS",
        "Nifty 50 ETF": "NIFTYBEES.NS",
        "Sensex ETF": "SETFNIFBK.NS",
    },
}


def format_indian_ticker(ticker: str, exchange: str = "NSE") -> str:
    """
    Append the appropriate exchange suffix to an Indian stock ticker.

    Parameters
    ----------
    ticker : str
        Base ticker symbol (e.g., 'RELIANCE').
    exchange : str
        Exchange code – 'NSE' or 'BSE'.

    Returns
    -------
    str
        Formatted ticker symbol (e.g., 'RELIANCE.NS').
    """
    suffix = INDIAN_SUFFIX.get(exchange.upper(), ".NS")
    if not ticker.endswith(suffix):
        return f"{ticker.upper()}{suffix}"
    return ticker.upper()


def fetch_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV historical data for a given ticker symbol.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS').
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    interval : str
        Data interval – '1d', '1wk', '1mo', etc.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume.

    Raises
    ------
    ValueError
        If the ticker is invalid or no data is returned.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
        )

        if df is None or df.empty:
            raise ValueError(
                f"No data returned for ticker '{ticker}'. "
                "Please verify the symbol and date range."
            )

        # Standardize column names
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        # Keep only OHLCV columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in required_cols if c in df.columns]]

        # Drop rows with all NaN OHLC values
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

        if len(df) < 30:
            raise ValueError(
                f"Insufficient data for '{ticker}': only {len(df)} rows returned. "
                "Try extending the date range."
            )

        return df

    except Exception as e:
        raise ValueError(f"Error fetching data for '{ticker}': {str(e)}")


def get_ticker_info(ticker: str) -> dict:
    """
    Retrieve basic metadata for a ticker (name, sector, currency, etc.).

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.

    Returns
    -------
    dict
        Dictionary with ticker metadata fields.
    """
    try:
        info = yf.Ticker(ticker).info
        return {
            "name": info.get("longName") or info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
            "market_cap": info.get("marketCap", None),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice", None),
        }
    except Exception:
        return {
            "name": ticker,
            "sector": "N/A",
            "industry": "N/A",
            "currency": "N/A",
            "exchange": "N/A",
            "market_cap": None,
            "current_price": None,
        }


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute common technical indicators and append them to the DataFrame.

    Indicators added
    ----------------
    - SMA_20, SMA_50, SMA_200  : Simple Moving Averages
    - EMA_12, EMA_26           : Exponential Moving Averages
    - RSI_14                   : Relative Strength Index
    - MACD, MACD_Signal, MACD_Hist : MACD components
    - BB_Upper, BB_Middle, BB_Lower : Bollinger Bands
    - ATR_14                   : Average True Range
    - Daily_Return             : Daily percentage return
    - Cumulative_Return        : Cumulative return from first date

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame (must contain at least Close, High, Low).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional indicator columns.
    """
    data = df.copy()
    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    # ── Moving Averages ────────────────────────────────────────────────
    for window in [20, 50, 200]:
        data[f"SMA_{window}"] = close.rolling(window=window).mean()

    data["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    data["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # ── MACD ──────────────────────────────────────────────────────────
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    # ── RSI ───────────────────────────────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # ── Bollinger Bands ───────────────────────────────────────────────
    bb_mid = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    data["BB_Upper"] = bb_mid + 2 * bb_std
    data["BB_Middle"] = bb_mid
    data["BB_Lower"] = bb_mid - 2 * bb_std

    # ── ATR ───────────────────────────────────────────────────────────
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["ATR_14"] = tr.rolling(window=14).mean()

    # ── Returns ───────────────────────────────────────────────────────
    data["Daily_Return"] = close.pct_change()
    data["Cumulative_Return"] = (1 + data["Daily_Return"]).cumprod() - 1

    return data


def validate_date_range(start_date: str, end_date: str) -> tuple[bool, str]:
    """
    Validate that a date range is logically consistent.

    Parameters
    ----------
    start_date : str
        Start date string in 'YYYY-MM-DD' format.
    end_date : str
        End date string in 'YYYY-MM-DD' format.

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message) where error_message is empty if valid.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        today = datetime.today()

        if start >= end:
            return False, "Start date must be before end date."
        if end > today:
            return False, "End date cannot be in the future."
        if (end - start).days < 30:
            return False, "Date range must span at least 30 days."

        return True, ""
    except ValueError as e:
        return False, f"Invalid date format: {str(e)}"


def export_to_csv(df: pd.DataFrame, filename: str = "backtest_results.csv") -> bytes:
    """
    Serialize a DataFrame to CSV bytes for Streamlit download.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export.
    filename : str
        Suggested filename (unused here; handled by Streamlit caller).

    Returns
    -------
    bytes
        UTF-8 encoded CSV bytes.
    """
    return df.to_csv(index=True).encode("utf-8")
