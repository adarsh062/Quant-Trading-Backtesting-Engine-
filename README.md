# 📈 QuantLab – Professional Quant Trading Backtesting Platform

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.19%2B-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A production-grade quantitative backtesting platform for algorithmic trading strategies on US and Indian equity markets.**

[Features](#features) · [Architecture](#architecture) · [Installation](#installation) · [Usage](#usage) · [Strategies](#strategies) · [Roadmap](#roadmap)

</div>

---

## 🧭 Project Overview

**QuantLab** is a full-stack backtesting platform built entirely in Python, designed to simulate algorithmic trading strategies on real historical market data. It provides institutional-quality performance analytics and interactive visualizations — all within a sleek, dark-themed web interface.

The project demonstrates core competencies in:

- **Quantitative Finance** – strategy design, risk/return attribution, drawdown analysis
- **Financial Engineering** – backtesting engine, commission modelling, signal generation
- **Data Science** – time-series processing, statistical metrics, portfolio analytics
- **Software Engineering** – modular architecture, clean APIs, production-ready code

---

## ✨ Features

### 📊 Market Data
| Feature | Details |
|---|---|
| Data Source | Yahoo Finance via `yfinance` |
| US Stocks | AAPL, MSFT, TSLA, NVDA, AMZN, GOOGL, META, JPM, SPY, QQQ |
| Indian Stocks | RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS, WIPRO.NS + more |
| Custom Tickers | Any valid Yahoo Finance symbol |
| Date Range | Fully configurable with validation |
| Data Interval | Daily (1d) |

### 🎯 Trading Strategies

#### 1. Moving Average Crossover
- Configurable short/long windows (5–200 days)
- SMA or EMA variant selectable
- Golden/Death Cross entry & exit logic

#### 2. RSI Mean-Reversion
- Configurable RSI period (7–21 days)
- Custom oversold (10–40) and overbought (60–90) thresholds
- State-machine position tracking

#### 3. MACD Signal-Line Crossover
- Configurable fast (5–20), slow (15–50), and signal (5–15) periods
- Histogram-based momentum confirmation
- Bullish/bearish crossover detection

### 📈 Performance Analytics

| Metric | Description |
|---|---|
| Total Return (%) | Overall portfolio return |
| Annualized Return (%) | CAGR over the backtest period |
| Sharpe Ratio | Risk-adjusted return vs risk-free rate |
| Sortino Ratio | Downside risk-adjusted return |
| Calmar Ratio | CAGR / Max Drawdown |
| Max Drawdown (%) | Worst peak-to-trough loss |
| Win Rate (%) | % of profitable trades |
| Profit Factor | Gross profit / Gross loss |
| Avg Win / Avg Loss | Mean returns on winning vs losing trades |
| Total Trades | Number of completed round-trips |
| Buy & Hold Comparison | Strategy vs passive investing benchmark |

### 📉 Visualizations
1. **Candlestick Chart** – OHLC price + buy/sell signal markers + MA overlays + Bollinger Bands
2. **Equity Curve** – Portfolio value over time vs buy-and-hold baseline
3. **Drawdown Chart** – Waterfall-style rolling drawdown from equity peak
4. **RSI Panel** – Oscillator with overbought/oversold bands
5. **MACD Panel** – MACD line, signal line, and colored histogram bars
6. **Monthly Returns Heatmap** – Calendar-style heat map of monthly P&L
7. **Trade P&L Distribution** – Histogram of per-trade returns
8. **Strategy Comparison Bar Chart** – Multi-metric grouped bar comparison

### 🏆 Multi-Strategy Comparison
- Run all 3 strategies simultaneously on the same data
- Automatic best-strategy detection (by Sharpe Ratio)
- Side-by-side comparison table with colour highlights
- Combined equity curve overlay for visual comparison

### 💾 Export
- Download backtest data as CSV
- Download trade log as CSV
- Download multi-strategy comparison as CSV

---

## 🏗️ Architecture

```
quant-backtester/
│
├── app.py              # Streamlit UI – layout, controls, rendering
├── data_loader.py      # yfinance data fetching & technical indicators
├── strategies.py       # Strategy classes + backtesting engine
├── analytics.py        # Performance metrics & comparison utilities
├── charts.py           # Plotly chart builders (dark theme)
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── assets/             # Logo, screenshots
```

### Data Flow

```
User Input (Sidebar)
        │
        ▼
data_loader.fetch_stock_data()
        │
        ▼
strategies.generate_signals()   ──►  strategies.run_backtest()
        │                                       │
        ▼                                       ▼
strategies.extract_trades()      analytics.calculate_metrics()
        │                                       │
        └────────────────┬──────────────────────┘
                         ▼
              charts.*  (Plotly figures)
                         │
                         ▼
              Streamlit Dashboard
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `data_loader.py` | Fetching OHLCV data, computing 10+ technical indicators, ticker validation |
| `strategies.py` | Strategy signal generation, backtesting engine, commission modelling, trade extraction |
| `analytics.py` | Sharpe/Sortino/Calmar ratios, drawdown analysis, trade statistics, comparison tables |
| `charts.py` | 8 interactive Plotly charts, consistent dark theme, buy/sell signal overlays |
| `app.py` | Streamlit UI, sidebar controls, metric cards, tab layout, CSV export |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip (or conda)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/quant-backtester.git
cd quant-backtester

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501` in your default browser.

---

## 🚀 Usage

1. **Select Market** – Choose US Stocks, Indian Stocks (NSE), or enter a custom ticker
2. **Set Date Range** – Pick your historical backtesting window (min 30 days)
3. **Choose Strategy** – Single strategy or compare all three simultaneously
4. **Configure Parameters** – Tune MA windows, RSI thresholds, MACD periods via sliders
5. **Set Portfolio** – Define initial capital and commission percentage
6. **Run Backtest** – Click the "🚀 Run Backtest" button
7. **Analyse Results** – Explore metrics, charts, and trade log across tabs
8. **Export** – Download results as CSV files

---

## 📐 Strategies In Depth

### Moving Average Crossover
```
Entry  : Short MA crosses ABOVE Long MA  (Golden Cross → BUY)
Exit   : Short MA crosses BELOW Long MA  (Death Cross  → SELL)
Default: SMA(20) vs SMA(50)
```

### RSI Mean-Reversion
```
Entry  : RSI < 30  (Oversold  → BUY)
Exit   : RSI > 70  (Overbought → SELL)
Default: RSI(14), Oversold=30, Overbought=70
```

### MACD Crossover
```
Entry  : MACD line crosses ABOVE Signal line  (Bullish → BUY)
Exit   : MACD line crosses BELOW Signal line  (Bearish → SELL)
Default: EMA(12) − EMA(26), Signal EMA(9)
```

---

## 📸 Screenshots

> *Launch the app and run a backtest to see the live interactive dashboard.*

| Section | Description |
|---|---|
| Sidebar | Strategy selection, parameter sliders, portfolio settings |
| Metrics Row | 12 colour-coded KPI cards (green=positive, red=negative) |
| Price Chart | Candlestick + MA overlays + buy/sell triangle markers |
| Equity Curve | Multi-strategy comparison + buy-and-hold baseline |
| Drawdown | Waterfall area chart of portfolio drawdowns |
| Heatmap | Monthly returns calendar in red/green |
| Trade Log | Sortable table of all trades with P&L colouring |

---

## 🔮 Future Improvements

- [ ] **Short-selling support** – Allow bearish positions in strategies
- [ ] **Portfolio backtesting** – Multi-asset allocation & rebalancing
- [ ] **Options strategies** – Covered calls, protective puts
- [ ] **Walk-forward optimisation** – Avoid in-sample overfitting
- [ ] **Monte Carlo simulation** – Confidence intervals on equity curves
- [ ] **Live paper trading** – Connect to broker API (Zerodha Kite, Alpaca)
- [ ] **Sentiment analysis** – News/social media signal integration
- [ ] **ML-based strategies** – LSTM price prediction, Random Forest signals
- [ ] **Sector rotation** – Relative strength momentum across sectors
- [ ] **Risk management** – Position sizing, stop-loss, take-profit rules
- [ ] **Factor models** – Fama-French 3/5 factor attribution

---

## 🧑‍💻 Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| Streamlit | 1.32+ | Web application framework |
| Pandas | 2.1+ | Data manipulation & time-series |
| NumPy | 1.26+ | Numerical computing |
| yfinance | 0.2.36+ | Historical market data API |
| Plotly | 5.19+ | Interactive visualizations |

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## ⚠️ Disclaimer

This platform is built for **educational and research purposes only**. Past performance of backtested strategies does **not** guarantee future results. Do not use this software for actual financial trading decisions without consulting a qualified financial advisor.

---

<div align="center">

Built with ❤️ for Quant Research internship applications — Futures First · Graviton · QuadeEye · Tower Research · IMC Trading

</div>
