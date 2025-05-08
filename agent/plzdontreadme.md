# Final Project: Financial & Alternative Data Pipeline

A Python project template for downloading and processing financial time series (stocks, benchmarks, crypto, DeFi, macro) and Reddit data for factor construction and predictive modeling.

## Repository Structure

```
agent/
├── agent_config.py         # Load environment variables and API keys
├── data_loader.py          # Fetch raw data (stock prices, macro series, crypto/Reddit posts)
├── preprocessor.py         # Clean and impute raw datasets
├── sentiment_analysis.py   # Convert crypto discussion posts into daily sentiment scores
├── feature_engineer.py     # All feature construction (returns, rolling stats, RSI, MACD, etc.)
├── analysis.py             # Compute technical indicators (SMA, EMA, RSI, MACD, BBANDS, ATR) and raw signals
├── forecast.py             # Load trained models (Sklearn or GRU), produce N‑step forecasts
├── strategy.py             # Apply trading rules (e.g. minimum holding period, max trades per week) to signals
├── portfolio.py            # Position sizing, trade execution, portfolio P&L, Sharpe ratio, max drawdown
├── report_generate.py      # Generate a DOCX report: LLM summary + equity‑curve plots
├── tradesystem.py          # Orchestrate full pipeline: load → preprocess → features → predict → strategy → portfolio → report
└── main.py                 # Entry point: instantiate TradingSystem and kick off `run()`

```

## Prerequisites

- Python 3.8+  
- A Reddit “script” app (client ID, secret, user‑agent)  
- FRED API key  
- CoinGecko API key 

### Install dependencies

```
pip install -r requirements.txt
```
