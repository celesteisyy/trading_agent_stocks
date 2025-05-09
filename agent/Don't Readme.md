# Crypto Sentiment-Driven Trading System

A comprehensive trading system that leverages cryptocurrency Reddit sentiment to identify and trade correlated technology stocks. The system combines natural language processing, technical analysis, and machine learning to generate trading signals and simulate portfolio performance.

## Overview

This trading system analyzes sentiment from cryptocurrency discussions on Reddit, identifies technology stocks that exhibit correlation with this sentiment, and executes trades based on a combination of sentiment signals, technical indicators, and predictive models. The system provides full backtesting capabilities, performance metrics, and interactive visualization.

## Key Features

- **Crypto-to-Tech Stock Correlation**: Automatically selects technology stocks with the highest correlation to cryptocurrency sentiment
- **Advanced Sentiment Analysis**: Uses FinBERT transformer models for finance-specific sentiment analysis
- **Multi-source Signal Integration**: Combines Reddit sentiment, technical indicators, and ML predictions with configurable weights
- **Machine Learning Support**: Compatible with GRU deep learning models and sklearn models
- **Comprehensive Technical Analysis**: Calculates SMA, EMA, RSI, MACD, Bollinger Bands, and other indicators
- **Portfolio Simulation**: Simulates trading performance with transaction costs and trading rules
- **Interactive Dashboard**: Visualizes performance metrics, sentiment trends, and trade signals
- **DOCX Report Generation**: Creates detailed performance reports with metrics and charts

## System Architecture

```
┌─ main.py ───────────────┐
│  Simple entry point     │
│  Loads configuration    │
│  Initializes system     │
└───────────┬─────────────┘
            ▼
┌─ tradingsystem.py ──────┐
│  Master orchestrator    │
│  Coordinates workflow   │
│  Manages pipeline flow  │
└┬──────┬──────┬──────┬───┘
 │      │      │      │
 ▼      ▼      ▼      ▼
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│Data │ │Sent.│ │Anal.│ │Port.│
│Proc.│ │Anal.│ │Strat│ │Mgr. │
└─────┘ └─────┘ └─────┘ └─────┘
```

- **data_processor.py**: Handles data loading and preprocessing
- **sentiment_analyzer.py**: Analyzes sentiment from Reddit posts
- **analysis_strategy.py**: Generates trading signals
- **portfolio_manager.py**: Simulates portfolio and generates reports

## Installation

### Prerequisites
- Python 3.8 or higher
- API keys for:
  - Reddit (for sentiment analysis)
  - FinancialModelingPrep (for stock data)
  - FRED (for economic data)

### Setup

* Clone the repository:
   ```bash
   git clone https://github.com/yourusername/crypto-sentiment-trading.git
   cd crypto-sentiment-trading
   ```

* Create a configuration file named `fin580.env` with your API keys:
   ```
   # Trading system configuration
   START_DATE=2024-01-01
   END_DATE=2024-12-31
   MODEL_TYPE=gru
   SENTIMENT_METHOD=transformer
   MIN_HOLDING_PERIOD=5
   MAX_TRADES_PER_WEEK=3
   INITIAL_CAPITAL=100000.0
   OUTPUT_DIR=output
   REDDIT_SUBREDDIT=CryptoCurrency
   USE_SENTIMENT_STRATEGY=true
   SENTIMENT_THRESHOLD=0.3
   SENTIMENT_CHANGE_THRESHOLD=0.1
   NUM_STOCKS=5
   SENTIMENT_WEIGHT=0.6
   TECHNICAL_WEIGHT=0.4

   # API keys (required)
   FRED_API_KEY=your_fred_api_key
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=your_reddit_client_secret
   REDDIT_USER_AGENT=your_reddit_user_agent
   FMP_API_KEY=your_fmp_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Basic Usage

Run the trading system with the configuration from your `fin580.env` file:

```bash
python main.py
```

This will execute the full trading pipeline, including:
1. Selecting technology stocks with highest crypto sentiment correlation
2. Analyzing Reddit posts for sentiment
3. Generating trading signals
4. Simulating portfolio performance
5. Creating a performance report
6. Launching an interactive dashboard

### Using a GRU Model

To use a GRU deep learning model for price prediction, place your trained PyTorch model at:
```
agent/models/gru_model.pt
```

If the GRU model is not found, the system will automatically fall back to a simpler sklearn model.

### Custom Configuration

You can modify the `fin580.env` file to customize:
- Date ranges
- Sentiment methods and thresholds
- Trading parameters
- Signal weights
- Initial capital

## Module Details

### 1. main.py
Simple entry point that loads configuration, initializes the TradingSystem, and runs it.

### 2. tradingsystem.py
Master orchestrator that coordinates the entire workflow, manages component initialization, and executes the pipeline.

### 3. data_processor.py
Handles data loading and preprocessing, including:
- Stock price data with fallbacks (yfinance → FMP API → local CSV)
- Reddit posts from specified subreddits
- Sector-based ticker selection
- Data cleaning and normalization

### 4. sentiment_analyzer.py
Specializes in sentiment analysis from text data:
- Supports FinBERT transformer models and VADER
- Handles text chunking for long Reddit posts
- Aggregates sentiment scores by day
- Correlates sentiment with price movements

### 5. analysis_strategy.py
Combines technical analysis and trading strategy:
- Calculates technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Integrates ML model predictions (GRU or sklearn)
- Generates weighted signals from multiple sources
- Applies trading rules (holding periods, frequency limits)

### 6. portfolio_manager.py
Simulates portfolio performance and generates visualizations:
- Backtests trading strategies with transaction costs
- Calculates comprehensive performance metrics
- Generates DOCX reports with charts
- Creates interactive dashboards with Gradio

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| START_DATE | Start date for analysis | 2023-01-01 |
| END_DATE | End date for analysis | 2023-12-31 |
| MODEL_TYPE | Prediction model type ('gru' or 'sklearn') | gru |
| SENTIMENT_METHOD | Sentiment analysis method ('transformer' or 'vader') | transformer |
| MIN_HOLDING_PERIOD | Minimum days to hold a position | 5 |
| MAX_TRADES_PER_WEEK | Maximum trades allowed per 7-day period | 3 |
| INITIAL_CAPITAL | Starting portfolio value | 100000.0 |
| OUTPUT_DIR | Directory for saving reports | output |
| REDDIT_SUBREDDIT | Subreddit to analyze for sentiment | CryptoCurrency |
| SENTIMENT_THRESHOLD | Minimum sentiment score to consider | 0.3 |
| SENTIMENT_CHANGE_THRESHOLD | Minimum change in sentiment for signal | 0.1 |
| NUM_STOCKS | Number of stocks to select | 5 |
| SENTIMENT_WEIGHT | Weight for sentiment signals (0-1) | 0.6 |
| TECHNICAL_WEIGHT | Weight for technical signals (0-1) | 0.4 |

## Dependencies

The system requires the following key dependencies:
- pandas, numpy: Data handling and numerical calculations
- yfinance: Yahoo Finance data download
- praw: Reddit API wrapper
- transformers, torch: For FinBERT sentiment analysis
- nltk: For VADER sentiment analysis
- matplotlib: For visualization
- gradio: For interactive dashboards
- python-docx: For report generation
- ccxt: For cryptocurrency data (optional)
- scipy: For statistical calculations

See `requirements.txt` for the full list of dependencies.

## Example Dashboard

The system generates an interactive dashboard with multiple views:

1. **Performance Metrics**: Key metrics including returns, Sharpe ratio, and drawdown
2. **Equity Curve**: Portfolio value over time
3. **Drawdown Chart**: Visualization of portfolio drawdowns
4. **Sentiment vs. Price**: Comparison of Reddit sentiment and stock price
5. **Trade Signals**: Buy/sell signal visualization
6. **Returns Distribution**: Histogram of daily returns
7. **Recent Trades**: Table of most recent trades

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.