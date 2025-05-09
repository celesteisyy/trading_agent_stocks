# Crypto Sentiment Trading System

A trading system that uses cryptocurrency Reddit sentiment to identify and trade correlated technology stocks, enriched with macroeconomic and DeFi data analysis.

## Core Features

- Analyzes Reddit r/CryptoCurrency sentiment using FinBERT
- Automatically selects tech stocks with highest crypto-sentiment correlation
- Integrates macroeconomic data (FRED) and DeFi market data for comprehensive analysis
- Combines sentiment signals, technical indicators, and ML predictions
- Supports GRU and sklearn prediction models
- Generates interactive dashboard and performance reports

## Overall Workflow

Since we often get fellows asking about the detailed working logic of this multi‑agent system, here’s a brief overview of its workflow.


## Repository Structure

```
# Execution Flow
├── main.py                     # 1. System entry point
├── fin580.env                  # 2. Configuration file with API keys
├── tradingsystem.py            # 3. Main orchestrator
├── data_processor.py           # 4. Data loading and preprocessing
├── sentiment_analyzer.py       # 5. Reddit sentiment analysis
├── analysis_strategy.py        # 6. Signal generation and strategy
├── portfolio_manager.py        # 7. Portfolio simulation and reporting

# Supporting Directories
.
├── main.py                     # Entry point: orchestrates data fetching, preprocessing, strategy execution, and reporting
├── fin580.env.example          # Example environment file (copy to fin580.env and fill in your keys)
├── README.md                   # Project overview and instructions
├── requirements.txt            # Python dependencies
├── data_processor.py           # Data fetching and cleaning logic
├── sentiment_analyzer.py       # Module for Reddit and other text sentiment scoring
├── analysis_strategy.py        # Trading signal generation strategies
├── gru_model.py                # GRU model definition and training
├── portfolio_manager.py        # Backtesting and portfolio management
├── tradingsystem.py            # High-level orchestration and execution
│
├── data/                       # Raw and example data
│   ├── AAPL.csv                # Sample stock price data
│   └── GSPC.csv                # Sample index price data
│
├── notebooks/                  # Jupyter notebooks for demos and testing
│   ├── crypto_data.ipynb
│   ├── fmp_localdata.ipynb
│   └── reddit_sent.ipynb
│
├── output/                     # Generated outputs: reports, logs, visualizations
│   └── trading_report.docx
│
├── .gitignore
└── LICENSE
```

```
Note:
The data/reddit/submissions/ folder (monthly .zst archives) is not included in this repository due to file‑size limits.
Please download those Reddit submission files separately (e.g. from the official archive) and place them under data/reddit/submissions/ before running the pipeline.
```

## Quick Installation

```bash
# Clone repository and move into directory
git clone https://github.com/yourusername/crypto-sentiment-trading.git
cd crypto-sentiment-trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a file named `fin580.env` with your API keys:

```
# Required settings
START_DATE=2024-01-01
END_DATE=2024-12-31
REDDIT_SUBREDDIT=CryptoCurrency
OUTPUT_DIR=output

# API keys (all required)
FRED_API_KEY=your_fred_api_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_reddit_user_agent
FMP_API_KEY=your_fmp_api_key
```

## Usage

Run the system:

```bash
python main.py
```

The system will:
1. Select tech stocks correlated with crypto sentiment
2. Analyze Reddit posts and process macroeconomic indicators
3. Collect DeFi market data for additional analytical support
4. Generate trading signals with multi-source data integration
5. Simulate portfolio performance
6. Create performance report and dashboard

## System Architecture

- **main.py**: System entry point
- **tradingsystem.py**: Main orchestrator
- **data_processor.py**: Data loading (stocks, crypto, macro, DeFi) and preprocessing
- **sentiment_analyzer.py**: Reddit sentiment analysis
- **analysis_strategy.py**: Signal generation and strategy with multi-source data
- **portfolio_manager.py**: Portfolio simulation and reporting

## Data Sources

- **Stock Data**: Yahoo Finance, FinancialModelingPrep API
- **Sentiment Data**: Reddit API (r/CryptoCurrency)
- **Macroeconomic Data**: FRED Economic Data API
- **DeFi Market Data**: CCXT library for cryptocurrency exchanges

## Requirements

- Python 3.8+
- Reddit API access
- FinancialModelingPrep API key
- FRED API key
- Internet access for cryptocurrency exchange data

## Acknowledgements

This project was jointly developed by Celeste (Yueying) Huang and Stella (Lechen) Gong.
Reddit data from: https://academictorrents.com/details/ba051999301b109eab37d16f027b3f49ade2de13/tech&filelist=1 (The raw data is too big to upload)
