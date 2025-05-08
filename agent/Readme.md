# Crypto Sentiment Trading System

A trading system that uses cryptocurrency Reddit sentiment to identify and trade correlated technology stocks.

## Core Features

- Analyzes Reddit r/CryptoCurrency sentiment using FinBERT
- Automatically selects tech stocks with highest crypto-sentiment correlation
- Integrates macroeconomic data (FRED) and DeFi market data for comprehensive analysis
- Combines sentiment signals, technical indicators, and ML predictions
- Supports GRU and sklearn prediction models
- Generates interactive dashboard and performance reports

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

# Create required directories
mkdir -p agent/models output
```

## Configuration

Create a file named `fin580.env` with your API keys:

```
# Required settings
START_DATE=2020-01-01
END_DATE=2025-01-01
REDDIT_SUBREDDIT=CryptoCurrency
OUTPUT_DIR=output

# API keys
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
2. Analyze Reddit posts
3. Generate trading signals
4. Simulate portfolio performance
5. Create performance report and dashboard

## System Architecture

- **main.py**: System entry point
- **tradingsystem.py**: Main orchestrator
- **data_processor.py**: Data loading and preprocessing
- **sentiment_analyzer.py**: Reddit sentiment analysis
- **analysis_strategy.py**: Signal generation and strategy
- **portfolio_manager.py**: Portfolio simulation and reporting

## Requirements

- Python 3.8+
- Reddit API access
- FinancialModelingPrep API key
- FRED API key (optional)

## Acknowledgements

This project was jointly developed by Celeste (Yueying) Huang and Stella (Lechen) Gong.