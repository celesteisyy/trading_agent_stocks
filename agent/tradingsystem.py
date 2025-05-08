import warnings
import pandas as pd
import pickle
import os
import numpy as np
from data_loader import DataLoader
from preprocessor import Preprocessor
from feature_engineer import FeatureEngineer
from sentiment_analysis import SentimentAnalyzer
from analysis import AnalysisAgent
from forecast import ForecastAgent
from strategy import StrategyAgent
from portfolio import PortfolioManagerAgent
from report_generate import ReportGenerator

# Define RandomWalkModel directly in this file to match the class in forecast.py
class RandomWalkModel:
    """
    Simple model that generates forecasts based on random walk principles.
    This matches the implementation used in forecast.py's unit test.
    """
    def predict(self, X):
        """
        Predict next values by adding small random noise to the input values.
        X is expected to be a list of lists, where each inner list has one element.
        """
        # Add small random noise to last_val
        return [x[0] + np.random.randn() * 0.1 for x in X]

warnings.filterwarnings('ignore')

class TradingSystem:
    """
    Orchestrates the full trading pipeline by sequentially invoking:
      DataLoader → Preprocessor/FeatureEngineer → SentimentAnalyzer →
      ForecastAgent → StrategyAgent → PortfolioManagerAgent → ReportGenerator
    """
    def __init__(
        self,
        start_date: str,
        end_date: str,
        tickers: list,
        model_path: str,
        model_type: str = 'sklearn',
        sentiment_method: str = 'vader',
        min_holding: int = 5,
        max_trades: int = 3,
        initial_capital: float = 100000.0
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers
        self.model_path = model_path
        self.model_type = model_type
        self.sentiment_method = sentiment_method
        self.min_holding = min_holding
        self.max_trades = max_trades
        self.initial_capital = initial_capital

    def run(self):
        # 1. Load data (handles fallback internally)
        loader = DataLoader(self.start_date, self.end_date)
        price_df = loader.load_stock_prices(self.tickers)
        # If MultiIndex (ticker, field), extract field level
        if isinstance(price_df.columns, pd.MultiIndex):
            # Extract the second level (field names) and title-case
            price_df.columns = [col[1].strip().title() for col in price_df.columns]
        else:
            # Title-case simple column names
            price_df.columns = [str(c).strip().title() for c in price_df.columns]
        # Ensure required OHLC column exists
        if 'Close' not in price_df.columns:
            raise KeyError(f"price_df missing 'Close' after normalization: {price_df.columns.tolist()}")
        
        macro_df = loader.load_macro_series(['GDP', 'UNRATE'])
        reddit_df = loader.load_reddit_range(
            subreddit='CryptoCurrency',
            after=self.start_date,
            before=self.end_date,
            max_posts=500
        )

        # 2. Clean, impute, and create features
        fe = FeatureEngineer()
        feature_df = fe.fit_transform(price_df, macro_df)

        # 3. Sentiment scoring
        try:
            sa = SentimentAnalyzer(method=self.sentiment_method)
            _, sentiment_daily = sa.analyze_posts(reddit_df)
        except Exception as e:
            warnings.warn(f'Sentiment analysis failed: {str(e)}; using zero series')
            sentiment_daily = pd.DataFrame(0, index=feature_df.index, columns=['sentiment'])

        # 4. Forecast signals and apply strategy
        forecast_agent = ForecastAgent(
            df_raw=price_df,
            model_path=self.model_path,
            model_type=self.model_type
        )
        class ForecastWrapper:
            def __init__(self, agent): self.agent = agent
            def predict(self, feat, sent=None):
                return self.agent.generate_signals(n_steps=len(feat))

        strategy = StrategyAgent(
            forecast_agent=ForecastWrapper(forecast_agent),
            min_holding_period=self.min_holding,
            max_trades_per_week=self.max_trades
        )
        orders_df = strategy.generate_orders(feature_df, sentiment_daily, price_df)

        # 5. Backtest portfolio and compute metrics
        pm = PortfolioManagerAgent(initial_capital=self.initial_capital)
        portfolio_df = pm.backtest(orders_df)
        metrics = pm.compute_performance(portfolio_df)

        # 6. Generate report
        ReportGenerator(portfolio_df, metrics).create_report()

        return {
            'orders': orders_df,
            'portfolio': portfolio_df,
            'metrics': metrics
        }

if __name__ == '__main__':
    # First, create the dummy model if it doesn't exist
    model_path = 'agent/dummy_model.pkl'
    model_dir = os.path.dirname(model_path)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Create the model file if it doesn't exist
    if not os.path.exists(model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(RandomWalkModel(), f)
        print(f"Created dummy model at {model_path}")
    
    # Now run the trading system
    ts = TradingSystem(
        start_date='2023-01-01',  # Reduced timeframe to avoid rate limiting
        end_date='2023-12-31',
        tickers=['AAPL'],
        model_path=model_path
    )
    
    try:
        res = ts.run()
        print('Metrics:', res['metrics'])
    except Exception as e:
        print(f"Error running trading system: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()