import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import pickle

# Import optimized modules
from data_processor import DataProcessor
from sentiment_analyzer import SentimentAnalyzer
from analysis_strategy import AnalysisStrategy
from portfolio_manager import PortfolioManager

class RandomWalkModel:
    """Dummy model that predicts random walk values when no real model is available."""
    def predict(self, X):
        import numpy as np
        return [x[0] + np.random.randn() * 0.1 for x in X]

    
class TradingSystem:
    """
    Master orchestrator for the entire trading system workflow:
    1. Data loading and preprocessing
    2. Stock selection based on crypto sentiment correlation
    3. Sentiment analysis of Reddit posts
    4. Technical analysis and signal generation
    5. Portfolio simulation and performance measurement
    6. Report generation and visualization
    """
    
    def __init__(self, config: Dict[str, any]):
        """
        Initialize trading system with configuration.
        
        Args:
            config: Configuration parameters dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Create necessary directories
        os.makedirs(config.get('output_dir', 'agent/output'), exist_ok=True)
        os.makedirs('agent/models', exist_ok=True)
    
    def select_stocks_by_sentiment(self) -> List[str]:
        """
        Select technology stocks from S&P 500 with highest correlation to crypto sentiment.
        
        Returns:
            List of selected ticker symbols
        """
        self.logger.info("Selecting S&P 500 tech stocks based on crypto sentiment correlation...")
        
        # Calculate selection period
        lookback_days = 180  # Use 6 months of data for correlation analysis
        selection_start = pd.Timestamp(self.config['start_date']) - pd.Timedelta(days=lookback_days)
        selection_start = selection_start.strftime('%Y-%m-%d')
        
        # Initialize data processor
        data_processor = DataProcessor(
            start_date=selection_start,
            end_date=self.config['end_date']
        )
        
        # Get S&P 500 technology stocks
        sp500_tech_stocks = data_processor.get_sp500_tech_stocks()
        
        if not sp500_tech_stocks:
            self.logger.warning("No S&P 500 tech stocks found, using default tickers")
            return ['AAPL', 'MSFT', 'GOOG', 'NVDA', 'AMD']
        

        # Create a combined list, prioritizing S&P 500 tech stocks
        tech_tickers = list(sp500_tech_stocks)
        
        
        # Limit to 30 stocks for initial analysis to avoid API rate limits
        if len(tech_tickers) > 30:
            self.logger.info(f"Limiting initial analysis to 30 tech stocks, prioritizing S&P 500 members")
            # Ensure we preserve the S&P 500 stocks at the beginning of the list
            tech_tickers = tech_tickers[:30]
        
        # Load Reddit data
        reddit_subreddit = self.config.get('reddit_subreddit', 'CryptoCurrency')
        reddit_df = data_processor.load_reddit_posts(reddit_subreddit, limit=500)
        
        # Analyze sentiment
        sentiment_method = self.config.get('sentiment_method', 'transformer')
        transformer_model = 'ProsusAI/finbert' if sentiment_method == 'transformer' else None
        sentiment_analyzer = SentimentAnalyzer(
            method=sentiment_method,
            transformer_model=transformer_model
        )
        _, daily_sentiment = sentiment_analyzer.analyze_posts(reddit_df)
        
        # Calculate correlation for each stock
        correlations = {}
        
        # Process stocks in batches to avoid rate limits
        batch_size = 5
        for i in range(0, len(tech_tickers), batch_size):
            batch_tickers = tech_tickers[i:i+batch_size]
            self.logger.info(f"Processing batch: {', '.join(batch_tickers)}")
            
            try:
                # Load price data
                price_df = data_processor.load_stock_prices(batch_tickers)
                
                # Calculate correlation for each ticker
                for ticker in batch_tickers:
                    try:
                        # Extract close price for ticker
                        if isinstance(price_df.columns, pd.MultiIndex):
                            if (ticker, 'Close') in price_df.columns:
                                close_series = price_df[(ticker, 'Close')]
                            else:
                                # Try lowercase 'close'
                                potential_columns = [col for col in price_df.columns if col[0] == ticker]
                                if potential_columns:
                                    for col in potential_columns:
                                        if col[1].lower() == 'close':
                                            close_series = price_df[col]
                                            break
                                    else:
                                        raise KeyError(f"No Close column found for {ticker}")
                                else:
                                    raise KeyError(f"No columns found for {ticker}")
                        else:
                            # Single ticker case
                            close_series = price_df['Close']
                        
                        # Convert to returns
                        returns = close_series.pct_change().dropna()
                        
                        # Ensure date formats match for joining
                        sentiment_dates = pd.to_datetime(daily_sentiment['date']) if 'date' in daily_sentiment.columns else daily_sentiment.index
                        sentiment_values = daily_sentiment['avg_compound'].values
                        sentiment_series = pd.Series(
                            sentiment_values, 
                            index=sentiment_dates
                        )
                        
                        # Join with sentiment and calculate correlation
                        aligned_dates = returns.index.intersection(sentiment_series.index)
                        if len(aligned_dates) > 5:  # Need sufficient data points
                            stock_returns = returns.loc[aligned_dates]
                            sent_values = sentiment_series.loc[aligned_dates]
                            corr = stock_returns.corr(sent_values)
                            if not np.isnan(corr):
                                correlations[ticker] = abs(corr)  # Use absolute correlation
                    except Exception as e:
                        self.logger.warning(f"Error processing {ticker}: {str(e)}")
            
            except Exception as e:
                self.logger.warning(f"Error fetching batch data: {str(e)}")
        
        # Sort stocks by correlation and select top N
        sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        self.logger.info(f"Sentiment correlations: {sorted_correlations[:10]}")
        
        num_stocks = int(self.config.get('num_stocks', 5))
        selected_tickers = [ticker for ticker, corr in sorted_correlations[:num_stocks]]
        
        # If no correlations were found, use default S&P 500 tech stocks
        if not selected_tickers:
            self.logger.warning("No correlation-based tickers found, using default S&P 500 tech stocks")
            selected_tickers = ['AAPL', 'MSFT', 'GOOG', 'NVDA', 'AMD']
        
        self.logger.info(f"Selected {len(selected_tickers)} S&P 500 tech stocks with highest correlation: {selected_tickers}")
        
        return selected_tickers
    
    def run(self):
        """
        Execute the full trading pipeline.
        
        Returns:
            Dictionary with results
        """
        # Select stocks if not already specified
        tickers = self.config.get('tickers')
        if not tickers:
            tickers = self.select_stocks_by_sentiment()
            self.config['tickers'] = tickers

        # Ensure we have at least some default tickers
        if not tickers:
            self.logger.warning("No tickers selected or specified, using default S&P 500 tech stocks")
            tickers = ['AAPL', 'MSFT', 'GOOG', 'NVDA', 'AMD']  # DEFAULT S&P 500 tech stocks
            self.config['tickers'] = tickers

        # 2. Initialize core components
        data_processor = DataProcessor(
            start_date=self.config['start_date'],
            end_date=self.config['end_date'])
        
        # 2. Initialize core components
        data_processor = DataProcessor(
            start_date=self.config['start_date'],
            end_date=self.config['end_date'])
        
        sentiment_analyzer = SentimentAnalyzer(
            method=self.config.get('sentiment_method', 'transformer'),
            transformer_model='ProsusAI/finbert' if self.config.get('sentiment_method') == 'transformer' else None
        )

        self.config['model_type'] = 'gru'
        gru_model_path = self._find_model_path()

        analysis_strategy = AnalysisStrategy(
            sentiment_threshold=float(self.config.get('sentiment_threshold', 0.3)),
            sentiment_change_threshold=float(self.config.get('sentiment_change_threshold', 0.1)),
            min_holding_period=int(self.config.get('min_holding', 5)),
            max_trades_per_week=int(self.config.get('max_trades', 3)),
            model_path=gru_model_path,
            model_type='gru',
            sentiment_weight=float(self.config.get('sentiment_weight', 0.6)),
            technical_weight=float(self.config.get('technical_weight', 0.4))
        )
        
        portfolio_manager = PortfolioManager(
            initial_capital=float(self.config.get('initial_capital', 100000.0)),
            output_dir=self.config.get('output_dir', 'agent/output')
        )
        
        # 3. Load price data
        self.logger.info(f"Loading price data for {tickers}")
        price_df = data_processor.load_stock_prices(tickers)

        if not price_df.empty:
            try:
                price_df = self._fix_price_columns(price_df)
            except ValueError as e:
                self.logger.error(f"Error fixing price columns: {e}")
                # Print available columns to help diagnose
                self.logger.info(f"Available columns: {price_df.columns.tolist()}")
                raise
        
        # Normalize columns for single ticker or first ticker
        if isinstance(price_df.columns, pd.MultiIndex):
            if len(tickers) > 1:
                # Use the first ticker for trading
                ticker_to_use = tickers[0]
                price_df = price_df[ticker_to_use].copy()
            else:
                # Flatten the columns
                price_df.columns = [col[1] for col in price_df.columns]

        
        # 4. Load Reddit data and analyze sentiment
        self.logger.info("Loading and analyzing Reddit data")
        reddit_df = data_processor.load_reddit_posts(
            self.config.get('reddit_subreddit', 'CryptoCurrency')
        )
        
        _, sentiment_df = sentiment_analyzer.analyze_posts(reddit_df)
        
        # Ensure sentiment has datetime index
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        sentiment_df.set_index('date', inplace=True)
        
        # 5. Calculate technical indicators and generate signals
        self.logger.info("Calculating technical indicators")
        indicators_df = analysis_strategy.calculate_technical_indicators(price_df)
        
        self.logger.info("Generating technical signals")
        tech_signals = analysis_strategy.generate_technical_signals(indicators_df)
        
        # 6. Generate strategy signals
        self.logger.info("Generating combined strategy signals")
        signals_df = analysis_strategy.generate_strategy_signals(
            price_df=price_df,
            sentiment_df=sentiment_df,
            technical_signals_df=tech_signals
        )
        
        # 7. Run portfolio simulation
        self.logger.info("Running portfolio simulation")
        portfolio_df = portfolio_manager.backtest(signals_df)
        
        # 8. Calculate performance metrics
        self.logger.info("Calculating performance metrics")
        metrics = portfolio_manager.compute_performance(portfolio_df)
        
        # 9. Generate report
        self.logger.info("Generating performance report")
        ticker_str = tickers[0] if len(tickers) == 1 else f"{len(tickers)} Tech Stocks"
        report_path = portfolio_manager.generate_report(
            portfolio_df=portfolio_df,
            metrics=metrics,
            ticker=ticker_str
        )
        
        # 10. Create and launch dashboard
        self.logger.info("Creating interactive dashboard")
        results = {
            'portfolio': portfolio_df,
            'orders': signals_df,
            'metrics': metrics,
            'sentiment': sentiment_df,
            'technical': tech_signals
        }
        
        dashboard = portfolio_manager.create_dashboard(results, self.config)
        dashboard.launch(share=False)
        
        return results
    
    def _find_model_path(self) -> str:
        """
        Find or create the GRU model path.
        Returns only the GRU model path, with no fallback to other model types.
        """
        self.config['model_type'] = 'gru'
        model_path = os.path.join('agent', 'models', 'gru_model.pt')
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Just return the path - will be created in AnalysisStrategy if not found
        return model_path
    def _fix_price_columns(self, price_df):
        """
        Fix column names in the price DataFrame to ensure 'Close' column exists.
        
        Args:
            price_df: DataFrame with price data
            
        Returns:
            DataFrame with corrected column names
        """
        # First check if Close already exists
        if 'Close' in price_df.columns:
            return price_df
        
        # Make a copy to avoid modifying the original
        df = price_df.copy()
        
        # Try to find alternate column names for close price
        close_candidates = [
            col for col in df.columns 
            if any(term in col.lower() for term in ['close', 'price', 'last', 'adj'])
        ]
        
        if close_candidates:
            # Use the first match
            df['Close'] = df[close_candidates[0]]
            self.logger.info(f"Using '{close_candidates[0]}' as 'Close' column")
            return df
        
        # If still no suitable column, check if there's any numerical column we can use
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            df['Close'] = df[numeric_cols[0]]
            self.logger.info(f"Using numeric column '{numeric_cols[0]}' as 'Close'")
            return df
        
        # If we have a multi-index DataFrame, check columns at level 1
        if isinstance(df.columns, pd.MultiIndex):
            level1_close = [(l0, l1) for l0, l1 in df.columns if 'close' in l1.lower()]
            if level1_close:
                # Extract this column and make it a non-multi-index DataFrame
                df = pd.DataFrame({'Close': df[level1_close[0]]})
                return df
        
        # If we get here, we couldn't find any usable price column
        self.logger.error("Could not find a suitable column to use as 'Close'")
        raise ValueError("No suitable price column found to use as 'Close'")



if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with minimal configuration
    config = {
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'tickers': ['AAPL'],
        'model_type': 'sklearn',
        'sentiment_method': 'vader'
    }
    
    # Run trading system
    trading_system = TradingSystem(config)
    try:
        results = trading_system.run()
        print(f"Trading completed. Final equity: ${results['portfolio']['equity'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"Error running trading system: {str(e)}")
        import traceback
        traceback.print_exc()