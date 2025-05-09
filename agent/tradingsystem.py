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
                # ensure we have a flat 'Close' as expected by the correlation logic
                try:
                    price_df = self._fix_price_columns(price_df)
                except ValueError as e:
                    self.logger.error(f"Cannot locate Close column in batch {batch_tickers}: {e}")
                    continue
                
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

        # Initialize core components
        data_processor = DataProcessor(
            start_date=self.config['start_date'],
            end_date=self.config['end_date'])
        
        sentiment_analyzer = SentimentAnalyzer(
            method=self.config.get('sentiment_method', 'transformer'),
            transformer_model='ProsusAI/finbert' if self.config.get('sentiment_method') == 'transformer' else None
        )

        # Set model type to gru
        self.config['model_type'] = 'gru'
        model_dir = os.path.join('agent', 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Load price data
        self.logger.info(f"Loading price data for {tickers}")
        price_df = data_processor.load_stock_prices(tickers)
        price_df = price_df.loc[self.config['start_date']:self.config['end_date']]
        
        # Log available columns for debugging
        self.logger.info(f"Available columns: {price_df.columns.tolist()}")
        
        # Handle the flattened structure with ticker_name_Close format
        if 'Close' not in price_df.columns:
            # Check for ticker-specific Close columns
            ticker_close_cols = [col for col in price_df.columns if col.endswith('_Close')]
            
            if ticker_close_cols:
                # Use the first ticker
                ticker_to_use = ticker_close_cols[0].split('_')[0]
                self.logger.info(f"Using {ticker_to_use} for trading")
                
                # Extract columns for this ticker
                ticker_cols = [col for col in price_df.columns if col.startswith(f"{ticker_to_use}_")]
                ticker_df = price_df[ticker_cols].copy()
                
                # Rename columns to remove ticker prefix
                rename_map = {col: col.split('_', 1)[1] for col in ticker_cols}
                ticker_df = ticker_df.rename(columns=rename_map)
                
                # Now we should have a 'Close' column
                if 'Close' in ticker_df.columns:
                    price_df = ticker_df
                    self.logger.info(f"Processed {ticker_to_use} data with columns: {price_df.columns.tolist()}")
                else:
                    self.logger.error(f"No 'Close' column found after processing {ticker_to_use} data")
                    raise ValueError(f"Failed to process {ticker_to_use} data correctly")
            else:
                self.logger.error("No '_Close' columns found in price data")
                raise ValueError("No suitable price columns found")
        
        # Initialize strategy
        analysis_strategy = AnalysisStrategy(
            sentiment_threshold=float(self.config.get('sentiment_threshold', 0.3)),
            sentiment_change_threshold=float(self.config.get('sentiment_change_threshold', 0.1)),
            min_holding_period=int(self.config.get('min_holding', 5)),
            max_trades_per_week=int(self.config.get('max_trades', 3)),
            model_path=None,  # Will train a new model
            model_type='gru',
            sentiment_weight=float(self.config.get('sentiment_weight', 0.6)),
            technical_weight=float(self.config.get('technical_weight', 0.4))
        )
        
        portfolio_manager = PortfolioManager(
            initial_capital=float(self.config.get('initial_capital', 100000.0)),
            output_dir=self.config.get('output_dir', 'agent/output')
        )
        
        # Load Reddit data and analyze sentiment
        self.logger.info("Loading and analyzing Reddit data")
        reddit_df = data_processor.load_reddit_posts(
            self.config.get('reddit_subreddit', 'CryptoCurrency')
        )
        
        _, sentiment_df = sentiment_analyzer.analyze_posts(reddit_df)
        
        # Ensure sentiment has datetime index
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        sentiment_df.set_index('date', inplace=True)
        
        # Calculate technical indicators and generate signals
        self.logger.info("Calculating technical indicators")
        indicators_df = analysis_strategy.calculate_technical_indicators(price_df)
        
        self.logger.info("Generating technical signals")
        tech_signals = analysis_strategy.generate_technical_signals(indicators_df)
        
        # Generate strategy signals
        self.logger.info("Generating combined strategy signals")
        signals_df = analysis_strategy.generate_strategy_signals(
            price_df=price_df,
            sentiment_df=sentiment_df,
            technical_signals_df=tech_signals
        )
        signals_df = signals_df.loc[self.config['start_date']:self.config['end_date']]
        
        # Generate baseline signals (without sentiment)
        self.logger.info("Generating baseline signals without sentiment")
        baseline_signals = analysis_strategy.generate_baseline_signals(
            price_df=price_df,
            technical_signals_df=tech_signals
        )
        baseline_signals = baseline_signals.loc[self.config['start_date']:self.config['end_date']]
        
        # Run backtest for both strategies
        self.logger.info("Running portfolio simulations")
        portfolio_df = portfolio_manager.backtest(signals_df)
        baseline_portfolio_df = portfolio_manager.backtest(baseline_signals)
        
        # Calculate metrics
        self.logger.info("Calculating performance metrics")
        metrics = portfolio_manager.compute_performance(portfolio_df)
        baseline_metrics = portfolio_manager.compute_performance(baseline_portfolio_df)
        
        # Analyze sentiment contribution
        self.logger.info("Analyzing sentiment contribution")
        sentiment_analysis = portfolio_manager.analyze_sentiment_contribution(
            base_portfolio=baseline_portfolio_df,
            full_portfolio=portfolio_df,
            sentiment_df=sentiment_df
        )
        
        # Generate reports
        self.logger.info("Generating reports")
        ticker_str = tickers[0] if len(tickers) == 1 else f"{len(tickers)} Tech Stocks"
        
        report_path = portfolio_manager.generate_report(
            portfolio_df=portfolio_df,
            metrics=metrics,
            ticker=ticker_str
        )
        
        sentiment_report_path = portfolio_manager.generate_sentiment_report(
            sentiment_analysis=sentiment_analysis
        )
        
        self.logger.info(f"Reports generated at:\n- {report_path}\n- {sentiment_report_path}")
        
        # Create dashboard
        self.logger.info("Creating interactive dashboard")
        results = {
            'portfolio': portfolio_df,
            'base_portfolio': baseline_portfolio_df,
            'sentiment_portfolio': portfolio_df,
            'orders': signals_df,
            'baseline_orders': baseline_signals,
            'metrics': metrics,
            'baseline_metrics': baseline_metrics,
            'sentiment': sentiment_df,
            'technical': tech_signals,
            'sentiment_analysis': sentiment_analysis
        }
        
        dashboard = portfolio_manager.create_dashboard(results, self.config)
        dashboard.launch(share=True)
        
        return results


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
