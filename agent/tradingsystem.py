import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

# Import optimized modules
from data_processor import DataProcessor
from sentiment_analyzer import SentimentAnalyzer
from analysis_strategy import AnalysisStrategy
from portfolio_manager import PortfolioManager

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
        os.makedirs(config.get('output_dir', 'output'), exist_ok=True)
        os.makedirs('agent/models', exist_ok=True)
    
    def select_stocks_by_sentiment(self) -> List[str]:
        """
        Select technology stocks with highest correlation to crypto sentiment.
        
        Returns:
            List of selected ticker symbols
        """
        self.logger.info("Selecting stocks based on crypto sentiment correlation...")
        
        # Calculate selection period
        lookback_days = 180  # Use 6 months of data for correlation analysis
        selection_start = pd.Timestamp(self.config['start_date']) - pd.Timedelta(days=lookback_days)
        selection_start = selection_start.strftime('%Y-%m-%d')
        
        # Initialize data processor
        data_processor = DataProcessor(
            start_date=selection_start,
            end_date=self.config['end_date']
        )
        
        # Get technology sector stocks
        tech_tickers = data_processor.get_tickers_by_sector("Technology")
        if not tech_tickers:
            self.logger.warning("No technology tickers found, using default tickers")
            return ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD']
        
        # Limit to 30 stocks for initial analysis to avoid API rate limits
        if len(tech_tickers) > 30:
            self.logger.info(f"Limiting initial analysis to 30 tech stocks")
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
                            close_series = price_df[(ticker, 'Close')]
                        else:
                            # Single ticker case
                            close_series = price_df['Close']
                        
                        # Convert to returns
                        returns = close_series.pct_change().dropna()
                        
                        # Ensure date formats match for joining
                        sentiment_dates = pd.to_datetime(daily_sentiment['date'])
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
        self.logger.info(f"Selected {len(selected_tickers)} stocks with highest correlation: {selected_tickers}")
        
        return selected_tickers
    
    def run(self):
        """
        Execute the full trading pipeline.
        
        Returns:
            Dictionary with results
        """
        # 1. Select stocks if not already specified
        tickers = self.config.get('tickers')
        if not tickers:
            tickers = self.select_stocks_by_sentiment()
            self.config['tickers'] = tickers
        
        self.logger.info(f"Trading analysis will use tickers: {tickers}")
        
        # 2. Initialize core components
        data_processor = DataProcessor(
            start_date=self.config['start_date'],
            end_date=self.config['end_date']
        )
        
        sentiment_analyzer = SentimentAnalyzer(
            method=self.config.get('sentiment_method', 'transformer'),
            transformer_model='ProsusAI/finbert' if self.config.get('sentiment_method') == 'transformer' else None
        )
        
        analysis_strategy = AnalysisStrategy(
            sentiment_threshold=float(self.config.get('sentiment_threshold', 0.3)),
            sentiment_change_threshold=float(self.config.get('sentiment_change_threshold', 0.1)),
            min_holding_period=int(self.config.get('min_holding', 5)),
            max_trades_per_week=int(self.config.get('max_trades', 3)),
            model_path=self._find_model_path(),
            model_type=self.config.get('model_type', 'gru'),
            sentiment_weight=float(self.config.get('sentiment_weight', 0.6)),
            technical_weight=float(self.config.get('technical_weight', 0.4))
        )
        
        portfolio_manager = PortfolioManager(
            initial_capital=float(self.config.get('initial_capital', 100000.0)),
            output_dir=self.config.get('output_dir', 'output')
        )
        
        # 3. Load price data
        self.logger.info(f"Loading price data for {tickers}")
        price_df = data_processor.load_stock_prices(tickers)
        
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
    
    def _find_model_path(self) -> Optional[str]:
        """Find appropriate model path based on configuration."""
        if self.config.get('model_type') == 'gru':
            model_path = os.path.join('agent', 'models', 'gru_model.pt')
            if not os.path.exists(model_path):
                self.logger.warning("GRU model not found, falling back to sklearn model")
                self.config['model_type'] = 'sklearn'
                model_path = os.path.join('agent', 'models', 'dummy_model.pkl')
        else:
            model_path = os.path.join('agent', 'models', 'dummy_model.pkl')
        
        # Create dummy sklearn model if needed
        if self.config['model_type'] == 'sklearn' and not os.path.exists(model_path):
            import pickle
            
            class RandomWalkModel:
                def predict(self, X):
                    return [x[0] + np.random.randn() * 0.1 for x in X]
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(RandomWalkModel(), f)
            self.logger.info(f"Created dummy model at {model_path}")
        
        return model_path


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