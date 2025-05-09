import pandas as pd
import numpy as np
import os
import logging
import torch
from datetime import timedelta
from typing import Dict, List, Optional, Union, Tuple
from gru_model import GRUModelTrainer

class AnalysisStrategy:
    """
    Comprehensive analysis and strategy module that:
    1. Combines technical analysis, sentiment data, and ML predictions
    2. Generates trading signals with configurable weights
    3. Evaluates the contribution of sentiment analysis to strategy performance
    4. Supports GRU model for price prediction
    """
    
    def __init__(
        self,
        sentiment_threshold: float = 0.3,
        sentiment_change_threshold: float = 0.1,
        min_holding_period: int = 5,
        max_trades_per_week: int = 3,
        model_path: Optional[str] = None,
        model_type: str = 'gru',
        sentiment_weight: float = 0.6,
        technical_weight: float = 0.4,
        seq_len: int = 60
    ):
        """
        Initialize the analysis and strategy system.
        
        Args:
            sentiment_threshold: Minimum sentiment score to consider (absolute value)
            sentiment_change_threshold: Minimum change in sentiment to generate a signal
            min_holding_period: Days to hold a position before new trade
            max_trades_per_week: Max trades in any rolling 7-day window
            model_path: Path to the prediction model file
            model_type: Type of model ('gru')
            sentiment_weight: Weight for sentiment signals (0-1)
            technical_weight: Weight for technical signals (0-1)
            seq_len: Sequence length for GRU model input
        """
        self.logger = logging.getLogger(__name__)
        self.sentiment_threshold = sentiment_threshold
        self.sentiment_change_threshold = sentiment_change_threshold
        self.min_holding = pd.Timedelta(days=min_holding_period)
        self.max_trades = max_trades_per_week
        self.sentiment_weight = sentiment_weight
        self.technical_weight = technical_weight
        self.seq_len = seq_len
        
        # Model related attributes
        self.model_path = model_path
        self.model_type = 'gru'  # Always use GRU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize GRU trainer
        self.gru_trainer = GRUModelTrainer(seq_len=seq_len, device=self.device)
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.gru_trainer.load_model(model_path)
    
    def calculate_technical_indicators(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators from price data.
        
        Args:
            price_df: DataFrame with price data (needs 'Close' column)
            
        Returns:
            DataFrame with technical indicators
        """
        if 'Close' not in price_df.columns:
            raise ValueError("price_df must have a 'Close' column")
            
        df = price_df.copy()
        
        # Simple Moving Averages
        for window in [5, 20, 60]:
            df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
        
        # Exponential Moving Averages
        for window in [5, 20, 60]:
            df[f'ema_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        roll_up = up.ewm(com=13, adjust=False).mean()  # 14-day RSI
        roll_down = down.ewm(com=13, adjust=False).mean()
        rs = roll_up / roll_down
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = rolling_mean + 2 * rolling_std
        df['bb_lower'] = rolling_mean - 2 * rolling_std
        
        return df
    
    def generate_technical_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical trading signals.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            DataFrame of technical signals
        """
        signals = pd.DataFrame(index=df.index)
        
        # RSI signals
        signals['rsi_sig'] = 0
        signals.loc[df['rsi_14'] < 30, 'rsi_sig'] = 1
        signals.loc[df['rsi_14'] > 70, 'rsi_sig'] = -1
        
        # MACD crossover signals
        signals['macd_sig'] = 0
        macd = df['macd']
        macd_sig = df['macd_signal']
        crossover_up = (macd > macd_sig) & (macd.shift(1) <= macd_sig.shift(1))
        crossover_down = (macd < macd_sig) & (macd.shift(1) >= macd_sig.shift(1))
        signals.loc[crossover_up, 'macd_sig'] = 1
        signals.loc[crossover_down, 'macd_sig'] = -1
        
        # Bollinger Bands signals
        signals['bb_sig'] = 0
        signals.loc[df['Close'] < df['bb_lower'], 'bb_sig'] = 1
        signals.loc[df['Close'] > df['bb_upper'], 'bb_sig'] = -1
        
        # SMA crossover signals
        signals['sma_sig'] = 0
        sma20 = df['sma_20']
        sma_up = (df['Close'] > sma20) & (df['Close'].shift(1) <= sma20.shift(1))
        sma_down = (df['Close'] < sma20) & (df['Close'].shift(1) >= sma20.shift(1))
        signals.loc[sma_up, 'sma_sig'] = 1
        signals.loc[sma_down, 'sma_sig'] = -1
        
        # Combined signal
        signals['combined_sig'] = signals.sum(axis=1)
        # Normalize to -1, 0, 1
        signals['combined_sig'] = signals['combined_sig'].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        
        return signals
    
    def _predict_with_gru(self, price_series, sentiment_series=None, technical_indicators=None):
        """
        Generate prediction using GRU model with multiple features.
        
        Args:
            price_series: Series of price data
            sentiment_series: Series of sentiment data (optional)
            technical_indicators: DataFrame of technical indicators (optional)
            
        Returns:
            Signal value: 1 (buy), -1 (sell), or 0 (hold)
        """
        # Check if we need to train a model
        if self.gru_trainer.model is None and len(price_series) > self.seq_len * 2:
            self.logger.info("No GRU model loaded, training a new one...")
            
            # Prepare data for training
            price_df = pd.DataFrame({'Close': price_series})
            
            sentiment_df = None
            if sentiment_series is not None and not sentiment_series.empty:
                sentiment_df = pd.DataFrame({'avg_compound': sentiment_series})
                sentiment_df.index = price_series.index
            
            # Train the model
            self.gru_trainer.train(
                price_df=price_df,
                sentiment_df=sentiment_df,
                tech_indicators=technical_indicators,
                epochs=50,
                batch_size=32
            )
            
            # Save the model if path specified
            if self.model_path:
                self.gru_trainer.save_model(self.model_path)
        
        # If still no model, return neutral signal
        if self.gru_trainer.model is None:
            return 0
        
        # Prepare features for prediction
        features = []
        
        # Add price data
        price_vals = price_series.values[-self.seq_len:]
        if len(price_vals) < self.seq_len:
            price_vals = np.pad(price_vals, (self.seq_len - len(price_vals), 0), 'constant')
        features.append(price_vals)
        
        # Add sentiment if available
        if sentiment_series is not None:
            sent_vals = sentiment_series.values[-self.seq_len:]
            if len(sent_vals) < self.seq_len:
                sent_vals = np.pad(sent_vals, (self.seq_len - len(sent_vals), 0), 'constant')
            features.append(sent_vals)
        else:
            # Add zeros for sentiment
            features.append(np.zeros(self.seq_len))
        
        # Add technical indicators if available
        tech_features = ['rsi_14', 'macd', 'bb_upper']
        if technical_indicators is not None:
            for feature in tech_features:
                if feature in technical_indicators.columns:
                    tech_vals = technical_indicators[feature].values[-self.seq_len:]
                    if len(tech_vals) < self.seq_len:
                        tech_vals = np.pad(tech_vals, (self.seq_len - len(tech_vals), 0), 'constant')
                    features.append(tech_vals)
                else:
                    features.append(np.zeros(self.seq_len))
        else:
            # Add zeros for missing indicators
            for _ in range(len(tech_features)):
                features.append(np.zeros(self.seq_len))
        
        # Stack features
        feature_array = np.column_stack(features)
        
        # Get current price
        current_price = float(price_series.iloc[-1])
        
        # Use GRU trainer to get trading signal
        return self.gru_trainer.predict_signal(current_price, feature_array, threshold=0.01)
    
    def generate_strategy_signals(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        technical_signals_df: Optional[pd.DataFrame] = None,
        macro_df: Optional[pd.DataFrame] = None,
        defi_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate combined trading signals from multiple sources.
        
        Args:
            price_df: DataFrame with price data
            sentiment_df: DataFrame with sentiment data
            technical_signals_df: Optional DataFrame with technical signals
            macro_df: Optional DataFrame with macroeconomic data
            defi_df: Optional DataFrame with DeFi market data
            
        Returns:
            DataFrame with signals and trading orders
        """
        # 1. Ensure we have required columns
        if 'Close' not in price_df.columns:
            raise ValueError("price_df must have a 'Close' column")
        if 'avg_compound' not in sentiment_df.columns:
            raise ValueError("sentiment_df must have an 'avg_compound' column")
        
        # 2. Prepare dataframe
        df = pd.DataFrame(index=price_df.index)
        df['price'] = price_df['Close']
        df['sentiment'] = np.nan
        
        # 3. Calculate technical indicators if not provided
        technical_indicators = None
        if 'rsi_14' not in price_df.columns:
            # Calculate technical indicators if not already in price_df
            technical_indicators = self.calculate_technical_indicators(price_df)
        else:
            technical_indicators = price_df
        
        # 4. Align sentiment data with price dates
        sentiment_dates = sentiment_df.index
        for date in df.index:
            # Find the closest sentiment date before or on the current date
            closest_date = sentiment_dates[sentiment_dates <= date]
            if not closest_date.empty:
                closest_date = closest_date[-1]
                df.loc[date, 'sentiment'] = sentiment_df.loc[closest_date, 'avg_compound']
        
        # 5. Fill missing sentiment values with previous value
        df['sentiment'] = df['sentiment'].fillna(method='ffill')
        
        # 6. Calculate sentiment change
        df['sentiment_change'] = df['sentiment'].diff()
        
        # 7. Generate model-based prediction signals with multiple features
        df['model_signal'] = 0
        sentiment_series = df['sentiment']
        
        # For each day, predict using appropriate window of data
        for i in range(min(self.seq_len, 5), len(df)):
            # Extract feature windows
            price_window = df['price'].iloc[i-min(self.seq_len, 5):i]
            sentiment_window = sentiment_series.iloc[i-min(self.seq_len, 5):i]
            tech_window = technical_indicators.iloc[i-min(self.seq_len, 5):i] if technical_indicators is not None else None
            
            # Predict with GRU model
            df.iloc[i, df.columns.get_loc('model_signal')] = self._predict_with_gru(
                price_window, sentiment_window, tech_window)
        
        # 8. Generate sentiment-based signals
        df['sentiment_signal'] = 0
        
        # Buy signal: Sentiment above threshold and significant positive change
        buy_condition = (
            (df['sentiment'] > self.sentiment_threshold) & 
            (df['sentiment_change'] > self.sentiment_change_threshold)
        )
        df.loc[buy_condition, 'sentiment_signal'] = 1
        
        # Sell signal: Sentiment below negative threshold or significant negative change
        sell_condition = (
            (df['sentiment'] < -self.sentiment_threshold) | 
            (df['sentiment_change'] < -self.sentiment_change_threshold)
        )
        df.loc[sell_condition, 'sentiment_signal'] = -1
        
        # 9. Incorporate technical signals if provided
        df['technical_signal'] = 0
        if technical_signals_df is not None:
            # Align technical signals with our index
            aligned_tech = technical_signals_df.reindex(df.index)
            
            # Sum up all technical signals if multiple columns
            if isinstance(aligned_tech, pd.DataFrame) and len(aligned_tech.columns) > 1:
                tech_cols = [col for col in aligned_tech.columns if col.endswith('_sig')]
                if tech_cols:
                    df['technical_signal'] = aligned_tech[tech_cols].sum(axis=1)
                    # Normalize to -1, 0, 1
                    df['technical_signal'] = df['technical_signal'].apply(
                        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
                    )
            elif 'combined_sig' in aligned_tech.columns:
                df['technical_signal'] = aligned_tech['combined_sig']
        
        # 10. Calculate weighted combination of signals
        model_weight = max(0, 1 - self.sentiment_weight - self.technical_weight)
        df['raw_signal'] = (
            self.sentiment_weight * df['sentiment_signal'] + 
            self.technical_weight * df['technical_signal'] +
            model_weight * df['model_signal']
        )
        
        # 11. Discretize to -1, 0, 1
        df['raw_signal'] = df['raw_signal'].apply(
            lambda x: 1 if x > 0.3 else (-1 if x < -0.3 else 0)
        )
        
        # 12. Apply trading rules
        df['order'] = 0
        last_trade = None
        trades = []
        
        for ts, row in df.iterrows():
            sig = row['raw_signal']
            if sig == 0:
                continue
                
            # Minimum holding period
            if last_trade and ts - last_trade < self.min_holding:
                continue
                
            # Maximum trades in past 7 days
            window_start = ts - timedelta(days=7)
            recent = [t for t in trades if t >= window_start]
            if len(recent) >= self.max_trades:
                continue
                
            df.loc[ts, 'order'] = sig
            last_trade = ts
            trades.append(ts)
        
        return df[['raw_signal', 'sentiment_signal', 'technical_signal', 'model_signal', 
                  'order', 'price', 'sentiment', 'sentiment_change']]
    
    def generate_baseline_signals(
        self,
        price_df: pd.DataFrame,
        technical_signals_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate baseline trading signals without using sentiment data.
        Used for A/B testing to measure sentiment contribution.
        
        Args:
            price_df: DataFrame with price data
            technical_signals_df: Optional DataFrame with technical signals
            
        Returns:
            DataFrame with baseline signals and trading orders
        """
        # Create a dummy sentiment DataFrame filled with zeros
        dates = price_df.index
        dummy_sentiment = pd.DataFrame({
            'avg_compound': np.zeros(len(dates))
        }, index=dates)
        
        # Save original sentiment weight
        original_weight = self.sentiment_weight
        
        # Set sentiment weight to 0 for baseline
        self.sentiment_weight = 0.0
        
        # Generate signals without sentiment influence
        baseline_signals = self.generate_strategy_signals(
            price_df=price_df,
            sentiment_df=dummy_sentiment,
            technical_signals_df=technical_signals_df
        )
        
        # Restore original sentiment weight
        self.sentiment_weight = original_weight
        
        return baseline_signals
    
    def evaluate_sentiment_contribution(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        technical_signals_df: Optional[pd.DataFrame] = None,
        portfolio_manager=None
    ) -> Dict:
        """
        Evaluate the contribution of sentiment analysis to strategy performance.
        
        Args:
            price_df: DataFrame with price data
            sentiment_df: DataFrame with sentiment data
            technical_signals_df: Optional DataFrame with technical signals
            portfolio_manager: PortfolioManager instance for backtesting
            
        Returns:
            Dictionary with sentiment contribution analysis
        """
        self.logger.info("Evaluating sentiment contribution to strategy performance...")
        
        # Generate signals with sentiment
        signals_with_sentiment = self.generate_strategy_signals(
            price_df=price_df,
            sentiment_df=sentiment_df,
            technical_signals_df=technical_signals_df
        )
        
        # Generate baseline signals without sentiment
        baseline_signals = self.generate_baseline_signals(
            price_df=price_df,
            technical_signals_df=technical_signals_df
        )
        
        # If portfolio manager is provided, run backtests
        if portfolio_manager:
            # Backtest with sentiment
            portfolio_with_sentiment = portfolio_manager.backtest(signals_with_sentiment)
            metrics_with_sentiment = portfolio_manager.compute_performance(portfolio_with_sentiment)
            
            # Backtest without sentiment
            portfolio_baseline = portfolio_manager.backtest(baseline_signals)
            metrics_baseline = portfolio_manager.compute_performance(portfolio_baseline)
            
            # Calculate differences in key metrics
            return_diff = metrics_with_sentiment['total_return'] - metrics_baseline['total_return']
            sharpe_diff = metrics_with_sentiment['sharpe_ratio'] - metrics_baseline['sharpe_ratio']
            drawdown_diff = metrics_baseline['max_drawdown'] - metrics_with_sentiment['max_drawdown']
            
            # Calculate percentage contribution
            if metrics_with_sentiment['total_return'] != 0:
                pct_contribution = return_diff / abs(metrics_with_sentiment['total_return']) * 100
            else:
                pct_contribution = 0
            
            # Identify trades that are uniquely due to sentiment
            sentiment_driven_trades = self._identify_sentiment_driven_trades(
                signals_with_sentiment, 
                baseline_signals
            )
            
            # Prepare results
            contribution_analysis = {
                'return_difference': return_diff,
                'sharpe_difference': sharpe_diff,
                'drawdown_difference': drawdown_diff,
                'percentage_contribution': pct_contribution,
                'sentiment_driven_trades': sentiment_driven_trades,
                'portfolio_with_sentiment': portfolio_with_sentiment,
                'portfolio_baseline': portfolio_baseline,
                'metrics_with_sentiment': metrics_with_sentiment,
                'metrics_baseline': metrics_baseline
            }
            
            self.logger.info(f"Sentiment contribution: {return_diff:.2%} to total return, {pct_contribution:.2f}% of strategy performance")
            
            return contribution_analysis
        else:
            # Without portfolio manager, just return the signals
            return {
                'signals_with_sentiment': signals_with_sentiment,
                'baseline_signals': baseline_signals
            }
    
    def _identify_sentiment_driven_trades(
        self,
        signals_with_sentiment: pd.DataFrame,
        baseline_signals: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Identify trades that are uniquely driven by sentiment signals.
        
        Args:
            signals_with_sentiment: Signals generated with sentiment
            baseline_signals: Signals generated without sentiment
            
        Returns:
            DataFrame with sentiment-driven trades
        """
        # Find dates where orders differ
        diff_mask = signals_with_sentiment['order'] != baseline_signals['order']
        sentiment_trades = signals_with_sentiment[diff_mask].copy()
        
        # Add baseline order for comparison
        sentiment_trades['baseline_order'] = baseline_signals.loc[diff_mask, 'order']
        
        # Filter to trades where sentiment order is non-zero
        sentiment_trades = sentiment_trades[sentiment_trades['order'] != 0]
        
        return sentiment_trades