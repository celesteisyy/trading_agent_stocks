import pandas as pd
import numpy as np
import os
import logging
import pickle
import torch
from datetime import timedelta
from typing import Dict, List, Optional, Union, Tuple

class AnalysisStrategy:
    """
    Comprehensive analysis and strategy module that:
    1. Combines technical analysis, sentiment data, and ML predictions
    2. Generates trading signals with configurable weights
    3. Enforces trading rules (holding periods, frequency limits)
    4. Supports multiple model types (GRU or sklearn)
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
            model_type: Type of model ('gru' or 'sklearn')
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
        self.model = None
        self.model_type = model_type.lower()
        self.device = None
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load prediction model from file.
        
        Args:
            model_path: Path to model file
        """
        try:
            if self.model_type == 'sklearn':
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.logger.info(f"Loaded sklearn model from {model_path}")
            elif self.model_type == 'gru':
                # Use GPU if available
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model = torch.load(model_path, map_location=self.device)
                self.model.to(self.device).eval()
                self.logger.info(f"Loaded GRU model from {model_path} (device: {self.device})")
            else:
                self.logger.error(f"Unsupported model_type: {self.model_type}")
                raise ValueError(f"Unsupported model_type: {self.model_type}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def predict_with_model(self, price_series: pd.Series) -> int:
        """
        Generate prediction signal using loaded model.
        
        Args:
            price_series: Series of price data
            
        Returns:
            Signal value: 1 (buy), -1 (sell), or 0 (hold)
        """
        if self.model is None:
            return 0  # Neutral if no model
            
        if self.model_type == 'gru':
            return self._predict_with_gru(price_series)
        else:
            return self._predict_with_sklearn(price_series)
    
    def _predict_with_gru(self, price_series: pd.Series) -> int:
        """
        Generate prediction using GRU model.
        
        Args:
            price_series: Series of price data
            
        Returns:
            Signal value: 1 (buy), -1 (sell), or 0 (hold)
        """
        # Prepare input sequence (last seq_len values)
        seq_vals = price_series.values[-self.seq_len:]
        if len(seq_vals) < self.seq_len:
            # Pad with zeros if not enough data
            seq_vals = np.pad(seq_vals, (self.seq_len - len(seq_vals), 0), 'constant')
        
        # Convert to tensor
        seq = (
            torch.tensor(seq_vals, dtype=torch.float32)
                .unsqueeze(0)      # batch dim
                .unsqueeze(-1)     # feature dim
                .to(self.device)
        )
        
        # Get prediction
        with torch.no_grad():
            pred = self.model(seq)
        
        # Convert to signal: 1 if predicted price > current price, -1 if lower, 0 if same
        current_price = float(price_series.iloc[-1])
        predicted_price = float(pred.squeeze(0).cpu().numpy()[0])
        
        if predicted_price > current_price * 1.01:  # 1% threshold
            return 1
        elif predicted_price < current_price * 0.99:  # -1% threshold
            return -1
        else:
            return 0
    
    def _predict_with_sklearn(self, price_series: pd.Series) -> int:
        """
        Generate prediction using sklearn model.
        
        Args:
            price_series: Series of price data
            
        Returns:
            Signal value: 1 (buy), -1 (sell), or 0 (hold)
        """
        # Get last price
        last_val = float(price_series.iloc[-1])
        
        # Predict next price
        try:
            pred = self.model.predict([[last_val]])[0]
            
            # Convert to signal
            if pred > last_val * 1.01:  # 1% threshold
                return 1
            elif pred < last_val * 0.99:  # -1% threshold
                return -1
            else:
                return 0
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return 0
    
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
        
        # ATR (Average True Range)
        if all(col in df.columns for col in ['High', 'Low']):
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_14'] = true_range.rolling(window=14).mean()
        
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
        
        return signals
    
    def generate_strategy_signals(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        technical_signals_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate combined trading signals from multiple sources.
        
        Args:
            price_df: DataFrame with price data
            sentiment_df: DataFrame with sentiment data
            technical_signals_df: Optional DataFrame with technical signals
            
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
        
        # 3. Align sentiment data with price dates
        sentiment_dates = sentiment_df.index
        for date in df.index:
            # Find the closest sentiment date before or on the current date
            closest_date = sentiment_dates[sentiment_dates <= date]
            if not closest_date.empty:
                closest_date = closest_date[-1]
                df.loc[date, 'sentiment'] = sentiment_df.loc[closest_date, 'avg_compound']
        
        # 4. Fill missing sentiment values with previous value
        df['sentiment'] = df['sentiment'].fillna(method='ffill')
        
        # 5. Calculate sentiment change
        df['sentiment_change'] = df['sentiment'].diff()
        
        # 6. Generate model-based prediction signals
        df['model_signal'] = 0
        if self.model is not None:
            # For each day, predict using appropriate window of data
            for i in range(min(self.seq_len, 5), len(df)):
                window = df['price'].iloc[i-min(self.seq_len, 5):i]
                df.iloc[i, df.columns.get_loc('model_signal')] = self.predict_with_model(window)
        
        # 7. Generate sentiment-based signals
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
        
        # 8. Incorporate technical signals if provided
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
        
        # 9. Calculate weighted combination of signals
        model_weight = max(0, 1 - self.sentiment_weight - self.technical_weight)
        df['raw_signal'] = (
            self.sentiment_weight * df['sentiment_signal'] + 
            self.technical_weight * df['technical_signal'] +
            model_weight * df['model_signal']
        )
        
        # 10. Discretize to -1, 0, 1
        df['raw_signal'] = df['raw_signal'].apply(
            lambda x: 1 if x > 0.3 else (-1 if x < -0.3 else 0)
        )
        
        # 11. Apply trading rules
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


if __name__ == '__main__':
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the analysis strategy with sample data
    import matplotlib.pyplot as plt
    from pandas import date_range
    
    # Create sample data
    dates = date_range('2023-01-01', periods=100, freq='D')
    
    # Price data
    close = 100 + np.cumsum(np.random.randn(len(dates)))
    high = close + np.random.rand(len(dates)) * 5
    low = close - np.random.rand(len(dates)) * 5
    open_price = close.shift(1).fillna(close[0])
    volume = np.random.randint(1000, 10000, size=len(dates))
    
    price_df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)
    
    # Sentiment data
    sentiment = np.random.normal(0.2, 0.3, size=len(dates))
    # Add some trend and noise
    for i in range(1, len(sentiment)):
        sentiment[i] = 0.8 * sentiment[i-1] + 0.2 * sentiment[i]
    
    sentiment_df = pd.DataFrame({
        'avg_compound': sentiment
    }, index=dates)
    
    # Create the analysis strategy
    strategy = AnalysisStrategy(
        sentiment_threshold=0.2,
        sentiment_change_threshold=0.05,
        min_holding_period=3,
        max_trades_per_week=2
    )
    
    # Calculate technical indicators
    indicators_df = strategy.calculate_technical_indicators(price_df)
    
    # Generate technical signals
    tech_signals = strategy.generate_technical_signals(indicators_df)
    
    # Generate combined strategy signals
    signals_df = strategy.generate_strategy_signals(price_df, sentiment_df, tech_signals)
    
    # Print trading statistics
    trades = signals_df[signals_df['order'] != 0]
    print(f"Generated {len(trades)} trades")
    print(f"Buy signals: {len(trades[trades['order'] > 0])}")
    print(f"Sell signals: {len(trades[trades['order'] < 0])}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot price
    ax1.plot(signals_df.index, signals_df['price'], label='Price')
    
    # Plot buy signals
    buys = signals_df[signals_df['order'] > 0]
    ax1.scatter(buys.index, buys['price'], marker='^', color='green', s=100, label='Buy')
    
    # Plot sell signals
    sells = signals_df[signals_df['order'] < 0]
    ax1.scatter(sells.index, sells['price'], marker='v', color='red', s=100, label='Sell')
    
    ax1.set_title('Price and Trading Signals')
    ax1.set_ylabel('Price')
    ax1.legend()
    
    # Plot sentiment
    ax2.plot(signals_df.index, signals_df['sentiment'], label='Sentiment')
    ax2.set_title('Sentiment')
    ax2.set_ylabel('Sentiment Score')
    ax2.set_xlabel('Date')
    
    plt.tight_layout()
    plt.show()