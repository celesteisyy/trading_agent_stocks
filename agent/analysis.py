# Compute technical indicators (SMA, EMA, RSI, MACD, BBANDS, ATR) and raw signals
import pandas as pd
import numpy as np

class AnalysisAgent:
    """
    AnalysisAgent for calculating technical indicators and generating raw trading signals.

    Indicators computed:
      - Simple Moving Averages (SMA)
      - Exponential Moving Averages (EMA)
      - Relative Strength Index (RSI)
      - Moving Average Convergence Divergence (MACD)
      - Bollinger Bands (upper/lower)
      - Average True Range (ATR)

    Raw signals generated:
      - RSI-based: buy when oversold (RSI < 30), sell when overbought (RSI > 70)
      - MACD crossover: buy on bullish crossover, sell on bearish crossover
      - Bollinger Bands: buy when price < lower band, sell when price > upper band
      - SMA crossover: buy/sell when price crosses 20-day SMA
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize AnalysisAgent.

        Args:
            df: Price DataFrame indexed by datetime with columns: 'Open', 'High', 'Low', 'Close', 'Volume'
        """
        self.df = df.copy()
        self.indicators = pd.DataFrame(index=self.df.index)
        self.signals = pd.DataFrame(index=self.df.index)

    def calculate_sma(self, window: int):
        """Compute simple moving average."""
        self.indicators[f'sma_{window}'] = self.df['Close'].rolling(window=window).mean()

    def calculate_ema(self, window: int):
        """Compute exponential moving average."""
        self.indicators[f'ema_{window}'] = (
            self.df['Close'].ewm(span=window, adjust=False).mean()
        )

    def calculate_rsi(self, window: int = 14):
        """Compute Relative Strength Index (RSI)."""
        delta = self.df['Close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        # use Wilder's smoothing via EWM
        roll_up = up.ewm(com=window-1, adjust=False).mean()
        roll_down = down.ewm(com=window-1, adjust=False).mean()
        rs = roll_up / roll_down
        self.indicators[f'rsi_{window}'] = 100 - (100 / (1 + rs))

    def calculate_macd(self, fast: int = 12, slow: int = 26, signal_window: int = 9):
        """Compute MACD line and signal line."""
        ema_fast = self.df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['Close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        self.indicators['macd'] = macd_line
        self.indicators['macd_signal'] = signal_line

    def calculate_bollinger_bands(self, window: int = 20, num_std: float = 2.0):
        """Compute upper and lower Bollinger Bands."""
        rolling_mean = self.df['Close'].rolling(window).mean()
        rolling_std = self.df['Close'].rolling(window).std()
        self.indicators['bb_upper'] = rolling_mean + num_std * rolling_std
        self.indicators['bb_lower'] = rolling_mean - num_std * rolling_std

    def calculate_atr(self, window: int = 14):
        """Compute Average True Range (ATR)."""
        high_low = self.df['High'] - self.df['Low']
        high_close = (self.df['High'] - self.df['Close'].shift()).abs()
        low_close = (self.df['Low'] - self.df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.indicators[f'atr_{window}'] = true_range.rolling(window=window).mean()

    def compute_all_indicators(self):
        """Convenience method to compute all indicators with default settings."""
        for w in [5, 20, 60]:
            self.calculate_sma(w)
            self.calculate_ema(w)
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_atr()

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate raw buy/sell signals based on technical indicators.

        Returns:
            DataFrame of signals with columns: 'rsi_sig', 'macd_sig', 'bb_sig', 'sma_sig'.
        """
        df = self.df
        ind = self.indicators
        signals = pd.DataFrame(index=df.index)

        # RSI signals
        signals['rsi_sig'] = 0
        signals.loc[ind['rsi_14'] < 30, 'rsi_sig'] = 1
        signals.loc[ind['rsi_14'] > 70, 'rsi_sig'] = -1

        # MACD crossover signals
        macd = ind['macd']
        macd_sig = ind['macd_signal']
        crossover_up = (macd > macd_sig) & (macd.shift(1) <= macd_sig.shift(1))
        crossover_down = (macd < macd_sig) & (macd.shift(1) >= macd_sig.shift(1))
        signals['macd_sig'] = 0
        signals.loc[crossover_up, 'macd_sig'] = 1
        signals.loc[crossover_down, 'macd_sig'] = -1

        # Bollinger Bands signals
        signals['bb_sig'] = 0
        signals.loc[df['Close'] < ind['bb_lower'], 'bb_sig'] = 1
        signals.loc[df['Close'] > ind['bb_upper'], 'bb_sig'] = -1

        # SMA(20) crossover signals
        sma20 = ind['sma_20']
        sma_up = (df['Close'] > sma20) & (df['Close'].shift(1) <= sma20.shift(1))
        sma_down = (df['Close'] < sma20) & (df['Close'].shift(1) >= sma20.shift(1))
        signals['sma_sig'] = 0
        signals.loc[sma_up, 'sma_sig'] = 1
        signals.loc[sma_down, 'sma_sig'] = -1

        self.signals = signals
        return signals


if __name__ == '__main__':
    # Self-check / example usage
    from data_loader import DataLoader

    # Load a single-ticker price series for testing
    loader = DataLoader(start_date='2021-01-01', end_date='2025-05-01')
    df_price = loader.load_stock_prices(['AAPL'])
    # If resulting df has MultiIndex columns, flatten for one ticker
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = df_price.columns.droplevel(0)

    # Instantiate and run
    agent = AnalysisAgent(df_price)
    agent.compute_all_indicators()
    signals = agent.generate_signals()

    # Print head of indicators and signals for manual inspection
    print(agent.indicators[['sma_20','rsi_14','macd','macd_signal','bb_upper','bb_lower','atr_14']].dropna().head())
    print(signals.dropna().head())

    # this is what you may need if encounter YF rate limit while still need to test the system
#    dates = pd.date_range('2021-01-01', periods=200, freq='B')

    # 2) Simulate a random‑walk close price around 100
#    np.random.seed(42)
#    close = 100 + np.cumsum(np.random.randn(len(dates)))

    # 3) Build O/H/L/C/Volume columns
#    df = pd.DataFrame({
#        'Open':  close + np.random.randn(len(dates))*0.5,
#        'High':  close + np.abs(np.random.randn(len(dates))*1.0),
#        'Low':   close - np.abs(np.random.randn(len(dates))*1.0),
#        'Close': close,
#        'Volume': np.random.randint(100, 1000, size=len(dates))
#    }, index=dates)

    # 4) Run the AnalysisAgent
#    agent  = AnalysisAgent(df)
#   agent.compute_all_indicators()
#    signals = agent.generate_signals()
#    print(signals.head())
#    """
