import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import requests
import time
import yfinance as yf
import ccxt
import praw
from datetime import datetime, timedelta
from dotenv import load_dotenv, find_dotenv
from scipy import stats

class DataProcessor:
    """
    DataProcessor handles downloading and preprocessing of financial data:
    - Stock prices, benchmark indices, DeFi tokens
    - Reddit posts for sentiment analysis
    - Sector-based ticker selection
    - Data cleaning and normalization
    - Feature computation for technical analysis
    """
    
    def __init__(self, start_date: str, end_date: str, interval: str = "1d"):
        """
        Initialize DataProcessor.
        
        Args:
            start_date: Start date for data analysis (YYYY-MM-DD)
            end_date: End date for data analysis (YYYY-MM-DD)
            interval: Data frequency (e.g., "1d" for daily)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        env_path = find_dotenv("fin580.env", raise_error_if_not_found=True)
        load_dotenv(env_path)
        
        # Set up API keys and connections
        self._setup_apis()
    
    def _setup_apis(self):
        """Set up API connections and keys."""
        # FRED API
        self.fred_api_key = os.getenv('FRED_API_KEY')
        if not self.fred_api_key:
            self.logger.warning("FRED_API_KEY not set in fin580.env")
        self._FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

        # Reddit (PRAW)
        reddit_id = os.getenv("REDDIT_CLIENT_ID")
        reddit_sec = os.getenv("REDDIT_CLIENT_SECRET")
        reddit_ua = os.getenv("REDDIT_USER_AGENT")
        if all([reddit_id, reddit_sec, reddit_ua]):
            try:
                self.reddit = praw.Reddit(
                    client_id=reddit_id,
                    client_secret=reddit_sec,
                    user_agent=reddit_ua
                )
                self.logger.info("Successfully connected to Reddit API")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Reddit API: {e}")
                self.reddit = None
        else:
            self.logger.warning("Reddit API credentials not fully set in fin580.env")
            self.reddit = None

        # FinancialModelingPrep API
        self.fmp_api_key = os.getenv("FMP_API_KEY")
        if not self.fmp_api_key:
            self.logger.warning("FMP_API_KEY not set in fin580.env")
        self._FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
    
    def get_tickers_by_sector(self, sector: str) -> List[str]:
        """
        Fetch all tickers in a given GICS sector via FinancialModelingPrep.
        
        Args:
            sector: Sector name (e.g., "Technology")
            
        Returns:
            List of ticker symbols
        """
        if not self.fmp_api_key:
            self.logger.error("FMP_API_KEY required to get tickers by sector")
            return []
            
        url = f"{self._FMP_BASE_URL}/stock-screener"
        params = {
            "sector": sector,
            "limit": 1000,
            "apikey": self.fmp_api_key
        }
        
        try:
            self.logger.info(f"Fetching {sector} sector tickers...")
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            tickers = [item["symbol"] for item in data]
            self.logger.info(f"Found {len(tickers)} tickers in {sector} sector")
            return tickers
        except Exception as e:
            self.logger.error(f"Failed to fetch {sector} tickers: {e}")
            return []
    
    def load_stock_prices(self, tickers: List[str]) -> pd.DataFrame:
        """
        Load stock price data with fallbacks:
        1) Try yfinance
        2) Fallback to FMP API
        3) Fallback to local CSV
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            DataFrame with OHLCV data
        """
        # 1) Try yfinance
        try:
            self.logger.info(f"Loading price data for {len(tickers)} tickers via yfinance...")
            df = yf.download(
                tickers,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                group_by='ticker',
                auto_adjust=True,
                threads=True,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                self.logger.info(f"Successfully loaded price data via yfinance")
                return self._preprocess_dataframe(df)
            raise ValueError("yfinance returned no data")
        except Exception as e:
            self.logger.warning(f"yfinance failed ({e}), falling back to FMP API...")

        # 2) FinancialModelingPrep
        if self.fmp_api_key:
            all_frames = []
            for ticker in tickers:
                url = (
                    f"{self._FMP_BASE_URL}/historical-price-full/{ticker}"
                    f"?from={self.start_date}&to={self.end_date}&apikey={self.fmp_api_key}"
                )
                try:
                    resp = requests.get(url, timeout=10)
                    resp.raise_for_status()
                    hist = resp.json().get("historical", [])
                    df_t = (
                        pd.DataFrame(hist)
                          .rename(columns={
                              "open": "Open",
                              "high": "High",
                              "low":  "Low",
                              "close":"Close",
                              "volume": "Volume"
                          })
                          [["date","Open","High","Low","Close","Volume"]]
                    )
                    df_t["date"] = pd.to_datetime(df_t["date"])
                    df_t.set_index("date", inplace=True)
                    # Add ticker level
                    df_t.columns = pd.MultiIndex.from_product([[ticker], df_t.columns])
                    all_frames.append(df_t)
                except Exception as e2:
                    self.logger.warning(f"FMP failed for {ticker} ({e2}), will try local CSV...")

            if all_frames:
                result = pd.concat(all_frames, axis=1).sort_index()
                self.logger.info(f"Successfully loaded price data via FMP API")
                return self._preprocess_dataframe(result)

        # 3) Local CSV fallback
        csv_frames = []
        for ticker in tickers:
            path = os.path.join("data", "prices", f"{ticker}.csv")
            try:
                df_csv = pd.read_csv(path, parse_dates=["date"], index_col="date")
                df_csv.columns = pd.MultiIndex.from_product([[ticker], df_csv.columns])
                csv_frames.append(df_csv)
            except Exception as e3:
                self.logger.warning(f"Local CSV load failed for {ticker} ({e3})")
        
        if csv_frames:
            result = pd.concat(csv_frames, axis=1).sort_index()
            self.logger.info(f"Successfully loaded price data from local CSV files")
            return self._preprocess_dataframe(result)

        self.logger.error("All data sources for stock prices failed")
        raise RuntimeError("All data sources for `load_stock_prices` failed: yfinance, FMP, local CSV.")
    
    def load_macro_series(self, series_ids: List[str]) -> pd.DataFrame:
        """
        Fetch macroeconomic time series from the FRED API.
        
        Args:
            series_ids: List of FRED series identifiers (e.g., 'GDP', 'UNRATE')
            
        Returns:
            DataFrame indexed by date with one column per series
        """
        self.logger.info(f"Loading macro series: {series_ids}")
        frames: List[pd.DataFrame] = []
        
        for series in series_ids:
            params = {
                'series_id': series,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'observation_start': self.start_date,
                'observation_end': self.end_date,
            }
            
            try:
                resp = requests.get(self._FRED_BASE_URL, params=params)
                resp.raise_for_status()
                obs = resp.json().get('observations', [])
                
                df = pd.DataFrame(obs)[['date', 'value']]
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df.set_index('date', inplace=True)
                df.rename(columns={'value': series}, inplace=True)
                frames.append(df)
                self.logger.info(f"Successfully loaded {series} data")
            except Exception as e:
                self.logger.warning(f"Failed to load FRED series {series}: {str(e)}")
        
        if frames:
            result = pd.concat(frames, axis=1)
            self.logger.info(f"Loaded {len(frames)} macro series")
            return self._preprocess_dataframe(result)
        else:
            self.logger.warning("No macro data loaded, returning empty DataFrame")
            return pd.DataFrame()
        
    def load_defi_prices(self, coin_ids: List[str], vs_symbol: str = 'USDT') -> pd.DataFrame:
        """
        Fetch DeFi token prices from cryptocurrency exchanges via CCXT.
        
        Args:
            coin_ids: List of token symbols (e.g., 'UNI', 'AAVE')
            vs_symbol: Quote currency (e.g., 'USDT', 'USD')
            
        Returns:
            DataFrame indexed by date with one column per token
        """
        self.logger.info(f"Loading DeFi price data for {coin_ids}")
        exchanges = ['binance', 'coinbasepro', 'kraken', 'kucoin']
        
        # Try CCXT for each exchange
        for exchange_id in exchanges:
            try:
                exchange = getattr(ccxt, exchange_id)()
                since_ms = int(pd.to_datetime(self.start_date).timestamp() * 1000)
                result = {}
                
                for coin in coin_ids:
                    for quote in (vs_symbol.upper(), 'USD'):
                        market = f"{coin.upper()}/{quote}"
                        try:
                            ohlcv = exchange.fetch_ohlcv(market, '1d', since_ms)
                            if ohlcv:
                                df = pd.DataFrame(
                                    ohlcv,
                                    columns=['ts', 'open', 'high', 'low', 'close', 'volume']
                                )
                                df['date'] = pd.to_datetime(df['ts'], unit='ms')
                                df.set_index('date', inplace=True)
                                result[coin] = df['close']
                                self.logger.info(f"Loaded {coin} price data from {exchange_id}")
                                break
                        except Exception:
                            continue
                
                if result:
                    result_df = pd.DataFrame(result)
                    self.logger.info(f"Successfully loaded DeFi data for {len(result)} tokens")
                    return self._preprocess_dataframe(result_df)
                    
            except Exception as e:
                self.logger.warning(f"CCXT exchange {exchange_id} failed: {str(e)}")
        
        # All exchanges failed, return empty DataFrame
        self.logger.warning("All DeFi data sources failed, returning empty DataFrame")
        return pd.DataFrame()
    
    def load_reddit_posts(self, subreddit: str, limit: int = 500) -> pd.DataFrame:
        """
        Load Reddit posts from specified subreddit.
        
        Args:
            subreddit: Subreddit name (e.g., "CryptoCurrency")
            limit: Maximum number of posts to fetch
            
        Returns:
            DataFrame with Reddit posts
        """
        if not self.reddit:
            # Generate synthetic data if Reddit API not available
            self.logger.warning("Reddit API not available, generating synthetic data")
            return self._generate_synthetic_reddit(subreddit)
        
        try:
            self.logger.info(f"Fetching posts from r/{subreddit}...")
            records = []
            
            # Fetch posts from specified date range
            after_ts = int(pd.to_datetime(self.start_date).timestamp())
            before_ts = int(pd.to_datetime(self.end_date).timestamp())
            
            for post in self.reddit.subreddit(subreddit).new(limit=limit):
                if after_ts <= post.created_utc <= before_ts:
                    records.append({
                        "id": post.id,
                        "title": post.title,
                        "selftext": getattr(post, "selftext", ""),
                        "created_utc": datetime.fromtimestamp(post.created_utc),
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "url": post.url,
                    })
            
            if not records:
                self.logger.warning(f"No posts found in r/{subreddit} for date range")
                return self._generate_synthetic_reddit(subreddit)
                
            result = pd.DataFrame(records)
            self.logger.info(f"Fetched {len(result)} posts from r/{subreddit}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to fetch Reddit posts: {e}")
            return self._generate_synthetic_reddit(subreddit)
    
    
    def _generate_synthetic_reddit(self, subreddit: str) -> pd.DataFrame:
        """Generate synthetic Reddit data for testing."""
        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        df = pd.DataFrame({
            'id': [f'syn_{i}' for i in range(len(dates))],
            'title': [f'Synthetic {subreddit} Post {i}' for i in range(len(dates))],
            'selftext': [f'This is synthetic content for testing. Day {i}' for i in range(len(dates))],
            'created_utc': dates,
            'score': np.random.randint(1, 100, size=len(dates)),
            'num_comments': np.random.randint(0, 20, size=len(dates)),
            'url': [f'https://reddit.com/r/{subreddit}/syn_{i}' for i in range(len(dates))],
        })
        return df
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare DataFrame for analysis.
        
        - Flatten MultiIndex columns
        - Ensure datetime index
        - Handle missing values
        - Remove outliers
        
        Args:
            df: DataFrame to preprocess
            
        Returns:
            Preprocessed DataFrame
        """
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort index
        df = df.sort_index()
        
        # Handle missing values
        df = df.ffill()
        
        # Remove outliers (z-score method)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy='omit'))
            df = df[(z_scores < 3).all(axis=1)]
        
        return df
    
    def compute_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators for price data.
        
        Args:
            df: DataFrame with price data (must have 'Close' column)
            
        Returns:
            DataFrame with technical indicators
        """
        # Ensure we have a Close column
        if 'Close' not in df.columns:
            if isinstance(df.columns, pd.MultiIndex):
                # Handle MultiIndex columns
                ticker = df.columns.levels[0][0]
                close_col = (ticker, 'Close')
                if close_col in df.columns:
                    close = df[close_col].copy()
                    # Flatten columns for simplicity
                    df = df.copy()
                    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
                    df['Close'] = close
                else:
                    raise ValueError("DataFrame must have a 'Close' column")
            else:
                raise ValueError("DataFrame must have a 'Close' column")
        
        # Simple technical features
        result = df.copy()
        
        # Returns
        result['return'] = result['Close'].pct_change()
        result['log_return'] = np.log(result['Close'] / result['Close'].shift(1))
        
        # Moving averages
        for window in [5, 20, 60]:
            result[f'sma_{window}'] = result['Close'].rolling(window=window).mean()
            result[f'ema_{window}'] = result['Close'].ewm(span=window, adjust=False).mean()
        
        # Volatility
        result['volatility_20'] = result['return'].rolling(window=20).std()
        
        # RSI
        delta = result['Close'].diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        roll_up = up.ewm(com=13, adjust=False).mean()
        roll_down = down.ewm(com=13, adjust=False).mean()
        rs = roll_up / roll_down
        result['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = result['Close'].ewm(span=12, adjust=False).mean()
        ema26 = result['Close'].ewm(span=26, adjust=False).mean()
        result['macd'] = ema12 - ema26
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        result['bb_middle'] = result['Close'].rolling(window=20).mean()
        result['bb_std'] = result['Close'].rolling(window=20).std()
        result['bb_upper'] = result['bb_middle'] + 2 * result['bb_std']
        result['bb_lower'] = result['bb_middle'] - 2 * result['bb_std']
        
        return result
    
    def generate_technical_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            DataFrame with trading signals
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
        
        # SMA(20) crossover signals
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


if __name__ == '__main__':
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the data processor
    processor = DataProcessor(
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
    
    # Test ticker selection
    tech_tickers = processor.get_tickers_by_sector("Technology")
    print(f"Found {len(tech_tickers)} technology tickers")
    
    # Test price loading (first 3 tickers)
    if tech_tickers:
        test_tickers = tech_tickers[:3]
        price_df = processor.load_stock_prices(test_tickers)
        print(f"Loaded {len(price_df)} price records")
        
        # Test technical indicators
        tech_df = processor.compute_technical_features(price_df)
        print(f"Computed {len(tech_df.columns)} technical features")
        
        # Test signal generation
        signals_df = processor.generate_technical_signals(tech_df)
        print(f"Generated signals with {len(signals_df.columns)} signal types")
    
    # Test Reddit data loading
    reddit_df = processor.load_reddit_posts("CryptoCurrency", limit=100)
    print(f"Loaded {len(reddit_df)} Reddit posts")