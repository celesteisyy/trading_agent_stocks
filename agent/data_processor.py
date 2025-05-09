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
import zstandard
import io, json
from datetime import datetime, timedelta
from dotenv import load_dotenv, find_dotenv
from scipy import stats
import zstandard as zstd
from datetime import datetime, timedelta
import concurrent.futures

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
        self.zst_dir = './data/reddit/submissions'
        
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
    

    def get_sp500_tech_stocks(self) -> List[str]:
        """
        Get all stocks in the S&P 500 technology sector.

        Tries the following methods:
        1. Fetch from the FMP API.
        2. If the API call fails, falls back to a hard‑coded list of common tech stocks.

        Returns:
            List[str]: A list of S&P 500 technology sector ticker symbols.
        """
        self.logger.info("Fetching S&P 500 technology sector stocks")
        sp500_tech_stocks = []
        
        # FMP API
        if self.fmp_api_key:
            try:
                url = f"{self._FMP_BASE_URL}/sp500_constituent?apikey={self.fmp_api_key}"
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                sp500_stocks = resp.json()
                
                sp500_tech_stocks = [
                    stock['symbol'] for stock in sp500_stocks
                    if stock['sector'].lower() in ['technology', 'information technology', 'it']
                ]
                
                self.logger.info(f"Successfully fetched {len(sp500_tech_stocks)} S&P 500 tech stocks from API")
                return sp500_tech_stocks
            except Exception as e:
                self.logger.warning(f"Failed to fetch S&P 500 tech stocks from API: {e}")
        
        default_tech_stocks = [
            'AAPL',  # Apple
            'MSFT',  # Microsoft
            'NVDA',  # NVIDIA
            'GOOGL', # Alphabet (Google) Class A
            'GOOG',  # Alphabet (Google) Class C
            'META',  # Meta Platforms (Facebook)
            'AVGO',  # Broadcom
            'ADBE',  # Adobe
            'CRM',   # Salesforce
            'CSCO',  # Cisco
            'ORCL',  # Oracle
            'ACN',   # Accenture
            'INTC',  # Intel
            'IBM',   # IBM
            'TXN',   # Texas Instruments
            'AMD',   # Advanced Micro Devices
            'QCOM',  # Qualcomm
            'AMAT',  # Applied Materials
            'ADI',   # Analog Devices
            'MU',    # Micron Technology
            'NOW',   # ServiceNow
            'INTU',  # Intuit
            'PYPL',  # PayPal
            'FISV',  # Fiserv
            'LRCX',  # Lam Research
            'ADSK',  # Autodesk
            'SNPS',  # Synopsys
            'KLAC',  # KLA Corporation
            'CDNS',  # Cadence Design Systems
            'NXPI',  # NXP Semiconductors
        ]
        
        self.logger.info(f"Using default list of {len(default_tech_stocks)} S&P 500 tech stocks")
        return default_tech_stocks

    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare a raw price-DataFrame for downstream use:
         - Flatten a MultiIndex column into single-level columns
         - Ensure the index is a DatetimeIndex
         - Sort by date
        """
        import pandas as pd

        # 1) If it's a multi‑ticker frame, flatten columns ("AAPL_Close", "MSFT_Open", …)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # 2) Make sure the index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 3) Sort by date, drop any exact duplicates
        df = df.sort_index().loc[~df.index.duplicated(keep='first')]

        return df

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
            # Try with .csv extension
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
            path = os.path.join(base, f"{ticker}.csv")
            
            try:
                if os.path.exists(path):
                    self.logger.info(f"Loading {ticker} data from {path}")
                    df_csv = pd.read_csv(path, parse_dates=["date"], index_col="date")
                    
                    # Check column names and standardize if needed
                    if 'close' in [col.lower() for col in df_csv.columns]:
                        # Get the actual case-sensitive column name
                        close_col = next(col for col in df_csv.columns if col.lower() == 'close')
                        # Rename to standard 'Close'
                        if close_col != 'Close':
                            df_csv = df_csv.rename(columns={close_col: 'Close'})
                    
                    # Add ticker level if needed
                    if len(tickers) > 1:
                        df_csv.columns = pd.MultiIndex.from_product([[ticker], df_csv.columns])
                    
                    csv_frames.append(df_csv)
                    self.logger.info(f"Successfully loaded {ticker} from local CSV")
                else:
                    self.logger.warning(f"No local CSV file found for {ticker}")
            except Exception as e3:
                self.logger.warning(f"Local CSV load failed for {ticker} ({e3})")

        # Handle the case where we have data in the current working directory
        if not csv_frames and os.path.exists("AAPL.csv") and 'AAPL' in tickers:
            try:
                self.logger.info("Trying to load AAPL from current directory")
                df_csv = pd.read_csv("AAPL.csv", parse_dates=["date"], index_col="date")
                if 'close' in [col.lower() for col in df_csv.columns]:
                    close_col = next(col for col in df_csv.columns if col.lower() == 'close')
                    if close_col != 'Close':
                        df_csv = df_csv.rename(columns={close_col: 'Close'})
                if len(tickers) > 1:
                    df_csv.columns = pd.MultiIndex.from_product([['AAPL'], df_csv.columns])
                csv_frames.append(df_csv)
                self.logger.info("Successfully loaded AAPL from current directory")
            except Exception as e:
                self.logger.warning(f"Failed to load AAPL from current directory: {e}")

        if csv_frames:
            result = pd.concat(csv_frames, axis=1).sort_index()
            self.logger.info(f"Successfully loaded price data from local CSV files")
            df = self._preprocess_dataframe(result)
            df = df.loc[self.start_date:self.end_date]
            return df

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
        Efficiently load Reddit posts from local .zst archives:
        1. Prioritizes newest .zst files
        2. Uses sampling to quickly scan large files
        3. Implements early termination strategies
        4. Falls back to synthetic data if needed
        """
        # Skip PRAW (API) entirely and focus on .zst files
        candidate_dirs = [
            os.path.join(os.path.dirname(__file__), "data", "reddit", "submissions"),
            "./data/reddit/submissions",
            "data/reddit/submissions",
            os.path.abspath(os.path.join("data", "reddit", "submissions"))
        ]
        base_dir = next((d for d in candidate_dirs if os.path.isdir(d)), None)
        if not base_dir:
            self.logger.warning("[Data] Local Reddit directory not found; using synthetic data.")
            return self._generate_synthetic_reddit(subreddit)

        # Find and sort .zst files by name (assuming YYYY-MM format in filenames)
        zst_files = [
            os.path.join(base_dir, f)
            for f in sorted(os.listdir(base_dir), reverse=True)  # Reverse to get newest first
            if f.lower().endswith('.zst')
        ]
        
        if not zst_files:
            self.logger.warning("[Data] No .zst files found; using synthetic data.")
            return self._generate_synthetic_reddit(subreddit)
            
        self.logger.info(f"Found {len(zst_files)} .zst files in {base_dir}, processing newest first.")

        def _process_file_with_sampling(path, target_sample_size=3000):
            """Process a .zst file with intelligent sampling to avoid reading the entire file."""
            try:
                # Verify file has valid zstd header
                with open(path, 'rb') as fh:
                    if fh.read(4) != b"\x28\xb5\x2f\xfd":
                        self.logger.warning(f"Skipping non‑zstd file: {path}")
                        return None

                # Get file size to determine sampling strategy
                file_size = os.path.getsize(path)
                
                # For very large files (>100MB), use aggressive sampling
                if file_size > 100_000_000:  # 100MB
                    sample_rate = max(1, int(file_size / (2_000_000 * target_sample_size)))  # Read ~every Nth post
                    self.logger.info(f"Large file detected ({file_size/1_000_000:.1f}MB), sampling 1/{sample_rate} lines")
                else:
                    sample_rate = 1  # Read every post for smaller files
                
                posts = []
                line_counter = 0
                posts_found = 0
                
                # Set up decompression stream with a limited buffer size to control memory usage
                dctx = zstd.ZstdDecompressor()
                with open(path, 'rb') as compressed:
                    with dctx.stream_reader(compressed) as reader:
                        text_reader = io.TextIOWrapper(reader, encoding='utf-8')
                        
                        # Use a 50MB read limit to avoid memory issues on very large files
                        bytes_read = 0
                        max_bytes = 50_000_000  # 25MB limit
                        
                        for line in text_reader:
                            line_counter += 1
                            bytes_read += len(line)
                            
                            # Apply sampling rate to avoid processing every line in large files
                            if line_counter % sample_rate != 0:
                                continue
                                
                            try:
                                # Fast check if subreddit is likely in this line before full JSON parse
                                if f'"subreddit":"{subreddit}"' not in line and f'"subreddit": "{subreddit}"' not in line:
                                    # Quick reject for obvious non-matches
                                    continue
                                    
                                post = json.loads(line)
                                if post.get('subreddit', '').lower() == subreddit.lower():
                                    # Extract only needed fields to save memory
                                    filtered_post = {
                                        'id': post.get('id', ''),
                                        'title': post.get('title', ''),
                                        'selftext': post.get('selftext', ''),
                                        'created_utc': post.get('created_utc', 0),
                                        'score': post.get('score', 0),
                                        'num_comments': post.get('num_comments', 0),
                                        'url': post.get('url', '')
                                    }
                                    posts.append(filtered_post)
                                    posts_found += 1
                                    
                                    # Stop after finding enough posts
                                    if posts_found >= target_sample_size:
                                        self.logger.info(f"Found {posts_found} posts in {path}, stopping early")
                                        break
                            except json.JSONDecodeError:
                                continue
                            
                            # Early termination if read limit exceeded
                            if bytes_read > max_bytes:
                                self.logger.info(f"Reached {max_bytes/1_000_000:.1f}MB read limit with {posts_found} posts in {path}")
                                break

                if not posts:
                    return None

                # Convert to DataFrame
                df = pd.DataFrame(posts)
                # Convert timestamp to datetime
                if 'created_utc' in df.columns:
                    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
                
                return df

            except zstd.ZstdError as e:
                self.logger.error(f"Decompression failed for {path}: {e}")
                return None
            except Exception as e:
                self.logger.error(f"Error processing {path}: {e}")
                return None

        # Process only the newest files first (most relevant data)
        max_files_to_process = min(5, len(zst_files))
        files_to_process = zst_files[:max_files_to_process]
        self.logger.info(f"Processing {len(files_to_process)} newest .zst files out of {len(zst_files)} total")

        # Calculate posts to request from each file based on limit
        posts_per_file = max(100, limit // max_files_to_process)
        
        all_posts = []
        total_posts = 0
        
        # Process files sequentially to avoid excessive memory usage with multiple threads
        # For large archives, sequential processing with smart sampling is often faster than
        # parallel processing with full file reads
        for file_path in files_to_process:
            self.logger.info(f"Processing {os.path.basename(file_path)}")
            df = _process_file_with_sampling(file_path, target_sample_size=posts_per_file)
            
            if df is not None and not df.empty:
                all_posts.append(df)
                total_posts += len(df)
                
                if total_posts >= limit:
                    self.logger.info(f"Reached target of {limit} posts, stopping file processing")
                    break
        
        if not all_posts:
            self.logger.warning(f"No posts found in .zst files for r/{subreddit}, generating synthetic data")
            return self._generate_synthetic_reddit(subreddit)
        
        # Combine results and sort by date
        result_df = pd.concat(all_posts, ignore_index=True)
        
        if 'created_utc' in result_df.columns:
            result_df = result_df.sort_values('created_utc', ascending=False).reset_index(drop=True)
        
        # Limit to requested number
        result_df = result_df.head(limit)
        
        self.logger.info(f"Successfully loaded {len(result_df)} posts for r/{subreddit} from .zst files")
        return result_df
    
    def compute_technical_features(
        self,
        df: pd.DataFrame,
        ticker: str = None,
        macro_df: Optional[pd.DataFrame] = None,
        defi_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute technical indicators for price data and merge macro/DeFi features.
        Args:
            df: DataFrame with price data (must have 'Close' column)
            ticker: Optional ticker symbol to select specific close column when multiple exist
            macro_df: Optional DataFrame with macroeconomic series indexed by date
            defi_df: Optional DataFrame with DeFi price series indexed by date
        Returns:
            DataFrame with technical indicators plus macro/DeFi features
        """
        # Build mapping from lowercase column names to original names
        lower_to_orig = {col.lower(): col for col in df.columns}

        # Determine which column to treat as Close
        if 'close' in lower_to_orig:
            close_orig = lower_to_orig['close']
        else:
            candidates = [orig for low, orig in lower_to_orig.items() if low.endswith('_close')]
            if ticker is not None:
                ticker_close = f"{ticker}_Close"
                if ticker_close in df.columns:
                    close_orig = ticker_close
                else:
                    raise ValueError(f"Could not find close column for ticker {ticker}")
            elif len(candidates) == 1:
                close_orig = candidates[0]
            elif len(candidates) > 1:
                first_ticker = candidates[0].split('_')[0]
                self.logger.warning(
                    f"Found multiple '_close' columns: {candidates!r}. Using {first_ticker}'s close by default."
                )
                close_orig = candidates[0]
            else:
                raise ValueError(
                    "DataFrame must have one column named 'Close' or exactly one ending with '_close'."
                )

        # Copy and assign selected column to 'Close'
        df = df.copy()
        df['Close'] = df[close_orig]

        # Initialize result with base features
        result = df.copy()

        # Returns
        result['return'] = result['Close'].pct_change()
        result['log_return'] = np.log(result['Close'] / result['Close'].shift(1))

        # Moving averages and EMAs
        for window in [5, 20, 60]:
            result[f'sma_{window}'] = result['Close'].rolling(window).mean()
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

        # Merge macroeconomic features if provided
        if macro_df is not None and not macro_df.empty:
            macro_aligned = macro_df.reindex(result.index).ffill().bfill()
            for col in macro_aligned.columns:
                result[f'macro_{col}'] = macro_aligned[col]

        # Merge DeFi features if provided
        if defi_df is not None and not defi_df.empty:
            defi_aligned = defi_df.reindex(result.index).ffill().bfill()
            for col in defi_aligned.columns:
                result[f'defi_{col}'] = defi_aligned[col]

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

        # SMA crossover signals
        signals['sma_sig'] = 0
        sma20 = df['sma_20']
        sma_up = (df['Close'] > sma20) & (df['Close'].shift(1) <= sma20.shift(1))
        sma_down = (df['Close'] < sma20) & (df['Close'].shift(1) >= sma20.shift(1))
        signals.loc[sma_up, 'sma_sig'] = 1
        signals.loc[sma_down, 'sma_sig'] = -1

        # Combined signal
        signals['combined_sig'] = signals.sum(axis=1)
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