import os
from dotenv import load_dotenv,find_dotenv
import pandas as pd
from typing import List, Dict, Optional
import requests
import time
import yfinance as yf
import ccxt
from concurrent.futures import ThreadPoolExecutor
import praw

from datetime import datetime

# Load environment variables from fin580.env
env_path = find_dotenv("fin580.env", raise_error_if_not_found=True)
load_dotenv(env_path)


class DataLoader:
    """
    DataLoader handles downloading of various financial time series:
      - Equity prices via yfinance
      - Benchmark indices via yfinance
      - Cryptocurrency prices via yfinance
      - DeFi token prices via CCXT
      - Macroeconomic series via the FRED API
      - Sector‑based tickers via FinancialModelingPrep API
      - Reddit posts via PRAW
    """
    def __init__(self, start_date: str, end_date: str, interval: str = "1d"):
        """
        Initialize the DataLoader with time range parameters.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date:   End date in 'YYYY-MM-DD' format
            interval:   Data interval ('1d', '1wk', '1mo', etc.)
        """
        self.start_date = start_date
        self.end_date   = end_date
        self.interval   = interval

        # FRED API
        self.fred_api_key = os.getenv('FRED_API_KEY')
        if not self.fred_api_key:
            raise ValueError("FRED_API_KEY must be set in fin580.env")
        self._FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

        # Reddit (PRAW)
        reddit_id  = os.getenv("REDDIT_CLIENT_ID")
        reddit_sec = os.getenv("REDDIT_CLIENT_SECRET")
        reddit_ua  = os.getenv("REDDIT_USER_AGENT")
        if not all([reddit_id, reddit_sec, reddit_ua]):
            raise ValueError(
                "REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT must be set in fin580.env"
            )
        self.reddit = praw.Reddit(
            client_id=reddit_id,
            client_secret=reddit_sec,
            user_agent=reddit_ua
        )

        # FinancialModelingPrep API
        self.fmp_api_key = os.getenv("FMP_API_KEY")
        if not self.fmp_api_key:
            raise ValueError("FMP_API_KEY must be set in fin580.env")
        self._FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

    def load_stock_prices(self, tickers: List[str]) -> pd.DataFrame:
        """
        Download historical adjusted prices for equity tickers.
        Returns a MultiIndex DataFrame with ('Ticker', ['Open','High','Low','Close','Volume']).
        """
        return yf.download(
            tickers,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            group_by='ticker',
            auto_adjust=True,
            threads=True,
        )

    def load_benchmark_indices(self, symbols: List[str]) -> pd.DataFrame:
        """
        Download benchmark index data (e.g., ^GSPC, ^RUI, ^RUT).
        Returns a DataFrame with columns ['Open','High','Low','Close','Volume'].
        """
        return yf.download(
            symbols,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            auto_adjust=True,
            threads=True,
        )

    def load_defi_prices(self,coin_ids: List[str],vs_symbol: str = 'USDT',ccxt_exchanges: List[str] = ['kraken', 'coinbasepro', 'bitfinex'],timeframe: str = '1d') -> pd.DataFrame:
        """
        Fetch closing prices for DeFi tokens, trying multiple quote currencies:

        1) For each CCXT exchange in order:
        • Try market SYMBOL/VS (e.g. UNI/USDT)
        • If that fails, try SYMBOL/USD
        • Move on as soon as one quote works per coin
        2) If no exchange yields any data for any coin, return an empty DataFrame.

        Returns:
            DataFrame indexed by timestamp, one column per token.
        """
        # Step 1: CCXT path
        for exchange_id in ccxt_exchanges:
            try:
                exchange = getattr(ccxt, exchange_id)()
                since_ms = exchange.parse8601(f"{self.start_date}T00:00:00Z")
                result: Dict[str, pd.Series] = {}

                for coin in coin_ids:
                    for quote in (vs_symbol.upper(), 'USD'):
                        symbol = f"{coin.upper()}/{quote}"
                        try:
                            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since_ms)
                            df = pd.DataFrame(
                                ohlcv,
                                columns=['ts', 'open', 'high', 'low', 'close', 'volume']
                            )
                            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                            df.set_index('ts', inplace=True)
                            result[coin] = df['close']
                            break  # stop trying quotes for this coin
                        except Exception:
                            continue  # try next quote

                if result:
                    return pd.DataFrame(result)

            except Exception:
                # this exchange failed entirely → try next exchange
                continue

        # Step 2: if we fall through all exchanges without any data
        return pd.DataFrame()

    def load_macro_series(self, series_ids: List[str]) -> pd.DataFrame:
        """
        Fetch macroeconomic time series from the FRED API.
        Returns a DataFrame indexed by date with one column per series.
        """
        frames: List[pd.DataFrame] = []
        for series in series_ids:
            params = {
                'series_id':         series,
                'api_key':           self.fred_api_key,
                'file_type':         'json',
                'observation_start': self.start_date,
                'observation_end':   self.end_date,
            }
            resp = requests.get(self._FRED_BASE_URL, params=params)
            resp.raise_for_status()
            obs = resp.json().get('observations', [])

            df = pd.DataFrame(obs)[['date', 'value']]
            df['date']  = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df.set_index('date', inplace=True)
            df.rename(columns={'value': series}, inplace=True)
            frames.append(df)

        return pd.concat(frames, axis=1) if frames else pd.DataFrame()

    def get_tickers_by_sector(self, sector: str) -> List[str]:
        """
        Fetch all tickers in a given GICS sector via FinancialModelingPrep.
        """
        url = f"{self._FMP_BASE_URL}/stock-screener"
        params = {
            "sector": sector,
            "limit":  1000,
            "apikey": self.fmp_api_key
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return [item["symbol"] for item in data]

    def load_tech_and_fintech(self) -> pd.DataFrame:
        """
        Download historical prices for all Technology and Financial Services tickers.
        """
        tech     = self.get_tickers_by_sector("Technology")
        fintech  = self.get_tickers_by_sector("Financial Services")
        tickers  = tech + fintech
        return self.load_stock_prices(tickers)

    def load_reddit_latest(self, subreddit: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch the latest `limit` posts from a subreddit via PRAW.
        """
        records: List[Dict[str, Optional[object]]] = []
        for post in self.reddit.subreddit(subreddit).new(limit=limit):
            records.append({
                "id":           post.id,
                "title":        post.title,
                "selftext":     getattr(post, "selftext", None),
                "created_utc":  datetime.fromtimestamp(post.created_utc),
                "score":        post.score,
                "num_comments": post.num_comments,
                "url":          post.url,
            })
        return pd.DataFrame(records)

    def load_reddit_range(
        self,
        subreddit: str,
        after: str,
        before: str,
        max_posts: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch up to `max_posts` from a subreddit, filtered by UTC date range.
        `after` / `before` in 'YYYY-MM-DD' format.
        """
        after_ts  = int(pd.to_datetime(after).timestamp())
        before_ts = int(pd.to_datetime(before).timestamp())
        records: List[Dict[str, Optional[object]]] = []
        for post in self.reddit.subreddit(subreddit).new(limit=max_posts):
            if after_ts <= post.created_utc <= before_ts:
                records.append({
                    "id":           post.id,
                    "title":        post.title,
                    "selftext":     getattr(post, "selftext", None),
                    "created_utc":  datetime.fromtimestamp(post.created_utc),
                    "score":        post.score,
                    "num_comments": post.num_comments,
                    "url":          post.url,
                })
        return pd.DataFrame(records)


# Example usage (for testing)
if __name__ == "__main__":
    loader = DataLoader(start_date="2020-01-01", end_date="2025-05-01")

    # use smallest sample as possible to avoid rate limit
    tech_one   = loader.get_tickers_by_sector("Technology")[:1]
    fin_one    = loader.get_tickers_by_sector("Financial Services")[:1]
    tests = {
        "Equity":      (loader.load_stock_prices,       [['AAPL']]),
        "Benchmark":   (loader.load_benchmark_indices, [['^GSPC']]),
        "DeFi":        (loader.load_defi_prices,        [['uniswap']]),
        "Macro":       (loader.load_macro_series,       [['GDP','UNRATE']]),
        "Tech+FinTech":(loader.load_stock_prices,       [tech_one + fin_one]),
        "Reddit":      (loader.load_reddit_latest,      ['CryptoCurrency', 2]),
    }

    for name, (func, args) in tests.items():
        if name in {"Equity","Benchmark","Crypto","Tech+FinTech"}:
            time.sleep(1)
        df = func(*args)
        print(f"{name} data sample:\n{df.head()}\n")


