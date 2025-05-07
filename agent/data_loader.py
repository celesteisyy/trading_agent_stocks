import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import requests
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import praw
from datetime import datetime

# Load environment variables from fin580.env
load_dotenv('fin580.env')

class DataLoader:
    """
    DataLoader handles downloading of various financial time series:
      - Equity prices via yfinance
      - Benchmark indices via yfinance
      - Cryptocurrency prices via yfinance
      - DeFi token prices via CoinGecko API
      - Macroeconomic series via the FRED API
      - Reddit posts via PRAW
    """

    def __init__(self, start_date: str, end_date: str, interval: str = "1d"):
        """
        Initialize the DataLoader with time range parameters.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1d', '1wk', '1mo', etc.)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

        # FRED
        self.fred_api_key = os.getenv('FRED_API_KEY')
        if not self.fred_api_key:
            raise ValueError("FRED_API_KEY must be set in fin580.env")
        self._FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

        # CoinGecko
        self.gecko_api_key = os.getenv('GECKO_API_KEY')
        if not self.gecko_api_key:
            raise ValueError("GECKO_API_KEY must be set in fin580.env")
        self._COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3/coins"

        # Reddit (PRAW)
        reddit_id     = os.getenv("REDDIT_CLIENT_ID")
        reddit_sec    = os.getenv("REDDIT_CLIENT_SECRET")
        reddit_ua     = os.getenv("REDDIT_USER_AGENT")
        if not all([reddit_id, reddit_sec, reddit_ua]):
            raise ValueError("REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT must be set in fin580.env")
        self.reddit = praw.Reddit(
            client_id=reddit_id,
            client_secret=reddit_sec,
            user_agent=reddit_ua
        )

    def load_stock_prices(self, tickers: List[str]) -> pd.DataFrame:
        # ... unchanged ...
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
        # ... unchanged ...
        return yf.download(
            symbols,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            auto_adjust=True,
            threads=True,
        )

    def load_crypto_prices(self, symbols: List[str]) -> pd.DataFrame:
        # ... unchanged ...
        tickers = [f"{sym}-USD" for sym in symbols]
        return yf.download(
            tickers,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            group_by='ticker',
            auto_adjust=True,
            threads=True,
        )

    def load_defi_prices(self, coin_ids: List[str], vs_currency: str = 'usd') -> pd.DataFrame:
        # ... unchanged ...
        results = {}
        for coin in coin_ids:
            url = f"{self._COINGECKO_BASE_URL}/{coin}/market_chart"
            params = {'vs_currency': vs_currency, 'days': 'max'}
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            prices = resp.json().get('prices', [])
            df = pd.DataFrame(prices, columns=['timestamp', coin])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            results[coin] = df[coin]
        return pd.DataFrame(results)

    def load_macro_series(self, series_ids: List[str]) -> pd.DataFrame:
        # ... unchanged ...
        frames = []
        for series in series_ids:
            params = {
                'series_id': series,
                'api_key':   self.fred_api_key,
                'file_type': 'json',
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

    def load_reddit_latest(self, subreddit: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch the latest `limit` posts from a subreddit.
        """
        records = []
        for post in self.reddit.subreddit(subreddit).new(limit=limit):
            records.append({
                "id":           post.id,
                "title":        post.title,
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
        records = []
        for post in self.reddit.subreddit(subreddit).new(limit=max_posts):
            if after_ts <= post.created_utc <= before_ts:
                records.append({
                    "id":           post.id,
                    "title":        post.title,
                    "created_utc":  datetime.fromtimestamp(post.created_utc),
                    "score":        post.score,
                    "num_comments": post.num_comments,
                    "url":          post.url,
                })
        return pd.DataFrame(records)


# Example usage (for testing)
if __name__ == "__main__":
    loader = DataLoader(start_date="2020-01-01", end_date="2025-01-01")
    
    # test reddit
    df_latest = loader.load_reddit_latest("CryptoCurrency", limit=5)
    print("Latest posts:\n", df_latest.head())

    df_range = loader.load_reddit_range("CryptoCurrency", after="2025-04-30", before="2025-05-07", max_posts=200)
    print("Range posts:\n", df_range.head())
