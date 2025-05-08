import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from typing import List, Dict, Optional
import requests
import time
import yfinance as yf
import ccxt
import praw
from datetime import datetime

# Load environment variables from fin580.env
env_path = find_dotenv("fin580.env", raise_error_if_not_found=True)
load_dotenv(env_path)


class DataLoader:
    """
    DataLoader handles downloading of various financial time series:
      - Equity prices via yfinance (fallback to FMP, then local CSV)
      - Benchmark indices via yfinance (fallback to FRED, then local CSV)
      - DeFi tokens via CCXT
      - Macro series via the FRED API
      - Sector tickers via FinancialModelingPrep API
      - Reddit posts via PRAW
    """
    def __init__(self, start_date: str, end_date: str, interval: str = "1d"):
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
        1) Try yfinance
        2) Fallback to FMP API
        3) Fallback to local CSV in data/prices/{TICKER}.csv
        Returns a DataFrame with a DatetimeIndex and a MultiIndex on columns [Ticker, OHLCV].
        """
        # 1) yfinance
        try:
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
                return df
            raise ValueError("yfinance returned no data")
        except Exception as e:
            print(f"yfinance failed ({e}), falling back to FMP API...")

        # 2) FinancialModelingPrep
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
                          "close":"Close"
                      })
                      [["date","Open","High","Low","Close","volume"]]
                )
                df_t["date"] = pd.to_datetime(df_t["date"])
                df_t.set_index("date", inplace=True)
                # add ticker level
                df_t.columns = pd.MultiIndex.from_product([[ticker], df_t.columns])
                all_frames.append(df_t)
            except Exception as e2:
                print(f"FMP failed for {ticker} ({e2}), will try local CSV...")

        if all_frames:
            return pd.concat(all_frames, axis=1).sort_index()

        # 3) Local CSV fallback
        csv_frames = []
        for ticker in tickers:
            path = os.path.join("data", "prices", f"{ticker}.csv")
            try:
                df_csv = pd.read_csv(path, parse_dates=["date"], index_col="date")
                df_csv.columns = pd.MultiIndex.from_product([[ticker], df_csv.columns])
                csv_frames.append(df_csv)
            except Exception as e3:
                print(f"Local CSV load failed for {ticker} ({e3})")
        if csv_frames:
            return pd.concat(csv_frames, axis=1).sort_index()

        raise RuntimeError(
            "All data sources for `load_stock_prices` failed: yfinance, FMP, local CSV.")

    def load_benchmark_indices(self, symbols: List[str]) -> pd.DataFrame:
        """
        1) Try yfinance
        2) Fallback to FRED (close only)
        3) Fallback to local CSV in data/{SYMBOL}.csv
        Returns a DataFrame indexed by date with one column per symbol.
        """
        # 1) yfinance
        try:
            df = yf.download(
                symbols,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                group_by='ticker',
                auto_adjust=False,
                threads=True,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                # flatten to single-level columns if needed
                if isinstance(df.columns, pd.MultiIndex):
                    # extract the 'Close' price for each symbol
                    closes = {sym: df[sym]['Close'] for sym in symbols}
                    df_close = pd.DataFrame(closes)
                    df_close.index.name = "Date"
                    return df_close
                else:
                    # single symbol case
                    return df[['Close']].rename(columns={'Close': symbols[0]})
            raise ValueError("yfinance returned no data")
        except Exception as e:
            print(f"yfinance for indices failed ({e}), falling back to FRED...")

        # 2) FRED fallback (close only)
        frames: List[pd.DataFrame] = []
        for sym in symbols:
            series_id = sym.lstrip('^').upper()
            try:
                params = {
                    'series_id':       series_id,
                    'api_key':         self.fred_api_key,
                    'file_type':       'json',
                    'observation_start': self.start_date,
                    'observation_end':   self.end_date,
                }
                resp = requests.get(self._FRED_BASE_URL, params=params)
                resp.raise_for_status()
                obs = resp.json().get('observations', [])
                df_s = pd.DataFrame(obs)[['date','value']]
                df_s['date']  = pd.to_datetime(df_s['date'])
                df_s.set_index('date', inplace=True)
                df_s.rename(columns={'value': sym}, inplace=True)
                frames.append(df_s)
            except Exception as e2:
                print(f" FRED failed for {sym} ({e2}), trying local CSV...")

        if frames:
            return pd.concat(frames, axis=1)

        # 3) Local CSV fallback
        csv_frames = []
        for sym in symbols:
            fname = sym.lstrip('^') + ".csv"
            path = os.path.join("data", fname)
            try:
                df_csv = pd.read_csv(path, parse_dates=["date"], index_col="date")
                # we expect at least a 'close' column
                if "close" in df_csv.columns:
                    df_idx = df_csv[["close"]].rename(columns={'close': sym})
                elif "Close" in df_csv.columns:
                    df_idx = df_csv[["Close"]].rename(columns={'Close': sym})
                else:
                    raise KeyError("no close column")
                csv_frames.append(df_idx)
            except Exception as e3:
                print(f"Local CSV load failed for {sym} ({e3})")
        if csv_frames:
            return pd.concat(csv_frames, axis=1)

        raise RuntimeError("All data sources for `load_benchmark_indices` failed: yfinance, FRED, local CSV.")
    
    def load_defi_prices(self, coin_ids: List[str], vs_symbol: str = 'USDT', ccxt_exchanges: List[str] = ['kraken', 'coinbasepro', 'bitfinex'], timeframe: str = '1d') -> pd.DataFrame:
        """
        Fetch closing prices for DeFi tokens with fallbacks:

        1) CCXT exchanges (e.g. UNI/USDT or UNI/USD) :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}  
        2) FMP Crypto API (/historical-price-full/crypto/{symbol})  
        3) Local CSVs at ./data/defi/{coin}.csv  

        Returns:
            DataFrame indexed by date (or timestamp), one column per token.
        """
        # Try CCXT
        for exchange_id in ccxt_exchanges:
            try:
                exchange = getattr(ccxt, exchange_id)()
                since_ms = exchange.parse8601(f"{self.start_date}T00:00:00Z")
                result: Dict[str, pd.Series] = {}

                for coin in coin_ids:
                    for quote in (vs_symbol.upper(), 'USD'):
                        market = f"{coin.upper()}/{quote}"
                        try:
                            ohlcv = exchange.fetch_ohlcv(market, timeframe, since_ms)
                            df = pd.DataFrame(
                                ohlcv,
                                columns=['ts', 'open', 'high', 'low', 'close', 'volume']
                            )
                            df['date'] = pd.to_datetime(df['ts'], unit='ms')
                            df.set_index('date', inplace=True)
                            result[coin] = df['close']
                            break
                        except Exception:
                            continue

                if result:
                    return pd.DataFrame(result)

            except Exception:
                continue

        print(" CCXT path yielded no data, falling back to FMP Crypto API...")

        # Try FinancialModelingPrep Crypto endpoint
        all_frames = []
        for coin in coin_ids:
            url = (
                f"{self._FMP_BASE_URL}/historical-price-full/crypto/{coin.upper()}-{vs_symbol.upper()}"
                f"?from={self.start_date}&to={self.end_date}&apikey={self.fmp_api_key}"
            )
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                hist = r.json().get("historical", [])
                if not hist:
                    continue
                df_c = (
                    pd.DataFrame(hist)
                      .rename(columns={'close': coin})
                      [['date', coin]]
                )
                df_c['date'] = pd.to_datetime(df_c['date'])
                df_c.set_index('date', inplace=True)
                all_frames.append(df_c)
            except Exception:
                continue

        if all_frames:
            return pd.concat(all_frames, axis=1).sort_index()

        print("FMP Crypto API yielded no data, falling back to local CSV...")

        # Local CSV fallback
        csv_frames = []
        for coin in coin_ids:
            path = os.path.join("data", "defi", f"{coin}.csv")
            try:
                df_csv = pd.read_csv(path, parse_dates=["date"], index_col="date")
                # assume CSV has a 'close' column
                if 'close' in df_csv.columns:
                    df_csv = df_csv[['close']].rename(columns={'close': coin})
                else:
                    raise KeyError("no 'close' column")
                csv_frames.append(df_csv)
            except Exception as e:
                print(f"Local CSV load failed for {coin} ({e})")

        if csv_frames:
            return pd.concat(csv_frames, axis=1).sort_index()

        # NOTHING WORKED
        raise RuntimeError("All data sources for `load_defi_prices` failed: CCXT, FMP Crypto API, local CSV.")
    
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

    def load_reddit_range(self,subreddit: str,after: str,before: str,max_posts: int = 1000) -> pd.DataFrame:
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
