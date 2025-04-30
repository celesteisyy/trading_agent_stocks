import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import requests
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

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
        # Load FRED API key from environment
        self.fred_api_key = os.getenv('FRED_API_KEY')
        if not self.fred_api_key:
            raise ValueError(
                "FRED_API_KEY must be set in fin580.env for macro data retrieval."
            )
        # Load Gecko Crypto API key
        self.gecko_api_key = os.getenv('GECKO_API_KEY')
        if not self.gecko_api_key:
            raise ValueError(
                "GECKO_API_KEY must be set in fin580.env for crypto data retrieval."
            )
        # Base URLs as class constants
        self._COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3/coins"
        self._FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def load_stock_prices(self, tickers: List[str]) -> pd.DataFrame:
        """
        Download historical adjusted prices for equity tickers.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            MultiIndex DataFrame: Outer='Ticker', Inner=['Open','High','Low','Close','Volume']
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
        
        Args:
            symbols: List of index symbols
            
        Returns:
            DataFrame with columns ['Open','High','Low','Close','Volume'] for each symbol
        """
        return yf.download(
            symbols,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            auto_adjust=True,
            threads=True,
        )

    def load_crypto_prices(self, symbols: List[str]) -> pd.DataFrame:
        """
        Download major cryptocurrency prices (e.g., BTC-USD, ETH-USD).
        
        Args:
            symbols: List of crypto symbols without '-USD'
            
        Returns:
            Same structure as load_stock_prices
        """
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
        """
        Fetch historical price data for DeFi tokens via the CoinGecko API.
        
        Args:
            coin_ids: List of CoinGecko identifiers (e.g., 'uniswap', 'aave')
            vs_currency: Currency to get prices in (default: 'usd')
            
        Returns:
            DataFrame indexed by timestamp with one column per token
        """
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
        """
        Fetch macroeconomic time series from the FRED API.
        Requires FRED_API_KEY set in fin580.env.
        
        Args:
            series_ids: List of FRED series identifiers
            
        Returns:
            DataFrame indexed by date with one column per series
        """
        frames = []
        for series in series_ids:
            params = {
                'series_id': series,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'observation_start': self.start_date,
                'observation_end': self.end_date,
            }
            
            resp = requests.get(self._FRED_BASE_URL, params=params)
            resp.raise_for_status()
            obs = resp.json().get('observations', [])
            
            df = pd.DataFrame(obs)[['date', 'value']]
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df.set_index('date', inplace=True)
            df.rename(columns={'value': series}, inplace=True)
            frames.append(df)
            
        return pd.concat(frames, axis=1) if frames else pd.DataFrame()


# Example usage (for testing)
if __name__ == "__main__":
    loader = DataLoader(start_date="2020-01-01", end_date="2025-01-01")
    
    # Test equity data loading
    equities = ['AAPL', 'MSFT', 'GOOGL']
    equity_data = loader.load_stock_prices(equities)
    
    # Test index data loading
    benchmarks = ['^GSPC', '^RUI', '^RUT']
    benchmark_data = loader.load_benchmark_indices(benchmarks)
    
    # Test cryptocurrency data loading
    cryptos = ['BTC', 'ETH', 'XRP']
    crypto_data = loader.load_crypto_prices(cryptos)
    
    print(f"Data loading test completed successfully")