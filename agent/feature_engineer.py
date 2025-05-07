import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from preprocessor import Preprocessor
from data_loader import DataLoader
# Optionally import sentiment analysis module
# from sentiment_analysis import analyze_sentiment


class FeatureEngineer:
    """
    FeatureEngineer constructs features from cleaned financial data and Reddit posts.

    Methods:
      - fit_transform: run full pipeline
      - add_return_features: computes pct_change and log_returns
      - add_rolling_features: computes SMA, EMA, rolling volatility
      - add_technical_indicators: RSI, MACD, momentum
      - add_macro_growth: computes growth rates for macro series
      - add_sentiment_features (optional): sentiment from Reddit text
    """
    def __init__(
        self,
        preprocessor: Optional[Preprocessor] = None,
        windows: Dict[str, int] = None
    ):
        self.preprocessor = preprocessor or Preprocessor()
        # default window sizes in days for rolling features
        self.windows = windows or {"short": 5, "medium": 20, "long": 60}

    def fit_transform(
        self,
        df_price: pd.DataFrame,
        df_macro: pd.DataFrame = None,
        df_reddit: pd.DataFrame = None
    ) -> pd.DataFrame:
        # Preprocess price data
        df = self.preprocessor.fit_transform(df_price)

        # Price-based features
        df = self._add_return_features(df)
        df = self._add_rolling_features(df)
        df = self._add_technical_indicators(df)

        # Macro features
        if df_macro is not None:
            df_macro = self.preprocessor.fit_transform(df_macro)
            df = df.join(self._add_macro_growth(df_macro))

        # Sentiment features
        # if df_reddit is not None:
        #     sentiments = analyze_sentiment(df_reddit["title"] + " " + df_reddit["selftext"])
        #     df = df.join(sentiments)

        return df

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # assumes a column named 'Close'
        df["return"] = df["Close"].pct_change()
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for name, window in self.windows.items():
            df[f"sma_{window}"] = df["Close"].rolling(window).mean()
            df[f"ema_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
            df[f"volatility_{window}"] = df["return"].rolling(window).std()
        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # RSI
        delta = df["Close"].diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        roll_up = up.ewm(com=13, adjust=False).mean()
        roll_down = down.ewm(com=13, adjust=False).mean()
        rs = roll_up / roll_down
        df["rsi_14"] = 100 - (100 / (1 + rs))
        # MACD
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        return df

    def _add_macro_growth(self, df_macro: pd.DataFrame) -> pd.DataFrame:
        # computes period-over-period growth rates
        growth = df_macro.pct_change().add_suffix("_growth")
        return growth

if __name__ == "__main__":

    loader = DataLoader(start_date="2020-01-01",
                        end_date="2025-05-01",
                        interval="1d") 

    df_price = loader.load_stock_prices(["AAPL", "MSFT"])  
    df_macro = loader.load_macro_series(["GDP", "UNRATE"])  
    df_reddit = loader.load_reddit_latest("CryptoCurrency", limit=5)  

    fe = FeatureEngineer()
    df_features = fe.fit_transform(df_price, df_macro, df_reddit)

    print(df_features.head())
