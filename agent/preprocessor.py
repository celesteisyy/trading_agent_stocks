import pandas as pd
import numpy as np
from scipy import stats

class Preprocessor:
    """
    Preprocessor for financial time series and tabular data.
    Performs:
      - Flattening multi-index columns
      - Ensuring datetime index
      - Sorting index
      - Resampling (optional)
      - Missing value handling
      - Outlier removal (z-score)
    """
    def __init__(self, resample_freq: str = None, missing_method: str = 'ffill', z_thresh: float = 3.0):
        """
        Args:
            resample_freq: frequency string for resampling (e.g., 'B', 'D', '1H'). If None, no resampling.
            missing_method: one of ['ffill', 'bfill', 'interpolate', 'drop']
            z_thresh: threshold for z-score based outlier removal
        """
        self.resample_freq = resample_freq
        self.missing_method = missing_method
        self.z_thresh = z_thresh

    def flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten MultiIndex columns (e.g. ('AAPL','Close') -> 'AAPL_Close')"""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [f"{lvl0}_{lvl1}" for lvl0, lvl1 in df.columns]
        return df

    def ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert index to DatetimeIndex and sort"""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    def resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample to specified frequency using last observation"""
        if self.resample_freq:
            df = df.resample(self.resample_freq).last()
        return df

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to chosen method"""
        if self.missing_method == 'ffill':
            return df.ffill()
        if self.missing_method == 'bfill':
            return df.bfill()
        if self.missing_method == 'interpolate':
            return df.interpolate(method='time')
        if self.missing_method == 'drop':
            return df.dropna()
        raise ValueError(f"Unknown missing_method: {self.missing_method}")

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where any numeric column has |z-score| >= z_thresh"""
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            return df
        z_scores = np.abs(stats.zscore(numeric, nan_policy='omit'))
        mask = (z_scores < self.z_thresh).all(axis=1)
        return df.loc[mask]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full preprocessing pipeline on DataFrame"""
        df = self.flatten_columns(df)
        df = self.ensure_datetime_index(df)
        df = self.resample(df)
        df = self.handle_missing(df)
        df = self.remove_outliers(df)
        return df


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader

    loader = DataLoader(start_date="2020-01-01", end_date="2025-01-01")
    df_stock = loader.load_stock_prices(["AAPL", "MSFT"])

    pre = Preprocessor(resample_freq='B', missing_method='ffill', z_thresh=3.0)
    df_clean = pre.fit_transform(df_stock)
    print(df_clean.head())
