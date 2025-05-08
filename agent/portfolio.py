import pandas as pd
from pandas import date_range
import numpy as np


class PortfolioManagerAgent:
    """
    PortfolioManagerAgent for position sizing, trade execution, and performance metrics.

    Methods:
      - backtest: simulate equity curve given orders and prices
      - compute_performance: total return, annualized Sharpe ratio, max drawdown
    """
    def __init__(
        self,
        initial_capital: float = 100000.0,
        freq: str = 'B'
    ):
        """
        Args:
            initial_capital: starting portfolio value
            freq: data frequency ('B' for business days) for annualization
        """
        self.initial_capital = initial_capital
        self.freq = freq
        # approximate trading days per year
        self.annual_factor = 252 if freq == 'B' else 365

    def backtest(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate portfolio given orders.

        Args:
            orders_df: DataFrame with index dates and columns ['order', 'price']
                       where order=1 buys (enter), -1 sells (exit).

        Returns:
            portfolio_df: DataFrame with columns ['price', 'order', 'position', 'returns', 'equity']
        """
        df = orders_df.copy().sort_index()
        # daily price returns
        df['price_return'] = df['price'].pct_change().fillna(0)
        # build position series: 1 after buy, 0 after sell
        # map orders: buy->1, sell->0, no order->NaN
        pos = df['order'].replace({1: 1, -1: 0})
        df['position'] = pos.ffill().fillna(0)
        # strategy returns
        df['returns'] = df['position'].shift(1).fillna(0) * df['price_return']
        # equity curve
        df['equity'] = (1 + df['returns']).cumprod() * self.initial_capital
        return df

    def compute_performance(self, portfolio_df: pd.DataFrame) -> dict:
        """
        Compute performance metrics.

        Args:
            portfolio_df: output of backtest()

        Returns:
            metrics: dict with 'total_return', 'sharpe_ratio', 'max_drawdown'
        """
        # total return
        equity = portfolio_df['equity']
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        # daily strategy returns
        rets = portfolio_df['returns']
        # annualized Sharpe ratio (assume risk-free ~0)
        if rets.std() != 0:
            sharpe = (rets.mean() / rets.std()) * np.sqrt(self.annual_factor)
        else:
            sharpe = np.nan
        # max drawdown
        rolling_max = equity.cummax()
        drawdown = equity / rolling_max - 1
        max_drawdown = drawdown.min()
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }

if __name__ == '__main__':
    # Self-check / example usage

    # Generate synthetic price series
    dates = date_range('2025-01-01', periods=100, freq='B')
    price = 100 + np.cumsum(np.random.randn(len(dates)))
    price_df = pd.DataFrame({'price': price}, index=dates)

    # Synthetic orders: buy on day 10, sell on day 40, buy on day 60
    orders = pd.DataFrame(index=dates)
    orders['order'] = 0
    orders.loc[dates[9], 'order'] = 1
    orders.loc[dates[39], 'order'] = -1
    orders.loc[dates[59], 'order'] = 1
    orders['price'] = price_df['price']

    # Run backtest
    pm = PortfolioManagerAgent(initial_capital=100000)
    port = pm.backtest(orders)
    metrics = pm.compute_performance(port)

    # Print sample and metrics
    print("Equity head:", port[['equity']].dropna().head())
    print("Performance:", metrics)
