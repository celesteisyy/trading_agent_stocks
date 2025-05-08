import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import io
from docx import Document
from docx.shared import Inches
import gradio as gr
from typing import Dict, List, Tuple, Optional, Union

class PortfolioManager:
    """
    PortfolioManager handles:
    1. Portfolio simulation based on trading signals
    2. Performance metrics calculation
    3. Report generation
    4. Dashboard visualization
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        freq: str = 'B',
        transaction_cost: float = 0.001,
        output_dir: str = 'output'
    ):
        """
        Initialize portfolio manager.
        
        Args:
            initial_capital: Starting portfolio value
            freq: Data frequency ('B' for business days, 'D' for calendar days)
            transaction_cost: Cost per trade as fraction of trade value
            output_dir: Directory for saving reports
        """
        self.logger = logging.getLogger(__name__)
        self.initial_capital = initial_capital
        self.freq = freq
        self.transaction_cost = transaction_cost
        self.output_dir = output_dir
        
        # Set annualization factor based on frequency
        self.annual_factor = 252 if freq == 'B' else 365
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def backtest(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate portfolio performance from trading signals.
        
        Args:
            orders_df: DataFrame with index dates and columns ['order', 'price']
                       where order=1 buys (enter), -1 sells (exit).
        
        Returns:
            DataFrame with portfolio performance metrics
        """
        self.logger.info("Running portfolio backtest...")
        df = orders_df.copy().sort_index()
        
        # Ensure we have required columns
        if 'order' not in df.columns or 'price' not in df.columns:
            raise ValueError("orders_df must have 'order' and 'price' columns")
        
        # Calculate daily price returns
        df['price_return'] = df['price'].pct_change().fillna(0)
        
        # Build position series: 1 after buy, 0 after sell
        # Map orders: buy->1, sell->0, no order->NaN
        pos = df['order'].replace({1: 1, -1: 0})
        df['position'] = pos.ffill().fillna(0)
        
        # Number of trades (changes in position)
        df['trade'] = df['position'].diff().fillna(0).abs()
        df['trade_cost'] = df['trade'] * df['price'] * self.transaction_cost
        
        # Strategy returns
        df['returns'] = df['position'].shift(1).fillna(0) * df['price_return'] - df['trade_cost'] / self.initial_capital
        
        # Equity curve
        df['equity'] = (1 + df['returns']).cumprod() * self.initial_capital
        
        # Drawdown calculation
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] / df['peak'] - 1) * 100  # as percentage
        
        self.logger.info(f"Backtest complete. Final equity: ${df['equity'].iloc[-1]:.2f}")
        return df
    
    def compute_performance(self, portfolio_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics from portfolio results.
        
        Args:
            portfolio_df: Output from backtest() method
            
        Returns:
            Dictionary of performance metrics
        """
        # Extract required data
        equity = portfolio_df['equity']
        returns = portfolio_df['returns']
        
        # Total return
        total_return = equity.iloc[-1] / self.initial_capital - 1
        
        # Annualized return
        n_periods = len(returns)
        years = n_periods / self.annual_factor
        ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(self.annual_factor)
        
        # Sharpe ratio (assume risk-free ~0)
        sharpe = ann_return / volatility if volatility != 0 else 0
        
        # Maximum drawdown
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Calmar ratio
        calmar = ann_return / abs(max_drawdown / 100) if max_drawdown != 0 else 0
        
        # Win rate
        daily_wins = (returns > 0).sum()
        win_rate = daily_wins / len(returns) if len(returns) > 0 else 0
        
        # Number of trades
        n_trades = portfolio_df['trade'].sum() / 2  # divide by 2 since each round-trip is 2 position changes
        
        # Average trade return
        trade_returns = []
        in_trade = False
        entry_price = 0
        
        for i, row in portfolio_df.iterrows():
            if row['trade'] == 1 and not in_trade:  # Enter trade
                entry_price = row['price']
                in_trade = True
            elif row['trade'] == 1 and in_trade:  # Exit trade
                if entry_price > 0:
                    trade_return = (row['price'] / entry_price) - 1
                    trade_returns.append(trade_return)
                in_trade = False
        
        avg_trade = np.mean(trade_returns) if trade_returns else 0
        
        # Return all metrics
        metrics = {
            'total_return': total_return,
            'ann_return': ann_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown / 100,  # Convert back to decimal
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'n_trades': n_trades,
            'avg_trade': avg_trade
        }
        
        self.logger.info(f"Performance metrics calculated: Total Return: {total_return:.2%}, Sharpe: {sharpe:.2f}")
        return metrics
    
    def generate_report(
        self,
        portfolio_df: pd.DataFrame,
        metrics: Dict[str, float],
        ticker: str = "Stock",
        report_path: Optional[str] = None
    ) -> str:
        """
        Generate a DOCX report with portfolio performance.
        
        Args:
            portfolio_df: Output from backtest() method
            metrics: Performance metrics dictionary
            ticker: Stock ticker or name for report title
            report_path: Optional custom path for the report
            
        Returns:
            Path to the generated report
        """
        if report_path is None:
            report_path = os.path.join(self.output_dir, 'trading_report.docx')
        
        doc = Document()
        doc.add_heading(f'Trading System Report: {ticker}', level=1)
        
        # Add executive summary
        doc.add_heading('Executive Summary', level=2)
        doc.add_paragraph(f"This report presents the performance of a trading strategy applied to {ticker} "
                         f"using a combination of technical analysis, sentiment analysis, and predictive models.")
                         
        summary_text = (
            f"The strategy achieved a total return of {metrics['total_return']:.2%} "
            f"({metrics['ann_return']:.2%} annualized) with a Sharpe ratio of {metrics['sharpe_ratio']:.2f}. "
            f"Maximum drawdown was {metrics['max_drawdown']:.2%}. "
            f"The strategy executed {metrics['n_trades']:.0f} trades with a win rate of {metrics['win_rate']:.2%}."
        )
        doc.add_paragraph(summary_text)
        
        # Performance Metrics Table
        doc.add_heading('Performance Metrics', level=2)
        table = doc.add_table(rows=1, cols=2)
        table.style = 'Table Grid'
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Metric'
        hdr_cells[1].text = 'Value'
        
        metrics_to_show = [
            ('Total Return', f"{metrics['total_return']:.2%}"),
            ('Annualized Return', f"{metrics['ann_return']:.2%}"),
            ('Volatility', f"{metrics['volatility']:.2%}"),
            ('Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}"),
            ('Maximum Drawdown', f"{metrics['max_drawdown']:.2%}"),
            ('Calmar Ratio', f"{metrics['calmar_ratio']:.2f}"),
            ('Win Rate', f"{metrics['win_rate']:.2%}"),
            ('Number of Trades', f"{metrics['n_trades']:.0f}"),
            ('Average Trade Return', f"{metrics['avg_trade']:.2%}")
        ]
        
        for metric, value in metrics_to_show:
            row_cells = table.add_row().cells
            row_cells[0].text = metric
            row_cells[1].text = value
        
        # Equity Curve Plot
        doc.add_heading('Equity Curve', level=2)
        img_stream = io.BytesIO()
        plt.figure(figsize=(8, 4))
        plt.plot(portfolio_df.index, portfolio_df['equity'])
        plt.title('Portfolio Equity Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(img_stream, format='png')
        plt.close()
        img_stream.seek(0)
        doc.add_picture(img_stream, width=Inches(6))
        
        # Drawdown Plot
        doc.add_heading('Drawdown', level=2)
        img_stream = io.BytesIO()
        plt.figure(figsize=(8, 4))
        plt.fill_between(portfolio_df.index, portfolio_df['drawdown'], 0, color='red', alpha=0.3)
        plt.plot(portfolio_df.index, portfolio_df['drawdown'], color='red')
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(img_stream, format='png')
        plt.close()
        img_stream.seek(0)
        doc.add_picture(img_stream, width=Inches(6))
        
        # Save the document
        doc.save(report_path)
        self.logger.info(f"Report saved to {report_path}")
        
        return report_path
    
    def create_dashboard(
        self,
        results: Dict[str, pd.DataFrame],
        config: Dict[str, any]
    ) -> gr.Blocks:
        """
        Create an interactive dashboard for visualizing results.
        
        Args:
            results: Dictionary with 'portfolio', 'orders', 'metrics', 'sentiment' data
            config: Configuration parameters
            
        Returns:
            Gradio interface
        """
        # Extract data from results
        portfolio_df = results['portfolio']
        metrics = results['metrics']
        orders_df = results['orders']
        sentiment_df = results.get('sentiment', pd.DataFrame())
        
        # Dashboard components
        def plot_equity_curve():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(portfolio_df.index, portfolio_df['equity'])
            ax.set_title('Equity Curve')
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value ($)')
            ax.grid(True)
            return fig
        
        def plot_drawdown():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.fill_between(portfolio_df.index, portfolio_df['drawdown'], 0, color='red', alpha=0.3)
            ax.plot(portfolio_df.index, portfolio_df['drawdown'], color='red')
            ax.set_title('Portfolio Drawdown')
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.grid(True)
            return fig
        
        def plot_returns_histogram():
            fig, ax = plt.subplots(figsize=(10, 6))
            portfolio_df['returns'].hist(bins=30, ax=ax)
            ax.set_title('Returns Distribution')
            ax.set_xlabel('Daily Return')
            ax.set_ylabel('Frequency')
            ax.grid(True)
            return fig
        
        def plot_trades():
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot price
            ax.plot(orders_df.index, orders_df['price'], color='gray', alpha=0.7)
            
            # Plot buy points
            buys = orders_df[orders_df['order'] > 0]
            ax.scatter(buys.index, buys['price'], marker='^', color='green', s=100, label='Buy')
            
            # Plot sell points
            sells = orders_df[orders_df['order'] < 0]
            ax.scatter(sells.index, sells['price'], marker='v', color='red', s=100, label='Sell')
            
            ax.set_title('Trade Signals')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True)
            return fig
        
        def plot_sentiment_vs_price():
            # Check if we have sentiment and price data
            if sentiment_df.empty or 'avg_compound' not in sentiment_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No sentiment data available", 
                       horizontalalignment='center', verticalalignment='center')
                return fig
            
            # Create figure with two Y axes
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Plot price on primary Y-axis
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price', color='tab:blue')
            ax1.plot(orders_df.index, orders_df['price'], color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            
            # Create secondary Y-axis for sentiment
            ax2 = ax1.twinx()
            ax2.set_ylabel('Sentiment Score', color='tab:red')
            
            # Plot sentiment, handling date alignment
            if isinstance(sentiment_df.index, pd.DatetimeIndex):
                dates = sentiment_df.index
            else:
                dates = pd.to_datetime(sentiment_df['date'])
            
            ax2.plot(dates, sentiment_df['avg_compound'], color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            
            # Add buy/sell points
            buys = orders_df[orders_df['order'] > 0]
            ax1.scatter(buys.index, buys['price'], marker='^', color='green', s=100, label='Buy')
            
            sells = orders_df[orders_df['order'] < 0]
            ax1.scatter(sells.index, sells['price'], marker='v', color='red', s=100, label='Sell')
            
            ax1.legend(loc='upper left')
            fig.tight_layout()
            fig.suptitle('Price vs. Reddit Sentiment', y=1.02)
            
            return fig
        
        def get_metrics_table():
            metrics_to_show = [
                ('Total Return', f"{metrics['total_return'] * 100:.2f}%"),
                ('Annualized Return', f"{metrics.get('ann_return', 0) * 100:.2f}%"),
                ('Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}"),
                ('Max Drawdown', f"{metrics['max_drawdown'] * 100:.2f}%"),
                ('Win Rate', f"{metrics.get('win_rate', 0) * 100:.2f}%"),
                ('Number of Trades', f"{metrics.get('n_trades', 0):.0f}")
            ]
            metrics_df = pd.DataFrame(metrics_to_show, columns=['Metric', 'Value'])
            return metrics_df
        
        def get_sentiment_stats():
            if sentiment_df.empty or 'avg_compound' not in sentiment_df.columns:
                return pd.DataFrame({
                    'Metric': ['No sentiment data available'],
                    'Value': ['']
                })
            
            sentiment_series = sentiment_df['avg_compound']
            stats = pd.DataFrame({
                'Metric': [
                    'Mean Sentiment',
                    'Median Sentiment',
                    'Max Sentiment',
                    'Min Sentiment',
                    'Sentiment Volatility'
                ],
                'Value': [
                    f"{sentiment_series.mean():.4f}",
                    f"{sentiment_series.median():.4f}",
                    f"{sentiment_series.max():.4f}",
                    f"{sentiment_series.min():.4f}",
                    f"{sentiment_series.std():.4f}"
                ]
            })
            return stats
        
        # Create Gradio interface
        title = "Crypto-Sentiment Trading Dashboard"
        tickers = config.get('tickers', ['Stock'])
        ticker_str = ', '.join(tickers)
        
        with gr.Blocks(title=title) as dashboard:
            gr.Markdown(f"# {title}\n### Technology Stock(s): {ticker_str}")
            
            with gr.Row():
                with gr.Column():
                    gr.Dataframe(get_metrics_table(), label="Performance Metrics")
                
                with gr.Column():
                    gr.Markdown(f"""
                    ## Trading Parameters
                    - Start Date: {config.get('start_date', 'N/A')}
                    - End Date: {config.get('end_date', 'N/A')}
                    - Initial Capital: ${config.get('initial_capital', 100000.0)}
                    - Min Holding Period: {config.get('min_holding', 5)} days
                    - Max Trades per Week: {config.get('max_trades', 3)}
                    - Sentiment Method: {config.get('sentiment_method', 'transformer')}
                    - Model Type: {config.get('model_type', 'gru')}
                    - Reddit Subreddit: {config.get('reddit_subreddit', 'CryptoCurrency')}
                    """)
            
            gr.Markdown("## Equity Curve")
            gr.Plot(plot_equity_curve)
            
            gr.Markdown("## Drawdown")
            gr.Plot(plot_drawdown)
            
            # Sentiment and Price Correlation
            gr.Markdown("## Reddit Sentiment vs. Stock Price")
            gr.Plot(plot_sentiment_vs_price)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Sentiment Statistics")
                    gr.Dataframe(get_sentiment_stats())
                
                with gr.Column():
                    gr.Markdown("## Returns Distribution")
                    gr.Plot(plot_returns_histogram)
            
            gr.Markdown("## Trade Signals")
            gr.Plot(plot_trades)
            
            gr.Markdown("## Recent Trades")
            trades = orders_df[orders_df['order'] != 0].tail(10)
            trades_display = trades.reset_index()
            
            # Adapt column names based on what's available
            display_cols = ['index', 'order', 'price']
            new_names = ['Date', 'Order', 'Price']
            
            # Add sentiment column if available
            if 'sentiment' in trades_display.columns:
                display_cols.append('sentiment')
                new_names.append('Sentiment')
                
            trades_display = trades_display[display_cols]
            trades_display.columns = new_names
            
            # Format date and add trade type
            trades_display['Date'] = pd.to_datetime(trades_display['Date']).dt.strftime('%Y-%m-%d')
            trades_display['Type'] = trades_display['Order'].apply(lambda x: 'Buy' if x > 0 else 'Sell')
            
            # Show dataframe
            final_cols = ['Date', 'Type', 'Price']
            if 'Sentiment' in trades_display.columns:
                final_cols.append('Sentiment')
                
            gr.Dataframe(trades_display[final_cols])
            
        return dashboard


if __name__ == '__main__':
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Test portfolio manager with sample data
    from pandas import date_range
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing
    
    # Create sample data
    dates = date_range('2023-01-01', periods=252, freq='B')
    
    # Generate price series (random walk)
    price = 100 + np.cumsum(np.random.randn(len(dates)))
    
    # Generate orders at random intervals
    orders = np.zeros(len(dates))
    for i in range(10, len(dates), 30):  # Every ~30 days
        orders[i] = 1  # Buy
        if i + 15 < len(dates):
            orders[i + 15] = -1  # Sell 15 days later
    
    orders_df = pd.DataFrame({'price': price, 'order': orders}, index=dates)
    
    # Create a sentiment series
    sentiment = np.random.normal(0.2, 0.3, size=len(dates))
    for i in range(1, len(sentiment)):
        sentiment[i] = 0.8 * sentiment[i-1] + 0.2 * sentiment[i]  # Add some autocorrelation
    
    sentiment_df = pd.DataFrame({'avg_compound': sentiment}, index=dates)
    
    # Create portfolio manager
    portfolio_manager = PortfolioManager(initial_capital=100000.0)
    
    # Backtest portfolio
    portfolio_df = portfolio_manager.backtest(orders_df)
    
    # Calculate performance metrics
    metrics = portfolio_manager.compute_performance(portfolio_df)
    print("Performance Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    
    # Generate report
    report_path = portfolio_manager.generate_report(
        portfolio_df=portfolio_df,
        metrics=metrics,
        ticker="SAMPLE"
    )
    print(f"Report saved to: {report_path}")
    
    # Test dashboard creation
    results = {
        'portfolio': portfolio_df,
        'orders': orders_df,
        'metrics': metrics,
        'sentiment': sentiment_df
    }
    
    config = {
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'tickers': ['SAMPLE'],
        'initial_capital': 100000.0
    }
    
    dashboard = portfolio_manager.create_dashboard(results, config)
    print("Dashboard created successfully")
    # Note: In a real scenario, you would call dashboard.launch() here