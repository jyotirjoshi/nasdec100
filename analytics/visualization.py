"""
Visualization tools for trading results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os


class Visualizer:
    def __init__(self, portfolio_data=None, metrics=None):
        """
        Initialize Visualizer

        Args:
            portfolio_data (pandas.DataFrame): Portfolio data
            metrics (dict): Performance metrics
        """
        self.portfolio_data = portfolio_data
        self.metrics = metrics
        self.logger = logging.getLogger(__name__)

    def set_data(self, portfolio_data, metrics=None):
        """
        Set data for visualization

        Args:
            portfolio_data (pandas.DataFrame): Portfolio data
            metrics (dict): Performance metrics
        """
        self.portfolio_data = portfolio_data
        if metrics:
            self.metrics = metrics

    def plot_equity_curve(self, benchmark=None, figsize=(12, 6)):
        """
        Plot equity curve

        Args:
            benchmark (pandas.Series): Benchmark data for comparison
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        if self.portfolio_data is None or self.portfolio_data.empty:
            self.logger.warning("No portfolio data to plot")
            return None

        if 'total' not in self.portfolio_data.columns:
            self.logger.warning("Portfolio data missing 'total' column")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        # Plot equity curve
        ax.plot(self.portfolio_data.index, self.portfolio_data['total'],
                label='Portfolio', linewidth=2)

        # Add benchmark if provided
        if benchmark is not None:
            if len(benchmark) != len(self.portfolio_data):
                self.logger.warning("Benchmark data length doesn't match portfolio data")
            else:
                ax.plot(self.portfolio_data.index, benchmark,
                        label='Benchmark', linewidth=1, alpha=0.7, linestyle='--')

        # Format dates on x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Add metrics if available
        if self.metrics:
            metrics_text = f"Total Return: {self.metrics.get('total_return', 0):.2%}\n"
            metrics_text += f"Sharpe: {self.metrics.get('annualized_sharpe', 0):.2f}\n"
            metrics_text += f"Max DD: {self.metrics.get('max_drawdown', 0):.2%}"

            # Add text box
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)

        # Set labels and title
        ax.set_title('Portfolio Equity Curve')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_drawdowns(self, figsize=(12, 6)):
        """
        Plot drawdowns

        Args:
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        if self.portfolio_data is None or self.portfolio_data.empty:
            self.logger.warning("No portfolio data to plot")
            return None

        if 'total' not in self.portfolio_data.columns:
            self.logger.warning("Portfolio data missing 'total' column")
            return None

        # Calculate drawdowns
        equity = self.portfolio_data['total']
        peak = equity.cummax()
        drawdown = (equity - peak) / peak

        fig, ax = plt.subplots(figsize=figsize)

        # Plot drawdowns
        ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)

        # Format dates on x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Set labels and title
        ax.set_title('Portfolio Drawdowns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)

        # Add max drawdown line
        max_dd = drawdown.min()
        ax.axhline(y=max_dd, color='r', linestyle='--', alpha=0.7,
                   label=f'Max Drawdown: {max_dd:.2%}')
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_returns_distribution(self, figsize=(12, 6)):
        """
        Plot returns distribution

        Args:
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        if self.portfolio_data is None or self.portfolio_data.empty:
            self.logger.warning("No portfolio data to plot")
            return None

        if 'total' not in self.portfolio_data.columns:
            self.logger.warning("Portfolio data missing 'total' column")
            return None

        # Calculate returns
        returns = self.portfolio_data['total'].pct_change().dropna()

        fig, ax = plt.subplots(figsize=figsize)

        # Plot returns distribution
        sns.histplot(returns, kde=True, ax=ax)

        # Add vertical line at mean
        mean_return = returns.mean()
        ax.axvline(mean_return, color='r', linestyle='--',
                   label=f'Mean: {mean_return:.4%}')

        # Add normal distribution for comparison
        x = np.linspace(returns.min(), returns.max(), 100)
        from scipy.stats import norm
        params = norm.fit(returns)
        dist = norm(*params)
        ax.plot(x, dist.pdf(x) * len(returns) * (returns.max() - returns.min()) / 30,
                'r-', alpha=0.5, label='Normal Fit')

        # Set labels and title
        ax.set_title('Returns Distribution')
        ax.set_xlabel('Daily Return')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_monthly_returns_heatmap(self, figsize=(12, 8)):
        """
        Plot monthly returns heatmap

        Args:
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        if self.portfolio_data is None or self.portfolio_data.empty:
            self.logger.warning("No portfolio data to plot")
            return None

        if 'total' not in self.portfolio_data.columns:
            self.logger.warning("Portfolio data missing 'total' column")
            return None

        # Calculate returns
        returns = self.portfolio_data['total'].pct_change().dropna()

        # Create monthly returns
        returns.index = pd.to_datetime(returns.index)
        monthly_returns = returns.groupby([returns.index.year, returns.index.month]).apply(
            lambda x: (1 + x).prod() - 1
        )

        # Convert to DataFrame with multi-index
        monthly_returns = monthly_returns.reset_index()
        monthly_returns.columns = ['Year', 'Month', 'Return']

        # Pivot to get years as rows and months as columns
        heatmap_data = monthly_returns.pivot(index='Year', columns='Month', values='Return')

        # Replace month numbers with names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        heatmap_data.columns = [month_names[m - 1] for m in heatmap_data.columns]

        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.2%', cmap='RdYlGn', center=0,
                    linewidths=1, ax=ax)

        # Set labels and title
        ax.set_title('Monthly Returns Heatmap')
        ax.set_ylabel('Year')

        plt.tight_layout()
        return fig

    def plot_trades(self, trades_data, figsize=(12, 6)):
        """
        Plot trades on equity curve

        Args:
            trades_data (pandas.DataFrame): Trades data
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        if self.portfolio_data is None or self.portfolio_data.empty:
            self.logger.warning("No portfolio data to plot")
            return None

        if trades_data is None or len(trades_data) == 0:
            self.logger.warning("No trades data to plot")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        # Plot equity curve
        ax.plot(self.portfolio_data.index, self.portfolio_data['total'],
                label='Portfolio', linewidth=2)

        # Plot entries and exits
        for _, trade in trades_data.iterrows():
            # Check if required columns exist
            if 'entry_time' not in trade or 'exit_time' not in trade:
                continue

            # Get entry and exit points
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']

            # Find portfolio values at entry and exit
            try:
                entry_idx = self.portfolio_data.index.get_indexer([entry_time], method='nearest')[0]
                exit_idx = self.portfolio_data.index.get_indexer([exit_time], method='nearest')[0]

                entry_value = self.portfolio_data['total'].iloc[entry_idx]
                exit_value = self.portfolio_data['total'].iloc[exit_idx]

                # Plot entry and exit markers
                if trade.get('type') == 'long':
                    ax.plot(entry_time, entry_value, 'g^', markersize=8)  # Green triangle up for long entry
                    ax.plot(exit_time, exit_value, 'rv', markersize=8)  # Red triangle down for long exit
                else:
                    ax.plot(entry_time, entry_value, 'rv', markersize=8)  # Red triangle down for short entry
                    ax.plot(exit_time, exit_value, 'g^', markersize=8)  # Green triangle up for short exit
            except Exception as e:
                self.logger.warning(f"Error plotting trade: {e}")

        # Format dates on x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Set labels and title
        ax.set_title('Portfolio Equity Curve with Trades')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_interactive_chart(self):
        """
        Create interactive chart using Plotly

        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        if self.portfolio_data is None or self.portfolio_data.empty:
            self.logger.warning("No portfolio data to plot")
            return None

        # Create subplots: 2 rows, 1 column
        fig = make_subplots(rows=3, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.05,
                            row_heights=[0.6, 0.2, 0.2])

        # Add equity curve
        fig.add_trace(
            go.Scatter(x=self.portfolio_data.index, y=self.portfolio_data['total'],
                       name='Portfolio Value', line={'width': 2}),
            row=1, col=1
        )

        # Add drawdown chart
        if 'total' in self.portfolio_data.columns:
            equity = self.portfolio_data['total']
            peak = equity.cummax()
            drawdown = (equity - peak) / peak

            fig.add_trace(
                go.Scatter(x=drawdown.index, y=drawdown,
                           name='Drawdown', line={'color': 'red', 'width': 1},
                           fill='tozeroy', fillcolor='rgba(255,0,0,0.2)'),
                row=2, col=1
            )

        # Add position chart if available
        if 'position' in self.portfolio_data.columns:
            fig.add_trace(
                go.Scatter(x=self.portfolio_data.index, y=self.portfolio_data['position'],
                           name='Position', line={'color': 'purple', 'width': 1}),
                row=3, col=1
            )

        # Update layout
        fig.update_layout(
            title='Interactive Portfolio Performance',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            yaxis2_title='Drawdown',
            yaxis3_title='Position',
            legend_title='Metrics',
            hovermode='x unified'
        )

        # Add performance metrics as annotations if available
        if self.metrics:
            annotations = []
            metrics_text = (
                f"Return: {self.metrics.get('total_return', 0):.2%}<br>"
                f"Sharpe: {self.metrics.get('annualized_sharpe', 0):.2f}<br>"
                f"Max DD: {self.metrics.get('max_drawdown', 0):.2%}<br>"
                f"Trades: {self.metrics.get('number_of_trades', 0)}"
            )

            annotations.append(dict(
                x=0.01,
                y=0.99,
                xref="paper",
                yref="paper",
                text=metrics_text,
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255, 255, 240, 0.8)",
                bordercolor="gray",
                borderwidth=1,
                borderpad=4
            ))

            fig.update_layout(annotations=annotations)

        return fig

    def save_plots(self, directory='plots'):
        """
        Save all plots to a directory

        Args:
            directory (str): Directory to save plots

        Returns:
            list: Paths to saved plots
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        saved_files = []

        # Generate and save each plot
        plots = [
            ('equity_curve', self.plot_equity_curve()),
            ('drawdowns', self.plot_drawdowns()),
            ('returns_distribution', self.plot_returns_distribution()),
            ('monthly_returns', self.plot_monthly_returns_heatmap())
        ]

        for name, fig in plots:
            if fig is not None:
                filename = os.path.join(directory, f"{name}.png")
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_files.append(filename)

        # Save interactive plot if available
        try:
            interactive_fig = self.create_interactive_chart()
            if interactive_fig is not None:
                filename = os.path.join(directory, "interactive_chart.html")
                interactive_fig.write_html(filename)
                saved_files.append(filename)
        except Exception as e:
            self.logger.error(f"Error saving interactive chart: {e}")

        return saved_files