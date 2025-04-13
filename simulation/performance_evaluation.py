"""
Performance evaluation for trading strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime, timedelta


class PerformanceEvaluator:
    def __init__(self):
        """Initialize Performance Evaluator"""
        self.logger = logging.getLogger(__name__)

    def evaluate_strategy(self, strategy, data, initial_capital=100000):
        """
        Evaluate a trading strategy on historical data

        Args:
            strategy: Trading strategy to evaluate
            data (pandas.DataFrame): Historical market data
            initial_capital (float): Initial capital

        Returns:
            dict: Performance metrics
        """
        # Generate signals
        signals = strategy.generate_signals(data)

        # Calculate position sizes
        positions = strategy.calculate_position_sizes(signals, data, initial_capital)

        # Run backtest
        from simulation.backtester import Backtester
        backtester = Backtester(data, config=None)
        backtester.initial_capital = initial_capital
        backtester.run(strategy=strategy)

        # Get performance metrics
        metrics = backtester.get_performance_metrics()

        return metrics

    def compare_strategies(self, strategies, data, initial_capital=100000):
        """
        Compare multiple trading strategies

        Args:
            strategies (list): List of strategy instances
            data (pandas.DataFrame): Historical market data
            initial_capital (float): Initial capital

        Returns:
            pandas.DataFrame: Comparison results
        """
        results = []

        for strategy in strategies:
            # Evaluate strategy
            metrics = self.evaluate_strategy(strategy, data, initial_capital)

            # Add strategy name
            metrics['strategy'] = strategy.name

            # Add to results
            results.append(metrics)

        # Convert to DataFrame
        comparison = pd.DataFrame(results)

        # Reorder columns
        if not comparison.empty and 'strategy' in comparison.columns:
            cols = ['strategy'] + [col for col in comparison.columns if col != 'strategy']
            comparison = comparison[cols]

        return comparison

    def plot_equity_curves(self, strategies, data, initial_capital=100000, figsize=(12, 6)):
        """
        Plot equity curves for multiple strategies

        Args:
            strategies (list): List of strategy instances
            data (pandas.DataFrame): Historical market data
            initial_capital (float): Initial capital
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        from simulation.backtester import Backtester

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Add benchmark (buy and hold)
        benchmark_equity = initial_capital * (1 + data['close'].pct_change().fillna(0)).cumprod()
        ax.plot(data.index, benchmark_equity, label='Buy & Hold', linestyle='--', alpha=0.7)

        # Run backtest for each strategy
        backtester = Backtester(data, config=None)
        backtester.initial_capital = initial_capital

        for strategy in strategies:
            # Run backtest
            backtester.run(strategy=strategy)

            # Plot equity curve
            ax.plot(backtester.portfolio.index, backtester.portfolio['total'],
                    label=strategy.name)

        # Add labels and legend
        ax.set_title('Equity Curve Comparison')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_drawdowns(self, strategies, data, initial_capital=100000, figsize=(12, 6)):
        """
        Plot drawdowns for multiple strategies

        Args:
            strategies (list): List of strategy instances
            data (pandas.DataFrame): Historical market data
            initial_capital (float): Initial capital
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        from simulation.backtester import Backtester

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Run backtest for each strategy
        backtester = Backtester(data, config=None)
        backtester.initial_capital = initial_capital

        for strategy in strategies:
            # Run backtest
            backtester.run(strategy=strategy)

            # Plot drawdowns
            ax.fill_between(backtester.portfolio.index, backtester.portfolio['drawdown'], 0,
                            alpha=0.3, label=f"{strategy.name} Drawdown")

        # Add labels
        ax.set_title('Strategy Drawdowns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_trade_analysis(self, trades_df, figsize=(15, 10)):
        """
        Plot trade analysis charts

        Args:
            trades_df (pandas.DataFrame): Trades data
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        if trades_df.empty:
            self.logger.warning("No trades to analyze")
            return None

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Profit distribution
        sns.histplot(trades_df['profit'], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Profit Distribution')
        axes[0, 0].axvline(0, color='r', linestyle='--')

        # 2. Cumulative profit
        cum_profit = trades_df['profit'].cumsum()
        axes[0, 1].plot(cum_profit.index, cum_profit.values)
        axes[0, 1].set_title('Cumulative Profit')
        axes[0, 1].grid(True)

        # 3. Win/Loss by trade type
        trades_df['result'] = trades_df['profit'].apply(lambda x: 'Win' if x > 0 else 'Loss')
        win_loss = pd.crosstab(trades_df['type'], trades_df['result'])
        win_loss.plot(kind='bar', stacked=True, ax=axes[1, 0])
        axes[1, 0].set_title('Win/Loss by Trade Type')

        # 4. Profit by month
        if 'entry_time' in trades_df.columns:
            # Convert entry_time to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(trades_df['entry_time']):
                trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

            # Group by month
            trades_df['month'] = trades_df['entry_time'].dt.strftime('%Y-%m')
            monthly_profit = trades_df.groupby('month')['profit'].sum().reset_index()

            # Plot
            sns.barplot(x='month', y='profit', data=monthly_profit, ax=axes[1, 1])
            axes[1, 1].set_title('Monthly Profit')
            axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=90)
        else:
            axes[1, 1].set_title('Monthly Profit (No date data available)')

        plt.tight_layout()
        return fig