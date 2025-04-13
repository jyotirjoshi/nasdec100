"""
Performance metrics calculation for trading strategies
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging
from datetime import datetime, timedelta


class PerformanceAnalyzer:
    def __init__(self, portfolio_data=None):
        """
        Initialize Performance Analyzer

        Args:
            portfolio_data (pandas.DataFrame): Portfolio data with 'total' value column
        """
        self.portfolio_data = portfolio_data
        self.logger = logging.getLogger(__name__)
        self.metrics = {}

    def set_data(self, portfolio_data):
        """
        Set portfolio data for analysis

        Args:
            portfolio_data (pandas.DataFrame): Portfolio data with 'total' value column
        """
        self.portfolio_data = portfolio_data

    def calculate_metrics(self):
        """
        Calculate comprehensive performance metrics

        Returns:
            dict: Performance metrics
        """
        if self.portfolio_data is None or self.portfolio_data.empty:
            self.logger.error("No portfolio data available")
            return {}

        # Make sure 'total' column exists
        if 'total' not in self.portfolio_data.columns:
            self.logger.error("Portfolio data missing 'total' column")
            return {}

        # Extract time series
        equity = self.portfolio_data['total']

        # Calculate returns
        returns = equity.pct_change().dropna()

        # Calculate metrics
        self._calculate_basic_metrics(equity, returns)
        self._calculate_risk_metrics(returns)
        self._calculate_drawdown_metrics(equity)
        self._calculate_trading_metrics()

        return self.metrics

    def _calculate_basic_metrics(self, equity, returns):
        """Calculate basic performance metrics"""
        # Total return
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

        # Annualized return
        days = (equity.index[-1] - equity.index[0]).days
        years = max(days / 365, 1 / 252)  # Avoid division by zero
        annualized_return = (1 + total_return) ** (1 / years) - 1

        # Average daily return
        avg_daily_return = returns.mean()

        # Store metrics
        self.metrics.update({
            'total_return': total_return,
            'annualized_return': annualized_return,
            'avg_daily_return': avg_daily_return
        })

    def _calculate_risk_metrics(self, returns):
        """Calculate risk-adjusted performance metrics"""
        # Risk metrics
        volatility = returns.std()
        ann_volatility = volatility * np.sqrt(252)  # Annualized

        # Downside deviation
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() if len(negative_returns) > 0 else 0

        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = returns.mean() / volatility if volatility > 0 else 0
        ann_sharpe = sharpe_ratio * np.sqrt(252)  # Annualized

        # Sortino ratio
        sortino_ratio = returns.mean() / downside_deviation if downside_deviation > 0 else 0
        ann_sortino = sortino_ratio * np.sqrt(252)  # Annualized

        # Calmar ratio
        max_dd = self._calculate_max_drawdown(self.portfolio_data['total'])
        calmar_ratio = self.metrics['annualized_return'] / abs(max_dd) if max_dd != 0 else 0

        # Risk-adjusted return
        risk_adjusted_return = self.metrics['total_return'] / ann_volatility if ann_volatility > 0 else 0

        # Store metrics
        self.metrics.update({
            'volatility': volatility,
            'annualized_volatility': ann_volatility,
            'downside_deviation': downside_deviation,
            'sharpe_ratio': sharpe_ratio,
            'annualized_sharpe': ann_sharpe,
            'sortino_ratio': sortino_ratio,
            'annualized_sortino': ann_sortino,
            'calmar_ratio': calmar_ratio,
            'risk_adjusted_return': risk_adjusted_return
        })

    def _calculate_drawdown_metrics(self, equity):
        """Calculate drawdown metrics"""
        # Calculate drawdown series
        peak = equity.cummax()
        drawdown = (equity - peak) / peak

        # Maximum drawdown
        max_drawdown = drawdown.min()

        # Average drawdown
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0

        # Drawdown duration
        underwater = drawdown < 0
        underwater_periods = []
        start = None

        for i, is_underwater in enumerate(underwater):
            if is_underwater and start is None:
                start = i
            elif not is_underwater and start is not None:
                underwater_periods.append(i - start)
                start = None

        if start is not None:  # Still underwater at the end
            underwater_periods.append(len(underwater) - start)

        avg_drawdown_duration = np.mean(underwater_periods) if underwater_periods else 0
        max_drawdown_duration = max(underwater_periods) if underwater_periods else 0

        # Store metrics
        self.metrics.update({
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_drawdown_duration': max_drawdown_duration
        })

    def _calculate_trading_metrics(self):
        """Calculate trading-specific metrics"""
        # Check if position column exists
        if 'position' not in self.portfolio_data.columns:
            return

        positions = self.portfolio_data['position']

        # Find trades by detecting position changes
        position_changes = positions.diff().fillna(0)
        trades = position_changes[position_changes != 0]

        # Count trades
        num_trades = len(trades)

        # Average trade duration
        trade_start_idx = []
        trade_end_idx = []
        in_trade = False

        for i, pos in enumerate(positions):
            if not in_trade and pos != 0:
                trade_start_idx.append(i)
                in_trade = True
            elif in_trade and pos == 0:
                trade_end_idx.append(i)
                in_trade = False

        if in_trade:  # Still in a position at the end
            trade_end_idx.append(len(positions) - 1)

        if len(trade_start_idx) == len(trade_end_idx) and len(trade_start_idx) > 0:
            trade_durations = [
                (positions.index[end] - positions.index[start]).total_seconds() / 3600  # In hours
                for start, end in zip(trade_start_idx, trade_end_idx)
            ]
            avg_trade_duration = np.mean(trade_durations)
        else:
            avg_trade_duration = 0

        # Store metrics
        self.metrics.update({
            'number_of_trades': num_trades,
            'avg_trade_duration_hours': avg_trade_duration
        })

        # If trades list is available with profit information
        if hasattr(self, 'trades_list') and self.trades_list:
            trades_df = pd.DataFrame(self.trades_list)
            if 'profit' in trades_df.columns:
                winning_trades = trades_df[trades_df['profit'] > 0]
                losing_trades = trades_df[trades_df['profit'] < 0]

                win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
                avg_win = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
                avg_loss = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0
                profit_factor = -winning_trades['profit'].sum() / losing_trades['profit'].sum() if len(
                    losing_trades) > 0 and losing_trades['profit'].sum() != 0 else float('inf')

                # Store metrics
                self.metrics.update({
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'expectancy': win_rate * avg_win + (1 - win_rate) * avg_loss
                })

    def _calculate_max_drawdown(self, equity):
        """Calculate maximum drawdown"""
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return drawdown.min()

    def set_trades_list(self, trades_list):
        """
        Set list of executed trades for analysis

        Args:
            trades_list (list): List of trade dictionaries
        """
        self.trades_list = trades_list

    def generate_report(self):
        """
        Generate a formatted performance report

        Returns:
            str: Formatted report string
        """
        if not self.metrics:
            self.calculate_metrics()

        if not self.metrics:
            return "No performance data available."

        report = "Performance Report\n"
        report += "=================\n\n"

        # Return metrics
        report += "Return Metrics:\n"
        report += f"  Total Return: {self.metrics.get('total_return', 0):.2%}\n"
        report += f"  Annualized Return: {self.metrics.get('annualized_return', 0):.2%}\n"
        report += f"  Average Daily Return: {self.metrics.get('avg_daily_return', 0):.4%}\n\n"

        # Risk metrics
        report += "Risk Metrics:\n"
        report += f"  Annualized Volatility: {self.metrics.get('annualized_volatility', 0):.2%}\n"
        report += f"  Sharpe Ratio: {self.metrics.get('annualized_sharpe', 0):.2f}\n"
        report += f"  Sortino Ratio: {self.metrics.get('annualized_sortino', 0):.2f}\n"
        report += f"  Calmar Ratio: {self.metrics.get('calmar_ratio', 0):.2f}\n\n"

        # Drawdown metrics
        report += "Drawdown Metrics:\n"
        report += f"  Maximum Drawdown: {self.metrics.get('max_drawdown', 0):.2%}\n"
        report += f"  Average Drawdown: {self.metrics.get('avg_drawdown', 0):.2%}\n"
        report += f"  Max Drawdown Duration: {self.metrics.get('max_drawdown_duration', 0):.0f} periods\n\n"

        # Trading metrics
        report += "Trading Metrics:\n"
        report += f"  Number of Trades: {self.metrics.get('number_of_trades', 0)}\n"

        if 'win_rate' in self.metrics:
            report += f"  Win Rate: {self.metrics.get('win_rate', 0):.2%}\n"
            report += f"  Profit Factor: {self.metrics.get('profit_factor', 0):.2f}\n"
            report += f"  Average Win: ${self.metrics.get('avg_win', 0):.2f}\n"
            report += f"  Average Loss: ${self.metrics.get('avg_loss', 0):.2f}\n"
            report += f"  Expectancy: ${self.metrics.get('expectancy', 0):.2f}\n"

        return report