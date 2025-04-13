"""
Backtesting framework for trading strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from tqdm import tqdm
import math


class Backtester:
    def __init__(self, data, agent=None, config=None):
        """
        Initialize Backtester

        Args:
            data (pandas.DataFrame): Historical market data
            agent: RL agent (optional)
            config: Configuration settings
        """
        self.data = data
        self.agent = agent
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Trading parameters
        if config:
            self.initial_capital = config.INITIAL_CAPITAL
            self.commission = config.COMMISSION_PER_CONTRACT
            self.contract_size = config.CONTRACT_SIZE
            self.slippage_points = config.SLIPPAGE_POINTS
        else:
            self.initial_capital = 100000
            self.commission = 2.0
            self.contract_size = 20.0
            self.slippage_points = 0.5

        # Initialize portfolio metrics
        self.portfolio = None
        self.trades = []

    def run(self, strategy=None, window_size=20):
        """
        Run backtest using either an RL agent or traditional strategy

        Args:
            strategy: Traditional trading strategy (optional)
            window_size (int): Window size for RL agent's observation

        Returns:
            pandas.DataFrame: Backtest results
        """
        # Initialize portfolio tracking
        self.portfolio = pd.DataFrame(index=self.data.index)
        self.portfolio['position'] = 0
        self.portfolio['price'] = self.data['close']
        self.portfolio['cash'] = self.initial_capital
        self.portfolio['holdings'] = 0
        self.portfolio['total'] = self.initial_capital
        self.trades = []

        # Track trade metrics
        trade_id = 0
        current_position = 0
        entry_price = 0
        entry_time = None

        # Choose execution mode based on inputs
        use_agent = self.agent is not None
        use_strategy = strategy is not None

        if not use_agent and not use_strategy:
            self.logger.error("Either agent or strategy must be provided")
            return None

        # Prepare for RL agent if used
        if use_agent:
            # Create state window (default to zeros initially)
            state_window = np.zeros((window_size, len(self.data.columns)))

        # Run backtest
        for i in tqdm(range(window_size, len(self.data)), desc="Backtesting"):
            current_time = self.data.index[i]
            current_price = self.data['close'].iloc[i]

            # Get current position
            current_position = self.portfolio['position'].iloc[i - 1]
            self.portfolio.loc[current_time, 'position'] = current_position

            # Update cash and holdings
            self.portfolio.loc[current_time, 'cash'] = self.portfolio['cash'].iloc[i - 1]
            self.portfolio.loc[current_time, 'holdings'] = current_position * self.contract_size * current_price

            # Determine action
            action = 0  # Default to HOLD

            if use_agent:
                # Update state window
                state_window = np.roll(state_window, -1, axis=0)
                state_window[-1] = self.data.iloc[i - 1].values

                # Create state representation
                state = state_window.flatten()

                # Add position info to state
                position_info = np.array([
                    current_position,
                    self.portfolio['cash'].iloc[i - 1] / self.initial_capital,
                    self.portfolio['total'].iloc[i - 1] / self.initial_capital,
                    (i - (entry_time or i)) / 100  # Time since entry
                ])

                state = np.concatenate([state, position_info])

                # Get action from agent
                action = self.agent.act(state, training=False)

            elif use_strategy:
                # Get signals from traditional strategy
                signals = strategy.generate_signals(self.data.iloc[:i + 1])
                if not signals.empty and signals.iloc[-1] != 0:
                    # Convert signal to action (1=buy, 2=sell, 0=hold)
                    if signals.iloc[-1] > 0:
                        action = 1  # BUY
                    else:
                        action = 2  # SELL

            # Execute action
            if action == 1:  # BUY
                if current_position <= 0:
                    # Close any existing short position
                    if current_position < 0:
                        # Calculate profit/loss
                        points_profit = entry_price - (current_price + self.slippage_points)
                        trade_profit = abs(current_position) * self.contract_size * points_profit

                        # Update cash
                        self.portfolio.loc[current_time, 'cash'] += trade_profit - abs(
                            current_position) * self.commission

                        # Log trade
                        self.trades.append({
                            'id': trade_id,
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'exit_time': current_time,
                            'exit_price': current_price,
                            'position': current_position,
                            'profit': trade_profit,
                            'type': 'short'
                        })
                        trade_id += 1

                    # Enter long position (default 1 contract)
                    new_position = 1
                    entry_price = current_price + self.slippage_points
                    entry_time = current_time

                    # Update cash to account for commission
                    self.portfolio.loc[current_time, 'cash'] -= self.commission

                    # Update position
                    self.portfolio.loc[current_time, 'position'] = new_position

            elif action == 2:  # SELL
                if current_position >= 0:
                    # Close any existing long position
                    if current_position > 0:
                        # Calculate profit/loss
                        points_profit = (current_price - self.slippage_points) - entry_price
                        trade_profit = current_position * self.contract_size * points_profit

                        # Update cash
                        self.portfolio.loc[current_time, 'cash'] += trade_profit - current_position * self.commission

                        # Log trade
                        self.trades.append({
                            'id': trade_id,
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'exit_time': current_time,
                            'exit_price': current_price,
                            'position': current_position,
                            'profit': trade_profit,
                            'type': 'long'
                        })
                        trade_id += 1

                    # Enter short position (default 1 contract)
                    new_position = -1
                    entry_price = current_price - self.slippage_points
                    entry_time = current_time

                    # Update cash to account for commission
                    self.portfolio.loc[current_time, 'cash'] -= self.commission

                    # Update position
                    self.portfolio.loc[current_time, 'position'] = new_position

            elif action == 3:  # CLOSE
                if current_position != 0:
                    # Close any existing position
                    if current_position > 0:
                        # Calculate profit/loss for long
                        points_profit = (current_price - self.slippage_points) - entry_price
                        trade_profit = current_position * self.contract_size * points_profit

                        # Log trade
                        self.trades.append({
                            'id': trade_id,
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'exit_time': current_time,
                            'exit_price': current_price,
                            'position': current_position,
                            'profit': trade_profit,
                            'type': 'long'
                        })
                        trade_id += 1

                    else:
                        # Calculate profit/loss for short
                        points_profit = entry_price - (current_price + self.slippage_points)
                        trade_profit = abs(current_position) * self.contract_size * points_profit

                        # Log trade
                        self.trades.append({
                            'id': trade_id,
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'exit_time': current_time,
                            'exit_price': current_price,
                            'position': current_position,
                            'profit': trade_profit,
                            'type': 'short'
                        })
                        trade_id += 1

                    # Update cash to account for profit/loss and commission
                    self.portfolio.loc[current_time, 'cash'] += trade_profit - abs(current_position) * self.commission

                    # Close position
                    self.portfolio.loc[current_time, 'position'] = 0
                    entry_price = 0
                    entry_time = None

            # Update total value
            self.portfolio.loc[current_time, 'total'] = self.portfolio.loc[current_time, 'cash'] + self.portfolio.loc[
                current_time, 'holdings']

        # Calculate performance metrics
        self._calculate_performance_metrics()

        # Convert trades to DataFrame
        if self.trades:
            self.trades = pd.DataFrame(self.trades)

        return self.portfolio

    def _calculate_performance_metrics(self):
        """Calculate portfolio performance metrics"""
        # Calculate daily returns
        self.portfolio['daily_return'] = self.portfolio['total'].pct_change()

        # Calculate cumulative returns
        self.portfolio['cumulative_return'] = (1 + self.portfolio['daily_return']).cumprod() - 1

        # Calculate drawdowns
        self.portfolio['peak'] = self.portfolio['total'].cummax()
        self.portfolio['drawdown'] = (self.portfolio['total'] - self.portfolio['peak']) / self.portfolio['peak']

    def get_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if self.portfolio is None:
            self.logger.error("No backtest results available")
            return None

        # Basic metrics
        total_return = (self.portfolio['total'].iloc[-1] - self.initial_capital) / self.initial_capital

        # Annualized return (assuming 252 trading days per year)
        days = (self.portfolio.index[-1] - self.portfolio.index[0]).days
        years = days / 365
        annualized_return = (1 + total_return) ** (1 / max(years, 1 / 252)) - 1

        # Risk metrics
        daily_returns = self.portfolio['daily_return'].dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * (252 ** 0.5)  # Annualized
        sortino_ratio = daily_returns.mean() / daily_returns[daily_returns < 0].std() * (252 ** 0.5)
        max_drawdown = self.portfolio['drawdown'].min()

        # Trade metrics
        if hasattr(self, 'trades') and len(self.trades) > 0:
            num_trades = len(self.trades)
            profitable_trades = sum(self.trades['profit'] > 0)
            win_rate = profitable_trades / num_trades if num_trades > 0 else 0
            avg_profit = self.trades['profit'].mean() if num_trades > 0 else 0
            profit_factor = abs(self.trades[self.trades['profit'] > 0]['profit'].sum() /
                                self.trades[self.trades['profit'] < 0]['profit'].sum()) if num_trades > 0 else 0
        else:
            num_trades = 0
            win_rate = 0
            avg_profit = 0
            profit_factor = 0

        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'number_of_trades': num_trades,
            'win_rate': win_rate,
            'average_profit': avg_profit,
            'profit_factor': profit_factor
        }

        return metrics

    def plot_results(self, figsize=(15, 10)):
        """Plot backtest results"""
        if self.portfolio is None:
            self.logger.error("No backtest results available")
            return

        metrics = self.get_performance_metrics()

        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1, 1]})

        # Plot portfolio value
        axes[0].plot(self.portfolio.index, self.portfolio['total'])
        axes[0].set_title('Portfolio Value')
        axes[0].grid(True)

        # Plot position
        axes[1].plot(self.portfolio.index, self.portfolio['position'])
        axes[1].set_title('Position')
        axes[1].grid(True)

        # Plot drawdown
        axes[2].fill_between(self.portfolio.index, self.portfolio['drawdown'], 0, color='red', alpha=0.3)
        axes[2].set_title('Drawdown')
        axes[2].grid(True)

        # Add annotations with metrics
        metrics_text = f"Total Return: {metrics['total_return']:.2%}\n"
        metrics_text += f"Annualized Return: {metrics['annualized_return']:.2%}\n"
        metrics_text += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        metrics_text += f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
        metrics_text += f"Number of Trades: {metrics['number_of_trades']}\n"
        metrics_text += f"Win Rate: {metrics['win_rate']:.2%}\n"

        # Add text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axes[0].text(0.05, 0.95, metrics_text, transform=axes[0].transAxes, fontsize=10,
                     verticalalignment='top', bbox=props)

        plt.tight_layout()
        return fig