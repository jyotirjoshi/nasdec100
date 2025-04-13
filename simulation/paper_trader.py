"""
Paper trading simulation for trading strategies and RL agents
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging
import threading
import queue


class PaperTrader:
    def __init__(self, data_fetcher, config):
        """
        Initialize Paper Trading environment

        Args:
            data_fetcher: Live data fetcher instance
            config: Configuration settings
        """
        self.data_fetcher = data_fetcher
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Trading parameters
        self.initial_capital = config.INITIAL_CAPITAL
        self.commission = config.COMMISSION_PER_CONTRACT
        self.contract_size = config.CONTRACT_SIZE
        self.slippage_points = config.SLIPPAGE_POINTS

        # Trading state
        self.cash = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
        self.pnl = 0
        self.trades = []

        # Threading components
        self.running = False
        self.trading_thread = None
        self.command_queue = queue.Queue()

        # Trading agent/strategy
        self.agent = None
        self.strategy = None

        # Performance tracking
        self.portfolio_history = []

    def set_agent(self, agent):
        """Set RL agent for trading"""
        self.agent = agent
        self.logger.info("RL agent set for paper trading")

    def set_strategy(self, strategy):
        """Set traditional strategy for trading"""
        self.strategy = strategy
        self.logger.info(f"Strategy '{strategy.name}' set for paper trading")

    def run(self):
        """Start paper trading in a separate thread"""
        if self.running:
            self.logger.warning("Paper trading is already running")
            return

        if not self.agent and not self.strategy:
            self.logger.error("No agent or strategy set for paper trading")
            return

        # Start data fetcher if not already running
        if hasattr(self.data_fetcher, 'start') and callable(self.data_fetcher.start):
            self.data_fetcher.start()

        # Start trading thread
        self.running = True
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()

        self.logger.info("Paper trading started")

    def stop(self):
        """Stop paper trading"""
        if not self.running:
            return

        self.running = False
        if self.trading_thread:
            self.command_queue.put(('stop', None))
            self.trading_thread.join(timeout=5)

        # Stop data fetcher
        if hasattr(self.data_fetcher, 'stop') and callable(self.data_fetcher.stop):
            self.data_fetcher.stop()

        self.logger.info("Paper trading stopped")

    def _trading_loop(self):
        """Main trading loop executed in a separate thread"""
        last_check_time = datetime.now()
        state_window = None
        window_size = 20  # Default window size for RL agent

        while self.running:
            try:
                # Process any pending commands
                while not self.command_queue.empty():
                    cmd, data = self.command_queue.get()
                    if cmd == 'stop':
                        return

                # Check if market is open
                if not self.data_fetcher.is_market_open():
                    time.sleep(60)  # Sleep for a minute and check again
                    continue

                # Rate limiting - check only every 5 seconds
                current_time = datetime.now()
                if (current_time - last_check_time).total_seconds() < 5:
                    time.sleep(0.1)
                    continue

                last_check_time = current_time

                # Get latest data
                latest_data = self.data_fetcher.get_latest_data(window_size * 2)
                if latest_data is None or latest_data.empty:
                    self.logger.warning("No data available")
                    time.sleep(5)
                    continue

                # Determine action based on agent or strategy
                action = 0  # Default to HOLD

                if self.agent:
                    # Prepare state for RL agent
                    if state_window is None or len(state_window) != window_size:
                        # Initialize state window
                        state_window = latest_data.iloc[-window_size:].values

                    else:
                        # Update state window
                        state_window = np.roll(state_window, -1, axis=0)
                        state_window[-1] = latest_data.iloc[-1].values

                    # Create state representation
                    state = state_window.flatten()

                    # Add position info to state
                    position_info = np.array([
                        self.position,
                        self.cash / self.initial_capital,
                        (self.cash + self.position * self.contract_size * latest_data['close'].iloc[
                            -1]) / self.initial_capital,
                        (datetime.now() - (self.entry_time or datetime.now())).total_seconds() / 3600
                        # Hours since entry
                    ])

                    state = np.concatenate([state, position_info])

                    # Get action from agent
                    action = self.agent.act(state, training=False)

                elif self.strategy:
                    # Get signals from traditional strategy
                    signals = self.strategy.generate_signals(latest_data)
                    if not signals.empty and signals.iloc[-1] != 0:
                        # Convert signal to action (1=buy, 2=sell, 0=hold)
                        if signals.iloc[-1] > 0:
                            action = 1  # BUY
                        else:
                            action = 2  # SELL

                # Execute action if not HOLD
                if action != 0:
                    self._execute_action(action, latest_data)

                # Log current state
                self._log_portfolio_state(latest_data)

            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(5)

    def _execute_action(self, action, data):
        """
        Execute trading action

        Args:
            action (int): Action to take (1=buy, 2=sell, 3=close)
            data (pandas.DataFrame): Latest market data
        """
        current_time = datetime.now()
        current_price = data['close'].iloc[-1]

        if action == 1:  # BUY
            if self.position <= 0:
                # Close any existing short position
                if self.position < 0:
                    # Calculate profit/loss
                    points_profit = self.entry_price - (current_price + self.slippage_points)
                    trade_profit = abs(self.position) * self.contract_size * points_profit

                    # Update cash
                    self.cash += trade_profit - abs(self.position) * self.commission

                    # Log trade
                    trade = {
                        'entry_time': self.entry_time,
                        'entry_price': self.entry_price,
                        'exit_time': current_time,
                        'exit_price': current_price,
                        'position': self.position,
                        'profit': trade_profit,
                        'type': 'short'
                    }
                    self.trades.append(trade)
                    self.logger.info(f"Closed short position: {trade}")

                # Enter long position (default 1 contract)
                self.position = 1
                self.entry_price = current_price + self.slippage_points
                self.entry_time = current_time

                # Update cash to account for commission
                self.cash -= self.commission

                self.logger.info(f"Opened long position at {self.entry_price}")

        elif action == 2:  # SELL
            if self.position >= 0:
                # Close any existing long position
                if self.position > 0:
                    # Calculate profit/loss
                    points_profit = (current_price - self.slippage_points) - self.entry_price
                    trade_profit = self.position * self.contract_size * points_profit

                    # Update cash
                    self.cash += trade_profit - self.position * self.commission

                    # Log trade
                    trade = {
                        'entry_time': self.entry_time,
                        'entry_price': self.entry_price,
                        'exit_time': current_time,
                        'exit_price': current_price,
                        'position': self.position,
                        'profit': trade_profit,
                        'type': 'long'
                    }
                    self.trades.append(trade)
                    self.logger.info(f"Closed long position: {trade}")

                # Enter short position (default 1 contract)
                self.position = -1
                self.entry_price = current_price - self.slippage_points
                self.entry_time = current_time

                # Update cash to account for commission
                self.cash -= self.commission

                self.logger.info(f"Opened short position at {self.entry_price}")

        elif action == 3:  # CLOSE
            if self.position != 0:
                # Close any existing position
                if self.position > 0:
                    # Calculate profit/loss for long
                    points_profit = (current_price - self.slippage_points) - self.entry_price
                    trade_profit = self.position * self.contract_size * points_profit

                    # Log trade
                    trade = {
                        'entry_time': self.entry_time,
                        'entry_price': self.entry_price,
                        'exit_time': current_time,
                        'exit_price': current_price,
                        'position': self.position,
                        'profit': trade_profit,
                        'type': 'long'
                    }
                    self.trades.append(trade)
                    self.logger.info(f"Closed long position: {trade}")

                else:
                    # Calculate profit/loss for short
                    points_profit = self.entry_price - (current_price + self.slippage_points)
                    trade_profit = abs(self.position) * self.contract_size * points_profit

                    # Log trade
                    trade = {
                        'entry_time': self.entry_time,
                        'entry_price': self.entry_price,
                        'exit_time': current_time,
                        'exit_price': current_price,
                        'position': self.position,
                        'profit': trade_profit,
                        'type': 'short'
                    }
                    self.trades.append(trade)
                    self.logger.info(f"Closed short position: {trade}")

                # Update cash to account for profit/loss and commission
                self.cash += trade_profit - abs(self.position) * self.commission

                # Close position
                self.position = 0
                self.entry_price = 0
                self.entry_time = None

    def _log_portfolio_state(self, data):
        """Log current portfolio state and update history"""
        current_time = datetime.now()
        current_price = data['close'].iloc[-1]

        # Calculate unrealized PnL if in position
        unrealized_pnl = 0
        if self.position > 0:
            unrealized_pnl = self.position * self.contract_size * (current_price - self.entry_price)
        elif self.position < 0:
            unrealized_pnl = abs(self.position) * self.contract_size * (self.entry_price - current_price)

        # Calculate total portfolio value
        portfolio_value = self.cash + unrealized_pnl

        # Add to history
        self.portfolio_history.append({
            'time': current_time,
            'price': current_price,
            'position': self.position,
            'cash': self.cash,
            'unrealized_pnl': unrealized_pnl,
            'portfolio_value': portfolio_value
        })

        # Trim history if it gets too large
        if len(self.portfolio_history) > 10000:
            self.portfolio_history = self.portfolio_history[-5000:]

        # Log portfolio state every 10 minutes
        if len(self.portfolio_history) % 120 == 0:  # Assuming 5-second checks
            self.logger.info(
                f"Portfolio value: ${portfolio_value:.2f}, "
                f"Position: {self.position}, "
                f"Unrealized PnL: ${unrealized_pnl:.2f}"
            )

    def get_portfolio_summary(self):
        """Get summary of current portfolio state"""
        if not self.portfolio_history:
            return {
                'current_value': self.initial_capital,
                'total_return': 0,
                'position': 0
            }

        current = self.portfolio_history[-1]

        # Calculate metrics
        total_return = (current['portfolio_value'] - self.initial_capital) / self.initial_capital

        # Calculate drawdown
        peak_value = max([h['portfolio_value'] for h in self.portfolio_history])
        drawdown = (current['portfolio_value'] - peak_value) / peak_value if peak_value > 0 else 0

        # Trade statistics
        num_trades = len(self.trades)
        profitable_trades = sum(1 for trade in self.trades if trade['profit'] > 0)
        win_rate = profitable_trades / num_trades if num_trades > 0 else 0

        summary = {
            'current_value': current['portfolio_value'],
            'cash': current['cash'],
            'unrealized_pnl': current['unrealized_pnl'],
            'total_return': total_return,
            'drawdown': drawdown,
            'position': current['position'],
            'num_trades': num_trades,
            'win_rate': win_rate
        }

        return summary

    def get_trades(self):
        """Get list of completed trades"""
        return self.trades

    def get_portfolio_history(self):
        """Get portfolio history as a DataFrame"""
        if not self.portfolio_history:
            return pd.DataFrame()

        return pd.DataFrame(self.portfolio_history)