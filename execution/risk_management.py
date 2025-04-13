"""
Risk management system for trading
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class RiskManager:
    def __init__(self, config):
        """
        Initialize Risk Manager

        Args:
            config: Configuration settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Risk parameters
        self.max_position_size = config.MAX_POSITION_SIZE
        self.max_risk_per_trade = config.MAX_RISK_PER_TRADE
        self.max_daily_loss = config.MAX_DAILY_LOSS

        # Risk tracking
        self.daily_pnl = 0
        self.last_reset_day = datetime.now().date()
        self.trades_today = []
        self.drawdown = 0
        self.peak_value = config.INITIAL_CAPITAL

        # Market state
        self.latest_market_data = None

    def update_market_data(self, market_data):
        """Update latest market data"""
        self.latest_market_data = market_data

    def reset_daily_stats(self):
        """Reset daily statistics"""
        today = datetime.now().date()
        if today > self.last_reset_day:
            self.logger.info("Resetting daily risk statistics")
            self.daily_pnl = 0
            self.trades_today = []
            self.last_reset_day = today

    def update_portfolio_value(self, portfolio_value):
        """
        Update portfolio value and calculate drawdown

        Args:
            portfolio_value (float): Current portfolio value
        """
        # Update peak value
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        # Calculate drawdown
        self.drawdown = (self.peak_value - portfolio_value) / self.peak_value if self.peak_value > 0 else 0

    def record_trade(self, trade):
        """
        Record a completed trade

        Args:
            trade (dict): Trade details
        """
        self.trades_today.append(trade)
        self.daily_pnl += trade.get('profit', 0)

    def check_action(self, action, current_position, portfolio_value, current_price):
        """
        Check if an action is allowed by risk management rules

        Args:
            action (int): Action to check (0=hold, 1=buy, 2=sell, 3=close)
            current_position (int): Current position size
            portfolio_value (float): Current portfolio value
            current_price (float): Current market price

        Returns:
            bool: Whether the action is allowed
        """
        # Reset daily stats if needed
        self.reset_daily_stats()

        # Update portfolio value and drawdown
        self.update_portfolio_value(portfolio_value)

        # Always allow closing positions
        if action == 3:
            return True

        # Check for daily loss limit
        if self.daily_pnl < -portfolio_value * self.max_daily_loss:
            self.logger.warning("Daily loss limit reached, no new trades allowed")
            return False

        # Check for excessive drawdown (> 20%)
        if self.drawdown > 0.2:
            self.logger.warning(f"Excessive drawdown ({self.drawdown:.2%}), no new trades allowed")
            return False

        # Check for position limits
        if action == 1:  # BUY
            if current_position >= self.max_position_size:
                self.logger.warning(f"Maximum long position reached ({current_position})")
                return False
        elif action == 2:  # SELL
            if current_position <= -self.max_position_size:
                self.logger.warning(f"Maximum short position reached ({current_position})")
                return False

        # Check volatility conditions
        if self.latest_market_data is not None:
            # Check if volatility is too high for new positions
            if 'atr_14' in self.latest_market_data.columns:
                atr = self.latest_market_data['atr_14'].iloc[-1]
                avg_atr = self.latest_market_data['atr_14'].rolling(5).mean().iloc[-1]

                # If current ATR is more than 2x the average, limit new positions
                if atr > 2 * avg_atr and action != 3:
                    self.logger.warning(f"Volatility too high (ATR: {atr:.2f}, Avg: {avg_atr:.2f})")
                    return False

        # All checks passed
        return True

    def get_position_size(self, side, price, portfolio_value, current_position):
        """
        Calculate position size based on risk parameters

        Args:
            side (str): Trade side ('long' or 'short')
            price (float): Current price
            portfolio_value (float): Current portfolio value
            current_position (int): Current position size

        Returns:
            int: Recommended position size
        """
        # Make sure we have latest market data
        if self.latest_market_data is None or self.latest_market_data.empty:
            self.logger.warning("No market data available for position sizing")
            return 1  # Default to minimum size

        # Get ATR for stop placement
        if 'atr_14' in self.latest_market_data.columns:
            atr = self.latest_market_data['atr_14'].iloc[-1]
        else:
            # Fallback to calculating a simple range-based volatility
            recent_data = self.latest_market_data.iloc[-14:]
            price_range = (recent_data['high'] - recent_data['low']).mean()
            atr = price_range

        # Calculate stop distance based on ATR
        stop_distance = atr * self.config.STOP_LOSS_ATR_MULTIPLIER

        # Calculate dollar risk per contract
        dollar_risk_per_contract = stop_distance * self.config.CONTRACT_SIZE

        # Calculate maximum dollar risk for this trade
        max_dollar_risk = portfolio_value * self.max_risk_per_trade

        # Calculate position size
        if dollar_risk_per_contract > 0:
            position_size = max_dollar_risk / dollar_risk_per_contract
            position_size = int(position_size)  # Round down to integer
        else:
            position_size = 1  # Minimum size

        # Limit by max position size
        position_size = min(position_size, self.max_position_size)

        # Adjust for current position
        if side == 'long':
            # Limit long positions
            position_size = min(position_size, self.max_position_size - current_position)
        else:  # short
            # Limit short positions
            position_size = min(position_size, self.max_position_size + current_position)

        # Ensure at least 1 contract
        position_size = max(1, position_size)

        return position_size

    def get_stop_loss(self, side, entry_price):
        """
        Calculate stop loss price based on ATR

        Args:
            side (str): Trade side ('long' or 'short')
            entry_price (float): Entry price

        Returns:
            float: Recommended stop loss price
        """
        if self.latest_market_data is None or self.latest_market_data.empty:
            # Default to 2% stop if no market data
            return entry_price * 0.98 if side == 'long' else entry_price * 1.02

        # Get ATR for stop placement
        if 'atr_14' in self.latest_market_data.columns:
            atr = self.latest_market_data['atr_14'].iloc[-1]
        else:
            # Fallback to calculating a simple range-based volatility
            recent_data = self.latest_market_data.iloc[-14:]
            price_range = (recent_data['high'] - recent_data['low']).mean()
            atr = price_range

        # Calculate stop distance
        stop_distance = atr * self.config.STOP_LOSS_ATR_MULTIPLIER

        # Calculate stop price
        if side == 'long':
            stop_price = entry_price - stop_distance
        else:  # short
            stop_price = entry_price + stop_distance

        return stop_price

    def get_take_profit(self, side, entry_price):
        """
        Calculate take profit price based on ATR and risk:reward ratio

        Args:
            side (str): Trade side ('long' or 'short')
            entry_price (float): Entry price

        Returns:
            float: Recommended take profit price
        """
        if self.latest_market_data is None or self.latest_market_data.empty:
            # Default to 4% target if no market data
            return entry_price * 1.04 if side == 'long' else entry_price * 0.96

        # Get ATR for target placement
        if 'atr_14' in self.latest_market_data.columns:
            atr = self.latest_market_data['atr_14'].iloc[-1]
        else:
            # Fallback to calculating a simple range-based volatility
            recent_data = self.latest_market_data.iloc[-14:]
            price_range = (recent_data['high'] - recent_data['low']).mean()
            atr = price_range

        # Calculate target distance (risk:reward ratio typically 1:1.5 or 1:2)
        target_distance = atr * self.config.TAKE_PROFIT_ATR_MULTIPLIER

        # Calculate target price
        if side == 'long':
            target_price = entry_price + target_distance
        else:  # short
            target_price = entry_price - target_distance

        return target_price

    def get_risk_report(self):
        """
        Generate risk management report

        Returns:
            dict: Risk metrics
        """
        report = {
            'daily_pnl': self.daily_pnl,
            'trades_today': len(self.trades_today),
            'drawdown': self.drawdown,
            'peak_value': self.peak_value,
            'max_position_allowed': self.max_position_size,
            'daily_loss_limit': self.max_daily_loss,
            'daily_loss_limit_amount': self.peak_value * self.max_daily_loss
        }

        return report