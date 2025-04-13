"""
Base class for all trading strategies
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import logging


class Strategy(ABC):
    def __init__(self, name, params=None):
        """
        Initialize the strategy

        Args:
            name (str): Name of the strategy
            params (dict): Strategy parameters
        """
        self.name = name
        self.params = params or {}
        self.logger = logging.getLogger(f"strategy.{name}")

    @abstractmethod
    def generate_signals(self, data):
        """
        Generate trading signals based on the strategy

        Args:
            data (pandas.DataFrame): Market data with indicators

        Returns:
            pandas.Series: Trading signals (1 for buy, -1 for sell, 0 for hold)
        """
        pass

    def calculate_position_sizes(self, signals, data, capital, risk_per_trade=0.02):
        """
        Calculate position sizes based on signals and risk management

        Args:
            signals (pandas.Series): Trading signals
            data (pandas.DataFrame): Market data
            capital (float): Available capital
            risk_per_trade (float): Maximum risk per trade as fraction of capital

        Returns:
            pandas.Series: Position sizes in number of contracts
        """
        # Default implementation uses fixed fraction sizing with ATR-based stops

        # Make sure we have ATR calculated
        if 'atr_14' not in data.columns:
            self.logger.warning("ATR not found in data for position sizing")
            atr = (data['high'] - data['low']).rolling(window=14).mean()
        else:
            atr = data['atr_14']

        # Calculate position sizes
        position_sizes = pd.Series(0.0, index=signals.index)

        # Process actual signals (non-zero)
        active_signals = signals[signals != 0]

        for idx in active_signals.index:
            signal = active_signals[idx]

            if signal == 0:
                continue

            # Get ATR value for volatility-based position sizing
            current_atr = atr.loc[idx]

            # Calculate dollar risk per contract based on ATR
            # Use 2x ATR as initial stop loss distance
            risk_per_contract = current_atr * 2.0 * 20.0  # NQ is $20 per point

            # Calculate position size based on risk
            max_risk_amount = capital * risk_per_trade
            if risk_per_contract > 0:
                contracts = max_risk_amount / risk_per_contract
                # Round down to nearest integer
                contracts = max(1, int(contracts))
            else:
                contracts = 1  # Default to 1 contract if ATR is zero

            position_sizes.loc[idx] = contracts * signal

        return position_sizes

    def set_parameters(self, params):
        """Update strategy parameters"""
        self.params.update(params)
        self.logger.info(f"Updated parameters: {params}")

    def get_parameters(self):
        """Get current strategy parameters"""
        return self.params.copy()