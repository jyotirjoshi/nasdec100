"""
Mean reversion strategies implementation
"""

import numpy as np
import pandas as pd
from strategies.strategy_base import Strategy


class RSIStrategy(Strategy):
    def __init__(self, period=14, oversold=30, overbought=70):
        """
        Initialize RSI Mean Reversion strategy

        Args:
            period (int): RSI period
            oversold (int): Oversold threshold
            overbought (int): Overbought threshold
        """
        super().__init__(
            name="RSI_MeanReversion",
            params={
                "period": period,
                "oversold": oversold,
                "overbought": overbought
            }
        )

    def generate_signals(self, data):
        """Generate buy/sell signals based on RSI indicators"""
        period = self.params["period"]
        oversold = self.params["oversold"]
        overbought = self.params["overbought"]

        # Ensure RSI is calculated
        rsi_col = f'rsi_{period}'
        if rsi_col not in data.columns:
            self.logger.warning(f"{rsi_col} not found in data")
            return pd.Series(0, index=data.index)

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Buy when RSI crosses from below oversold to above oversold
        buy_signals = (data[rsi_col].shift(1) <= oversold) & (data[rsi_col] > oversold)

        # Sell when RSI crosses from above overbought to below overbought
        sell_signals = (data[rsi_col].shift(1) >= overbought) & (data[rsi_col] < overbought)

        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals


class BollingerBandsStrategy(Strategy):
    def __init__(self, period=20, std_dev=2.0):
        """
        Initialize Bollinger Bands Mean Reversion strategy

        Args:
            period (int): Bollinger Bands period
            std_dev (float): Standard deviation multiplier
        """
        super().__init__(
            name="BollingerBands",
            params={
                "period": period,
                "std_dev": std_dev
            }
        )

    def generate_signals(self, data):
        """Generate buy/sell signals based on Bollinger Bands"""
        # Ensure Bollinger Bands are calculated
        if 'bb_high' not in data.columns or 'bb_low' not in data.columns:
            self.logger.warning("Bollinger Bands not found in data")
            return pd.Series(0, index=data.index)

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Price touching or crossing bands
        price_below_lower = data['close'] <= data['bb_low']
        price_above_upper = data['close'] >= data['bb_high']

        # Oversold/Overbought conditions with confirmation
        oversold = price_below_lower & (data['close'] > data['close'].shift(1))
        overbought = price_above_upper & (data['close'] < data['close'].shift(1))

        # Mean reversion logic
        signals[oversold] = 1  # Buy when oversold and price starts rising
        signals[overbought] = -1  # Sell when overbought and price starts falling

        return signals


class MeanReversionZScore(Strategy):
    def __init__(self, lookback_period=20, entry_threshold=2.0, exit_threshold=0.0):
        """
        Initialize Z-Score Mean Reversion strategy

        Args:
            lookback_period (int): Period for calculating mean and standard deviation
            entry_threshold (float): Z-score threshold for entry
            exit_threshold (float): Z-score threshold for exit
        """
        super().__init__(
            name="ZScore_MeanReversion",
            params={
                "lookback_period": lookback_period,
                "entry_threshold": entry_threshold,
                "exit_threshold": exit_threshold
            }
        )

    def generate_signals(self, data):
        """Generate buy/sell signals based on Z-Score Mean Reversion"""
        # Get parameters
        lookback = self.params["lookback_period"]
        entry_threshold = self.params["entry_threshold"]
        exit_threshold = self.params["exit_threshold"]

        # Calculate rolling mean and standard deviation
        rolling_mean = data['close'].rolling(window=lookback).mean()
        rolling_std = data['close'].rolling(window=lookback).std()

        # Calculate Z-score
        data['zscore'] = (data['close'] - rolling_mean) / rolling_std

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Buy when Z-score is below negative threshold
        buy_signals = data['zscore'] < -entry_threshold

        # Sell when Z-score is above positive threshold
        sell_signals = data['zscore'] > entry_threshold

        # Exit positions when Z-score crosses exit threshold
        exit_longs = (data['zscore'].shift(1) < exit_threshold) & \
                     (data['zscore'] >= exit_threshold)

        exit_shorts = (data['zscore'].shift(1) > -exit_threshold) & \
                      (data['zscore'] <= -exit_threshold)

        signals[buy_signals] = 1
        signals[sell_signals] = -1
        signals[exit_longs] = -1  # Exit long positions
        signals[exit_shorts] = 1  # Exit short positions

        return signals


class StochasticMeanReversion(Strategy):
    def __init__(self, k_period=14, d_period=3, oversold=20, overbought=80):
        """
        Initialize Stochastic Oscillator Mean Reversion strategy

        Args:
            k_period (int): %K period
            d_period (int): %D period
            oversold (int): Oversold threshold
            overbought (int): Overbought threshold
        """
        super().__init__(
            name="Stochastic_MeanReversion",
            params={
                "k_period": k_period,
                "d_period": d_period,
                "oversold": oversold,
                "overbought": overbought
            }
        )

    def generate_signals(self, data):
        """Generate buy/sell signals based on Stochastic Oscillator"""
        # Check if Stochastic Oscillator is already calculated
        if 'stoch_k' not in data.columns or 'stoch_d' not in data.columns:
            self.logger.warning("Stochastic Oscillator not found in data")
            return pd.Series(0, index=data.index)

        # Get parameters
        oversold = self.params["oversold"]
        overbought = self.params["overbought"]

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate signals - buy when both %K and %D cross up through oversold
        buy_signals = (data['stoch_k'].shift(1) <= oversold) & \
                      (data['stoch_k'] > oversold) & \
                      (data['stoch_d'] < 50)  # Confirm we're still in lower half

        # Sell when both %K and %D cross down through overbought
        sell_signals = (data['stoch_k'].shift(1) >= overbought) & \
                       (data['stoch_k'] < overbought) & \
                       (data['stoch_d'] > 50)  # Confirm we're still in upper half

        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals