"""
Breakout strategy implementations
"""

import numpy as np
import pandas as pd
from strategies.strategy_base import Strategy


class DonchianBreakout(Strategy):
    def __init__(self, channel_period=20, exit_period=10):
        """
        Initialize Donchian Channel Breakout strategy

        Args:
            channel_period (int): Period for Donchian Channel calculation
            exit_period (int): Period for exit channel calculation
        """
        super().__init__(
            name="DonchianBreakout",
            params={
                "channel_period": channel_period,
                "exit_period": exit_period
            }
        )

    def generate_signals(self, data):
        """Generate buy/sell signals based on Donchian Channel Breakout"""
        # Get parameters
        channel_period = self.params["channel_period"]
        exit_period = self.params["exit_period"]

        # Calculate Donchian Channels if not already in data
        if f'donchian_high_{channel_period}' not in data.columns:
            data[f'donchian_high_{channel_period}'] = data['high'].rolling(window=channel_period).max()
            data[f'donchian_low_{channel_period}'] = data['low'].rolling(window=channel_period).min()

        if f'donchian_high_{exit_period}' not in data.columns:
            data[f'donchian_high_{exit_period}'] = data['high'].rolling(window=exit_period).max()
            data[f'donchian_low_{exit_period}'] = data['low'].rolling(window=exit_period).min()

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Entry signals - price breaks out above/below channel
        buy_breakouts = (data['close'] > data[f'donchian_high_{channel_period}'].shift(1))
        sell_breakouts = (data['close'] < data[f'donchian_low_{channel_period}'].shift(1))

        # Exit signals - price breaks below/above shorter-term channel
        exit_longs = (data['close'] < data[f'donchian_low_{exit_period}'].shift(1))
        exit_shorts = (data['close'] > data[f'donchian_high_{exit_period}'].shift(1))

        # Apply signals
        signals[buy_breakouts] = 1
        signals[sell_breakouts] = -1
        signals[exit_longs] = -1  # Exit long positions
        signals[exit_shorts] = 1  # Exit short positions

        return signals


class VolatilityBreakout(Strategy):
    def __init__(self, atr_period=14, atr_multiplier=2.0, lookback=3):
        """
        Initialize Volatility Breakout strategy

        Args:
            atr_period (int): Period for ATR calculation
            atr_multiplier (float): Multiplier for ATR to determine breakout threshold
            lookback (int): Lookback period for reference price
        """
        super().__init__(
            name="VolatilityBreakout",
            params={
                "atr_period": atr_period,
                "atr_multiplier": atr_multiplier,
                "lookback": lookback
            }
        )

    def generate_signals(self, data):
        """Generate buy/sell signals based on Volatility Breakout"""
        # Get parameters
        atr_period = self.params["atr_period"]
        atr_multiplier = self.params["atr_multiplier"]
        lookback = self.params["lookback"]

        # Calculate ATR if not already in data
        atr_col = f'atr_{atr_period}'
        if atr_col not in data.columns:
            self.logger.warning(f"{atr_col} not found in data")
            return pd.Series(0, index=data.index)

        # Calculate reference price - typically previous close
        reference_price = data['close'].shift(1)

        # Calculate breakout thresholds
        upper_threshold = reference_price + (data[atr_col].shift(1) * atr_multiplier)
        lower_threshold = reference_price - (data[atr_col].shift(1) * atr_multiplier)

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate breakout signals
        buy_signals = data['close'] > upper_threshold
        sell_signals = data['close'] < lower_threshold

        # Apply signals
        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals


class PriceChannelBreakout(Strategy):
    def __init__(self, channel_period=20, confirmation_bars=2):
        """
        Initialize Price Channel Breakout strategy

        Args:
            channel_period (int): Period for price channel calculation
            confirmation_bars (int): Number of bars for breakout confirmation
        """
        super().__init__(
            name="PriceChannelBreakout",
            params={
                "channel_period": channel_period,
                "confirmation_bars": confirmation_bars
            }
        )

    def generate_signals(self, data):
        """Generate buy/sell signals based on Price Channel Breakout"""
        # Get parameters
        channel_period = self.params["channel_period"]
        confirmation_bars = self.params["confirmation_bars"]

        # Calculate price channels
        data['upper_channel'] = data['high'].rolling(window=channel_period).max()
        data['lower_channel'] = data['low'].rolling(window=channel_period).min()

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate potential breakout signals
        potential_buys = data['close'] > data['upper_channel'].shift(1)
        potential_sells = data['close'] < data['lower_channel'].shift(1)

        # Apply confirmation logic if required
        if confirmation_bars > 1:
            for i in range(1, confirmation_bars):
                # Check if price remained above/below channel for required number of bars
                potential_buys = potential_buys & (
                        data['close'].shift(i) > data['upper_channel'].shift(i + 1)
                )
                potential_sells = potential_sells & (
                        data['close'].shift(i) < data['lower_channel'].shift(i + 1)
                )

        # Apply signals
        signals[potential_buys] = 1
        signals[potential_sells] = -1

        return signals


class VolumeBreakout(Strategy):
    def __init__(self, price_period=20, volume_period=20, volume_factor=1.5):
        """
        Initialize Volume-Confirmed Breakout strategy

        Args:
            price_period (int): Period for price range
            volume_period (int): Period for volume average
            volume_factor (float): Factor for volume confirmation
        """
        super().__init__(
            name="VolumeBreakout",
            params={
                "price_period": price_period,
                "volume_period": volume_period,
                "volume_factor": volume_factor
            }
        )

    def generate_signals(self, data):
        """Generate buy/sell signals based on Volume-Confirmed Breakout"""
        # Get parameters
        price_period = self.params["price_period"]
        volume_period = self.params["volume_period"]
        volume_factor = self.params["volume_factor"]

        # Calculate price channels
        data['upper_channel'] = data['high'].rolling(window=price_period).max()
        data['lower_channel'] = data['low'].rolling(window=price_period).min()

        # Calculate volume average
        data['volume_avg'] = data['volume'].rolling(window=volume_period).mean()

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate breakout signals with volume confirmation
        buy_signals = (
                (data['close'] > data['upper_channel'].shift(1)) &
                (data['volume'] > data['volume_avg'].shift(1) * volume_factor)
        )

        sell_signals = (
                (data['close'] < data['lower_channel'].shift(1)) &
                (data['volume'] > data['volume_avg'].shift(1) * volume_factor)
        )

        # Apply signals
        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals