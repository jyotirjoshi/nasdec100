"""
Volatility-based strategy implementations
"""

import numpy as np
import pandas as pd
from strategies.strategy_base import Strategy


class ATRChannelStrategy(Strategy):
    def __init__(self, atr_period=14, channel_multiplier=2.0):
        """
        Initialize ATR Channel strategy

        Args:
            atr_period (int): Period for ATR calculation
            channel_multiplier (float): Multiplier for ATR to determine channel width
        """
        super().__init__(
            name="ATRChannel",
            params={
                "atr_period": atr_period,
                "channel_multiplier": channel_multiplier
            }
        )

    def generate_signals(self, data):
        """Generate buy/sell signals based on ATR Channels"""
        # Get parameters
        atr_period = self.params["atr_period"]
        multiplier = self.params["channel_multiplier"]

        # Calculate ATR if not already in data
        atr_col = f'atr_{atr_period}'
        if atr_col not in data.columns:
            self.logger.warning(f"{atr_col} not found in data")
            return pd.Series(0, index=data.index)

        # Calculate ATR channels
        data['atr_middle'] = data['close'].rolling(window=atr_period).mean()
        data['atr_upper'] = data['atr_middle'] + (data[atr_col] * multiplier)
        data['atr_lower'] = data['atr_middle'] - (data[atr_col] * multiplier)

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate signals
        buy_signals = (
                (data['close'].shift(1) <= data['atr_lower'].shift(1)) &
                (data['close'] > data['atr_lower'])
        )

        sell_signals = (
                (data['close'].shift(1) >= data['atr_upper'].shift(1)) &
                (data['close'] < data['atr_upper'])
        )

        # Apply signals
        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals


class VolatilityExpansionStrategy(Strategy):
    def __init__(self, period=20, threshold_factor=1.5):
        """
        Initialize Volatility Expansion strategy

        Args:
            period (int): Period for volatility calculation
            threshold_factor (float): Factor to determine expansion threshold
        """
        super().__init__(
            name="VolatilityExpansion",
            params={
                "period": period,
                "threshold_factor": threshold_factor
            }
        )

    def generate_signals(self, data):
        """Generate signals based on volatility expansion"""
        # Get parameters
        period = self.params["period"]
        threshold_factor = self.params["threshold_factor"]

        # Calculate volatility (standard deviation of returns)
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(window=period).std()
        data['volatility_sma'] = data['volatility'].rolling(window=period).mean()

        # Calculate volatility ratio
        data['volatility_ratio'] = data['volatility'] / data['volatility_sma']

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate signals
        # Buy when volatility expands and price is rising
        buy_signals = (
                (data['volatility_ratio'] > threshold_factor) &
                (data['returns'] > 0)
        )

        # Sell when volatility expands and price is falling
        sell_signals = (
                (data['volatility_ratio'] > threshold_factor) &
                (data['returns'] < 0)
        )

        # Apply signals
        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals


class KeltnerChannelStrategy(Strategy):
    def __init__(self, ema_period=20, atr_period=10, atr_multiplier=2.0):
        """
        Initialize Keltner Channel strategy

        Args:
            ema_period (int): Period for EMA calculation
            atr_period (int): Period for ATR calculation
            atr_multiplier (float): Multiplier for ATR
        """
        super().__init__(
            name="KeltnerChannel",
            params={
                "ema_period": ema_period,
                "atr_period": atr_period,
                "atr_multiplier": atr_multiplier
            }
        )

    def generate_signals(self, data):
        """Generate buy/sell signals based on Keltner Channels"""
        # Check if Keltner Channels are already calculated
        ema_period = self.params["ema_period"]
        keltner_cols = [f'keltner_high_{ema_period}', f'keltner_mid_{ema_period}', f'keltner_low_{ema_period}']

        if not all(col in data.columns for col in keltner_cols):
            self.logger.warning("Keltner Channel indicators not found in data")
            return pd.Series(0, index=data.index)

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate signals - Buy when price crosses above lower band after being below it
        buy_signals = (
                (data['close'].shift(1) <= data[f'keltner_low_{ema_period}'].shift(1)) &
                (data['close'] > data[f'keltner_low_{ema_period}'])
        )

        # Sell when price crosses below upper band after being above it
        sell_signals = (
                (data['close'].shift(1) >= data[f'keltner_high_{ema_period}'].shift(1)) &
                (data['close'] < data[f'keltner_high_{ema_period}'])
        )

        # Apply signals
        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals


class BollingerSqueezeStrategy(Strategy):
    def __init__(self, bb_period=20, keltner_period=20, bb_std=2.0, keltner_factor=1.5):
        """
        Initialize Bollinger Squeeze strategy

        Args:
            bb_period (int): Period for Bollinger Bands
            keltner_period (int): Period for Keltner Channels
            bb_std (float): Standard deviation for Bollinger Bands
            keltner_factor (float): ATR factor for Keltner Channels
        """
        super().__init__(
            name="BollingerSqueeze",
            params={
                "bb_period": bb_period,
                "keltner_period": keltner_period,
                "bb_std": bb_std,
                "keltner_factor": keltner_factor
            }
        )

    def generate_signals(self, data):
        """Generate signals based on Bollinger Squeeze (volatility contraction)"""
        # Check if required indicators are in data
        bb_cols = ['bb_high', 'bb_low', 'bb_width']
        keltner_cols = [f'keltner_high_{self.params["keltner_period"]}',
                        f'keltner_low_{self.params["keltner_period"]}']

        if not all(col in data.columns for col in bb_cols + keltner_cols):
            self.logger.warning("Required indicators for Bollinger Squeeze not found in data")
            return pd.Series(0, index=data.index)

        # Calculate if Bollinger Bands are inside Keltner Channels (squeeze)
        data['squeeze'] = (
                (data['bb_high'] < data[f'keltner_high_{self.params["keltner_period"]}']) &
                (data['bb_low'] > data[f'keltner_low_{self.params["keltner_period"]}'])
        )

        # Detect squeeze release
        data['squeeze_release'] = (
                (data['squeeze'].shift(1) == True) &
                (data['squeeze'] == False)
        )

        # Calculate momentum at squeeze release
        data['momentum'] = data['close'] - data['close'].shift(1)

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate signals - Buy on squeeze release with upward momentum
        buy_signals = (
                data['squeeze_release'] &
                (data['momentum'] > 0)
        )

        # Sell on squeeze release with downward momentum
        sell_signals = (
                data['squeeze_release'] &
                (data['momentum'] < 0)
        )

        # Apply signals
        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals