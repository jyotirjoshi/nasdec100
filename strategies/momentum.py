"""
Momentum strategy implementations
"""

import numpy as np
import pandas as pd
from strategies.strategy_base import Strategy


class RelativeStrengthMomentum(Strategy):
    def __init__(self, lookback_period=14, smoothing_period=3):
        """
        Initialize Relative Strength Momentum strategy

        Args:
            lookback_period (int): Period for momentum calculation
            smoothing_period (int): Period for signal smoothing
        """
        super().__init__(
            name="RelativeStrengthMomentum",
            params={
                "lookback_period": lookback_period,
                "smoothing_period": smoothing_period
            }
        )

    def generate_signals(self, data):
        """Generate signals based on Relative Strength Momentum"""
        # Get parameters
        lookback = self.params["lookback_period"]
        smoothing = self.params["smoothing_period"]

        # Calculate momentum (rate of change)
        data['momentum'] = data['close'].pct_change(periods=lookback) * 100

        # Smooth momentum
        data['smooth_momentum'] = data['momentum'].rolling(window=smoothing).mean()

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate signals
        # Buy when momentum turns positive
        buy_signals = (data['smooth_momentum'].shift(1) <= 0) & (data['smooth_momentum'] > 0)

        # Sell when momentum turns negative
        sell_signals = (data['smooth_momentum'].shift(1) >= 0) & (data['smooth_momentum'] < 0)

        # Apply signals
        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals


class MACDMomentum(Strategy):
    def __init__(self, fast_period=12, slow_period=26, signal_period=9, hist_threshold=0):
        """
        Initialize MACD Momentum strategy

        Args:
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period
            hist_threshold (float): MACD histogram threshold for signals
        """
        super().__init__(
            name="MACDMomentum",
            params={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
                "hist_threshold": hist_threshold
            }
        )

    def generate_signals(self, data):
        """Generate signals based on MACD momentum"""
        # Check if MACD is already calculated
        if not all(col in data.columns for col in ['macd', 'macd_signal', 'macd_diff']):
            self.logger.warning("MACD indicators not found in data")
            return pd.Series(0, index=data.index)

        # Get histogram threshold
        hist_threshold = self.params["hist_threshold"]

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate signals
        # Buy when MACD crosses above signal line and histogram exceeds threshold
        buy_signals = (
                (data['macd'].shift(1) <= data['macd_signal'].shift(1)) &
                (data['macd'] > data['macd_signal']) &
                (data['macd_diff'] > hist_threshold)
        )

        # Sell when MACD crosses below signal line and histogram is below negative threshold
        sell_signals = (
                (data['macd'].shift(1) >= data['macd_signal'].shift(1)) &
                (data['macd'] < data['macd_signal']) &
                (data['macd_diff'] < -hist_threshold)
        )

        # Apply signals
        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals


class PriceMomentumOscillator(Strategy):
    def __init__(self, lookback_period=14, overbought=80, oversold=20, signal_period=9):
        """
        Initialize Price Momentum Oscillator strategy

        Args:
            lookback_period (int): Lookback period for PMO
            overbought (int): Overbought threshold
            oversold (int): Oversold threshold
            signal_period (int): Signal line period
        """
        super().__init__(
            name="PriceMomentumOscillator",
            params={
                "lookback_period": lookback_period,
                "overbought": overbought,
                "oversold": oversold,
                "signal_period": signal_period
            }
        )

    def _calculate_pmo(self, data, lookback, signal_period):
        """Calculate Price Momentum Oscillator (PMO)"""
        # First smoothing - 10% of lookback
        first_smooth = int(lookback * 0.1)
        # Second smoothing - 30% of lookback
        second_smooth = int(lookback * 0.3)

        # Calculate Rate of Change
        data['roc'] = data['close'].pct_change(periods=1) * 100

        # First smoothing of ROC
        data['roc_ema'] = data['roc'].ewm(span=first_smooth, adjust=False).mean()

        # Second smoothing for PMO line
        data['pmo'] = data['roc_ema'].ewm(span=second_smooth, adjust=False).mean()

        # Signal line - EMA of PMO
        data['pmo_signal'] = data['pmo'].ewm(span=signal_period, adjust=False).mean()

        # PMO histogram
        data['pmo_hist'] = data['pmo'] - data['pmo_signal']

        return data

    def generate_signals(self, data):
        """Generate signals based on Price Momentum Oscillator"""
        # Get parameters
        lookback = self.params["lookback_period"]
        overbought = self.params["overbought"]
        oversold = self.params["oversold"]
        signal_period = self.params["signal_period"]

        # Calculate PMO if not already in data
        if 'pmo' not in data.columns:
            data = self._calculate_pmo(data, lookback, signal_period)

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate signals
        # Buy when PMO crosses above signal line from below oversold
        buy_signals = (
                (data['pmo'].shift(1) <= data['pmo_signal'].shift(1)) &
                (data['pmo'] > data['pmo_signal']) &
                (data['pmo'] < oversold)
        )

        # Sell when PMO crosses below signal line from above overbought
        sell_signals = (
                (data['pmo'].shift(1) >= data['pmo_signal'].shift(1)) &
                (data['pmo'] < data['pmo_signal']) &
                (data['pmo'] > overbought)
        )

        # Apply signals
        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals


class PPOMomentum(Strategy):
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        """
        Initialize PPO (Percentage Price Oscillator) Momentum strategy

        Args:
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period
        """
        super().__init__(
            name="PPOMomentum",
            params={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period
            }
        )

    def _calculate_ppo(self, data, fast_period, slow_period, signal_period):
        """Calculate Percentage Price Oscillator (PPO)"""
        # Calculate fast and slow EMAs
        fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()

        # Calculate PPO line
        data['ppo'] = ((fast_ema - slow_ema) / slow_ema) * 100

        # Calculate signal line
        data['ppo_signal'] = data['ppo'].ewm(span=signal_period, adjust=False).mean()

        # Calculate histogram
        data['ppo_hist'] = data['ppo'] - data['ppo_signal']

        return data

    def generate_signals(self, data):
        """Generate signals based on Percentage Price Oscillator"""
        # Get parameters
        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]
        signal_period = self.params["signal_period"]

        # Calculate PPO if not already in data
        if 'ppo' not in data.columns:
            data = self._calculate_ppo(data, fast_period, slow_period, signal_period)

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate signals
        # Buy when PPO crosses above signal line
        buy_signals = (
                (data['ppo'].shift(1) <= data['ppo_signal'].shift(1)) &
                (data['ppo'] > data['ppo_signal'])
        )

        # Sell when PPO crosses below signal line
        sell_signals = (
                (data['ppo'].shift(1) >= data['ppo_signal'].shift(1)) &
                (data['ppo'] < data['ppo_signal'])
        )

        # Apply signals
        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals