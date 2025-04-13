"""
Trend following strategies implementation
"""

import numpy as np
import pandas as pd
from strategies.strategy_base import Strategy


class MovingAverageCrossover(Strategy):
    def __init__(self, fast_period=9, slow_period=21):
        """
        Initialize Moving Average Crossover strategy

        Args:
            fast_period (int): Fast moving average period
            slow_period (int): Slow moving average period
        """
        super().__init__(
            name="MA_Crossover",
            params={
                "fast_period": fast_period,
                "slow_period": slow_period
            }
        )

    def generate_signals(self, data):
        """Generate buy/sell signals based on moving average crossovers"""
        # Get parameters
        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]

        # Make sure necessary columns exist
        if f'sma_{fast_period}' not in data.columns or f'sma_{slow_period}' not in data.columns:
            self.logger.warning(f"Required SMA columns not found, calculating them now")
            data[f'sma_{fast_period}'] = data['close'].rolling(window=fast_period).mean()
            data[f'sma_{slow_period}'] = data['close'].rolling(window=slow_period).mean()

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Calculate crossover signals
        fast_ma = data[f'sma_{fast_period}']
        slow_ma = data[f'sma_{slow_period}']

        # Buy signal: fast MA crosses above slow MA
        buy_signals = (fast_ma.shift(1) <= slow_ma.shift(1)) & (fast_ma > slow_ma)

        # Sell signal: fast MA crosses below slow MA
        sell_signals = (fast_ma.shift(1) >= slow_ma.shift(1)) & (fast_ma < slow_ma)

        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals


class MACDStrategy(Strategy):
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        """
        Initialize MACD strategy

        Args:
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period
        """
        super().__init__(
            name="MACD",
            params={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period
            }
        )

    def generate_signals(self, data):
        """Generate buy/sell signals based on MACD crossovers"""
        # Check if MACD is already calculated
        if 'macd' not in data.columns or 'macd_signal' not in data.columns:
            # Calculate MACD indicators
            fast_period = self.params["fast_period"]
            slow_period = self.params["slow_period"]
            signal_period = self.params["signal_period"]

            # Calculate MACD line
            data['macd'] = data['close'].ewm(span=fast_period, adjust=False).mean() - \
                           data['close'].ewm(span=slow_period, adjust=False).mean()

            # Calculate signal line
            data['macd_signal'] = data['macd'].ewm(span=signal_period, adjust=False).mean()

            # Calculate MACD histogram
            data['macd_hist'] = data['macd'] - data['macd_signal']

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Buy signal: MACD crosses above signal line
        buy_signals = (data['macd'].shift(1) <= data['macd_signal'].shift(1)) & \
                      (data['macd'] > data['macd_signal'])

        # Sell signal: MACD crosses below signal line
        sell_signals = (data['macd'].shift(1) >= data['macd_signal'].shift(1)) & \
                       (data['macd'] < data['macd_signal'])

        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals


class ParabolicSAR(Strategy):
    def __init__(self, af_start=0.02, af_increment=0.02, af_max=0.2):
        """
        Initialize Parabolic SAR strategy

        Args:
            af_start (float): Starting acceleration factor
            af_increment (float): Acceleration factor increment
            af_max (float): Maximum acceleration factor
        """
        super().__init__(
            name="ParabolicSAR",
            params={
                "af_start": af_start,
                "af_increment": af_increment,
                "af_max": af_max
            }
        )

    def _calculate_psar(self, high, low, close):
        """Calculate Parabolic SAR values"""
        af_start = self.params["af_start"]
        af_increment = self.params["af_increment"]
        af_max = self.params["af_max"]

        n = len(high)
        psar = close.copy()
        bull = True  # Current trend direction (True=Bull, False=Bear)
        af = af_start
        ep = low[0]  # Extreme point
        hp = high.copy()  # High points
        lp = low.copy()  # Low points

        # Initialize first SAR value
        psar[0] = close[0]

        for i in range(1, n):
            # Calculate SAR
            if bull:
                psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
                # Ensure SAR is below the prior two lows
                psar[i] = min(psar[i], lp[i - 1], lp[i - 2] if i > 1 else lp[i - 1])

                # Check if trend reversal
                if psar[i] > low[i]:
                    bull = False
                    psar[i] = ep
                    ep = low[i]
                    af = af_start
                else:
                    # Update EP and AF if new high
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_increment, af_max)
            else:
                # Bear trend
                psar[i] = psar[i - 1] - af * (psar[i - 1] - ep)
                # Ensure SAR is above the prior two highs
                psar[i] = max(psar[i], hp[i - 1], hp[i - 2] if i > 1 else hp[i - 1])

                # Check if trend reversal
                if psar[i] < high[i]:
                    bull = True
                    psar[i] = ep
                    ep = high[i]
                    af = af_start
                else:
                    # Update EP and AF if new low
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_increment, af_max)

        return psar

    def generate_signals(self, data):
        """Generate buy/sell signals based on Parabolic SAR crossovers"""
        # Calculate Parabolic SAR if not already in data
        if 'psar' not in data.columns:
            data['psar'] = self._calculate_psar(data['high'], data['low'], data['close'])

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate signals based on price crossing PSAR
        buy_signals = (data['close'].shift(1) <= data['psar'].shift(1)) & \
                      (data['close'] > data['psar'])

        sell_signals = (data['close'].shift(1) >= data['psar'].shift(1)) & \
                       (data['close'] < data['psar'])

        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals


class ADXTrendFollower(Strategy):
    def __init__(self, adx_period=14, adx_threshold=25, di_period=14):
        """
        Initialize ADX Trend Following strategy

        Args:
            adx_period (int): ADX period
            adx_threshold (int): ADX threshold for trend strength
            di_period (int): DI+/DI- period
        """
        super().__init__(
            name="ADX_Trend",
            params={
                "adx_period": adx_period,
                "adx_threshold": adx_threshold,
                "di_period": di_period
            }
        )

    def generate_signals(self, data):
        """Generate buy/sell signals based on ADX and DI crossovers"""
        adx_period = self.params["adx_period"]
        adx_threshold = self.params["adx_threshold"]
        di_period = self.params["di_period"]

        # Check if ADX and DI are already calculated
        adx_col = f'adx_{adx_period}'
        dip_col = f'dip_{di_period}'
        din_col = f'din_{di_period}'

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate signals based on ADX and DI
        strong_trend = data[adx_col] > adx_threshold
        di_buy = (data[dip_col].shift(1) <= data[din_col].shift(1)) & \
                 (data[dip_col] > data[din_col])
        di_sell = (data[dip_col].shift(1) >= data[din_col].shift(1)) & \
                  (data[dip_col] < data[din_col])

        # Only generate signals when ADX indicates strong trend
        signals[strong_trend & di_buy] = 1
        signals[strong_trend & di_sell] = -1

        return signals


class IchimokuStrategy(Strategy):
    def __init__(self, tenkan_period=9, kijun_period=26, senkou_span_b_period=52):
        """
        Initialize Ichimoku Cloud strategy

        Args:
            tenkan_period (int): Tenkan-sen (Conversion Line) period
            kijun_period (int): Kijun-sen (Base Line) period
            senkou_span_b_period (int): Senkou Span B (Leading Span B) period
        """
        super().__init__(
            name="Ichimoku",
            params={
                "tenkan_period": tenkan_period,
                "kijun_period": kijun_period,
                "senkou_span_b_period": senkou_span_b_period
            }
        )

    def generate_signals(self, data):
        """Generate buy/sell signals based on Ichimoku Cloud"""
        # Use pre-calculated Ichimoku components if available
        ichimoku_cols = ['ichimoku_conv', 'ichimoku_base', 'ichimoku_a', 'ichimoku_b']
        if not all(col in data.columns for col in ichimoku_cols):
            self.logger.warning("Ichimoku indicators not found in data")
            return pd.Series(0, index=data.index)

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # TK Cross - Tenkan crosses above Kijun
        tk_buy = (data['ichimoku_conv'].shift(1) <= data['ichimoku_base'].shift(1)) & \
                 (data['ichimoku_conv'] > data['ichimoku_base'])

        tk_sell = (data['ichimoku_conv'].shift(1) >= data['ichimoku_base'].shift(1)) & \
                  (data['ichimoku_conv'] < data['ichimoku_base'])

        # Price relative to cloud
        price_above_cloud = (data['close'] > data['ichimoku_a']) & \
                            (data['close'] > data['ichimoku_b'])

        price_below_cloud = (data['close'] < data['ichimoku_a']) & \
                            (data['close'] < data['ichimoku_b'])

        # Generate signals - Buy when TK cross and price above cloud
        signals[tk_buy & price_above_cloud] = 1

        # Sell when TK cross and price below cloud
        signals[tk_sell & price_below_cloud] = -1

        return signals