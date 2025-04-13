"""
Data preprocessing and feature engineering for trading data
"""

import pandas as pd
import numpy as np
import logging
import ta  # Technical Analysis library
import pandas_ta as pta  # Extended Technical Analysis


class DataProcessor:
    def __init__(self, window_size=20, feature_list=None):
        """
        Initialize DataProcessor

        Args:
            window_size (int): Size of lookback window for features
            feature_list (list): List of features to include, None for all
        """
        self.window_size = window_size
        self.feature_list = feature_list
        self.logger = logging.getLogger(__name__)
        self.feature_stats = {}  # To store statistics for normalization

    def process_data(self, df):
        """
        Process raw market data, calculate technical indicators and normalize

        Args:
            df (pandas.DataFrame): Raw market data with OHLCV columns

        Returns:
            pandas.DataFrame: Processed data with additional features
        """
        if df is None or df.empty:
            self.logger.warning("Empty DataFrame provided for processing")
            return None

        # Make a copy to avoid modifying the original
        data = df.copy()

        # Add basic price features
        self._add_price_features(data)

        # Add technical indicators
        self._add_technical_indicators(data)

        # Add volatility indicators
        self._add_volatility_indicators(data)

        # Add volume indicators
        self._add_volume_indicators(data)

        # Add advanced features
        self._add_advanced_features(data)

        # Drop rows with NaN values (usually at the beginning due to indicators requiring lookback)
        data = data.dropna()

        # Normalize features
        data = self._normalize_features(data)

        self.logger.info(f"Processed data, shape: {data.shape}, features: {list(data.columns)}")

        return data

    def create_state_representation(self, data, lookback=10):
        """
        Create a state representation for RL agent from processed data

        Args:
            data (pandas.DataFrame): Processed market data
            lookback (int): Number of time steps to include in state

        Returns:
            numpy.ndarray: State representation for RL agent
        """
        if data is None or data.empty or len(data) < lookback:
            self.logger.warning(f"Insufficient data for state representation, need {lookback} rows")
            return None

        # Get the most recent data
        recent_data = data.iloc[-lookback:]

        # Stack all features into a 2D array (time x features)
        state = recent_data.values

        return state

    def _add_price_features(self, data):
        """Add basic price-based features"""
        # Log returns
        data['log_return'] = np.log(data['close'] / data['close'].shift(1))

        # Price differences
        data['price_diff'] = data['close'] - data['open']

        # Candle properties
        data['candle_size'] = data['high'] - data['low']
        data['body_size'] = abs(data['close'] - data['open'])
        data['body_ratio'] = data['body_size'] / data['candle_size']

        # Upper and lower shadows
        data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
        data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']

        # Previous candles
        for i in range(1, 6):
            data[f'close_lag_{i}'] = data['close'].shift(i)
            data[f'return_lag_{i}'] = data['log_return'].shift(i)

    def _add_technical_indicators(self, data):
        """Add technical indicators"""
        # Moving Averages
        for window in [5, 9, 21, 50, 200]:
            data[f'sma_{window}'] = ta.trend.sma_indicator(data['close'], window=window)
            data[f'ema_{window}'] = ta.trend.ema_indicator(data['close'], window=window)

        # MACD
        macd = ta.trend.MACD(data['close'], window_slow=26, window_fast=12, window_sign=9)
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_diff'] = macd.macd_diff()

        # RSI
        for window in [6, 14, 21]:
            data[f'rsi_{window}'] = ta.momentum.RSIIndicator(data['close'], window=window).rsi()

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'], window=14, smooth_window=3)
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['close'], window=20, window_dev=2)
        data['bb_high'] = bollinger.bollinger_hband()
        data['bb_mid'] = bollinger.bollinger_mavg()
        data['bb_low'] = bollinger.bollinger_lband()
        data['bb_width'] = (data['bb_high'] - data['bb_low']) / data['bb_mid']
        data['bb_pct'] = (data['close'] - data['bb_low']) / (data['bb_high'] - data['bb_low'])

        # Ichimoku Cloud
        try:
            ichimoku = ta.trend.IchimokuIndicator(data['high'], data['low'], window1=9, window2=26, window3=52)
            data['ichimoku_a'] = ichimoku.ichimoku_a()
            data['ichimoku_b'] = ichimoku.ichimoku_b()
            data['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
            data['ichimoku_base'] = ichimoku.ichimoku_base_line()
        except Exception as e:
            self.logger.warning(f"Could not calculate Ichimoku Cloud: {str(e)}")

    def _add_volatility_indicators(self, data):
        """Add volatility indicators"""
        # Average True Range (ATR)
        for window in [7, 14, 21]:
            data[f'atr_{window}'] = ta.volatility.AverageTrueRange(
                data['high'], data['low'], data['close'], window=window
            ).average_true_range()

        # Historical Volatility
        for window in [10, 21, 63]:
            data[f'hist_vol_{window}'] = data['log_return'].rolling(window=window).std() * np.sqrt(252)

        # Donchian Channels
        for window in [20, 55]:
            data[f'donchian_high_{window}'] = data['high'].rolling(window=window).max()
            data[f'donchian_low_{window}'] = data['low'].rolling(window=window).min()
            data[f'donchian_mid_{window}'] = (data[f'donchian_high_{window}'] + data[f'donchian_low_{window}']) / 2

        # Keltner Channels
        for window in [20]:
            try:
                keltner = ta.volatility.KeltnerChannel(
                    data['high'], data['low'], data['close'], window=window, window_atr=window
                )
                data[f'keltner_high_{window}'] = keltner.keltner_channel_hband()
                data[f'keltner_mid_{window}'] = keltner.keltner_channel_mband()
                data[f'keltner_low_{window}'] = keltner.keltner_channel_lband()
            except Exception as e:
                self.logger.warning(f"Could not calculate Keltner Channels: {str(e)}")

    def _add_volume_indicators(self, data):
        """Add volume-based indicators"""
        # On-Balance Volume (OBV)
        data['obv'] = ta.volume.OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()

        # Volume-Weighted Average Price (VWAP) - Daily calculation
        try:
            # Calculate intraday VWAP
            data['vwap'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data[
                'volume'].cumsum()
        except Exception as e:
            self.logger.warning(f"Could not calculate VWAP: {str(e)}")

        # Chaikin Money Flow
        data['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
            data['high'], data['low'], data['close'], data['volume'], window=20
        ).chaikin_money_flow()

        # Volume-weighted RSI
        try:
            data['volume_rsi'] = pta.vrsi(data['close'], data['volume'], length=14)
        except Exception as e:
            self.logger.warning(f"Could not calculate Volume RSI: {str(e)}")

        # Ease of Movement
        data['eom'] = ta.volume.EaseOfMovementIndicator(
            data['high'], data['low'], data['volume']
        ).ease_of_movement()

        # Volume Rate of Change
        data['volume_roc'] = data['volume'].pct_change(periods=1)

        # Volume moving averages
        for window in [10, 20, 50]:
            data[f'volume_sma_{window}'] = data['volume'].rolling(window=window).mean()
            data[f'vol_ratio_{window}'] = data['volume'] / data[f'volume_sma_{window}']

    def _add_advanced_features(self, data):
        """Add more advanced and derived features"""
        # Price momentum indicators
        for window in [3, 5, 10, 21]:
            data[f'momentum_{window}'] = data['close'] / data['close'].shift(window) - 1

        # Directional Movement Index (DMI)
        for window in [14]:
            try:
                adx = ta.trend.ADXIndicator(data['high'], data['low'], data['close'], window=window)
                data[f'adx_{window}'] = adx.adx()
                data[f'dip_{window}'] = adx.adx_pos()
                data[f'din_{window}'] = adx.adx_neg()
            except Exception as e:
                self.logger.warning(f"Could not calculate ADX: {str(e)}")

        # Fibonacci ratios - distance from current price to key Fibonacci levels from recent high/low
        try:
            window = 100  # Look back for high/low
            recent_high = data['high'].rolling(window=window).max()
            recent_low = data['low'].rolling(window=window).min()
            range_size = recent_high - recent_low

            # Key Fibonacci retracement levels
            for level, name in [(0.236, '236'), (0.382, '382'), (0.5, '50'), (0.618, '618'), (0.786, '786')]:
                retracement_level = recent_high - (range_size * level)
                data[f'fib_{name}_dist'] = (data['close'] - retracement_level) / data['close']
        except Exception as e:
            self.logger.warning(f"Could not calculate Fibonacci levels: {str(e)}")

        # Custom indicators
        try:
            # TrendScore - combination of multiple trend indicators
            data['trend_score'] = (
                                          (data['close'] > data['sma_50']).astype(int) +
                                          (data['sma_9'] > data['sma_21']).astype(int) +
                                          (data['macd'] > data['macd_signal']).astype(int) +
                                          (data['rsi_14'] > 50).astype(int)
                                  ) / 4.0

            # Volatility ratio - current vs historical
            data['vol_ratio'] = data['atr_7'] / data['atr_21']

            # Support/Resistance Proximity
            data['sup_res_zone'] = (
                                           (data['close'] - data['bb_low']).abs() < 0.2 * data['bb_width']
                                   ).astype(int) + (
                                           (data['close'] - data['bb_high']).abs() < 0.2 * data['bb_width']
                                   ).astype(int)

        except Exception as e:
            self.logger.warning(f"Could not calculate custom indicators: {str(e)}")

    def _normalize_features(self, data):
        """Normalize features using z-score scaling"""
        # List of columns to normalize (skip date and boolean columns)
        skip_cols = ['time', 'date']
        numeric_cols = [col for col in data.columns if col not in skip_cols]

        # Store mean and std for future use on new data
        if not self.feature_stats:
            self.feature_stats = {
                'mean': data[numeric_cols].mean(),
                'std': data[numeric_cols].std()
            }

        # Replace 0 std with 1 to avoid division by zero
        std = self.feature_stats['std'].copy()
        std[std == 0] = 1

        # Apply normalization
        for col in numeric_cols:
            try:
                data[col] = (data[col] - self.feature_stats['mean'][col]) / std[col]
            except KeyError:
                # Skip if column not in stats (new feature)
                data[col] = (data[col] - data[col].mean()) / (data[col].std() or 1)

        return data