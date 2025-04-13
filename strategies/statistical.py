"""
Statistical strategy implementations
"""

import numpy as np
import pandas as pd
from strategies.strategy_base import Strategy
from scipy.stats import linregress


class LinearRegressionChannel(Strategy):
    def __init__(self, period=20, std_dev=2.0):
        """
        Initialize Linear Regression Channel strategy

        Args:
            period (int): Period for linear regression calculation
            std_dev (float): Standard deviation for channel width
        """
        super().__init__(
            name="LinearRegressionChannel",
            params={
                "period": period,
                "std_dev": std_dev
            }
        )

    def _calculate_regression_channel(self, data, period, std_dev):
        """Calculate linear regression channel"""
        y = data['close'].values
        x = np.arange(len(y))

        # Initialize result arrays
        regression_line = np.zeros_like(y)
        upper_channel = np.zeros_like(y)
        lower_channel = np.zeros_like(y)

        # Calculate regression for each window
        for i in range(period, len(y)):
            # Get window data
            window_y = y[i - period:i]
            window_x = np.arange(period)

            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = linregress(window_x, window_y)

            # Calculate regression line
            y_pred = intercept + slope * window_x

            # Calculate standard error
            std_error = np.sqrt(np.sum((window_y - y_pred) ** 2) / (period - 2))

            # Calculate regression line and channels for current point
            regression_line[i] = intercept + slope * period
            upper_channel[i] = regression_line[i] + std_dev * std_error
            lower_channel[i] = regression_line[i] - std_dev * std_error

        return regression_line, upper_channel, lower_channel

    def generate_signals(self, data):
        """Generate signals based on Linear Regression Channel"""
        # Get parameters
        period = self.params["period"]
        std_dev = self.params["std_dev"]

        # Calculate linear regression channel
        regression, upper, lower = self._calculate_regression_channel(data, period, std_dev)

        # Add calculated channels to data
        data['reg_line'] = regression
        data['reg_upper'] = upper
        data['reg_lower'] = lower

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate signals - Buy when price touches lower channel and starts rising
        buy_signals = (
                (data['close'].shift(1) <= data['reg_lower'].shift(1)) &
                (data['close'] > data['close'].shift(1))
        )

        # Sell when price touches upper channel and starts falling
        sell_signals = (
                (data['close'].shift(1) >= data['reg_upper'].shift(1)) &
                (data['close'] < data['close'].shift(1))
        )

        # Apply signals
        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals


class MeanReversionStatArb(Strategy):
    def __init__(self, lookback=20, entry_z=2.0, exit_z=0.5):
        """
        Initialize Mean Reversion Statistical Arbitrage strategy

        Args:
            lookback (int): Lookback period for calculating z-score
            entry_z (float): Z-score threshold for entry
            exit_z (float): Z-score threshold for exit
        """
        super().__init__(
            name="MeanReversionStatArb",
            params={
                "lookback": lookback,
                "entry_z": entry_z,
                "exit_z": exit_z
            }
        )

    def generate_signals(self, data):
        """Generate signals based on statistical mean reversion"""
        # Get parameters
        lookback = self.params["lookback"]
        entry_z = self.params["entry_z"]
        exit_z = self.params["exit_z"]

        # Calculate rolling mean and standard deviation
        data['rolling_mean'] = data['close'].rolling(window=lookback).mean()
        data['rolling_std'] = data['close'].rolling(window=lookback).std()

        # Calculate z-score
        data['zscore'] = (data['close'] - data['rolling_mean']) / data['rolling_std']

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate signals
        # Buy when z-score is below negative entry threshold
        buy_signals = data['zscore'] < -entry_z

        # Sell when z-score is above positive entry threshold
        sell_signals = data['zscore'] > entry_z

        # Exit long when z-score rises above negative exit threshold
        exit_longs = (data['zscore'].shift(1) < -exit_z) & (data['zscore'] >= -exit_z)

        # Exit short when z-score falls below positive exit threshold
        exit_shorts = (data['zscore'].shift(1) > exit_z) & (data['zscore'] <= exit_z)

        # Apply signals
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        signals[exit_longs] = -1  # Exit long positions
        signals[exit_shorts] = 1  # Exit short positions

        return signals


class KalmanFilterStrategy(Strategy):
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2, entry_threshold=1.0):
        """
        Initialize Kalman Filter strategy

        Args:
            process_variance (float): Process variance for Kalman Filter
            measurement_variance (float): Measurement variance for Kalman Filter
            entry_threshold (float): Threshold for entry signals
        """
        super().__init__(
            name="KalmanFilter",
            params={
                "process_variance": process_variance,
                "measurement_variance": measurement_variance,
                "entry_threshold": entry_threshold
            }
        )

    def _kalman_filter(self, data, process_var, measurement_var):
        """Apply Kalman Filter to price series"""
        # Initialize state and covariance
        x = data.iloc[0]  # Initial state estimate
        p = 1.0  # Initial estimate covariance

        # Initialize result arrays
        filtered = np.zeros(len(data))
        residuals = np.zeros(len(data))

        # Apply Kalman Filter
        for i in range(len(data)):
            # Prediction step
            x_pred = x
            p_pred = p + process_var

            # Update step
            k = p_pred / (p_pred + measurement_var)  # Kalman gain
            x = x_pred + k * (data.iloc[i] - x_pred)
            p = (1 - k) * p_pred

            # Store results
            filtered[i] = x
            residuals[i] = data.iloc[i] - x

        return filtered, residuals

    def generate_signals(self, data):
        """Generate signals based on Kalman Filter"""
        # Get parameters
        process_var = self.params["process_variance"]
        measurement_var = self.params["measurement_variance"]
        entry_threshold = self.params["entry_threshold"]

        # Apply Kalman Filter
        filtered, residuals = self._kalman_filter(data['close'], process_var, measurement_var)

        # Add results to data
        data['kf_filtered'] = filtered
        data['kf_residual'] = residuals

        # Calculate residual z-score
        data['residual_mean'] = data['kf_residual'].rolling(window=20).mean()
        data['residual_std'] = data['kf_residual'].rolling(window=20).std()
        data['residual_zscore'] = (data['kf_residual'] - data['residual_mean']) / data['residual_std']

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate signals
        # Buy when price is significantly below filter estimate (negative residual z-score)
        buy_signals = data['residual_zscore'] < -entry_threshold

        # Sell when price is significantly above filter estimate (positive residual z-score)
        sell_signals = data['residual_zscore'] > entry_threshold

        # Apply signals
        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals


class HurstExponentStrategy(Strategy):
    def __init__(self, lookback=100, hurst_lookback=40, mean_reversion_threshold=0.4, trend_threshold=0.6):
        """
        Initialize Hurst Exponent strategy

        Args:
            lookback (int): Lookback period for calculations
            hurst_lookback (int): Lookback period for Hurst exponent
            mean_reversion_threshold (float): Threshold for mean-reversion regime
            trend_threshold (float): Threshold for trending regime
        """
        super().__init__(
            name="HurstExponent",
            params={
                "lookback": lookback,
                "hurst_lookback": hurst_lookback,
                "mean_reversion_threshold": mean_reversion_threshold,
                "trend_threshold": trend_threshold
            }
        )

    def _calculate_hurst_exponent(self, price_series, max_lag=20):
        """Calculate Hurst Exponent using R/S analysis"""
        lags = range(2, max_lag)
        tau = [np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]

    def generate_signals(self, data):
        """Generate signals based on Hurst Exponent"""
        # Get parameters
        lookback = self.params["lookback"]
        hurst_lookback = self.params["hurst_lookback"]
        mean_reversion_threshold = self.params["mean_reversion_threshold"]
        trend_threshold = self.params["trend_threshold"]

        # Initialize Hurst exponent series
        data['hurst'] = np.nan

        # Calculate Hurst exponent for each window
        for i in range(hurst_lookback, len(data)):
            price_window = data['close'].iloc[i - hurst_lookback:i].values
            data.loc[data.index[i], 'hurst'] = self._calculate_hurst_exponent(price_window)

        # Calculate z-score for mean reversion when appropriate
        data['zscore'] = (data['close'] - data['close'].rolling(window=lookback).mean()) / \
                         data['close'].rolling(window=lookback).std()

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate signals based on regime
        for i in range(lookback, len(data)):
            hurst = data['hurst'].iloc[i]
            zscore = data['zscore'].iloc[i]

            if pd.isna(hurst):
                continue

            if hurst < mean_reversion_threshold:
                # Mean reversion regime
                if zscore < -2.0:
                    signals.iloc[i] = 1  # Buy when oversold
                elif zscore > 2.0:
                    signals.iloc[i] = -1  # Sell when overbought
            elif hurst > trend_threshold:
                # Trending regime - follow 5-period momentum
                momentum = (data['close'].iloc[i] / data['close'].iloc[i - 5] - 1) * 100
                if momentum > 0.5:
                    signals.iloc[i] = 1  # Buy when momentum is positive
                elif momentum < -0.5:
                    signals.iloc[i] = -1  # Sell when momentum is negative

        return signals