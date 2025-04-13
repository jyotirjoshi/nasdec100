"""
Advanced feature engineering for trading data
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
import logging
import talib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    def __init__(self):
        """Initialize FeatureEngineer"""
        self.logger = logging.getLogger(__name__)

    def extract_features(self, data, add_talib=True, add_patterns=True,
                         add_cycles=True, add_ml_features=True):
        """
        Extract advanced features from OHLCV data

        Args:
            data (pandas.DataFrame): DataFrame with OHLCV data
            add_talib (bool): Whether to add TA-Lib indicators
            add_patterns (bool): Whether to add candlestick patterns
            add_cycles (bool): Whether to add cyclical features
            add_ml_features (bool): Whether to add ML-derived features

        Returns:
            pandas.DataFrame: DataFrame with added features
        """
        # Make a copy to avoid modifying the original
        df = data.copy()

        # Add TA-Lib indicators
        if add_talib:
            df = self._add_talib_indicators(df)

        # Add candlestick patterns
        if add_patterns:
            df = self._add_candlestick_patterns(df)

        # Add cyclical features
        if add_cycles:
            df = self._add_cyclical_features(df)

        # Add machine learning features
        if add_ml_features:
            df = self._add_ml_features(df)

        return df

    def _add_talib_indicators(self, df):
        """Add technical indicators using TA-Lib"""
        try:
            # Make sure necessary columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.warning(f"Missing required columns for TA-Lib indicators")
                return df

            # Overlap Studies
            df['bband_upper'], df['bband_middle'], df['bband_lower'] = talib.BBANDS(
                df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
            )

            # Momentum Indicators
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
            df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
            df['willr'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
            df['ultosc'] = talib.ULTOSC(df['high'], df['low'], df['close'],
                                        timeperiod1=7, timeperiod2=14, timeperiod3=28)

            # Volume Indicators
            df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
            df['adosc'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'],
                                      fastperiod=3, slowperiod=10)

            # Volatility Indicators
            df['natr'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
            df['trange'] = talib.TRANGE(df['high'], df['low'], df['close'])

            # Cycle Indicators
            df['ht_dcperiod'] = talib.HT_DCPERIOD(df['close'])
            df['ht_dcphase'] = talib.HT_DCPHASE(df['close'])
            df['sine'], df['leadsine'] = talib.HT_SINE(df['close'])

            # Statistic Functions
            df['linear_reg'] = talib.LINEARREG(df['close'], timeperiod=14)
            df['linear_reg_angle'] = talib.LINEARREG_ANGLE(df['close'], timeperiod=14)
            df['linear_reg_slope'] = talib.LINEARREG_SLOPE(df['close'], timeperiod=14)

            # Price Transform
            df['wclprice'] = talib.WCLPRICE(df['high'], df['low'], df['close'])

        except Exception as e:
            self.logger.error(f"Error adding TA-Lib indicators: {str(e)}")

        return df

    def _add_candlestick_patterns(self, df):
        """Add candlestick pattern recognition"""
        try:
            # Basic patterns
            pattern_funcs = {
                'doji': talib.CDLDOJI,
                'engulfing': talib.CDLENGULFING,
                'hammer': talib.CDLHAMMER,
                'hanging_man': talib.CDLHANGINGMAN,
                'shooting_star': talib.CDLSHOOTINGSTAR,
                'morning_star': talib.CDLMORNINGSTAR,
                'evening_star': talib.CDLEVENINGSTAR,
                'three_white_soldiers': talib.CDL3WHITESOLDIERS,
                'three_black_crows': talib.CDL3BLACKCROWS
            }

            for name, func in pattern_funcs.items():
                df[f'pattern_{name}'] = func(
                    df['open'], df['high'], df['low'], df['close']
                )

            # Aggregate pattern signal
            pattern_cols = [col for col in df.columns if col.startswith('pattern_')]
            if pattern_cols:
                # Normalize pattern values to -1, 0, 1
                for col in pattern_cols:
                    df[col] = np.sign(df[col])

                # Calculate bullish pattern count
                df['bullish_patterns'] = df[pattern_cols].apply(
                    lambda x: (x > 0).sum(), axis=1
                )

                # Calculate bearish pattern count
                df['bearish_patterns'] = df[pattern_cols].apply(
                    lambda x: (x < 0).sum(), axis=1
                )

                # Calculate overall pattern signal
                df['pattern_signal'] = df['bullish_patterns'] - df['bearish_patterns']

        except Exception as e:
            self.logger.error(f"Error adding candlestick patterns: {str(e)}")

        return df

    def _add_cyclical_features(self, df):
        """Add cyclical time features"""
        try:
            # Make sure the index is a datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.warning("DataFrame index is not DatetimeIndex, skipping cyclical features")
                return df

            # Time components
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['week_of_year'] = df.index.isocalendar().week
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter

            # Encode cyclical features using sine and cosine transformations
            # This preserves the cyclical nature of time variables

            # Hour of day (0-23)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

            # Day of week (0-6)
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)

            # Day of month (1-31)
            df['day_of_month_sin'] = np.sin(2 * np.pi * (df['day_of_month'] - 1) / 31.0)
            df['day_of_month_cos'] = np.cos(2 * np.pi * (df['day_of_month'] - 1) / 31.0)

            # Month (1-12)
            df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12.0)
            df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12.0)

            # Market session flags
            df['is_us_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16) &
                                       (df['day_of_week'] < 5)).astype(int)
            df['is_asia_market_open'] = ((df['hour'] >= 19) | (df['hour'] < 2) &
                                         (df['day_of_week'] < 5)).astype(int)
            df['is_europe_market_open'] = ((df['hour'] >= 3) & (df['hour'] < 11) &
                                           (df['day_of_week'] < 5)).astype(int)

            # Drop the raw cyclical columns
            df = df.drop(['hour', 'day_of_week', 'day_of_month', 'week_of_year',
                          'month', 'quarter'], axis=1)

        except Exception as e:
            self.logger.error(f"Error adding cyclical features: {str(e)}")

        return df

    def _add_ml_features(self, df):
        """Add machine learning derived features"""
        try:
            # Make sure we have enough data
            if len(df) < 50:
                self.logger.warning("Not enough data for ML features")
                return df

            # Moving window PCA on price and volume features
            window_sizes = [20, 50]
            feature_sets = {
                'price': ['open', 'high', 'low', 'close'],
                'volume': ['volume']
            }

            for window in window_sizes:
                for feature_set_name, features in feature_sets.items():
                    # Skip if not all features are available
                    if not all(f in df.columns for f in features):
                        continue

                    # Apply rolling PCA for dimensionality reduction
                    df[f'pca_{feature_set_name}_{window}'] = self._rolling_pca(
                        df[features], window=window
                    )

            # Autocorrelation features
            for lag in [1, 5, 10]:
                df[f'close_autocorr_{lag}'] = df['close'].autocorr(lag=lag)

            # Spectral features - detect cycles in price data
            if len(df) >= 128:  # Need significant data for FFT
                # Get close price
                close_price = df['close'].values

                # Apply FFT
                fft_values = np.fft.fft(close_price - np.mean(close_price))
                fft_abs = np.abs(fft_values)

                # Get dominant frequency
                dominant_freq_idx = np.argmax(fft_abs[1:len(fft_abs) // 2]) + 1

                # Calculate cycle period
                cycle_period = len(close_price) / dominant_freq_idx if dominant_freq_idx > 0 else len(close_price)

                # Add as feature
                df['dominant_cycle_period'] = cycle_period

                # Add amplitude feature
                df['dominant_cycle_amplitude'] = fft_abs[dominant_freq_idx] / len(
                    close_price) if dominant_freq_idx > 0 else 0

                # Add phase of current point in dominant cycle
                df['cycle_phase'] = np.remainder(np.arange(len(df)) / cycle_period * 2 * np.pi, 2 * np.pi)
                df['cycle_phase_sin'] = np.sin(df['cycle_phase'])
                df['cycle_phase_cos'] = np.cos(df['cycle_phase'])

            # Non-linear transformations and ratios
            for col in ['close', 'volume']:
                if col in df.columns:
                    # Log transforms
                    df[f'log_{col}'] = np.log1p(df[col])

                    # Power transforms
                    df[f'sqrt_{col}'] = np.sqrt(np.abs(df[col]))
                    df[f'square_{col}'] = df[col] ** 2

            # Lomb-Scargle periodogram (for non-uniformly sampled data)
            # This feature is useful for detecting cycles in noisy data
            try:
                from scipy.signal import lombscargle

                close_price = df['close'].values
                x = np.arange(len(close_price))
                f = np.linspace(0.01, 0.5, 100)  # frequencies to evaluate

                pgram = lombscargle(x, close_price - np.mean(close_price), 2 * np.pi * f)

                # Get the top frequency
                top_freq_idx = np.argmax(pgram)
                top_freq = f[top_freq_idx]

                # Add periodic features
                df['ls_top_cycle_freq'] = top_freq
                df['ls_top_cycle_period'] = 1.0 / top_freq if top_freq > 0 else 999
                df['ls_top_cycle_power'] = pgram[top_freq_idx]

            except Exception as e:
                self.logger.warning(f"Could not compute Lomb-Scargle periodogram: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error adding ML features: {str(e)}")

        return df

    def _rolling_pca(self, data, window=20, n_components=1):
        """Apply PCA in a rolling window to extract temporal structure"""
        values = np.zeros(len(data))
        values[:] = np.nan

        for i in range(window, len(data)):
            window_data = data.iloc[i - window:i].values

            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(window_data)

            # Apply PCA
            pca = PCA(n_components=min(n_components, window_data.shape[1]))
            pca_result = pca.fit_transform(scaled_data)

            # Use the last value of the first principal component
            values[i] = pca_result[-1, 0]

        return values