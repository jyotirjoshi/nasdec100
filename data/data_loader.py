"""
Module for loading historical trading data
"""

import os
import pandas as pd
import logging
from datetime import datetime, timedelta


class DataLoader:
    def __init__(self, data_path):
        """
        Initialize DataLoader with path to historical data file

        Args:
            data_path (str): Path to the CSV file containing historical data
        """
        self.data_path = data_path
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """
        Load historical trading data from CSV file

        Returns:
            pandas.DataFrame: Historical data with datetime index
        """
        self.logger.info(f"Loading data from {self.data_path}")

        try:
            # Load data from CSV
            df = pd.read_csv(self.data_path)

            # Convert time column to datetime if it exists
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)

            self.logger.info(f"Loaded {len(df)} data points from {df.index.min()} to {df.index.max()}")

            # Ensure the necessary columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Required column {col} missing from data")
                    raise ValueError(f"Required column {col} missing from data")

            # Sort by time
            df = df.sort_index()

            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                self.logger.warning(f"Data contains missing values: {missing_values[missing_values > 0].to_dict()}")

            # Check for duplicates
            duplicates = df.index.duplicated().sum()
            if duplicates > 0:
                self.logger.warning(f"Data contains {duplicates} duplicate timestamps, removing duplicates")
                df = df[~df.index.duplicated(keep='first')]

            return df

        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise

    def split_data(self, data, train_ratio=0.7, validation_ratio=0.15):
        """
        Split data into training, validation, and test sets

        Args:
            data (pandas.DataFrame): Historical data
            train_ratio (float): Ratio of data to use for training
            validation_ratio (float): Ratio of data to use for validation

        Returns:
            tuple: (train_data, validation_data, test_data)
        """
        data_sorted = data.sort_index()
        n = len(data_sorted)

        train_end_idx = int(n * train_ratio)
        val_end_idx = int(n * (train_ratio + validation_ratio))

        train_data = data_sorted.iloc[:train_end_idx]
        validation_data = data_sorted.iloc[train_end_idx:val_end_idx]
        test_data = data_sorted.iloc[val_end_idx:]

        self.logger.info(
            f"Data split - Train: {len(train_data)}, Validation: {len(validation_data)}, Test: {len(test_data)}")

        return train_data, validation_data, test_data

    def resample_data(self, data, timeframe):
        """
        Resample data to a different timeframe

        Args:
            data (pandas.DataFrame): Historical data
            timeframe (str): Target timeframe (e.g., '15min', '1H', '1D')

        Returns:
            pandas.DataFrame: Resampled data
        """
        resampled = data.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Remove any NaN values that might have been created during resampling
        resampled = resampled.dropna()

        self.logger.info(f"Resampled data from {data.index[0]} to {data.index[-1]} with timeframe {timeframe}")

        return resampled