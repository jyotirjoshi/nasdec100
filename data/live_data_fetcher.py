"""
Module for fetching live market data from free sources
"""

import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from threading import Thread, Lock, Event
import websocket
import json
import re


class LiveDataFetcher:
    """
    Fetches live market data using multiple free sources
    Implements fallback mechanisms if one source fails
    """

    def __init__(self, symbol="NQ=F", interval="5m", buffer_size=100):
        """
        Initialize the LiveDataFetcher

        Args:
            symbol (str): The ticker symbol to fetch (default: NQ=F for Nasdaq E-mini futures)
            interval (str): Data interval (default: 5m)
            buffer_size (int): Number of data points to keep in buffer
        """
        self.symbol = symbol
        self.interval = interval
        self.buffer_size = buffer_size
        self.logger = logging.getLogger(__name__)
        self.data_buffer = []
        self.buffer_lock = Lock()
        self.stop_event = Event()
        self.data_thread = None
        self.last_update = None
        self.websocket = None
        self.available_sources = [
            self._fetch_from_yfinance,
            self._fetch_from_alphavantage,
            self._fetch_from_twelvedata
        ]

    def start(self):
        """Start fetching live data in a background thread"""
        if self.data_thread is None or not self.data_thread.is_alive():
            self.stop_event.clear()
            self.data_thread = Thread(target=self._data_fetching_loop)
            self.data_thread.daemon = True
            self.data_thread.start()
            self.logger.info("Live data fetching started")

            # Also try to establish websocket connection if possible
            self._try_websocket_connection()

    def stop(self):
        """Stop fetching live data"""
        if self.data_thread and self.data_thread.is_alive():
            self.stop_event.set()
            if self.websocket:
                self.websocket.close()
            self.data_thread.join(timeout=5)
            self.logger.info("Live data fetching stopped")

    def get_latest_data(self, n=1):
        """
        Get the latest n data points

        Args:
            n (int): Number of data points to retrieve

        Returns:
            pandas.DataFrame: Latest market data
        """
        with self.buffer_lock:
            if not self.data_buffer:
                return None

            data = self.data_buffer[-n:] if n < len(self.data_buffer) else self.data_buffer
            df = pd.DataFrame(data)

            if not df.empty:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)

        return df

    def get_historical_and_latest_data(self, days=5):
        """
        Get historical data and append latest live data

        Args:
            days (int): Number of historical days to fetch

        Returns:
            pandas.DataFrame: Combined historical and live data
        """
        # First get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        historical_data = self._fetch_historical_data(start_date, end_date)

        # Get latest live data
        live_data = self.get_latest_data(100)  # Get quite a few points to ensure overlap

        if historical_data is None or live_data is None:
            return historical_data if historical_data is not None else live_data

        # Combine, making sure to avoid duplicates
        combined = pd.concat([historical_data, live_data])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()

        return combined

    def is_market_open(self):
        """Check if the market is currently open"""
        now = datetime.now()

        # US futures markets trading hours (CME Globex)
        # Sunday - Friday: 6:00 PM - 5:00 PM ET (next day)
        # Friday close at 5:00 PM ET, reopen Sunday at 6:00 PM ET

        weekday = now.weekday()  # 0 = Monday, 6 = Sunday

        if weekday == 5:  # Saturday
            return False

        current_hour = now.hour
        current_minute = now.minute
        current_time = current_hour * 100 + current_minute  # Convert to HHMM format

        # Sunday open at 18:00 ET through Friday close at 17:00 ET
        if weekday == 6:  # Sunday
            return current_time >= 1800  # After 6:00 PM
        elif weekday == 4:  # Friday
            return current_time < 1700  # Before 5:00 PM
        else:  # Monday to Thursday
            return True  # Always open

    def _data_fetching_loop(self):
        """Background loop that continuously fetches market data"""
        while not self.stop_event.is_set():
            try:
                # Try each data source until one succeeds
                data_point = None
                for source_func in self.available_sources:
                    try:
                        data_point = source_func()
                        if data_point:
                            break
                    except Exception as e:
                        self.logger.warning(f"Failed to fetch data from source {source_func.__name__}: {str(e)}")

                if data_point:
                    with self.buffer_lock:
                        self.data_buffer.append(data_point)
                        # Keep buffer at specified size
                        if len(self.data_buffer) > self.buffer_size:
                            self.data_buffer = self.data_buffer[-self.buffer_size:]

                    self.last_update = datetime.now()
                    self.logger.debug(f"Updated live data: {data_point}")
                else:
                    self.logger.warning("Failed to fetch data from all sources")

                # Wait before next fetch to avoid API rate limits
                # For 5-minute data, we can poll every minute
                self.stop_event.wait(60)
            except Exception as e:
                self.logger.error(f"Error in data fetching loop: {str(e)}")
                self.stop_event.wait(60)  # Wait a bit longer if there was an error

    def _try_websocket_connection(self):
        """Try to establish a websocket connection for real-time data if possible"""
        try:
            # This is a placeholder - real implementation would use actual endpoints
            # Most free data sources don't provide websocket APIs
            # This would be implemented with a paid service in production
            pass
        except Exception as e:
            self.logger.warning(f"Could not establish websocket connection: {str(e)}")

    def _fetch_from_yfinance(self):
        """Fetch latest data point from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period="1d", interval=self.interval)
            if not data.empty:
                latest = data.iloc[-1]
                return {
                    'time': data.index[-1],
                    'open': latest.Open,
                    'high': latest.High,
                    'low': latest.Low,
                    'close': latest.Close,
                    'volume': latest.Volume
                }
            return None
        except Exception as e:
            self.logger.warning(f"Yahoo Finance fetch error: {str(e)}")
            return None

    def _fetch_from_alphavantage(self):
        """Fetch latest data point from Alpha Vantage (requires free API key)"""
        try:
            # This is a placeholder - you would need to add your Alpha Vantage API key
            api_key = "demo"  # Replace with your API key
            interval_map = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "60m": "60min"}
            av_interval = interval_map.get(self.interval, "5min")

            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={self.symbol}&interval={av_interval}&apikey={api_key}"
            r = requests.get(url)
            data = r.json()

            if "Time Series" in data:
                time_series_key = f"Time Series ({av_interval})"
                latest_time = list(data[time_series_key].keys())[0]
                latest_data = data[time_series_key][latest_time]

                return {
                    'time': datetime.strptime(latest_time, "%Y-%m-%d %H:%M:%S"),
                    'open': float(latest_data["1. open"]),
                    'high': float(latest_data["2. high"]),
                    'low': float(latest_data["3. low"]),
                    'close': float(latest_data["4. close"]),
                    'volume': float(latest_data["5. volume"])
                }
            return None
        except Exception as e:
            self.logger.warning(f"Alpha Vantage fetch error: {str(e)}")
            return None

    def _fetch_from_twelvedata(self):
        """Fetch latest data point from Twelve Data (requires free API key)"""
        try:
            # This is a placeholder - you would need to add your Twelve Data API key
            api_key = "demo"  # Replace with your API key

            url = f"https://api.twelvedata.com/time_series?symbol={self.symbol}&interval={self.interval}&apikey={api_key}&source=docs"
            r = requests.get(url)
            data = r.json()

            if "values" in data and len(data["values"]) > 0:
                latest = data["values"][0]

                return {
                    'time': datetime.strptime(latest["datetime"], "%Y-%m-%d %H:%M:%S"),
                    'open': float(latest["open"]),
                    'high': float(latest["high"]),
                    'low': float(latest["low"]),
                    'close': float(latest["close"]),
                    'volume': float(latest["volume"])
                }
            return None
        except Exception as e:
            self.logger.warning(f"Twelve Data fetch error: {str(e)}")
            return None

    def _fetch_historical_data(self, start_date, end_date):
        """Fetch historical data for a specific time range"""
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(start=start_date, end=end_date, interval=self.interval)

            if not data.empty:
                # Rename columns to match our expected format
                data = data.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })

                # Select only the columns we need
                data = data[['open', 'high', 'low', 'close', 'volume']]

                return data

            return None
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            return None