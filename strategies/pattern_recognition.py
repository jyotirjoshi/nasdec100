"""
Pattern recognition strategy implementations
"""

import numpy as np
import pandas as pd
from strategies.strategy_base import Strategy


class CandlestickPatterns(Strategy):
    def __init__(self, confirmation_period=3):
        """
        Initialize Candlestick Pattern Recognition strategy

        Args:
            confirmation_period (int): Number of bars to confirm pattern
        """
        super().__init__(
            name="CandlestickPatterns",
            params={
                "confirmation_period": confirmation_period
            }
        )

    def _identify_doji(self, data, idx, body_threshold=0.1):
        """Identify a doji candlestick pattern"""
        if idx < 0 or idx >= len(data):
            return False

        # Calculate body size as percentage of candle size
        candle_size = data['high'].iloc[idx] - data['low'].iloc[idx]
        if candle_size == 0:
            return False

        body_size = abs(data['close'].iloc[idx] - data['open'].iloc[idx])
        body_percentage = body_size / candle_size

        # Doji has very small body
        return body_percentage <= body_threshold

    def _identify_hammer(self, data, idx, body_pos_threshold=0.3, lower_shadow_threshold=2.0):
        """Identify a hammer candlestick pattern"""
        if idx < 0 or idx >= len(data):
            return False

        open_price = data['open'].iloc[idx]
        close_price = data['close'].iloc[idx]
        high_price = data['high'].iloc[idx]
        low_price = data['low'].iloc[idx]

        # Calculate body and shadows
        body_size = abs(close_price - open_price)
        if body_size == 0:
            return False

        candle_size = high_price - low_price
        if candle_size == 0:
            return False

        body_position = (min(open_price, close_price) - low_price) / candle_size
        lower_shadow = min(open_price, close_price) - low_price
        lower_to_body_ratio = lower_shadow / body_size if body_size > 0 else 0

        # Hammer has small body near the top and long lower shadow
        return (body_position >= 1 - body_pos_threshold) and (lower_to_body_ratio >= lower_shadow_threshold)

    def _identify_engulfing(self, data, idx, bullish=True):
        """Identify bullish or bearish engulfing pattern"""
        if idx < 1 or idx >= len(data):
            return False

        curr_open = data['open'].iloc[idx]
        curr_close = data['close'].iloc[idx]
        prev_open = data['open'].iloc[idx - 1]
        prev_close = data['close'].iloc[idx - 1]

        if bullish:
            # Bullish engulfing: Current candle is bullish (close > open)
            # and completely engulfs previous bearish candle
            return (curr_close > curr_open and  # Current candle is bullish
                    prev_close < prev_open and  # Previous candle is bearish
                    curr_open <= prev_close and  # Current open below previous close
                    curr_close >= prev_open)  # Current close above previous open
        else:
            # Bearish engulfing: Current candle is bearish (close < open)
            # and completely engulfs previous bullish candle
            return (curr_close < curr_open and  # Current candle is bearish
                    prev_close > prev_open and  # Previous candle is bullish
                    curr_open >= prev_close and  # Current open above previous close
                    curr_close <= prev_open)  # Current close below previous open

    def generate_signals(self, data):
        """Generate signals based on candlestick patterns"""
        # Initialize pattern columns if not already in data
        pattern_cols = ['doji', 'hammer', 'shooting_star',
                        'bullish_engulfing', 'bearish_engulfing']

        for col in pattern_cols:
            if col not in data.columns:
                data[col] = 0

        # Identify patterns for each candle
        for i in range(1, len(data)):
            # Check for doji
            if self._identify_doji(data, i):
                data.loc[data.index[i], 'doji'] = 1

            # Check for hammer
            if self._identify_hammer(data, i, body_pos_threshold=0.3, lower_shadow_threshold=2.0):
                data.loc[data.index[i], 'hammer'] = 1

            # Check for shooting star (inverse hammer)
            if self._identify_hammer(data, i, body_pos_threshold=0.3, lower_shadow_threshold=2.0):
                # For shooting star, swap high and low in the calculation
                data.loc[data.index[i], 'shooting_star'] = 1

            # Check for bullish engulfing
            if self._identify_engulfing(data, i, bullish=True):
                data.loc[data.index[i], 'bullish_engulfing'] = 1

            # Check for bearish engulfing
            if self._identify_engulfing(data, i, bullish=False):
                data.loc[data.index[i], 'bearish_engulfing'] = 1

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Generate signals based on patterns and market context
        for i in range(2, len(data)):
            # Bullish signals
            if (data['bullish_engulfing'].iloc[i] == 1 or
                data['hammer'].iloc[i] == 1) and \
                    (data['close'].iloc[i] > data['close'].iloc[i - 1]):
                signals.iloc[i] = 1

            # Bearish signals
            if (data['bearish_engulfing'].iloc[i] == 1 or
                data['shooting_star'].iloc[i] == 1) and \
                    (data['close'].iloc[i] < data['close'].iloc[i - 1]):
                signals.iloc[i] = -1

            # Doji at support/resistance can signal reversal
            if data['doji'].iloc[i] == 1:
                # At resistance
                if data['close'].iloc[i] >= data['high'].rolling(20).max().iloc[i - 1]:
                    signals.iloc[i] = -1
                # At support
                elif data['close'].iloc[i] <= data['low'].rolling(20).min().iloc[i - 1]:
                    signals.iloc[i] = 1

        return signals


class ChartPatterns(Strategy):
    def __init__(self, lookback_period=20):
        """
        Initialize Chart Pattern Recognition strategy

        Args:
            lookback_period (int): Lookback period for pattern identification
        """
        super().__init__(
            name="ChartPatterns",
            params={
                "lookback_period": lookback_period
            }
        )

    def _identify_double_bottom(self, data, idx, lookback=20, tolerance=0.03):
        """Identify double bottom pattern"""
        if idx < lookback + 5:
            return False

        # Get window data
        window = data.iloc[idx - lookback:idx + 1]

        # Find the two lowest points in the window
        lowest_idx = window['low'].argmin()
        lowest = window['low'].iloc[lowest_idx]

        # Find second lowest point that's at least 5 bars away from first
        window_without_lowest = window.drop(window.index[lowest_idx])
        second_lowest_idx = window_without_lowest['low'].argmin()
        second_lowest = window_without_lowest['low'].iloc[second_lowest_idx]

        # Calculate the time between bottoms
        time_between = abs(lowest_idx - second_lowest_idx)

        # Check if prices are within tolerance
        price_within_tolerance = (abs(second_lowest - lowest) / lowest) <= tolerance

        # Check if there's a peak in between the two bottoms
        if lowest_idx < second_lowest_idx:
            in_between = window.iloc[lowest_idx:second_lowest_idx]
        else:
            in_between = window.iloc[second_lowest_idx:lowest_idx]

        peak_in_between = in_between['high'].max() > window['high'].iloc[0]

        # Confirm with price action - current price above both bottoms
        price_confirming = data['close'].iloc[idx] > max(lowest, second_lowest)

        # Double bottom criteria
        return (price_within_tolerance and
                time_between >= 5 and
                peak_in_between and
                price_confirming)

    def _identify_double_top(self, data, idx, lookback=20, tolerance=0.03):
        """Identify double top pattern"""
        if idx < lookback + 5:
            return False

        # Get window data
        window = data.iloc[idx - lookback:idx + 1]

        # Find the two highest points in the window
        highest_idx = window['high'].argmax()
        highest = window['high'].iloc[highest_idx]

        # Find second highest point that's at least 5 bars away from first
        window_without_highest = window.drop(window.index[highest_idx])
        second_highest_idx = window_without_highest['high'].argmax()
        second_highest = window_without_highest['high'].iloc[second_highest_idx]

        # Calculate the time between tops
        time_between = abs(highest_idx - second_highest_idx)

        # Check if prices are within tolerance
        price_within_tolerance = (abs(second_highest - highest) / highest) <= tolerance

        # Check if there's a valley in between the two tops
        if highest_idx < second_highest_idx:
            in_between = window.iloc[highest_idx:second_highest_idx]
        else:
            in_between = window.iloc[second_highest_idx:highest_idx]

        valley_in_between = in_between['low'].min() < window['low'].iloc[0]

        # Confirm with price action - current price below both tops
        price_confirming = data['close'].iloc[idx] < min(highest, second_highest)

        # Double top criteria
        return (price_within_tolerance and
                time_between >= 5 and
                valley_in_between and
                price_confirming)

    def _identify_head_and_shoulders(self, data, idx, lookback=30, shoulder_tolerance=0.05):
        """Identify head and shoulders pattern"""
        if idx < lookback + 10:
            return False

        # Get window data
        window = data.iloc[idx - lookback:idx + 1]

        # Need to identify three peaks with the middle one being highest
        # First find all local maxima
        peaks = []
        for i in range(1, len(window) - 1):
            if (window['high'].iloc[i] > window['high'].iloc[i - 1] and
                    window['high'].iloc[i] > window['high'].iloc[i + 1]):
                peaks.append((i, window['high'].iloc[i]))

        if len(peaks) < 3:
            return False

        # Find the highest peak (head)
        head_idx, head_value = max(peaks, key=lambda x: x[1])

        # Find left shoulder (before head)
        left_shoulder = [(i, v) for i, v in peaks if i < head_idx]
        if not left_shoulder:
            return False
        left_shoulder_idx, left_shoulder_value = max(left_shoulder, key=lambda x: x[1])

        # Find right shoulder (after head)
        right_shoulder = [(i, v) for i, v in peaks if i > head_idx]
        if not right_shoulder:
            return False
        right_shoulder_idx, right_shoulder_value = max(right_shoulder, key=lambda x: x[1])

        # Check if shoulders are at similar levels
        if abs(left_shoulder_value - right_shoulder_value) / head_value > shoulder_tolerance:
            return False

        # Check if head is higher than shoulders
        if head_value <= max(left_shoulder_value, right_shoulder_value):
            return False

        # Find neckline (support line connecting troughs between peaks)
        left_trough_idx = window['low'].iloc[left_shoulder_idx:head_idx].argmin() + left_shoulder_idx
        right_trough_idx = window['low'].iloc[head_idx:right_shoulder_idx].argmin() + head_idx

        left_trough_value = window['low'].iloc[left_trough_idx]
        right_trough_value = window['low'].iloc[right_trough_idx]

        # Check if price has broken below neckline
        neckline_value_at_end = left_trough_value + (right_trough_value - left_trough_value) * \
                                (len(window) - left_trough_idx) / (right_trough_idx - left_trough_idx)

        price_below_neckline = data['close'].iloc[idx] < neckline_value_at_end

        return price_below_neckline

    def generate_signals(self, data):
        """Generate signals based on chart patterns"""
        # Get parameters
        lookback = self.params["lookback_period"]

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Identify patterns and generate signals
        for i in range(lookback + 10, len(data)):
            # Bullish double bottom
            if self._identify_double_bottom(data, i, lookback):
                signals.iloc[i] = 1

            # Bearish double top
            if self._identify_double_top(data, i, lookback):
                signals.iloc[i] = -1

            # Bearish head and shoulders
            if self._identify_head_and_shoulders(data, i, lookback):
                signals.iloc[i] = -1

        return signals


class FibonacciRetracement(Strategy):
    def __init__(self, lookback_period=100, key_levels=(0.382, 0.5, 0.618)):
        """
        Initialize Fibonacci Retracement strategy

        Args:
            lookback_period (int): Period to look back for swing high/low
            key_levels (tuple): Key Fibonacci retracement levels
        """
        super().__init__(
            name="FibonacciRetracement",
            params={
                "lookback_period": lookback_period,
                "key_levels": key_levels
            }
        )

    def _find_swing_high_low(self, data, lookback=100):
        """Find recent swing high and low"""
        if len(data) < lookback:
            lookback = len(data) - 1

        window = data.iloc[-lookback:]
        swing_high = window['high'].max()
        swing_high_idx = window['high'].argmax()
        swing_low = window['low'].min()
        swing_low_idx = window['low'].argmin()

        # Determine which came first
        if swing_high_idx < swing_low_idx:
            # Downtrend: high first, then low
            direction = "down"
        else:
            # Uptrend: low first, then high
            direction = "up"

        return swing_high, swing_low, direction

    def _calculate_fib_levels(self, high, low):
        """Calculate Fibonacci retracement levels"""
        diff = high - low
        levels = {}

        for level in self.params["key_levels"]:
            if high > low:  # If we're in an uptrend
                levels[level] = high - (diff * level)
            else:  # If we're in a downtrend
                levels[level] = low + (diff * level)

        return levels

    def generate_signals(self, data):
        """Generate signals based on Fibonacci retracements"""
        # Get parameters
        lookback = self.params["lookback_period"]

        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Skip if we don't have enough data
        if len(data) < lookback + 5:
            return signals

        # Find swing high and low
        swing_high, swing_low, direction = self._find_swing_high_low(data.iloc[:-5], lookback)

        # Calculate Fibonacci levels
        fib_levels = self._calculate_fib_levels(swing_high, swing_low)

        # Generate signals based on price interaction with Fibonacci levels
        for i in range(-5, 0):
            current_price = data['close'].iloc[i]

            if direction == "down":
                # In a downtrend, look for support at Fibonacci levels
                for level, price in fib_levels.items():
                    # If price approaches a Fibonacci level from above and bounces
                    if (data['low'].iloc[i - 1] <= price * 1.01 and
                            data['close'].iloc[i - 1] <= data['open'].iloc[i - 1] and
                            data['close'].iloc[i] > data['open'].iloc[i]):
                        signals.iloc[i] = 1  # Bullish signal
                        break
            else:
                # In an uptrend, look for resistance at Fibonacci levels
                for level, price in fib_levels.items():
                    # If price approaches a Fibonacci level from below and reverses
                    if (data['high'].iloc[i - 1] >= price * 0.99 and
                            data['close'].iloc[i - 1] >= data['open'].iloc[i - 1] and
                            data['close'].iloc[i] < data['open'].iloc[i]):
                        signals.iloc[i] = -1  # Bearish signal
                        break

        return signals


class HarmonicPatterns(Strategy):
    def __init__(self, tolerance=0.05):
        """
        Initialize Harmonic Pattern Recognition strategy

        Args:
            tolerance (float): Tolerance for pattern ratio matching
        """
        super().__init__(
            name="HarmonicPatterns",
            params={
                "tolerance": tolerance
            }
        )

    def _is_ratio_match(self, actual, target, tolerance):
        """Check if actual ratio matches target within tolerance"""
        return abs(actual - target) <= tolerance

    def _identify_gartley(self, points, is_bullish=True):
        """Identify Gartley pattern"""
        # Extract points
        X, A, B, C, D = points

        # Calculate ratios
        AB = abs(B - A)
        BC = abs(C - B)
        CD = abs(D - C)
        XA = abs(A - X)

        # Gartley pattern ratios
        # AB should be 0.618 of XA
        # BC should be 0.382 or 0.886 of AB
        # CD should be 1.27 or 1.618 of BC

        ab_xa_ratio = AB / XA
        bc_ab_ratio = BC / AB
        cd_bc_ratio = CD / BC

        tolerance = self.params["tolerance"]

        # Check ratios
        ab_check = self._is_ratio_match(ab_xa_ratio, 0.618, tolerance)
        bc_check = (self._is_ratio_match(bc_ab_ratio, 0.382, tolerance) or
                    self._is_ratio_match(bc_ab_ratio, 0.886, tolerance))
        cd_check = (self._is_ratio_match(cd_bc_ratio, 1.27, tolerance) or
                    self._is_ratio_match(cd_bc_ratio, 1.618, tolerance))

        # Direction check
        direction_check = False
        if is_bullish:
            direction_check = (X > A and B > A and C > B and D > C)
        else:
            direction_check = (X < A and B < A and C < B and D < C)

        return ab_check and bc_check and cd_check and direction_check

    def _find_zigzag_points(self, data, min_swing_size=0.01):
        """Find zigzag points for harmonic pattern identification"""
        # Initialize
        zigzag = []
        direction = None
        last_price = data['close'].iloc[0]
        last_idx = 0

        # Find swing highs and lows
        for i in range(1, len(data)):
            price = data['close'].iloc[i]
            price_change = (price - last_price) / last_price

            if abs(price_change) >= min_swing_size:
                if (direction is None) or (direction == "up" and price < last_price) or (
                        direction == "down" and price > last_price):
                    # Add the last point as a zigzag point
                    zigzag.append((last_idx, last_price))

                    # Change direction
                    direction = "down" if price < last_price else "up"
                    last_price = price
                    last_idx = i
                elif ((direction == "up" and price > last_price) or
                      (direction == "down" and price < last_price)):
                    # Update the extreme in current direction
                    last_price = price
                    last_idx = i

        # Add the last point
        if zigzag and last_idx > zigzag[-1][0]:
            zigzag.append((last_idx, last_price))

        return zigzag

    def generate_signals(self, data):
        """Generate signals based on harmonic patterns"""
        # Initialize signals series
        signals = pd.Series(0, index=data.index)

        # Find zigzag points
        zigzag_points = self._find_zigzag_points(data)

        # Need at least 5 points for a harmonic pattern
        if len(zigzag_points) < 5:
            return signals

        # Look for patterns in the most recent points
        for i in range(4, len(zigzag_points)):
            # Get the last 5 points for pattern
            pattern_points = [point[1] for point in zigzag_points[i - 4:i + 1]]

            # Check for bullish Gartley
            if self._identify_gartley(pattern_points, is_bullish=True):
                signals.iloc[zigzag_points[i][0]] = 1

            # Check for bearish Gartley
            if self._identify_gartley(pattern_points, is_bullish=False):
                signals.iloc[zigzag_points[i][0]] = -1

        return signals