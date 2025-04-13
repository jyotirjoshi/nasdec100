"""
Trading environment for reinforcement learning
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
import logging


class TradingEnvironment(gym.Env):
    """OpenAI Gym environment for trading using reinforcement learning"""

    def __init__(self, data, config, window_size=20):
        """
        Initialize the trading environment

        Args:
            data (pandas.DataFrame): Historical market data
            config (module): Configuration settings
            window_size (int): Observation window size
        """
        super(TradingEnvironment, self).__init__()

        self.data = data
        self.config = config
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)

        # Trading params
        self.initial_balance = config.INITIAL_CAPITAL
        self.transaction_cost = config.COMMISSION_PER_CONTRACT
        self.contract_size = config.CONTRACT_SIZE

        # Episode params
        self._start_tick = window_size
        self._end_tick = len(self.data) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._balance = None
        self._action_history = None

        # Define action and observation space
        # Actions: 0 (hold), 1 (buy), 2 (sell), 3 (close)
        self.action_space = spaces.Discrete(4)

        # State space: market features + position info
        feature_count = len(data.columns) * window_size + 4  # +4 for current position and balance
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_count,), dtype=np.float32
        )

    def reset(self):
        """Reset the environment for a new episode"""
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick
        self._position = 0
        self._position_history = [0] * self._start_tick
        self._total_reward = 0.0
        self._total_profit = 0.0
        self._balance = self.initial_balance
        self._action_history = [0] * self._start_tick

        return self._get_observation()

    def step(self, action):
        """
        Take a step in the environment

        Args:
            action (int): Action to take (0=hold, 1=buy, 2=sell, 3=close)

        Returns:
            tuple: (observation, reward, done, info)
        """
        if self._done:
            self.logger.warning("Trading episode is done, reset the environment")
            return self._get_observation(), 0.0, self._done, {}

        self._current_tick += 1

        # Check if episode is done
        if self._current_tick >= self._end_tick:
            self._done = True

        # Execute the action
        step_reward = self._take_action(action)
        self._total_reward += step_reward

        # Update position history
        self._position_history.append(self._position)

        # Update action history
        self._action_history.append(action)

        # Get current observation
        observation = self._get_observation()

        info = {
            'current_tick': self._current_tick,
            'current_price': self.data['close'].iloc[self._current_tick],
            'position': self._position,
            'balance': self._balance,
            'total_reward': self._total_reward,
            'total_profit': self._total_profit
        }

        return observation, step_reward, self._done, info

    def _take_action(self, action):
        """
        Execute trading action

        Args:
            action (int): Action to take (0=hold, 1=buy, 2=sell, 3=close)

        Returns:
            float: Reward from the action
        """
        current_price = self.data['close'].iloc[self._current_tick]
        reward = 0.0

        # Hold position
        if action == 0:
            pass

        # Buy position
        elif action == 1:
            if self._position <= 0:
                # Close any existing short position
                if self._position < 0:
                    close_profit = -self._position * self.contract_size * (
                            self.data['close'].iloc[self._last_trade_tick] - current_price
                    )
                    self._total_profit += close_profit
                    self._balance += close_profit - abs(self._position) * self.transaction_cost
                    reward += close_profit

                # Open new long position
                self._position = 1  # Simple version: always trade 1 contract
                self._last_trade_tick = self._current_tick
                self._balance -= self.transaction_cost

        # Sell position
        elif action == 2:
            if self._position >= 0:
                # Close any existing long position
                if self._position > 0:
                    close_profit = self._position * self.contract_size * (
                            current_price - self.data['close'].iloc[self._last_trade_tick]
                    )
                    self._total_profit += close_profit
                    self._balance += close_profit - abs(self._position) * self.transaction_cost
                    reward += close_profit

                # Open new short position
                self._position = -1  # Simple version: always trade 1 contract
                self._last_trade_tick = self._current_tick
                self._balance -= self.transaction_cost

        # Close position
        elif action == 3:
            if self._position > 0:
                # Close long position
                close_profit = self._position * self.contract_size * (
                        current_price - self.data['close'].iloc[self._last_trade_tick]
                )
                self._total_profit += close_profit
                self._balance += close_profit - abs(self._position) * self.transaction_cost
                reward += close_profit
                self._position = 0

            elif self._position < 0:
                # Close short position
                close_profit = -self._position * self.contract_size * (
                        self.data['close'].iloc[self._last_trade_tick] - current_price
                )
                self._total_profit += close_profit
                self._balance += close_profit - abs(self._position) * self.transaction_cost
                reward += close_profit
                self._position = 0

        # Calculate unrealized PnL if still in position
        if self._position > 0:
            unrealized_profit = self._position * self.contract_size * (
                    current_price - self.data['close'].iloc[self._last_trade_tick]
            )
            reward += unrealized_profit * 0.1  # Smaller reward for unrealized profit
        elif self._position < 0:
            unrealized_profit = -self._position * self.contract_size * (
                    self.data['close'].iloc[self._last_trade_tick] - current_price
            )
            reward += unrealized_profit * 0.1  # Smaller reward for unrealized profit

        return reward

    def _get_observation(self):
        """
        Get current observation (state) of the environment

        Returns:
            numpy.ndarray: Current state observation
        """
        # Get historical market data for the observation window
        start_idx = max(0, self._current_tick - self.window_size + 1)
        end_idx = self._current_tick + 1

        # Get window of market data
        window_data = self.data.iloc[start_idx:end_idx]

        # If window is smaller than window_size, pad with zeros
        if len(window_data) < self.window_size:
            padding = self.window_size - len(window_data)
            padded_data = pd.DataFrame(
                0,
                index=range(padding),
                columns=window_data.columns
            )
            window_data = pd.concat([padded_data, window_data], ignore_index=True)

        # Convert market data to flat numpy array
        market_data = window_data.values.flatten()

        # Add position and balance info to the observation
        position_info = np.array([
            self._position,
            self._balance / self.initial_balance,
            self._total_profit / self.initial_balance,
            (self._current_tick - self._last_trade_tick) / 100  # Time since last trade (normalized)
        ])

        # Combine market data and position info
        observation = np.concatenate([market_data, position_info])

        return observation

    def render(self, mode='human'):
        """
        Render the environment

        Args:
            mode (str): Rendering mode ('human' for human-readable output)
        """
        if mode == 'human':
            print(f"Tick: {self._current_tick}, Position: {self._position}, "
                  f"Balance: {self._balance:.2f}, Total Profit: {self._total_profit:.2f}")