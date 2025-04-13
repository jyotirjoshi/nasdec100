"""
Configuration settings for the Nasdaq-100 E-mini Trading Bot
"""

import os
from pathlib import Path

# Paths
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT_DIR / "data_files"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"
PLOT_DIR = ROOT_DIR / "plots" # For saving plots

# Ensure directories exist
for dir_path in [DATA_DIR, MODEL_DIR, LOG_DIR, PLOT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# --- Data Settings ---
# HISTORICAL_DATA_PATH = r"C:\Users\spars\Desktop\kise pata\elite_bot\NQ_emini_5min_complete_1y_2025-04-10.csv"
HISTORICAL_DATA_PATH = r"C:\Users\spars\Desktop\kise pata\star_war\data\NQ_emini_5min_complete_1y_2025-04-10.csv" # Use the path from the error message
TIMEFRAMES = ["1min", "5min", "15min", "1h"] # Added 1min
DEFAULT_TIMEFRAME = "5min"
EXTERNAL_SYMBOLS = ['^VIX', 'ES=F', 'ZN=F'] # Example: VIX, S&P500 Futures, 10Y Treasury Futures

# --- Live Data Settings ---
LIVE_DATA_SOURCE = 'yfinance' # Options: 'yfinance', 'alphavantage', 'twelvedata', 'ibkr', 'polygon', etc.
LIVE_DATA_INTERVAL = '1m' # Use '1m' or '5m' depending on broker/API capability
API_KEY = os.getenv("YOUR_API_KEY") # Set environment variable or replace None
API_SECRET = os.getenv("YOUR_API_SECRET") # Set environment variable or replace None
# IBKR Settings (Example)
IBKR_HOST = '127.0.0.1'
IBKR_PORT = 7497 # 7497 for TWS, 4001 for Gateway default
IBKR_CLIENT_ID = 101 # Use a unique client ID

# --- Trading Settings ---
SYMBOL_FUTURES = "NQ=F" # Symbol for data fetching (Yahoo Finance format)
SYMBOL_BROKER = "NQ" # Symbol format for your specific broker (e.g., 'NQ H4' for March 2024 contract)
CONTRACT_EXPIRY = "202406" # Example: June 2024 expiry - IMPORTANT: Update this regularly or use continuous contracts if broker supports
INITIAL_CAPITAL = 100000.0
CONTRACT_SIZE = 20.0  # NQ E-mini contract size ($ per point)
COMMISSION_PER_CONTRACT = 2.10 # Realistic commission (round trip)
SLIPPAGE_POINTS = 0.25 # Slippage per side (in NQ points, e.g., 0.25 points = $5)
VARIABLE_SLIPPAGE = True # Use volatility-based slippage in RL environment

# --- RL Settings ---
EPISODES = 1000 # Total training episodes
WARMUP_EPISODES = 50 # Episodes to fill buffer before training starts
BATCH_SIZE = 64
LEARNING_RATE = 0.0005 # Slightly lower LR can help stability
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0 # Start with high exploration
EPSILON_END = 0.05 # Keep some exploration
EPSILON_DECAY = 0.998 # Slower decay
TARGET_UPDATE_FREQUENCY = 10 # Episodes between target network updates
MEMORY_CAPACITY = 50000 # Adjust based on RAM
RL_REWARD_SHARPE_WEIGHT = 0.05 # Weight for Sharpe ratio in reward (0 to disable)
RL_WINDOW_SIZE = 30 # Observation window for RL agent

# --- Walk-Forward Settings ---
# Lengths are in number of data points (bars)
WF_TRAIN_LEN = 252 * 24 * 6 # Approx 6 months of 5-min data
WF_TEST_LEN = 252 * 24 * 1   # Approx 1 month of 5-min data
WF_RETRAIN_EVERY = WF_TEST_LEN * 3 # Retrain every 3 months

# --- Trading Strategy Parameters ---
# Example: Using Ensemble - configure weights and specific params here
ENSEMBLE_STRATEGIES_CONFIG = {
    "MovingAverageCrossover": {"weight": 1.0, "params": {"fast_period": 9, "slow_period": 21}},
    "MACDStrategy": {"weight": 1.0, "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9}},
    "RSIStrategy": {"weight": 0.8, "params": {"period": 14, "oversold": 30, "overbought": 70}},
    "BollingerBandsStrategy": {"weight": 0.8, "params": {"period": 20, "std_dev": 2.0}},
    "DonchianBreakout": {"weight": 0.7, "params": {"channel_period": 20, "exit_period": 10}},
    "ATRChannelStrategy": {"weight": 0.9, "params": {"atr_period": 14, "channel_multiplier": 2.0}},
    # Add more strategies as needed...
    # "MLStrategy": {"weight": 1.5, "params": {"model_type": "random_forest", "lookback_period": 20}} # Example ML
}
# Default strategy if not specified via command line
DEFAULT_STRATEGY = 'ensemble'

# --- Risk Management ---
MAX_POSITION_SIZE = 5  # Maximum number of contracts allowed open
MAX_RISK_PER_TRADE = 0.015  # Maximum risk per trade as fraction of portfolio value
MAX_DAILY_LOSS = 0.03  # Maximum daily loss fraction before stopping trading for the day
STOP_LOSS_ATR_MULTIPLIER = 1.5 # ATR multiplier for stop loss distance
TAKE_PROFIT_ATR_MULTIPLIER = 2.5 # ATR multiplier for take profit distance

# --- Live Trading Settings ---
LIVE_TRADING_MODE = 'paper' # 'paper' or 'live'
BROKER_TYPE = 'simulated' # 'simulated' or 'interactive_brokers', 'polygon', etc.
TRADING_HOURS_START = "09:30"  # Eastern Time (adjust for NQ futures hours, e.g., Globex)
TRADING_HOURS_END = "16:00"    # Eastern Time (adjust for NQ futures hours)

# --- Logging ---
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR
TRADE_LOG_ENABLED = True # Log individual trades to a separate file