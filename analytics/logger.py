"""
Logging setup for the trading bot
"""

import logging
import logging.handlers
import os
from datetime import datetime
import sys


def setup_logger(log_file=None, log_level=logging.INFO):
    """
    Configure logging for the trading bot

    Args:
        log_file (str): Path to log file
        log_level (int): Logging level (default: INFO)
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    # Add file handler if log file is specified
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)

        # Create rotating file handler (10MB max, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    # Log startup message
    logging.info(f"Logger initialized at level {logging.getLevelName(log_level)}")
    if log_file:
        logging.info(f"Logs will be saved to {os.path.abspath(log_file)}")

    return logger


def get_named_logger(name):
    """
    Get a named logger

    Args:
        name (str): Logger name

    Returns:
        logging.Logger: Named logger
    """
    return logging.getLogger(name)


class TradeLogger:
    """Specialized logger for trade events"""

    def __init__(self, log_dir='logs', enable_trade_log=True):
        """
        Initialize TradeLogger

        Args:
            log_dir (str): Directory for trade logs
            enable_trade_log (bool): Whether to enable trade logging
        """
        self.log_dir = log_dir
        self.enable_trade_log = enable_trade_log
        self.logger = logging.getLogger('trades')

        if enable_trade_log:
            # Create trades directory
            os.makedirs(log_dir, exist_ok=True)

            # Create trade log file handler
            trade_log_file = os.path.join(log_dir, f'trades_{datetime.now().strftime("%Y%m%d")}.log')
            handler = logging.FileHandler(trade_log_file)

            # Create formatter
            formatter = logging.Formatter('%(asctime)s,%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)

            # Add handler to logger
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

            # Add header if file is new
            if os.path.getsize(trade_log_file) == 0:
                self.logger.info("timestamp,action,side,size,price,value,commission,position,pnl")

    def log_trade(self, action, side, size, price, value, commission, position, pnl):
        """
        Log a trade

        Args:
            action (str): Trade action ('entry', 'exit')
            side (str): Trade side ('buy', 'sell')
            size (int): Trade size
            price (float): Trade price
            value (float): Trade value
            commission (float): Trade commission
            position (int): Position after trade
            pnl (float): Trade P&L
        """
        if not self.enable_trade_log:
            return

        log_msg = f"{action},{side},{size},{price:.2f},{value:.2f},{commission:.2f},{position},{pnl:.2f}"
        self.logger.info(log_msg)

    def log_signal(self, timestamp, strategy, signal, confidence, price):
        """
        Log a trading signal

        Args:
            timestamp: Signal timestamp
            strategy (str): Strategy name
            signal (int): Signal value (1=buy, -1=sell, 0=hold)
            confidence (float): Signal confidence
            price (float): Current price
        """
        if not self.enable_trade_log:
            return

        signal_type = "BUY" if signal > 0 else "SELL" if signal < 0 else "HOLD"
        log_msg = f"SIGNAL,{strategy},{signal_type},{confidence:.2f},{price:.2f}"
        self.logger.info(log_msg)