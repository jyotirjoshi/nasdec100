"""
Main entry point for the Nasdaq-100 E-mini Trading Bot
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from datetime import datetime

from data.data_loader import DataLoader
from data.live_data_fetcher import LiveDataFetcher
from data.data_processor import DataProcessor
from rl.environment import TradingEnvironment
from rl.agent import DQNAgent
from rl.trainer import RLTrainer
from simulation.backtester import Backtester
from simulation.paper_trader import PaperTrader
from execution.order_manager import OrderManager
from execution.broker_interface import BrokerInterface
from execution.risk_management import RiskManager
from analytics.performance_metrics import PerformanceAnalyzer
from analytics.visualization import Visualizer
from analytics.logger import setup_logger
import config


def setup_logging():
    """Set up logging configuration"""
    log_file = Path(config.LOG_DIR) / f"trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logger(log_file)
    logging.info("Starting Nasdaq-100 E-mini Trading Bot")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Nasdaq-100 E-mini Trading Bot')
    parser.add_argument('--mode', type=str, choices=['train', 'backtest', 'paper_trade', 'live_trade'],
                        default='train', help='Operation mode')
    parser.add_argument('--strategy', type=str, default='ensemble',
                        help='Trading strategy to use')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved RL model')
    parser.add_argument('--data_source', type=str, choices=['historical', 'live', 'both'],
                        default='historical', help='Data source to use')

    return parser.parse_args()


def main():
    """Main function to run the trading bot"""
    args = parse_arguments()
    setup_logging()

    # Initialize data components
    data_loader = DataLoader(config.HISTORICAL_DATA_PATH)
    live_data_fetcher = LiveDataFetcher() if args.data_source in ['live', 'both'] else None
    data_processor = DataProcessor()

    # Load and process historical data
    logging.info("Loading historical data...")
    historical_data = data_loader.load_data()
    processed_data = data_processor.process_data(historical_data)

    # Initialize components based on mode
    if args.mode == 'train':
        logging.info(f"Starting training mode with {args.strategy} strategy...")
        # Create trading environment
        env = TradingEnvironment(processed_data, config)

        # Create RL agent
        agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            learning_rate=config.LEARNING_RATE,
            gamma=config.GAMMA,
            epsilon_start=config.EPSILON_START,
            epsilon_end=config.EPSILON_END,
            epsilon_decay=config.EPSILON_DECAY,
            memory_capacity=config.MEMORY_CAPACITY,
            batch_size=config.BATCH_SIZE
        )

        # Train agent
        trainer = RLTrainer(env, agent, config)
        trained_agent = trainer.train(episodes=config.EPISODES)

        # Save trained model
        model_path = Path(config.MODEL_DIR) / f"{args.strategy}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        trained_agent.save_model(str(model_path))
        logging.info(f"Model saved to {model_path}")

    elif args.mode == 'backtest':
        logging.info("Starting backtest mode...")
        # Load trained model if provided
        agent = None
        if args.model_path:
            agent = DQNAgent.load_model(args.model_path)

        # Create backtester and run backtesting
        backtester = Backtester(processed_data, agent, config)
        results = backtester.run()

        # Analyze and visualize results
        analyzer = PerformanceAnalyzer(results)
        metrics = analyzer.calculate_metrics()
        visualizer = Visualizer(results, metrics)
        visualizer.plot_results()
        logging.info(f"Backtest results: {metrics}")

    elif args.mode == 'paper_trade':
        logging.info("Starting paper trading mode...")
        # Initialize paper trader
        paper_trader = PaperTrader(live_data_fetcher, config)

        # Load trained model
        agent = None
        if args.model_path:
            agent = DQNAgent.load_model(args.model_path)
            paper_trader.set_agent(agent)

        # Start paper trading
        paper_trader.run()

    elif args.mode == 'live_trade':
        logging.info("Starting live trading mode...")
        # Make sure we have a trained agent
        if not args.model_path:
            logging.error("Model path must be provided for live trading")
            sys.exit(1)

        agent = DQNAgent.load_model(args.model_path)

        # Initialize risk manager
        risk_manager = RiskManager(config)

        # Initialize broker interface
        broker = BrokerInterface(config)

        # Initialize order manager
        order_manager = OrderManager(broker, risk_manager, config)

        # Start live trading
        while True:
            if live_data_fetcher.is_market_open():
                # Fetch latest data
                latest_data = live_data_fetcher.get_latest_data()
                processed_latest = data_processor.process_data(latest_data)

                # Get trading action from agent
                state = data_processor.create_state_representation(processed_latest)
                action = agent.act(state, training=False)

                # Execute trading action
                order_manager.execute_action(action, processed_latest)

                # Log and monitor
                logging.info(
                    f"Executed action: {action}, current portfolio value: {order_manager.get_portfolio_value()}")
            else:
                logging.info("Market closed. Waiting...")
                # Wait until market opens

    logging.info("Trading bot execution completed")


if __name__ == "__main__":
    main()