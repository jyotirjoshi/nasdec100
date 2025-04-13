"""
Order management system for trade execution
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import threading
import queue
import time


class OrderManager:
    def __init__(self, broker, risk_manager, config):
        """
        Initialize Order Manager

        Args:
            broker: Broker interface for order execution
            risk_manager: Risk management module
            config: Configuration settings
        """
        self.broker = broker
        self.risk_manager = risk_manager
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Order tracking
        self.active_orders = {}  # Order ID -> Order details
        self.filled_orders = []
        self.current_position = 0
        self.position_value = 0

        # Portfolio tracking
        self.initial_capital = config.INITIAL_CAPITAL
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital

        # Threading components
        self.command_queue = queue.Queue()
        self.order_thread = None
        self.running = False

        # Market data
        self.latest_data = None

    def start(self):
        """Start order management thread"""
        if self.running:
            self.logger.warning("Order manager already running")
            return

        self.running = True
        self.order_thread = threading.Thread(target=self._order_management_loop)
        self.order_thread.daemon = True
        self.order_thread.start()
        self.logger.info("Order manager started")

    def stop(self):
        """Stop order management thread"""
        if not self.running:
            return

        self.running = False
        if self.order_thread:
            self.command_queue.put(('stop', None))
            self.order_thread.join(timeout=5)
        self.logger.info("Order manager stopped")

    def execute_action(self, action, market_data):
        """
        Execute a trading action

        Args:
            action (int): Action to take (0=hold, 1=buy, 2=sell, 3=close)
            market_data (pandas.DataFrame): Latest market data
        """
        self.latest_data = market_data

        if action == 0:  # HOLD
            return

        # Make sure we have necessary data
        if market_data is None or market_data.empty:
            self.logger.warning("Cannot execute action without market data")
            return

        # Get current price
        current_price = market_data['close'].iloc[-1]

        # Update portfolio value
        self._update_portfolio_value(current_price)

        # Check if action is allowed by risk manager
        if not self.risk_manager.check_action(action, self.current_position,
                                              self.portfolio_value, current_price):
            self.logger.warning(f"Action {action} rejected by risk manager")
            return

        # Process action
        if action == 1:  # BUY
            # Determine position size
            size = self.risk_manager.get_position_size(
                'long', current_price, self.portfolio_value, self.current_position
            )

            if size > 0:
                # Create buy order
                self._submit_order('buy', size, current_price)

        elif action == 2:  # SELL
            # Determine position size
            size = self.risk_manager.get_position_size(
                'short', current_price, self.portfolio_value, self.current_position
            )

            if size > 0:
                # Create sell order
                self._submit_order('sell', size, current_price)

        elif action == 3:  # CLOSE
            if self.current_position != 0:
                # Close position
                side = 'sell' if self.current_position > 0 else 'buy'
                size = abs(self.current_position)
                self._submit_order(side, size, current_price, order_type='close')

    def _submit_order(self, side, size, price, order_type='market'):
        """
        Submit an order to the broker

        Args:
            side (str): Order side ('buy' or 'sell')
            size (int): Order size (number of contracts)
            price (float): Current market price
            order_type (str): Order type ('market', 'limit', 'close')
        """
        # Create order request
        order_request = {
            'side': side,
            'size': size,
            'price': price,
            'type': order_type,
            'timestamp': datetime.now()
        }

        # Submit to order queue
        self.command_queue.put(('order', order_request))
        self.logger.info(f"Order queued: {side} {size} contracts at {price}")

    def _order_management_loop(self):
        """Background thread for order management"""
        while self.running:
            try:
                # Process commands from queue
                try:
                    cmd, data = self.command_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Process command
                if cmd == 'stop':
                    break
                elif cmd == 'order':
                    self._process_order(data)

                # Update order status
                self._update_orders()

                # Update portfolio value
                if self.latest_data is not None:
                    current_price = self.latest_data['close'].iloc[-1]
                    self._update_portfolio_value(current_price)

            except Exception as e:
                self.logger.error(f"Error in order management loop: {str(e)}")

    def _process_order(self, order_request):
        """Process an order request"""
        # Submit order to broker
        try:
            order_id = self.broker.submit_order(
                order_request['side'],
                order_request['size'],
                order_request['price'],
                order_request['type']
            )

            if order_id:
                # Track order
                self.active_orders[order_id] = order_request
                order_request['id'] = order_id
                order_request['status'] = 'open'

                self.logger.info(f"Order submitted: {order_id}")
            else:
                self.logger.warning("Order submission failed")

        except Exception as e:
            self.logger.error(f"Order submission error: {str(e)}")

    def _update_orders(self):
        """Update status of active orders"""
        for order_id in list(self.active_orders.keys()):
            try:
                # Check order status
                status = self.broker.get_order_status(order_id)

                # Update order status
                self.active_orders[order_id]['status'] = status

                # Process filled orders
                if status == 'filled':
                    self._process_fill(order_id)

                # Process canceled orders
                elif status == 'canceled':
                    self.logger.info(f"Order {order_id} was canceled")
                    del self.active_orders[order_id]

            except Exception as e:
                self.logger.error(f"Error updating order {order_id}: {str(e)}")

    def _process_fill(self, order_id):
        """Process a filled order"""
        order = self.active_orders[order_id]

        # Get fill details from broker
        fill_details = self.broker.get_fill_details(order_id)

        # Update order with fill details
        order.update(fill_details)

        # Update position
        size_adjustment = order['size']
        if order['side'] == 'sell':
            size_adjustment = -size_adjustment

        previous_position = self.current_position
        self.current_position += size_adjustment

        # Update cash
        fill_price = fill_details.get('fill_price', order['price'])
        commission = fill_details.get('commission', self.config.COMMISSION_PER_CONTRACT * order['size'])

        # Calculate trade value
        contract_value = order['size'] * self.config.CONTRACT_SIZE * fill_price

        # Update cash based on position change
        if order['side'] == 'buy':
            self.cash -= contract_value + commission
        else:  # sell
            self.cash += contract_value - commission

        # Move to filled orders list
        self.filled_orders.append(order)
        del self.active_orders[order_id]

        self.logger.info(
            f"Order {order_id} filled: {order['side']} {order['size']} at {fill_price}, "
            f"Position: {previous_position} -> {self.current_position}, "
            f"Cash: ${self.cash:.2f}"
        )

    def _update_portfolio_value(self, current_price):
        """Update portfolio value based on current position and price"""
        # Calculate position value
        self.position_value = self.current_position * self.config.CONTRACT_SIZE * current_price

        # Calculate total portfolio value
        self.portfolio_value = self.cash + self.position_value

    def get_portfolio_value(self):
        """Get current portfolio value"""
        return self.portfolio_value

    def get_position(self):
        """Get current position"""
        return {
            'size': self.current_position,
            'value': self.position_value
        }

    def get_portfolio_summary(self):
        """Get portfolio summary"""
        return {
            'cash': self.cash,
            'position_size': self.current_position,
            'position_value': self.position_value,
            'total_value': self.portfolio_value,
            'return_pct': (self.portfolio_value / self.initial_capital - 1) * 100,
            'active_orders': len(self.active_orders),
            'filled_orders': len(self.filled_orders)
        }