"""
Reinforcement Learning Agent for Trading
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import logging
import os
from collections import deque


class DQNNetwork(nn.Module):
    """Deep Q-Network Model"""

    def __init__(self, state_size, action_size, hidden_size=64):
        """
        Initialize the DQN model

        Args:
            state_size (int): Size of state vector
            action_size (int): Number of possible actions
            hidden_size (int): Size of hidden layers
        """
        super(DQNNetwork, self).__init__()

        # Layers
        self.fc1 = nn.Linear(state_size, hidden_size * 2)
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """Forward pass through the network"""
        # Use batch norm only during training (batch size > 1)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            use_bn = False
        else:
            use_bn = True

        x = F.relu(self.fc1(x))
        if use_bn:
            x = self.bn1(x)
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        if use_bn:
            x = self.bn2(x)
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        if use_bn:
            x = self.bn3(x)

        return self.fc4(x)


class DQNAgent:
    """Agent using Deep Q-Learning with Experience Replay"""

    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_capacity=10000, batch_size=64):
        """
        Initialize the DQN Agent

        Args:
            state_size (int): Size of state vector
            action_size (int): Number of possible actions
            learning_rate (float): Learning rate for optimizer
            gamma (float): Discount factor
            epsilon_start (float): Starting exploration rate
            epsilon_end (float): Minimum exploration rate
            epsilon_decay (float): Decay rate for exploration
            memory_capacity (int): Size of replay memory
            batch_size (int): Batch size for training
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.logger = logging.getLogger(__name__)

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Network and Target Network
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.update_target_network()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def update_target_network(self):
        """Update target network with Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """
        Choose action based on epsilon-greedy policy

        Args:
            state (numpy.ndarray): Current state
            training (bool): Whether the agent is in training mode

        Returns:
            int: Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        self.q_network.train()

        return torch.argmax(action_values).item()

    def replay(self):
        """Train on batch of experiences from replay memory"""
        if len(self.memory) < self.batch_size:
            return

        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)

        # Prepare batch tensors
        states = torch.FloatTensor([experience[0] for experience in batch]).to(self.device)
        actions = torch.LongTensor([experience[1] for experience in batch]).to(self.device)
        rewards = torch.FloatTensor([experience[2] for experience in batch]).to(self.device)
        next_states = torch.FloatTensor([experience[3] for experience in batch]).to(self.device)
        dones = torch.FloatTensor([experience[4] for experience in batch]).to(self.device)

        # Compute Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            target_q_values = rewards + (1 - dones) * self.gamma * \
                              self.target_network(next_states).max(1)[0]

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradient problem
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save_model(self, path):
        """
        Save model weights to disk

        Args:
            path (str): File path for saving the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        self.logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path, state_size=None, action_size=None):
        """
        Load model weights from disk

        Args:
            path (str): File path for loading the model
            state_size (int): Size of state vector
            action_size (int): Number of possible actions

        Returns:
            DQNAgent: Loaded agent
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load model data
        checkpoint = torch.load(path, map_location=torch.device('cpu'))

        # Extract model info
        q_network_state = checkpoint['q_network_state_dict']
        target_network_state = checkpoint.get('target_network_state_dict')
        optimizer_state = checkpoint.get('optimizer_state_dict')
        epsilon = checkpoint.get('epsilon', 0.01)

        # Infer state and action sizes from model if not provided
        if state_size is None or action_size is None:
            # Get first layer weight shape for state_size
            first_layer = next(iter(q_network_state.items()))
            if 'weight' in first_layer[0]:
                state_size = first_layer[1].shape[1]

            # Get last layer weight shape for action_size
            last_layer = list(q_network_state.items())[-2]
            if 'weight' in last_layer[0]:
                action_size = last_layer[1].shape[0]

        # Create new agent
        agent = cls(state_size, action_size)

        # Load weights
        agent.q_network.load_state_dict(q_network_state)
        if target_network_state:
            agent.target_network.load_state_dict(target_network_state)
        else:
            agent.update_target_network()

        # Load optimizer state
        if optimizer_state:
            agent.optimizer.load_state_dict(optimizer_state)

        # Set exploration rate
        agent.epsilon = epsilon

        return agent