"""
Advanced reinforcement learning models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DuelingDQN(nn.Module):
    """Dueling DQN architecture with separate value and advantage streams"""

    def __init__(self, state_size, action_size, hidden_size=128):
        """
        Initialize the Dueling DQN model

        Args:
            state_size (int): Size of state vector
            action_size (int): Number of possible actions
            hidden_size (int): Size of hidden layers
        """
        super(DuelingDQN, self).__init__()

        # Feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size)
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )

    def forward(self, state):
        """Forward pass through the network"""
        # Handle single state (inference mode)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Extract features
        features = self.feature_layer(state)

        # Calculate value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage (Q = V + A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class A2CNetwork(nn.Module):
    """Actor-Critic Network for continuous action spaces"""

    def __init__(self, state_size, action_size, hidden_size=128):
        """
        Initialize the Actor-Critic model

        Args:
            state_size (int): Size of state vector
            action_size (int): Number of continuous actions
            hidden_size (int): Size of hidden layers
        """
        super(A2CNetwork, self).__init__()

        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Actor (policy) network
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))

        # Critic (value) network
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state):
        """
        Forward pass through network

        Args:
            state: Input state

        Returns:
            tuple: (action_mean, action_log_std, value)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Shared features
        features = self.shared_layers(state)

        # Actor output
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)

        # Critic output
        value = self.critic(features)

        return action_mean, action_log_std, value

    def get_action(self, state):
        """
        Sample action from policy

        Args:
            state: Input state

        Returns:
            tuple: (action, log_prob, value)
        """
        action_mean, action_log_std, value = self.forward(state)

        # Create normal distribution
        action_std = torch.exp(action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)

        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value


class PPONetwork(nn.Module):
    """PPO Network with separate actor and critic"""

    def __init__(self, state_size, action_size, hidden_size=128, discrete=True):
        """
        Initialize PPO Network

        Args:
            state_size (int): Size of state vector
            action_size (int): Number of actions
            hidden_size (int): Size of hidden layers
            discrete (bool): Whether action space is discrete
        """
        super(PPONetwork, self).__init__()

        self.discrete = discrete

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        # Actor (policy) head
        if discrete:
            self.actor = nn.Linear(hidden_size, action_size)
        else:
            self.actor_mean = nn.Linear(hidden_size, action_size)
            self.actor_log_std = nn.Parameter(torch.zeros(1, action_size))

        # Critic (value) head
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state):
        """Forward pass through network"""
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Shared features
        features = self.shared(state)

        # Actor output
        if self.discrete:
            # Discrete actions (categorical distribution)
            action_probs = F.softmax(self.actor(features), dim=-1)
            dist = torch.distributions.Categorical(action_probs)
        else:
            # Continuous actions (normal distribution)
            action_mean = self.actor_mean(features)
            action_std = torch.exp(self.actor_log_std)
            dist = torch.distributions.Normal(action_mean, action_std)

        # Critic output
        value = self.critic(features)

        return dist, value