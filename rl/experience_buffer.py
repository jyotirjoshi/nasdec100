"""
Experience replay buffer for reinforcement learning
"""

import numpy as np
import random
from collections import namedtuple, deque
import torch

# Define experience tuple structure
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples"""

    def __init__(self, capacity, batch_size, device):
        """
        Initialize ReplayBuffer

        Args:
            capacity (int): Maximum size of buffer
            batch_size (int): Size of training batch
            device: PyTorch device
        """
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert to torch tensors
        states = torch.FloatTensor(
            np.vstack([e.state for e in experiences if e is not None])
        ).to(self.device)
        actions = torch.LongTensor(
            np.vstack([e.action for e in experiences if e is not None])
        ).to(self.device)
        rewards = torch.FloatTensor(
            np.vstack([e.reward for e in experiences if e is not None])
        ).to(self.device)
        next_states = torch.FloatTensor(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).to(self.device)
        dones = torch.FloatTensor(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)


class PrioritizedReplayBuffer:
    """Prioritized Replay Buffer for RL agents"""

    def __init__(self, capacity, batch_size, device, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Initialize Prioritized Replay Buffer

        Args:
            capacity (int): Maximum size of buffer
            batch_size (int): Size of training batch
            device: PyTorch device
            alpha (float): How much prioritization to use (0 = uniform, 1 = full prioritization)
            beta_start (float): Starting importance sampling correction factor
            beta_frames (int): Number of frames over which to anneal beta to 1.0
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def beta_by_frame(self, frame_idx):
        """Calculate beta value for importance sampling"""
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory with maximum priority"""
        max_prio = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append(Experience(state, action, reward, next_state, done))
        else:
            self.memory[self.position] = Experience(state, action, reward, next_state, done)

        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        """Sample a batch of experiences based on priority"""
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]

        # Calculate probabilities
        probs = prios ** self.alpha
        probs /= probs.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        # Calculate importance sampling weights
        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Calculate weights
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(self.device)

        # Convert to torch tensors
        states = torch.FloatTensor(
            np.vstack([s.state for s in samples])
        ).to(self.device)
        actions = torch.LongTensor(
            np.vstack([s.action for s in samples])
        ).to(self.device)
        rewards = torch.FloatTensor(
            np.vstack([s.reward for s in samples])
        ).to(self.device)
        next_states = torch.FloatTensor(
            np.vstack([s.next_state for s in samples])
        ).to(self.device)
        dones = torch.FloatTensor(
            np.vstack([s.done for s in samples]).astype(np.uint8)
        ).to(self.device)

        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)