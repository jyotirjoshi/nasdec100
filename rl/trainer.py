"""
Trainer for reinforcement learning agent
"""

import numpy as np
import torch
import time
import logging
import os
from tqdm import tqdm


class RLTrainer:
    def __init__(self, env, agent, config):
        """
        Initialize the RL Trainer

        Args:
            env: Trading environment
            agent: RL agent
            config: Configuration settings
        """
        self.env = env
        self.agent = agent
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Training parameters
        self.episodes = config.EPISODES
        self.warmup_episodes = config.WARMUP_EPISODES
        self.batch_size = config.BATCH_SIZE
        self.target_update_freq = config.TARGET_UPDATE_FREQUENCY

        # Performance tracking
        self.episode_rewards = []
        self.episode_profits = []
        self.episode_lengths = []
        self.epsilon_history = []
        self.loss_history = []

    def train(self, episodes=None):
        """
        Train the agent

        Args:
            episodes (int): Number of episodes to train for

        Returns:
            agent: Trained agent
        """
        if episodes is None:
            episodes = self.episodes

        self.logger.info(f"Starting training for {episodes} episodes")
        start_time = time.time()

        # Main training loop
        for episode in tqdm(range(episodes), desc="Training"):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = []

            # Episode loop
            done = False
            while not done:
                # Choose action
                action = self.agent.act(state, training=True)

                # Take action
                next_state, reward, done, info = self.env.step(action)

                # Remember experience
                self.agent.remember(state, action, reward, next_state, done)

                # Update state
                state = next_state
                episode_reward += reward

                # Experience replay
                if len(self.agent.memory) >= self.batch_size:
                    loss = self.agent.replay()
                    if loss is not None:
                        episode_loss.append(loss)

                # Update target network periodically
                if episode % self.target_update_freq == 0:
                    self.agent.update_target_network()

            # Track performance metrics
            self.episode_rewards.append(episode_reward)
            self.episode_profits.append(info['total_profit'])
            self.episode_lengths.append(self.env._current_tick - self.env._start_tick)
            self.epsilon_history.append(self.agent.epsilon)

            # Track loss
            if episode_loss:
                avg_loss = sum(episode_loss) / len(episode_loss)
                self.loss_history.append(avg_loss)
            else:
                self.loss_history.append(0)

            # Logging
            if episode % 10 == 0:
                self.logger.info(
                    f"Episode: {episode}, Reward: {episode_reward:.2f}, "
                    f"Profit: ${info['total_profit']:.2f}, "
                    f"Epsilon: {self.agent.epsilon:.4f}, "
                    f"Avg Loss: {self.loss_history[-1]:.6f}"
                )

            # Save checkpoint periodically
            if episode > 0 and episode % 100 == 0:
                checkpoint_path = os.path.join(
                    self.config.MODEL_DIR, f"dqn_checkpoint_ep{episode}.pth"
                )
                self.agent.save_model(checkpoint_path)

        # Final metrics
        train_time = time.time() - start_time
        self.logger.info(f"Training completed in {train_time:.2f} seconds")
        self.logger.info(f"Final epsilon: {self.agent.epsilon:.4f}")

        # Calculate performance statistics
        avg_reward = sum(self.episode_rewards[-100:]) / min(100, len(self.episode_rewards))
        avg_profit = sum(self.episode_profits[-100:]) / min(100, len(self.episode_profits))
        self.logger.info(f"Average reward (last 100 episodes): {avg_reward:.2f}")
        self.logger.info(f"Average profit (last 100 episodes): ${avg_profit:.2f}")

        # Save final model
        final_model_path = os.path.join(self.config.MODEL_DIR, "dqn_final.pth")
        self.agent.save_model(final_model_path)

        return self.agent

    def evaluate(self, episodes=10):
        """
        Evaluate the agent without exploration

        Args:
            episodes (int): Number of episodes for evaluation

        Returns:
            tuple: (average_reward, average_profit)
        """
        self.logger.info(f"Evaluating agent for {episodes} episodes")
        eval_rewards = []
        eval_profits = []

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0

            # Episode loop
            done = False
            while not done:
                # Choose action (no exploration)
                action = self.agent.act(state, training=False)

                # Take action
                next_state, reward, done, info = self.env.step(action)

                # Update state
                state = next_state
                episode_reward += reward

            # Track performance
            eval_rewards.append(episode_reward)
            eval_profits.append(info['total_profit'])

        # Calculate statistics
        avg_reward = sum(eval_rewards) / len(eval_rewards)
        avg_profit = sum(eval_profits) / len(eval_profits)

        self.logger.info(f"Evaluation results - Avg reward: {avg_reward:.2f}, Avg profit: ${avg_profit:.2f}")

        return avg_reward, avg_profit