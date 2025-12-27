#!/usr/bin/env python3
"""
Advanced AI Trainer with Reinforcement Learning
"""

import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import pickle
from datetime import datetime

# Define experience tuple
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer for reinforcement learning"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample random batch of experiences"""
        if len(self.buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.FloatTensor([exp.state for exp in batch])
        actions = torch.LongTensor([exp.action for exp in batch])
        rewards = torch.FloatTensor([exp.reward for exp in batch])
        next_states = torch.FloatTensor([exp.next_state for exp in batch])
        dones = torch.FloatTensor([exp.done for exp in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    """Deep Q-Network for AI decision making"""
    def __init__(self, state_size=6, action_size=5, hidden_size=128):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x):
        x = self.network(x)
        
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values

class AdvancedAITrainer:
    """Advanced AI trainer with reinforcement learning"""
    def __init__(self):
        self.state_size = 6
        self.action_size = 5
        self.hidden_size = 128
        
        # Initialize networks
        self.policy_net = DQN(self.state_size, self.action_size, self.hidden_size)
        self.target_net = DQN(self.state_size, self.action_size, self.hidden_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer and replay buffer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Training parameters
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005  # Soft update parameter
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Checkpoint paths
        self.checkpoint_dir = 'training/models'
        self.replay_buffer_file = os.path.join(self.checkpoint_dir, 'replay_buffer.pkl')
        self.policy_net_file = os.path.join(self.checkpoint_dir, 'policy_net.pth')
        self.target_net_file = os.path.join(self.checkpoint_dir, 'target_net.pth')
        self.training_log_file = os.path.join(self.checkpoint_dir, 'training_log.json')
        
        # Load from checkpoint if exists
        self.load_checkpoint()
        
        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'loss_history': [],
            'epsilon_history': [],
            'success_rate': 0
        }
    
    def load_checkpoint(self):
        """Load model and replay buffer from checkpoint"""
        try:
            if os.path.exists(self.policy_net_file):
                self.policy_net.load_state_dict(torch.load(self.policy_net_file))
                print("✓ Loaded policy network from checkpoint")
            
            if os.path.exists(self.target_net_file):
                self.target_net.load_state_dict(torch.load(self.target_net_file))
                print("✓ Loaded target network from checkpoint")
            
            if os.path.exists(self.replay_buffer_file):
                with open(self.replay_buffer_file, 'rb') as f:
                    self.replay_buffer.buffer = pickle.load(f)
                print(f"✓ Loaded {len(self.replay_buffer)} experiences from replay buffer")
            
            if os.path.exists(self.training_log_file):
                with open(self.training_log_file, 'r') as f:
                    self.training_stats = json.load(f)
                print("✓ Loaded training statistics")
                
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    def save_checkpoint(self):
        """Save model and replay buffer to checkpoint"""
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            # Save networks
            torch.save(self.policy_net.state_dict(), self.policy_net_file)
            torch.save(self.target_net.state_dict(), self.target_net_file)
            
            # Save replay buffer
            with open(self.replay_buffer_file, 'wb') as f:
                pickle.dump(list(self.replay_buffer.buffer), f)
            
            # Save training statistics
            with open(self.training_log_file, 'w') as f:
                json.dump(self.training_stats, f, indent=2)
            
            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join('training/checkpoints', timestamp)
            os.makedirs(backup_dir, exist_ok=True)
            
            torch.save(self.policy_net.state_dict(), os.path.join(backup_dir, 'policy_net.pth'))
            
            print("✓ Checkpoint saved successfully")
            return True
            
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return False
    
    def get_action(self, state, training=True):
        """Get action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.action_size - 1)
        
        # Exploit: use policy network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.push(experience)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return 0
        
        states, actions, rewards, next_states, dones = batch
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update statistics
        self.training_stats['loss_history'].append(loss.item())
        self.training_stats['epsilon_history'].append(self.epsilon)
        
        return loss.item()
    
    def soft_update(self):
        """Soft update of the target network"""
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                             self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )
    
    def train(self, episodes=100):
        """Train the AI for specified number of episodes"""
        if len(self.replay_buffer) < self.batch_size:
            print("Not enough experiences in replay buffer")
            return 0
        
        print(f"Starting training with {len(self.replay_buffer)} experiences...")
        
        total_reward = 0
        losses = []
        
        for episode in range(episodes):
            episode_reward = 0
            episode_loss = 0
            steps = 0
            
            # Simulate multiple training steps per episode
            for _ in range(10):  # 10 training steps per episode
                loss = self.train_step()
                if loss > 0:
                    episode_loss += loss
                    steps += 1
            
            if steps > 0:
                avg_loss = episode_loss / steps
                losses.append(avg_loss)
                
                if episode % 10 == 0:
                    print(f"Episode {episode}, Avg Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.3f}")
            
            self.training_stats['episodes'] += 1
        
        # Save checkpoint
        self.save_checkpoint()
        
        # Calculate progress based on loss reduction
        if len(losses) > 0:
            initial_loss = losses[0]
            final_loss = losses[-1]
            
            if initial_loss > 0:
                loss_reduction = (initial_loss - final_loss) / initial_loss
                progress = min(100, max(0, loss_reduction * 100))
            else:
                progress = 50  # Default progress if no loss reduction
        else:
            progress = 0
        
        print(f"Training complete. Progress: {progress:.1f}%")
        return progress
    
    def process_game_data(self, match_data):
        """Process game match data and convert to experiences"""
        winner = match_data.get('winner')
        moves = match_data.get('moves', [])
        
        experiences_added = 0
        
        for i, move in enumerate(moves):
            state = self._extract_state(move['game_state'])
            action = self._action_to_index(move['action'])
            
            # Calculate reward based on move outcome
            reward = self._calculate_reward(move, winner)
            
            # Get next state if available
            next_state = None
            if i < len(moves) - 1:
                next_state = self._extract_state(moves[i + 1]['game_state'])
            else:
                # If last move, use current state (terminal)
                next_state = state
            
            done = 1 if i == len(moves) - 1 else 0
            
            # Add experience
            self.add_experience(state, action, reward, next_state, done)
            experiences_added += 1
        
        print(f"Processed {experiences_added} experiences from match")
        return experiences_added
    
    def _extract_state(self, game_state):
        """Extract normalized state from game state"""
        return [
            game_state['player_x'] / 9.0,
            game_state['player_y'] / 9.0,
            game_state['player_health'] / 100.0,
            game_state['ai_x'] / 9.0,
            game_state['ai_y'] / 9.0,
            game_state['ai_health'] / 100.0
        ]
    
    def _action_to_index(self, action):
        """Convert action string to index"""
        action_map = {
            'move_up': 0,
            'move_down': 1,
            'move_left': 2,
            'move_right': 3,
            'shoot': 4
        }
        return action_map.get(action, 0)
    
    def _index_to_action(self, index):
        """Convert index to action string"""
        action_map = [
            'move_up',
            'move_down',
            'move_left',
            'move_right',
            'shoot'
        ]
        return action_map[index]
    
    def _calculate_reward(self, move, winner):
        """Calculate reward for a move"""
        reward = 0
        
        # Base reward for taking action
        reward += 0.1
        
        # Extra reward for shooting
        if move['action'] == 'shoot':
            reward += 0.5
        
        # Reward for winning moves
        if move['actor'] == winner:
            reward += 1.0
        
        # Reward for strategic positioning
        game_state = move['game_state']
        player_x, player_y = game_state['player_x'], game_state['player_y']
        ai_x, ai_y = game_state['ai_x'], game_state['ai_y']
        
        distance = abs(player_x - ai_x) + abs(player_y - ai_y)
        
        if move['actor'] == 'ai':
            # AI gets rewarded for maintaining optimal distance (3-5 cells)
            if 3 <= distance <= 5:
                reward += 0.3
            # Penalty for getting too close
            elif distance < 2:
                reward -= 0.2
        
        return reward
    
    def get_stats(self):
        """Get training statistics"""
        return {
            'experiences': len(self.replay_buffer),
            'episodes': self.training_stats['episodes'],
            'epsilon': self.epsilon,
            'avg_loss': np.mean(self.training_stats['loss_history'][-100:]) if self.training_stats['loss_history'] else 0,
            'success_rate': self.training_stats['success_rate']
        }
    
    def predict_best_action(self, game_state):
        """Predict best action for given game state"""
        state = self._extract_state(game_state)
        action_index = self.get_action(state, training=False)
        return self._index_to_action(action_index)

# Export functions for use in main server
def create_ai_trainer():
    """Create and return an AI trainer instance"""
    return AdvancedAITrainer()

if __name__ == '__main__':
    # Test the AI trainer
    trainer = AdvancedAITrainer()
    
    print("AI Trainer initialized successfully!")
    print(f"Replay buffer size: {len(trainer.replay_buffer)}")
    print(f"Epsilon: {trainer.epsilon}")
    print(f"Checkpoint loaded: {os.path.exists(trainer.policy_net_file)}")
    
    # Test prediction
    test_state = {
        'player_x': 0,
        'player_y': 0,
        'player_health': 100,
        'ai_x': 9,
        'ai_y': 9,
        'ai_health': 100
    }
    
    action = trainer.predict_best_action(test_state)
    print(f"Test prediction: {action}")