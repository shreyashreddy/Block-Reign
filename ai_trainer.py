#!/usr/bin/env python3
import json
import os
import random
import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from collections import deque, namedtuple
import pickle
from datetime import datetime

# Define experience tuple 💭
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer for reinforcement learning"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity) # 💾 Buffer storage
    
    def push(self, experience):
        """Add experience to buffer"""
        self.buffer.append(experience) # ➕ Append experience
    
    def sample(self, batch_size):
        """Sample random batch of experiences"""
        if (len(self.buffer) < batch_size): # ❌ Not enough data
            return None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False) # 🎲 Random indices
        batch = [self.buffer[i] for i in indices] # 📦 Collect batch
        
        states = torch.FloatTensor([exp.state for exp in batch]) # 🔢 State tensors
        actions = torch.LongTensor([exp.action for exp in batch]) # 🚀 Action tensors
        rewards = torch.FloatTensor([exp.reward for exp in batch]) # 💰 Reward tensors
        next_states = torch.FloatTensor([exp.next_state for exp in batch]) # ➡️ Next state tensors
        dones = torch.FloatTensor([exp.done for exp in batch]) # ✅ Done tensors
        
        return states, actions, rewards, next_states, dones # 📤 Return batch
    
    def __len__(self):
        return len(self.buffer) # 📏 Buffer size

class DQN(nn.Module):
    """Deep Q-Network for AI decision making"""
    def __init__(self, state_size=6, action_size=5, hidden_size=128):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size), # 🧱 Input layer
            nn.ReLU(), # ✨ Activation function
            nn.Linear(hidden_size, hidden_size), # 🧱 Hidden layer
            nn.ReLU(), # ✨ Activation function
            nn.Linear(hidden_size, hidden_size), # 🧱 Hidden layer
            nn.ReLU(), # ✨ Activation function
            nn.Linear(hidden_size, action_size) # 🧱 Output layer
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), # 🧱 Value hidden
            nn.ReLU(), # ✨ Activation
            nn.Linear(hidden_size, 1) # 🧱 Value output
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), # 🧱 Adv hidden
            nn.ReLU(), # ✨ Activation
            nn.Linear(hidden_size, action_size) # 🧱 Adv output
        )
    
    def forward(self, x):
        x = self.network(x) # 🧠 Process input
        
        value = self.value_stream(x) # 📊 Calculate value
        advantages = self.advantage_stream(x) # 📈 Calculate advantages
        
        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True)) # 🧮 Q-value calculation
        
        return q_values # ➡️ Return Q-values

class AdvancedAITrainer:
    """Advanced AI trainer with reinforcement learning"""
    def __init__(self):
        self.state_size = 6 # 📏 State dimension
        self.action_size = 5 # 📏 Action dimension
        self.hidden_size = 128 # 📏 Hidden layer size
        
        # Initialize networks
        self.policy_net = DQN(self.state_size, self.action_size, self.hidden_size) # 🤖 Policy network
        self.target_net = DQN(self.state_size, self.action_size, self.hidden_size) # 🎯 Target network
        self.target_net.load_state_dict(self.policy_net.state_dict()) # 🔄 Copy weights
        
        # Initialize optimizer and replay buffer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001) # ⚙️ Adam optimizer
        self.replay_buffer = ReplayBuffer(capacity=10000) # 📦 Replay buffer
        
        # Training parameters
        self.batch_size = 64 # 🔢 Batch size
        self.gamma = 0.99  # Discount factor 📉
        self.tau = 0.005  # Soft update parameter 🛠️
        self.epsilon = 1.0  # Exploration rate 🌟
        self.epsilon_min = 0.01 # Minimum exploration 🤏
        self.epsilon_decay = 0.995 # Exploration decay rate 📉
        
        # Checkpoint paths
        self.checkpoint_dir = 'training/models' # 📂 Checkpoint directory
        self.replay_buffer_file = os.path.join(self.checkpoint_dir, 'replay_buffer.pkl') # 💾 Replay buffer file
        self.policy_net_file = os.path.join(self.checkpoint_dir, 'policy_net.pth') # 💾 Policy net file
        self.target_net_file = os.path.join(self.checkpoint_dir, 'target_net.pth') # 💾 Target net file
        self.training_log_file = os.path.join(self.checkpoint_dir, 'training_log.json') # 📈 Training log file
        
        # Load from checkpoint if exists
        self.load_checkpoint() # 🗄️ Load existing checkpoint
        
        # Training statistics
        self.training_stats = {
            'episodes': 0, # 🔢 Episode count
            'total_reward': 0, # 💰 Total reward
            'loss_history': [], # 📉 Loss history
            'epsilon_history': [], # 🌟 Epsilon history
            'success_rate': 0 # 🏆 Success rate
        }
    
    def load_checkpoint(self):
        """Load model and replay buffer from checkpoint"""
        try:
            if os.path.exists(self.policy_net_file): # 🔍 Policy net exists
                self.policy_net.load_state_dict(torch.load(self.policy_net_file)) # 💽 Load policy weights
                print("✓ Loaded policy network from checkpoint") # ✅ Success message
            
            if os.path.exists(self.target_net_file): # 🔍 Target net exists
                self.target_net.load_state_dict(torch.load(self.target_net_file)) # 💽 Load target weights
                print("✓ Loaded target network from checkpoint") # ✅ Success message
            
            if os.path.exists(self.replay_buffer_file): # 🔍 Replay buffer exists
                with open(self.replay_buffer_file, 'rb') as f: # 📖 Open buffer file
                    self.replay_buffer.buffer = pickle.load(f) # 💽 Load buffer data
                print(f"✓ Loaded {len(self.replay_buffer)} experiences from replay buffer") # ✅ Success message
            
            if os.path.exists(self.training_log_file): # 🔍 Log file exists
                with open(self.training_log_file, 'r') as f: # 📖 Open log file
                    self.training_stats = json.load(f) # 💽 Load stats
                print("✓ Loaded training statistics") # ✅ Success message
                
        except Exception as e: # ❌ Error occurred
            print(f"Error loading checkpoint: {e}") # 😥 Error message
    
    def save_checkpoint(self):
        """Save model and replay buffer to checkpoint"""
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True) # 📂 Create directory
            
            # Save networks
            torch.save(self.policy_net.state_dict(), self.policy_net_file) # 💾 Save policy
            torch.save(self.target_net.state_dict(), self.target_net_file) # 💾 Save target
            
            # Save replay buffer
            with open(self.replay_buffer_file, 'wb') as f: # 🗄️ Open buffer file
                pickle.dump(list(self.replay_buffer.buffer), f) # 💾 Save buffer
            
            # Save training statistics
            with open(self.training_log_file, 'w') as f: # 🗄️ Open log file
                json.dump(self.training_stats, f, indent=2) # 💾 Save stats
            
            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # ⏰ Generate timestamp
            backup_dir = os.path.join('training/checkpoints', timestamp) # 📂 Backup directory
            os.makedirs(backup_dir, exist_ok=True) # 📂 Create backup dir
            
            torch.save(self.policy_net.state_dict(), os.path.join(backup_dir, 'policy_net.pth')) # 💾 Backup policy
            
            print("✓ Checkpoint saved successfully") # ✅ Success message
            return True # 👍 Success
            
        except Exception as e: # ❌ Error occurred
            print(f"Error saving checkpoint: {e}") # 😥 Error message
            return False # 👎 Failure
    
    def get_action(self, state, training=True):
        """Get action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon: # 🏃 Exploration phase
            # Explore: random action
            return random.randint(0, self.action_size - 1) # 🎲 Random action
        
        # Exploit: use policy network
        with torch.no_grad(): # 🚫 Disable gradient calculation
            state_tensor = torch.FloatTensor(state).unsqueeze(0) # 🔢 Convert state tensor
            q_values = self.policy_net(state_tensor) # 🧠 Predict Q-values
            action = q_values.argmax().item() # 🏆 Choose best action
        
        return action # ➡️ Return action

    import torch.nn as nn # type: ignore
import torch # type: ignore
import numpy as np
import os

class Experience:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states = torch.tensor([self.buffer[i].state for i in batch], dtype=torch.float32)
        actions = torch.tensor([self.buffer[i].action for i in batch], dtype=torch.int64)
        rewards = torch.tensor([self.buffer[i].reward for i in batch], dtype=torch.float32)
        next_states = torch.tensor([self.buffer[i].next_state for i in batch], dtype=torch.float32)
        dones = torch.tensor([self.buffer[i].done for i in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class PolicyNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class TargetNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(TargetNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class AdvancedAITrainer:
    def __init__(self, capacity=10000, batch_size=64, gamma=0.99, tau=0.001, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, lr=0.0001):
        self.input_size = 6  # state size
        self.output_size = 5 # action size
        self.capacity = capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        
        self.policy_net = PolicyNet(self.input_size, self.output_size)
        self.target_net = TargetNet(self.input_size, self.output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(self.capacity)
        
        self.policy_net_file = "policy_net.pth"
        self.target_net_file = "target_net.pth"
        self.optimizer_file = "optimizer.pth"
        self.replay_buffer_file = "replay_buffer.pth"
        self.training_stats_file = "training_stats.pth"
        
        self.training_stats = {
            'episodes': 0,
            'loss_history': [],
            'epsilon_history': [],
            'success_rate': 0.0  # Placeholder
        }
        
        self.load_checkpoint()
        
    def load_checkpoint(self):
        if os.path.exists(self.policy_net_file):
            self.policy_net.load_state_dict(torch.load(self.policy_net_file))
            self.target_net.load_state_dict(torch.load(self.target_net_file))
            self.optimizer.load_state_dict(torch.load(self.optimizer_file))
            
            # Load replay buffer
            if os.path.exists(self.replay_buffer_file):
                with open(self.replay_buffer_file, 'rb') as f:
                    buffer_data = pickle.load(f)
                    self.replay_buffer.buffer = buffer_data['buffer']
                    self.replay_buffer.position = buffer_data['position']
            
            # Load training stats
            if os.path.exists(self.training_stats_file):
                self.training_stats = torch.load(self.training_stats_file)
                self.epsilon = self.training_stats.get('epsilon', self.epsilon) # Load epsilon if present

            print("Checkpoint loaded successfully.")
        else:
            print("No checkpoint found. Starting fresh.")

    def save_checkpoint(self):
        torch.save(self.policy_net.state_dict(), self.policy_net_file)
        torch.save(self.target_net.state_dict(), self.target_net_file)
        torch.save(self.optimizer.state_dict(), self.optimizer_file)
        
        # Save replay buffer
        buffer_data = {
            'buffer': self.replay_buffer.buffer,
            'position': self.replay_buffer.position
        }
        with open(self.replay_buffer_file, 'wb') as f:
            pickle.dump(buffer_data, f)
        
        # Save training stats
        self.training_stats['epsilon'] = self.epsilon # Update epsilon in stats
        torch.save(self.training_stats, self.training_stats_file)
        print("Checkpoint saved.")

    def get_action(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return np.random.randint(self.output_size)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_values = self.policy_net(state)
            return torch.argmax(action_values, dim=1).item()

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