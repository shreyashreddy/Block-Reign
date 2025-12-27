#!/usr/bin/env python3
"""
Player vs AI Game Server - Clean Version
"""

import json
import os
import random
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle

os.makedirs('training/models', exist_ok=True)

app = Flask(__name__, 
            static_folder='game/static',
            template_folder='game/templates')
CORS(app)

class SimpleAI:
    """Simple Q-Learning AI"""
    
    def __init__(self):
        # Game parameters
        self.grid_size = 10
        self.actions = ['move_up', 'move_down', 'move_left', 'move_right', 'shoot']
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        self.idx_to_action = {i: a for i, a in enumerate(self.actions)}
        
        # Q-learning parameters
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
        # Statistics
        self.wins = 0
        self.losses = 0
        self.total_matches = 0
        
        # Files
        self.model_file = 'training/models/simple_ai.pkl'
        
        # Load existing model
        self.load_model()
        
        print(f"🤖 AI Initialized!")
        print(f"   Q-table: {len(self.q_table)} states")
        print(f"   Record: {self.wins} wins, {self.losses} losses")
        print(f"   Epsilon: {self.epsilon:.3f}")
    
    def get_state_key(self, game_state):
        # More detailed states = slower learning
        distance = abs(game_state['player_x'] - game_state['ai_x']) + abs(game_state['player_y'] - game_state['ai_y'])
        # 5 distance categories instead of 3
        if distance <= 1: dist = 'touching'
        elif distance <= 3: dist = 'very_close'
        elif distance <= 5: dist = 'close'
        elif distance <= 7: dist = 'medium'
        else: dist = 'far'
        
        # 4 health levels instead of 2
        if game_state['player_health'] > 75: ph = 'very_high'
        elif game_state['player_health'] > 50: ph = 'high'
        elif game_state['player_health'] > 25: ph = 'low'
        else: ph = 'very_low'
        
        # Same for AI health
        if game_state['ai_health'] > 75: ah = 'very_high'
        elif game_state['ai_health'] > 50: ah = 'high'
        elif game_state['ai_health'] > 25: ah = 'low'
        else: ah = 'very_low'
        
        # Add direction
        if game_state['player_x'] > game_state['ai_x']: dir_x = 'right'
        else: dir_x = 'left'
        
        if game_state['player_y'] > game_state['ai_y']: dir_y = 'below'
        else: dir_y = 'above'
        return f"{dist}_{ph}_{ah}_{dir_x}_{dir_y}"
    
    def choose_action(self, game_state):
        """Choose action using epsilon-greedy"""
        state_key = self.get_state_key(game_state)
        
        # Initialize Q-values for new state
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.actions}
        
        # Exploration vs Exploitation
        if random.random() < self.epsilon:
            # Explore with some intelligence
            actions = self.actions.copy()
            distance = abs(game_state['player_x'] - game_state['ai_x']) + abs(game_state['player_y'] - game_state['ai_y'])
            
            if distance <= 2 and game_state['ai_health'] > 30:
                # Close and healthy - prefer shooting
                if 'shoot' in actions and random.random() < 0.7:
                    action = 'shoot'
                else:
                    action = random.choice(actions)
            elif distance > 5:
                # Far away - prefer moving
                move_actions = [a for a in actions if a.startswith('move_')]
                action = random.choice(move_actions) if move_actions else random.choice(actions)
            else:
                action = random.choice(actions)
            
            return action, self.epsilon, 'exploring'
        else:
            # Exploit: choose best known action
            q_values = self.q_table[state_key]
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = random.choice(best_actions)
            
            return action, max_q, 'exploiting'
    
    def learn_from_match(self, match_moves, winner):
        """Learn from a completed match"""
        if not winner or winner not in ['player', 'ai']:
            print(f"⚠️ Match completed with no clear winner: {winner}")
            return
        
        self.total_matches += 1
        
        if winner == 'ai':
            self.wins += 1
        else:
            self.losses += 1
        
        # Process each AI move
        for i, move in enumerate(match_moves):
            if move['player'] != 'ai':
                continue
            
            # Get state before move
            if i == 0:
                continue
            
            prev_move = match_moves[i-1]
            state_before = {
                'player_x': prev_move['playerPosition']['x'],
                'player_y': prev_move['playerPosition']['y'],
                'player_health': prev_move['playerHealth'],
                'ai_x': prev_move['aiPosition']['x'],
                'ai_y': prev_move['aiPosition']['y'],
                'ai_health': prev_move['aiHealth']
            }
            
            state_key = self.get_state_key(state_before)
            action = move['action']
            
            # Calculate reward
            reward = self.calculate_reward(state_before, action, winner, i == len(match_moves)-1)
            
            # Initialize if new state
            if state_key not in self.q_table:
                self.q_table[state_key] = {a: 0.0 for a in self.actions}
            
            # Get next state's max Q
            next_max_q = 0
            if i < len(match_moves) - 1:
                next_state = {
                    'player_x': move['playerPosition']['x'],
                    'player_y': move['playerPosition']['y'],
                    'player_health': move['playerHealth'],
                    'ai_x': move['aiPosition']['x'],
                    'ai_y': move['aiPosition']['y'],
                    'ai_health': move['aiHealth']
                }
                next_state_key = self.get_state_key(next_state)
                if next_state_key in self.q_table:
                    next_max_q = max(self.q_table[next_state_key].values())
            
            # Q-learning update
            current_q = self.q_table[state_key][action]
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
            self.q_table[state_key][action] = new_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Save model
        self.save_model()
        
        print(f"✅ AI learned from match. Wins: {self.wins}, Losses: {self.losses}")
    
    def calculate_reward(self, state, action, winner, is_last_move):
        """Calculate reward for an action"""
        reward = 0
        
        distance = abs(state['player_x'] - state['ai_x']) + abs(state['player_y'] - state['ai_y'])
        ai_health = state['ai_health']
        player_health = state['player_health']
        
        # Action-based rewards
        if action == 'shoot':
            if distance <= 2:
                reward += 1.5  # Good shot at close range
            elif distance <= 4:
                reward += 0.5  # Medium range shot
            else:
                reward -= 0.8  # Wasteful shot
        else:  # Movement
            # Good to move toward injured player
            if player_health < 50 and distance > 2:
                reward += 0.3
            # Good to retreat when AI health is low
            elif ai_health < 30 and distance < 3:
                reward += 0.4
            else:
                reward -= 0.1  # Small penalty for moving
        
        # Match outcome reward (only for last move)
        if is_last_move:
            if winner == 'ai':
                reward += 3.0
            elif winner == 'player':
                reward -= 1.5
        
        return reward
    
    def save_model(self):
        """Save Q-table to file"""
        try:
            model_data = {
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'wins': self.wins,
                'losses': self.losses,
                'total_matches': self.total_matches
            }
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self):
        """Load Q-table from file"""
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.q_table = model_data.get('q_table', {})
                self.epsilon = model_data.get('epsilon', 1.0)
                self.wins = model_data.get('wins', 0)
                self.losses = model_data.get('losses', 0)
                self.total_matches = model_data.get('total_matches', 0)
                
                return True
        except:
            pass
        
        return False

# Initialize AI
ai = SimpleAI()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ai_move', methods=['POST'])
def ai_move():
    try:
        game_state = request.json
        
        action, value, mode = ai.choose_action(game_state)
        
        return jsonify({
            'action': action,
            'epsilon': ai.epsilon,
            'mode': mode,
            'success': True
        })
        
    except Exception as e:
        print(f"AI move error: {e}")
        return jsonify({
            'action': random.choice(['move_up', 'move_down', 'move_left', 'move_right', 'shoot']),
            'success': False
        }), 500

@app.route('/learn_from_match', methods=['POST'])
def learn_from_match():
    try:
        data = request.json
        moves = data.get('moves', [])
        winner = data.get('winner')
        match_num = data.get('match_number', 1)
        
        print(f"\n🏁 Match {match_num} complete - Winner: {winner}")
        
        ai.learn_from_match(moves, winner)
        
        return jsonify({
            'success': True,
            'ai_stats': {
                'wins': ai.wins,
                'losses': ai.losses,
                'epsilon': ai.epsilon,
                'q_table_size': len(ai.q_table)
            }
        })
        
    except Exception as e:
        print(f"Learn error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🤖 PLAYER vs AI - GRID BATTLE")
    print("="*60)
    print("Controls: Arrow Keys = Move, Space = Shoot")
    print(f"URL: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)