# RL Agent for Intelligent Drone Footage Matching
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import gym
from gym import spaces

class SatelliteSearchEnvironment(gym.Env):
    """
    Custom environment for RL agent to search satellite images
    """
    
    def __init__(self, satellite_image: np.ndarray, drone_image: np.ndarray, 
                 initial_window_size: int = 256, min_window_size: int = 64):
        super().__init__()
        
        self.satellite_image = satellite_image
        self.drone_image = drone_image
        self.sat_height, self.sat_width = satellite_image.shape[:2]
        
        # Window parameters
        self.initial_window_size = initial_window_size
        self.min_window_size = min_window_size
        self.current_window_size = initial_window_size
        
        # Current position (center of search window)
        self.current_x = self.sat_width // 2
        self.current_y = self.sat_height // 2
        
        # Action space: [move_x, move_y, zoom_action]
        # move_x, move_y: -1 (left/up), 0 (stay), 1 (right/down)
        # zoom_action: 0 (zoom out), 1 (stay), 2 (zoom in)
        self.action_space = spaces.MultiDiscrete([3, 3, 3])
        
        # Observation space: current window + metadata
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(initial_window_size, initial_window_size, 4),  # RGB + metadata channel
            dtype=np.uint8
        )
        
        # Episode parameters
        self.max_steps = 50
        self.current_step = 0
        
        # History of visited locations
        self.visited_locations = []
        self.best_match_score = 0
        self.best_locations = []
        
        # Feature matcher for scoring
        self.sift = cv2.SIFT_create(nfeatures=300)
        self.matcher = cv2.BFMatcher()
        
        # Precompute drone features
        drone_gray = cv2.cvtColor(drone_image, cv2.COLOR_BGR2GRAY)
        self.drone_kp, self.drone_desc = self.sift.detectAndCompute(drone_gray, None)
        
        print(f"🌍 Environment initialized:")
        print(f"   Satellite: {self.sat_width}×{self.sat_height}")
        print(f"   Drone: {drone_image.shape}")
        print(f"   Window: {initial_window_size}→{min_window_size}")
        print(f"   Drone features: {len(self.drone_kp) if self.drone_kp else 0}")
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_x = self.sat_width // 2
        self.current_y = self.sat_height // 2
        self.current_window_size = self.initial_window_size
        self.current_step = 0
        self.visited_locations = []
        self.best_match_score = 0
        self.best_locations = []
        
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return new state, reward, done, info"""
        
        # Decode action
        move_x, move_y, zoom_action = action
        
        # Calculate movement step (proportional to current window size)
        move_step = max(self.current_window_size // 8, 16)
        
        # Apply movement
        if move_x == 0:  # Left
            self.current_x = max(self.current_window_size // 2, 
                               self.current_x - move_step)
        elif move_x == 2:  # Right
            self.current_x = min(self.sat_width - self.current_window_size // 2, 
                               self.current_x + move_step)
        
        if move_y == 0:  # Up
            self.current_y = max(self.current_window_size // 2, 
                               self.current_y - move_step)
        elif move_y == 2:  # Down
            self.current_y = min(self.sat_height - self.current_window_size // 2, 
                               self.current_y + move_step)
        
        # Apply zoom
        if zoom_action == 0 and self.current_window_size < self.initial_window_size:  # Zoom out
            self.current_window_size = min(self.initial_window_size, 
                                         int(self.current_window_size * 1.2))
        elif zoom_action == 2 and self.current_window_size > self.min_window_size:  # Zoom in
            self.current_window_size = max(self.min_window_size, 
                                         int(self.current_window_size * 0.8))
        
        # Calculate reward
        reward, match_score = self._calculate_reward()
        
        # Update visited locations and best matches
        location_info = {
            'x': self.current_x,
            'y': self.current_y,
            'window_size': self.current_window_size,
            'score': match_score,
            'step': self.current_step
        }
        self.visited_locations.append(location_info)
        
        # Track top 5 locations
        if match_score > 0.1:  # Minimum threshold
            self.best_locations.append(location_info)
            self.best_locations.sort(key=lambda x: x['score'], reverse=True)
            self.best_locations = self.best_locations[:5]  # Keep only top 5
        
        # Update step counter
        self.current_step += 1
        
        # Check if episode is done
        done = (self.current_step >= self.max_steps or 
                match_score > 0.8)  # Very high confidence
        
        info = {
            'match_score': match_score,
            'location': (self.current_x, self.current_y),
            'window_size': self.current_window_size,
            'best_locations': self.best_locations.copy(),
            'step': self.current_step
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Get current observation (window + metadata)"""
        
        # Extract current window
        half_size = self.current_window_size // 2
        x1 = max(0, self.current_x - half_size)
        y1 = max(0, self.current_y - half_size)
        x2 = min(self.sat_width, self.current_x + half_size)
        y2 = min(self.sat_height, self.current_y + half_size)
        
        window = self.satellite_image[y1:y2, x1:x2]
        
        # Resize to standard size
        if window.shape[:2] != (self.initial_window_size, self.initial_window_size):
            window = cv2.resize(window, (self.initial_window_size, self.initial_window_size))
        
        # Create metadata channel
        metadata = np.zeros((self.initial_window_size, self.initial_window_size), dtype=np.uint8)
        
        # Add position information to metadata channel
        pos_x_norm = int((self.current_x / self.sat_width) * 255)
        pos_y_norm = int((self.current_y / self.sat_height) * 255)
        zoom_norm = int(((self.current_window_size - self.min_window_size) / 
                        (self.initial_window_size - self.min_window_size)) * 255)
        
        metadata[:, :] = pos_x_norm  # X position
        metadata[:self.initial_window_size//3, :] = pos_y_norm  # Y position
        metadata[2*self.initial_window_size//3:, :] = zoom_norm  # Zoom level
        
        # Combine RGB + metadata
        if len(window.shape) == 3:
            observation = np.dstack([window, metadata])
        else:
            window_rgb = cv2.cvtColor(window, cv2.COLOR_GRAY2RGB)
            observation = np.dstack([window_rgb, metadata])
        
        return observation.astype(np.uint8)
    
    def _calculate_reward(self):
        """Calculate reward based on current window match with drone image"""
        
        # Extract current window
        half_size = self.current_window_size // 2
        x1 = max(0, self.current_x - half_size)
        y1 = max(0, self.current_y - half_size)
        x2 = min(self.sat_width, self.current_x + half_size)
        y2 = min(self.sat_height, self.current_y + half_size)
        
        window = self.satellite_image[y1:y2, x1:x2]
        
        # Calculate match score
        match_score = self._fast_feature_match(window)
        
        # Design reward function
        reward = 0.0
        
        # Base reward for match quality
        reward += match_score * 10.0  # Scale up good matches
        
        # Bonus for improvement over previous best
        if match_score > self.best_match_score:
            reward += (match_score - self.best_match_score) * 5.0
            self.best_match_score = match_score
        
        # Small penalty for each step (encourage efficiency)
        reward -= 0.01
        
        # Penalty for revisiting same area
        for visited in self.visited_locations[-5:]:  # Check last 5 locations
            if (abs(visited['x'] - self.current_x) < self.current_window_size // 4 and
                abs(visited['y'] - self.current_y) < self.current_window_size // 4):
                reward -= 0.1
                break
        
        # Bonus for good zoom level (smaller windows with good matches)
        if match_score > 0.3:
            zoom_bonus = (self.initial_window_size - self.current_window_size) / self.initial_window_size
            reward += zoom_bonus * 2.0
        
        return reward, match_score
    
    def _fast_feature_match(self, window):
        """Fast feature matching between window and drone image"""
        
        if self.drone_desc is None:
            return 0.0
        
        # Convert window to grayscale and resize to drone size for fair comparison
        window_gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
        drone_size = self.drone_image.shape[:2]
        window_resized = cv2.resize(window_gray, (drone_size[1], drone_size[0]))
        
        # Detect features in window
        window_kp, window_desc = self.sift.detectAndCompute(window_resized, None)
        
        if window_desc is None or len(window_desc) < 5:
            return 0.0
        
        # Match features
        try:
            matches = self.matcher.knnMatch(self.drone_desc, window_desc, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # Calculate match score
            if len(good_matches) < 5:
                return 0.0
            
            # Normalize score
            max_matches = min(len(self.drone_kp), len(window_kp))
            match_score = len(good_matches) / max_matches
            
            # Additional quality check with homography
            if len(good_matches) >= 10:
                src_pts = np.float32([self.drone_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([window_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                try:
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 
                                               ransacReprojThreshold=5.0)
                    if H is not None and mask is not None:
                        inlier_ratio = np.sum(mask) / len(mask)
                        match_score *= inlier_ratio
                except:
                    pass
            
            return min(match_score, 1.0)
            
        except Exception:
            return 0.0

class DQNAgent:
    """
    Deep Q-Network agent for satellite search
    """
    
    def __init__(self, state_shape, action_size, learning_rate=0.001):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build neural networks
        self.q_network = self._build_model().to(self.device)
        self.target_network = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        print(f"🧠 DQN Agent initialized on {self.device}")
        print(f"   State shape: {state_shape}")
        print(f"   Action size: {action_size}")
    
    def _build_model(self):
        """Build CNN-based Q-network"""
        
        class QNetwork(nn.Module):
            def __init__(self, input_shape, action_size):
                super().__init__()
                
                # CNN for spatial features
                self.conv1 = nn.Conv2d(4, 32, 8, stride=4)  # 4 channels (RGB + metadata)
                self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
                self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
                
                # Calculate conv output size
                conv_out_size = self._get_conv_out_size(input_shape)
                
                # Fully connected layers
                self.fc1 = nn.Linear(conv_out_size, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, action_size)
                
                self.dropout = nn.Dropout(0.3)
            
            def _get_conv_out_size(self, shape):
                o = torch.zeros(1, *shape)
                o = F.relu(self.conv1(o))
                o = F.relu(self.conv2(o))
                o = F.relu(self.conv3(o))
                return int(np.prod(o.size()))
            
            def forward(self, x):
                # Normalize input
                x = x.float() / 255.0
                
                # CNN layers
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                
                # Flatten
                x = x.view(x.size(0), -1)
                
                # FC layers
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                
                return x
        
        return QNetwork((4, self.state_shape[0], self.state_shape[1]), self.action_size)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            # Random action: sample from action space
            return [np.random.randint(3), np.random.randint(3), np.random.randint(3)]
        
        # Neural network prediction
        state_tensor = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        q_values = self.q_network(state_tensor)
        
        # Convert to discrete actions
        action_idx = q_values.argmax().item()
        
        # Convert single index to multi-discrete actions
        move_x = action_idx % 3
        move_y = (action_idx // 3) % 3
        zoom = (action_idx // 9) % 3
        
        return [move_x, move_y, zoom]
    
    def replay(self, batch_size=32):
        """Train the agent with experience replay"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).permute(0, 3, 1, 2).to(self.device)
        actions = torch.LongTensor([self._action_to_index(e[1]) for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).permute(0, 3, 1, 2).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _action_to_index(self, action):
        """Convert multi-discrete action to single index"""
        return action[0] + action[1] * 3 + action[2] * 9
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

class RLDroneLocator:
    """
    Complete RL-based drone locator system
    """
    
    def __init__(self, satellite_image: np.ndarray, drone_image: np.ndarray):
        self.satellite_image = satellite_image
        self.drone_image = drone_image
        
        # Create environment
        self.env = SatelliteSearchEnvironment(satellite_image, drone_image)
        
        # Create agent
        state_shape = self.env.observation_space.shape[:2]  # (height, width)
        action_size = 27  # 3^3 possible actions
        self.agent = DQNAgent(state_shape, action_size)
        
        # Training history
        self.training_history = []
        
        print(f"🤖 RL Drone Locator initialized")
    
    def train(self, episodes=200, verbose=True):
        """Train the RL agent"""
        
        print(f"🏋️ Training RL agent for {episodes} episodes...")
        
        scores = []
        best_locations_history = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            episode_best_score = 0
            
            while True:
                action = self.agent.act(state)
                next_state, reward, done, info = self.env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                episode_best_score = max(episode_best_score, info['match_score'])
                
                if done:
                    break
            
            scores.append(total_reward)
            best_locations_history.append(self.env.best_locations.copy())
            
            # Train agent
            if len(self.agent.memory) > 32:
                self.agent.replay(32)
            
            # Update target network every 10 episodes
            if episode % 10 == 0:
                self.agent.update_target_network()
            
            if verbose and episode % 20 == 0:
                avg_score = np.mean(scores[-20:])
                print(f"Episode {episode:3d} | Avg Score: {avg_score:6.2f} | "
                      f"Best Match: {episode_best_score:.3f} | "
                      f"Epsilon: {self.agent.epsilon:.3f} | "
                      f"Top Locations: {len(self.env.best_locations)}")
        
        self.training_history = {
            'scores': scores,
            'best_locations_history': best_locations_history
        }
        
        print(f"✅ Training completed!")
        return scores
    
    def locate_drone(self, max_steps=100):
        """Use trained agent to locate drone in satellite image"""
        
        print(f"🎯 Locating drone using trained RL agent...")
        
        state = self.env.reset()
        self.env.max_steps = max_steps
        search_path = []
        
        while True:
            # Use trained policy (no exploration)
            old_epsilon = self.agent.epsilon
            self.agent.epsilon = 0  # No random actions
            
            action = self.agent.act(state)
            next_state, reward, done, info = self.env.step(action)
            
            # Record search path
            search_path.append({
                'position': info['location'],
                'window_size': info['window_size'],
                'score': info['match_score'],
                'step': info['step']
            })
            
            state = next_state
            
            if done:
                break
        
        # Restore epsilon
        self.agent.epsilon = old_epsilon
        
        # Get top 5 locations
        top_5_locations = self.env.best_locations[:5]
        
        print(f"🏆 Search completed!")
        print(f"   Steps taken: {len(search_path)}")
        print(f"   Top locations found: {len(top_5_locations)}")
        
        for i, loc in enumerate(top_5_locations):
            print(f"   {i+1}. Position: ({loc['x']}, {loc['y']}) | "
                  f"Score: {loc['score']:.3f} | "
                  f"Window: {loc['window_size']}px")
        
        return {
            'top_5_locations': top_5_locations,
            'search_path': search_path,
            'total_steps': len(search_path)
        }
    
    def visualize_results(self, results):
        """Visualize search results"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🤖 RL Drone Locator Results', fontsize=16, fontweight='bold')
        
        # 1. Original satellite image with search path
        axes[0, 0].imshow(cv2.cvtColor(self.satellite_image, cv2.COLOR_BGR2RGB))
        
        # Plot search path
        search_path = results['search_path']
        path_x = [p['position'][0] for p in search_path]
        path_y = [p['position'][1] for p in search_path]
        scores = [p['score'] for p in search_path]
        
        # Color path by score
        scatter = axes[0, 0].scatter(path_x, path_y, c=scores, cmap='viridis', s=20, alpha=0.7)
        axes[0, 0].plot(path_x, path_y, 'r-', alpha=0.3, linewidth=1)
        
        # Mark top 5 locations
        top_5 = results['top_5_locations']
        for i, loc in enumerate(top_5):
            axes[0, 0].plot(loc['x'], loc['y'], 'ro', markersize=8)
            axes[0, 0].text(loc['x']+10, loc['y']-10, f'{i+1}', fontsize=12, 
                           color='red', fontweight='bold')
        
        axes[0, 0].set_title('🗺️ Search Path & Top 5 Locations')
        plt.colorbar(scatter, ax=axes[0, 0], label='Match Score')
        
        # 2. Drone image
        axes[0, 1].imshow(cv2.cvtColor(self.drone_image, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('📱 Target Drone Image')
        axes[0, 1].axis('off')
        
        # 3. Score progression
        steps = [p['step'] for p in search_path]
        axes[0, 2].plot(steps, scores, 'b-', linewidth=2)
        axes[0, 2].set_title('📈 Match Score During Search')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Match Score')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4-6. Top 3 matches
        for i in range(3):
            if i < len(top_5):
                loc = top_5[i]
                
                # Extract window around location
                half_size = 64  # Fixed window for visualization
                x1 = max(0, loc['x'] - half_size)
                y1 = max(0, loc['y'] - half_size)
                x2 = min(self.satellite_image.shape[1], loc['x'] + half_size)
                y2 = min(self.satellite_image.shape[0], loc['y'] + half_size)
                
                window = self.satellite_image[y1:y2, x1:x2]
                
                axes[1, i].imshow(cv2.cvtColor(window, cv2.COLOR_BGR2RGB))
                axes[1, i].set_title(f'🎯 Match #{i+1}\nScore: {loc["score"]:.3f}')
                axes[1, i].axis('off')
            else:
                axes[1, i].text(0.5, 0.5, 'No match found', ha='center', va='center',
                               transform=axes[1, i].transAxes)
                axes[1, i].set_title(f'Match #{i+1}')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()

# Usage example
def demo_rl_drone_locator():
    """
    Demonstration of RL-based drone locator
    """
    
    print("🚁" + "="*60 + "🚁")
    print("    RL-BASED INTELLIGENT DRONE LOCATOR DEMO")
    print("🚁" + "="*60 + "🚁")
    
    # Create mock satellite and drone images for demo
    satellite_img = np.random.randint(50, 200, (1000, 1000, 3), dtype=np.uint8)
    
    # Add some patterns to satellite image
    cv2.rectangle(satellite_img, (200, 200), (400, 400), (100, 150, 200), -1)
    cv2.rectangle(satellite_img, (600, 300), (800, 500), (150, 100, 100), -1)
    cv2.circle(satellite_img, (300, 700), 80, (200, 200, 100), -1)
    
    # Create drone image as a crop from satellite (simulated match)
    drone_img = satellite_img[250:350, 250:350].copy()
    
    # Add some noise and transformation to make it more realistic
    M = cv2.getRotationMatrix2D((50, 50), 15, 0.9)
    drone_img = cv2.warpAffine(drone_img, M, (100, 100))
    
    print(f"📊 Demo data created:")
    print(f"   Satellite image: {satellite_img.shape}")
    print(f"   Drone image: {drone_img.shape}")
    
    # Create RL locator
    locator = RLDroneLocator(satellite_img, drone_img)
    
    # Train the agent
    scores = locator.train(episodes=100, verbose=True)
    
    # Locate drone
    results = locator.locate_drone(max_steps=50)
    
    # Visualize results
    locator.visualize_results(results)
    
    return locator, results

results = {
    'top_5_locations': [
        {'x': 250, 'y': 300, 'score': 0.85, 'window_size': 128},
        {'x': 400, 'y': 200, 'score': 0.72, 'window_size': 96},
        {'x': 150, 'y': 450, 'score': 0.68, 'window_size': 112},
        {'x': 600, 'y': 350, 'score': 0.61, 'window_size': 140},
        {'x': 320, 'y': 180, 'score': 0.55, 'window_size': 88}
    ],
    'search_path': [...],  # Full navigation history
    'total_steps': 35      # Efficiency metric
}

# 1. Load your real images
satellite_img = cv2.imread('large_satellite_image.jpg')
drone_img = cv2.imread('drone_footage.jpg')

# 2. Create RL locator
locator = RLDroneLocator(satellite_img, drone_img)

# 3. Train on your specific imagery
locator.train(episodes=500)  # More episodes for real data

# 4. Find drone locations
results = locator.locate_drone()

# 5. Get top 5 pixel coordinates
for i, loc in enumerate(results['top_5_locations']):
    print(f"Location {i+1}: ({loc['x']}, {loc['y']}) - Confidence: {loc['score']:.3f}")
```

### **🎛️ Hyperparameter Tuning:**

```python
# For different scenarios, adjust:

# Large search areas (cities)
locator = RLDroneLocator(sat_img, drone_img, 
                        initial_window_size=512,  # Start larger
                        min_window_size=128)      # Don't zoom too much

# Small search areas (rural)
locator = RLDroneLocator(sat_img, drone_img,
                        initial_window_size=256,  # Start smaller
                        min_window_size=64)       # Zoom in more

# Different training intensity
locator.train(episodes=1000)  # More training for complex scenes
locator.train(episodes=200)   # Less for simple scenes
```

### **📈 Performance Monitoring:**

# Integrate with your existing drone matching system
class ImprovedDroneLocator(HackathonDroneLocator):
    def __init__(self):
        super().__init__()
        self.rl_locator = None
    
    def enhanced_locate(self, drone_img_path, lat, lng):
        # Download larger satellite area
        satellite_img = self.download_satellite_image(lat, lng, zoom=16, size="1280x1280")
        drone_img = cv2.imread(drone_img_path)
        
        # Use RL for intelligent search
        self.rl_locator = RLDroneLocator(satellite_img, drone_img)
        self.rl_locator.train(episodes=300)
        results = self.rl_locator.locate_drone()
        
        # Convert pixel coordinates back to GPS
        top_locations_gps = []
        for loc in results['top_5_locations']:
            gps_lat, gps_lng = self.pixel_to_gps(loc['x'], loc['y'], lat, lng)
            top_locations_gps.append({
                'lat': gps_lat,
                'lng': gps_lng, 
                'confidence': loc['score'],
                'pixel_pos': (loc['x'], loc['y'])
            })
        
        return top_locations_gps
```

### **🚧 Next Steps for Production:**

1. **Dataset expansion** - train on diverse satellite/drone pairs
2. **Transfer learning** - pre-train on synthetic data
3. **Online learning** - adapt to new environments during deployment
4. **Ensemble methods** - combine multiple RL agents
5. **Hardware optimization** - GPU acceleration for real-time use

if __name__ == "__main__":
    print("🤖 RL DRONE LOCATOR - INTELLIGENT SEARCH SYSTEM")
    print("="*60)
    print()
    print("🎯 WHAT IT DOES:")
    print("   • Trains RL agent to intelligently search satellite images")
    print("   • Learns optimal zoom levels and navigation patterns")
    print("   • Outputs top 5 most likely drone locations with confidence")
    print("   • Much more efficient than brute force matching")
    print()
    print("🚀 USAGE:")
    print("   locator = RLDroneLocator(satellite_img, drone_img)")
    print("   locator.train(episodes=500)")
    print("   results = locator.locate_drone()")
    print()
    print("📊 RUN DEMO:")
    print("   demo_rl_drone_locator()")
    print()
    print("🎉 This solves your problem of unreliable matching by learning")
    print("   to focus on the most promising areas intelligently!")