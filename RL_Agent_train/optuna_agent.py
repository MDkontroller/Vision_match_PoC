# Optuna Hyperparameter Optimization for Drone Localization RL Agent
import optuna
import torch
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any

# Import your existing classes (assuming they're in the same file or imported)
# from drone_localization_agent import DroneLocalizationEnvironment, PPODroneAgent, DroneLocalizationNetwork

class OptimizedDroneLocalizationNetwork(torch.nn.Module):
    """Optimizable version of the drone localization network"""
    
    def __init__(self, grid_size: int, feature_dim: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.grid_size = grid_size
        self.num_locations = grid_size * grid_size
        
        # CNN backbone for image processing
        self.tif_encoder = self._create_cnn_encoder(feature_dim, dropout_rate)
        self.crop_encoder = self._create_cnn_encoder(feature_dim, dropout_rate)
        
        # Cross-attention between TIF and crop
        self.cross_attention = torch.nn.MultiheadAttention(feature_dim, num_heads=num_heads, batch_first=True)
        
        # Spatial reasoning
        self.spatial_reasoning = torch.nn.Sequential(
            torch.nn.Linear(feature_dim * 2, feature_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(feature_dim, feature_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate)
        )
        
        # Output heads
        self.location_head = torch.nn.Linear(feature_dim, self.num_locations)
        self.value_head = torch.nn.Linear(feature_dim, 1)
        
    def _create_cnn_encoder(self, feature_dim: int, dropout_rate: float):
        """Create optimizable CNN encoder"""
        import torchvision.models as models
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        
        encoder = torch.nn.Sequential(
            *list(resnet.children())[:-2],
            torch.nn.AdaptiveAvgPool2d((8, 8)),
            torch.nn.Flatten(),
            torch.nn.Linear(512 * 64, feature_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate)
        )
        
        return encoder
    
    def forward(self, tif_image: torch.Tensor, crop_image: torch.Tensor):
        # Encode images
        tif_features = self.tif_encoder(tif_image)
        crop_features = self.crop_encoder(crop_image)
        
        # Cross-attention
        tif_attended, _ = self.cross_attention(
            crop_features.unsqueeze(1),
            tif_features.unsqueeze(1),
            tif_features.unsqueeze(1)
        )
        
        # Combine features
        combined_features = torch.cat([
            tif_attended.squeeze(1), 
            crop_features
        ], dim=1)
        
        # Spatial reasoning
        spatial_features = self.spatial_reasoning(combined_features)
        
        # Output predictions
        location_logits = self.location_head(spatial_features)
        location_probs = torch.nn.functional.softmax(location_logits, dim=1)
        value = self.value_head(spatial_features)
        
        return location_probs, value, location_logits

class OptimizedDroneLocalizationTrainer:
    """Optimizable trainer with configurable hyperparameters"""
    
    def __init__(self, trial_params: Dict[str, Any]):
        self.params = trial_params
        
        # Initialize environment
        self.env = DroneLocalizationEnvironment(
            tif_image_path=None,
            crops_metadata_path=None, 
            grid_size=trial_params['grid_size']
        )
        
        # Initialize optimized agent
        self.agent = self._create_optimized_agent(trial_params)
        
        # Training metrics
        self.episode_rewards = []
        self.similarity_scores = []
        self.convergence_episodes = []
        
    def _create_optimized_agent(self, params: Dict[str, Any]):
        """Create agent with optimized hyperparameters"""
        
        # Create optimized network
        network = OptimizedDroneLocalizationNetwork(
            grid_size=params['grid_size'],
            feature_dim=params['feature_dim'],
            num_heads=params['num_heads'],
            dropout_rate=params['dropout_rate']
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=params['learning_rate'])
        
        # Create agent-like object
        class OptimizedAgent:
            def __init__(self, network, optimizer, params):
                self.network = network
                self.optimizer = optimizer
                self.grid_size = params['grid_size']
                self.device = network.location_head.weight.device
                
                # PPO hyperparameters
                self.gamma = params['gamma']
                self.eps_clip = params['eps_clip']
                self.k_epochs = params['k_epochs']
                self.entropy_coef = params['entropy_coef']
                self.value_coef = params['value_coef']
                
                self.memory = []
            
            def select_top3_actions(self, tif_image, crop_image):
                # Preprocess images
                tif_tensor = self._preprocess_image(tif_image).to(self.device)
                crop_tensor = self._preprocess_image(crop_image).to(self.device)
                
                with torch.no_grad():
                    location_probs, value, logits = self.network(tif_tensor, crop_tensor)
                    
                    # Get top 3 predictions
                    top_probs, top_indices = torch.topk(location_probs.squeeze(), k=3)
                    
                    # Convert to grid coordinates
                    locations = []
                    probabilities = []
                    
                    for i in range(3):
                        idx = top_indices[i].item()
                        prob = top_probs[i].item()
                        
                        grid_y = idx // self.grid_size
                        grid_x = idx % self.grid_size
                        
                        locations.append((grid_x, grid_y))
                        probabilities.append(prob)
                
                return locations, probabilities, logits.squeeze()
            
            def _preprocess_image(self, image):
                import cv2
                resized = cv2.resize(image, (224, 224))
                tensor = torch.from_numpy(resized).float() / 255.0
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
                return tensor
            
            def store_experience(self, state, action_logits, reward, value):
                self.memory.append({
                    'state': state,
                    'action_logits': action_logits.detach().cpu(),
                    'reward': reward,
                    'value': value.detach().cpu()
                })
            
            def update_policy(self):
                if len(self.memory) < 16:  # Minimum batch size
                    return
                
                # Simplified PPO update for optimization
                returns = []
                advantages = []
                gae = 0
                
                for i in reversed(range(len(self.memory))):
                    if i == len(self.memory) - 1:
                        next_value = 0
                    else:
                        next_value = self.memory[i + 1]['value']
                    
                    reward = self.memory[i]['reward']
                    value = self.memory[i]['value']
                    
                    returns.insert(0, reward + self.gamma * next_value)
                    
                    delta = reward + self.gamma * next_value - value
                    gae = delta + self.gamma * 0.95 * gae
                    advantages.insert(0, gae)
                
                # Convert to tensors
                returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
                advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # PPO update
                for _ in range(self.k_epochs):
                    total_loss = 0
                    
                    for i, experience in enumerate(self.memory):
                        state = experience['state']
                        old_logits = experience['action_logits'].to(self.device)
                        
                        # Forward pass
                        tif_tensor = self._preprocess_image(state['tif_image']).to(self.device)
                        crop_tensor = self._preprocess_image(state['crop_image']).to(self.device)
                        
                        location_probs, value, new_logits = self.network(tif_tensor, crop_tensor)
                        
                        # Policy loss
                        old_probs = torch.nn.functional.softmax(old_logits, dim=0)
                        new_probs = torch.nn.functional.softmax(new_logits.squeeze(), dim=0)
                        
                        ratio = (new_probs + 1e-8) / (old_probs + 1e-8)
                        
                        surr1 = ratio * advantages[i]
                        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[i]
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value loss
                        value_loss = torch.nn.functional.mse_loss(value.squeeze(), returns[i])
                        
                        # Entropy loss
                        entropy = -torch.sum(new_probs * torch.log(new_probs + 1e-8))
                        
                        total_loss += (policy_loss + 
                                     self.value_coef * value_loss - 
                                     self.entropy_coef * entropy)
                    
                    # Update
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                    self.optimizer.step()
                
                # Clear memory
                self.memory = []
        
        return OptimizedAgent(network, optimizer, params)
    
    def train_and_evaluate(self, num_episodes: int = 300, update_frequency: int = 16) -> float:
        """Train and return objective score for Optuna"""
        
        print(f"   Training with params: {self.params}")
        
        episode_rewards = []
        similarity_scores = []
        
        for episode in range(num_episodes):
            # Reset environment
            tif_image, crop_image, state = self.env.reset()
            
            # Agent prediction
            locations, probabilities, action_logits = self.agent.select_top3_actions(tif_image, crop_image)
            
            # Calculate reward
            reward, similarities = self.env.calculate_reward(locations, probabilities)
            
            # Get value estimate
            with torch.no_grad():
                tif_tensor = self.agent._preprocess_image(tif_image).to(self.agent.device)
                crop_tensor = self.agent._preprocess_image(crop_image).to(self.agent.device)
                _, value, _ = self.agent.network(tif_tensor, crop_tensor)
            
            # Store experience
            self.agent.store_experience(state, action_logits, reward, value)
            
            # Update policy
            if (episode + 1) % update_frequency == 0:
                self.agent.update_policy()
            
            # Track metrics
            episode_rewards.append(reward)
            similarity_scores.append(max(similarities))
            
            # Early stopping if not improving
            if episode > 100 and episode % 50 == 0:
                recent_avg = np.mean(episode_rewards[-50:])
                older_avg = np.mean(episode_rewards[-100:-50])
                
                if recent_avg <= older_avg * 1.01:  # Less than 1% improvement
                    print(f"   Early stopping at episode {episode} (no improvement)")
                    break
        
        # Calculate final objective
        final_episodes = min(100, len(episode_rewards))
        final_reward = np.mean(episode_rewards[-final_episodes:])
        final_similarity = np.mean(similarity_scores[-final_episodes:])
        max_confidence = max([max(self.agent.select_top3_actions(
            *self.env.reset()[:2])[1]) for _ in range(10)])
        
        # Combined objective (reward + similarity + confidence)
        objective = 0.4 * final_reward + 0.4 * final_similarity + 0.2 * max_confidence
        
        print(f"   Final objective: {objective:.4f} (reward: {final_reward:.3f}, "
              f"similarity: {final_similarity:.3f}, confidence: {max_confidence:.3f})")
        
        return objective

def objective(trial):
    """Optuna objective function"""
    
    # Sample hyperparameters
    params = {
        # Grid and architecture
        'grid_size': trial.suggest_int('grid_size', 20, 35),
        'feature_dim': trial.suggest_categorical('feature_dim', [256, 384, 512, 768]),
        'num_heads': trial.suggest_categorical('num_heads', [4, 6, 8, 12]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        
        # Learning parameters
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'gamma': trial.suggest_float('gamma', 0.95, 0.999),
        
        # PPO parameters
        'eps_clip': trial.suggest_float('eps_clip', 0.1, 0.3),
        'k_epochs': trial.suggest_int('k_epochs', 2, 8),
        'entropy_coef': trial.suggest_float('entropy_coef', 0.001, 0.1, log=True),
        'value_coef': trial.suggest_float('value_coef', 0.1, 1.0),
    }
    
    try:
        # Create trainer with sampled parameters
        trainer = OptimizedDroneLocalizationTrainer(params)
        
        # Train and evaluate
        objective_score = trainer.train_and_evaluate(num_episodes=300)
        
        return objective_score
        
    except Exception as e:
        print(f"   Trial failed with error: {e}")
        return 0.0  # Return poor score for failed trials

def run_optimization(n_trials: int = 50, timeout_hours: int = 8):
    """Run Optuna optimization"""
    
    # Setup logging
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler())
    optuna.logging.set_verbosity(logging.INFO)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50)
    )
    
    print(f"🔍 Starting Optuna Optimization")
    print(f"   Trials: {n_trials}")
    print(f"   Timeout: {timeout_hours} hours")
    print(f"   Start time: {datetime.now()}")
    
    # Run optimization
    start_time = time.time()
    study.optimize(
        objective, 
        n_trials=n_trials,
        timeout=timeout_hours * 3600,
        show_progress_bar=True
    )
    
    end_time = time.time()
    
    # Results
    print(f"\n🎉 Optimization Complete!")
    print(f"   Duration: {(end_time - start_time) / 3600:.2f} hours")
    print(f"   Trials completed: {len(study.trials)}")
    print(f"   Best value: {study.best_value:.4f}")
    
    print(f"\n🏆 Best Parameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    # Save results
    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials),
        'duration_hours': (end_time - start_time) / 3600,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = f"optuna_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   💾 Results saved to: {results_file}")
    
    # Plot optimization history (if possible)
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Optimization history
        optuna.visualization.matplotlib.plot_optimization_history(study, ax=ax1)
        ax1.set_title('Optimization History')
        
        # Parameter importances
        optuna.visualization.matplotlib.plot_param_importances(study, ax=ax2)
        ax2.set_title('Parameter Importances')
        
        plt.tight_layout()
        plt.savefig(f"optuna_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=300)
        plt.show()
        
    except ImportError:
        print("   📊 Install matplotlib and plotly for visualization")
    
    return study

def create_optimized_trainer(best_params: Dict[str, Any]):
    """Create trainer with best parameters found by Optuna"""
    
    print(f"🚀 Creating optimized trainer with best parameters:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    return OptimizedDroneLocalizationTrainer(best_params)

def quick_optimization_overnight():
    """Quick function to run overnight optimization"""
    
    print("🌙 Starting Overnight Optimization...")
    print("   This will run for 8 hours or 50 trials (whichever comes first)")
    print("   Go to sleep! Results will be ready in the morning 😴")
    
    study = run_optimization(n_trials=50, timeout_hours=8)
    
    print(f"\n☀️ Good morning! Optimization complete.")
    print(f"   Use these parameters for your next training:")
    print(f"   {study.best_params}")
    
    return study

if __name__ == "__main__":
    print("🔍 OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*50)
    print()
    print("🎯 WHAT IT OPTIMIZES:")
    print("   • Grid size (20-35)")
    print("   • Network architecture (feature_dim, num_heads, dropout)")
    print("   • Learning rate (1e-5 to 1e-2)")
    print("   • PPO hyperparameters (eps_clip, k_epochs, entropy)")
    print("   • Reward function weights")
    print()
    print("⏰ OVERNIGHT MODE:")
    print("   # Perfect for while you sleep!")
    print("   study = quick_optimization_overnight()")
    print()
    print("🔬 MANUAL MODE:")
    print("   # Custom trials and timeout")
    print("   study = run_optimization(n_trials=100, timeout_hours=12)")
    print()
    print("🚀 DEPLOY BEST:")
    print("   # Use best parameters found")
    print("   trainer = create_optimized_trainer(study.best_params)")
    print("   trainer.train_and_evaluate(num_episodes=1000)")
    print()
    print("😴 Sweet dreams! Wake up to optimized hyperparameters!")