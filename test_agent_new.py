import sys
from collections import defaultdict
from datetime import datetime
import torch
import torch.optim as optim
from stable_baselines3 import DQN
from callback import RewardLoggerCallback
from custumCNN import CustomCNN
from new_custum_map_GP import GridMappingEnv
from new_custum_map_GP import train_observer_on_env  
from doubleCNN import DoubleCNNExtractor
import sys
from collections import defaultdict
from datetime import datetime
import torch
import torch.optim as optim
from stable_baselines3 import DQN
from callback import RewardLoggerCallback
from custumCNN import CustomCNN
from new_custum_map_GP import GridMappingEnv
from new_custum_map_GP import train_observer_on_env  
from doubleCNN import DoubleCNNExtractor
from helper import save_dict, load_models
import tqdm as tqdm
import numpy as np
import time
import random

from mixed_exploration_policy import MixedExplorationPolicy
from replay_buffer import prefill_replay_buffer
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import os


def create_env(size, step, base_model, ig_model, strategy, render=False, 
               bootstrap_mode=True, device='cpu'):
    """
    Create environment with support for bootstrap mode.
    
    Args:
        bootstrap_mode: if True, use bootstrap from observations (Phase 1)
                       if False, use LSTM ↔ GP loop (Phase 2)
    """
    env = GridMappingEnv(
        n=size, 
        max_steps=step,
        ig_model=ig_model,
        base_model=base_model,
        strategy=strategy,
        render_mode='human' if render else None,
        device=device,
        bootstrap_gp_from_obs=bootstrap_mode,  # ← Training phase control
        prob_temperature=1.5,
        bootstrap_var=1.0
    )
    return env


def set_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)


# ============================================================================
# NEW FUNCTION: Pre-training Observer LSTM
# ============================================================================
def pretrain_observer(env, device, num_episodes=20, num_epochs=30, lr=1e-3):
    """
    Pre-training of the Observer LSTM before RL training.
    
    PHASE 1: Bootstrap from observations
    - Generate random trajectories to collect data
    - Train LSTM using statistics from direct observations
    
    Args:
        env: GridMappingEnv with bootstrap_gp_from_obs=True
        device: 'cuda' or 'cpu'
        num_episodes: episodes for data collection
        num_epochs: training epochs
        lr: learning rate
    
    Returns:
        dict with pre-training metrics
    """
    print("\n" + "=" * 70)
    print("PRE-TRAINING OBSERVER LSTM (Fase 1: Bootstrap)")
    print("=" * 70)
    
    observer = env.observer
    optimizer = optim.Adam(observer.parameters(), lr=lr, weight_decay=1e-5)
    
    pretrain_metrics = {
        'losses': [],
        'num_episodes': num_episodes,
        'num_epochs': num_epochs
    }
    
    # 1. DATA COLLECTION with random trajectories
    print(f"\n[1/3] Data Collection: {num_episodes} episodes with random actions...")
    for episode in tqdm.tqdm(range(num_episodes), desc="Data Collection"):
        obs = env.reset()
        done = False
        step_count = 0
        max_steps_per_episode = 100
        
        while not done and step_count < max_steps_per_episode:
            action = env.action_space.sample()  # random action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
    
    # Count cells with data
    cells_with_data = 0
    total_observations = 0
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            obs_count = np.count_nonzero(np.any(env.state[r, c]['obs'] != 0, axis=1))
            if obs_count > 0:
                cells_with_data += 1
                total_observations += obs_count
    
    print(f"  ✓ Cells with data: {cells_with_data}/{env.grid_size**2}")
    print(f"  ✓ Total observations: {total_observations}")
    
    # 2. SUPERVISED TRAINING
    print(f"\n[2/3] Training Observer: {num_epochs} epochs...")
    observer.train()
    
    for epoch in tqdm.tqdm(range(num_epochs), desc="Observer Training"):
        loss = train_observer_on_env(env, observer, optimizer, device=device)
        pretrain_metrics['losses'].append(loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs}: Loss = {loss:.6f}")
    
    final_loss = pretrain_metrics['losses'][-1]
    print(f"\n  ✓ Final Loss: {final_loss:.6f}")
    
    # 3. EVALUATION
    print("\n[3/3] Evaluation of prediction accuracy...")
    observer.eval()
    
    correct_predictions = 0
    total_cells_with_obs = 0
    
    for r in range(1, env.n + 1):
        for c in range(1, env.n + 1):
            cell = env.state[r, c]
            if np.any(cell['obs'] != 0):
                total_cells_with_obs += 1
                if cell['marker_pred'] == 1:
                    correct_predictions += 1
    
    accuracy = correct_predictions / max(total_cells_with_obs, 1)
    pretrain_metrics['accuracy'] = accuracy
    pretrain_metrics['correct_predictions'] = correct_predictions
    pretrain_metrics['total_cells_with_obs'] = total_cells_with_obs
    
    print(f"  ✓ Accuracy: {accuracy:.2%} ({correct_predictions}/{total_cells_with_obs})")
    print("=" * 70 + "\n")
    
    return pretrain_metrics


# ============================================================================
# NEW FUNCTION: Fine-tuning Observer during RL
# ============================================================================
def finetune_observer_during_rl(env, device, optimizer, finetune_every=5000):
    """
    Periodic fine-tuning of the Observer during RL training.
    
    PHASE 2: LSTM ↔ GP loop
    - Use data collected by the RL agent
    - LSTM uses feedback from the current GP
    
    Args:
        env: GridMappingEnv with bootstrap_gp_from_obs=False
        device: device
        optimizer: Observer's optimizer
        finetune_every: how many RL steps between fine-tuning
    """
    print(f"\n→ Fine-tuning Observer (step {env.current_steps})...")
    
    env.observer.train()
    loss = train_observer_on_env(env, env.observer, optimizer, device=device)
    env.observer.eval()
    
    print(f"  Observer Loss: {loss:.6f}")
    return loss


# ============================================================================
# CUSTOM CALLBACK for periodic fine-tuning
# ============================================================================
class ObserverFinetuneCallback(CallbackList):
    """
    Callback to perform periodic fine-tuning of the Observer during RL training.
    """
    def __init__(self, env, device, finetune_every=5000, num_epochs_per_finetune=5):
        super().__init__([])
        self.env = env
        self.device = device
        self.finetune_every = finetune_every
        self.num_epochs_per_finetune = num_epochs_per_finetune
        self.finetune_losses = []
        self.optimizer = optim.Adam(env.observer.parameters(), lr=5e-4)
    
    def _on_step(self) -> bool:
        # Periodic fine-tuning
        if self.n_calls % self.finetune_every == 0 and self.n_calls > 0:
            print(f"\n{'='*70}")
            print(f"FINE-TUNING OBSERVER @ step {self.n_calls}")
            print(f"{'='*70}")
            
            # Switch to LSTM ↔ GP mode if needed
            was_bootstrap = self.env.bootstrap_gp_from_obs
            if was_bootstrap:
                print("→ Switching to LSTM ↔ GP mode (Phase 2)...")
                self.env.bootstrap_gp_from_obs = False
            
            # Fine-tune for a few epochs
            for epoch in range(self.num_epochs_per_finetune):
                loss = train_observer_on_env(
                    self.env, self.env.observer, self.optimizer, device=self.device
                )
                self.finetune_losses.append(loss)
                
                if epoch == 0 or epoch == self.num_epochs_per_finetune - 1:
                    print(f"  Epoch {epoch+1}/{self.num_epochs_per_finetune}: Loss = {loss:.6f}")
            
            print(f"{'='*70}\n")
        
        return True


# ============================================================================
# MODIFIED TRAIN FUNCTION
# ============================================================================
def train(episodes, render, strategy, device, buffer_size=1_000_000,
          use_observer_pretraining=True, use_observer_finetuning=True,
          pretrain_episodes=20, pretrain_epochs=30):
    """
    Training with integrated Observer LSTM.
    
    Pipeline:
    1. Pre-training Observer (Phase 1: Bootstrap from observations)
    2. RL training of the DQN agent
    3. Periodic fine-tuning of the Observer (Phase 2: LSTM ↔ GP loop)
    
    Args:
        use_observer_pretraining: if True, pre-train Observer before RL
        use_observer_finetuning: if True, perform periodic fine-tuning during RL
        pretrain_episodes: episodes for pre-training data collection
        pretrain_epochs: epochs for Observer pre-training
    """
    dir_path = strategy
    print("Start train")
    if strategy == "random_agent":
        return
    
    if strategy != 'policy2_ig_reward':
        _, strategy = strategy.split("_")
    else:
        strategy = 'ig_reward'

    train_data = defaultdict(list)

    checkpoint_dir = f"./data/{dir_path}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup callbacks
    reward_logger = RewardLoggerCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=checkpoint_dir,
        name_prefix="dqn_exploration_ig_reward_env_20x20_doubleCNN_expov8_ig_policy_checkpoint"
    )

    # Load models
    base_model, ig_model = load_models(device)
    
    # Create environment in initial bootstrap mode (Phase 1)
    env = create_env(
        size=20, 
        step=1000, 
        base_model=base_model, 
        ig_model=ig_model, 
        render=render, 
        strategy=strategy,
        bootstrap_mode=True,  # ← Phase 1
        device=device
    )

    # ========================================================================
    # PRE-TRAINING OBSERVER (optional but recommended)
    # ========================================================================
    if use_observer_pretraining:
        pretrain_metrics = pretrain_observer(
            env, 
            device, 
            num_episodes=pretrain_episodes,
            num_epochs=pretrain_epochs,
            lr=1e-3
        )
        train_data['observer_pretraining'] = pretrain_metrics
        
        # Save pre-trained Observer
        torch.save(
            env.observer.state_dict(), 
            f"{checkpoint_dir}/observer_pretrained.pth"
        )
        print(f"✓ Observer pre-trained saved in {checkpoint_dir}/observer_pretrained.pth\n")

    # ========================================================================
    # SETUP DQN AGENT
    # ========================================================================
    policy_kwargs = dict(
         features_extractor_class=DoubleCNNExtractor,
         features_extractor_kwargs=dict(extra_pov_radius=8),
    )

    model_dqn = DQN(
        policy=MixedExplorationPolicy,
        env=env,
        policy_kwargs={
            **policy_kwargs, 
            'env': env, 
            'p_ig_start': 1.0, 
            'p_ig_end': 0.0,
            'p_ig_decay_steps': 10000, 
            'strategy': 'entropy'
        },
        buffer_size=buffer_size,
        device=device,
        verbose=1,
    )

    # ========================================================================
    # CALLBACK FOR PERIODIC FINE-TUNING (Phase 2)
    # ========================================================================
    callbacks = [reward_logger, checkpoint_callback]
    
    if use_observer_finetuning:
        observer_finetune_callback = ObserverFinetuneCallback(
            env=env,
            device=device,
            finetune_every=5000,  # every 5k steps
            num_epochs_per_finetune=5  # 5 fine-tuning epochs
        )
        callbacks.append(observer_finetune_callback)
        print("✓ Observer fine-tuning enabled (every 5000 steps)\n")
    
    callback = CallbackList(callbacks)

    # ========================================================================
    # RL TRAINING
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING RL AGENT (DQN)")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    model_dqn.learn(total_timesteps=episodes, callback=callback)
    end_time = time.time()
    training_time = end_time - start_time

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    train_data["episode_rewards"] = [float(r) for r in reward_logger.episode_rewards]
    train_data["episode_cells_marker_pred_1"] = reward_logger.episode_cells_marker_pred_1
    train_data["episode_cells_seen_pov"] = reward_logger.episode_cells_seen_pov
    train_data["episode_steps"] = reward_logger.episode_steps
    train_data["training_time_seconds"] = training_time
    
    # Save fine-tuning losses if available
    if use_observer_finetuning:
        train_data["observer_finetune_losses"] = observer_finetune_callback.finetune_losses

    save_dict(
        train_data, 
        f"./data/{dir_path}/train_data_ig_reward_env_50x50_doubleCNN_expov8_ig_policy_{current_datetime}.json"
    )

    # Save DQN model
    model_dqn.save(
        f"./data/{dir_path}/dqn_exploration_ig_reward_env_50x50_doubleCNN_expov8_ig_policy_{current_datetime}"
    )
    
    # Save final Observer
    torch.save(
        env.observer.state_dict(), 
        f"{checkpoint_dir}/observer_final.pth"
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"✓ Total time: {training_time:.2f}s")
    print(f"✓ DQN saved in: {checkpoint_dir}/")
    print(f"✓ Observer saved in: {checkpoint_dir}/observer_final.pth")
    print("=" * 70 + "\n")
    
    del model_dqn


# ============================================================================
# TEST FUNCTION (unchanged, but can use trained Observer)
# ============================================================================
def test(render, strategy, initial_seed=42, num_runs=10, load_observer=True):
    """
    Test with option to load pre-trained Observer.
    
    Args:
        load_observer: if True, load observer_final.pth if available
    """
    print("Test strategy: " + strategy)
    dir_path = strategy
    
    if strategy != "random_agent":
        if strategy != 'policy2_ig_reward':
            _, strategy = strategy.split("_")
        else:
            strategy = 'ig_reward'
        model_dqn = DQN.load(
            f"./data/{dir_path}/dqn_exploration_ig_reward_env_20x20_doubleCNN_expov8_ig_policy"
        )
        model_dqn.policy.p_ig_start = 0

    test_data = defaultdict(list)
    base_model, ig_model = load_models()
    env = create_env(
        size=20, 
        step=1000, 
        base_model=base_model, 
        ig_model=ig_model, 
        render=render, 
        strategy=strategy,
        bootstrap_mode=False,  # use LSTM ↔ GP mode for testing
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Load trained Observer if available
    if load_observer:
        observer_path = f"./data/{dir_path}/observer_final.pth"
        if os.path.exists(observer_path):
            env.observer.load_state_dict(torch.load(observer_path))
            print(f"✓ Observer loaded from: {observer_path}\n")
        else:
            print(f"⚠ Observer not found at: {observer_path}")
            print("  Using untrained Observer\n")

    # Lists to keep track of metrics for each run
    cumulative_rewards_per_run = []
    cells_marker_pred_1_per_run = []
    cells_seen_pov_per_run = []
    total_steps_per_run = []
    cells_marker_pred_1_each_step = []
    total_position_per_run = []

    for run in tqdm.tqdm(range(num_runs)):
        seed = initial_seed + run
        obs, info = env.reset(seed=seed)
        cumulative_reward = 0.0
        steps = 0
        cells_marker_pred_1_run = []
        positions = [env.agent_pos.copy()]

        while True:
            if strategy != "random_agent":
                action, _states = model_dqn.predict(obs)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            cumulative_reward += reward
            steps += 1
            pos = env.agent_pos.copy()
            positions.append(pos)

            cells_marker_pred_1_run.append(sum(
                1 for row in env.state[1:env.n + 1, 1:env.n + 1]
                for cell in row if cell['marker_pred'] == 1
            ))

            if terminated or truncated:
                break

        cells_marker_pred_1_each_step.append(cells_marker_pred_1_run)
        total_position_per_run.append(positions)

        cells_marker_pred_1 = sum(
            1 for row in env.state[1:env.n + 1, 1:env.n + 1]
            for cell in row if cell['marker_pred'] == 1
        )

        cells_seen_pov = sum(
            1 for row in env.state[1:env.n + 1, 1:env.n + 1]
            for cell in row if sum(cell['pov']) == 9
        )

        cumulative_rewards_per_run.append(cumulative_reward)
        cells_marker_pred_1_per_run.append(cells_marker_pred_1)
        cells_seen_pov_per_run.append(cells_seen_pov)
        total_steps_per_run.append(steps)

    max_length = max(len(sotto_lista) for sotto_lista in cells_marker_pred_1_each_step)
    data_padded = np.full((len(cells_marker_pred_1_each_step), max_length), np.nan)

    for i, lst in enumerate(cells_marker_pred_1_each_step):
        data_padded[i, :len(lst)] = lst

    cells_marker_pred_1_std = np.nanstd(data_padded, axis=0, ddof=1)
    cells_marker_pred_1_mean = np.nanmean(data_padded, axis=0)

    print("Cumulative Rewards per Run:", cumulative_rewards_per_run)
    print("Cells with Correct Marker Prediction per Run:", cells_marker_pred_1_per_run)
    print("Cells Seen from 9 POVs per Run:", cells_seen_pov_per_run)

    test_data["cells_marker_pred_1_mean"] = cells_marker_pred_1_mean.tolist()
    test_data["cells_marker_pred_1_std"] = cells_marker_pred_1_std.tolist()
    test_data["cumulative_rewards_per_run"] = cumulative_rewards_per_run
    test_data["cells_marker_pred_1_per_run"] = cells_marker_pred_1_per_run
    test_data["cells_seen_pov_per_run"] = cells_seen_pov_per_run
    test_data["total_steps_per_run"] = total_steps_per_run
    test_data["total_position_per_run"] = total_position_per_run

    save_dict(
        test_data, 
        f"./data/{dir_path}/test_data_ig_reward_env_20x20_doubleCNN_expov8_ig_policy.json"
    )


# ============================================================================
# TRAIN WITH MULTIPLE SEEDS
# ============================================================================
def train_multiple_seeds(seeds, episodes, render, strategy, device, 
                         buffer_size=1_000_000,
                         use_observer_pretraining=True,
                         use_observer_finetuning=True):
    """
    Training with multiple seeds and Observer integration.
    """
    results = {}
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"TRAINING WITH SEED {seed}")
        print(f"{'='*70}\n")
        
        # seed-specific directory
        dir_path = os.path.join(strategy, f"seed_{seed}")
        
        # set global seed
        set_seed(seed)
        
        # run train
        train_data = train(
            episodes=episodes,
            render=render,
            strategy=dir_path,
            device=device,
            buffer_size=buffer_size,
            use_observer_pretraining=use_observer_pretraining,
            use_observer_finetuning=use_observer_finetuning
        )
        
        results[seed] = train_data
    
    return results


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    strategy = sys.argv[1]

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    episodes = 50_000
    seeds = [0, 42, 123, 999, 2024, 7, 88, 256, 512, 1024]

    # Training with Observer integration
    train(
        episodes=episodes, 
        render=False, 
        strategy=strategy, 
        device=device,
        use_observer_pretraining=True,   # ← Enable pre-training (Phase 1)
        use_observer_finetuning=True,    # ← Enable fine-tuning (Phase 2)
        pretrain_episodes=20,            # episodes for data collection
        pretrain_epochs=30               # epochs for pre-training
    )
    
    # Test with trained Observer
    test(
        render=False, 
        strategy=strategy, 
        initial_seed=42, 
        num_runs=20,
        load_observer=True  # ← Load trained Observer
    )