import sys
from collections import defaultdict
from datetime import datetime
import torch
from stable_baselines3 import DQN
from callback import RewardLoggerCallback
from custumCNN import CustomCNN
# from custum_map import GridMappingEnv
from new_custum_map_simple_GP import GridMappingEnv
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


def create_env(size, step, base_model, ig_model, strategy, render=False):
    env = GridMappingEnv(n=size, max_steps=step,
                         ig_model=ig_model,
                         base_model=base_model,
                         strategy=strategy,
                         render_mode='human' if render else None,
                         device=device)
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

def train_multiple_seeds(seeds, episodes, render, strategy, device, buffer_size=1_000_000):
    results = {}
    for seed in seeds:
        print(f"\n=== Training with seed {seed} ===")
        
        # directory specifica per seed
        dir_path = os.path.join(strategy, f"seed_{seed}")
        
        # set seed globale
        set_seed(seed)
        
        # lancia train
        train_data = train(
            episodes=episodes,
            render=render,
            strategy=dir_path,  # passo una directory unica per seed
            device=device,
            buffer_size=buffer_size
        )
        
        results[seed] = train_data
    
    return results


def train(episodes, render, strategy, device, buffer_size=1_000_000):
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

    reward_logger = RewardLoggerCallback()
    # Callback per salvare il modello ogni 10.000 step (puoi cambiare il valore)
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=checkpoint_dir,
        name_prefix="dqn_exploration_ig_reward_env_20x20_doubleCNN_expov8_ig_policy_checkpoint"
    )

    # Unisci callback personalizzato e salvataggio
    callback = CallbackList([reward_logger, checkpoint_callback])
    base_model, ig_model = load_models(device)
    env = create_env(size=50, step=3000, base_model=base_model, ig_model=ig_model, render=render, strategy=strategy)

    # MlpPolicy: rete completamente connessa (https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)
    # input: obs --> [3, 3, 9, 17] ---> flatten = 1377
    # dim hidden layer --> 64 units
    # output: action --> 4
    # model_dqn = DQN("MlpPolicy", env, verbose=1, buffer_size=800000)

    policy_kwargs = dict(
         features_extractor_class=DoubleCNNExtractor,
         features_extractor_kwargs=dict(extra_pov_radius=8),
    )

    # policy_kwargs = dict(
    #    features_extractor_class=CustomCNN,
    #    features_extractor_kwargs=dict(features_dim=256),
    # )

    # model_dqn = DQN(
    #    "MlpPolicy",
    #    env,
    #    policy_kwargs=policy_kwargs,
    #    verbose=1,
    #    buffer_size=buffer_size,
    #    device=device,
    # )

    model_dqn = DQN(
        policy=MixedExplorationPolicy,
        env=env,
        policy_kwargs={**policy_kwargs, 'env': env, 'p_ig_start': 1.0, 'p_ig_end': 0.0,
                       'p_ig_decay_steps': 10000, 'strategy': 'entropy'},
        buffer_size=buffer_size,
        device=device,
        verbose=1,
    )
    start_time = time.time()

    # prefill_replay_buffer(model_dqn, env, ig_model, steps=1000)

    model_dqn.learn(total_timesteps=episodes, callback=callback)
    end_time = time.time()

    training_time = end_time - start_time

    train_data["episode_rewards"] = [float(r) for r in reward_logger.episode_rewards]
    train_data["episode_cells_marker_pred_1"] = reward_logger.episode_cells_marker_pred_1
    train_data["episode_cells_seen_pov"] = reward_logger.episode_cells_seen_pov
    train_data["episode_steps"] = reward_logger.episode_steps
    train_data["training_time_seconds"] = training_time

    save_dict(train_data, f"./data/{dir_path}/train_data_ig_reward_env_50x50_doubleCNN_expov8_ig_policy_{current_datetime}.json")

    model_dqn.save(f"./data/{dir_path}/dqn_exploration_ig_reward_env_50x50_doubleCNN_expov8_ig_policy_{current_datetime}")
    print("Stop train")
    del model_dqn


def test(render, strategy, initial_seed=42, num_runs=10):
    print("Test strategy: " + strategy)
    dir_path = strategy
    if strategy != "random_agent":
        if strategy != 'policy2_ig_reward':
            _, strategy = strategy.split("_")
        else:
            strategy = 'ig_reward'
        model_dqn = DQN.load(f"./data/{dir_path}/dqn_exploration_ig_reward_env_20x20_doubleCNN_expov8_ig_policy")
        model_dqn.policy.p_ig_start = 0

    test_data = defaultdict(list)
    base_model, ig_model = load_models()
    env = create_env(size=50, step=3000, base_model=base_model, ig_model=ig_model, render=render, strategy=strategy)


    # Lista per tenere traccia delle metriche per ogni run
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
        # Per memorizzare la posizione dell'agente a ogni step
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

        # Aggiorna il conteggio delle celle viste da 9 punti di vista e delle celle con marker predetto corretto
        cells_marker_pred_1 = sum(
            1 for row in env.state[1:env.n + 1, 1:env.n + 1]
            for cell in row if cell['marker_pred'] == 1
        )

        cells_seen_pov = sum(
            1 for row in env.state[1:env.n + 1, 1:env.n + 1]
            for cell in row if sum(cell['pov']) == 9
        )

        # Salva le metriche per il run corrente
        cumulative_rewards_per_run.append(cumulative_reward)
        cells_marker_pred_1_per_run.append(cells_marker_pred_1)
        cells_seen_pov_per_run.append(cells_seen_pov)
        total_steps_per_run.append(steps)

    max_length = max(len(sotto_lista) for sotto_lista in cells_marker_pred_1_each_step)

    # Shape: (run, step)
    # Creazione di un array 2D con padding di NaN per le liste più corte
    data_padded = np.full((len(cells_marker_pred_1_each_step), max_length), np.nan)

    # Riempire con i valori reali disponibili
    for i, lst in enumerate(cells_marker_pred_1_each_step):
        data_padded[i, :len(lst)] = lst  # Copia i valori effettivi

    # Calcola la deviazione standard e media ignorando i NaN
    cells_marker_pred_1_std = np.nanstd(data_padded, axis=0, ddof=1)
    cells_marker_pred_1_mean = np.nanmean(data_padded, axis=0)

    # Stampa le metriche
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

    save_dict(test_data, f"./data/{dir_path}/test_data_ig_reward_env_20x20_doubleCNN_expov8_ig_policy.json")


current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
strategy = sys.argv[1]

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"

# buffer_size = 100_000 if use_cuda else 50_000
episodes = 50_000
seeds = [0, 42, 123, 999, 2024, 7, 88, 256, 512, 1024]

train(episodes=episodes, render=False, strategy=strategy, device=device)
test(render=False, strategy=strategy, initial_seed=42, num_runs=20)
