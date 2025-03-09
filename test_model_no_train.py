from collections import defaultdict
from helper import save_dict, load_models
from custum_map import GridMappingEnv
from datetime import datetime
import numpy as np
import tqdm as tqdm


def create_env(size, step, base_model, ig_model, strategy, render=False):
    env = GridMappingEnv(n=size, max_steps=step,
                         ig_model=ig_model,
                         base_model=base_model,
                         strategy=strategy,
                         render_mode='human' if render else None)
    return env


def test(render, strategy, initial_seed=42, num_runs=10):
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("Test strategy: " + strategy)
    test_data = defaultdict(list)
    base_model, ig_model = load_models()
    env = create_env(size=20, step=1000, base_model=base_model, ig_model=ig_model, render=render, strategy=strategy)

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
            each_action = [0, 1, 2, 3]
            best_score = 0
            for action in each_action:
                action_score = env.step_score(action)
                if action_score > best_score:
                    best_score = action_score
                    best_action = action

            if best_score == 0:
                best_action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(best_action)

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

    max_length = max(len(lst) for lst in cells_marker_pred_1_each_step)

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

    save_dict(test_data, f"./data/{strategy}/test_data_{strategy}_{current_datetime}.json")


strategy = "no_train"
test(render=False, strategy=strategy, initial_seed=42, num_runs=20)
