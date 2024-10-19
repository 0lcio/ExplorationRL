from collections import defaultdict
from helper import save_dict, load_models
from custum_map import GridMappingEnv
from datetime import datetime


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

    for run in range(num_runs):
        seed = initial_seed + run
        obs, info = env.reset(seed=seed)
        cumulative_reward = 0.0
        steps = 0
        cells_marker_pred_1_run = []
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
            cells_marker_pred_1_run.append(sum(
                1 for row in env.state[1:env.n + 1, 1:env.n + 1]
                for cell in row if cell['marker_pred'] == 1
            ))

            if terminated or truncated:
                break

        cells_marker_pred_1_each_step.append(cells_marker_pred_1_run)

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

    somme = [0] * max_length
    conteggi = [0] * max_length

    # Sommare gli elementi corrispondenti di ogni lista
    for sotto_lista in cells_marker_pred_1_each_step:
        for i, valore in enumerate(sotto_lista):
            somme[i] += valore
            conteggi[i] += 1

    # Calcolare la media per ogni posizione
    cells_marker_pred_1_mean = [somme[i] / conteggi[i] for i in range(max_length)]
    # Stampa le metriche
    print("Cumulative Rewards per Run:", cumulative_rewards_per_run)
    print("Cells with Correct Marker Prediction per Run:", cells_marker_pred_1_per_run)
    print("Cells Seen from 9 POVs per Run:", cells_seen_pov_per_run)

    test_data["cells_marker_pred_1_mean"] = cells_marker_pred_1_mean
    test_data["cumulative_rewards_per_run"] = cumulative_rewards_per_run
    test_data["cells_marker_pred_1_per_run"] = cells_marker_pred_1_per_run
    test_data["cells_seen_pov_per_run"] = cells_seen_pov_per_run
    test_data["total_steps_per_run"] = total_steps_per_run

    save_dict(test_data, f"./data/{strategy}/test_data_{strategy}_{current_datetime}.json")


strategy = "no_train"
test(render=False, strategy=strategy, initial_seed=42, num_runs=20)
