from stable_baselines3.common.buffers import ReplayBuffer
import torch
import numpy as np


def prefill_replay_buffer(model_dqn, env, ig_model, steps=1000):
    obs = env.reset()
    for _ in range(steps):
        # Converti obs in formato accettabile da ig_model (es. torch tensor)
        action = select_action_with_ig_model(env, ig_model)

        new_obs, reward, done, info = env.step(action)
        # Salva transizione nel replay buffer del modello DQN
        model_dqn.replay_buffer.add(obs, new_obs, action, reward, done, infos=None)
        obs = new_obs if not done else env.reset()


def select_action_with_ig_model(env, ig_model, strategy='entropy', device='cpu'):
    """
    obs: numpy array of shape [3, 3, 9, 17]
    """
    utilities = np.zeros((3, 3))

    ax, ay = env.agent_pos
    for i in range(-1, 2):  # Da -1 a 1 (inclusi)
        for j in range(-1, 2):  # Da -1 a 1 (inclusi)
            nx, ny = ax + i, ay + j
            if 0 <= nx < env.grid_size and 0 <= ny < env.grid_size:
                cell_obs = env.state[nx, ny]['obs']
                cell_tensor = torch.tensor(cell_obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = ig_model(cell_tensor)  # each of shape [1, 9]

                entropy = output['pred_entropy']
                loss = output['pred_loss']

                if strategy == 'entropy':
                    utilities[i + 1, j + 1] = torch.min(entropy).item()
                elif strategy == 'loss':
                    utilities[i + 1, j + 1] = torch.min(loss).item()
                elif strategy == 'random':
                    utilities[i + 1, j + 1] = np.random.rand()

    # Massima utilità (esplorazione desiderabile)
    best_i, best_j = np.unravel_index(np.argmin(utilities), (3, 3))

    # Posizione corrente è centro (1, 1)
    delta_i = best_i - 1
    delta_j = best_j - 1

    if abs(delta_i) > abs(delta_j):
        action = 0 if delta_i == -1 else 1  # up or down
    elif abs(delta_j) > 0:
        action = 2 if delta_j == -1 else 3  # left or right
    else:
        action = np.random.choice([0, 1, 2, 3])  # se siamo già sulla cella target

    return action
