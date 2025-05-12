from stable_baselines3.dqn.policies import DQNPolicy
import torch
import numpy as np


class MixedExplorationPolicy(DQNPolicy):
    def __init__(self, *args, env, p_ig_start=1.0, p_ig_end=0.0, p_ig_decay_steps=10000,
                 strategy='entropy', **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        self.p_ig_start = p_ig_start
        self.p_ig_end = p_ig_end
        self.p_ig_decay_steps = p_ig_decay_steps
        self.step_count = 0
        self.strategy = strategy

    def _predict(self, observation, deterministic=False):
        # Calcola la probabilità attuale di usare ig_model
        p_ig = max(self.p_ig_end,
                   self.p_ig_start - (self.step_count / self.p_ig_decay_steps) * (self.p_ig_start - self.p_ig_end))
        self.step_count += 1

        # Decide stocasticamente se usare ig_model
        if np.random.rand() < p_ig:
            action = self.select_action_with_ig_model()
        else:
            action = super()._predict(observation, deterministic)
        return action

    def select_action_with_ig_model(self):
        agent_x, agent_y = self.env.agent_pos
        grid_values = np.full((3, 3), -np.inf)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x, y = agent_x + dx, agent_y + dy
                if 0 <= x < self.env.grid_size and 0 <= y < self.env.grid_size:
                    obs_seq = self.env.state[x, y]["obs"]
                    input_tensor = torch.tensor(obs_seq, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, 9, 17]
                    with torch.no_grad():
                        output = self.env.ig_model(input_tensor)  # [1, 9]
                    entropy = output['pred_entropy']
                    loss = output['pred_loss']

                    if self.strategy == 'entropy':
                        grid_values[dy + 1, dx + 1] = entropy.mean().item()
                    else:
                        grid_values[dy + 1, dx + 1] = loss.mean().item()

        best_dy, best_dx = np.unravel_index(np.argmin(grid_values), (3, 3))
        dx = best_dx - 1
        dy = best_dy - 1
        action = self.direction_to_action(dx, dy)

        return torch.tensor([action])

    def direction_to_action(self, dx, dy):
        if dx == 0 and dy == -1:
            return 0  # su
        elif dx == 0 and dy == 1:
            return 1  # giù
        elif dx == -1 and dy == 0:
            return 2  # sinistra
        elif dx == 1 and dy == 0:
            return 3  # destra
        elif abs(dx) + abs(dy) == 2:  # movimento diagonale
            # scegli una direzione valida che si avvicina
            if np.random.rand() < 0.5:
                return self.direction_to_action(dx, 0)  # passo orizzontale
            else:
                return self.direction_to_action(0, dy)  # passo verticale
        else:
            # fallback se dx, dy fuori range [-1, 1]
            return np.random.choice([0, 1, 2, 3])
