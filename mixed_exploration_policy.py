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
        grid_min, grid_max = 1, self.env.n
        best_score = -np.inf
        best_action = None
        for action in [0, 1, 2, 3]:
            grid_values = []
            new_x, new_y = self.get_new_position(action, agent_x, agent_y)

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    x, y = new_x + dx, new_y + dy
                    if grid_min <= x <= grid_max and grid_min <= y <= grid_max:
                        cell = self.env.state[x, y]
                        current_entropy = self.env.state[x, y]["current_entropy"]
                        input_array = self.env.update_cell(cell, dx, dy, update=False)
                        if isinstance(input_array, int) and input_array == 0:
                            # cella già visitata del quel punto di vista
                            obs_seq = self.env.state[x, y]["obs"]
                            input_array = torch.tensor(obs_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                            non_zero_mask = (input_array != 0).any(dim=2)
                            input_array = input_array[non_zero_mask]

                        with torch.no_grad():
                            output = self.env.ig_model(torch.tensor(input_array))  # [1, 9]
                        entropy = output['pred_entropy']

                        gain = current_entropy.item() - entropy.min().item()
                        if gain < 0:
                            gain = 0

                        if self.strategy == 'entropy':
                            grid_values.append(gain)

            if grid_values:
                score = np.sum(grid_values)
                if score > best_score:
                    best_score = score
                    best_action = action

        if best_action is None:
            best_action = np.random.choice([0, 1, 2, 3])

        return torch.tensor([best_action])

    def get_new_position(self, action, x, y):
        if action == 0:
            return x - 1, y  # su
        elif action == 1:
            return x, y + 1  # destra
        elif action == 2:
            return x + 1, y  # giù
        elif action == 3:
            return x, y - 1  # sinistra

