import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy
import matplotlib.pyplot as plt
import pygame

# ------------------------------------------------------------
# La differenza principale con custum_map è che qui l'asseganzione di ogni
# cella non è caso, ma segue un campo latente generato con un filtro gaussiano.
# ------------------------------------------------------------
def gaussian_random_field(n_cell=(20, 20), cluster_radius=3, binary=False):
    """
    Genera un campo continuo correlato usando un filtro gaussiano.
    """
    noise = np.random.randn(*n_cell)
    field = gaussian_filter(noise, sigma=cluster_radius)
    field = (field - field.min()) / (field.max() - field.min())  # normalizza a [0,1]
    if binary:
        field = (field > 0.5).astype(float)
    return field


def create_binned_field(field, n_bins):
    """
    Divide il campo continuo in n_bins intervalli discreti.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    binned_field = np.digitize(field, bin_edges) - 1
    binned_field = np.clip(binned_field, 0, n_bins - 1)
    return binned_field, bin_edges


class GridMappingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n=5, max_steps=300, render_mode=None,
                 ig_model=None, base_model=None,
                 dataset_path='./data/final_output.csv', strategy=None):
        super(GridMappingEnv, self).__init__()
        self.n = n
        self.grid_size = n + 2  # include bordo
        self.ig_model = ig_model
        self.base_model = base_model
        self.strategy = f"pred_{strategy}"

        self.state = np.array([
            [{'pov': np.zeros(9, dtype=np.int32),
              'best_next_pov': -1,
              'id': None,
              'marker_pred': 0,
              'obs': np.zeros((9, 17), dtype=np.float32),
              'current_entropy': entropy(torch.full((8,), 1 / 8))}
             for _ in range(self.grid_size)]
            for _ in range(self.grid_size)
        ])

        self.agent_pos = [1, 1]
        self.max_steps = max_steps
        self.current_steps = 0
        self.render_mode = render_mode

        self.dataset = pd.read_csv(dataset_path)
        self._init_observation_space(extra_pov_radius=8)

        # Parametri per il campo latente
        self.num_latent_states = 8
        self.cluster_radius = 2.5

        # Rendering
        self.window_size = 600
        self.cell_size = self.window_size // self.grid_size
        self.window = None
        self.clock = None

    # ------------------------------------------------------------
    # Generazione del campo latente
    # ------------------------------------------------------------
    def _generate_latent_field(self):
        field = gaussian_random_field(n_cell=(self.n, self.n),
                                      cluster_radius=self.cluster_radius,
                                      binary=False)
        binned_field, _ = create_binned_field(field, self.num_latent_states)
        binned_field = binned_field + 1  # range [1, num_latent_states]
        return binned_field

    # ------------------------------------------------------------
    # Assegnazione coerente del dataset alle celle
    # ------------------------------------------------------------
    def _assign_ids_to_cells(self):
        self.hidden_state = self._generate_latent_field()

        for i in range(1, self.n + 1):
            for j in range(1, self.n + 1):
                state_value = int(self.hidden_state[i - 1, j - 1])
                candidates = self.dataset[self.dataset['MARKER_COUNT'] == state_value]

                if len(candidates) > 0:
                    row = candidates.sample(n=1).iloc[0]
                else:
                    row = self.dataset.sample(n=1).iloc[0]

                self.state[i, j]['id'] = {
                    'IMAGE_ID': row['IMAGE_ID'],
                    'BOX_COUNT': row['BOX_COUNT'],
                    'MARKER_COUNT': state_value
                }

    # ------------------------------------------------------------
    # Aggiornamento credenze (belief propagation semplice)
    # ------------------------------------------------------------
    def _update_neighbor_beliefs(self, cell_x, cell_y, observed_entropy):
        """
        Aggiorna le entropie delle celle vicine in base alla distanza
        dalla cella osservata.
        """
        sigma = 1.5  # forza della correlazione
        for i in range(1, self.n + 1):
            for j in range(1, self.n + 1):
                dist = np.sqrt((i - cell_x) ** 2 + (j - cell_y) ** 2)
                weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
                old_entropy = self.state[i, j]['current_entropy']
                # Riduci l'entropia delle celle vicine proporzionalmente al peso
                new_entropy = old_entropy * (1 - 0.3 * weight) + observed_entropy * (0.3 * weight)
                self.state[i, j]['current_entropy'] = new_entropy

    # ------------------------------------------------------------
    # Funzioni standard Gym
    # ------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([
            [{'pov': np.zeros(9, dtype=np.int32),
              'best_next_pov': -1,
              'id': None,
              'marker_pred': 0,
              'obs': np.zeros((9, 17), dtype=np.float32),
              'current_entropy': entropy(torch.full((8,), 1 / 8))}
             for _ in range(self.grid_size)]
            for _ in range(self.grid_size)
        ])

        self.agent_pos = [1, 1]
        self._assign_ids_to_cells()
        self.current_steps = 0

        if self.render_mode == 'human':
            self.render()

        return self._get_observation(), {}

    def _move_agent(self, action):
        if action == 0:  # su
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:  # destra
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 2:  # giù
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)
        elif action == 3:  # sinistra
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)

    def step(self, action):
        self.current_steps += 1
        prev_pos = list(self.agent_pos)
        self._move_agent(action)

        ax, ay = self.agent_pos
        cell_info = self.state[ax, ay]['id']
        true_marker = cell_info['MARKER_COUNT']

        # Stima del modello base
        pred = torch.ones(8) / 8 if self.base_model is None else self.base_model(torch.randn(1, 17))
        ent = entropy(pred)

        # Aggiornamento credenze dei vicini
        self._update_neighbor_beliefs(ax, ay, ent)

        reward = -ent  # reward = minore entropia → migliore osservazione
        terminated = self.current_steps >= self.max_steps
        truncated = False

        if self.render_mode == 'human':
            self.render()

        return self._get_observation(), reward, terminated, truncated, {}

    # ------------------------------------------------------------
    # Osservazione e spazio
    # ------------------------------------------------------------
    def _get_observation(self):
        obs = torch.zeros((3, 3, 18))
        ax, ay = self.agent_pos

        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = ax + i, ay + j
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    cell_obs = self.state[nx, ny]['obs']
                    curr_entropy = self.state[nx, ny]['current_entropy'].unsqueeze(0).detach()
                    cell_povs = torch.tensor(self.state[nx, ny]['pov'], dtype=torch.float32).unsqueeze(0).detach()
                    obs[i + 1, j + 1] = torch.cat((curr_entropy, torch.zeros(1, 8), cell_povs), dim=1)

        return obs.detach()

    def _init_observation_space(self, extra_pov_radius=1):
        n_center = 3 * 3 * 18
        n = 2 * extra_pov_radius + 3
        n_pov_cells = n * n
        pov_size = len(self.state[0, 0]['pov'])
        total_obs_len = n_center + n_pov_cells * pov_size
        self.observation_space = spaces.Box(low=0, high=1, shape=(total_obs_len,), dtype=np.float32)
    
    def render(self, mode='human'):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Calcola l'offset per centrare la griglia
        offset = (self.window_size - self.grid_size * self.cell_size) // 2

        # Riempie lo sfondo con bianco
        self.window.fill((255, 255, 255))

        # Disegna la griglia
        font = pygame.font.SysFont('Arial', 20)  # Font per il conteggio delle visite

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.state[i, j]
                if 1 <= i <= self.n and 1 <= j <= self.n:  # Celle all'interno della griglia originale
                    visits = np.sum(cell['pov'])
                    green_value = min(255, visits * 25)  # Sfumatura di verde: più visite, più scuro
                    cell_color = (200 - green_value // 2, 255 - green_value, 200 - green_value // 2)
                else:
                    cell_color = (255, 255, 255)  # Celle del bordo esterno in bianco

                pygame.draw.rect(self.window, cell_color,
                                 pygame.Rect(offset + j * self.cell_size, offset + i * self.cell_size, self.cell_size,
                                             self.cell_size))

                # Disegna il conteggio delle visite sopra la cella con un colore specifico se la stima è corretta
                if 1 <= i <= self.n and 1 <= j <= self.n:
                    if cell['marker_pred'] == 1:
                        text_color = (0, 0, 255)  # Verde per stima corretta
                    else:
                        text_color = (0, 0, 0)  # Nero per stima non corretta

                    visit_text = font.render(str(visits), True, text_color)
                    text_rect = visit_text.get_rect(center=(offset + j * self.cell_size + self.cell_size // 2,
                                                            offset + i * self.cell_size + self.cell_size // 2))
                    self.window.blit(visit_text, text_rect)

        # Disegna l'agente
        agent_center = (
            offset + (self.agent_pos[1]) * self.cell_size + self.cell_size // 2,  # Coordinata x del centro del cerchio
            offset + (self.agent_pos[0]) * self.cell_size + self.cell_size // 2  # Coordinata y del centro del cerchio
        )
        agent_radius = self.cell_size // 3  # Raggio del cerchio

        pygame.draw.circle(self.window, (255, 0, 0), agent_center, agent_radius)

        pygame.display.update()
        self.clock.tick(10)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
