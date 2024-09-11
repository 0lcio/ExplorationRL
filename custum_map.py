import gymnasium as gym
import torch
from gymnasium import spaces
import numpy as np
import pygame
import pandas as pd


class GridMappingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n=5, max_steps=300, render_mode=None, ig_model=None, base_model=None,
                 dataset_path='./data/final_output.csv', strategy=None):
        super(GridMappingEnv, self).__init__()
        self.n = n  # Dimensione della griglia originale
        self.grid_size = n + 2  # Dimensione della griglia con bordo
        self.ig_model = ig_model  # Modello per il miglior punto di vista successivo
        self.base_model = base_model  # Modello per la stima dello stato delle celle
        self.state = np.array(
            [[{'pov': np.zeros(9, dtype=np.int32), 'best_next_pov': -1, 'id': None, 'marker_pred': 0}
              for _ in range(self.grid_size)] for _ in range(self.grid_size)])
        self.agent_pos = [1, 1]  # Posizione iniziale dell'agente
        self.max_steps = max_steps
        self.current_steps = 0
        self.render_mode = render_mode

        # Spazio d'azione e osservazione
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 3, 9), dtype=np.int32)

        # Caricamento dataset
        self.dataset = pd.read_csv(dataset_path)

        # Inizializza Pygame
        self.window_size = 600
        self.cell_size = self.window_size // self.grid_size
        self.window = None
        self.clock = None

        # Selezione della strategia
        self.strategy = f"pred_{strategy}"
        # self.unexplored_cells_count = n * n * 9

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Usa np_random per creare il generatore di numeri casuali
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Resetta stato e agent position
        self.state = np.array(
            [[{'pov': np.zeros(9, dtype=np.int32), 'best_next_pov': -1, 'id': None, 'marker_pred': 0}
              for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        )
        self.agent_pos = [1, 1]
        self._assign_ids_to_cells()
        self._update_pov(self.agent_pos)
        self.current_steps = 0

        if self.render_mode == 'human':
            self.render()

        return self._get_observation(), {}

    def _assign_ids_to_cells(self):
        for i in range(1, self.n + 1):
            for j in range(1, self.n + 1):
                random_row = self.dataset.sample(n=1, random_state=self.np_random.integers(0, 2 ** 32 - 1)).iloc[0]
                self.state[i, j]['id'] = {
                    'IMAGE_ID': random_row['IMAGE_ID'],
                    'BOX_COUNT': random_row['BOX_COUNT'],
                    'MARKER_COUNT': random_row['MARKER_COUNT']
                }

    def step(self, action):
        self.current_steps += 1

        # Salva la posizione precedente
        prev_pos = list(self.agent_pos)

        # Esegui l'azione
        if action == 0:  # su
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:  # destra
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 2:  # giù
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)
        elif action == 3:  # sinistra
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)

        # Calcola il reward
        new_pov_observed, best_next_pov_visited = self._update_pov(self.agent_pos)
        reward = self._calculate_reward(new_pov_observed, best_next_pov_visited, prev_pos)

        # Verifica combinata delle condizioni per terminare l'episodio
        all_cells_correct = True
        all_wrong_cells_visited_9_pov = True

        for row in self.state[1:self.n + 1, 1:self.n + 1]:
            for cell in row:
                if cell['marker_pred'] == 0:
                    all_cells_correct = False  # Trovata cella non corretta
                    # Verifica se la cella non corretta è stata visitata da tutti e 9 i punti di vista
                    if cell['marker_pred'] == 0 and sum(cell['pov']) != 9:
                        all_wrong_cells_visited_9_pov = False  # Trovata cella non completamente visitata
                        break  # Uscita anticipata, la condizione è falsa

        # Termina l'episodio se entrambe le condizioni sono soddisfatte
        terminated = all_cells_correct or all_wrong_cells_visited_9_pov
        truncated = self.current_steps >= self.max_steps

        if terminated:
            reward += 20

        if self.render_mode == 'human':
            self.render()

        return self._get_observation(), reward, terminated, truncated, {}

    def _calculate_reward(self, new_pov_observed, best_next_pov_visited, prev_pos):
        # Inizializza il reward
        reward = new_pov_observed * 1
        reward += best_next_pov_visited * 8.0

        # Penalizzazione se l'agente rimane fermo
        if self.agent_pos == prev_pos:
            reward -= 2  # Penalità per stato fermo

        # for i in range(-1, 2):
        #     for j in range(-1, 2):
        #         nx, ny = self.agent_pos[0] + i, self.agent_pos[1] + j
        #         if 1 <= nx <= self.n and 1 <= ny <= self.n:
        #             cell = self.state[nx, ny]
        #             if sum(cell['pov']) != 9 and cell['marker_pred'] == 0:
        #                 reward += 2

        return reward

    def _update_pov(self, agent_pos):
        ax, ay = agent_pos
        new_pov_count = 0
        best_next_pov_visited = 0
        grid_min, grid_max = 1, self.n

        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = ax + i, ay + j
                if grid_min <= nx <= grid_max and grid_min <= ny <= grid_max:
                    cell = self.state[nx, ny]
                    pov_index = (i + 1) * 3 + (j + 1)

                    if cell['pov'][pov_index] == 0:
                        cell['pov'][pov_index] = 1
                        # se osserva una cella con stima sbagliata da una nuova posizione
                        if cell['marker_pred'] == 0:
                            new_pov_count += 1

                        if cell['best_next_pov'] == pov_index:
                            best_next_pov_visited += 1

                        # Riduci il contatore di celle non completamente esplorate
                        # self.unexplored_cells_count -= 1

                    # Aggiornamento dello stato della cella
                    self._update_cell_state(cell)

        return new_pov_count, best_next_pov_visited

    def _update_cell_state(self, cell):
        observed_indices = np.flatnonzero(cell['pov'])

        input_list = []
        filtered_data = self.dataset[
            (self.dataset["IMAGE_ID"] == cell["id"]['IMAGE_ID']) &
            (self.dataset["BOX_COUNT"] == cell["id"]['BOX_COUNT']) &
            (self.dataset["MARKER_COUNT"] == cell["id"]['MARKER_COUNT'])
            ]

        for pov in observed_indices:
            row = filtered_data[filtered_data["POV_ID"] == pov + 1]
            if not row.empty:
                dist_prob = np.array([row[f"P{i}"] for i in range(8)]).flatten()
                pov_id_hot = np.zeros(9)
                pov_id_hot[pov] = 1
                input_list.append(np.concatenate((pov_id_hot, dist_prob)))

        input_array = np.array(input_list, dtype=np.float32)
        input_tensor = torch.tensor(input_array)

        if len(observed_indices) != 9:
            if self.strategy == 'pred_random' or self.strategy == "pred_random_agent":
                next_best_pov = torch.randint(0, 9, (1,)).item()
            else:
                ig_prediction = self.ig_model(input_tensor)[self.strategy]
                next_best_pov = int(torch.argmin(ig_prediction).item())

            cell['best_next_pov'] = next_best_pov
        else:
            cell['best_next_pov'] = -1

        base_model_pred = self.base_model(input_tensor)
        if torch.argmax(base_model_pred, 1) == cell["id"]['MARKER_COUNT']:
            cell['marker_pred'] = 1

    def _get_observation(self):
        obs = np.zeros((3, 3, 9), dtype=np.int32)
        ax, ay = self.agent_pos

        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = ax + i, ay + j
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    obs[i + 1, j + 1] = self.state[nx, ny]['pov']

        return obs

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
