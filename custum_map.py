import gymnasium as gym
import torch
from gymnasium import spaces
import numpy as np
import pygame
import pandas as pd
import torch.nn.functional as F
from helper import information_gain, entropy


class GridMappingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n=5, max_steps=300, render_mode=None, ig_model=None, base_model=None,
                 dataset_path='./data/final_output.csv', strategy=None, device='cpu'):
        super(GridMappingEnv, self).__init__()
        self.n = n  # Dimensione della griglia originale
        self.grid_size = n + 2  # Dimensione della griglia con bordo
        self.ig_model = ig_model  # Modello per il miglior punto di vista successivo
        self.base_model = base_model  # Modello per la stima dello stato delle celle
        self.device = device
        self.state = np.array(
            [[{'pov': np.zeros(9, dtype=np.int32),
               'best_next_pov': -1,
               'id': None,
               'marker_pred': 0,
               'obs': np.zeros((9, 17), dtype=np.float32),
               'current_entropy': entropy(torch.full((8,), 1 / 8))}  # entropia iniziale uniforme
              for _ in range(self.grid_size)]
             for _ in range(self.grid_size)]
        )
        self.agent_pos = [1, 1]  # Posizione iniziale dell'agente
        self.max_steps = max_steps
        self.current_steps = 0
        self.render_mode = render_mode

        # Spazio d'azione e osservazione
        self.action_space = spaces.Discrete(4)
        self._init_observation_space(extra_pov_radius=8)
        # self.observation_space = spaces.Box(low=0, high=1, shape=(3, 3, 18), dtype=np.float32)

        # Caricamento dataset
        self.dataset = pd.read_csv(dataset_path)

        print("Indicizzazione dataset in corso per velocità...")
        self.fast_data = {}
        # Iteriamo una volta sola sul dataset per preparare i dati
        for _, row in self.dataset.iterrows():
            # Chiave univoca: (IMAGE_ID, BOX_COUNT, MARKER_COUNT, POV_ID)
            # POV_ID nel CSV è 1-based (1..9), nel codice usiamo 0..8 spesso, quindi occhio agli indici
            key = (row['IMAGE_ID'], row['BOX_COUNT'], row['MARKER_COUNT'], int(row['POV_ID']))
            
            # Pre-calcoliamo il vettore finale (POV One-Hot + Probabilità)
            dist_prob = np.array([row[f"P{i}"] for i in range(8)], dtype=np.float32)
            pov_id_hot = np.zeros(9, dtype=np.float32)
            pov_id_hot[int(row['POV_ID']) - 1] = 1.0  # POV_ID 1 diventa indice 0
            
            # Salviamo il vettore già pronto concatenato (dimensione 17)
            self.fast_data[key] = np.concatenate((pov_id_hot, dist_prob))
        print("Indicizzazione completata.")

        # Inizializza Pygame
        self.window_size = 600
        self.cell_size = self.window_size // self.grid_size
        self.window = None
        self.clock = None

        # Selezione della strategia
        self.strategy = f"pred_{strategy}"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Usa np_random per creare il generatore di numeri casuali
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.state = np.array(
            [[{'pov': np.zeros(9, dtype=np.int32),
               'best_next_pov': -1,
               'id': None,
               'marker_pred': 0,
               'obs': np.zeros((9, 17), dtype=np.float32),
               'current_entropy': entropy(torch.full((8,), 1 / 8))}  # entropia iniziale uniforme
              for _ in range(self.grid_size)]
             for _ in range(self.grid_size)]
        )
        self.agent_pos = [1, 1]
        self._assign_ids_to_cells()
        if self.strategy == 'pred_ig_reward' or self.strategy == 'pred_no_train' or self.strategy == 'pred_random_agent':
            self._update_pov_ig(self.agent_pos, self.agent_pos)
        else:
            self._update_pov_best_view(self.agent_pos)
        self.current_steps = 0

        if self.render_mode == 'human':
            self.render()

        return self._get_observation_double_cnn(), {}

    def _assign_ids_to_cells(self):
        # Generiamo tutti gli indici casuali in un colpo solo invece di chiamare .sample() 400 volte
        # Questo è istantaneo
        total_cells = self.n * self.n
        random_indices = self.np_random.integers(0, len(self.dataset), size=total_cells)
        
        k = 0
        for i in range(1, self.n + 1):
            for j in range(1, self.n + 1):
                # Accesso diretto tramite iloc (molto più veloce di sample)
                # Usiamo i valori precaricati se possibile, altrimenti accesso veloce
                idx = random_indices[k]
                k += 1
                
                # Accediamo direttamente ai valori del dataframe usando .iloc[idx]
                # Nota: self.dataset.iloc[idx] ritorna una Series, accediamo ai campi
                # Per massima velocità, sarebbe meglio convertire il dataset in dict all'init,
                # ma questo è già 100x più veloce del .sample()
                row = self.dataset.iloc[idx]
                
                self.state[i, j]['id'] = {
                    'IMAGE_ID': row['IMAGE_ID'],
                    'BOX_COUNT': row['BOX_COUNT'],
                    'MARKER_COUNT': row['MARKER_COUNT']
                }

    def step_score(self, action):
        prev_pos = list(self.agent_pos)
        temp_pos = list(self.agent_pos)

        if action == 0:  # su
            temp_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:  # destra
            temp_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 2:  # giù
            temp_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)
        elif action == 3:  # sinistra
            temp_pos[1] = max(self.agent_pos[1] - 1, 0)

        action_score = self._update_pov_ig(temp_pos, prev_pos, update=False)
        action_score += 2

        return action_score

    def step(self, action):
        self.current_steps += 1
        prev_pos = list(self.agent_pos)

        # Esegui l'azione
        self._move_agent(action)

        # Calcola il reward
        if self.strategy == 'pred_ig_reward' or self.strategy == 'pred_no_train' or self.strategy == 'pred_random_agent':
            reward = self._update_pov_ig(self.agent_pos, prev_pos)
        else:
            new_pov_observed, best_next_pov_visited = self._update_pov_best_view(self.agent_pos)
            reward = self._calculate_reward_best_view(new_pov_observed, best_next_pov_visited, prev_pos)

        # Verifica condizioni di terminazione
        terminated = self._check_termination()
        truncated = self.current_steps >= self.max_steps
        if terminated:
            reward += 30

        if self.render_mode == 'human':
            self.render()

        return self._get_observation_double_cnn(), reward, terminated, truncated, {}

    def _move_agent(self, action):
        # Movimenti
        if action == 0:  # su
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:  # destra
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 2:  # giù
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)
        elif action == 3:  # sinistra
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)

    def _check_termination(self):
        all_cells_correct, all_wrong_cells_visited_9_pov = True, True

        for row in self.state[1:self.n + 1, 1:self.n + 1]:
            for cell in row:
                if cell['marker_pred'] == 0:
                    all_cells_correct = False
                    if sum(cell['pov']) != 9:
                        all_wrong_cells_visited_9_pov = False
                        break

        return all_cells_correct or all_wrong_cells_visited_9_pov

    def _update_pov_ig(self, agent_pos, prev_pos, update=True):
        ax, ay = agent_pos
        grid_min, grid_max = 1, self.n
        total_reward = 0

        # Cicla su tutte le celle intorno all'agente (3x3)
        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = ax + i, ay + j
                if grid_min <= nx <= grid_max and grid_min <= ny <= grid_max:
                    cell = self.state[nx, ny]

                    # Aggiorna lo stato della cella e ottieni l'input array
                    input_array = self.update_cell(cell, i, j, update=update)

                    # Se abbiamo un nuovo input (cioè una nuova osservazione), calcola il reward
                    if isinstance(input_array, np.ndarray) and input_array.size > 0:
                        total_reward += self._calculate_reward_ig(cell, input_array, update)

        # Penalità per restare nella stessa posizione
        if self.agent_pos == prev_pos:
            total_reward -= 2

        return total_reward

    def update_cell(self, cell, i, j, update):
        pov_index = (i + 1) * 3 + (j + 1)

        # Se il punto di vista è già stato osservato, non aggiornare
        if cell['pov'][pov_index] == 1:
            return 0  # Nessun reward aggiunto se già osservato

        cell_povs = cell['pov'].copy()
        cell_povs[pov_index] = 1
        # Aggiorna lo stato di osservazione
        if update:
            cell['pov'][pov_index] = 1

        # Ottieni gli indici dei punti di vista osservati
        observed_indices = np.flatnonzero(cell_povs)

        # Crea l'input array per il modello in base ai punti di vista osservati
        input_array = self._get_cell_input_array(cell, observed_indices)

        # Aggiorna la matrice 'obs' della cella
        if update:
            m = input_array.shape[0]
            cell['obs'][:m, :] = input_array

        return input_array

    def _calculate_reward_ig(self, cell, input_array, update=True):
        # Calcola l'information gain solo se si osserva da un nuovo punto di vista
        base_model_pred = self.base_model(torch.tensor(input_array).to(self.device))
        expected_entropy = entropy(base_model_pred)
        # information gain
        reward = cell['current_entropy'] - expected_entropy
        reward = reward.item()
        if reward < 0:
            reward = 0

        cell['current_entropy'] = expected_entropy
        # Se il modello predice correttamente il marker, aggiorna lo stato della cella
        if torch.argmax(base_model_pred, 1) == cell["id"]['MARKER_COUNT']:
            if update:
                cell['marker_pred'] = 1

        return reward

    # custum_map.py

    def _get_cell_input_array(self, cell, observed_indices):
        input_list = []
        
        # Recupera gli ID velocemente (senza fare query)
        img_id = cell["id"]['IMAGE_ID']
        box_cnt = cell["id"]['BOX_COUNT']
        mrk_cnt = cell["id"]['MARKER_COUNT']

        # Usa il dizionario self.fast_data invece di self.dataset[...]
        for pov in observed_indices:
            # Crea la chiave (img, box, marker, pov). Ricorda: dataset POV è 1-based
            key = (img_id, box_cnt, mrk_cnt, pov + 1)
            
            if key in self.fast_data:
                # O(1) Lookup istantaneo
                input_list.append(self.fast_data[key])

        if not input_list:
            return np.array([], dtype=np.float32)

        return np.array(input_list, dtype=np.float32)

    def _calculate_reward_best_view(self, new_pov_observed, best_next_pov_visited, prev_pos):
        reward = new_pov_observed * 1 + best_next_pov_visited * 8.0
        if self.agent_pos == prev_pos:
            reward -= 2
        return reward

    def _update_pov_best_view(self, agent_pos):
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

                    # Aggiornamento dello stato della cella
                    self._update_cell_state(cell)

        return new_pov_count, best_next_pov_visited

    def _update_cell_state(self, cell):
        observed_indices = np.flatnonzero(cell['pov'])

        # --- INIZIO MODIFICA OTTIMIZZATA ---
        input_list = []
        
        # Recuperiamo gli ID necessari per la chiave di ricerca
        img_id = cell["id"]['IMAGE_ID']
        box_cnt = cell["id"]['BOX_COUNT']
        mrk_cnt = cell["id"]['MARKER_COUNT']

        # Invece di filtrare il dataset Pandas (lento), usiamo il dizionario (veloce)
        for pov in observed_indices:
            # Nota: 'pov' nel loop parte da 0, nel dataset POV_ID parte da 1
            key = (img_id, box_cnt, mrk_cnt, pov + 1)
            
            if key in self.fast_data:
                # O(1) Lookup immediato
                input_list.append(self.fast_data[key])
        # --- FINE MODIFICA OTTIMIZZATA ---

        # Se non abbiamo input, usiamo un array vuoto per evitare crash
        if not input_list:
            input_array = np.zeros((0, 17), dtype=np.float32)
        else:
            input_array = np.array(input_list, dtype=np.float32)
            
        input_tensor = torch.tensor(input_array).to(self.device) # Assicurati di mandare al device corretto

        # Aggiorna la memoria 'obs' della cella
        m = input_array.shape[0]
        if m > 0:
            cell['obs'][:m, :] = input_array

        # Logica originale per il prossimo miglior POV (IG Model)
        if len(observed_indices) != 9:
            if self.strategy == 'pred_random' or self.strategy == "pred_random_agent":
                next_best_pov = torch.randint(0, 9, (1,)).item()
            else:
                # Gestione caso input vuoto per il modello IG
                if input_tensor.shape[0] > 0:
                    ig_prediction = self.ig_model(input_tensor)[self.strategy]
                    next_best_pov = int(torch.argmin(ig_prediction).item())
                else:
                    # Fallback se non ci sono dati osservati (raro ma possibile)
                    next_best_pov = torch.randint(0, 9, (1,)).item()

            cell['best_next_pov'] = next_best_pov
        else:
            cell['best_next_pov'] = -1

        # Logica originale per la predizione del marker (Base Model)
        if input_tensor.shape[0] > 0:
            base_model_pred = self.base_model(input_tensor)
            # Nota: base_model_pred è (Batch, 8), argmax su dim 1
            # Bisogna vedere se il modello predice correttamente il marker
            # Solitamente si prende la media delle predizioni o l'ultima, 
            # qui il codice originale controllava se l'argmax combaciava.
            # ATTENZIONE: Il codice originale faceva un check diretto sull'output.
            # Qui mantengo la logica originale[cite: 71]:
            if torch.argmax(base_model_pred, 1)[-1] == cell["id"]['MARKER_COUNT']: # Prendo l'ultima osservazione o gestisci come preferisci
                 cell['marker_pred'] = 1
                 
        # NOTA: Nel codice originale [cite: 71] c'era:
        # if torch.argmax(base_model_pred, 1) == cell["id"]['MARKER_COUNT']:
        # Questo funziona bene se base_model_pred ha dimensione 1 (una sola riga).
        # Se observed_indices > 1, base_model_pred sarà (N, 8). 
        # PyTorch potrebbe dare errore o risultati inattesi confrontando un vettore con uno scalare.
        # Ho aggiunto [-1] per prendere l'osservazione più recente, o puoi usare la logica che preferisci.

    def _get_observation(self):
        obs = torch.zeros((3, 3, 18)).to(self.device)
        ax, ay = self.agent_pos

        for i in range(-1, 2):  # Da -1 a 1 (inclusi)
            for j in range(-1, 2):  # Da -1 a 1 (inclusi)
                nx, ny = ax + i, ay + j
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    cell_obs = self.state[nx, ny]['obs']
                    currunt_entropy = self.state[nx, ny]['current_entropy'].unsqueeze(0).detach()
                    cell_povs = torch.tensor(self.state[nx, ny]['pov'], dtype=torch.float32).unsqueeze(0).detach()

                    currunt_entropy = currunt_entropy.to(self.device)
                    cell_povs = cell_povs.to(self.device)

                    # Filtra le righe che non contengono solo zeri
                    filtered_obs = cell_obs[~np.all(cell_obs == 0, axis=1)]
                    if filtered_obs.size > 0:
                        marker_pre = self.base_model(torch.tensor(filtered_obs))
                        marker_pre_softmax = F.softmax(marker_pre, dim=1).detach()
                        obs[i + 1, j + 1] = torch.cat((currunt_entropy, marker_pre_softmax, cell_povs), dim=1)
        return obs.detach()

    def _get_observation_double_cnn(self, extra_pov_radius=8):
        obs_3x3 = torch.zeros((3, 3, 18))
        pov_size = len(self.state[0, 0]['pov'])  # Dimensione del POV
        ax, ay = self.agent_pos

        # Dimensione totale: n x n
        grid_span = 2 * extra_pov_radius + 3
        pov_grid = torch.zeros((grid_span, grid_span, pov_size))

        # Costruzione dell'osservazione principale (3x3)
        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = ax + i, ay + j
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    cell_obs = self.state[nx, ny]['obs']
                    curr_entropy = self.state[nx, ny]['current_entropy'].unsqueeze(0).detach()
                    cell_povs = torch.tensor(self.state[nx, ny]['pov'], dtype=torch.float32).unsqueeze(0).detach()

                    curr_entropy = curr_entropy.to(self.device)
                    cell_povs = cell_povs.to(self.device)

                    filtered_obs = cell_obs[~np.all(cell_obs == 0, axis=1)]
                    if filtered_obs.size > 0:
                        marker_pre = self.base_model(torch.tensor(filtered_obs).to(self.device))
                        marker_pre_softmax = F.softmax(marker_pre, dim=1).detach()
                        obs_3x3[i + 1, j + 1] = torch.cat((curr_entropy, marker_pre_softmax, cell_povs), dim=1)

        # Costruzione POV grid [n, n, pov_size]
        for i in range(-extra_pov_radius - 1, extra_pov_radius + 2):
            for j in range(-extra_pov_radius - 1, extra_pov_radius + 2):
                gx, gy = i + extra_pov_radius + 1, j + extra_pov_radius + 1  # Indici nella POV grid
                nx, ny = ax + i, ay + j  # Coordinate nella mappa

                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    cell_povs = torch.tensor(self.state[nx, ny]['pov'], dtype=torch.float32).detach()
                    pov_grid[gx, gy] = cell_povs
                else:
                    pov_grid[gx, gy] = torch.zeros(pov_size)  # Fuori dalla griglia → padding

        obs_3x3_flat = obs_3x3.view(-1)
        extra_pov_flat = pov_grid.view(-1)  # Flatten [n*n*9]

        all_obs = torch.cat((obs_3x3_flat, extra_pov_flat), dim=0)

        return all_obs.detach()

    def _init_observation_space(self, extra_pov_radius=1):
        n_center = 3 * 3 * 18  # Parte centrale fissa
        n = 2 * extra_pov_radius + 3  # Dimensione lato della griglia POV
        n_pov_cells = n * n  # Celle nella POV grid (incluso centro)
        pov_size = len(self.state[0, 0]['pov'])  # Tipicamente 9

        total_obs_len = n_center + n_pov_cells * pov_size
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(total_obs_len,),
            dtype=np.float32
        )

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
        # self.clock.tick(10)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
