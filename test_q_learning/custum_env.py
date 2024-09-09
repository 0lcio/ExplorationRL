import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class GridMappingEnv(gym.Env):
    """
    Custom Environment per la mappatura di un'area nxn con visualizzazione Pygame.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, n=5, max_steps=100):
        super(GridMappingEnv, self).__init__()
        self.n = n  # Dimensione della griglia
        self.state = np.zeros((n, n), dtype=np.int32)  # Stato rappresentato come griglia n x n
        self.agent_pos = [0, 0]  # Posizione iniziale dell'agente
        self.max_steps = max_steps  # Numero massimo di passi consentiti
        self.current_steps = 0  # Inizializza il contatore dei passi

        # Definiamo lo spazio d'azione (4 azioni: su, destra, giù, sinistra)
        self.action_space = spaces.Discrete(4)
        # Lo stato ora è l'indice della cella dell'agente: da 0 a n*n - 1
        self.observation_space = spaces.Discrete(n * n)

        # Inizializza Pygame
        self.window_size = 600  # Dimensione della finestra
        self.cell_size = self.window_size // self.n  # Dimensione delle celle

        self.window = None
        self.clock = None

    def reset(self):
        # Resetta la griglia e l'agente
        self.state = np.zeros((self.n, self.n), dtype=np.int32)
        self.agent_pos = [0, 0]
        self.state[self.agent_pos[0], self.agent_pos[1]] = 1
        self.current_steps = 0  # Resetta il contatore dei passi
        return self._get_state()  # Ritorna l'osservazione iniziale

    def step(self, action):
        # Incrementa il contatore dei passi
        self.current_steps += 1

        # Definiamo le regole di movimento dell'agente
        if action == 0:  # su
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:  # destra
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.n - 1)
        elif action == 2:  # giù
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.n - 1)
        elif action == 3:  # sinistra
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)

        # Segna la cella come visitata
        self.state[self.agent_pos[0], self.agent_pos[1]] = 1

        # Calcola il reward
        reward = -0.1  # Penalità costante per incoraggiare il movimento

        # Controlla se l'agente ha visitato tutte le celle o ha raggiunto il massimo dei passi
        if np.all(self.state == 1):
            reward = 1  # Grande reward quando tutte le celle sono visitate
            done = True
        elif self.current_steps >= self.max_steps:
            done = True  # Termina l'episodio se supera il numero massimo di passi
        else:
            done = False

        return self._get_state(), reward, done, {}

    def _get_state(self):
        """
        Restituisce l'indice dello stato corrente in base alla posizione dell'agente.
        """
        return self.agent_pos[0] * self.n + self.agent_pos[1]

    def render(self, mode='human'):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Riempie lo sfondo con bianco
        self.window.fill((255, 255, 255))

        # Disegna la griglia
        for i in range(self.n):
            for j in range(self.n):
                cell_color = (200, 200, 200) if self.state[i, j] == 0 else (0, 255, 0)
                pygame.draw.rect(self.window, cell_color,
                                 pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))

        # Disegna l'agente
        agent_center = (
            self.agent_pos[1] * self.cell_size + self.cell_size // 2,  # Coordinata x del centro del cerchio
            self.agent_pos[0] * self.cell_size + self.cell_size // 2  # Coordinata y del centro del cerchio
        )
        agent_radius = self.cell_size // 3  # Raggio del cerchio

        pygame.draw.circle(self.window, (255, 0, 0), agent_center, agent_radius)

        pygame.display.update()
        self.clock.tick(10)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
