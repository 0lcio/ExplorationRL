import sys
import numpy as np
import pygame
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.utils import seeding
from typing import Optional

# Azioni
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Mappa
MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}

class FrozenLakeCustomEnv(Env):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="4x4",
        is_slippery=True,
        max_steps: int = 100,
    ):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            new_row, new_col = inc(row, col, action)
            new_state = to_s(new_row, new_col)
            new_letter = desc[new_row, new_col]
            terminated = bytes(new_letter) in b"GH"
            reward = float(new_letter == b"G")
            return new_state, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (1.0 / 3.0, *update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0

        # Initialize rendering
        self.window_size = 600  # Dimensione della finestra
        self.window = None
        self.clock = None

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a
        self.current_step += 1

        if self.current_step >= self.max_steps:
            t = True  # Termina l'episodio se si raggiunge il massimo dei passi

        if self.render_mode == "human":
            self.render()

        return int(s), r, t, False, {"prob": p}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        self.current_step = 0

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        elif self.render_mode == "human":
            self._render_gui()
        else:
            raise NotImplementedError(f"Render mode {self.render_mode} not supported.")

    def _render_text(self):
        desc = self.desc.tolist()
        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        return "\n".join("".join(line) for line in desc) + "\n"

    def _render_gui(self):
        # Importare pygame per la gestione della finestra
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Disegna la griglia
        cell_size = self.window_size // self.nrow
        for row in range(self.nrow):
            for col in range(self.ncol):
                cell_value = self.desc[row, col].decode("utf-8")
                color = (255, 255, 255)  # Default: Bianco

                if cell_value == "H":
                    color = (0, 0, 0)  # Buco: Nero
                elif cell_value == "G":
                    color = (0, 255, 0)  # Obiettivo: Verde
                elif cell_value == "S":
                    color = (0, 0, 255)  # Inizio: Blu

                pygame.draw.rect(
                    self.window,
                    color,
                    pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                )

        # Disegna il giocatore
        row, col = self.s // self.ncol, self.s % self.ncol
        pygame.draw.circle(
            self.window,
            (255, 0, 0),  # Giocatore: Rosso
            (col * cell_size + cell_size // 2, row * cell_size + cell_size // 2),
            cell_size // 3
        )

        # Aggiorna la finestra
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
