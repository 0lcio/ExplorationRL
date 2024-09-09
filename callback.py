from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggerCallback(BaseCallback):
    """
    Callback per registrare i reward cumulativi per ogni episodio,
    il numero di celle con `marker_pred` a 1, il numero di celle viste da 9 punti di vista,
    e il numero di step per ogni episodio.
    """

    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []  # Lista per tenere traccia dei reward per episodio
        self.episode_cells_marker_pred_1 = []  # Lista per il numero di celle con marker_pred a 1 per episodio
        self.episode_cells_seen_pov = []  # Lista per il numero di celle viste da 9 POV per episodio
        self.episode_steps = []  # Lista per il numero di step per episodio
        self.episode_reward = 0.0  # Reward cumulativo per l'episodio corrente

        # Insiemi per tracciare le celle contate durante l'episodio
        self.cells_marker_pred_1_set = set()
        self.cells_seen_pov_set = set()
        self.steps = 0  # Contatore per gli step dell'episodio corrente

    def _on_step(self) -> bool:
        # Aggiungi il reward corrente al reward cumulativo
        self.episode_reward += self.locals["rewards"][0]
        env = self.locals["env"].envs[0]

        # Incrementa il contatore degli step
        self.steps += 1

        # Aggiorna i conteggi solo per le celle che non sono state contate prima
        new_cells_marker_pred_1 = self._count_new_cells_with_marker_pred_1(env)
        new_cells_seen_9_pov = self._count_new_cells_seen_pov(env)

        # Aggiungi le nuove celle contate ai set
        self.cells_marker_pred_1_set.update(new_cells_marker_pred_1)
        self.cells_seen_pov_set.update(new_cells_seen_9_pov)

        # Quando l'episodio termina, salva le metriche
        if self.locals["dones"][0]:
            # Salva il reward cumulativo
            self.episode_rewards.append(self.episode_reward)

            # Salva le metriche
            self.episode_cells_marker_pred_1.append(len(self.cells_marker_pred_1_set))
            self.episode_cells_seen_pov.append(len(self.cells_seen_pov_set))
            self.episode_steps.append(self.steps)  # Salva il numero di step per l'episodio corrente

            # Resetta le variabili per il prossimo episodio
            self.episode_reward = 0.0
            self.cells_marker_pred_1_set.clear()
            self.cells_seen_pov_set.clear()
            self.steps = 0  # Resetta il contatore degli step

        return True

    def _count_new_cells_with_marker_pred_1(self, env):
        """
        Conta le nuove celle con `marker_pred` a 1 nell'ambiente.
        Ritorna l'insieme di coordinate delle celle nuove.
        """
        new_cells = set()
        for x in range(1, env.unwrapped.n + 1):
            for y in range(1, env.unwrapped.n + 1):
                if env.unwrapped.state[x, y]['marker_pred'] == 1:
                    if (x, y) not in self.cells_marker_pred_1_set:
                        new_cells.add((x, y))
        return new_cells

    def _count_new_cells_seen_pov(self, env):
        """
        Conta le nuove celle viste da 0 punti di vista differenti nell'ambiente.
        Ritorna l'insieme di coordinate delle celle nuove.
        """
        new_cells = set()
        for x in range(1, env.unwrapped.n + 1):
            for y in range(1, env.unwrapped.n + 1):
                if sum(env.unwrapped.state[x, y]['pov']) == 0:
                    if (x, y) not in self.cells_seen_pov_set:
                        new_cells.add((x, y))
        return new_cells
