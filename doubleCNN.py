from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

class DoubleCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_extra_cells=16):
        """
        observation_space: spazio delle osservazioni (flattened)
        n_extra_cells: numero di celle extra; deve essere un quadrato perfetto (ad es. 16 per griglia 4x4)
        """
        # Impostiamo un features_dim temporaneo; verrà usato lo strato finale per ottenere 64 feature.
        super().__init__(observation_space, features_dim=64)

        self.n_extra_cells = n_extra_cells  # deve essere un quadrato perfetto
        self.pov_size = 9
        self.obs3x3_len = 3 * 3 * 18  # 162
        self.num_features = 18
        self.extra_len = self.n_extra_cells * self.pov_size  # ad es. 16*9 = 144

        # Ramo per le osservazioni 3x3 (input: [B, 18, 3, 3])
        self.cnn_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_features, out_channels=32, kernel_size=3, stride=1, padding=1),  # mantiene [B,32,3,3]
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),                # [B,64,3,3]
            nn.ReLU(),
            nn.Flatten(),  # dimensione: 64 * 3 * 3 = 576
        )

        # Per le celle extra, assumiamo che siano disposte in una griglia quadrata
        self.extra_grid = int(self.n_extra_cells ** 0.5)
        assert self.extra_grid ** 2 == self.n_extra_cells, "n_extra_cells deve essere un quadrato perfetto"

        # Ramo CNN per le celle extra:
        self.extra_cnn = nn.Sequential(
            nn.Conv2d(in_channels=self.pov_size, out_channels=32, kernel_size=3, padding=1),  # [B,32,extra_grid,extra_grid]
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),               # [B,32,extra_grid,extra_grid]
            nn.ReLU(),
            nn.Flatten(),  # dimensione: 32 * extra_grid * extra_grid = 32 * n_extra_cells
        )

        # Calcolo dinamico della dimensione combinata usando un dummy forward
        dummy_obs3x3 = torch.zeros(1, self.num_features, 3, 3)  # dimensione: [1, 18, 3, 3]
        obs3x3_out_dim = self.cnn_3x3(dummy_obs3x3).shape[1]

        dummy_extra = torch.zeros(1, self.pov_size, self.extra_grid, self.extra_grid)  # [1, 9, extra_grid, extra_grid]
        extra_out_dim = self.extra_cnn(dummy_extra).shape[1]

        combined_dim = obs3x3_out_dim + extra_out_dim  # dimensione combinata

        # Strato finale: ora robusto, basato sul calcolo dinamico
        self.final = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
        )

        # Aggiorniamo la proprietà features_dim per BaseFeaturesExtractor
        self._features_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Estrai e processa il ramo 3x3
        obs3x3_flat = x[:, :self.obs3x3_len]  # [B, 162]
        # Ripristina la forma come immagine: [B, 3, 3, 18] e poi permuta → [B, 18, 3, 3]
        obs3x3 = obs3x3_flat.view(batch_size, 3, 3, 18).permute(0, 3, 1, 2)
        obs3x3_feat = self.cnn_3x3(obs3x3)  # [B, 576] (o altro numero calcolato)

        # Estrai e processa il ramo delle celle extra
        extra_flat = x[:, self.obs3x3_len:self.obs3x3_len + self.extra_len]  # [B, n_extra_cells * pov_size]
        # Riordina per avere una griglia: da [B, n_extra_cells, pov_size]
        extra_pov = extra_flat.reshape(batch_size, self.n_extra_cells, self.pov_size)
        # Risistema come immagine: [B, extra_grid, extra_grid, pov_size] e permuta → [B, pov_size, extra_grid, extra_grid]
        extra_pov = extra_pov.view(batch_size, self.extra_grid, self.extra_grid, self.pov_size).permute(0, 3, 1, 2)
        extra_feat = self.extra_cnn(extra_pov)  # [B, 32 * n_extra_cells]

        # Concatenazione dei due rami
        combined = torch.cat((obs3x3_feat, extra_feat), dim=1)

        return self.final(combined)
