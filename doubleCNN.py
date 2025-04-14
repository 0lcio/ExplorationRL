from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

class DoubleCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, extra_pov_radius=1):
        super().__init__(observation_space, features_dim=64)

        self.pov_size = 9
        self.num_features = 18
        self.obs3x3_len = 3 * 3 * self.num_features  # 162

        self.grid_size = 2 * extra_pov_radius + 3
        self.n_pov_cells = self.grid_size ** 2
        self.extra_len = self.n_pov_cells * self.pov_size

        # Ramo CNN per la parte centrale (3x3)
        self.cnn_3x3 = nn.Sequential(
            nn.Conv2d(self.num_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Ramo CNN per la POV completa [grid_size x grid_size]
        self.extra_cnn = nn.Sequential(
            nn.Conv2d(self.pov_size, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calcolo dinamico delle dimensioni per il layer fully-connected
        dummy_obs3x3 = torch.zeros(1, self.num_features, 3, 3)
        dummy_extra = torch.zeros(1, self.pov_size, self.grid_size, self.grid_size)

        obs3x3_out_dim = self.cnn_3x3(dummy_obs3x3).shape[1]
        extra_out_dim = self.extra_cnn(dummy_extra).shape[1]

        self.final = nn.Sequential(
            nn.Linear(obs3x3_out_dim + extra_out_dim, 64),
            nn.ReLU()
        )

        self._features_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Separa e prepara il blocco centrale
        obs3x3_flat = x[:, :self.obs3x3_len]
        obs3x3 = obs3x3_flat.view(batch_size, 3, 3, self.num_features).permute(0, 3, 1, 2)
        obs3x3_feat = self.cnn_3x3(obs3x3)

        # Separa e prepara la griglia POV
        extra_flat = x[:, self.obs3x3_len:self.obs3x3_len + self.extra_len]
        extra_pov = extra_flat.view(batch_size, self.grid_size, self.grid_size, self.pov_size).permute(0, 3, 1, 2)
        extra_feat = self.extra_cnn(extra_pov)

        combined = torch.cat((obs3x3_feat, extra_feat), dim=1)
        return self.final(combined)
