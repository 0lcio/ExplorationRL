from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
import torch.nn as nn
import torch


class DoubleCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, n_extra_cells=7, pov_size=9):
        # Calcola la dimensione dell'input flat
        super().__init__(observation_space, features_dim=64)  # Output della rete

        self.n_extra_cells = n_extra_cells
        self.pov_size = pov_size
        self.obs3x3_len = 3 * 3 * 18
        self.extra_len = self.n_extra_cells * self.pov_size

        self.cnn_3x3 = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=2),  # [B, 18, 3, 3] -> [B, 32, 2, 2]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
        )

        self.extra_processor = nn.Sequential(
            nn.Linear(pov_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        self.final = nn.Sequential(
            nn.Linear(64 + self.n_extra_cells * 16, 64),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        obs3x3_flat = x[:, :self.obs3x3_len]
        extra_flat = x[:, self.obs3x3_len:self.obs3x3_len + self.extra_len]

        obs3x3 = obs3x3_flat.view(batch_size, 3, 3, 18).permute(0, 3, 1, 2)  # [B, 18, 3, 3]
        obs3x3_feat = self.cnn_3x3(obs3x3)

        extra_pov = extra_flat.view(batch_size * self.n_extra_cells, self.pov_size)
        extra_feat = self.extra_processor(extra_pov)
        extra_feat = extra_feat.view(batch_size, -1)

        combined = torch.cat((obs3x3_feat, extra_feat), dim=1)
        return self.final(combined)
