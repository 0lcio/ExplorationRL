from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import math


class DoubleCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, extra_pov_radius=1, features_dim=64):
        super().__init__(observation_space, features_dim=features_dim)

        self.pov_size = 9
        self.num_features = 18
        self.obs3x3_len = 3 * 3 * self.num_features  # 162

        self.grid_size = 2 * extra_pov_radius + 3
        self.n_pov_cells = self.grid_size ** 2
        self.extra_len = self.n_pov_cells * self.pov_size

        # CNN branch for the central part (3x3)
        self.cnn_3x3 = nn.Sequential(
            nn.Conv2d(self.num_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Create dynamic CNN architecture based on extra_pov_radius
        self.extra_cnn = self._create_extra_cnn(extra_pov_radius)

        # Calculate dimensions dynamically for the fully-connected layer
        dummy_obs3x3 = torch.zeros(1, self.num_features, 3, 3)
        dummy_extra = torch.zeros(1, self.pov_size, self.grid_size, self.grid_size)

        obs3x3_out_dim = self.cnn_3x3(dummy_obs3x3).shape[1]
        extra_out_dim = self.extra_cnn(dummy_extra).shape[1]

        self.final = nn.Sequential(
            nn.Linear(obs3x3_out_dim + extra_out_dim, features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim

    def _create_extra_cnn(self, extra_pov_radius):
        """
        Create a CNN architecture for the POV grid based on extra_pov_radius.

        Returns exact architectures for radius=1, 3, and 8 as specified in the original code.
        """
        if extra_pov_radius == 1:
            # Architecture for extra_pov_radius=1 (grid_size=5)
            return nn.Sequential(
                nn.Conv2d(self.pov_size, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
        elif extra_pov_radius == 3:
            # Architecture for extra_pov_radius=3 (grid_size=9)
            return nn.Sequential(
                nn.Conv2d(self.pov_size, 32, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3),
                nn.ReLU(),
                nn.Flatten()
            )
        elif extra_pov_radius == 8:
            # Architecture for extra_pov_radius=8 (grid_size=19)
            return nn.Sequential(
                nn.Conv2d(self.pov_size, 32, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(32, 64, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3),
                nn.ReLU(),
                nn.Flatten()
            )
        else:
            # For other values, create a dynamic architecture based on grid size
            # Initial layers are common to all architectures
            layers = [
                nn.Conv2d(self.pov_size, 32, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3),
                nn.ReLU()
            ]

            effective_size = self.grid_size - 4  # After first two convolutions

            # Add MaxPooling for larger grids
            if effective_size > 6:
                layers.append(nn.MaxPool2d(kernel_size=2))
                effective_size = math.floor(effective_size / 2)

            # Add more convolution layers as needed
            channels = 32
            while effective_size > 3:
                next_channels = min(channels * 2, 128)  # Cap channel growth
                layers.extend([
                    nn.Conv2d(channels, next_channels, kernel_size=3),
                    nn.ReLU()
                ])
                channels = next_channels
                effective_size -= 2

                # Add another pooling if needed
                if effective_size > 6:
                    layers.append(nn.MaxPool2d(kernel_size=2))
                    effective_size = math.floor(effective_size / 2)

            layers.append(nn.Flatten())
            return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Separate and prepare the central block
        obs3x3_flat = x[:, :self.obs3x3_len]
        obs3x3 = obs3x3_flat.view(batch_size, 3, 3, self.num_features).permute(0, 3, 1, 2)
        obs3x3_feat = self.cnn_3x3(obs3x3)

        # Separate and prepare the POV grid
        extra_flat = x[:, self.obs3x3_len:self.obs3x3_len + self.extra_len]
        extra_pov = extra_flat.view(batch_size, self.grid_size, self.grid_size, self.pov_size).permute(0, 3, 1, 2)
        extra_feat = self.extra_cnn(extra_pov)

        combined = torch.cat((obs3x3_feat, extra_feat), dim=1)
        return self.final(combined)