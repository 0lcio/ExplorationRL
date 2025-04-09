import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    Feature extractor basato su CNN per osservazioni di dimensione [batch, 3, 3, 9].
    Ogni cella contiene 9 feature estratte da un modello intermedio.
    """

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        # Estrarre la forma dell'input: (3, 3, 18)
        obs_shape = observation_space.shape  # (H, W, num_features)
        num_features = obs_shape[2]  # 18 feature per cella

        # CNN per elaborare l'osservazione 3x3 con 9 canali di input
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=num_features, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calcola la dimensione dell'output della CNN
        dummy_input = torch.zeros(1, num_features, obs_shape[0], obs_shape[1])  # [1, 18, 3, 3]
        with torch.no_grad():
            cnn_output_dim = self.cnn(dummy_input).shape[1]  # Ottiene il numero di feature dopo la CNN

        # Proiezione nello spazio latente
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        """
        Forward pass della CNN:
        - Permuta le dimensioni per adattarle all'input della CNN.
        - Passa l'input attraverso la CNN per estrarre feature spaziali.
        - Proietta il risultato in uno spazio latente di dimensione `features_dim`.
        """
        # Permuta: [batch, 3, 3, 18] → [batch, 18, 3, 3] (feature come canali di input)
        observations = observations.permute(0, 3, 1, 2)

        # Passaggio nella CNN per estrarre feature
        features = self.cnn(observations)  # Output: [batch, CNN_output_dim]

        # Proiezione nello spazio latente finale
        return self.fc(features)
