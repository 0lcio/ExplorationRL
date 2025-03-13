import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    Feature extractor basato su CNN per elaborare osservazioni in ambienti 3x3 con viste multiple.
    L'input è una griglia 3x3, dove ogni cella può essere osservata da diverse prospettive (POV),
    e ogni POV ha 17 feature associate.

    Il modello:
    - Combina tutte le viste disponibili come canali di input.
    - Utilizza una CNN per estrarre caratteristiche spaziali condivise.
    - Proietta le feature estratte in uno spazio latente compatto.
    """

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        # Estrai la forma dell'input: (3, 3, 9, 17)
        obs_shape = observation_space.shape  # (H, W, num_views, num_features)
        num_views = obs_shape[2]  # Numero di punti di vista (POV) per cella
        num_features = obs_shape[3]  # Numero di feature per ogni POV

        # CNN per elaborare l'osservazione combinata
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=num_features * num_views, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calcola la dimensione dell'output della CNN
        dummy_input = torch.zeros(1, num_features * num_views, obs_shape[0], obs_shape[1])
        with torch.no_grad():
            cnn_output_dim = self.cnn(dummy_input).shape[1]  # Ottiene il numero di feature dopo la CNN

        # Proiezione nello spazio latente
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        """
        Forward pass della CNN.
        - Permuta le dimensioni per adattarle all'input della CNN.
        - Unisce le feature di tutte le viste in un unico tensore.
        - Passa l'input attraverso la CNN per estrarre le feature.
        - Proietta il risultato in uno spazio latente di dimensione `features_dim`.
        """

        # Permuta: [batch, H, W, num_views, num_features] → [batch, num_features, num_views, H, W]
        observations = observations.permute(0, 4, 3, 1, 2)

        # Raggruppa i POV come canali di input: [batch, num_features * num_views, H, W]
        batch_size, num_features, num_views, h, w = observations.shape
        observations = observations.reshape(batch_size, num_features * num_views, h, w)

        # Passaggio nella CNN per estrarre feature
        features = self.cnn(observations)  # Output: [batch, CNN_output_dim]

        # Proiezione nello spazio latente finale
        return self.fc(features)
