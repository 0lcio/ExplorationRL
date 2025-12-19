import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    """
    Feature extractor basato su CNN per osservazioni.
    FIX: Adattato per prendere in input un vettore piatto e ricostruire la vista 3x3.
    """

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        # --- FIX INIZIO ---
        # Non possiamo leggere obs_shape[2] perché l'input è piatto (es. 3411,)
        # Sappiamo dal codice dell'ambiente che ogni cella ha 18 feature (9 POV + 8 Prob + 1 Entropy)
        num_features = 18
        
        # Calcoliamo quanto è grande la parte "locale" (3x3 celle)
        self.local_view_len = 3 * 3 * num_features # 162
        # --- FIX FINE ---

        # CNN per elaborare l'osservazione 3x3 con 18 canali di input
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=num_features, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calcola la dimensione dell'output della CNN
        # Creiamo un dummy input della forma corretta [Batch, Canali, Altezza, Larghezza]
        dummy_input = torch.zeros(1, num_features, 3, 3) 
        
        with torch.no_grad():
            cnn_output_dim = self.cnn(dummy_input).shape[1]

        # Proiezione nello spazio latente
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        """
        Forward pass della CNN:
        1. Prende l'input piatto.
        2. Estrae solo la parte locale (primi 162 valori).
        3. Reshapa in 3x3x18.
        4. Passa alla CNN.
        """
        # 1. Gestione Input Piatto
        # observations arriva come [batch_size, total_len] (es. 3411)
        # Noi vogliamo solo i primi 162 valori che corrispondono alla griglia 3x3
        batch_size = observations.shape[0]
        
        # Prendi solo la parte locale
        local_obs_flat = observations[:, :self.local_view_len]
        
        # 2. Reshape: [batch, 3, 3, 18]
        # Ricostruiamo la griglia 3x3
        local_obs = local_obs_flat.view(batch_size, 3, 3, 18)

        # 3. Permuta: [batch, 3, 3, 18] -> [batch, 18, 3, 3]
        # PyTorch vuole i canali (18) come seconda dimensione
        local_obs = local_obs.permute(0, 3, 1, 2)

        # 4. Passaggio nella CNN
        features = self.cnn(local_obs)

        # Proiezione finale
        return self.fc(features)