# custum_map_patched.py
# Patched version of GridMappingEnv:
# - ObserverLSTM stateful per cell (streaming)
# - logits -> Gaussian message conversion (mu,var -> natural params h,J)
# - ObserverStateStore for per-cell hidden states
# - Global fusion (Lambda, eta) with Laplacian prior (dense, for now)
# - Integration into update_cell and _update_cell_state

import math 
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces

# ---------------------------
# Observer (stateful LSTM)
# ---------------------------
class ObserverLSTM(nn.Module):
    def __init__(self, dy=17, dc=9, n_classes=8, hidden=128, layers=1,
                 m0=0.0, s0=10.0):
        """
        Observer LSTM that receives observation vectors directly.
        
        Args:
            dy: observation feature dimension (default 17: 9 POV + 8 prob)
            dc: action/POV encoding dimension (default 9: one-hot POV)
            n_classes: number of prediction classes (default 8)
            hidden: LSTM hidden dimension
            layers: number of LSTM layers
            m0, s0: reference Gaussian prior
        """
        super().__init__()
        self.dy = dy
        self.dc = dc
        self.n_classes = n_classes
        self.hidden = hidden
        self.layers = layers
        
        # Reference prior
        self.m0 = float(m0)
        self.s0 = float(s0)
        self.s0_inv = 1.0 / self.s0

        # Input: observation + action + vis + gp_mean + gp_logvar + obs_count
        # All continuous features, no embeddings
        self.in_dim = dy + dc + 1 + 1 + 1 + 1
        
        self.lstm = nn.LSTM(self.in_dim, hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, hidden)
        self.logit_head = nn.Linear(hidden, n_classes)
        self.logvar_head = nn.Linear(hidden, 1)

    def step(self, y_t, a_t, vis_t, gp_mean_t, gp_logvar_t, obs_count_t, hx):
        """
        Single LSTM step.
        
        Args:
            y_t: observation [B, dy] (raw vector)
            a_t: action/POV [B, dc]
            vis_t: visibility [B, 1]
            gp_mean_t: current GP mean [B, 1]
            gp_logvar_t: current GP log-variance [B, 1]
            obs_count_t: observation count [B, 1]
            hx: LSTM hidden state
        """
        inp = torch.cat([y_t, a_t, vis_t, gp_mean_t, gp_logvar_t, obs_count_t], dim=-1).unsqueeze(1)
        out, hx_next = self.lstm(inp, hx)
        last = out[:, -1, :]
        
        h = torch.relu(self.fc(last))
        logits = self.logit_head(h)
        logvar = self.logvar_head(h)
        
        return logits, logvar, hx_next

    def forward_sequence(self, y_seq, a_seq, vis_seq, gp_mean_seq, gp_logvar_seq, 
                        obs_count_seq, lengths=None, hx=None):
        """
        Forward on a full sequence.
        
        Args:
            y_seq: observation sequence [B, T, dy]
            a_seq: action sequence [B, T, dc]
            vis_seq: visibility sequence [B, T, 1]
            gp_mean_seq: GP mean sequence [B, T, 1]
            gp_logvar_seq: GP log-var sequence [B, T, 1]
            obs_count_seq: observation count sequence [B, T, 1]
            lengths: actual sequence lengths
            hx: initial hidden state
        """
        inp = torch.cat([y_seq, a_seq, vis_seq, gp_mean_seq, gp_logvar_seq, obs_count_seq], dim=-1)
        
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                inp, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, hx_out = self.lstm(packed, hx)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, hx_out = self.lstm(inp, hx)
        
        last = out[:, -1, :]
        h = torch.relu(self.fc(last))
        logits = self.logit_head(h)
        logvar = self.logvar_head(h)
        
        return logits, logvar, hx_out

    def logits_to_site(self, logits, clamp_var_floor=1e-3):
        """Convert logits into site (h, J) via moment matching."""
        q = F.softmax(logits, dim=-1)
        k = torch.arange(1, self.n_classes + 1, dtype=q.dtype, device=q.device).unsqueeze(0)
        
        mu = (q * k).sum(-1, keepdim=True)
        var = (q * (k - mu)**2).sum(-1, keepdim=True)
        var = var.clamp_min(clamp_var_floor)
        
        s_inv = 1.0 / var
        h_site = (s_inv * mu - self.s0_inv * self.m0)
        J_site = (s_inv - self.s0_inv)
        
        return q, mu, var, h_site, J_site

# ----------------------------
# build RBF covariance matrix and precision
# ----------------------------
def build_rbf_precision(grid_H, grid_W, lengthscale=1.0, variance=1.0, jitter=1e-6, device='cpu'):
    """
    Build full covariance K for all cells and return Lambda0 = K^{-1}.
    """
    xs = np.arange(grid_H)
    ys = np.arange(grid_W)
    coords = np.array([[i, j] for i in xs for j in ys])
    N = coords.shape[0]
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1)
    K = variance * np.exp(-0.5 * d2 / (lengthscale**2))
    K += jitter * np.eye(N)
    K_t = torch.tensor(K, dtype=torch.float32, device=device)
    Lambda0 = torch.linalg.inv(K_t)
    return Lambda0, torch.tensor(coords, dtype=torch.float32, device=device)

# ---------------------------
# ObserverStateStore: manages per-cell state
# ---------------------------
class ObserverStateStore:
    def __init__(self, model: ObserverLSTM, H, W, device='cpu'):
        self.model = model
        self.device = device
        self.H = H
        self.W = W
        self.hiddens = {}
        self.last_msg = {}
        self.obs_count = {}

    def init_cell(self, cell_idx):
        """Initialize state for a cell."""
        h0 = torch.zeros(self.model.layers, 1, self.model.hidden, device=self.device)
        c0 = torch.zeros(self.model.layers, 1, self.model.hidden, device=self.device)
        self.hiddens[cell_idx] = (h0, c0)
        self.last_msg[cell_idx] = None
        self.obs_count[cell_idx] = 0

    @torch.no_grad()
    def step(self, cell_idx, y_t, a_t, vis_t, gp_mean_t, gp_logvar_t):
        """Performs an LSTM step for a cell."""
        hx = self.hiddens[cell_idx]
        device = self.device

        y_t = torch.as_tensor(y_t, dtype=torch.float32, device=device).unsqueeze(0)
        a_t = torch.as_tensor(a_t, dtype=torch.float32, device=device).unsqueeze(0)
        vis_t = torch.as_tensor([[float(vis_t)]], dtype=torch.float32, device=device)
        gp_mean_t = gp_mean_t.reshape(1, 1).to(device)
        gp_logvar_t = gp_logvar_t.reshape(1, 1).to(device)
        obs_count = torch.tensor([[float(self.obs_count[cell_idx])]], dtype=torch.float32, device=device)

        logits, logvar, hx_next = self.model.step(
            y_t, a_t, vis_t, gp_mean_t, gp_logvar_t, obs_count, hx
        )
        q, mu, var, h_site, J_site = self.model.logits_to_site(logits)

        self.hiddens[cell_idx] = hx_next
        self.obs_count[cell_idx] += 1

        msg = {
            'logits': logits.detach(), 'q': q.detach(),
            'mu': mu.detach(), 'var': var.detach(), 'logvar': logvar.detach(),
            'h_site': h_site.detach(), 'J_site': J_site.detach()
        }
        self.last_msg[cell_idx] = msg
        return msg

# ---------------------------
# Utility: convert prob -> site
# ---------------------------
def soft_site_from_probs(dist_prob, m0=0.0, s0=10.0, var_floor=1e-3, temperature=None, device='cpu'):
    """Convert probability distribution into site (h, J) via moment matching."""
    p = np.asarray(dist_prob, dtype=np.float32)
    if p.sum() == 0:
        p = np.ones_like(p) / float(len(p))
    else:
        p = p / (p.sum() + 1e-12)
    
    if temperature is not None and float(temperature) > 0 and float(temperature) != 1.0:
        p = p ** (1.0 / float(temperature))
        p = p / (p.sum() + 1e-12)
    
    k = np.arange(1, p.shape[0] + 1, dtype=np.float32)
    mu = float((p * k).sum())
    var = float((p * (k - mu) ** 2).sum())
    var = max(var, var_floor)
    
    s_inv = 1.0 / var
    s0_inv = 1.0 / float(s0)
    h = s_inv * mu - s0_inv * float(m0)
    J = s_inv - s0_inv
    
    return (torch.tensor([[h]], dtype=torch.float32, device=device), 
            torch.tensor([[J]], dtype=torch.float32, device=device), 
            mu, var, p)  

# ---------------------------
# The modified GridMappingEnv
# ---------------------------
class GridMappingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n=5, max_steps=300, render_mode=None, ig_model=None, base_model=None,
                 dataset_path='./data/final_output.csv', strategy=None, device='cpu',
                 bootstrap_gp_from_obs=True, prob_temperature=1.5, bootstrap_var=1.0):
        super(GridMappingEnv, self).__init__()
        self.n = n
        self.grid_size = n + 2
        self.ig_model = ig_model
        self.base_model = base_model
        self.dataset = pd.read_csv(dataset_path)

        print("Costruzione cache del dataset per velocità...")
        self._build_dataset_cache()
        print("Cache completata!")

        self.device = device

        # State per cell
        self.state = np.array(
            [[{'pov': np.zeros(9, dtype=np.int32),
               'best_next_pov': -1,
               'id': None,
               'marker_pred': 0,
               'obs': np.zeros((9, 17), dtype=np.float32),
               'current_entropy': torch.tensor(0.0),
               'gp_feats': {'pov_idx': [], 'gp_mean': [], 'gp_logvar': [], 'obs_count': []},
               '_gp_seen': set()}
              for _ in range(self.grid_size)]
             for _ in range(self.grid_size)]
        )

        self.agent_pos = [1, 1]
        self.max_steps = max_steps
        self.current_steps = 0
        self.render_mode = render_mode

        # Spaces
        self.action_space = spaces.Discrete(4)
        self._init_observation_space(extra_pov_radius=8)

        # Observer parameters
        self.N_CLASSES = 8
        self.dx = 1
        self.m0 = 0.0
        self.s0 = 10.0

        # LSTM Observer (no embedding)
        self.observer = ObserverLSTM(
            dy=17,  # 9 POV + 8 prob
            dc=9,   # one-hot POV
            n_classes=self.N_CLASSES,
            hidden=128,
            layers=1,
            m0=self.m0,
            s0=self.s0
        ).to(self.device)

        self.obs_store = ObserverStateStore(
            self.observer, H=self.grid_size, W=self.grid_size, device=self.device
        )

        # GP Prior
        Ncells = self.grid_size * self.grid_size
        D = Ncells * self.dx

        lengthscale = 2.0
        variance = 1.0
        jitter = 1e-5
        self.Lambda0, self._coords = build_rbf_precision(
            self.grid_size, self.grid_size,
            lengthscale=lengthscale,
            variance=variance,
            jitter=jitter,
            device=self.device
        )

        m0_vec = torch.full((D,), float(self.m0), device=self.device)
        self.eta0 = self.Lambda0 @ m0_vec
        self.Lambda = self.Lambda0.clone()
        self.eta = self.eta0.clone()
        self.global_mean = torch.linalg.solve(
            self.Lambda + 1e-6 * torch.eye(D, device=self.device),
            self.eta
        )
        self._msg_cache = {}
        self._global_cov = None

        # Bootstrap options
        self.bootstrap_gp_from_obs = bool(bootstrap_gp_from_obs)
        self.prob_temperature = float(prob_temperature)
        self.bootstrap_var = float(bootstrap_var)

        # Strategy
        self.strategy = f"pred_{strategy}" if strategy is not None else "pred_none"

        # Rendering
        self.window_size = 600
        self.cell_size = self.window_size // self.grid_size
        self.window = None
        self.clock = None

    def _build_dataset_cache(self):
        """Organizza il dataset in un dizionario per accesso istantaneo."""
        self.dataset_cache = {}
        # Raggruppa per le chiavi che usiamo per filtrare
        grouped = self.dataset.groupby(['IMAGE_ID', 'BOX_COUNT', 'MARKER_COUNT'])
        
        for name, group in grouped:
            # name è una tupla (image_id, box_count, marker_count)
            # group è il DataFrame contenente solo quelle righe
            self.dataset_cache[name] = group

    # -------------------------
    # env reset
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        self.state = np.array(
            [[{'pov': np.zeros(9, dtype=np.int32),
               'best_next_pov': -1,
               'id': None,
               'marker_pred': 0,
               'obs': np.zeros((9, 17), dtype=np.float32),
               'current_entropy': torch.tensor(0.0),
               'gp_feats': {'pov_idx': [], 'gp_mean': [], 'gp_logvar': [], 'obs_count': []},
               '_gp_seen': set()}
              for _ in range(self.grid_size)]
             for _ in range(self.grid_size)]
        )
        
        self.agent_pos = [1, 1]
        self._assign_ids_to_cells()

        # Reset GP
        D = (self.grid_size * self.grid_size) * self.dx
        self.Lambda = self.Lambda0.clone()
        self.eta = self.eta0.clone()
        self.global_mean = torch.linalg.solve(
            self.Lambda + 1e-6 * torch.eye(D, device=self.device), self.eta
        )
        self._msg_cache = {}

        # Reset observers
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_idx = r * self.grid_size + c
                self.obs_store.init_cell(cell_idx)

        if self.strategy in ['pred_ig_reward', 'pred_no_train', 'pred_random_agent']:
            self._update_pov_ig(self.agent_pos, self.agent_pos)
        else:
            self._update_pov_best_view(self.agent_pos)

        self.current_steps = 0
        if self.render_mode == 'human':
            self.render()

        return self._get_observation_double_cnn(), {}

    # -------------------------
    # assign random ids to cells
    # TODO: change to coherent assignment based on gaussian latent field
    # -------------------------
    def _assign_ids_to_cells(self):
        for i in range(1, self.n + 1):
            for j in range(1, self.n + 1):
                random_row = self.dataset.sample(n=1, random_state=self.np_random.integers(0, 2 ** 32 - 1)).iloc[0]
                self.state[i, j]['id'] = {
                    'IMAGE_ID': random_row['IMAGE_ID'],
                    'BOX_COUNT': random_row['BOX_COUNT'],
                    'MARKER_COUNT': random_row['MARKER_COUNT']
                }

    def _cell_index(self, r, c):
        return r * self.grid_size + c

    # -------------------------
    # replace message + global solve
    # -------------------------
    def replace_message_and_solve(self, cell_idx, new_h, new_J):
        dx = self.dx
        sl = slice(cell_idx * dx, (cell_idx + 1) * dx)

        # Remove previous message
        old = self._msg_cache.get(cell_idx, None)
        if old is not None:
            old_h = old['h'].squeeze().item()
            old_J = old['J'].squeeze().item()
            self.Lambda[sl, sl] = self.Lambda[sl, sl] - old_J
            self.eta[sl] = self.eta[sl] - old_h

        # Add new message
        h_add = new_h.squeeze().item()
        J_add = new_J.squeeze().item()

        self.Lambda[sl, sl] = self.Lambda[sl, sl] + J_add
        self.eta[sl] = self.eta[sl] + h_add

        self._msg_cache[cell_idx] = {
            'h': new_h.clone().to(self.device),
            'J': new_J.clone().to(self.device)
        }

        # Solve global system
        jitter = 1e-6
        try:
            m_vec = torch.linalg.solve(
                self.Lambda + jitter * torch.eye(self.Lambda.size(0), device=self.device),
                self.eta
            )
        except RuntimeError:
            m_vec = torch.linalg.solve(
                self.Lambda + 1e-3 * torch.eye(self.Lambda.size(0), device=self.device),
                self.eta
            )
        self.global_mean = m_vec

        # Compute covariance
        try:
            cov = torch.linalg.inv(
                self.Lambda + 1e-6 * torch.eye(self.Lambda.size(0), device=self.device)
            )
            self._global_cov = cov
        except RuntimeError:
            cov = torch.diag(1.0 / torch.diagonal(self.Lambda).clamp_min(1e-12))
            self._global_cov = cov

        return m_vec

    # -------------------------
    # step_score (unchanged semantics)
    # -------------------------
    def step_score(self, action):
        prev_pos = list(self.agent_pos)
        temp_pos = list(self.agent_pos)

        if action == 0:  # up
            temp_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:  # right
            temp_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 2:  # down
            temp_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)
        elif action == 3:  # left
            temp_pos[1] = max(self.agent_pos[1] - 1, 0)

        action_score = self._update_pov_ig(temp_pos, prev_pos, update=False)
        action_score += 2

        return action_score

    # -------------------------
    # step (main loop)
    # -------------------------
    def step(self, action):
        self.current_steps += 1
        prev_pos = list(self.agent_pos)

        # Execute the action
        self._move_agent(action)

        # Compute the reward
        if self.strategy in ('pred_ig_reward', 'pred_no_train', 'pred_random_agent'):
            reward = self._update_pov_ig(self.agent_pos, prev_pos)
        else:
            new_pov_observed, best_next_pov_visited = self._update_pov_best_view(self.agent_pos)
            reward = self._calculate_reward_best_view(new_pov_observed, best_next_pov_visited, prev_pos)

        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_steps >= self.max_steps
        if terminated:
            reward += 30

        if self.render_mode == 'human':
            self.render()

        return self._get_observation_double_cnn(), reward, terminated, truncated, {}

    def _move_agent(self, action):
        if action == 0:
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 2:
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)
        elif action == 3:
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)

    def _check_termination(self):
        all_cells_correct, all_wrong_cells_visited_9_pov = True, True
        for row in self.state[1:self.n + 1, 1:self.n + 1]:
            for cell in row:
                if cell['marker_pred'] == 0:
                    all_cells_correct = False
                    if sum(cell['pov']) != 9:
                        all_wrong_cells_visited_9_pov = False
                        break
        return all_cells_correct or all_wrong_cells_visited_9_pov

    # -------------------------
    # POV IG update (uses streaming observer)
    # -------------------------
    def _update_pov_ig(self, agent_pos, prev_pos, update=True):
        ax, ay = agent_pos
        grid_min, grid_max = 1, self.n
        total_reward = 0.0
        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = ax + i, ay + j
                if grid_min <= nx <= grid_max and grid_min <= ny <= grid_max:
                    cell = self.state[nx, ny]
                    input_array = self.update_cell(cell, i, j, update=update)
                    if isinstance(input_array, np.ndarray) and input_array.size > 0:
                        total_reward += self._calculate_reward_ig(cell, input_array, update)
        if self.agent_pos == prev_pos:
            total_reward -= 2
        return float(total_reward)

    def update_cell(self, cell, i, j, update):
        pov_index = (i + 1) * 3 + (j + 1)

        # If the viewpoint has already been observed, do not update
        if cell['pov'][pov_index] == 1:
            return 0  # No reward added if already observed

        cell_povs = cell['pov'].copy()
        cell_povs[pov_index] = 1
        # Update the observation state
        if update:
            cell['pov'][pov_index] = 1

        # Get indices of observed viewpoints
        observed_indices = np.flatnonzero(cell_povs)

        # Create the input array for the model based on observed viewpoints
        input_array = self._get_cell_input_array(cell, observed_indices)

        # Update the cell's 'obs' matrix
        if update:
            m = input_array.shape[0]
            cell['obs'][:m, :] = input_array

        return input_array

    def _get_cell_input_array(self, cell, observed_indices):
        input_list = []
        
        # --- VERSIONE VELOCE (CON CACHE) ---
        # Creiamo la chiave per cercare nel dizionario
        key = (cell["id"]['IMAGE_ID'], cell["id"]['BOX_COUNT'], cell["id"]['MARKER_COUNT'])
        
        # Recuperiamo i dati istantaneamente. Se non esistono, restituisce un DataFrame vuoto.
        filtered_data = self.dataset_cache.get(key, pd.DataFrame())
        # -----------------------------------

        for pov in observed_indices:
            row = filtered_data[filtered_data["POV_ID"] == pov + 1]
            if not row.empty:
                dist_prob = np.array([row[f"P{i}"] for i in range(8)]).flatten()
                pov_id_hot = np.zeros(9)
                pov_id_hot[pov] = 1
                input_list.append(np.concatenate((pov_id_hot, dist_prob)))

        return np.array(input_list, dtype=np.float32)

    def _calculate_reward_ig(self, cell, input_array, update=True):
        total_reward = 0.0
        
        for row in input_array:
            pov_onehot = row[:9]
            dist_prob = row[9:]
            y_t = row.copy().astype(np.float32)
            a_t = pov_onehot.astype(np.float32)

            # Identify cell coordinates
            if '_coords' not in cell or cell['_coords'] is None:
                found = False
                for rr in range(self.grid_size):
                    for cc in range(self.grid_size):
                        if self.state[rr, cc] is cell:
                            cell['_coords'] = (rr, cc)
                            found = True
                            break
                    if found:
                        break

            nx, ny = cell['_coords']
            cell_idx = self._cell_index(nx, ny)

            # Extract current GP info
            sl = slice(cell_idx * self.dx, (cell_idx + 1) * self.dx)
            gp_mean_cell = self.global_mean[sl].reshape(1, 1)
            approx_prec = self.Lambda[sl, sl].squeeze().item()
            gp_var_cell = 1.0 / max(approx_prec, 1e-9)
            gp_logvar_cell = torch.tensor(
                [[math.log(gp_var_cell)]], dtype=torch.float32, device=self.device
            )

            # PHASE 1: Bootstrap from observations or LSTM
            if self.bootstrap_gp_from_obs:
                # Bootstrap: build site from dist_prob
                h_site, J_site, mu_obs, var_obs, q_obs = soft_site_from_probs(
                    dist_prob, m0=self.m0, s0=self.s0,
                    var_floor=1e-3, temperature=self.prob_temperature, device=self.device
                )
                
                msg = {
                    'q': torch.tensor(q_obs, device=self.device).unsqueeze(0),
                    'mu': torch.tensor([[mu_obs]], device=self.device),
                    'var': torch.tensor([[var_obs]], device=self.device),
                    'h_site': h_site,
                    'J_site': J_site
                }
                
                # Store in gp_feats
                pov_idx = int(np.argmax(pov_onehot))
                gpbuf = cell.get('gp_feats', None)
                if gpbuf is not None and pov_idx not in cell.get('_gp_seen', set()):
                    gpbuf['pov_idx'].append(pov_idx)
                    gpbuf['gp_mean'].append(float(mu_obs))
                    gpbuf['gp_logvar'].append(float(math.log(var_obs)))
                    obs_cnt = float(self.obs_store.obs_count.get(cell_idx, 0))
                    gpbuf['obs_count'].append(obs_cnt)
                    cell['_gp_seen'].add(pov_idx)

                # Update observer store (maintains hx consistency)
                _ = self.obs_store.step(
                    cell_idx, y_t, a_t, vis_t=1,
                    gp_mean_t=torch.tensor([[mu_obs]], device=self.device),
                    gp_logvar_t=torch.tensor([[math.log(var_obs)]], device=self.device)
                )

            else:
                # PHASE 2: LSTM produces site
                msg = self.obs_store.step(
                    cell_idx, y_t, a_t, vis_t=1,
                    gp_mean_t=gp_mean_cell,
                    gp_logvar_t=gp_logvar_cell
                )
                
                # Store LSTM output in gp_feats
                pov_idx = int(np.argmax(pov_onehot))
                gpbuf = cell.get('gp_feats', None)
                if gpbuf is not None and pov_idx not in cell.get('_gp_seen', set()):
                    mu_pred = float(msg['mu'].squeeze().item())
                    var_pred = float(msg['var'].squeeze().item())
                    obs_cnt = float(self.obs_store.obs_count.get(cell_idx, 0) - 1)
                    gpbuf['pov_idx'].append(pov_idx)
                    gpbuf['gp_mean'].append(mu_pred)
                    gpbuf['gp_logvar'].append(float(math.log(max(var_pred, 1e-6))))
                    gpbuf['obs_count'].append(obs_cnt)
                    cell['_gp_seen'].add(pov_idx)

                h_site = msg['h_site']
                J_site = msg['J_site']

            # Use site to update GP
            if 'h_site' in msg and 'J_site' in msg:
                h_site_use = msg['h_site']
                J_site_use = msg['J_site']
            else:
                h_site_use = h_site
                J_site_use = J_site

            q = msg['q']
            mu = msg['mu']
            var = msg['var']

            # Compute current entropy
            current_entropy = -(q * (q + 1e-12).log()).sum().detach()
            cell['current_entropy'] = current_entropy

            # Check prediction correct
            pred_class = torch.argmax(q, dim=1).item() + 1
            if pred_class == cell['id']['MARKER_COUNT']:
                if update:
                    cell['marker_pred'] = 1

            # Compute Information Gain
            prev_ent = cell.get('_last_entropy', torch.tensor(1.0))
            ig = (prev_ent - current_entropy).item()
            if ig < 0:
                ig = 0.0
            total_reward += ig
            cell['_last_entropy'] = current_entropy

            # Update global GP
            self.replace_message_and_solve(cell_idx, h_site_use, J_site_use)

        return total_reward

    # -------------------------
    # best-view update (alternative strategy)
    # -------------------------
    def _update_pov_best_view(self, agent_pos):
        ax, ay = agent_pos
        new_pov_count = 0
        best_next_pov_visited = 0
        grid_min, grid_max = 1, self.n

        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = ax + i, ay + j
                if grid_min <= nx <= grid_max and grid_min <= ny <= grid_max:
                    cell = self.state[nx, ny]
                    pov_index = (i + 1) * 3 + (j + 1)
                    if cell['pov'][pov_index] == 0:
                        cell['pov'][pov_index] = 1
                        if cell['marker_pred'] == 0:
                            new_pov_count += 1
                        if cell['best_next_pov'] == pov_index:
                            best_next_pov_visited += 1
                    # Update the cell state with base_model (legacy)
                    self._update_cell_state(cell)
        return new_pov_count, best_next_pov_visited

    def _update_cell_state(self, cell):
        observed_indices = np.flatnonzero(cell['pov'])

        input_list = []
        
        # --- VERSIONE VELOCE (CON CACHE) ---
        key = (cell["id"]['IMAGE_ID'], cell["id"]['BOX_COUNT'], cell["id"]['MARKER_COUNT'])
        filtered_data = self.dataset_cache.get(key, pd.DataFrame())
        # -----------------------------------
        
        for pov in observed_indices:
            row = filtered_data[filtered_data["POV_ID"] == pov + 1]
            if not row.empty:
                dist_prob = np.array([row[f"P{i}"] for i in range(8)]).flatten()
                pov_id_hot = np.zeros(9)
                pov_id_hot[pov] = 1
                input_list.append(np.concatenate((pov_id_hot, dist_prob)))

        input_array = np.array(input_list, dtype=np.float32)
        if input_array.size > 0:
            input_tensor = torch.tensor(input_array).to(self.device)
        else:
            # Se è vuoto, creiamo un tensore vuoto sulla GPU per evitare errori
            input_tensor = torch.tensor([], device=self.device)

        m = input_array.shape[0]
        # Protezione: se m è 0 (nessun dato trovato), non fare nulla per evitare crash
        if m > 0:
            cell['obs'][:m, :] = input_array

        if len(observed_indices) != 9:
            if self.strategy == 'pred_random' or self.strategy == "pred_random_agent":
                next_best_pov = torch.randint(0, 9, (1,)).item()
            else:
                # Assicuriamoci che input_tensor non sia vuoto
                if input_tensor.nelement() > 0:
                    outputs = self.ig_model(input_tensor)
                    
                    # --- FIX KEYERROR ---
                    # Se la strategia è 'pred_ig_reward', dobbiamo leggere 'pred_entropy'
                    if self.strategy == 'pred_ig_reward':
                        target_key = 'pred_entropy'
                    else:
                        target_key = self.strategy
                    
                    # Controllo di sicurezza: se la chiave non esiste, usiamo 'pred_entropy' di default
                    if target_key not in outputs:
                        target_key = 'pred_entropy'
                        
                    ig_prediction = outputs[target_key]
                    # --------------------
                    
                    next_best_pov = int(torch.argmin(ig_prediction).item())
                else:
                    next_best_pov = -1 

            cell['best_next_pov'] = next_best_pov
        else:
            cell['best_next_pov'] = -1

        if input_tensor.nelement() > 0:
            base_model_pred = self.base_model(input_tensor)
            if torch.argmax(base_model_pred, 1) == cell["id"]['MARKER_COUNT']:
                cell['marker_pred'] = 1
    
    def _calculate_reward_best_view(self, new_pov_observed, best_next_pov_visited, prev_pos):
        reward = 0.0
        reward += new_pov_observed * 2
        reward += best_next_pov_visited * 5
        if self.agent_pos == prev_pos:
            reward -= 2
        return reward

    # -------------------------
    # Observations retrieval (for agent)
    # -------------------------
    def _get_observation(self):
        """3x3 observation around the agent."""
        obs = torch.zeros((3, 3, 18))
        ax, ay = self.agent_pos

        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = ax + i, ay + j
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    cell_obs = self.state[nx, ny]['obs']
                    curr_entropy = self.state[nx, ny]['current_entropy'].unsqueeze(0).detach()
                    cell_povs = torch.tensor(
                        self.state[nx, ny]['pov'], dtype=torch.float32
                    ).unsqueeze(0).detach()
                    
                    filtered_obs = cell_obs[~np.all(cell_obs == 0, axis=1)]
                    
                    if filtered_obs.size > 0:
                        if '_coords' not in self.state[nx, ny]:
                            self.state[nx, ny]['_coords'] = (nx, ny)
                        
                        cell_idx = self._cell_index(nx, ny)
                        last_msg = self.obs_store.last_msg.get(cell_idx)
                        
                        if last_msg is not None:
                            q = last_msg['q'].squeeze(0)
                        else:
                            try:
                                marker_pre = self.base_model(torch.tensor(filtered_obs))
                                marker_pre_softmax = F.softmax(marker_pre, dim=1).mean(dim=0).detach()
                                q = marker_pre_softmax
                            except Exception:
                                q = torch.ones(self.N_CLASSES) / float(self.N_CLASSES)
                        
                        obs[i + 1, j + 1] = torch.cat(
                            (curr_entropy, q, cell_povs), dim=1
                        ).squeeze(0)
        
        return obs.detach()

    def _get_observation_double_cnn(self, extra_pov_radius=8):
        """Extended observation: detailed 3x3 + larger POV grid."""
        obs_3x3 = torch.zeros((3, 3, 18))
        pov_size = len(self.state[0, 0]['pov'])
        ax, ay = self.agent_pos

        grid_span = 2 * extra_pov_radius + 3
        pov_grid = torch.zeros((grid_span, grid_span, pov_size))

        # Central 3x3 grid with detailed info
        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = ax + i, ay + j
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    cell_obs = self.state[nx, ny]['obs']
                    curr_entropy = self.state[nx, ny]['current_entropy'].unsqueeze(0).detach()
                    cell_povs = torch.tensor(
                        self.state[nx, ny]['pov'], dtype=torch.float32
                    ).unsqueeze(0).detach()
                    
                    filtered_obs = cell_obs[~np.all(cell_obs == 0, axis=1)]
                    
                    if filtered_obs.size > 0:
                        if '_coords' not in self.state[nx, ny]:
                            self.state[nx, ny]['_coords'] = (nx, ny)
                        
                        cell_idx = self._cell_index(nx, ny)
                        last_msg = self.obs_store.last_msg.get(cell_idx)
                        
                        if last_msg is not None:
                            q = last_msg['q'].squeeze(0)
                        else:
                            try:
                                marker_pre = self.base_model(torch.tensor(filtered_obs))
                                marker_pre_softmax = F.softmax(marker_pre, dim=1).mean(dim=0).detach()
                                q = marker_pre_softmax
                            except Exception:
                                q = torch.ones(self.N_CLASSES) / float(self.N_CLASSES)

                        curr_entropy_2d = curr_entropy.unsqueeze(0)
                        q_2d = q.unsqueeze(0)
                        curr_entropy_2d = curr_entropy_2d.to(self.device)
                        q_2d = q_2d.to(self.device)
                        cell_povs = cell_povs.to(self.device)
                        
                        obs_3x3[i + 1, j + 1] = torch.cat(
                            (curr_entropy_2d, q_2d, cell_povs), dim=1
                        ).squeeze(0)

        # Extended POV grid
        for i in range(-extra_pov_radius - 1, extra_pov_radius + 2):
            for j in range(-extra_pov_radius - 1, extra_pov_radius + 2):
                gx, gy = i + extra_pov_radius + 1, j + extra_pov_radius + 1
                nx, ny = ax + i, ay + j
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    cell_povs = torch.tensor(
                        self.state[nx, ny]['pov'], dtype=torch.float32
                    ).detach()
                    pov_grid[gx, gy] = cell_povs
                else:
                    pov_grid[gx, gy] = torch.zeros(pov_size)

        # Concatenate observations
        obs_3x3_flat = obs_3x3.view(-1)
        extra_pov_flat = pov_grid.view(-1)
        all_obs = torch.cat((obs_3x3_flat, extra_pov_flat), dim=0)
        
        return all_obs.detach()

    def _init_observation_space(self, extra_pov_radius=1):
        n_center = 3 * 3 * 18
        n = 2 * extra_pov_radius + 3
        n_pov_cells = n * n
        pov_size = len(self.state[0, 0]['pov'])
        total_obs_len = n_center + n_pov_cells * pov_size
        self.observation_space = spaces.Box(low=0, high=1, shape=(total_obs_len,), dtype=np.float32)

    # -------------------------
    # Rendering (kept minimal)
    # -------------------------
    def render(self, mode='human'):
        # optional pygame visualization: keep minimal to avoid dependency issues
        print(f"Agent pos: {self.agent_pos} step {self.current_steps}")
    
# ----------------------------
# Supervised training function
# Uses stored gp_feats to build sequences
# ----------------------------
def train_observer_on_env(env, observer_model, optimizer, device='cpu'):
    """
    Supervised training of the Observer LSTM.
    
    Uses data stored in gp_feats for each cell to build
    training sequences with (y, a, vis, gp_mean, gp_logvar, obs_count).
    """
    observer_model.train()
    criterion = nn.CrossEntropyLoss()

    sequences = []
    targets = []
    lengths = []

    # Collect sequences from all cells
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            cell = env.state[r, c]
            obs = cell['obs']
            obs_nonzero = obs[~np.all(obs == 0, axis=1)]
            
            if obs_nonzero.size == 0:
                continue

            T = obs_nonzero.shape[0]
            y_seq = torch.tensor(obs_nonzero, dtype=torch.float32, device=device)
            a_seq = torch.tensor(obs_nonzero[:, :9], dtype=torch.float32, device=device)
            vis_seq = torch.ones((T, 1), dtype=torch.float32, device=device)

            # Build GP sequences from stored gp_feats
            gpbuf = cell.get('gp_feats', None)
            gp_mean_seq = None
            gp_logvar_seq = None
            obs_count_seq = None

            if gpbuf and len(gpbuf['pov_idx']) > 0:
                gp_mean_list = []
                gp_logvar_list = []
                obs_count_list = []
                
                for t in range(T):
                    pov_onehot = obs_nonzero[t, :9]
                    pov_idx = int(np.argmax(pov_onehot))
                    
                    if pov_idx in gpbuf['pov_idx']:
                        k = gpbuf['pov_idx'].index(pov_idx)
                        gp_mean_list.append(gpbuf['gp_mean'][k])
                        gp_logvar_list.append(gpbuf['gp_logvar'][k])
                        obs_count_list.append(gpbuf['obs_count'][k])
                    else:
                        # Fallback: use current global mean
                        cell_idx = env._cell_index(r, c)
                        sl = slice(cell_idx * env.dx, (cell_idx + 1) * env.dx)
                        approx_mean = float(env.global_mean[sl].item())
                        approx_prec = env.Lambda[sl, sl].squeeze().item()
                        approx_var = 1.0 / max(approx_prec, 1e-9)
                        gp_mean_list.append(approx_mean)
                        gp_logvar_list.append(float(math.log(approx_var)))
                        obs_count_list.append(float(t))

                gp_mean_seq = torch.tensor(gp_mean_list, dtype=torch.float32, device=device).unsqueeze(-1)
                gp_logvar_seq = torch.tensor(gp_logvar_list, dtype=torch.float32, device=device).unsqueeze(-1)
                obs_count_seq = torch.tensor(obs_count_list, dtype=torch.float32, device=device).unsqueeze(-1)
            else:
                # Fallback: zero sequences
                gp_mean_seq = torch.zeros((T, 1), dtype=torch.float32, device=device)
                gp_logvar_seq = torch.zeros((T, 1), dtype=torch.float32, device=device)
                obs_count_seq = torch.arange(0, T, dtype=torch.float32, device=device).unsqueeze(-1) / 9.0

            sequences.append((y_seq, a_seq, vis_seq, gp_mean_seq, gp_logvar_seq, obs_count_seq))
            marker_count = cell['id']['MARKER_COUNT']

            num_classes = 8
            marker_count = max(0, min(marker_count, num_classes - 1))

            targets.append(marker_count)
            lengths.append(T)

    if len(sequences) == 0:
        return 0.0

    # Train on each sequence
    total_loss = 0.0
    for i, seq in enumerate(sequences):
        y_seq, a_seq, vis_seq, gp_mean_seq, gp_logvar_seq, obs_count_seq = seq

        # Batch size 1 per sequence
        y_b = y_seq.unsqueeze(0)
        a_b = a_seq.unsqueeze(0)
        vis_b = vis_seq.unsqueeze(0)
        gp_mean_b = gp_mean_seq.unsqueeze(0)
        gp_logvar_b = gp_logvar_seq.unsqueeze(0)
        obs_count_b = obs_count_seq.unsqueeze(0)
        lengths_b = torch.tensor([lengths[i]], dtype=torch.long, device=device)

        # Forward pass
        logits, logvar, _ = observer_model.forward_sequence(
            y_b, a_b, vis_b, gp_mean_b, gp_logvar_b, obs_count_b, 
            lengths=lengths_b
        )

        # Loss and backprop
        target = torch.tensor([targets[i]], dtype=torch.long, device=device)
        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(sequences)
