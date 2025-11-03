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
    def __init__(self, dy=17, dc=9, de=8, n_classes=8, hidden=128, layers=1,
                 m0=0.0, s0=10.0):  # ADDED m0, s0
        super().__init__()
        self.dy = dy
        self.dc = dc
        self.de = de
        self.n_classes = n_classes
        self.hidden = hidden
        self.layers = layers
        # NEW: reference prior
        self.m0 = float(m0)
        self.s0 = float(s0)
        self.s0_inv = 1.0 / self.s0

        # MODIFIED: Input dims now include gp_mean, gp_logvar, obs_count
        self.in_dim = dy + dc + 1 + 1 + 1 + 1  # +vis +gp_mean +gp_logvar +obs_count
        self.lstm = nn.LSTM(self.in_dim, hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden + de, hidden)
        self.logit_head = nn.Linear(hidden, n_classes)
        self.logvar_head = nn.Linear(hidden, 1)

    def step(self, y_t, a_t, vis_t, gp_mean_t, gp_logvar_t, obs_count_t, e, hx):
        """MODIFIED: added params gp_mean_t, gp_logvar_t, obs_count_t"""
        inp = torch.cat([y_t, a_t, vis_t, gp_mean_t, gp_logvar_t, obs_count_t], dim=-1).unsqueeze(1)
        out, hx_next = self.lstm(inp, hx)
        last = out[:, -1, :]
        feats = torch.cat([last, e], dim=-1)
        h = torch.relu(self.fc(feats))
        logits = self.logit_head(h)
        logvar = self.logvar_head(h)
        return logits, logvar, hx_next

    def forward_sequence(self, y_seq, a_seq, vis_seq, gp_mean_seq, gp_logvar_seq, obs_count_seq, e, lengths=None, hx=None):
        """MODIFIED: added GP sequence parameters"""
        inp = torch.cat([y_seq, a_seq, vis_seq, gp_mean_seq, gp_logvar_seq, obs_count_seq], dim=-1)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(inp, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, hx_out = self.lstm(packed, hx)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, hx_out = self.lstm(inp, hx)
        last = out[:, -1, :]
        feats = torch.cat([last, e], dim=-1)
        h = torch.relu(self.fc(feats))
        logits = self.logit_head(h)
        logvar = self.logvar_head(h)
        return logits, logvar, hx_out

    def logits_to_site(self, logits, clamp_var_floor=1e-3):
        """NEW METHOD: converts logits into Gaussian natural parameters"""
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
# ObserverStateStore: keeps hx and last message per cell
# ---------------------------
class ObserverStateStore:
    def __init__(self, model: ObserverLSTM, H, W, de, device='cpu'):
        self.model = model
        self.device = device
        self.H = H
        self.W = W
        self.de = de
        self.hiddens = {}
        self.embs = {}
        self.last_msg = {}
        self.obs_count = {}  # NEW: per-cell observation counter

    def init_cell(self, cell_idx, e_tensor):
        h0 = torch.zeros(self.model.layers, 1, self.model.hidden, device=self.device)
        c0 = torch.zeros(self.model.layers, 1, self.model.hidden, device=self.device)
        self.hiddens[cell_idx] = (h0, c0)
        self.embs[cell_idx] = e_tensor.to(self.device).reshape(1, -1)
        self.last_msg[cell_idx] = None
        self.obs_count[cell_idx] = 0  # NEW

    @torch.no_grad()
    def step(self, cell_idx, y_t, a_t, vis_t, gp_mean_t, gp_logvar_t):
        """MODIFIED: now receives gp_mean_t and gp_logvar_t"""
        hx = self.hiddens[cell_idx]
        e = self.embs[cell_idx]
        device = e.device

        y_t = torch.as_tensor(y_t, dtype=torch.float32, device=device).unsqueeze(0)
        a_t = torch.as_tensor(a_t, dtype=torch.float32, device=device).unsqueeze(0)
        vis_t = torch.as_tensor([[float(vis_t)]], dtype=torch.float32, device=device)
        gp_mean_t = gp_mean_t.reshape(1,1).to(device)
        gp_logvar_t = gp_logvar_t.reshape(1,1).to(device)
        obs_count = torch.tensor([[float(self.obs_count[cell_idx])]], dtype=torch.float32, device=device)

        logits, logvar, hx_next = self.model.step(y_t, a_t, vis_t, gp_mean_t, gp_logvar_t, obs_count, e, hx)
        q, mu, var, h_site, J_site = self.model.logits_to_site(logits)

        self.hiddens[cell_idx] = hx_next
        self.obs_count[cell_idx] += 1  # NEW: increment counter

        msg = {
            'logits': logits.detach(), 'q': q.detach(),
            'mu': mu.detach(), 'var': var.detach(), 'logvar': logvar.detach(),
            'h_site': h_site.detach(), 'J_site': J_site.detach()  # NEW
        }
        self.last_msg[cell_idx] = msg
        return msg  

# ---------------------------
# The modified GridMappingEnv
# ---------------------------
class GridMappingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n=5, max_steps=300, render_mode=None, ig_model=None, base_model=None,
                 dataset_path='./data/final_output.csv', strategy=None, device='cpu'):
        super(GridMappingEnv, self).__init__()
        self.n = n
        self.grid_size = n + 2
        self.ig_model = ig_model
        self.base_model = base_model  # previously LSTM pretrained; now we use ObserverLSTM for streaming
        self.dataset = pd.read_csv(dataset_path)
        self.device = device

        # state per cell (dictionary grid)
        self.state = np.array(
            [[{'pov': np.zeros(9, dtype=np.int32),
               'best_next_pov': -1,
               'id': None,
               'marker_pred': 0,
               'obs': np.zeros((9, 17), dtype=np.float32),
               'current_entropy': torch.tensor(0.0)}
              for _ in range(self.grid_size)]
             for _ in range(self.grid_size)]
        )

        # agent pos
        self.agent_pos = [1, 1]
        self.max_steps = max_steps
        self.current_steps = 0
        self.render_mode = render_mode

        # spaces
        self.action_space = spaces.Discrete(4)
        self._init_observation_space(extra_pov_radius=8)

        # integration: Observer & global fusion params
        # ordinal classes: assume N_CLASSES matches your previous base_model output (e.g., 8)
        self.N_CLASSES = 8
        # observer architecture
        self.cell_embedding_dim = 8

        self.dx = 1
        self.m0 = 0.0   # prior mean
        self.s0 = 10.0  # prior variance

        self.observer = ObserverLSTM(
            dy=17, dc=9, de=self.cell_embedding_dim, 
            n_classes=self.N_CLASSES,
            hidden=128, layers=1, 
            m0=self.m0, s0=self.s0  # ADDED prior params
        ).to(self.device)
        # store for per-cell hx and embeddings
        self.obs_store = ObserverStateStore(self.observer, H=self.grid_size, W=self.grid_size,
                                           de=self.cell_embedding_dim, device=self.device)

        Ncells = self.grid_size * self.grid_size
        D = Ncells * self.dx

        # build RBF precision (global prior)
        lengthscale = 2.0   # correlation lengthscale (tunable)
        variance = 1.0
        jitter = 1e-5
        self.Lambda0, self._coords = build_rbf_precision(
            self.grid_size, self.grid_size,
            lengthscale=lengthscale,
            variance=variance,
            jitter=jitter,
            device=self.device
        )

        # prior natural param
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

        # embeddings per cell (trainable embedding option: use nn.Embedding externally; here simple init)
        self.cell_embeddings = {}

        # strategy
        self.strategy = f"pred_{strategy}" if strategy is not None else "pred_none"

        # initialize ids and observers in reset()
        # rendering settings
        self.window_size = 600
        self.cell_size = self.window_size // self.grid_size
        self.window = None
        self.clock = None

    # -------------------------
    # env reset
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        # reset state fields
        self.state = np.array(
            [[{'pov': np.zeros(9, dtype=np.int32),
               'best_next_pov': -1,
               'id': None,
               'marker_pred': 0,
               'obs': np.zeros((9, 17), dtype=np.float32),
               'current_entropy': torch.tensor(0.0)}
              for _ in range(self.grid_size)]
             for _ in range(self.grid_size)]
        )
        self.agent_pos = [1, 1]
        self._assign_ids_to_cells()

        # reset fusion structures
        D = (self.grid_size * self.grid_size) * self.dx
        self.Lambda = self.Lambda0.clone()
        self.eta = self.eta0.clone()
        self.global_mean = torch.linalg.solve(self.Lambda + 1e-6 * torch.eye(D, device=self.device), self.eta)
        self._msg_cache = {}

        # init observers and embeddings per cell
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_idx = r * self.grid_size + c
                # create small random embedding; optionally replace with coordinates or precomputed visibility
                e = torch.randn(self.cell_embedding_dim, device=self.device) * 0.01
                self.cell_embeddings[cell_idx] = e
                self.obs_store.init_cell(cell_idx, e)

        # initialize pov choices for starting pos
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
        """COMPLETELY REWRITTEN to correctly handle scalars"""
        dx = self.dx
        sl = slice(cell_idx * dx, (cell_idx + 1) * dx)
        
        # remove old site
        old = self._msg_cache.get(cell_idx, None)
        if old is not None:
            old_h = old['h'].squeeze().item()  # MODIFIED: extract scalar
            old_J = old['J'].squeeze().item()  # MODIFIED: extract scalar
            self.Lambda[sl, sl] = self.Lambda[sl, sl] - old_J
            self.eta[sl] = self.eta[sl] - old_h
        
        # add new
        h_add = new_h.squeeze().item()  
        J_add = new_J.squeeze().item()
        
        self.Lambda[sl, sl] = self.Lambda[sl, sl] + J_add
        self.eta[sl] = self.eta[sl] + h_add
        
        # cache
        self._msg_cache[cell_idx] = {
            'h': new_h.clone().to(self.device), 
            'J': new_J.clone().to(self.device)
        }

        # solve
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

        # NEW: compute global covariance
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
        filtered_data = self.dataset[
            (self.dataset["IMAGE_ID"] == cell["id"]['IMAGE_ID']) &
            (self.dataset["BOX_COUNT"] == cell["id"]['BOX_COUNT']) &
            (self.dataset["MARKER_COUNT"] == cell["id"]['MARKER_COUNT'])
            ]

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
            
            # ensure coords
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
            
            # NEW BLOCK: prepare GP features
            sl = slice(cell_idx * self.dx, (cell_idx + 1) * self.dx)
            gp_mean_cell = self.global_mean[sl].reshape(1, 1)
            
            # approximate variance from diagonal precision
            approx_prec = self.Lambda[sl, sl].squeeze().item()
            gp_var_cell = 1.0 / max(approx_prec, 1e-9)
            gp_logvar_cell = torch.tensor(
                [[math.log(gp_var_cell)]], 
                dtype=torch.float32, 
                device=self.device
            )
            
            # MODIFIED: call with GP params
            msg = self.obs_store.step(
                cell_idx, y_t, a_t, vis_t=1,
                gp_mean_t=gp_mean_cell,
                gp_logvar_t=gp_logvar_cell
            )
            
            q = msg['q']
            mu = msg['mu']
            var = msg['var']
            
            current_entropy = -(q * (q + 1e-12).log()).sum().detach()
            cell['current_entropy'] = current_entropy
            
            pred_class = torch.argmax(q, dim=1).item() + 1
            if pred_class == cell['id']['MARKER_COUNT']:
                if update:
                    cell['marker_pred'] = 1
            
            prev_ent = cell.get('_last_entropy', torch.tensor(1.0))
            ig = (prev_ent - current_entropy).item()
            if ig < 0:
                ig = 0.0
            total_reward += ig
            cell['_last_entropy'] = current_entropy
            
            # MODIFIED: use msg['h_site'] and msg['J_site']
            h_site = msg['h_site']
            J_site = msg['J_site']
            self.replace_message_and_solve(cell_idx, h_site, J_site)
        
        return total_reward

    # -------------------------
    # best-view update (legacy behaviour kept)
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
        # Compute observed indices and generate input for model (same as before)
        observed_indices = np.flatnonzero(cell['pov'])
        input_list = []
        filtered_data = self.dataset[
            (self.dataset["IMAGE_ID"] == cell["id"]['IMAGE_ID']) &
            (self.dataset["BOX_COUNT"] == cell["id"]['BOX_COUNT']) &
            (self.dataset["MARKER_COUNT"] == cell["id"]['MARKER_COUNT'])
        ]
        for pov in observed_indices:
            row = filtered_data[filtered_data["POV_ID"] == pov + 1]
            if not row.empty:
                dist_prob = np.array([row[f"P{i}"] for i in range(8)]).flatten()
                pov_id_hot = np.zeros(9)
                pov_id_hot[pov] = 1
                input_list.append(np.concatenate((pov_id_hot, dist_prob)))

        input_array = np.array(input_list, dtype=np.float32)
        m = input_array.shape[0]
        if m > 0:
            for row in input_array:
                pov_onehot = row[:9]
                dist_prob = row[9:]
                y_t = row.copy()
                a_t = pov_onehot
                
                # ensure coords cached
                if '_coords' not in cell:
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
                
                sl = slice(cell_idx * self.dx, (cell_idx + 1) * self.dx)
                gp_mean_cell = self.global_mean[sl].reshape(1, 1)
                approx_prec = self.Lambda[sl, sl].squeeze().item()
                gp_var_cell = 1.0 / max(approx_prec, 1e-9)
                gp_logvar_cell = torch.tensor(
                    [[math.log(gp_var_cell)]], 
                    dtype=torch.float32, 
                    device=self.device
                )
                
                # CORRECT CALL:
                msg = self.obs_store.step(
                    cell_idx, y_t, a_t, vis_t=1,
                    gp_mean_t=gp_mean_cell,
                    gp_logvar_t=gp_logvar_cell
                )
                
                q = msg['q']
                current_entropy = -(q * (q + 1e-12).log()).sum().detach()
                cell['current_entropy'] = current_entropy
                pred_class = torch.argmax(q, dim=1).item() + 1
                if pred_class == cell['id']['MARKER_COUNT']:
                    cell['marker_pred'] = 1

        # Use observer streaming to produce updated belief if we want (consistent with _calculate_reward_ig)
        # Here we recompute messages for all observed views to ensure last_msg is consistent.
        # If training offline, you may skip this online recompute.
        if m > 0:
            for row in input_array:
                pov_onehot = row[:9]
                dist_prob = row[9:]
                y_t = row.copy()
                a_t = pov_onehot
                # ensure coords cached
                if '_coords' not in cell:
                    # attempt to find coords
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
                msg = self.obs_store.step(cell_idx, y_t, a_t, vis_t=1)
                q = msg['q']
                current_entropy = -(q * (q + 1e-12).log()).sum().detach()
                cell['current_entropy'] = current_entropy
                pred_class = torch.argmax(q, dim=1).item() + 1
                if pred_class == cell['id']['MARKER_COUNT']:
                    cell['marker_pred'] = 1

    # -------------------------
    # Observations retrieval (for agent)
    # -------------------------
    def _get_observation(self):
        obs = torch.zeros((3, 3, 18))
        ax, ay = self.agent_pos

        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = ax + i, ay + j
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    cell_obs = self.state[nx, ny]['obs']
                    curr_entropy = self.state[nx, ny]['current_entropy'].unsqueeze(0).detach()
                    cell_povs = torch.tensor(self.state[nx, ny]['pov'], dtype=torch.float32).unsqueeze(0).detach()
                    filtered_obs = cell_obs[~np.all(cell_obs == 0, axis=1)]
                    if filtered_obs.size > 0:
                        # use observer's last message q if present instead of calling base_model
                        # find cell index
                        if '_coords' not in self.state[nx, ny]:
                            self.state[nx, ny]['_coords'] = (nx, ny)
                        cell_idx = self._cell_index(nx, ny)
                        last_msg = self.obs_store.last_msg.get(cell_idx)
                        if last_msg is not None:
                            q = last_msg['q'].squeeze(0)
                        else:
                            # fallback: call base_model if provided (pretrained LSTM)
                            try:
                                marker_pre = self.base_model(torch.tensor(filtered_obs))
                                marker_pre_softmax = F.softmax(marker_pre, dim=1).mean(dim=0).detach()
                                q = marker_pre_softmax
                            except Exception:
                                q = torch.ones(self.N_CLASSES) / float(self.N_CLASSES)
                        obs[i + 1, j + 1] = torch.cat((curr_entropy, q, cell_povs), dim=1).squeeze(0)
        return obs.detach()

    def _get_observation_double_cnn(self, extra_pov_radius=8):
        obs_3x3 = torch.zeros((3, 3, 18))
        pov_size = len(self.state[0, 0]['pov'])  # Typically 9
        ax, ay = self.agent_pos

        grid_span = 2 * extra_pov_radius + 3
        pov_grid = torch.zeros((grid_span, grid_span, pov_size))

        # center 3x3
        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = ax + i, ay + j
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    cell_obs = self.state[nx, ny]['obs']
                    curr_entropy = self.state[nx, ny]['current_entropy'].unsqueeze(0).detach()
                    cell_povs = torch.tensor(self.state[nx, ny]['pov'], dtype=torch.float32).unsqueeze(0).detach()
                    filtered_obs = cell_obs[~np.all(cell_obs == 0, axis=1)]
                    if filtered_obs.size > 0:
                        # prefer observer's last message
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

                        curr_entropy_2d = curr_entropy.unsqueeze(0)  # From (1,) to (1, 1)
                        q_2d = q.unsqueeze(0)
                        curr_entropy_2d = curr_entropy_2d.to(self.device)
                        q_2d = q_2d.to(self.device)
                        cell_povs = cell_povs.to(self.device)
                        obs_3x3[i + 1, j + 1] = torch.cat((curr_entropy_2d, q_2d, cell_povs), dim=1).squeeze(0)

        # pov grid large neighborhood of pov occupancy (binary)
        for i in range(-extra_pov_radius - 1, extra_pov_radius + 2):
            for j in range(-extra_pov_radius - 1, extra_pov_radius + 2):
                gx, gy = i + extra_pov_radius + 1, j + extra_pov_radius + 1
                nx, ny = ax + i, ay + j
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    cell_povs = torch.tensor(self.state[nx, ny]['pov'], dtype=torch.float32).detach()
                    pov_grid[gx, gy] = cell_povs
                else:
                    pov_grid[gx, gy] = torch.zeros(pov_size)

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
# Training function (supervised)
# ----------------------------
def train_observer_on_env(env, observer_model, optimizer, device='cpu'):
    """Train observer using supervised learning on collected data"""
    observer_model.train()
    criterion = nn.CrossEntropyLoss()
    
    sequences = []
    targets = []
    embeddings = []
    lengths = []
    
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
            gp_mean_seq = torch.zeros((T, 1), dtype=torch.float32, device=device)
            gp_logvar_seq = torch.zeros((T, 1), dtype=torch.float32, device=device)
            obs_count_seq = torch.arange(0, T, dtype=torch.float32, device=device).unsqueeze(-1) / 9.0
            
            sequences.append((y_seq, a_seq, vis_seq, gp_mean_seq, gp_logvar_seq, obs_count_seq))
            targets.append(cell['id']['MARKER_COUNT'] - 1)
            
            cell_idx = env._cell_index(r, c)
            embeddings.append(env.cell_embeddings[cell_idx].unsqueeze(0))
            lengths.append(T)
    
    if len(sequences) == 0:
        return 0.0
    
    total_loss = 0.0
    for i, seq in enumerate(sequences):
        y_seq, a_seq, vis_seq, gp_mean_seq, gp_logvar_seq, obs_count_seq = seq
        
        y_b = y_seq.unsqueeze(0)
        a_b = a_seq.unsqueeze(0)
        vis_b = vis_seq.unsqueeze(0)
        gp_mean_b = gp_mean_seq.unsqueeze(0)
        gp_logvar_b = gp_logvar_seq.unsqueeze(0)
        obs_count_b = obs_count_seq.unsqueeze(0)
        e_b = embeddings[i].to(device)
        lengths_b = torch.tensor([lengths[i]], dtype=torch.long)
        
        logits, logvar, _ = observer_model.forward_sequence(
            y_b, a_b, vis_b, gp_mean_b, gp_logvar_b, obs_count_b, e_b, lengths=lengths_b
        )
        
        target = torch.tensor([targets[i]], dtype=torch.long, device=device)
        loss = criterion(logits, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(sequences)
