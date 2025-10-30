# custum_map_patched.py
# Versione patchata del tuo GridMappingEnv:
# - ObserverLSTM stateful per cella (streaming)
# - conversione logits -> gaussian message (mu,var -> natural params h,J)
# - ObserverStateStore per hidden states per cella
# - Global fusion (Lambda, eta) con Laplacian prior (dense, per ora)
# - Integrazione in update_cell e _update_cell_state

import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from gym import spaces

# ---------------------------
# Observer (stateful LSTM)
# ---------------------------
class ObserverLSTM(nn.Module):
    def __init__(self, dy=17, dc=9, de=8, n_classes=8, hidden=128, layers=1, diag_precision=True):
        """
        dy: feature dim per view (17)
        dc: pov one-hot dim (9)
        de: cell embedding dim
        n_classes: number of ordinal classes (1..N) -> logits output dim
        diag_precision: produce scalar variance for ordinal latent
        """
        super().__init__()
        self.dy = dy
        self.dc = dc
        self.de = de
        self.n_classes = n_classes
        self.hidden = hidden
        self.layers = layers
        self.in_dim = dy + dc + 1  # +1 for vis flag; cell embedding concatenated after LSTM
        self.lstm = nn.LSTM(self.in_dim, hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden + de, hidden)
        self.logit_head = nn.Linear(hidden, n_classes)
        self.diag_precision = diag_precision
        if diag_precision:
            # predict log-variance (scalar) for ordinal latent; clamp during usage
            self.logvar_head = nn.Linear(hidden, 1)
        else:
            # Not implemented full matrix precision in this patch
            raise NotImplementedError("Full-matrix precision not implemented in this patch")

    def forward_sequence(self, y_seq, a_seq, vis_seq, e, lengths=None, hx=None):
        """
        y_seq: [B,T,dy], a_seq: [B,T,dc], vis_seq: [B,T,1], e: [B,de]
        lengths: optional, for packing
        returns: logits [B,n_classes], logvar [B,1], hx_out
        """
        inp = torch.cat([y_seq, a_seq, vis_seq], dim=-1)  # [B,T,in_dim]
        if lengths is not None:
            # pack if lengths provided (not required if all same length)
            packed = torch.nn.utils.rnn.pack_padded_sequence(inp, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, hx_out = self.lstm(packed, hx)
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, hx_out = self.lstm(inp, hx)  # [B,T,H]
        last = out[:, -1, :]
        feats = torch.cat([last, e], dim=-1)
        h = torch.relu(self.fc(feats))
        logits = self.logit_head(h)
        logvar = self.logvar_head(h)
        return logits, logvar, hx_out

    def step(self, y_t, a_t, vis_t, e, hx):
        """
        single-step streaming update
        y_t: [B,dy], a_t: [B,dc], vis_t: [B,1], e: [B,de], hx: (h,c)
        return logits [B,n_classes], logvar [B,1], hx_next
        """
        inp = torch.cat([y_t, a_t, vis_t], dim=-1).unsqueeze(1)  # [B,1,in_dim]
        out, hx_next = self.lstm(inp, hx)
        last = out[:, -1, :]
        feats = torch.cat([last, e], dim=-1)
        h = torch.relu(self.fc(feats))
        logits = self.logit_head(h)
        logvar = self.logvar_head(h)
        return logits, logvar, hx_next

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
        # dictionaries keyed by cell_idx (r*W + c)
        self.hiddens = {}
        self.embs = {}
        self.last_msg = {}

    def init_cell(self, cell_idx, e_tensor):
        # initialize LSTM hidden/cell states for this cell
        h0 = torch.zeros(self.model.layers, 1, self.model.hidden, device=self.device)
        c0 = torch.zeros(self.model.layers, 1, self.model.hidden, device=self.device)
        self.hiddens[cell_idx] = (h0, c0)
        self.embs[cell_idx] = e_tensor.to(self.device).reshape(1, -1)
        self.last_msg[cell_idx] = None

    @torch.no_grad()
    def step(self, cell_idx, y_t, a_t, vis_t):
        """
        Process one new view for cell cell_idx.
        y_t: numpy array or tensor shape [dy]
        a_t: one-hot pov [dc]
        vis_t: 0/1 scalar
        Returns a dict: {'logits','q','mu','var','logvar'}
        """
        hx = self.hiddens[cell_idx]
        e = self.embs[cell_idx]  # [1,de]
        device = e.device

        y_t = torch.as_tensor(y_t, dtype=torch.float32, device=device).unsqueeze(0)  # [1,dy]
        a_t = torch.as_tensor(a_t, dtype=torch.float32, device=device).unsqueeze(0)  # [1,dc]
        vis_t = torch.as_tensor([[float(vis_t)]], dtype=torch.float32, device=device)  # [1,1]

        logits, logvar, hx_next = self.model.step(y_t, a_t, vis_t, e, hx)  # [1,N], [1,1]
        q = logits.log_softmax(-1).exp()  # [1,N]
        # convert categorical -> numeric mean & var on ordinal scale 1..N
        k = torch.arange(1, q.size(-1) + 1, dtype=torch.float32, device=device).unsqueeze(0)  # [1,N]
        mu = (q * k).sum(-1, keepdim=True)             # [1,1]
        var = (q * (k - mu)**2).sum(-1, keepdim=True)  # [1,1]
        var = var.clamp_min(1e-3)
        self.hiddens[cell_idx] = hx_next
        msg = {'logits': logits.detach(), 'q': q.detach(), 'mu': mu.detach(), 'var': var.detach(), 'logvar': logvar.detach()}
        self.last_msg[cell_idx] = msg
        return msg

# ---------------------------
# Helpers: categorical -> natural params (for Gaussian fusion)
# ---------------------------
def categorical_to_natural_params(mu, var, m0=0.0, s0=10.0):
    """
    mu: tensor [1,1], var: [1,1], m0,s0 scalars or tensors
    returns h: [1,1], J: [1,1,1] (we will use squeezed dims when fusing)
    """
    # ensure tensors
    if not torch.is_tensor(m0):
        m0 = torch.tensor(float(m0), dtype=mu.dtype, device=mu.device).reshape(1, 1)
    if not torch.is_tensor(s0):
        s0 = torch.tensor(float(s0), dtype=mu.dtype, device=mu.device).reshape(1, 1)
    s0_inv = 1.0 / s0
    s_inv = 1.0 / var
    # natural params: Lambda_post = S0^{-1} + S^{-1}, we provide site as difference
    J_site = (s_inv - s0_inv).reshape(1, 1, 1)  # [B,1,1]
    h_site = (s_inv * mu - s0_inv * m0).reshape(1, 1)  # [B,1]
    return h_site, J_site

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
        self.observer = ObserverLSTM(dy=17, dc=9, de=self.cell_embedding_dim, n_classes=self.N_CLASSES,
                                     hidden=128, layers=1, diag_precision=True).to(self.device)
        # store for per-cell hx and embeddings
        self.obs_store = ObserverStateStore(self.observer, H=self.grid_size, W=self.grid_size,
                                           de=self.cell_embedding_dim, device=self.device)

        # Gaussian fusion prior: dx = 1 (ordinal scalar)
        self.dx = 1
        self.m0 = 0.0   # reference prior mean (on ordinal numeric scale)
        self.s0 = 10.0  # reference prior variance (large = weak prior)
        Ncells = self.grid_size * self.grid_size
        D = Ncells * self.dx
        # Build dense Laplacian-like precision prior (small grid -> dense ok)
        self.Lambda0 = torch.zeros((D, D), dtype=torch.float32, device=self.device)
        lam = 0.5
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                i = r * self.grid_size + c
                sl = slice(i * self.dx, (i + 1) * self.dx)
                neighbors = []
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.grid_size and 0 <= cc < self.grid_size:
                        neighbors.append(rr * self.grid_size + cc)
                deg = len(neighbors)
                self.Lambda0[sl, sl] += deg * torch.eye(self.dx, device=self.device)
                for nb in neighbors:
                    jsl = slice(nb * self.dx, (nb + 1) * self.dx)
                    self.Lambda0[sl, jsl] += -1.0 * torch.eye(self.dx, device=self.device)
        # initial natural param
        m0_vec = torch.full((D,), float(self.m0), device=self.device)
        self.eta0 = self.Lambda0 @ m0_vec
        self.Lambda = self.Lambda0.clone()
        self.eta = self.eta0.clone()
        self.global_mean = torch.linalg.solve(self.Lambda + 1e-6 * torch.eye(D, device=self.device), self.eta)  # initial
        self._msg_cache = {}  # per-cell cached message for replace

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
        """
        new_h: [1,1], new_J: [1,1,1]
        Replace cached message for cell cell_idx (if any) and update Lambda, eta and global mean.
        """
        dx = self.dx
        sl = slice(cell_idx * dx, (cell_idx + 1) * dx)
        # subtract old
        old = self._msg_cache.get(cell_idx, None)
        if old is not None:
            old_h = old['h'].squeeze(0)  # [dx]
            old_J = old['J'].squeeze(0)  # [dx,dx]
            self.Lambda[sl, sl] = self.Lambda[sl, sl] - old_J
            self.eta[sl] = self.eta[sl] - old_h
        # add new
        h_add = new_h.squeeze(0).squeeze(-1).to(self.device)  # shape [dx]
        J_add = new_J.squeeze(0).squeeze(0).to(self.device)   # shape [dx,dx]
        self.Lambda[sl, sl] = self.Lambda[sl, sl] + J_add
        self.eta[sl] = self.eta[sl] + h_add
        # cache
        self._msg_cache[cell_idx] = {'h': new_h.clone().to(self.device), 'J': new_J.clone().to(self.device)}
        # numeric stability jitter
        jitter = 1e-6
        try:
            m_vec = torch.linalg.solve(self.Lambda + jitter * torch.eye(self.Lambda.size(0), device=self.device), self.eta)
        except RuntimeError:
            # fallback: add bigger jitter
            m_vec = torch.linalg.solve(self.Lambda + 1e-3 * torch.eye(self.Lambda.size(0), device=self.device), self.eta)
        self.global_mean = m_vec
        return m_vec

    # -------------------------
    # step_score (unchanged semantics)
    # -------------------------
    def step_score(self, action):
        prev_pos = list(self.agent_pos)
        temp_pos = list(self.agent_pos)

        if action == 0:  # su
            temp_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:  # destra
            temp_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 2:  # giù
            temp_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)
        elif action == 3:  # sinistra
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

        # Esegui l'azione
        self._move_agent(action)

        # Calcola il reward
        if self.strategy in ('pred_ig_reward', 'pred_no_train', 'pred_random_agent'):
            reward = self._update_pov_ig(self.agent_pos, prev_pos)
        else:
            new_pov_observed, best_next_pov_visited = self._update_pov_best_view(self.agent_pos)
            reward = self._calculate_reward_best_view(new_pov_observed, best_next_pov_visited, prev_pos)

        # Verifica condizioni di terminazione
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
        if cell['pov'][pov_index] == 1:
            return np.array([])

        cell_povs = cell['pov'].copy()
        cell_povs[pov_index] = 1
        if update:
            cell['pov'][pov_index] = 1

        observed_indices = np.flatnonzero(cell_povs)
        input_array = self._get_cell_input_array(cell, observed_indices)

        if update:
            m = input_array.shape[0]
            cell['obs'][:m, :] = input_array

        return input_array

    def _calculate_reward_ig(self, cell, input_array, update=True):
        # For each newly observed view (row) perform an observer step (streaming) and fuse
        total_reward = 0.0
        for row in input_array:
            pov_onehot = row[:9]       # first 9 entries
            dist_prob = row[9:]        # the 8 probs
            # choose the per-view y_t: we use the full 17-d row as y_t here
            y_t = row.copy().astype(np.float32)  # shape (17,)
            a_t = pov_onehot.astype(np.float32)
            # determine cell index in full grid indexing
            # find cell coordinates in state: this function is called inside loops where we know nx,ny
            # simpler: search for the exact cell index by scanning state arrays (costly but ok for small grids)
            # We'll require caller context; to avoid complexity, we assume caller sets a transient field on cell: cell['_coords']=(nx,ny)
            if '_coords' in cell and cell['_coords'] is not None:
                nx, ny = cell['_coords']
            else:
                # fallback (scan) - find first match
                found = False
                for rr in range(self.grid_size):
                    for cc in range(self.grid_size):
                        if self.state[rr, cc] is cell:
                            nx, ny = rr, cc
                            cell['_coords'] = (nx, ny)
                            found = True
                            break
                    if found:
                        break

            cell_idx = self._cell_index(nx, ny)
            msg = self.obs_store.step(cell_idx, y_t, a_t, vis_t=1)
            q = msg['q']    # [1,N]
            mu = msg['mu']  # [1,1]
            var = msg['var']  # [1,1]
            # entropy on categorical:
            current_entropy = -(q * (q + 1e-12).log()).sum().detach()
            cell['current_entropy'] = current_entropy
            # update marker_pred if confident
            pred_class = torch.argmax(q, dim=1).item() + 1  # classes 1..N
            if pred_class == cell['id']['MARKER_COUNT']:
                if update:
                    cell['marker_pred'] = 1
            # compute information gain reward: delta entropy (previous - new)
            # we have cell['current_entropy'] updated to new entropy, we need previous stored value
            # in this environment we stored previous in cell when last observed; use cell.get('_last_entropy')
            prev_ent = cell.get('_last_entropy', torch.tensor(1.0))  # default 1.0
            ig = (prev_ent - current_entropy).item()
            if ig < 0:
                ig = 0.0
            total_reward += ig
            cell['_last_entropy'] = current_entropy
            # convert to natural params and fuse
            h_site, J_site = categorical_to_natural_params(mu, var, m0=self.m0, s0=self.s0)
            # h_site: [1,1], J_site: [1,1,1] -> use replace_message_and_solve
            self.replace_message_and_solve(cell_idx, h_site.to(self.device), J_site.to(self.device))
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
                    # Aggiorna stato della cella con base_model (legacy)
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
            cell['obs'][:m, :] = input_array

        if len(observed_indices) != 9:
            if self.strategy in ('pred_random', 'pred_random_agent'):
                next_best_pov = torch.randint(0, 9, (1,)).item()
            else:
                # use ig_model or observer suggestions
                if self.ig_model is not None:
                    ig_prediction = self.ig_model(torch.tensor(input_array))[self.strategy]
                    next_best_pov = int(torch.argmin(ig_prediction).item())
                else:
                    next_best_pov = 0
            cell['best_next_pov'] = next_best_pov
        else:
            cell['best_next_pov'] = -1

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
                        obs_3x3[i + 1, j + 1] = torch.cat((curr_entropy, q, cell_povs), dim=1).squeeze(0)

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
