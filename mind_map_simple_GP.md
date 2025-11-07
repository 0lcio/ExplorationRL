# Gaussian Process-based Grid Mapping Environment 
This document summarizes the structure and usage of the environment defined in `new_custum_map_simple_GP.py`.

## Overview

`GridMappingEnv` is a Gym environment operating on a padded grid, generated based on gaussian distribution and. Each inner cell corresponds to an image sample with an 8-class latent marker (`MARKER_COUNT`). The agent moves on the grid, observes local **POVs** (9 viewpoints per cell), and receives rewards either for **entropy reduction** (IG strategy) or for visiting the **predicted best next POV** (heuristic strategy).

---

## Environment
  - `class GridMappingEnv(gym.Env)`
    - Core Gym API methods: `__init__()`, `reset()`, `step()`, `render()`, `close()`
    - Internal methods for movement, observation, reward calculation, termination checks, etc.
    - Strategy-specific methods for IG and best-view approaches.
    - `step()` consists of moving the agent, updating POVs, calculating rewards, checking termination, and returning observations:
      ```python
      obs, reward, terminated, truncated, info = env.step(action)
      ```

---
## Data Mapping
Based on Gaussian random field based grid, the images with corresponding box count and POVs are mapped to grid cells, to be observed the agent.
> The environment matches each grid cell to rows where `MARKER_COUNT` equals the latent class for that cell.
  - Generate Gaussian random field : `gaussian_random_field(n_cell=(20, 20), cluster_radius=3, binary=False) -> np.ndarray`
  - Discretize it into n-bins `create_binned_field(field, n_bins) -> (binned_field: np.ndarray[int], bin_edges)`
  - Match with images, CSV at `dataset_path` must contain (per row):
    - `IMAGE_ID`: unique identifier of the image/tile
    - `BOX_COUNT`: auxiliary integer (e.g., #boxes)
    - `MARKER_COUNT`: integer in `[0..9]`, used only `[1..8]` (target class)
    - `POV_ID`: integer in `[1..9]` (one of 9 view directions)
    - `P0..P7`: predicted class probabilities for the 8 classes (float in `[0,1]`)

---
## Environment

### Grid & State

- **Grid padding:** internal grid is `n × n`, environment grid is `(n+2) × (n+2)`.
- **Per-cell state dictionary:**

  - `pov`: `np.zeros(9, dtype=int)` — visited POV flags
  - `best_next_pov`: `int` in `{-1..8}` (−1 means all POVs complete)
  - `id`: `dict | None` with keys `{IMAGE_ID, BOX_COUNT, MARKER_COUNT}`
  - `marker_pred`: `0/1` (whether model predicts the correct class)
  - `obs`: `(9, 17)` float — per-POV features: 3by3 block -> 9 rows each with 17 features (`[ one_hot_pov(9) | P0..P7 (8) ]`)
  - `current_entropy`: scalar belief entropy (torch / float) initialized to uniform 8-class entropy [TODO: helper infogain check, doesn't store/reuse previous belief entropy, always computes from normal distribution]



### Actions

`spaces.Discrete(4)`:

- `0`: up, `1`: right, `2`: down, `3`: left  
  Movement is clamped within padded bounds. If after taking an action the agent’s position doesn’t change(either because it hit a boundary or repeated the same location intentionally), then it receives a negative reward (-2). Action is sampled via `env.action_space.sample()`.

### Observations (double-CNN encoding)

It combines two views of the environment:

1. Local 3×3 grid around the agent.
2. Extended POV grid with radius r.

Vector of length **3411**:

- Local **3×3** around agent (`obs_3x3`) → `3×3×18 = 162` features per cell:  
  `[ current_entropy (1) | softmax(base_model)(8) | pov_flags(9) ]`
- Extended **POV grid** with radius `r=8` → (+/-r + local 3) in x and y, 9POV for each `(2r+3)^2 * 9 = 19^2 * 9 = 3249`
- Concatenate : `162 + 3249 = 3411`

### Rewards

- **IG strategy:** sum of **entropy drop** `max(0, H_old − H_expected)` over newly observed cell-POVs; −2 if no movement.
- **Best-view strategy:** `+1` per new POV observed, `+8` if the new POV equals the **best_next_pov**; −2 if no movement.
- **Terminal bonus:** `+30` on successful termination.

### Termination

Episode ends early if:

1. All cells predicted correctly (`marker_pred==1` for every inner cell), **or**
2. Every _incorrect_ cell has been observed from **all 9** POVs.
   Truncates at `max_steps`.

---
## Strategy Notes

### IG-based (model-driven)

- Builds `(m, 17)` input per newly observed POV of each 3×3 neighborhood cell.
- `base_model(input_array)` → class probabilities; reward is the entropy reduction (prior entropy-posterior entropy after step).
- Updates `current_entropy` with posterior and sets `marker_pred=1` when argmax matches `MARKER_COUNT`.
- r=max{0, prior entropy-posterior entropy}, summed over all new POVs seen in the 3×3 window; then −2 if no movement.

### Heuristic “best-view”

- Marks observed POVs, computes `best_next_pov` per cell using `ig_model` or random policy.
- Reward favors visiting that best next POV soonest: r=1×new_pov_observed+8×best_next_pov_visited, then −2 if no movement. Here new_pov_observed counts new POVs on mispredicted cells only; best_next_pov_visited counts hits to the per-cell target POV.

---
## Class Reference

### Constructor

```python
GridMappingEnv(
    n: int = 5,
    max_steps: int = 300,
    render_mode: str | None = None,
    ig_model=None,
    base_model=None,
    dataset_path: str = "./data/final_output.csv",
    strategy: str | None = None,
    device: str = "cpu",
)
```


### Step
Here’s what env.step(action) does, in order:

1. Bookkeeping

```python
self.current_steps += 1
prev_pos = list(self.agent_pos).
```

2. Move the agent

Calls `_move_agent(action)` which clamps movement within the padded grid: `0: up, 1: right, 2: down, 3: left. `

3. Compute reward (by strategy)

- If strategy is one of `{'pred_ig_reward','pred_no_train','pred_random_agent'}` → IG mode:
  `reward = _update_pov_ig(self.agent_pos, prev_pos)`
  (loops over 3×3 neighborhood, registers new POVs, builds (m,17) inputs, sums entropy-drop rewards; −2 if didn’t move).
- Else → Best-view mode:
  `(new_pov_observed, best_next_pov_visited) = _update_pov_best_view(self.agent_pos)`
  `reward = _calculate_reward_best_view(..., prev_pos)` (reward = 1×new_pov_observed + 8×best_next_pov_visited, and −2 if didn’t move).

4. Check termination & truncation

`terminated = _check_termination()` (either all cells correct or every wrong cell seen from all 9 POVs).
`truncated = self.current_steps >= self.max_steps.`
If terminated, add a +30 terminal bonus to reward.

5. Return

`(observation, reward, terminated, truncated, info)` where
`observation = _get_observation_double_cnn()` (the 3×3 features + extended POV grid flattened).

### Gym API

```python
obs, info = env.reset(seed: int | None = None, options: dict | None = None)
obs, reward, terminated, truncated, info = env.step(action: int)
env.render(mode="human")
env.close()
```

### Key class functions

```python
# Latent field
_generate_latent_field() -> np.ndarray[int]  # shape (n, n), values in [1..8]
_assign_ids_to_cells() -> None               # binds dataset rows to cells

# Movement & termination
_move_agent(action: int) -> None
_check_termination() -> bool

# Strategies:

# 1. Ig_strategy functions:
_update_pov_ig(agent_pos, prev_pos, update=True) -> float
_calculate_reward_ig(cell, input_array, update=True) -> float

# 2. best view strategy functions:
_update_pov_best_view(agent_pos) -> tuple[int, bool]
_update_cell_state(cell) -> None
_calculate_reward_best_view(new_pov_observed: int, best_next_pov_visited: bool, prev_pos) -> float

# Cell updates
update_cell(cell: dict, i: int, j: int, update: bool) -> np.ndarray | int
_get_cell_input_array(cell: dict, observed_indices: list[int]) -> np.ndarray  # (m, 17)

# Observations
_get_observation_double_cnn(extra_pov_radius: int = 8) -> np.ndarray
_init_observation_space(extra_pov_radius: int = 8) -> None
```

---



## Quickstart

```python
from new_custum_map_simple_GP import GridMappingEnv

# Dummy models (replace with your trained models)
ig_model = ...
base_model = ...

env = GridMappingEnv(
    n=10,
    max_steps=500,
    ig_model=ig_model,
    base_model=base_model,
    dataset_path="./data/final_output.csv",
    strategy="pred_random",
    device="cpu",
)

obs, info = env.reset(seed=0)
done = False
total_reward = 0.0

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

env.close()
print("Episode reward:", total_reward)
```


## To review
Sorted based on the importance 
1. Index mismatch btw predicted class and MARKER_COUNT in dataset

    There was an index mismatch issue between the predicted class from the base model and the `MARKER_COUNT` used in _update_cell_state() and _calculate_reward_ig().

    - Based on generated field (with latent size 8 and adding +1 in bins in _generate_latent_field()), the range of field values is (1-8). Then this is used to collect samples from the dataset, specifically making MARKER_COUNT match with field value. Therefore, filtered MARKER_COUNT is also in range (1-8).  
    - In base model output size is 8 and using "torch.argmax(base_model_pred)" we get value in range (0-7). 
    - There was a mismatch between the range of values returned by argmax (0-7) and the MARKER_COUNT (1-8). This discrepancy could lead to incorrect updates of the cell state (best_view strategy) and reward calculations(ig strategy). The changes ensure that the comparison is valid by adjusting for this offset.

2. Softmax and IG calculation (helper functions)
  Raw predictions from base model are logits, so we need to convert them to probabilities using softmax before calculating entropy.

  - In information_gain() applies softmax and sends to entropy() where softmax is applied again. Therefore information gain calculation was incorrect.

  - Small note: in different places base_model output is used directly without softmax (e.g., in _update_cell_state() and _calculate_reward_ig()), which is correct for argmax comparison. but _get_observation_double_cnn() was using F.softmax and helper functions (entropy and information_gain) were applying torch.softmax. Both should be fine, but for consistency of conversion between torch and np arrays, and batch and single sample(e.g. dim=-1 and dim=1) handling, better to standardize it.

3. gym spaces definition for observation space bounds
  - In _init_observation_space() the observation space was defined with min and max values of 0 and 1, which is not accurate for the actual observation values. The current_entropy can be higher than 1 (up to log2(8)=3 for uniform distribution over 8 classes - 1/8). I don;t know how much gym rely on the defined observation space bounds, better to set more realistic bounds to avoid potential issues. 

4. not used functions
  - _get_observation() is defined but never used. The environment always uses _get_observation_double_cnn(). Consider removing unused functions to keep the code clean.
  - _update_neighbor_beliefs() is defined but never used. Consider removing it if not needed.

---
