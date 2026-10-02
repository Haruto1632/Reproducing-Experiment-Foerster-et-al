# Implementation Plan — Foerster et al. 2016 Switch Riddle Reproduction

## Directory Layout

```
research_reproduction/
└── switch_riddle/
    ├── REPRODUCTION_PLAN.md        (approved spec)
    ├── REPRODUCTION_PLAN_REVIEW.md (fidelity review)
    ├── IMPLEMENTATION_PLAN.md      (this file)
    ├── README.md
    ├── environment/
    │   ├── __init__.py
    │   └── switch_env.py           # SwitchRiddle environment class
    ├── agents/
    │   ├── __init__.py
    │   └── rnn_agent.py            # Shared C-Net / DRQN backbone
    ├── communication/
    │   ├── __init__.py
    │   ├── dru.py                  # Discretise/Regularise Unit
    │   ├── dial.py                 # DIAL trainer wrapper
    │   └── rial.py                 # RIAL trainer wrapper
    ├── training/
    │   ├── __init__.py
    │   └── trainer.py              # Batch runner, optimiser, counter tracking
    ├── evaluation/
    │   ├── __init__.py
    │   └── evaluator.py            # Greedy eval with discrete messages
    ├── configs/
    │   └── experiment_params.json  # All paper-specified hyperparameters
    ├── tests/
    │   ├── __init__.py
    │   └── test_switch_env.py      # Phase 1 unit tests
    └── results/                    # JSON logs + PNG plots
```

---

## Phase-by-Phase Breakdown

### PHASE 1 — Environment  *(implement now)*

**File:** `environment/switch_env.py`

**Class:** `SwitchRiddleEnv`

| Item | Source |
|------|--------|
| `n` agents, `T = 4n-6` | [A] Paper Sec 6.2 |
| Prisoner selected uniformly at random with replacement | [A] Paper Sec 6.2 |
| Observation `o ∈ {0,1}` | [A] Paper Sec 6.2 |
| Actions: `{0=None, 1=Tell}` in room; `{0=None}` outside | [A] Paper Sec 6.2 |
| 1-bit switch message; available only to room occupant | [A] Paper Sec 6.2 |
| Reward: 0 / +1 / -1 on Tell | [A] Paper Sec 6.2 |
| Terminal on Tell or `t == T` | [A] Paper Sec 6.2 |
| `visited` set tracking which agents entered the room | [B] implied by reward |

**Tests:** `tests/test_switch_env.py`
1. `test_time_horizon` — T = 4n-6 for n=3,4
2. `test_exactly_one_in_room` — exactly one agent has obs=1 per step
3. `test_legal_actions_in_room` — Tell/None valid when in room
4. `test_illegal_actions_outside_room` — only None valid outside room
5. `test_reward_tell_correct` — +1 when all visited and Tell
6. `test_reward_tell_incorrect` — -1 when not all visited and Tell
7. `test_reward_no_tell` — 0 at non-terminal steps
8. `test_terminal_on_tell` — episode ends on Tell
9. `test_terminal_on_horizon` — episode ends at T
10. `test_message_availability` — message only visible to room occupant
11. `test_visited_tracking` — visited set updated correctly
12. `test_no_info_leak_nocomm` — zeroed message cannot convey visited state
13. `test_reset` — env resets cleanly, visited cleared, step=0
14. `test_deterministic_scenario_n3` — hand-crafted 3-agent scenario
15. `test_deterministic_scenario_n4` — hand-crafted 4-agent scenario

---

### PHASE 2 — Shared Architecture

**File:** `agents/rnn_agent.py`

**Class:** `RNNAgent(nn.Module)`

| Component | Spec | Source |
|-----------|------|--------|
| Agent index embedding | `nn.Embedding(n, 128)` | [A] |
| Prev action embedding | `nn.Embedding(n_actions, 128)` | [A] |
| Prev message MLP | `BatchNorm1d → Linear(1→128) → ReLU` | [A] / [C] |
| Task obs network | `Linear(1→128)` — assumption: 1 linear layer | [A spec, C arch] |
| State `z` | Element-wise sum of 4 embeddings | [A] |
| RNN | `nn.GRU(128, 128, num_layers=2)` | [A] |
| Output MLP | `Linear(128,128) → ReLU → Linear(128, n_actions + n_messages)` | [A] |
| `h_0 = 0` | Zeros init | [A] |
| `m_{-1} = 0` | Zeros init | [C] |

---

### PHASE 3 — RIAL

**File:** `communication/rial.py`

- Two heads: `Q_u` (action) and `Q_m` (message)
- Separate ε-greedy for `u` and `m` [A]
- Independent Q-learning (no shared gradient across agents) [A]
- No experience replay [A]
- Shared / RIAL-NS variants

---

### PHASE 4 — DIAL

**File:** `communication/dru.py` + `communication/dial.py`

- DRU: `Logistic(N(m, σ=2))` during training [A]
- DRU: `1{m > 0}` during eval [A]
- Gradient flows sender→receiver via continuous message [A]
- Shared / DIAL-NS variants

---

### PHASE 5 — NoComm

Shared-parameter network, message input hardcoded to 0, message output ignored. [C]

---

### PHASE 6 — Training

**File:** `training/trainer.py`

| Param | Value | Source |
|-------|-------|--------|
| Optimiser | RMSProp | [A] |
| Momentum | 0.95 | [A] |
| LR | 5e-4 | [A] |
| ε | 0.05 | [A] |
| γ | 1.0 | [A] |
| Batch | 32 parallel episodes | [A] |
| Target reset | every 100 completed episodes | [A] |
| σ (DIAL noise) | 2 | [A] |

**Counters tracked:**
- `completed_episodes`
- `update_steps`
- `env_timesteps`
- `plotted_x_axis`
- `target_update_count`
- `last_target_update_episode`

---

## Known Assumptions (C)

- `TaskMLP`: 1-layer `Linear(1→128)` — paper says "task-specific network" without specifying depth/type
- `BatchNorm`: `BatchNorm1d` on scalar `m_{t-1}` before message MLP
- `NoComm`: zero-message; no gradient through comm channel
- `m_{-1} = 0` at episode start

## Known Ambiguities (D)

- **Epoch vs Episode:** Paper text says "5k episodes"; Figure 4 x-axis says "# Epochs". We track both separately and DO NOT equate them.
- **Target reset at batch boundaries:** Paper says every 100 episodes; batches are 32. Update fires at first batch boundary after each 100-episode threshold.

## Dependencies

```
torch >= 1.13
numpy
matplotlib
pytest
```
