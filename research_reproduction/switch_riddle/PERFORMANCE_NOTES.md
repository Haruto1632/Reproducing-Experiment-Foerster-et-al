# PERFORMANCE NOTES
## Switch Riddle Reproduction — Foerster et al. 2016

---

## Fidelity Legend

| Tag | Meaning |
|-----|---------|
| **[A]** | Directly specified by the paper |
| **[B]** | Strongly implied by the paper |
| **[C]** | Necessary implementation assumption (not specified) |
| **[E]** | Engineering optimization — does NOT change the scientific specification |

---

## 1. Original Implementation — Bottleneck Diagnosis

### Measured Performance (DIAL, n=3, seed=0, CUDA RTX 3050 Laptop)
- **Wall-clock time per 5000-epoch run:** ~100 minutes
- **GPU utilization:** 29–35%
- **GPU power:** 7 W / 80 W
- **GPU memory:** ~400 MiB / 4096 MiB
- **Total experiment runtime (5 alg × 5 seeds × n=3):** ~9–10 hours

### Root Causes

#### 1. Serial Python loop over 32 environments [E bottleneck]
```python
for b, env in enumerate(envs):   # ← 32 sequential Python calls
    result = env.step(acts_b)
```
The paper states "parallel episodes in batches of 32" **[A]**. The original
implementation ran 32 separate `SwitchRiddleEnv` Python objects sequentially
in a for-loop. This provided NO parallelism.

**Scientific note:** The paper does NOT specify that the environment must be
a tensor-vectorized GPU environment. Vectorization is an engineering choice **[E]**.
The transition semantics, reward, termination, and communication are unchanged.

#### 2. `.item()` GPU→CPU synchronization inside the timestep loop [E bottleneck]
```python
room_msg_val = float(new_msgs[in_room][b].item())   # GPU sync every step
raw = int(env_actions[a][b].item())                  # GPU sync per agent per step
```
Each `.item()` call forces a CUDA synchronization barrier. With B=32,
n=3, T=6, 5000 epochs: ≈2.88M synchronization barriers per seed.

#### 3. `torch.tensor(python_list, device=cuda)` inside loop [E bottleneck]
```python
obs_a = torch.tensor([...for b in range(B)], device=device)
```
Creates a new CUDA tensor from a Python list every timestep (90,000 times per seed).
Each call allocates host memory, fills it from a Python list, then copies to GPU.

#### 4. Scalar tensor list → `torch.stack()` for loss [E bottleneck]
```python
loss_terms.append((q_sa - target.detach()) ** 2)   # 576 individual tensors
loss = torch.stack(loss_terms).mean()               # 576 separate kernel launches
```
Should be a single vectorized loss over the entire batch.

#### 5. n separate GRU forward passes per timestep (shared-param case) [E bottleneck]
```python
for a in range(n):
    q_u, q_m, h_new = agent_net(obs_by_agent[a], ...)   # 3 separate calls
```
For shared parameters, all agents can be processed in one call with batch size n×B.

#### 6. Constant tensors reallocated every timestep [E waste]
```python
agent_ids = [torch.full((B,), a, device=device) for a in range(n)]  # every step
```
These are constants — should be pre-allocated once.

---

## 2. Optimized Implementation — Changes Made

### 2.1 Vectorized Batch Environment State **[E]**

Replaced 32 `SwitchRiddleEnv` Python objects stepped sequentially with a single
`BatchedSwitchRiddle` class that maintains all 32 episode states as integer tensors:

```python
class BatchedSwitchRiddle:
    # All state as CPU integer tensors — shape (B,)
    in_room:    torch.Tensor   # long, 0..n-1
    visited:    torch.Tensor   # long, bitmask per episode
    switch_state: torch.Tensor # long, 0 or 1
    step_count: torch.Tensor   # long
    done:       torch.Tensor   # bool
```

All 32 environments are stepped in a single vectorized Python call using
`torch.where`, boolean masking, and bitwise operations. No Python loop over
the batch dimension inside the timestep.

**Scientific invariants preserved:**
- Random prisoner selection with replacement **[A]**: `torch.randint` per
  active episode, identical distribution to `random.randrange`.
- Visited tracking **[B]**: bitmask per episode, semantically equivalent
  to the original `set()` per env.
- Reward: +1 for correct Tell, -1 for incorrect Tell, 0 for timeout **[A]**
- Termination: Tell or horizon T = 4n−6 **[A]**
- Switch state: 1-bit, written by room agent, readable by all **[A]**

**What is NOT changed:**
- Environment transition semantics
- Reward values
- Termination conditions
- Communication delivery rules
- Episode horizon T

### 2.2 Eliminated `.item()` from timestep loop **[E]**

All action masking, message routing, and reward accumulation now operate
on GPU tensors throughout the timestep. `.item()` is only called at
evaluation/logging time (outside the training loop).

### 2.3 Merged n-agent forward passes into single n×B call (shared param) **[E]**

For DIAL and RIAL (shared parameters), all n agents' observations are stacked
into a single tensor of shape `(n*B, obs_dim)` and processed in one GRU call:

```python
# Old: n separate calls of batch B
for a in range(n):
    q_u, q_m, h = agent(obs[a], ...)   # B items

# New: one call of batch n*B
q_u_all, q_m_all, h_all = agent(
    obs_stacked,          # (n*B, obs_dim)
    msg_stacked,          # (n*B, 1)
    act_stacked,          # (n*B,)
    aid_stacked,          # (n*B,)
    h_stacked,            # (2, n*B, 128)
)
# Then split: q_u_all.view(n, B, n_actions)
```

**Scientific note:** This is identical to n sequential calls for shared parameters
because the network weights are the same. The GRU hidden state is correctly
maintained per-agent-per-episode as before. This is purely a batching optimization.

For NS (non-shared) variants, agents still have separate networks and the merge
would require padding — kept as n separate calls to avoid complexity.

### 2.4 Pre-allocated constant tensors **[E]**

Agent ID tensors, zero-message tensors, and action-mask tensors are allocated
once before the epoch loop and reused:

```python
# Pre-allocate once
agent_id_tensors = [torch.full((B,), a, dtype=torch.long, device=device)
                    for a in range(n)]   # never reallocated
```

### 2.5 Vectorized loss accumulation **[E]**

Loss is accumulated as a running GPU sum tensor rather than a Python list
of scalar tensors:

```python
# Old: append 576 individual scalar tensors then torch.stack
loss_terms.append((q_sa - target) ** 2)

# New: accumulate into pre-allocated tensor, increment counter
loss_sum = loss_sum + batch_loss   # stays on GPU, no Python list
```

### 2.6 Observation construction from pre-computed GPU tensors **[E]**

Observations are read directly from the `in_room` tensor using scatter/comparison:
```python
# obs[a] = (in_room == a).float().unsqueeze(1)   — fully on GPU, no Python list
```

---

## 3. What Was NOT Changed

The following are all scientifically unchanged relative to the approved
REPRODUCTION_PLAN.md specification:

| Item | Status |
|------|--------|
| ε = 0.05 | Unchanged **[A]** |
| γ = 1.0 | Unchanged **[A]** |
| RMSProp, momentum=0.95, lr=5e-4 | Unchanged **[A]** |
| Batch size 32 parallel episodes | Unchanged **[A]** |
| Target network reset every 100 completed episodes | Unchanged **[A]** |
| DIAL DRU: σ=2, logistic+noise train, threshold eval | Unchanged **[A]** |
| RIAL: separate ε-greedy for Q_u and Q_m | Unchanged **[A]** |
| GRU architecture: 2-layer, 128 hidden | Unchanged **[A]** |
| 4 embedding streams summed element-wise | Unchanged **[A]** |
| Reward: +1 correct Tell, −1 incorrect Tell, 0 timeout | Unchanged **[A]** |
| T = 4n − 6 | Unchanged **[A]** |
| Random prisoner selection with replacement | Unchanged **[A]** |
| Communication: 1-bit switch as shared channel | Unchanged **[A]** |
| Random seeds | Unchanged **[C]** |
| NoComm: message stream zeroed | Unchanged **[C]** |
| Initial hidden state h_0 = 0 | Unchanged **[A]** |
| Initial message m_{-1} = 0 | Unchanged **[C]** |
| Counter tracking (episodes, epochs, timesteps) | Unchanged |

---

## 4. Equivalence Test Results

See `tests/test_equivalence.py` for the formal equivalence tests.

Summary:
- Environment mechanics: ✅ identical transition semantics
- Rewards: ✅ identical for same action sequence and same RNG seed
- Termination: ✅ identical
- Target update thresholds: ✅ fires at same completed-episode counts
- Communication delivery: ✅ identical switch-state routing

---

## 5. Benchmark Results

Measured on: RTX 3050 Laptop (4 GB), CUDA 12.4, PyTorch 2.6.0+cu124, n=3, NoComm, B=32, 50 epochs.

| Metric | Original | Optimized | Change |
|--------|----------|-----------|--------|
| Wall-clock time (50 epochs) | 18.25 s | 2.74 s | **6.7× faster** |
| Episodes / second | 88 | 583 | **+562%** |
| Projected full-run time (5 alg × 5 seeds × 5000 epochs) | ~9–10 hours | ~1.3–1.5 hours | — |
| GPU utilization | 29–35% | TBD (full run) | — |
| GPU power | ~7 W | TBD (full run) | — |
| Peak VRAM | ~400 MiB | TBD (full run) | — |

The 6.7× speedup is achieved entirely through engineering changes [E].
No scientific parameter was modified.

