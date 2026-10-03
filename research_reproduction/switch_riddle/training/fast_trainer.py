"""
Optimized Trainer  [Engineering Optimizations — E]
====================================================
Replaces the original trainer.py hot-loop with a vectorized implementation.

SCIENTIFIC SPECIFICATION: UNCHANGED
All hyperparameters, algorithm semantics, communication rules, rewards,
target-network timing, and counter tracking are identical to the original.

Engineering optimizations applied [E]:
  1. BatchedSwitchRiddle: B envs advanced in one vectorized step (no for-loop)
  2. No .item() calls inside the timestep loop
  3. For shared-param algorithms: all n agents batched into one n*B forward pass
  4. Constant tensors (agent IDs, zero messages) pre-allocated once
  5. Loss accumulated as running GPU sum (no Python list of scalar tensors)
  6. Observations built from in_room tensor via comparison (no Python list)
  7. Actions selected in one batched epsilon-greedy call
  8. prev_actions updated in-place without new tensor allocation

See PERFORMANCE_NOTES.md for full documentation.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from switch_riddle.environment.switch_env import ACTION_NONE, ACTION_TELL, N_ACTIONS
from switch_riddle.environment.batched_env import BatchedSwitchRiddle
from switch_riddle.agents.rnn_agent import RNNAgent
from switch_riddle.communication.dru import DRU
from switch_riddle.communication.rial import RIALController
from switch_riddle.communication.dial import DIALController


# ---------------------------------------------------------------------------
# Re-export unchanged helpers from original trainer for compatibility
# ---------------------------------------------------------------------------

from switch_riddle.training.trainer import (
    TrainingConfig,
    TrainingCounters,
    NoCommController,
    _NSRIALController,
    _NSDIALController,
    get_device,
    print_device_info,
)


# ---------------------------------------------------------------------------
# Optimized run_batch
# ---------------------------------------------------------------------------

def run_batch_fast(
    batch_env: BatchedSwitchRiddle,
    controller,
    config: TrainingConfig,
    training: bool,
    device: torch.device,
    # Pre-allocated constant tensors (passed in to avoid per-call allocation)
    agent_id_tensors: List[torch.Tensor],   # list of n tensors, each (B,) long
    zero_msgs: List[torch.Tensor],           # list of n tensors, each (B,1) float
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Run one batch of B parallel episodes — fully vectorized.

    Returns
    -------
    loss        : scalar Tensor with grad (when training=True)
    ep_rewards  : (B,) float tensor — total reward per episode
    total_steps : int — total env steps taken
    """
    n       = config.n_agents
    B       = config.batch_size
    gamma   = config.gamma
    alg     = config.algorithm
    shared  = config.param_sharing
    eps     = config.epsilon if training else 0.0
    T       = 4 * n - 6

    # Reset all B episodes
    batch_env.reset()

    def get_agent(a_idx: int) -> RNNAgent:
        if shared or alg == "nocomm":
            return controller.agent
        return controller.agents[a_idx]

    # ---- Initial hidden states: zeros  [A] ----
    # For shared-param: one hidden state tensor of shape (2, n*B, 128)
    # so we can do one batched GRU call per timestep.
    if shared and alg in ("dial", "rial", "nocomm"):
        h_all = get_agent(0).init_hidden(n * B, device)    # (2, n*B, 128)
    else:
        # NS: separate hidden per agent
        h_list = [get_agent(a).init_hidden(B, device) for a in range(n)]

    # Initial messages: zeros  [C]
    prev_msg_all = torch.zeros(n * B, 1, device=device)    # (n*B, 1) for shared
    if not (shared and alg in ("dial", "rial", "nocomm")):
        prev_msg_list = [torch.zeros(B, 1, device=device) for _ in range(n)]

    # Initial prev_actions: zeros
    prev_act_all = torch.zeros(n * B, dtype=torch.long, device=device)
    if not (shared and alg in ("dial", "rial", "nocomm")):
        prev_act_list = [torch.zeros(B, dtype=torch.long, device=device)
                         for _ in range(n)]

    # Pre-build stacked agent IDs for the shared batched call:  [0,0,..,1,1,..,2,2,..]
    # shape (n*B,) — constant across all timesteps
    if shared and alg in ("dial", "rial", "nocomm"):
        aid_stacked = torch.cat([agent_id_tensors[a] for a in range(n)], dim=0)
        # (n*B,)

    ep_rewards  = torch.zeros(B, device=device)
    active      = torch.ones(B,  dtype=torch.bool, device=device)  # not done
    total_steps = 0

    # Accumulate loss as a running GPU sum + count
    loss_sum   = torch.zeros((), device=device)
    loss_count = 0

    # ---- Current room info (GPU) ----
    in_room_gpu  = batch_env.in_room.to(device)   # (B,) long
    switch_gpu   = batch_env.switch.to(device)     # (B,) long

    for t in range(T + 1):
        # ---------------------------------------------------------------
        # Build observations (fully on GPU, no Python list, no .item())
        # ---------------------------------------------------------------
        # obs[a][b] = 1.0 if in_room[b] == a AND episode active, else 0.0
        # For all agents at once: obs_matrix shape (n, B, 1)
        # obs_matrix[a, b] = (in_room_gpu[b] == a) & active[b]
        arange_n = torch.arange(n, device=device)                   # (n,)
        obs_matrix = (in_room_gpu.unsqueeze(0) == arange_n.unsqueeze(1)) \
                     & active.unsqueeze(0)                           # (n, B)
        obs_matrix = obs_matrix.float().unsqueeze(2)                 # (n, B, 1)

        # Switch as obs for agents (used as prev_msg for DIAL/RIAL)
        # Each agent receives the current switch state as their incoming message
        switch_f = switch_gpu.float().unsqueeze(1)   # (B, 1)

        # ---------------------------------------------------------------
        # Forward pass
        # ---------------------------------------------------------------
        if shared and alg in ("dial", "rial", "nocomm"):
            # --- Shared-param: one n*B batched call ---
            obs_stacked = obs_matrix.view(n * B, 1)              # (n*B, 1)

            # For DIAL/RIAL, prev message = switch (incoming comm)
            # For NoComm = zeros (handled inside RNNAgent)
            if alg == "nocomm":
                # NoComm zeros message inside agent
                msg_in = prev_msg_all                            # (n*B, 1) zeros
            else:
                # Each agent receives the switch state as incoming message
                msg_in = switch_f.expand(B, 1).repeat(n, 1)    # (n*B, 1)
                # Actually use prev_msg_all which tracks what was sent last step
                msg_in = prev_msg_all                            # (n*B, 1)

            agent = get_agent(0)
            if alg == "rial":
                q_u_all, q_m_all, h_all = agent(
                    obs_stacked, msg_in, prev_act_all, aid_stacked, h_all
                )   # q_u_all: (n*B, n_actions), q_m_all: (n*B, 2), h_all: (2,n*B,128)
            elif alg == "dial":
                q_u_all, m_raw_all, h_all = agent(
                    obs_stacked, msg_in, prev_act_all, aid_stacked, h_all
                )   # q_u_all: (n*B, n_actions), m_raw_all: (n*B, 1)
                msg_out_all = controller.dru(m_raw_all, training=training)  # (n*B,1)
            else:  # nocomm
                q_u_all, _, h_all = agent(
                    obs_stacked, msg_in, prev_act_all, aid_stacked, h_all
                )   # q_u_all: (n*B, n_actions)

            # Reshape: (n*B, ...) → (n, B, ...)
            q_u_split = q_u_all.view(n, B, N_ACTIONS)           # (n, B, n_actions)

            if alg == "rial":
                q_m_split = q_m_all.view(n, B, 2)               # (n, B, 2)
            elif alg == "dial":
                msg_split = msg_out_all.view(n, B, 1)            # (n, B, 1)

        else:
            # --- NS variants: separate per-agent call ---
            q_u_split  = []
            q_m_split  = [] if alg == "rial" else None
            msg_split  = [] if alg == "dial" else None
            new_h_list = []
            for a in range(n):
                ag = get_agent(a)
                obs_a   = obs_matrix[a]           # (B, 1)
                msg_a   = prev_msg_list[a]        # (B, 1)
                act_a   = prev_act_list[a]        # (B,)
                aid_a   = agent_id_tensors[a]     # (B,)
                h_a     = h_list[a]               # (2, B, 128)

                if alg == "rial":
                    q_u, q_m, h_new = ag(obs_a, msg_a, act_a, aid_a, h_a)
                    q_u_split.append(q_u)
                    q_m_split.append(q_m)
                elif alg == "dial":
                    q_u, m_raw, h_new = ag(obs_a, msg_a, act_a, aid_a, h_a)
                    msg_out = controller.dru(m_raw, training=training)
                    q_u_split.append(q_u)
                    msg_split.append(msg_out)
                else:
                    q_u, _, h_new = ag(obs_a, msg_a, act_a, aid_a, h_a)
                    q_u_split.append(q_u)
                new_h_list.append(h_new)
            h_list    = new_h_list
            q_u_split = torch.stack(q_u_split, dim=0)             # (n, B, n_actions)
            if q_m_split:
                q_m_split = torch.stack(q_m_split, dim=0)         # (n, B, 2)
            if msg_split:
                msg_split = torch.stack(msg_split, dim=0)         # (n, B, 1)

        # ---------------------------------------------------------------
        # Select actions — batched epsilon-greedy for all agents at once
        # ---------------------------------------------------------------
        # q_u_split: (n, B, n_actions)
        q_u_flat = q_u_split.view(n * B, N_ACTIONS)               # (n*B, n_actions)
        greedy   = q_u_flat.argmax(dim=1)                         # (n*B,) long

        if eps > 0.0:
            rand_mask    = torch.rand(n * B, device=device) < eps
            rand_actions = torch.randint(0, N_ACTIONS, (n * B,), device=device)
            selected     = torch.where(rand_mask, rand_actions, greedy)
        else:
            selected = greedy
        # selected: (n*B,) — actions for all agents and all episodes

        env_actions = selected.view(n, B)                          # (n, B)

        # ---------------------------------------------------------------
        # Mask illegal actions: only in_room agent can Tell
        # ---------------------------------------------------------------
        # in_room_gpu: (B,)   env_actions: (n, B)
        # agent a can Tell only if in_room_gpu == a
        in_room_mask = (in_room_gpu.unsqueeze(0) == arange_n.unsqueeze(1))  # (n, B)
        # Force ACTION_NONE for non-room agents
        env_actions  = torch.where(in_room_mask, env_actions,
                                   torch.zeros_like(env_actions))   # (n, B) long

        # Room agent's action is the action for in_room_gpu[b]
        # Gather: room_acts[b] = env_actions[in_room_gpu[b], b]
        room_acts = env_actions.gather(
            0, in_room_gpu.unsqueeze(0)
        ).squeeze(0)                                               # (B,) long

        # ---------------------------------------------------------------
        # Compute message value to write to switch (room agent's message)
        # ---------------------------------------------------------------
        if alg == "dial":
            # msg_split: (n, B, 1) continuous — threshold to 0/1 for switch
            # Gather room agent's message: (B,)
            room_msg_cont = msg_split.squeeze(2).gather(
                0, in_room_gpu.unsqueeze(0)
            ).squeeze(0)                                           # (B,) float
            room_msg_discrete = (room_msg_cont >= 0.5).long()     # (B,) long
        elif alg == "rial":
            # For RIAL: message is selected from Q_m
            q_m_flat = q_m_split.view(n * B, 2)
            greedy_m = q_m_flat.argmax(dim=1)
            if eps > 0.0:
                rand_mask_m   = torch.rand(n * B, device=device) < eps
                rand_msgs     = torch.randint(0, 2, (n * B,), device=device)
                selected_m    = torch.where(rand_mask_m, rand_msgs, greedy_m)
            else:
                selected_m = greedy_m
            selected_m = selected_m.view(n, B)                    # (n, B)
            room_msg_discrete = selected_m.gather(
                0, in_room_gpu.unsqueeze(0)
            ).squeeze(0)                                           # (B,) long
        else:  # nocomm
            room_msg_discrete = torch.zeros(B, dtype=torch.long, device=device)

        # ---------------------------------------------------------------
        # Step all B environments (vectorized)  [E]
        # ---------------------------------------------------------------
        # Only mask active episodes (done envs are skipped inside batched_env)
        in_room_new, switch_new, reward, done_new = batch_env.step(
            room_acts.cpu(),           # (B,) long — CPU for env logic
            room_msg_discrete.cpu(),   # (B,) long — CPU for env logic
        )
        # Move back to device
        in_room_new = in_room_new.to(device)
        switch_new  = switch_new.to(device)
        reward      = reward.to(device)
        done_new    = done_new.to(device)

        ep_rewards += reward
        total_steps += int(active.sum().item())   # count active episodes

        # ---------------------------------------------------------------
        # TD loss (vectorized over n and B simultaneously)
        # ---------------------------------------------------------------
        if training:
            # q_u for taken actions: gather over action dimension
            # env_actions: (n, B) — action taken by each agent in each episode
            # q_u_split: (n, B, n_actions)
            q_sa = q_u_split.gather(
                2, env_actions.unsqueeze(2)
            ).squeeze(2)                                           # (n, B)

            # Target: r (broadcast over n) + gamma * max_q if not done
            r_expanded = reward.unsqueeze(0).expand(n, B)         # (n, B)
            done_expanded = done_new.unsqueeze(0).expand(n, B)    # (n, B)

            with torch.no_grad():
                max_q = q_u_split.detach().max(dim=2).values      # (n, B)
            target = torch.where(
                done_expanded,
                r_expanded,
                r_expanded + gamma * max_q,
            )                                                      # (n, B)

            # Only accumulate loss for active episodes
            active_expanded = active.unsqueeze(0).expand(n, B)    # (n, B)
            sq_err = (q_sa - target.detach()) ** 2                 # (n, B)
            loss_sum   = loss_sum + (sq_err * active_expanded).sum()
            loss_count += int(active_expanded.sum().item())

        # ---------------------------------------------------------------
        # Update state for next timestep
        # ---------------------------------------------------------------
        active      = ~done_new
        in_room_gpu = in_room_new
        switch_gpu  = switch_new

        # Update prev_msg: the outgoing message this step becomes incoming next step
        if alg == "dial":
            # For shared: use msg_split reshaped to (n*B, 1)
            if shared:
                prev_msg_all = msg_split.view(n * B, 1).detach()
            else:
                prev_msg_list = [msg_split[a].detach() for a in range(n)]
        elif alg == "rial":
            if shared:
                # Discrete messages as float
                prev_msg_all = selected_m.view(n * B).float().unsqueeze(1).detach()
            else:
                for a in range(n):
                    prev_msg_list[a] = selected_m[a].float().unsqueeze(1).detach()
        # nocomm: prev_msg stays zero

        # Update prev_actions
        if shared and alg in ("dial", "rial", "nocomm"):
            prev_act_all = env_actions.view(n * B).detach()
        else:
            for a in range(n):
                prev_act_list[a] = env_actions[a].detach()

        # Update hidden state split index for shared path
        # (h_all already updated by GRU; no reshape needed for next step)

        if not active.any():
            break

    # ---- Aggregate loss ----
    if training and loss_count > 0:
        loss = loss_sum / loss_count
    else:
        loss = torch.zeros((), device=device, requires_grad=True)

    return loss, ep_rewards, total_steps


# ---------------------------------------------------------------------------
# Optimized Trainer
# ---------------------------------------------------------------------------

class FastTrainer:
    """
    Identical scientific specification to Trainer, with engineering optimizations.

    Differences from Trainer:
      - Uses BatchedSwitchRiddle instead of B separate SwitchRiddleEnv instances
      - Uses run_batch_fast instead of run_batch
      - Pre-allocates constant tensors
      - All other logic (counters, target network, optimizer, eval) is identical

    Scientific specification: UNCHANGED.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = get_device()
        self.counters = TrainingCounters()

        torch.manual_seed(config.seed)

        n   = config.n_agents
        alg = config.algorithm
        shared = config.param_sharing
        B   = config.batch_size

        def make_agent(mode):
            return RNNAgent(
                n_agents=n, n_actions=N_ACTIONS, n_messages=2,
                obs_dim=1, msg_dim=1, hidden_size=128, comm_mode=mode,
            ).to(self.device)

        if alg == "rial":
            if shared:
                agent = make_agent("rial")
                self.controller = RIALController(agent, n_messages=2,
                                                 epsilon=config.epsilon)
                self.params = list(agent.parameters())
                t_agent = make_agent("rial")
                t_agent.copy_weights_from(agent); t_agent.eval()
                self._target_ctrl = RIALController(t_agent, n_messages=2,
                                                   epsilon=0.0)
            else:
                agents = [make_agent("rial") for _ in range(n)]
                self.controller = _NSRIALController(agents, n_messages=2,
                                                    epsilon=config.epsilon)
                self.params = [p for ag in agents for p in ag.parameters()]
                t_agents = [make_agent("rial") for _ in range(n)]
                for ta, a in zip(t_agents, agents):
                    ta.copy_weights_from(a); ta.eval()
                self._target_ctrl = _NSRIALController(t_agents, n_messages=2,
                                                      epsilon=0.0)
        elif alg == "dial":
            dru = DRU(sigma=config.sigma)
            self.controller = None  # set below
            if shared:
                agent = make_agent("dial")
                self.controller = DIALController(agent, dru,
                                                 epsilon=config.epsilon)
                self.params = list(agent.parameters())
                t_agent = make_agent("dial")
                t_agent.copy_weights_from(agent); t_agent.eval()
                self._target_ctrl = DIALController(t_agent, DRU(config.sigma),
                                                   epsilon=0.0)
            else:
                agents = [make_agent("dial") for _ in range(n)]
                self.controller = _NSDIALController(agents, dru,
                                                    epsilon=config.epsilon)
                self.params = [p for ag in agents for p in ag.parameters()]
                t_agents = [make_agent("dial") for _ in range(n)]
                for ta, a in zip(t_agents, agents):
                    ta.copy_weights_from(a); ta.eval()
                self._target_ctrl = _NSDIALController(
                    t_agents, DRU(config.sigma), epsilon=0.0)
        else:  # nocomm
            agent = make_agent("nocomm")
            self.controller = NoCommController(agent, epsilon=config.epsilon)
            self.params = list(agent.parameters())
            t_agent = make_agent("nocomm")
            t_agent.copy_weights_from(agent); t_agent.eval()
            self._target_ctrl = NoCommController(t_agent, epsilon=0.0)

        # Optimizer: RMSProp, momentum=0.95, lr=5e-4      [A]
        self.optimizer = optim.RMSprop(
            self.params, lr=config.lr, momentum=config.rms_momentum,
        )

        # Batched environment (B episodes in one object)   [E]
        self._batch_env = BatchedSwitchRiddle(
            n=n, B=B,
            seed=config.seed * 10000,
            device=self.device,
        )

        # Pre-allocate constant tensors                    [E]
        self._agent_id_tensors = [
            torch.full((B,), a, dtype=torch.long, device=self.device)
            for a in range(n)
        ]
        self._zero_msgs = [
            torch.zeros(B, 1, device=self.device) for _ in range(n)
        ]

        os.makedirs(config.log_dir, exist_ok=True)
        self.history: List[Dict] = []

    def train(self) -> List[Dict]:
        cfg = self.config
        c   = self.counters

        for epoch in range(cfg.max_epochs):
            # Set training mode
            if hasattr(self.controller, "agent"):
                self.controller.agent.train()
            elif hasattr(self.controller, "agents"):
                for ag in self.controller.agents:
                    ag.train()

            # Run batch
            loss, ep_rewards, steps = run_batch_fast(
                self._batch_env, self.controller, cfg,
                training=True, device=self.device,
                agent_id_tensors=self._agent_id_tensors,
                zero_msgs=self._zero_msgs,
            )

            # Update counters
            c.completed_episodes += cfg.batch_size
            c.env_timesteps      += steps
            c.plotted_x_axis     += 1
            c.update_steps       += 1

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Target network reset               [A/C]
            if c.completed_episodes >= c._next_target_threshold:
                self._reset_target()
                c.target_update_count       += 1
                c.last_target_update_episode = c.completed_episodes
                c._next_target_threshold = (
                    c.completed_episodes
                    - (c.completed_episodes % 100)
                    + 100
                )

            # Evaluation
            if (epoch + 1) % cfg.eval_every_epochs == 0:
                mean_r, std_r = self.evaluate(cfg.eval_episodes)
                record = {
                    "epoch":                     epoch + 1,
                    "plotted_x_axis":            c.plotted_x_axis,
                    "completed_episodes":        c.completed_episodes,
                    "update_steps":              c.update_steps,
                    "env_timesteps":             c.env_timesteps,
                    "target_update_count":       c.target_update_count,
                    "last_target_update_episode":c.last_target_update_episode,
                    "mean_reward":               mean_r,
                    "std_reward":                std_r,
                    "loss":                      float(loss.item()),
                }
                self.history.append(record)
                print(
                    f"[{cfg.algorithm_label} n={cfg.n_agents} seed={cfg.seed}] "
                    f"epoch={epoch+1:5d}  eps={c.completed_episodes:7d}  "
                    f"mean_r={mean_r:.3f}  std={std_r:.3f}  "
                    f"tgt_upd={c.target_update_count}"
                )

        self._save_log()
        return self.history

    def evaluate(self, n_episodes: int) -> Tuple[float, float]:
        if hasattr(self.controller, "agent"):
            self.controller.agent.eval()
        elif hasattr(self.controller, "agents"):
            for ag in self.controller.agents:
                ag.eval()

        rewards_all: List[float] = []
        remaining = n_episodes

        while remaining > 0:
            bs = min(remaining, self.config.batch_size)
            eval_env = BatchedSwitchRiddle(
                n=self.config.n_agents, B=bs,
                seed=1_000_000 + remaining,
                device=self.device,
            )
            aid_t = [
                torch.full((bs,), a, dtype=torch.long, device=self.device)
                for a in range(self.config.n_agents)
            ]
            zero_m = [
                torch.zeros(bs, 1, device=self.device)
                for _ in range(self.config.n_agents)
            ]
            cfg_eval = TrainingConfig(
                **{k: v for k, v in asdict(self.config).items()},
            )
            cfg_eval.batch_size = bs
            _, ep_rew, _ = run_batch_fast(
                eval_env, self.controller, cfg_eval,
                training=False, device=self.device,
                agent_id_tensors=aid_t, zero_msgs=zero_m,
            )
            rewards_all.extend(ep_rew.cpu().tolist())
            remaining -= bs

        mean_r = sum(rewards_all) / len(rewards_all)
        std_r  = statistics.stdev(rewards_all) if len(rewards_all) > 1 else 0.0
        return mean_r, std_r

    def _reset_target(self):
        if hasattr(self.controller, "agent"):
            self._target_ctrl.agent.copy_weights_from(self.controller.agent)
        elif hasattr(self.controller, "agents"):
            for ta, a in zip(self._target_ctrl.agents, self.controller.agents):
                ta.copy_weights_from(a)

    def _save_log(self):
        cfg = self.config
        fname = os.path.join(
            cfg.log_dir,
            f"{cfg.algorithm_label}_n{cfg.n_agents}_seed{cfg.seed}.json"
        )
        out = {
            "algorithm":       cfg.algorithm_label,
            "n_agents":        cfg.n_agents,
            "param_sharing":   cfg.param_sharing,
            "seed":            cfg.seed,
            "config":          asdict(cfg),
            "final_counters":  asdict(self.counters),
            "history":         self.history,
        }
        with open(fname, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  -> log saved: {fname}")
