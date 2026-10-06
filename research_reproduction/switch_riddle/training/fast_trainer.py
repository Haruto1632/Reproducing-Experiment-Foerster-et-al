"""
Fast Trainer — Faithful Implementation of Foerster et al. 2016 Algorithm 1
==========================================================================
Implements the Switch Riddle training loop faithful to Foerster et al. 2016:

  - Interrogation Room Communication:
    Only the agent in the room reads the switch (message left by previous occupant).
    Only the agent in the room writes to the switch.
    Non-room agents receive zero / blank message.
  - Action Masking:
    Only the agent in the room can choose 'Tell' (or 'None').
    Non-room agents can only choose 'None'.
  - Algorithm 1 TD Bootstrapping:
    Full episode rollout collected forward.
    Targets computed: y_t = r_t if done else r_t + gamma * next_val.
    next_val evaluated using target network theta^- with legal action masking:
      - in-room agent at t+1: max(Q(None), Q(Tell))
      - out-of-room agent at t+1: Q(None)
  - DIAL:
    Real-valued continuous message passed during training through DRU:
      Logistic(N(m, sigma=2)).
    Gradients backpropagate end-to-end through message channel.
    Binary threshold 1{m > 0} during evaluation.
  - RIAL:
    Independent Q-learning on discrete actions (Q_u) and messages (Q_m).
    Separate epsilon-greedy selection for u and m.
  - NoComm:
    Messages zeroed; only Q_u trained.
  - Parameter Sharing and Non-Sharing (-NS) variants supported.
  - Target Network Reset:
    Hard reset every 100 completed episodes [A].
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from switch_riddle.environment.switch_env import SwitchRiddleEnv, ACTION_NONE, ACTION_TELL, N_ACTIONS
from switch_riddle.environment.batched_env import BatchedSwitchRiddle
from switch_riddle.agents.rnn_agent import RNNAgent
from switch_riddle.communication.dru import DRU
from switch_riddle.communication.rial import RIALController
from switch_riddle.communication.dial import DIALController

# Re-export unchanged dataclasses and helpers
from switch_riddle.training.trainer import (
    TrainingConfig,
    TrainingCounters,
    NoCommController,
    _NSRIALController,
    _NSDIALController,
    get_device,
    print_device_info,
)


def _rial_message_td_loss(
    q_m: torch.Tensor,
    selected_messages: torch.Tensor,
    target_q_m_next: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    in_room: torch.Tensor,
    in_room_next: torch.Tensor,
    active: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Sum RIAL communication TD errors over legal room-agent actions."""
    n = q_m.shape[0]
    agent_indices = torch.arange(n, device=q_m.device).unsqueeze(1)
    current_mask = active.unsqueeze(0) & (in_room.unsqueeze(0) == agent_indices)
    next_mask = in_room_next.unsqueeze(0) == agent_indices
    q_m_taken = q_m.gather(2, selected_messages.unsqueeze(2)).squeeze(2)
    next_value = torch.where(
        next_mask,
        target_q_m_next.max(dim=2).values,
        torch.zeros_like(reward).unsqueeze(0).expand(n, -1),
    )
    target = torch.where(
        done.unsqueeze(0),
        reward.unsqueeze(0).expand(n, -1),
        reward.unsqueeze(0) + gamma * next_value,
    )
    return (((q_m_taken - target) ** 2) * current_mask).sum()


def run_batch_fast(
    batch_env: BatchedSwitchRiddle,
    controller,
    config: TrainingConfig,
    training: bool,
    device: torch.device,
    agent_id_tensors: List[torch.Tensor],
    zero_msgs: List[torch.Tensor],
    target_controller=None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Run one batch of B parallel episodes faithful to Algorithm 1.

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

    batch_env.reset()

    def get_agent(a_idx: int) -> RNNAgent:
        if shared or alg == "nocomm":
            return controller.agent
        return controller.agents[a_idx]

    def get_target_agent(a_idx: int) -> Optional[RNNAgent]:
        if target_controller is None:
            return None
        if shared or alg == "nocomm":
            return target_controller.agent
        return target_controller.agents[a_idx]

    arange_n = torch.arange(n, device=device)
    aid_stacked = torch.cat([agent_id_tensors[a] for a in range(n)], dim=0)

    # Initial recurrent hidden states: zeros [A]
    if shared and alg in ("dial", "rial", "nocomm"):
        h_all = get_agent(0).init_hidden(n * B, device)
        target_h_all = (
            get_target_agent(0).init_hidden(n * B, device)
            if target_controller is not None else None
        )
    else:
        h_list = [get_agent(a).init_hidden(B, device) for a in range(n)]
        target_h_list = (
            [get_target_agent(a).init_hidden(B, device) for a in range(n)]
            if target_controller is not None else None
        )

    # Initial prev_actions: zeros [A]
    if shared and alg in ("dial", "rial", "nocomm"):
        prev_act_all = torch.zeros(n * B, dtype=torch.long, device=device)
    else:
        prev_act_list = [torch.zeros(B, dtype=torch.long, device=device) for _ in range(n)]

    ep_rewards = torch.zeros(B, device=device)
    active = torch.ones(B, dtype=torch.bool, device=device)
    total_steps = 0

    # Room switch message: initial switch state is 0.0 [C]
    room_switch_msg = torch.zeros(B, 1, device=device)
    # DIAL's target policy has its own recurrently generated communication
    # stream, as in the released implementation. RIAL target inputs use the
    # messages actually emitted by the online policy.
    target_room_switch_msg = torch.zeros(B, 1, device=device)
    in_room_gpu = batch_env.in_room.to(device)

    rollout = []

    # -----------------------------------------------------------------------
    # Forward Pass Unroll
    # -----------------------------------------------------------------------
    for step in range(T + 1):
        target_h_all_next = None
        target_h_list_next = None
        target_room_msg_out = None

        # Observation matrix: (n, B, 1) — 1.0 if in_room and active, else 0.0
        in_room_mask = (in_room_gpu.unsqueeze(0) == arange_n.unsqueeze(1))
        obs_matrix = (in_room_mask & active.unsqueeze(0)).float().unsqueeze(2)

        # Message input routing:
        # Agent in room reads switch state. Agents outside room read 0.0.
        if alg == "nocomm":
            msg_matrix = torch.zeros(n, B, 1, device=device)
        else:
            msg_matrix = torch.where(
                in_room_mask.unsqueeze(2),
                room_switch_msg.unsqueeze(0).expand(n, B, 1),
                torch.zeros(n, B, 1, device=device)
            )
        if alg == "dial":
            target_msg_matrix = torch.where(
                in_room_mask.unsqueeze(2),
                target_room_switch_msg.unsqueeze(0).expand(n, B, 1),
                torch.zeros(n, B, 1, device=device),
            )
        else:
            target_msg_matrix = msg_matrix

        if shared and alg in ("dial", "rial", "nocomm"):
            obs_stacked = obs_matrix.view(n * B, 1)
            msg_stacked = msg_matrix.view(n * B, 1)
            agent = get_agent(0)

            if alg == "rial":
                q_u_all, q_m_all, h_all_next = agent(obs_stacked, msg_stacked, prev_act_all, aid_stacked, h_all)
                q_u_split = q_u_all.view(n, B, N_ACTIONS)
                q_m_split = q_m_all.view(n, B, 2)
                # Epsilon-greedy for discrete messages
                q_m_flat = q_m_split.view(n * B, 2)
                greedy_m = q_m_flat.argmax(dim=1)
                if eps > 0.0:
                    rand_m_mask = torch.rand(n * B, device=device) < eps
                    rand_msgs = torch.randint(0, 2, (n * B,), device=device)
                    sel_msgs = torch.where(rand_m_mask, rand_msgs, greedy_m).view(n, B)
                else:
                    sel_msgs = greedy_m.view(n, B)
                room_msg_disc = sel_msgs.gather(0, in_room_gpu.unsqueeze(0)).squeeze(0)
                room_msg_out = room_msg_disc.unsqueeze(1).float()
            elif alg == "dial":
                q_u_all, m_raw_all, h_all_next = agent(obs_stacked, msg_stacked, prev_act_all, aid_stacked, h_all)
                q_u_split = q_u_all.view(n, B, N_ACTIONS)
                m_dru_all = controller.dru(m_raw_all, training=training)
                m_dru_split = m_dru_all.view(n, B, 1)
                # Outgoing continuous message written to switch
                room_msg_out = m_dru_split.gather(0, in_room_gpu.unsqueeze(0).unsqueeze(2)).squeeze(0)
                room_msg_disc = (room_msg_out.detach() >= 0.5).long().squeeze(1)
                q_m_split = None
                sel_msgs = None

            else:  # nocomm
                q_u_all, _, h_all_next = agent(obs_stacked, msg_stacked, prev_act_all, aid_stacked, h_all)
                q_u_split = q_u_all.view(n, B, N_ACTIONS)
                room_msg_disc = torch.zeros(B, dtype=torch.long, device=device)
                room_msg_out = torch.zeros(B, 1, device=device)
                q_m_split = None
                sel_msgs = None

            if target_controller is not None:
                target_agent = get_target_agent(0)
                with torch.no_grad():
                    _, target_msg_all, target_h_all_next = target_agent(
                        obs_stacked, target_msg_matrix.view(n * B, 1),
                        prev_act_all, aid_stacked,
                        target_h_all,
                    )
                    if alg == "dial":
                        target_msg_all = target_controller.dru(
                            target_msg_all, training=training
                        )
                        target_msg_split = target_msg_all.view(n, B, 1)
                        target_room_msg_out = target_msg_split.gather(
                            0, in_room_gpu.unsqueeze(0).unsqueeze(2)
                        ).squeeze(0)

        else:
            # Non-shared (-NS) variants
            q_u_split = []
            q_m_split = [] if alg == "rial" else None
            m_dru_split = [] if alg == "dial" else None
            h_list_next = []
            target_h_list_next = [] if target_controller is not None else None

            for a in range(n):
                ag = get_agent(a)
                obs_a = obs_matrix[a]
                msg_a = msg_matrix[a]
                act_a = prev_act_list[a]
                aid_a = agent_id_tensors[a]
                h_a = h_list[a]

                if alg == "rial":
                    q_u, q_m, h_new = ag(obs_a, msg_a, act_a, aid_a, h_a)
                    q_u_split.append(q_u)
                    q_m_split.append(q_m)
                elif alg == "dial":
                    q_u, m_raw, h_new = ag(obs_a, msg_a, act_a, aid_a, h_a)
                    m_dru = controller.dru(m_raw, training=training)
                    q_u_split.append(q_u)
                    m_dru_split.append(m_dru)
                else:
                    q_u, _, h_new = ag(obs_a, msg_a, act_a, aid_a, h_a)
                    q_u_split.append(q_u)
                h_list_next.append(h_new)

                if target_controller is not None:
                    target_agent = get_target_agent(a)
                    with torch.no_grad():
                        _, target_msg, target_h_new = target_agent(
                            obs_a, target_msg_matrix[a], act_a, aid_a,
                            target_h_list[a],
                        )
                        if alg == "dial":
                            target_msg = target_controller.dru(
                                target_msg, training=training
                            )
                            if target_room_msg_out is None:
                                target_room_msg_out = torch.zeros(
                                    B, 1, device=device
                                )
                            target_room_msg_out = torch.where(
                                (in_room_gpu == a).unsqueeze(1),
                                target_msg,
                                target_room_msg_out,
                            )
                    target_h_list_next.append(target_h_new)


            q_u_split = torch.stack(q_u_split, dim=0)

            if alg == "rial":
                q_m_split = torch.stack(q_m_split, dim=0)
                q_m_flat = q_m_split.view(n * B, 2)
                greedy_m = q_m_flat.argmax(dim=1)
                if eps > 0.0:
                    rand_m_mask = torch.rand(n * B, device=device) < eps
                    rand_msgs = torch.randint(0, 2, (n * B,), device=device)
                    sel_msgs = torch.where(rand_m_mask, rand_msgs, greedy_m).view(n, B)
                else:
                    sel_msgs = greedy_m.view(n, B)
                room_msg_disc = sel_msgs.gather(0, in_room_gpu.unsqueeze(0)).squeeze(0)
                room_msg_out = room_msg_disc.unsqueeze(1).float()
            elif alg == "dial":
                m_dru_split = torch.stack(m_dru_split, dim=0)
                room_msg_out = m_dru_split.gather(0, in_room_gpu.unsqueeze(0).unsqueeze(2)).squeeze(0)
                room_msg_disc = (room_msg_out.detach() >= 0.5).long().squeeze(1)
                q_m_split = None
                sel_msgs = None
            else:
                room_msg_disc = torch.zeros(B, dtype=torch.long, device=device)
                room_msg_out = torch.zeros(B, 1, device=device)
                q_m_split = None
                sel_msgs = None

        # Action selection: epsilon-greedy on Q_u
        q_flat = q_u_split.view(n * B, N_ACTIONS)
        greedy = q_flat.argmax(dim=1)
        if eps > 0.0:
            rand_mask = torch.rand(n * B, device=device) < eps
            rand_acts = torch.randint(0, N_ACTIONS, (n * B,), device=device)
            sel_acts = torch.where(rand_mask, rand_acts, greedy).view(n, B)
        else:
            sel_acts = greedy.view(n, B)

        # Mask non-room agents: only agent in room can Tell
        env_acts = torch.where(in_room_mask, sel_acts, torch.zeros_like(sel_acts))
        room_acts = env_acts.gather(0, in_room_gpu.unsqueeze(0)).squeeze(0)

        # Advance environment
        in_room_next, _, reward, done_next = batch_env.step(room_acts.cpu(), room_msg_disc.cpu())
        in_room_next = in_room_next.to(device)
        reward = reward.to(device)
        done_next = done_next.to(device)

        ep_rewards += reward
        total_steps += int(active.sum().item())

        rollout.append({
            "q_u_split": q_u_split,
            "q_m_split": q_m_split,
            "sel_msgs": sel_msgs,
            "env_acts": env_acts,
            "reward": reward,
            "done_next": done_next,
            "active": active,
            "in_room": in_room_gpu,
            "in_room_next": in_room_next,
            "room_msg_out": room_msg_out,
            "target_room_msg_out": target_room_msg_out,
            "h_next": h_all_next if (shared and alg in ("dial", "rial", "nocomm")) else h_list_next,
            "target_h_next": (
                target_h_all_next if (shared and alg in ("dial", "rial", "nocomm"))
                else target_h_list_next
            ),
        })

        active = ~done_next
        in_room_gpu = in_room_next
        room_switch_msg = room_msg_out
        if alg == "dial" and target_room_msg_out is not None:
            target_room_switch_msg = target_room_msg_out

        if shared and alg in ("dial", "rial", "nocomm"):
            prev_act_all = env_acts.view(n * B).detach()
            h_all = h_all_next
            if target_controller is not None:
                target_h_all = target_h_all_next
        else:
            prev_act_list = [env_acts[a].detach() for a in range(n)]
            h_list = h_list_next
            if target_controller is not None:
                target_h_list = target_h_list_next

        if not active.any():
            break

    # -----------------------------------------------------------------------
    # Backward Pass (Algorithm 1)
    # -----------------------------------------------------------------------
    if training and target_controller is not None and len(rollout) > 0:
        loss = torch.tensor(0.0, device=device)
        total_transitions = 0

        for t_idx in range(len(rollout)):
            step_data = rollout[t_idx]
            q_u = step_data["q_u_split"]
            acts = step_data["env_acts"]
            r = step_data["reward"].unsqueeze(0).expand(n, B)
            d = step_data["done_next"].unsqueeze(0).expand(n, B)
            act_mask = step_data["active"].unsqueeze(0).expand(n, B)

            # Q(s_t, a_t)
            q_sa = q_u.gather(2, acts.unsqueeze(2)).squeeze(2)

            # Target computation: Algorithm 1 line 142
            with torch.no_grad():
                in_r_nxt = step_data["in_room_next"]
                d_nxt = step_data["done_next"]
                in_r_nxt_m = (in_r_nxt.unsqueeze(0) == arange_n.unsqueeze(1))
                obs_nxt_m = in_r_nxt_m & (~d_nxt).unsqueeze(0)

                if alg == "nocomm":
                    msg_nxt_m = torch.zeros(n, B, 1, device=device)
                else:
                    message_for_target = step_data["room_msg_out"]
                    if alg == "dial":
                        message_for_target = step_data["target_room_msg_out"]
                    if message_for_target is None:
                        message_for_target = target_room_switch_msg
                    msg_nxt_m = torch.where(
                        in_r_nxt_m.unsqueeze(2),
                        message_for_target.detach().unsqueeze(0).expand(n, B, 1),
                        torch.zeros(n, B, 1, device=device)
                    )

                if shared and alg in ("dial", "rial", "nocomm"):
                    obs_nxt_stk = obs_nxt_m.float().unsqueeze(2).view(n * B, 1)
                    msg_nxt_stk = msg_nxt_m.view(n * B, 1)
                    t_agent = get_target_agent(0)
                    q_tgt_all, q_m_tgt_all, _ = t_agent(
                        obs_nxt_stk,
                        msg_nxt_stk,
                        acts.view(n * B),
                        aid_stacked,
                        step_data["target_h_next"]
                    )
                    q_tgt_sp = q_tgt_all.view(n, B, N_ACTIONS)
                    q_m_tgt_sp = (
                        q_m_tgt_all.view(n, B, 2) if alg == "rial" else None
                    )
                else:
                    q_tgt_list = []
                    q_m_tgt_list = [] if alg == "rial" else None
                    for a in range(n):
                        t_agent = get_target_agent(a)
                        q_tgt, q_m_tgt, _ = t_agent(
                            obs_nxt_m[a].float().unsqueeze(1),
                            msg_nxt_m[a],
                            acts[a],
                            agent_id_tensors[a],
                            step_data["target_h_next"][a]
                        )
                        q_tgt_list.append(q_tgt)
                        if alg == "rial":
                            q_m_tgt_list.append(q_m_tgt)
                    q_tgt_sp = torch.stack(q_tgt_list, dim=0)
                    q_m_tgt_sp = (
                        torch.stack(q_m_tgt_list, dim=0)
                        if alg == "rial" else None
                    )

                # Legal action masking for next-state values
                next_val = torch.where(
                    in_r_nxt_m,
                    q_tgt_sp.max(dim=2).values,
                    q_tgt_sp[:, :, ACTION_NONE]
                )

            target = torch.where(d, r, r + gamma * next_val)
            sq_err = (q_sa - target) ** 2

            if alg == "rial" and step_data["q_m_split"] is not None:
                # Messages are legal only for the agent currently in the
                # room. The reference learns a separate communication Q
                # target and masks message TD error to legal comm actions.
                loss = loss + _rial_message_td_loss(
                    step_data["q_m_split"], step_data["sel_msgs"],
                    q_m_tgt_sp, step_data["reward"], step_data["done_next"],
                    step_data["in_room"], step_data["in_room_next"],
                    step_data["active"], gamma,
                )

            loss = loss + (sq_err * act_mask).sum()
            total_transitions += act_mask.sum()

        if total_transitions > 0:
            loss = loss / total_transitions
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    return loss, ep_rewards, total_steps


# ---------------------------------------------------------------------------
# Main FastTrainer
# ---------------------------------------------------------------------------

class FastTrainer:
    """
    Trainer faithful to Foerster et al. 2016 Algorithm 1.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = get_device()
        self.counters = TrainingCounters()

        torch.manual_seed(config.seed)

        n = config.n_agents
        alg = config.algorithm
        shared = config.param_sharing
        B = config.batch_size

        def make_agent(mode):
            return RNNAgent(
                n_agents=n, n_actions=N_ACTIONS, n_messages=2 if alg == "rial" else 1,
                obs_dim=1, msg_dim=1, hidden_size=128, comm_mode=mode,
            ).to(self.device)

        if alg == "rial":
            if shared:
                agent = make_agent("rial")
                self.controller = RIALController(agent, n_messages=2, epsilon=config.epsilon)
                self.params = list(agent.parameters())
                t_agent = make_agent("rial")
                t_agent.copy_weights_from(agent)
                t_agent.eval()
                self._target_ctrl = RIALController(t_agent, n_messages=2, epsilon=0.0)
            else:
                agents = [make_agent("rial") for _ in range(n)]
                self.controller = _NSRIALController(agents, n_messages=2, epsilon=config.epsilon)
                self.params = [p for ag in agents for p in ag.parameters()]
                t_agents = [make_agent("rial") for _ in range(n)]
                for ta, a in zip(t_agents, agents):
                    ta.copy_weights_from(a)
                    ta.eval()
                self._target_ctrl = _NSRIALController(t_agents, n_messages=2, epsilon=0.0)
        elif alg == "dial":
            dru = DRU(sigma=config.sigma)
            if shared:
                agent = make_agent("dial")
                self.controller = DIALController(agent, dru, epsilon=config.epsilon)
                self.params = list(agent.parameters())
                t_agent = make_agent("dial")
                t_agent.copy_weights_from(agent)
                t_agent.eval()
                self._target_ctrl = DIALController(t_agent, DRU(config.sigma), epsilon=0.0)
            else:
                agents = [make_agent("dial") for _ in range(n)]
                self.controller = _NSDIALController(agents, dru, epsilon=config.epsilon)
                self.params = [p for ag in agents for p in ag.parameters()]
                t_agents = [make_agent("dial") for _ in range(n)]
                for ta, a in zip(t_agents, agents):
                    ta.copy_weights_from(a)
                    ta.eval()
                self._target_ctrl = _NSDIALController(t_agents, DRU(config.sigma), epsilon=0.0)
        else:  # nocomm
            agent = make_agent("nocomm")
            self.controller = NoCommController(agent, epsilon=config.epsilon)
            self.params = list(agent.parameters())
            t_agent = make_agent("nocomm")
            t_agent.copy_weights_from(agent)
            t_agent.eval()
            self._target_ctrl = NoCommController(t_agent, epsilon=0.0)

        self.optimizer = optim.RMSprop(
            self.params, lr=config.lr, momentum=config.rms_momentum,
        )

        self._batch_env = BatchedSwitchRiddle(
            n=n, B=B, seed=config.seed * 10000, device=self.device,
        )

        self._train_envs = [
            SwitchRiddleEnv(n=n, seed=config.seed * 10000 + i)
            for i in range(config.batch_size)
        ]

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
        c = self.counters

        for epoch in range(cfg.max_epochs):
            if hasattr(self.controller, "agent"):
                self.controller.agent.train()
            elif hasattr(self.controller, "agents"):
                for ag in self.controller.agents:
                    ag.train()

            loss, ep_rewards, steps = run_batch_fast(
                self._batch_env, self.controller, cfg,
                training=True, device=self.device,
                agent_id_tensors=self._agent_id_tensors,
                zero_msgs=self._zero_msgs,
                target_controller=self._target_ctrl,
            )

            c.completed_episodes += cfg.batch_size
            c.env_timesteps += steps
            c.plotted_x_axis += 1
            c.update_steps += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Target network reset: every 100 completed episodes [A]
            if c.completed_episodes >= c._next_target_threshold:
                self._reset_target()
                c.target_update_count += 1
                c.last_target_update_episode = c.completed_episodes
                c._next_target_threshold = (
                    c.completed_episodes - (c.completed_episodes % 100) + 100
                )

            if (epoch + 1) % cfg.eval_every_epochs == 0:
                mean_r, std_r = self.evaluate(cfg.eval_episodes)
                record = {
                    "epoch": epoch + 1,
                    "plotted_x_axis": c.plotted_x_axis,
                    "completed_episodes": c.completed_episodes,
                    "update_steps": c.update_steps,
                    "env_timesteps": c.env_timesteps,
                    "target_update_count": c.target_update_count,
                    "last_target_update_episode": c.last_target_update_episode,
                    "mean_reward": mean_r,
                    "std_reward": std_r,
                    "loss": float(loss.item()),
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
                target_controller=None,
            )
            rewards_all.extend(ep_rew.cpu().tolist())
            remaining -= bs

        mean_r = sum(rewards_all) / len(rewards_all)
        std_r = statistics.stdev(rewards_all) if len(rewards_all) > 1 else 0.0
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
            "algorithm": cfg.algorithm_label,
            "n_agents": cfg.n_agents,
            "param_sharing": cfg.param_sharing,
            "seed": cfg.seed,
            "config": asdict(cfg),
            "final_counters": asdict(self.counters),
            "history": self.history,
        }
        with open(fname, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  -> log saved: {fname}")
