"""
NoComm Training Faithful to Foerster et al. Algorithm 1
=======================================================
Implements Algorithm 1 (lines 102-197 of Foerster et al. 2016) for NoComm:
  - Episode forward unroll: h_t = GRU(z_t, h_{t-1})
  - Actions selected epsilon-greedy from Q_t
  - Target network theta^- evaluates next step using h_t:
      y_t = r_t if done else r_t + gamma * max_u Q(o_{t+1}, h_t, u_t, a; theta^-)
  - Loss: MSE(Q_t[u_t], y_t)
  - Target network reset every 100 episodes
  - RMSProp(lr=5e-4, momentum=0.95), epsilon=0.05, gamma=1.0, B=32
"""

import sys
import os
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from switch_riddle.environment.batched_env import BatchedSwitchRiddle
from switch_riddle.agents.rnn_agent import RNNAgent
from switch_riddle.environment.switch_env import ACTION_NONE, ACTION_TELL, N_ACTIONS


def train_nocomm(max_epochs: int = 3000, seed: int = 0):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = 3
    B = 32
    T = 4 * n - 6  # 6
    gamma = 1.0
    lr = 5e-4
    eps = 0.05
    oracle_val = 20.0 / 27.0

    print("=" * 65)
    print(f"NoComm Training (Algorithm 1) | n={n} | B={B} | device={device} | seed={seed}")
    print("=" * 65)

    # Online network and target network
    agent = RNNAgent(n_agents=n, n_actions=N_ACTIONS, n_messages=1, hidden_size=128, comm_mode="nocomm").to(device)
    target_agent = RNNAgent(n_agents=n, n_actions=N_ACTIONS, n_messages=1, hidden_size=128, comm_mode="nocomm").to(device)
    target_agent.copy_weights_from(agent)
    target_agent.eval()

    optimizer = optim.RMSprop(agent.parameters(), lr=lr, momentum=0.95)
    env = BatchedSwitchRiddle(n=n, B=B, seed=seed * 1000, device=device)

    arange_n = torch.arange(n, device=device)
    aid_stacked = torch.cat([torch.full((B,), a, dtype=torch.long, device=device) for a in range(n)], dim=0)
    zero_msg = torch.zeros(n * B, 1, device=device)

    total_completed_episodes = 0
    target_reset_threshold = 100
    target_updates = 0

    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        agent.train()
        env.reset()

        h = agent.init_hidden(n * B, device)
        prev_act = torch.zeros(n * B, dtype=torch.long, device=device)
        active = torch.ones(B, dtype=torch.bool, device=device)
        in_room_gpu = env.in_room.to(device)

        # Store rollout steps for backward pass (Algorithm 1)
        rollout = []

        for step in range(T + 1):
            obs_matrix = (in_room_gpu.unsqueeze(0) == arange_n.unsqueeze(1)) & active.unsqueeze(0)
            obs_stacked = obs_matrix.float().unsqueeze(2).view(n * B, 1)

            # Online network forward pass
            q_u_all, _, h_next = agent(obs_stacked, zero_msg, prev_act, aid_stacked, h)
            q_u_split = q_u_all.view(n, B, N_ACTIONS)

            # Action selection: epsilon-greedy
            q_flat = q_u_split.view(n * B, N_ACTIONS)
            greedy = q_flat.argmax(dim=1)
            rand_mask = torch.rand(n * B, device=device) < eps
            rand_acts = torch.randint(0, N_ACTIONS, (n * B,), device=device)
            sel_acts = torch.where(rand_mask, rand_acts, greedy).view(n, B)

            # Mask non-room agents: only agent in room can Tell
            in_room_mask = (in_room_gpu.unsqueeze(0) == arange_n.unsqueeze(1))
            env_acts = torch.where(in_room_mask, sel_acts, torch.zeros_like(sel_acts))
            room_acts = env_acts.gather(0, in_room_gpu.unsqueeze(0)).squeeze(0)

            # Environment step
            in_room_next, _, reward, done_next = env.step(room_acts.cpu(), torch.zeros(B, dtype=torch.long))
            in_room_next = in_room_next.to(device)
            reward = reward.to(device)
            done_next = done_next.to(device)

            rollout.append({
                "q_u_split": q_u_split,
                "env_acts": env_acts,
                "reward": reward,
                "done_next": done_next,
                "active": active,
                "in_room_next": in_room_next,
                "h_next": h_next,  # hidden state after this step
            })

            active = ~done_next
            in_room_gpu = in_room_next
            prev_act = env_acts.view(n * B).detach()
            h = h_next

            if not active.any():
                break

        # -------------------------------------------------------------------
        # Backward Pass (Algorithm 1, lines 140-168)
        # -------------------------------------------------------------------
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
                # Evaluate next state with target network theta^-
                in_r_nxt = step_data["in_room_next"]
                d_nxt = step_data["done_next"]
                obs_nxt_m = (in_r_nxt.unsqueeze(0) == arange_n.unsqueeze(1)) & (~d_nxt).unsqueeze(0)
                obs_nxt_stk = obs_nxt_m.float().unsqueeze(2).view(n * B, 1)

                q_tgt_all, _, _ = target_agent(
                    obs_nxt_stk,
                    zero_msg,
                    acts.view(n * B),
                    aid_stacked,
                    step_data["h_next"]
                )
                max_next_q = q_tgt_all.view(n, B, N_ACTIONS).max(dim=2).values

            target = torch.where(d, r, r + gamma * max_next_q)

            sq_err = (q_sa - target) ** 2
            loss = loss + (sq_err * act_mask).sum()
            total_transitions += act_mask.sum()

        if total_transitions > 0:
            loss = loss / total_transitions
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_completed_episodes += B

        # Target network update: every 100 completed episodes [A]
        if total_completed_episodes >= target_reset_threshold:
            target_agent.copy_weights_from(agent)
            target_updates += 1
            target_reset_threshold += 100

        # Evaluation every 100 epochs
        if epoch % 100 == 0 or epoch == max_epochs:
            agent.eval()
            with torch.no_grad():
                EVAL_B = 200
                eval_env = BatchedSwitchRiddle(n=n, B=EVAL_B, seed=888888, device=device)
                eval_env.reset()
                h_ev = agent.init_hidden(n * EVAL_B, device)
                p_act_ev = torch.zeros(n * EVAL_B, dtype=torch.long, device=device)
                act_ev = torch.ones(EVAL_B, dtype=torch.bool, device=device)
                in_r_ev = eval_env.in_room.to(device)
                aid_ev = torch.cat([torch.full((EVAL_B,), a, dtype=torch.long, device=device) for a in range(n)], dim=0)
                ep_rewards = torch.zeros(EVAL_B, device=device)

                for _ in range(T + 1):
                    obs_m = (in_r_ev.unsqueeze(0) == arange_n.unsqueeze(1)) & act_ev.unsqueeze(0)
                    obs_stk = obs_m.float().unsqueeze(2).view(n * EVAL_B, 1)

                    q_ev, _, h_ev = agent(obs_stk, torch.zeros(n * EVAL_B, 1, device=device), p_act_ev, aid_ev, h_ev)
                    q_sp = q_ev.view(n, EVAL_B, N_ACTIONS)
                    greedy_ev = q_sp.argmax(dim=2)

                    in_r_m = (in_r_ev.unsqueeze(0) == arange_n.unsqueeze(1))
                    env_acts_ev = torch.where(in_r_m, greedy_ev, torch.zeros_like(greedy_ev))
                    room_a_ev = env_acts_ev.gather(0, in_r_ev.unsqueeze(0)).squeeze(0)

                    in_r_ev, _, rew, d_ev = eval_env.step(room_a_ev.cpu(), torch.zeros(EVAL_B, dtype=torch.long))
                    in_r_ev = in_r_ev.to(device)
                    ep_rewards += rew.to(device)
                    act_ev = ~d_ev.to(device)
                    p_act_ev = env_acts_ev.view(n * EVAL_B)

                    if not act_ev.any():
                        break

                mean_r = ep_rewards.mean().item()
                norm_r = mean_r / oracle_val
                elapsed = time.time() - t0
                print(f"Epoch {epoch:4d} | Eps: {total_completed_episodes:6d} | Mean Raw: {mean_r:+.4f} | Norm R: {norm_r:+.4f} | Tgt Upd: {target_updates:4d} | Time: {elapsed:.1f}s")

    print("\nTraining complete.")


if __name__ == "__main__":
    train_nocomm(max_epochs=2000, seed=0)
