"""
Diagnostic: Test NoComm learning with correct vs incorrect bootstrapping
========================================================================
Compares:
1. Current trainer logic: target bootstrapped from s_t (BROKEN)
2. Correct DRQN logic: target bootstrapped from s_{t+1} using target network
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from switch_riddle.environment.batched_env import BatchedSwitchRiddle
from switch_riddle.agents.rnn_agent import RNNAgent
from switch_riddle.environment.switch_env import ACTION_NONE, ACTION_TELL, N_ACTIONS


def run_experiment(correct_bootstrapping: bool, n_epochs: int = 1000, seed: int = 0):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = 3
    B = 32
    T = 6
    gamma = 1.0
    lr = 5e-4
    eps = 0.05

    env = BatchedSwitchRiddle(n=n, B=B, seed=seed * 1000, device=device)

    # Q network and target network
    agent = RNNAgent(n_agents=n, n_actions=N_ACTIONS, n_messages=1, hidden_size=128, comm_mode="nocomm").to(device)
    target_agent = RNNAgent(n_agents=n, n_actions=N_ACTIONS, n_messages=1, hidden_size=128, comm_mode="nocomm").to(device)
    target_agent.copy_weights_from(agent)
    target_agent.eval()

    optimizer = optim.RMSprop(agent.parameters(), lr=lr, momentum=0.95)

    arange_n = torch.arange(n, device=device)
    aid_stacked = torch.cat([torch.full((B,), a, dtype=torch.long, device=device) for a in range(n)], dim=0)

    # Pre-allocate zero msgs and prev actions
    zero_msg = torch.zeros(n * B, 1, device=device)

    eval_rewards = []

    for epoch in range(n_epochs):
        agent.train()
        env.reset()

        h = agent.init_hidden(n * B, device)
        prev_act = torch.zeros(n * B, dtype=torch.long, device=device)

        active = torch.ones(B, dtype=torch.bool, device=device)
        in_room_gpu = env.in_room.to(device)

        # Store rollout for full episode DRQN update
        # We store: obs_stacked, prev_act, aid_stacked, actions_taken, rewards, dones, active_mask
        trajectory = []

        loss_terms = []

        for t in range(T + 1):
            obs_matrix = (in_room_gpu.unsqueeze(0) == arange_n.unsqueeze(1)) & active.unsqueeze(0)
            obs_stacked = obs_matrix.float().unsqueeze(2).view(n * B, 1)

            # Forward
            q_u_all, _, h_next = agent(obs_stacked, zero_msg, prev_act, aid_stacked, h)
            q_u_split = q_u_all.view(n, B, N_ACTIONS)

            # Epsilon-greedy
            q_flat = q_u_split.view(n * B, N_ACTIONS)
            greedy = q_flat.argmax(dim=1)
            rand_mask = torch.rand(n * B, device=device) < eps
            rand_acts = torch.randint(0, N_ACTIONS, (n * B,), device=device)
            sel_acts = torch.where(rand_mask, rand_acts, greedy).view(n, B)

            # Mask non-room agents
            in_room_mask = (in_room_gpu.unsqueeze(0) == arange_n.unsqueeze(1))
            env_acts = torch.where(in_room_mask, sel_acts, torch.zeros_like(sel_acts))
            room_acts = env_acts.gather(0, in_room_gpu.unsqueeze(0)).squeeze(0)

            # Step env
            in_room_next, _, reward, done_next = env.step(room_acts.cpu(), torch.zeros(B, dtype=torch.long))
            in_room_next = in_room_next.to(device)
            reward = reward.to(device)
            done_next = done_next.to(device)

            trajectory.append({
                "q_u_split": q_u_split,
                "env_acts": env_acts,
                "reward": reward,
                "done_next": done_next,
                "active": active,
                "obs_stacked": obs_stacked,
                "prev_act": prev_act,
                "h": h,
            })

            # Advance
            active = ~done_next
            in_room_gpu = in_room_next
            prev_act = env_acts.view(n * B).detach()
            h = h_next

            if not active.any():
                break

        # Compute TD loss
        loss = torch.tensor(0.0, device=device)
        total_steps = len(trajectory)

        for step_idx in range(total_steps):
            step_data = trajectory[step_idx]
            q_u = step_data["q_u_split"]
            acts = step_data["env_acts"]
            r = step_data["reward"].unsqueeze(0).expand(n, B)
            d = step_data["done_next"].unsqueeze(0).expand(n, B)
            act_mask = step_data["active"].unsqueeze(0).expand(n, B)

            q_sa = q_u.gather(2, acts.unsqueeze(2)).squeeze(2)

            if not correct_bootstrapping:
                # BROKEN LOGIC: bootstrap from current step
                max_q = q_u.detach().max(dim=2).values
                target = torch.where(d, r, r + gamma * max_q)
            else:
                # CORRECT LOGIC: bootstrap from next step using target network
                if step_idx + 1 < total_steps:
                    next_step = trajectory[step_idx + 1]
                    with torch.no_grad():
                        # Target network evaluates next state
                        q_tgt, _, _ = target_agent(
                            next_step["obs_stacked"], zero_msg, next_step["prev_act"],
                            aid_stacked, next_step["h"]
                        )
                        max_next_q = q_tgt.view(n, B, N_ACTIONS).max(dim=2).values
                    target = torch.where(d, r, r + gamma * max_next_q)
                else:
                    # Final step in trajectory is terminal
                    target = r

            sq_err = (q_sa - target) ** 2
            loss = loss + (sq_err * act_mask).sum() / (act_mask.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Target network update every 100 episodes (~3 epochs)
        if (epoch + 1) % 3 == 0:
            target_agent.copy_weights_from(agent)

        # Eval every 100 epochs
        if (epoch + 1) % 100 == 0:
            agent.eval()
            with torch.no_grad():
                eval_env = BatchedSwitchRiddle(n=n, B=100, seed=999999, device=device)
                eval_env.reset()
                h_ev = agent.init_hidden(n * 100, device)
                p_act_ev = torch.zeros(n * 100, dtype=torch.long, device=device)
                act_ev = torch.ones(100, dtype=torch.bool, device=device)
                in_r_ev = eval_env.in_room.to(device)
                aid_ev = torch.cat([torch.full((100,), a, dtype=torch.long, device=device) for a in range(n)], dim=0)
                ep_r = torch.zeros(100, device=device)

                for t in range(T + 1):
                    obs_m = (in_r_ev.unsqueeze(0) == arange_n.unsqueeze(1)) & act_ev.unsqueeze(0)
                    obs_stk = obs_m.float().unsqueeze(2).view(n * 100, 1)

                    q_ev, _, h_ev = agent(obs_stk, torch.zeros(n * 100, 1, device=device), p_act_ev, aid_ev, h_ev)
                    q_sp = q_ev.view(n, 100, N_ACTIONS)
                    greedy_ev = q_sp.argmax(dim=2)

                    in_r_m = (in_r_ev.unsqueeze(0) == arange_n.unsqueeze(1))
                    env_acts_ev = torch.where(in_r_m, greedy_ev, torch.zeros_like(greedy_ev))
                    room_a_ev = env_acts_ev.gather(0, in_r_ev.unsqueeze(0)).squeeze(0)

                    in_r_ev, _, rew, d_ev = eval_env.step(room_a_ev.cpu(), torch.zeros(100, dtype=torch.long))
                    in_r_ev = in_r_ev.to(device)
                    ep_r += rew.to(device)
                    act_ev = ~d_ev.to(device)
                    p_act_ev = env_acts_ev.view(n * 100)

                    if not act_ev.any():
                        break

                mean_eval_r = ep_r.mean().item()
                norm_r = mean_eval_r / (20.0 / 27.0)
                eval_rewards.append(norm_r)
                print(f"Epoch {epoch+1:4d} | Mean Raw: {mean_eval_r:+.4f} | Normalized: {norm_r:+.4f}")

    return eval_rewards


if __name__ == "__main__":
    print("=== TESTING BROKEN (CURRENT) BOOTSTRAPPING ===")
    r_broken = run_experiment(correct_bootstrapping=False, n_epochs=500, seed=0)

    print("\n=== TESTING CORRECT (DRQN s_{t+1}) BOOTSTRAPPING ===")
    r_correct = run_experiment(correct_bootstrapping=True, n_epochs=500, seed=0)
