"""
Test DIAL and RIAL with legal action masking in Bellman target
=============================================================
Verifies whether restricting next-state Q-values to legal actions:
  - Inside room at t+1: max(Q(None), Q(Tell))
  - Outside room at t+1: Q(None) (only legal action)
stabilizes and achieves optimal communication protocols.
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
from switch_riddle.communication.dru import DRU
from switch_riddle.environment.switch_env import ACTION_NONE, ACTION_TELL, N_ACTIONS


def evaluate_agent(agent, alg, dru, n, device, oracle_val, seed=123456):
    agent.eval()
    T = 4 * n - 6
    arange_n = torch.arange(n, device=device)
    EVAL_B = 200
    with torch.no_grad():
        eval_env = BatchedSwitchRiddle(n=n, B=EVAL_B, seed=seed, device=device)
        eval_env.reset()
        h_ev = agent.init_hidden(n * EVAL_B, device)
        p_act_ev = torch.zeros(n * EVAL_B, dtype=torch.long, device=device)
        act_ev = torch.ones(EVAL_B, dtype=torch.bool, device=device)
        in_r_ev = eval_env.in_room.to(device)
        aid_ev = torch.cat([torch.full((EVAL_B,), a, dtype=torch.long, device=device) for a in range(n)], dim=0)
        ep_rewards = torch.zeros(EVAL_B, device=device)
        room_sw_ev = torch.zeros(EVAL_B, 1, device=device)

        for _ in range(T + 1):
            obs_m = (in_r_ev.unsqueeze(0) == arange_n.unsqueeze(1)) & act_ev.unsqueeze(0)
            obs_stk = obs_m.float().unsqueeze(2).view(n * EVAL_B, 1)

            in_r_m = (in_r_ev.unsqueeze(0) == arange_n.unsqueeze(1))
            msg_m = torch.where(
                in_r_m.unsqueeze(2),
                room_sw_ev.unsqueeze(0).expand(n, EVAL_B, 1),
                torch.zeros(n, EVAL_B, 1, device=device)
            )
            msg_stk = msg_m.view(n * EVAL_B, 1)

            if alg == "dial":
                q_ev, m_ev, h_ev = agent(obs_stk, msg_stk, p_act_ev, aid_ev, h_ev)
                q_sp = q_ev.view(n, EVAL_B, N_ACTIONS)
                m_bin = dru(m_ev, training=False).view(n, EVAL_B, 1)
                room_msg_ev = m_bin.gather(0, in_r_ev.unsqueeze(0).unsqueeze(2)).squeeze(0)
            elif alg == "rial":
                q_ev, q_m_ev, h_ev = agent(obs_stk, msg_stk, p_act_ev, aid_ev, h_ev)
                q_sp = q_ev.view(n, EVAL_B, N_ACTIONS)
                q_m_sp = q_m_ev.view(n, EVAL_B, 2)
                m_disc = q_m_sp.argmax(dim=2).unsqueeze(2).float()
                room_msg_ev = m_disc.gather(0, in_r_ev.unsqueeze(0).unsqueeze(2)).squeeze(0)
            else:  # nocomm
                q_ev, _, h_ev = agent(obs_stk, msg_stk, p_act_ev, aid_ev, h_ev)
                q_sp = q_ev.view(n, EVAL_B, N_ACTIONS)
                room_msg_ev = torch.zeros(EVAL_B, 1, device=device)

            greedy_ev = q_sp.argmax(dim=2)
            env_acts_ev = torch.where(in_r_m, greedy_ev, torch.zeros_like(greedy_ev))
            room_a_ev = env_acts_ev.gather(0, in_r_ev.unsqueeze(0)).squeeze(0)

            in_r_ev, _, rew, d_ev = eval_env.step(room_a_ev.cpu(), room_msg_ev.long().squeeze(1).cpu())
            in_r_ev = in_r_ev.to(device)
            ep_rewards += rew.to(device)
            act_ev = ~d_ev.to(device)
            p_act_ev = env_acts_ev.view(n * EVAL_B)
            room_sw_ev = room_msg_ev

            if not act_ev.any():
                break

        mean_r = ep_rewards.mean().item()
        norm_r = mean_r / oracle_val
        return mean_r, norm_r


def run_experiment(alg: str = "dial", max_epochs: int = 3000, seed: int = 0):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = 3
    B = 32
    T = 4 * n - 6  # 6
    gamma = 1.0
    lr = 5e-4
    eps = 0.05
    oracle_val = 20.0 / 27.0
    dru = DRU(sigma=2.0)

    print("=" * 65)
    print(f"Algorithm: {alg.upper()} | n={n} | B={B} | device={device} | seed={seed}")
    print("=" * 65)

    comm_mode = "dial" if alg == "dial" else ("rial" if alg == "rial" else "nocomm")
    agent = RNNAgent(n_agents=n, n_actions=N_ACTIONS, n_messages=2 if alg == "rial" else 1,
                     msg_dim=1, hidden_size=128, comm_mode=comm_mode).to(device)
    target_agent = RNNAgent(n_agents=n, n_actions=N_ACTIONS, n_messages=2 if alg == "rial" else 1,
                            msg_dim=1, hidden_size=128, comm_mode=comm_mode).to(device)
    target_agent.copy_weights_from(agent)
    target_agent.eval()

    optimizer = optim.RMSprop(agent.parameters(), lr=lr, momentum=0.95)
    env = BatchedSwitchRiddle(n=n, B=B, seed=seed * 1000, device=device)

    arange_n = torch.arange(n, device=device)
    aid_stacked = torch.cat([torch.full((B,), a, dtype=torch.long, device=device) for a in range(n)], dim=0)

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
        room_switch_msg = torch.zeros(B, 1, device=device)

        rollout = []

        for step in range(T + 1):
            obs_matrix = (in_room_gpu.unsqueeze(0) == arange_n.unsqueeze(1)) & active.unsqueeze(0)
            obs_stacked = obs_matrix.float().unsqueeze(2).view(n * B, 1)

            in_room_mask = (in_room_gpu.unsqueeze(0) == arange_n.unsqueeze(1))
            msg_matrix = torch.where(
                in_room_mask.unsqueeze(2),
                room_switch_msg.unsqueeze(0).expand(n, B, 1),
                torch.zeros(n, B, 1, device=device)
            )
            msg_stacked = msg_matrix.view(n * B, 1)

            if alg == "dial":
                q_u_all, m_raw_all, h_next = agent(obs_stacked, msg_stacked, prev_act, aid_stacked, h)
                q_u_split = q_u_all.view(n, B, N_ACTIONS)
                m_dru_all = dru(m_raw_all, training=True)
                m_dru_split = m_dru_all.view(n, B, 1)
                room_msg_out = m_dru_split.gather(0, in_room_gpu.unsqueeze(0).unsqueeze(2)).squeeze(0)
                room_msg_disc = (room_msg_out.detach() >= 0.5).long().squeeze(1)
                q_m_split = None
                sel_msgs = None
            elif alg == "rial":
                q_u_all, q_m_all, h_next = agent(obs_stacked, msg_stacked, prev_act, aid_stacked, h)
                q_u_split = q_u_all.view(n, B, N_ACTIONS)
                q_m_split = q_m_all.view(n, B, 2)
                # Epsilon-greedy for messages
                q_m_flat = q_m_split.view(n * B, 2)
                greedy_m = q_m_flat.argmax(dim=1)
                rand_m_mask = torch.rand(n * B, device=device) < eps
                rand_msgs = torch.randint(0, 2, (n * B,), device=device)
                sel_msgs = torch.where(rand_m_mask, rand_msgs, greedy_m).view(n, B)
                room_msg_disc = sel_msgs.gather(0, in_room_gpu.unsqueeze(0)).squeeze(0)
                room_msg_out = room_msg_disc.unsqueeze(1).float()
            else:  # nocomm
                q_u_all, _, h_next = agent(obs_stacked, msg_stacked, prev_act, aid_stacked, h)
                q_u_split = q_u_all.view(n, B, N_ACTIONS)
                room_msg_disc = torch.zeros(B, dtype=torch.long, device=device)
                room_msg_out = torch.zeros(B, 1, device=device)
                q_m_split = None
                sel_msgs = None

            # Action selection: epsilon-greedy
            q_flat = q_u_split.view(n * B, N_ACTIONS)
            greedy = q_flat.argmax(dim=1)
            rand_mask = torch.rand(n * B, device=device) < eps
            rand_acts = torch.randint(0, N_ACTIONS, (n * B,), device=device)
            sel_acts = torch.where(rand_mask, rand_acts, greedy).view(n, B)

            env_acts = torch.where(in_room_mask, sel_acts, torch.zeros_like(sel_acts))
            room_acts = env_acts.gather(0, in_room_gpu.unsqueeze(0)).squeeze(0)

            in_room_next, _, reward, done_next = env.step(room_acts.cpu(), room_msg_disc.cpu())
            in_room_next = in_room_next.to(device)
            reward = reward.to(device)
            done_next = done_next.to(device)

            rollout.append({
                "q_u_split": q_u_split,
                "q_m_split": q_m_split,
                "sel_msgs": sel_msgs,
                "env_acts": env_acts,
                "reward": reward,
                "done_next": done_next,
                "active": active,
                "in_room_next": in_room_next,
                "room_msg_out": room_msg_out,
                "h_next": h_next,
            })

            active = ~done_next
            in_room_gpu = in_room_next
            room_switch_msg = room_msg_out
            prev_act = env_acts.view(n * B).detach()
            h = h_next

            if not active.any():
                break

        # -------------------------------------------------------------------
        # Backward Pass (Algorithm 1 with legal action masking)
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

            q_sa = q_u.gather(2, acts.unsqueeze(2)).squeeze(2)

            with torch.no_grad():
                in_r_nxt = step_data["in_room_next"]
                d_nxt = step_data["done_next"]
                obs_nxt_m = (in_r_nxt.unsqueeze(0) == arange_n.unsqueeze(1)) & (~d_nxt).unsqueeze(0)
                obs_nxt_stk = obs_nxt_m.float().unsqueeze(2).view(n * B, 1)

                in_r_nxt_m = (in_r_nxt.unsqueeze(0) == arange_n.unsqueeze(1))
                msg_nxt_m = torch.where(
                    in_r_nxt_m.unsqueeze(2),
                    step_data["room_msg_out"].detach().unsqueeze(0).expand(n, B, 1),
                    torch.zeros(n, B, 1, device=device)
                )
                msg_nxt_stk = msg_nxt_m.view(n * B, 1)

                if alg == "dial":
                    q_tgt_all, _, _ = target_agent(obs_nxt_stk, msg_nxt_stk, acts.view(n * B), aid_stacked, step_data["h_next"])
                elif alg == "rial":
                    q_tgt_all, _, _ = target_agent(obs_nxt_stk, msg_nxt_stk, acts.view(n * B), aid_stacked, step_data["h_next"])
                else:
                    q_tgt_all, _, _ = target_agent(obs_nxt_stk, msg_nxt_stk, acts.view(n * B), aid_stacked, step_data["h_next"])

                q_tgt_sp = q_tgt_all.view(n, B, N_ACTIONS)

                # Legal action masking for next state value:
                # Agent in room at t+1 can choose Tell or None -> max(Q(None), Q(Tell))
                # Agent outside room at t+1 can ONLY choose None -> Q(None)
                next_val = torch.where(
                    in_r_nxt_m,
                    q_tgt_sp.max(dim=2).values,
                    q_tgt_sp[:, :, ACTION_NONE]
                )

            target = torch.where(d, r, r + gamma * next_val)
            sq_err = (q_sa - target) ** 2

            if alg == "rial" and step_data["q_m_split"] is not None:
                # RIAL also trains Q_m on the room agent's message
                q_m = step_data["q_m_split"]
                sel_m = step_data["sel_msgs"]
                q_m_sa = q_m.gather(2, sel_m.unsqueeze(2)).squeeze(2)
                sq_err_m = (q_m_sa - target) ** 2
                sq_err = sq_err + sq_err_m

            loss = loss + (sq_err * act_mask).sum()
            total_transitions += act_mask.sum()

        if total_transitions > 0:
            loss = loss / total_transitions
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_completed_episodes += B

        if total_completed_episodes >= target_reset_threshold:
            target_agent.copy_weights_from(agent)
            target_updates += 1
            target_reset_threshold += 100

        if epoch % 100 == 0 or epoch == max_epochs:
            mean_r, norm_r = evaluate_agent(agent, alg, dru, n, device, oracle_val, seed=777777)
            elapsed = time.time() - t0
            print(f"Epoch {epoch:4d} | Eps: {total_completed_episodes:6d} | Mean Raw: {mean_r:+.4f} | Norm R: {norm_r:+.4f} | Tgt Upd: {target_updates:4d} | Time: {elapsed:.1f}s")

    print(f"\n{alg.upper()} Training complete.")
    return agent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="dial")
    parser.add_argument("--epochs", type=int, default=2000)
    args = parser.parse_args()
    run_experiment(alg=args.alg, max_epochs=args.epochs)
