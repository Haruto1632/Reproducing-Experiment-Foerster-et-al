"""
Diagnose Q-values across timesteps during NoComm training
"""
import sys, os, torch, torch.optim as optim
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from switch_riddle.environment.batched_env import BatchedSwitchRiddle
from switch_riddle.agents.rnn_agent import RNNAgent
from switch_riddle.environment.switch_env import ACTION_NONE, ACTION_TELL, N_ACTIONS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n = 3
B = 32
T = 6
gamma = 1.0
lr = 5e-4
eps = 0.05

agent = RNNAgent(n_agents=n, n_actions=N_ACTIONS, n_messages=1, hidden_size=128, comm_mode="nocomm").to(device)
target_agent = RNNAgent(n_agents=n, n_actions=N_ACTIONS, n_messages=1, hidden_size=128, comm_mode="nocomm").to(device)
target_agent.copy_weights_from(agent)
target_agent.eval()

optimizer = optim.RMSprop(agent.parameters(), lr=lr, momentum=0.95)

env = BatchedSwitchRiddle(n=n, B=B, seed=0, device=device)
arange_n = torch.arange(n, device=device)
aid_stacked = torch.cat([torch.full((B,), a, dtype=torch.long, device=device) for a in range(n)], dim=0)
zero_msg = torch.zeros(n * B, 1, device=device)

print("Starting training...")
for epoch in range(1001):
    agent.train()
    env.reset()
    h = agent.init_hidden(n * B, device)
    h_tgt = target_agent.init_hidden(n * B, device)
    prev_act = torch.zeros(n * B, dtype=torch.long, device=device)
    active = torch.ones(B, dtype=torch.bool, device=device)
    in_room_gpu = env.in_room.to(device)

    # Rollout
    traj_q = []
    traj_acts = []
    traj_rewards = []
    traj_dones = []
    traj_active = []
    traj_next_q_tgt = []

    for t in range(T + 1):
        obs_matrix = (in_room_gpu.unsqueeze(0) == arange_n.unsqueeze(1)) & active.unsqueeze(0)
        obs_stacked = obs_matrix.float().unsqueeze(2).view(n * B, 1)

        q_u_all, _, h_next = agent(obs_stacked, zero_msg, prev_act, aid_stacked, h)
        q_u_split = q_u_all.view(n, B, N_ACTIONS)

        # Epsilon-greedy
        q_flat = q_u_split.view(n * B, N_ACTIONS)
        greedy = q_flat.argmax(dim=1)
        rand_mask = torch.rand(n * B, device=device) < eps
        rand_acts = torch.randint(0, N_ACTIONS, (n * B,), device=device)
        sel_acts = torch.where(rand_mask, rand_acts, greedy).view(n, B)

        in_room_mask = (in_room_gpu.unsqueeze(0) == arange_n.unsqueeze(1))
        env_acts = torch.where(in_room_mask, sel_acts, torch.zeros_like(sel_acts))
        room_acts = env_acts.gather(0, in_room_gpu.unsqueeze(0)).squeeze(0)

        in_room_next, _, reward, done_next = env.step(room_acts.cpu(), torch.zeros(B, dtype=torch.long))
        in_room_next = in_room_next.to(device)
        reward = reward.to(device)
        done_next = done_next.to(device)

        # Target network evaluates next state
        with torch.no_grad():
            obs_m_next = (in_room_next.unsqueeze(0) == arange_n.unsqueeze(1)) & (~done_next).unsqueeze(0)
            obs_stk_next = obs_m_next.float().unsqueeze(2).view(n * B, 1)
            q_tgt_all, _, h_tgt_next = target_agent(obs_stk_next, zero_msg, env_acts.view(n * B), aid_stacked, h_tgt)
            max_next_q = q_tgt_all.view(n, B, N_ACTIONS).max(dim=2).values

        traj_q.append(q_u_split)
        traj_acts.append(env_acts)
        traj_rewards.append(reward)
        traj_dones.append(done_next)
        traj_active.append(active)
        traj_next_q_tgt.append(max_next_q)

        active = ~done_next
        in_room_gpu = in_room_next
        prev_act = env_acts.view(n * B).detach()
        h = h_next
        h_tgt = h_tgt_next

        if not active.any():
            break

    # Loss
    loss = torch.tensor(0.0, device=device)
    loss_count = 0
    for step_idx in range(len(traj_q)):
        q_u = traj_q[step_idx]
        acts = traj_acts[step_idx]
        r = traj_rewards[step_idx].unsqueeze(0).expand(n, B)
        d = traj_dones[step_idx].unsqueeze(0).expand(n, B)
        act_mask = traj_active[step_idx].unsqueeze(0).expand(n, B)
        max_next_q = traj_next_q_tgt[step_idx]

        q_sa = q_u.gather(2, acts.unsqueeze(2)).squeeze(2)
        target = torch.where(d, r, r + gamma * max_next_q)

        sq_err = (q_sa - target) ** 2
        loss = loss + (sq_err * act_mask).sum()
        loss_count += act_mask.sum()

    if loss_count > 0:
        loss = loss / loss_count
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Target update every 3 epochs (~100 episodes)
    if (epoch + 1) % 3 == 0:
        target_agent.copy_weights_from(agent)

    if (epoch + 1) % 200 == 0:
        # Inspect Q-values on a fresh evaluation episode
        agent.eval()
        with torch.no_grad():
            eval_env = BatchedSwitchRiddle(n=n, B=100, seed=12345, device=device)
            eval_env.reset()
            h_ev = agent.init_hidden(n * 100, device)
            p_act_ev = torch.zeros(n * 100, dtype=torch.long, device=device)
            act_ev = torch.ones(100, dtype=torch.bool, device=device)
            in_r_ev = eval_env.in_room.to(device)
            aid_ev = torch.cat([torch.full((100,), a, dtype=torch.long, device=device) for a in range(n)], dim=0)
            ep_rewards_eval = torch.zeros(100, device=device)

            print(f"\n--- EPOCH {epoch+1} Q-VALUE SUMMARY ---")
            for t in range(T + 1):
                obs_m = (in_r_ev.unsqueeze(0) == arange_n.unsqueeze(1)) & act_ev.unsqueeze(0)
                obs_stk = obs_m.float().unsqueeze(2).view(n * 100, 1)

                q_ev, _, h_ev = agent(obs_stk, torch.zeros(n * 100, 1, device=device), p_act_ev, aid_ev, h_ev)
                q_sp = q_ev.view(n, 100, N_ACTIONS)

                # Room occupant's Q values
                # in_r_ev: (100,)
                q_room = q_sp.gather(0, in_r_ev.unsqueeze(0).unsqueeze(2).expand(1, 100, 2)).squeeze(0)
                # q_room: (100, 2) [None, Tell]
                mean_q_none = q_room[:, 0][act_ev].mean().item() if act_ev.any() else 0.0
                mean_q_tell = q_room[:, 1][act_ev].mean().item() if act_ev.any() else 0.0
                print(f"  Step t={t} (active={act_ev.sum().item():3d}): Q(None)={mean_q_none:+.4f}, Q(Tell)={mean_q_tell:+.4f}")

                greedy_ev = q_sp.argmax(dim=2)
                in_r_m = (in_r_ev.unsqueeze(0) == arange_n.unsqueeze(1))
                env_acts_ev = torch.where(in_r_m, greedy_ev, torch.zeros_like(greedy_ev))
                room_a_ev = env_acts_ev.gather(0, in_r_ev.unsqueeze(0)).squeeze(0)

                in_r_ev, _, rew, d_ev = eval_env.step(room_a_ev.cpu(), torch.zeros(100, dtype=torch.long))
                in_r_ev = in_r_ev.to(device)
                ep_rewards_eval += rew.to(device)
                act_ev = ~d_ev.to(device)
                p_act_ev = env_acts_ev.view(n * 100)

                if not act_ev.any():
                    break

            mean_eval_r = ep_rewards_eval.mean().item()
            print(f"  Eval Mean Raw Reward: {mean_eval_r:+.4f} (Normalized: {mean_eval_r / (20/27):+.4f})")
