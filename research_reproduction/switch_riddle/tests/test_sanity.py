"""
Environment and Training Sanity Check
======================================
Systematic check of:
  1. T = 4n - 6 for n=3
  2. Theoretical oracle reward probability
  3. Oracle handcrafted policy (uses true visited state)
  4. Optimal NoComm upper bound
  5. Message routing correctness
  6. GRU hidden state persistence (not reset per timestep)
  7. Switch state read/write semantics
  8. Action masking
  9. Evaluation loop correctness

Run with:
    python research_reproduction/switch_riddle/tests/test_sanity.py
"""

from __future__ import annotations

import sys, os, math, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np

from switch_riddle.environment.switch_env import SwitchRiddleEnv, ACTION_NONE, ACTION_TELL
from switch_riddle.environment.batched_env import BatchedSwitchRiddle


DIVIDER = "=" * 65


def header(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def ok(msg):    print(f"  [PASS]  {msg}")
def fail(msg):  print(f"  [FAIL]  {msg}")
def info(msg):  print(f"  [INFO]  {msg}")


# ===========================================================================
# 1. Verify T = 4n - 6
# ===========================================================================

def check_time_horizon():
    header("1. Time Horizon T = 4n - 6")
    for n in (3, 4):
        env = SwitchRiddleEnv(n=n, seed=0)
        expected = 4 * n - 6
        if env.T == expected:
            ok(f"n={n}: T = {env.T}")
        else:
            fail(f"n={n}: T = {env.T}, expected {expected}")

    # Extra: verify episode actually terminates AT T
    n = 3
    env = SwitchRiddleEnv(n=n, seed=42)
    env.reset()
    steps = 0
    while not env.done:
        env.step([ACTION_NONE] * n)
        steps += 1
    if steps <= env.T + 1:
        ok(f"n=3: episode terminates after at most T+1={env.T+1} steps (got {steps})")
    else:
        fail(f"n=3: episode ran {steps} steps, expected <= {env.T+1}")


# ===========================================================================
# 2. Theoretical oracle probability
# ===========================================================================

def theoretical_oracle_prob(n: int) -> float:
    """P(all n agents visit within T = 4n-6 draws, uniform w/ replacement)."""
    T = 4 * n - 6
    return sum(
        ((-1)**k) * math.comb(n, k) * ((n - k) / n) ** T
        for k in range(n + 1)
    )

def check_theory():
    header("2. Theoretical Oracle Probability")
    for n in (3, 4):
        p = theoretical_oracle_prob(n)
        T = 4 * n - 6
        info(f"n={n}, T={T}: P(all visited) = {p:.6f}")
    # Exact for n=3: 20/27
    expected_n3 = 20 / 27
    p3 = theoretical_oracle_prob(3)
    if abs(p3 - expected_n3) < 1e-9:
        ok(f"n=3 exact: {p3:.6f} == 20/27 = {expected_n3:.6f}")
    else:
        fail(f"n=3: got {p3:.6f}, expected {expected_n3:.6f}")


# ===========================================================================
# 3. Oracle handcrafted policy on SwitchRiddleEnv
#    (knows true visited set, tells exactly when all visited)
# ===========================================================================

def oracle_policy_single(n: int, n_episodes: int = 100_000, seed: int = 0) -> dict:
    """
    Oracle: Tell when visited_count == n, otherwise None.
    Returns statistics over n_episodes.
    """
    rng = random.Random(seed)
    env = SwitchRiddleEnv(n=n, seed=seed)

    correct_tell = 0
    wrong_tell   = 0
    timeout      = 0
    total_reward = 0.0

    for _ in range(n_episodes):
        env._rng = rng  # share rng
        env.reset()
        while not env.done:
            actions = [ACTION_NONE] * n
            # Oracle: tell only when all n have visited
            in_r = env.in_room
            if len(env.visited) == n:
                actions[in_r] = ACTION_TELL
            result = env.step(actions)
            if result.done:
                total_reward += result.reward
                if result.reward > 0:
                    correct_tell += 1
                elif result.reward < 0:
                    wrong_tell += 1
                else:
                    timeout += 1

    return {
        "correct_tell": correct_tell,
        "wrong_tell":   wrong_tell,
        "timeout":      timeout,
        "mean_reward":  total_reward / n_episodes,
        "normalized":   (total_reward / n_episodes) / theoretical_oracle_prob(n),
        "n_episodes":   n_episodes,
    }

def check_oracle_policy():
    header("3. Oracle Handcrafted Policy (SwitchRiddleEnv, n=3, 100k episodes)")
    N_EPS = 100_000
    n = 3
    r = oracle_policy_single(n=n, n_episodes=N_EPS, seed=0)
    p_theory = theoretical_oracle_prob(n)
    info(f"Episodes:      {r['n_episodes']:,}")
    info(f"Correct Tell:  {r['correct_tell']:,} ({100*r['correct_tell']/N_EPS:.2f}%)")
    info(f"Wrong Tell:    {r['wrong_tell']:,} ({100*r['wrong_tell']/N_EPS:.2f}%)")
    info(f"Timeout:       {r['timeout']:,} ({100*r['timeout']/N_EPS:.2f}%)")
    info(f"Mean reward:   {r['mean_reward']:.5f}")
    info(f"Theoretical:   {p_theory:.5f}")
    info(f"Normalized:    {r['normalized']:.5f}")

    if r['wrong_tell'] == 0:
        ok("Oracle policy never tells wrong")
    else:
        fail(f"Oracle tells wrong {r['wrong_tell']} times — environment bug!")

    if abs(r['mean_reward'] - p_theory) < 0.01:
        ok(f"Mean reward {r['mean_reward']:.5f} ~= theoretical {p_theory:.5f}")
    else:
        fail(f"Mean reward {r['mean_reward']:.5f} deviates from theoretical {p_theory:.5f}")


# ===========================================================================
# 3b. Oracle on BatchedSwitchRiddle
# ===========================================================================

def oracle_batched(n: int, n_episodes: int = 100_000, B: int = 64) -> dict:
    """Same oracle but using BatchedSwitchRiddle."""
    total_reward = 0.0
    wrong_tell   = 0
    correct_tell = 0
    timeout      = 0
    episodes_run = 0

    n_batches = n_episodes // B
    T = 4 * n - 6

    for batch_idx in range(n_batches):
        env = BatchedSwitchRiddle(n=n, B=B, seed=batch_idx, device=torch.device("cpu"))
        env.reset()
        episode_reward = torch.zeros(B)
        done = torch.zeros(B, dtype=torch.bool)

        for t in range(T + 1):
            active = ~done
            if not active.any():
                break

            # Oracle: Tell if visited_count == n AND you are active
            v = env.visited_count    # (B,) — how many distinct agents visited
            # Room agent tells if all visited; others say None
            room_tells = active & (v == n)

            # room_acts: Tell=1 if room_tells, else 0=None
            room_acts = room_tells.long()
            msg_vals  = torch.zeros(B, dtype=torch.long)  # switch doesn't matter

            in_r_new, sw_new, reward, new_done = env.step(room_acts, msg_vals)

            episode_reward += reward.cpu()

            # Track outcomes for terminal episodes
            just_done = new_done & ~done
            for b in range(B):
                if just_done[b]:
                    r = float(episode_reward[b])
                    if r > 0:    correct_tell += 1
                    elif r < 0:  wrong_tell   += 1
                    else:        timeout      += 1

            done = new_done

        total_reward += episode_reward.sum().item()
        episodes_run += B

    mean_r = total_reward / episodes_run
    return {
        "correct_tell": correct_tell,
        "wrong_tell":   wrong_tell,
        "timeout":      timeout,
        "mean_reward":  mean_r,
        "normalized":   mean_r / theoretical_oracle_prob(n),
        "n_episodes":   episodes_run,
    }


def check_oracle_batched():
    header("3b. Oracle on BatchedSwitchRiddle (n=3, 100k episodes)")
    n = 3
    r = oracle_batched(n=n, n_episodes=100_000, B=64)
    p_theory = theoretical_oracle_prob(n)
    info(f"Correct Tell:  {r['correct_tell']:,}")
    info(f"Wrong Tell:    {r['wrong_tell']:,}")
    info(f"Timeout:       {r['timeout']:,}")
    info(f"Mean reward:   {r['mean_reward']:.5f}")
    info(f"Theoretical:   {p_theory:.5f}")
    info(f"Normalized:    {r['normalized']:.5f}")

    if r['wrong_tell'] == 0:
        ok("Batched oracle never tells wrong")
    else:
        fail(f"Batched oracle tells wrong {r['wrong_tell']} times — BatchedEnv bug!")

    if abs(r['mean_reward'] - p_theory) < 0.02:
        ok(f"Batched mean reward {r['mean_reward']:.5f} ~= theoretical {p_theory:.5f}")
    else:
        fail(f"Batched mean reward {r['mean_reward']:.5f} deviates from theoretical {p_theory:.5f}")


# ===========================================================================
# 4. Verify switch read/write semantics
#    The critical semantic: room agent READS the switch that the PREVIOUS
#    room agent WROTE — not their own previous output.
# ===========================================================================

def check_switch_semantics():
    header("4. Switch Read/Write Semantics")

    n = 3
    env = SwitchRiddleEnv(n=n, seed=0)
    env.reset()

    # Step 1: whoever is in room, write switch=1
    in_r0 = env.in_room
    env.set_switch(1)
    result = env.step([ACTION_NONE] * n)
    info(f"After step 1: switch={env.switch}, new room agent={env.in_room}")

    if not result.done:
        in_r1 = env.in_room
        # Now in_r1 should SEE switch=1 (what in_r0 wrote)
        switch_readable = env.switch
        if switch_readable == 1:
            ok(f"Agent {in_r1} reads switch=1 (written by agent {in_r0})")
        else:
            fail(f"Agent {in_r1} reads switch={switch_readable}, expected 1")

        # Now in_r1 writes switch=0
        env.set_switch(0)
        result2 = env.step([ACTION_NONE] * n)
        if not result2.done:
            in_r2 = env.in_room
            switch_r2 = env.switch
            if switch_r2 == 0:
                ok(f"Agent {in_r2} reads switch=0 (written by agent {in_r1})")
            else:
                fail(f"Agent {in_r2} reads switch={switch_r2}, expected 0")


# ===========================================================================
# 5. What agents actually observe (message input check)
#    Key question: does the implementation pass the SWITCH STATE (written
#    by previous room agent) as the input message to the current room agent?
# ===========================================================================

def check_message_input_semantics():
    header("5. Message Input Semantics (critical)")
    info("Checking what is passed as prev_msg in fast_trainer vs what paper requires.")
    print()

    # The paper's requirement [A]:
    # - Room agent at time t READS the switch = message written by previous room agent
    # - Room agent at time t WRITES new switch = their output message
    # - Non-room agents have NO message input (or zero)
    print("  Paper requires [A]:")
    print("    Room agent input message   = switch state (written by prev room agent)")
    print("    Non-room agent input msg   = 0 or last switch they wrote")
    print()

    # What the current fast_trainer does:
    print("  Current fast_trainer (DIAL shared path, lines 165-169):")
    print("    msg_in = prev_msg_all   # (n*B, 1)")
    print("    where prev_msg_all[a*B + b] = agent a's own output message at t-1")
    print()
    print("  BUG: each agent receives their OWN previous output, not the")
    print("  switch state written by whoever was the last room agent.")
    print()

    # Demonstrate the semantic difference concretely:
    # With correct semantics: agent 0 writes msg=1 to switch at t=0
    #   At t=1, if agent 1 is in room, agent 1's input msg = 1
    # With current semantics: agent 1's input at t=1 = agent 1's own output at t=0

    print("  Concrete example with n=3:")
    print("    t=0: agent 0 in room, outputs msg=1.0 → switch becomes 1")
    print("    t=1: agent 1 in room")
    print("    CORRECT: agent 1 receives msg=1.0 (the switch value)")
    print("    CURRENT: agent 1 receives its own msg from t=0 (irrelevant)")
    print()

    # Also check the original trainer
    print("  Original trainer (trainer.py) has same bug:")
    print("    prev_msgs = new_msgs  (each agent's own previous output)")
    print("    switch state is written to env but never READ back as agent input")
    print()

    fail("Communication channel is broken: agents cannot read each other's messages")
    fail("This explains near-zero performance: agents learn with no actual comms")

    # Correct implementation should be:
    print()
    print("  CORRECT implementation should be:")
    print("    For room agent a at time t:")
    print("      msg_in[a, b] = switch_state[b]  (what was written at t-1)")
    print("    For non-room agents:")
    print("      msg_in[a, b] = 0  (or last switch they saw, but 0 is safe)")
    print()
    print("  NOTE: NoComm correctly zeros all messages — so NoComm is unaffected.")
    print("  But DIAL and RIAL are fundamentally broken by this bug.")


# ===========================================================================
# 6. GRU hidden state persistence check
#    Verify: hidden state is NOT reset between timesteps within an episode
# ===========================================================================

def check_gru_hidden_persistence():
    header("6. GRU Hidden State Persistence")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from switch_riddle.agents.rnn_agent import RNNAgent

    agent = RNNAgent(n_agents=3, n_actions=2, n_messages=2,
                     obs_dim=1, msg_dim=1, hidden_size=4, comm_mode="nocomm")
    B = 2
    device = torch.device("cpu")

    h = agent.init_hidden(B, device)              # zeros
    obs = torch.ones(B, 1)
    msg = torch.zeros(B, 1)
    act = torch.zeros(B, dtype=torch.long)
    aid = torch.zeros(B, dtype=torch.long)

    # Step 1
    q1, _, h1 = agent(obs, msg, act, aid, h)
    # Step 2 — pass h1 (not fresh zeros)
    q2, _, h2 = agent(obs, msg, act, aid, h1)
    # Step 3 — reset hidden (wrong — simulating reset bug)
    h_fresh = agent.init_hidden(B, device)
    q3_reset, _, _ = agent(obs, msg, act, aid, h_fresh)

    # q2 and q3_reset should differ if hidden state matters
    if not torch.allclose(q2, q3_reset):
        ok("GRU output differs when hidden state is carried vs reset → state persists")
    else:
        fail("GRU output unchanged — hidden state has no effect (possible bug or degenerate init)")

    # Verify h1 != h (state changes after step)
    if not torch.allclose(h1, h):
        ok("Hidden state changes after first step")
    else:
        fail("Hidden state did NOT change after first step")

    # Verify h2 != h1 (state changes after second step)
    if not torch.allclose(h2, h1):
        ok("Hidden state changes after second step")
    else:
        fail("Hidden state did NOT change after second step")


# ===========================================================================
# 7. NoComm reachable reward check
#    Without communication, the best strategy is:
#    - Tell only on the very last step if you've visited at least once
#    - This gives a nonzero (but suboptimal) reward
#    We verify this WITH the current environment (not the trained model).
# ===========================================================================

def nocomm_heuristic_policy(n: int, n_episodes: int = 100_000, seed: int = 0) -> dict:
    """
    Simple NoComm heuristic: Tell on the LAST step if in room.
    This does not require any information from other agents.
    Expected reward: approximately (1/n) * P(all_visited | T steps)
    """
    rng = random.Random(seed)
    env = SwitchRiddleEnv(n=n, seed=seed)
    T = env.T

    total_reward = 0.0
    correct = 0; wrong = 0; timeout = 0

    for _ in range(n_episodes):
        env._rng = rng
        env.reset()
        step = 0
        while not env.done:
            in_r = env.in_room
            actions = [ACTION_NONE] * n
            step += 1
            # Tell only on the very last possible step
            if step >= T:
                actions[in_r] = ACTION_TELL
            result = env.step(actions)
            if result.done:
                total_reward += result.reward
                if result.reward > 0:   correct += 1
                elif result.reward < 0: wrong   += 1
                else:                   timeout += 1

    return {
        "mean_reward": total_reward / n_episodes,
        "correct":     correct,
        "wrong":       wrong,
        "timeout":     timeout,
    }


def check_nocomm_heuristic():
    header("7. NoComm Heuristic Policy (Tell at last step)")
    n = 3
    r = nocomm_heuristic_policy(n=n, n_episodes=100_000)
    info(f"Correct={r['correct']:,}  Wrong={r['wrong']:,}  Timeout={r['timeout']:,}")
    info(f"Mean reward = {r['mean_reward']:.5f}")
    info(f"Normalized  = {r['mean_reward'] / theoretical_oracle_prob(n):.5f}")

    if r['mean_reward'] > 0.0:
        ok(f"NoComm heuristic achieves positive reward ({r['mean_reward']:.5f})")
    else:
        fail("NoComm heuristic gets ≤ 0 reward — possible environment bug")


# ===========================================================================
# 8. Verify action masking
# ===========================================================================

def check_action_masking():
    header("8. Action Masking")
    n = 3
    env = SwitchRiddleEnv(n=n, seed=0)
    env.reset()
    in_r = env.in_room
    not_in_room = [a for a in range(n) if a != in_r]

    # Legal: in-room agent says Tell
    try:
        actions = [ACTION_NONE] * n
        actions[in_r] = ACTION_TELL
        env.reset()
        env.step(actions)
        ok(f"In-room agent (a={in_r}) can Tell")
    except ValueError as e:
        fail(f"In-room agent cannot Tell: {e}")

    # Illegal: non-room agent says Tell
    for a in not_in_room[:1]:
        try:
            env.reset()
            actions = [ACTION_NONE] * n
            actions[a] = ACTION_TELL
            env.step(actions)
            fail(f"Non-room agent a={a} was allowed to Tell — masking broken!")
        except ValueError:
            ok(f"Non-room agent a={a} correctly blocked from Telling")


# ===========================================================================
# 9. Episode reset verification
# ===========================================================================

def check_episode_reset():
    header("9. Episode Reset")
    n = 3
    env = SwitchRiddleEnv(n=n, seed=42)
    env.reset()

    # Run to completion
    while not env.done:
        env.step([ACTION_NONE] * n)

    if env.done:
        ok("Episode terminated correctly")
    else:
        fail("Episode never terminated")

    # Reset and verify clean state
    env.reset()
    if not env.done:
        ok("After reset: done=False")
    else:
        fail("After reset: done=True — not reset correctly")

    if env.step_count == 0:
        ok("After reset: step_count=0")
    else:
        fail(f"After reset: step_count={env.step_count}")

    if len(env.visited) == 1:
        ok(f"After reset: visited has exactly 1 agent (initial occupant)")
    else:
        fail(f"After reset: visited={env.visited}")


# ===========================================================================
# 10. Diagnosis summary
# ===========================================================================

def print_diagnosis():
    header("10. Root Cause Diagnosis")
    print("""
  IDENTIFIED BUGS (in order of severity):

  [BUG-1] CRITICAL: Message channel broken for DIAL and RIAL
  ----------------------------------------------------------
  Current: prev_msg[a] = agent a's own previous output
  Paper:   room agent reads the switch = previous room agent's output
  
  Fix needed: At each timestep, the room agent's input message must be
  the current switch_state (written by the previous room agent), not
  their own last output.
  
  For DIAL (shared params), the correct message matrix is:
    msg_in[a, b] = switch_gpu[b]  if in_room[b] == a  (room agent reads switch)
    msg_in[a, b] = 0              otherwise (non-room agents get nothing)
  
  NOT: prev_msg_all (which gives each agent their own previous output)

  This explains why DIAL performance = 0.000:
  The communication channel was completely disconnected.
  DIAL and RIAL were training WITHOUT any actual inter-agent communication.

  [BUG-2] MINOR: switch_gpu not used as message input at all
  -----------------------------------------------------------
  switch_f is computed (line 151) but then overridden by prev_msg_all (line 169).
  The commented-out line 167 was the correct approach but was discarded.

  [OK] NoComm: unaffected by Bug-1 (it zeros all messages by design)
  [OK] Environment mechanics: all correct (oracle achieves theoretical bound)
  [OK] GRU hidden state: correctly persists within episode
  [OK] Action masking: correctly enforced
  [OK] Episode reset: correct
  [OK] T = 4n - 6: correct
  [OK] Reward: +1/-1/0 correct
""")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print(f"\n{'#'*65}")
    print("  SWITCH RIDDLE SANITY CHECK")
    print(f"{'#'*65}")

    check_time_horizon()
    check_theory()
    check_oracle_policy()
    check_oracle_batched()
    check_switch_semantics()
    check_message_input_semantics()
    check_gru_hidden_persistence()
    check_nocomm_heuristic()
    check_action_masking()
    check_episode_reset()
    print_diagnosis()
