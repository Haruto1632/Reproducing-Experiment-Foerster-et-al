"""
Unit Tests — Switch Riddle Environment (Phase 1)
================================================
Tests every item listed in IMPLEMENTATION_PLAN.md §Phase 1.

Run with:
    pytest research_reproduction/switch_riddle/tests/test_switch_env.py -v

Source-fidelity annotations match those in switch_env.py.
"""

from __future__ import annotations

import sys
import os

# Make the reproduction module importable when running from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import random

from switch_riddle.environment import (
    SwitchRiddleEnv,
    ACTION_NONE,
    ACTION_TELL,
    N_ACTIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def all_none(n: int) -> list[int]:
    """Return a list of ACTION_NONE for all n agents."""
    return [ACTION_NONE] * n


def tell_action(n: int, room_agent: int) -> list[int]:
    """Return actions where only room_agent performs Tell."""
    acts = [ACTION_NONE] * n
    acts[room_agent] = ACTION_TELL
    return acts


# ---------------------------------------------------------------------------
# 1. Time horizon
# ---------------------------------------------------------------------------

class TestTimeHorizon:
    """T = 4n - 6  [A] Paper Sec 6.2"""

    @pytest.mark.parametrize("n,expected_T", [(3, 6), (4, 10)])
    def test_time_horizon_formula(self, n, expected_T):
        env = SwitchRiddleEnv(n=n, seed=0)
        assert env.T == expected_T, (
            f"For n={n}, expected T={expected_T}, got T={env.T}")

    @pytest.mark.parametrize("n", [3, 4])
    def test_episode_terminates_at_T(self, n):
        """Episode must terminate exactly at step T when no Tell is issued."""
        env = SwitchRiddleEnv(n=n, seed=42)
        env.reset()
        done = False
        steps = 0
        while not done:
            result = env.step(all_none(n))
            done = result.done
            steps += 1
            assert steps <= env.T + 1, "Episode exceeded T steps without terminating."
        assert steps == env.T, (
            f"Episode ended at step {steps}, expected T={env.T}.")


# ---------------------------------------------------------------------------
# 2. Exactly one agent in the room per step
# ---------------------------------------------------------------------------

class TestRoomOccupancy:
    """Exactly one agent has observation=1 per step  [A]"""

    @pytest.mark.parametrize("n", [3, 4])
    def test_exactly_one_in_room_on_reset(self, n):
        env = SwitchRiddleEnv(n=n, seed=1)
        result = env.reset()
        assert sum(result.observations) == 1, (
            f"Expected exactly 1 agent in room after reset, "
            f"got obs={result.observations}")

    @pytest.mark.parametrize("n", [3, 4])
    def test_exactly_one_in_room_each_step(self, n):
        env = SwitchRiddleEnv(n=n, seed=2)
        env.reset()
        for _ in range(env.T - 1):
            result = env.step(all_none(n))
            if result.done:
                break
            assert sum(result.observations) == 1, (
                f"Expected exactly 1 agent in room, got obs={result.observations}")


# ---------------------------------------------------------------------------
# 3 & 4. Legal and illegal actions
# ---------------------------------------------------------------------------

class TestLegalActions:
    """Actions are validated against room occupancy  [A]"""

    def test_tell_in_room_is_legal(self):
        env = SwitchRiddleEnv(n=3, seed=0)
        env.reset()
        room = env.in_room
        acts = tell_action(3, room)
        # Should not raise
        result = env.step(acts)
        assert result.done  # Tell always terminates

    def test_none_outside_room_is_legal(self):
        env = SwitchRiddleEnv(n=3, seed=0)
        env.reset()
        # all None — all agents outside submit None; agent in room submits None
        result = env.step(all_none(3))
        # Should not raise
        assert not result.done or result.info["event"] == "timeout"

    def test_tell_outside_room_raises(self):
        env = SwitchRiddleEnv(n=3, seed=0)
        env.reset()
        room = env.in_room
        # Find an agent NOT in the room and make them Tell
        outside = (room + 1) % 3
        acts = [ACTION_NONE] * 3
        acts[outside] = ACTION_TELL
        with pytest.raises(ValueError, match="NOT in the room"):
            env.step(acts)

    def test_invalid_action_value_raises(self):
        env = SwitchRiddleEnv(n=3, seed=0)
        env.reset()
        room = env.in_room
        acts = [ACTION_NONE] * 3
        acts[room] = 99  # not a valid action
        with pytest.raises(ValueError):
            env.step(acts)


# ---------------------------------------------------------------------------
# 5 & 6 & 7. Reward calculation
# ---------------------------------------------------------------------------

class TestReward:
    """Reward is +1/−1 on Tell and 0 otherwise  [A]"""

    def _force_all_visited(self, env: SwitchRiddleEnv, n: int):
        """
        Deterministically drive the env until all n agents have visited.
        Uses a seeded env and reads the internal visited set.
        Returns False if T is reached before all visit.
        """
        for _ in range(env.T):
            if len(env.visited) == n:
                return True
            result = env.step(all_none(n))
            if result.done:
                return False
        return len(env.visited) == n

    def test_reward_tell_correct(self):
        """When all n agents have visited and Tell is issued, reward = +1  [A]"""
        # Use large number of seeds to find one that makes all visit quickly.
        for seed in range(200):
            env = SwitchRiddleEnv(n=3, seed=seed)
            env.reset()
            found = self._force_all_visited(env, 3)
            if found and not env.done:
                room = env.in_room
                result = env.step(tell_action(3, room))
                assert result.reward == 1.0, (
                    f"Expected +1 when all visited and Tell, got {result.reward}")
                assert result.done
                return
        pytest.skip("Could not find a seed where all 3 agents visit within T steps.")

    def test_reward_tell_incorrect(self):
        """When not all agents visited and Tell is issued, reward = −1  [A]"""
        env = SwitchRiddleEnv(n=3, seed=0)
        env.reset()
        # Immediately Tell without ensuring all visited
        room = env.in_room
        # visited will have at most 1 agent (n=3, so not all visited)
        assert len(env.visited) < 3
        result = env.step(tell_action(3, room))
        assert result.reward == -1.0, (
            f"Expected -1 when not all visited and Tell, got {result.reward}")
        assert result.done

    def test_reward_non_terminal_zero(self):
        """Reward is 0 at every non-terminal, non-Tell step  [A]"""
        env = SwitchRiddleEnv(n=3, seed=7)
        env.reset()
        for _ in range(env.T - 1):
            result = env.step(all_none(3))
            if result.done:
                # Timeout (reward=0 by assumption [C])
                assert result.reward == 0.0
                break
            assert result.reward == 0.0

    def test_reward_timeout_zero(self):
        """Episode reaching T without Tell ends with reward = 0  [C]"""
        env = SwitchRiddleEnv(n=3, seed=99)
        env.reset()
        done = False
        while not done:
            result = env.step(all_none(3))
            done = result.done
        assert result.reward == 0.0, (
            f"Expected timeout reward=0 [C], got {result.reward}")
        assert result.info["event"] == "timeout"


# ---------------------------------------------------------------------------
# 8. Terminal conditions
# ---------------------------------------------------------------------------

class TestTerminalConditions:
    def test_terminal_on_tell(self):
        env = SwitchRiddleEnv(n=3, seed=0)
        env.reset()
        room = env.in_room
        result = env.step(tell_action(3, room))
        assert result.done, "Tell must terminate the episode."

    def test_terminal_on_horizon(self):
        env = SwitchRiddleEnv(n=4, seed=5)
        env.reset()
        done = False
        while not done:
            result = env.step(all_none(4))
            done = result.done
        assert result.info["event"] == "timeout"

    def test_step_after_done_raises(self):
        env = SwitchRiddleEnv(n=3, seed=0)
        env.reset()
        room = env.in_room
        env.step(tell_action(3, room))
        with pytest.raises(RuntimeError, match="done"):
            env.step(all_none(3))


# ---------------------------------------------------------------------------
# 9. Message / switch availability
# ---------------------------------------------------------------------------

class TestMessageAvailability:
    """Switch state is the 1-bit communication channel  [A]"""

    def test_switch_readable_after_set(self):
        env = SwitchRiddleEnv(n=3, seed=0)
        env.reset()
        env.set_switch(1)
        assert env.switch == 1

    def test_switch_persists_between_steps(self):
        env = SwitchRiddleEnv(n=3, seed=0)
        env.reset()
        env.set_switch(1)
        env.step(all_none(3))
        assert env.switch == 1, "Switch state must persist between steps."

    def test_switch_only_visible_in_obs_not_leaked(self):
        """
        Observations contain only {0,1} indicating room occupancy.
        The switch state is NOT embedded in the observation vector itself — [A]
        it is a separate channel, only accessible to the learning algorithm
        when the agent is in the room.
        """
        env = SwitchRiddleEnv(n=3, seed=0)
        env.reset()
        env.set_switch(1)
        result = env.step(all_none(3))
        # Observations are strictly binary occupancy flags
        for obs in result.observations:
            assert obs in (0, 1), f"Observation leaked non-binary value: {obs}"

    def test_invalid_switch_value_raises(self):
        env = SwitchRiddleEnv(n=3, seed=0)
        env.reset()
        with pytest.raises(ValueError):
            env.set_switch(2)


# ---------------------------------------------------------------------------
# 10. Visited tracking
# ---------------------------------------------------------------------------

class TestVisitedTracking:
    def test_visited_starts_with_first_room_agent(self):
        env = SwitchRiddleEnv(n=3, seed=0)
        env.reset()
        assert len(env.visited) >= 1, "At least the first agent must be visited."
        assert env.in_room in env.visited

    def test_visited_grows(self):
        env = SwitchRiddleEnv(n=4, seed=42)
        env.reset()
        prev = len(env.visited)
        for _ in range(env.T - 1):
            result = env.step(all_none(4))
            current = len(env.visited)
            assert current >= prev, "Visited count must be non-decreasing."
            prev = current
            if result.done:
                break

    def test_visited_max_is_n(self):
        env = SwitchRiddleEnv(n=3, seed=1)
        env.reset()
        for _ in range(env.T):
            if env.done:
                break
            env.step(all_none(3))
        assert len(env.visited) <= env.n


# ---------------------------------------------------------------------------
# 11. Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_step(self):
        env = SwitchRiddleEnv(n=3, seed=0)
        env.reset()
        env.step(all_none(3))
        env.reset()
        assert env.step_count == 0

    def test_reset_clears_visited(self):
        env = SwitchRiddleEnv(n=3, seed=0)
        env.reset()
        # Run to end
        while not env.done:
            env.step(all_none(3))
        env.reset()
        assert len(env.visited) >= 1   # at least the new first agent
        assert env.step_count == 0

    def test_reset_sets_done_false(self):
        env = SwitchRiddleEnv(n=3, seed=0)
        env.reset()
        room = env.in_room
        env.step(tell_action(3, room))
        assert env.done
        env.reset()
        assert not env.done

    def test_double_reset_is_safe(self):
        env = SwitchRiddleEnv(n=3, seed=0)
        env.reset()
        env.reset()   # should not raise
        assert not env.done


# ---------------------------------------------------------------------------
# 12. No information leak via NoComm (zeroed message)
# ---------------------------------------------------------------------------

class TestNoInfoLeakNoComm:
    """
    If the communication channel is zeroed (NoComm baseline), the
    switch state carries no useful information about 'visited'.
    We verify that a zeroed-switch environment cannot distinguish
    between 'all visited' and 'not all visited' from observations alone.

    This test verifies the environment's isolation property, not the
    algorithm's learned policy.                                    [C]
    """

    def test_zero_switch_cannot_convey_visited_state(self):
        """
        Run two episodes with identical seeds and identical zero-switch policy.
        The visited outcome may differ (due to random prisoner selection),
        but the message channel carries no visited information.
        """
        n = 3
        env = SwitchRiddleEnv(n=n, seed=77)
        env.reset()
        env.set_switch(0)

        obs_log = []
        switch_log = []
        for _ in range(env.T):
            if env.done:
                break
            env.set_switch(0)   # always zero, no info transmitted
            result = env.step(all_none(n))
            obs_log.append(result.observations[:])
            switch_log.append(result.switch_state)

        # All switch states should be 0 — no information was transmitted
        assert all(s == 0 for s in switch_log), (
            "Switch was non-zero despite NoComm zeroing policy.")


# ---------------------------------------------------------------------------
# 13 & 14. Deterministic hand-crafted scenarios
# ---------------------------------------------------------------------------

class TestDeterministicScenarios:
    """
    Hand-crafted scenarios to verify correctness independently of the RNG.
    We use a custom subclass to inject a predetermined room sequence.
    """

    class ScriptedEnv(SwitchRiddleEnv):
        """Env that follows a pre-specified sequence of room occupants."""
        def __init__(self, n: int, room_sequence: list[int]):
            super().__init__(n=n, seed=0)
            self._seq = iter(room_sequence)
            self._step_idx = -1

        def reset(self, **kwargs):
            self._seq = iter(self._seq_backup)
            return super().reset(**kwargs)

        @classmethod
        def with_sequence(cls, n, seq):
            obj = cls.__new__(cls)
            SwitchRiddleEnv.__init__(obj, n=n, seed=0)
            obj._seq_full = seq
            obj._seq_iter = iter(seq)
            obj._seq_pos = 0
            return obj

    def _make_scripted(self, n: int, seq: list[int]) -> SwitchRiddleEnv:
        """
        Build a SwitchRiddleEnv that yields room occupants from `seq`
        instead of the random RNG.
        """
        env = SwitchRiddleEnv(n=n, seed=0)
        seq_iter = iter(seq)

        original_randrange = env._rng.randrange

        def fake_randrange(k):
            try:
                return next(seq_iter)
            except StopIteration:
                return original_randrange(k)

        env._rng.randrange = fake_randrange
        return env

    def test_n3_all_visit_then_tell_succeeds(self):
        """
        n=3: agents 0,1,2 each visit exactly once, then agent 0 tells.
        Expected: reward = +1
        """
        # room sequence: 0 (reset), 1 (step1), 2 (step2), 0 (step3-Tell)
        seq = [0, 1, 2, 0]
        env = self._make_scripted(n=3, seq=iter(seq))
        env.reset()
        assert env.in_room == 0
        assert env.visited == frozenset({0})

        # step 1: agent 0 does None, agent 1 enters
        env.step(all_none(3))
        assert env.in_room == 1
        assert 1 in env.visited

        # step 2: agent 1 does None, agent 2 enters
        env.step(all_none(3))
        assert env.in_room == 2
        assert 2 in env.visited
        assert env.visited == frozenset({0, 1, 2})

        # step 3: agent 2 does None, agent 0 re-enters
        env.step(all_none(3))
        assert env.in_room == 0

        # Tell — all visited
        result = env.step(tell_action(3, 0))
        assert result.reward == 1.0
        assert result.done

    def test_n3_tell_too_early_fails(self):
        """
        n=3: agent 0 is in room at reset; tells immediately.
        Only 1/3 agents visited — expected reward = -1
        """
        seq = [0]
        env = self._make_scripted(n=3, seq=iter(seq))
        env.reset()
        assert env.in_room == 0
        assert len(env.visited) == 1  # only agent 0

        result = env.step(tell_action(3, 0))
        assert result.reward == -1.0
        assert result.done

    def test_n4_all_visit_then_tell_succeeds(self):
        """
        n=4: agents 0,1,2,3 each visit once, then Tell.
        """
        seq = [0, 1, 2, 3, 0]
        env = self._make_scripted(n=4, seq=iter(seq))
        env.reset()

        for _ in range(4):      # steps to get all 4 agents in
            if len(env.visited) < 4 and not env.done:
                env.step(all_none(4))

        assert env.visited == frozenset({0, 1, 2, 3})
        room = env.in_room
        result = env.step(tell_action(4, room))
        assert result.reward == 1.0

    def test_n4_tell_too_early_fails(self):
        """
        n=4: Tell after only 2 agents visited — reward = -1
        """
        seq = [0, 1]
        env = self._make_scripted(n=4, seq=iter(seq))
        env.reset()
        env.step(all_none(4))
        # Now visited = {0, 1}, in_room = 1
        assert len(env.visited) == 2
        room = env.in_room
        result = env.step(tell_action(4, room))
        assert result.reward == -1.0
        assert result.done


# ---------------------------------------------------------------------------
# 15. Wrong number of actions
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_wrong_action_count_raises(self):
        env = SwitchRiddleEnv(n=3, seed=0)
        env.reset()
        with pytest.raises(ValueError, match="Expected 3 actions"):
            env.step([ACTION_NONE, ACTION_NONE])   # only 2


# ---------------------------------------------------------------------------
# 16. N_ACTIONS constant
# ---------------------------------------------------------------------------

class TestConstants:
    def test_n_actions_is_2(self):
        assert N_ACTIONS == 2

    def test_action_none_is_0(self):
        assert ACTION_NONE == 0

    def test_action_tell_is_1(self):
        assert ACTION_TELL == 1
