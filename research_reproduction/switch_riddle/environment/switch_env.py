"""
Switch Riddle Environment
=========================
Implements the Switch Riddle described in:

  Foerster et al. (2016) — Learning to Communicate with Deep Multi-Agent
  Reinforcement Learning. Sec 6.2.

Source fidelity legend used in comments:
  [A] Directly specified by paper
  [B] Strongly implied by paper
  [C] Necessary implementation assumption
  [D] Unknown / unresolved
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants / action encoding
# ---------------------------------------------------------------------------

ACTION_NONE = 0   # available to every agent in every position  [A]
ACTION_TELL = 1   # available only when agent is in the room     [A]

N_ACTIONS = 2     # {None, Tell}


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Return value of SwitchRiddleEnv.step()."""
    observations: List[int]          # o_t^a for each agent  [A]
    switch_state: int                # current switch bit     [A]
    reward: float                    # shared reward          [A]
    done: bool                       # episode terminated     [A]
    info: Dict                       # diagnostics            [C]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SwitchRiddleEnv:
    """
    Switch Riddle environment faithful to Foerster et al. 2016, Sec 6.2.

    The environment manages ONE episode at a time.  For batched training the
    caller is responsible for running multiple independent instances.

    Parameters
    ----------
    n : int
        Number of agents (prisoners).  Paper targets n=3 and n=4.   [A]
    seed : Optional[int]
        RNG seed for reproducibility.                                [C]
    """

    def __init__(self, n: int, seed: Optional[int] = None):
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        self.n: int = n
        self.T: int = 4 * n - 6     # maximum steps per episode      [A]
        self._rng = random.Random(seed)

        # Episode state (initialised on reset)
        self._step: int = 0
        self._switch: int = 0       # 1-bit shared switch            [A]
        self._in_room: int = -1     # index of current room occupant [A]
        self._visited: set = set()  # which agents have been in room [B]
        self._done: bool = True     # force reset before first step

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, switch_init: Optional[int] = None) -> StepResult:
        """
        Reset the environment and return the initial observation.

        Parameters
        ----------
        switch_init : Optional[int]
            Force the initial switch state (0 or 1).  Used only for
            deterministic test scenarios.  In normal training the initial
            switch state is set by the agent who first enters the room [C].
            The paper does not specify the initial switch state; we
            initialise it to 0.                                        [C]
        """
        self._step = 0
        self._visited = set()
        self._done = False

        # Initial switch state — NOT SPECIFIED IN PAPER.               [C]
        # We default to 0; test cases may override.
        self._switch = switch_init if switch_init is not None else 0

        # Choose the first room occupant
        self._in_room = self._rng.randrange(self.n)   # uniform, w/ replacement [A]
        self._visited.add(self._in_room)

        return self._make_result(reward=0.0, done=False,
                                 extra={"event": "reset"})

    def step(self, actions: List[int]) -> StepResult:
        """
        Advance the environment by one time-step.

        Parameters
        ----------
        actions : List[int]
            actions[a] is the action chosen by agent a.
            Legal values:
              - Agent in room  : ACTION_NONE (0) or ACTION_TELL (1)  [A]
              - Agent outside  : ACTION_NONE (0)                     [A]
            Illegal actions raise ValueError.

        Returns
        -------
        StepResult
        """
        if self._done:
            raise RuntimeError("Episode is done; call reset() first.")
        if len(actions) != self.n:
            raise ValueError(f"Expected {self.n} actions, got {len(actions)}.")

        self._validate_actions(actions)

        room_agent = self._in_room
        room_action = actions[room_agent]

        # ---- Check Tell ------------------------------------------------ [A]
        if room_action == ACTION_TELL:
            success = (len(self._visited) == self.n)
            reward = 1.0 if success else -1.0
            self._done = True
            return self._make_result(
                reward=reward, done=True,
                extra={"event": "tell",
                       "success": success,
                       "visited": set(self._visited)})

        # ---- Non-terminal step: agent may flip the switch -------------- [A]
        # The switch IS the message — no separate message action in the env.
        # The agent's choice of what to write to the switch is the
        # communication action (m_t^a), handled by the learning algorithm.
        # Here the env simply persists the switch state between room visits.

        self._step += 1

        # ---- Check time horizon --------------------------------------- [A]
        if self._step >= self.T:
            # Reached maximum steps without Tell — episode ends, reward 0. [C]
            # The paper does not explicitly state the reward on timeout;
            # reward=0 at non-terminal steps is the only stated rule.
            # A timeout is treated as a non-Tell terminal: reward = 0.    [C]
            self._done = True
            return self._make_result(
                reward=0.0, done=True,
                extra={"event": "timeout", "visited": set(self._visited)})

        # ---- Choose next room occupant --------------------------------- [A]
        self._in_room = self._rng.randrange(self.n)
        self._visited.add(self._in_room)

        return self._make_result(reward=0.0, done=False,
                                 extra={"event": "step"})

    def set_switch(self, value: int) -> None:
        """
        Allow the learning algorithm to write the switch state.

        The switch is the 1-bit communication channel.  The agent in the
        room writes it; the next agent in the room reads it.           [A]
        """
        if value not in (0, 1):
            raise ValueError(f"Switch value must be 0 or 1, got {value}.")
        self._switch = value

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def switch(self) -> int:
        """Current switch state (1-bit)."""
        return self._switch

    @property
    def in_room(self) -> int:
        """Index of the agent currently in the interrogation room."""
        return self._in_room

    @property
    def visited(self) -> frozenset:
        """Frozenset of agent indices that have visited the room."""
        return frozenset(self._visited)

    @property
    def step_count(self) -> int:
        """Number of steps taken in the current episode (not counting reset)."""
        return self._step

    @property
    def done(self) -> bool:
        return self._done

    @property
    def max_steps(self) -> int:
        return self.T

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_actions(self, actions: List[int]) -> None:
        """Raise ValueError if any agent provides an illegal action."""
        for a, act in enumerate(actions):
            if a == self._in_room:
                if act not in (ACTION_NONE, ACTION_TELL):
                    raise ValueError(
                        f"Agent {a} is in the room; legal actions are "
                        f"ACTION_NONE(0) and ACTION_TELL(1), got {act}.")
            else:
                if act != ACTION_NONE:
                    raise ValueError(
                        f"Agent {a} is NOT in the room; only ACTION_NONE(0) "
                        f"is legal, got {act}.")

    def _make_result(self, reward: float, done: bool, extra: Dict) -> StepResult:
        """Build a StepResult for the current state."""
        obs = [1 if a == self._in_room else 0 for a in range(self.n)]  # [A]
        info = {
            "step": self._step,
            "in_room": self._in_room,
            "switch": self._switch,
            "visited_count": len(self._visited),
            **extra,
        }
        return StepResult(
            observations=obs,
            switch_state=self._switch,
            reward=reward,
            done=done,
            info=info,
        )
