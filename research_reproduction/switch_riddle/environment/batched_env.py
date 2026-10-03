"""
Batched Switch Riddle Environment  [Engineering Optimization — E]
==================================================================
Maintains B independent Switch Riddle episodes as tensors.
Advances all B episodes in ONE vectorized step with no Python
loop over the batch dimension.

SCIENTIFIC SPECIFICATION IS UNCHANGED:
  - Random prisoner selection with replacement           [A]
  - T = 4n - 6                                           [A]
  - Reward: +1 correct Tell, -1 wrong Tell, 0 timeout   [A]
  - 1-bit switch, written by room agent                  [A]
  - Termination: Tell or horizon                         [A]
  - No information leak across agents                    [A]

Engineering changes  [E]:
  - All B episode states stored as CPU long/bool tensors
  - step() is vectorized — no Python for-loop over B
  - Visited tracking: bitmask per episode (int64)
    Semantically identical to a set(); bit a = (visited_mask >> a) & 1
  - RNG: torch.Generator (same uniform distribution as random.randrange)
  - Observations returned as GPU float tensors directly

The original SwitchRiddleEnv is untouched; it is used for unit tests
and equivalence verification.
"""

from __future__ import annotations

import torch
from typing import Tuple


class BatchedSwitchRiddle:
    """
    Vectorized batch of B independent Switch Riddle episodes.

    State tensors (on CPU, long/bool, shape (B,)):
        _in_room    : index of current room occupant, 0..n-1
        _visited    : bitmask — bit a is set iff agent a has visited
        _switch     : 1-bit switch state, 0 or 1
        _step_count : steps taken in current episode
        _done       : episode finished flag

    Parameters
    ----------
    n      : number of agents                        [A]
    B      : batch size = 32 parallel episodes       [A]
    seed   : RNG seed for prisoner selection         [C]
    device : device for the output tensors returned by step()
    """

    def __init__(
        self,
        n: int,
        B: int,
        seed: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        self.n = n
        self.B = B
        self.T = 4 * n - 6          # horizon                         [A]
        self.device = device
        self._all_visited_mask = (1 << n) - 1   # bitmask when all visited

        self._gen = torch.Generator()
        self._gen.manual_seed(seed)

        # Pre-allocate state tensors (CPU)
        self._in_room    = torch.zeros(B, dtype=torch.long)
        self._visited    = torch.zeros(B, dtype=torch.long)
        self._switch     = torch.zeros(B, dtype=torch.long)
        self._step_count = torch.zeros(B, dtype=torch.long)
        self._done       = torch.ones(B,  dtype=torch.bool)   # force reset

        # Pre-allocate bit-shift table: bit_of[a] = 1 << a  (shape n)
        self._bit_of = (1 << torch.arange(n, dtype=torch.long))   # (n,)

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all B episodes simultaneously (fully vectorized)."""
        B, n = self.B, self.n
        self._step_count.zero_()
        self._switch.zero_()
        self._done.fill_(False)

        # Choose first room occupant: uniform random in [0, n)          [A]
        self._in_room = torch.randint(0, n, (B,), generator=self._gen)

        # Mark initial occupant in the visited bitmask
        # visited[b] = bit_of[in_room[b]]
        self._visited = self._bit_of[self._in_room]    # (B,) — fully vectorized

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(
        self,
        room_actions: torch.Tensor,   # (B,) long — 0=None, 1=Tell (already masked)
        msg_values:   torch.Tensor,   # (B,) long — 0 or 1 — written to switch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Advance all B episodes by one timestep (vectorized, no Python loop).

        Parameters
        ----------
        room_actions : (B,) long — action of current room agent (0 or 1)
        msg_values   : (B,) long — switch value the room agent writes

        Returns  (on self.device)
        -------
        obs_in_room : (B,) long  — index of agent in room next step
        switch_obs  : (B,) long  — switch state after this step
        reward      : (B,) float — reward this step (non-zero only at terminal)
        done        : (B,) bool  — episode terminated
        """
        B = self.B
        active = ~self._done                               # (B,) bool

        # --- 1. Write switch for active episodes                       [A] ---
        self._switch = torch.where(
            active,
            msg_values.cpu() & 1,   # ensure 0 or 1
            self._switch,
        )

        reward = torch.zeros(B, dtype=torch.float32)

        # --- 2. Tell action                                            [A] ---
        telling = active & (room_actions.cpu() == 1)

        if telling.any():
            all_visited = (self._visited == self._all_visited_mask)
            correct_tell = telling & all_visited
            wrong_tell   = telling & ~all_visited

            reward = torch.where(correct_tell, torch.ones(B),  reward)
            reward = torch.where(wrong_tell,   torch.full((B,), -1.0), reward)
            self._done = self._done | telling

        # --- 3. Non-tell active episodes ---
        non_tell_active = active & ~telling

        if non_tell_active.any():
            # Increment step counter for non-tell active episodes
            self._step_count = self._step_count + non_tell_active.long()

            # Timeout: step_count >= T                                  [A]
            timed_out = non_tell_active & (self._step_count >= self.T)
            self._done = self._done | timed_out

            # Advance room for episodes that are still going
            advancing = non_tell_active & ~timed_out
            if advancing.any():
                # New room occupant                                      [A]
                new_room = torch.randint(0, self.n, (B,), generator=self._gen)
                self._in_room = torch.where(advancing, new_room, self._in_room)
                # Update visited bitmask: OR in bit for new room agent
                new_bits = self._bit_of[self._in_room]     # (B,)
                self._visited = torch.where(
                    advancing,
                    self._visited | new_bits,
                    self._visited,
                )

        # --- 4. Return observation tensors on target device ---
        # obs: index of current room agent (will be read by next forward pass)
        obs_in_room = self._in_room.to(self.device)                  # (B,) long
        switch_obs  = self._switch.to(self.device)                   # (B,) long
        reward_out  = reward.to(self.device)                         # (B,) float
        done_out    = self._done.to(self.device)                     # (B,) bool

        return obs_in_room, switch_obs, reward_out, done_out

    # ------------------------------------------------------------------
    # Convenience properties (CPU tensors)
    # ------------------------------------------------------------------

    @property
    def in_room(self) -> torch.Tensor:
        """(B,) long — current room agent index (CPU)."""
        return self._in_room

    @property
    def switch(self) -> torch.Tensor:
        """(B,) long — current switch state (CPU)."""
        return self._switch

    @property
    def done(self) -> torch.Tensor:
        """(B,) bool (CPU)."""
        return self._done

    @property
    def visited_count(self) -> torch.Tensor:
        """(B,) long — number of distinct agents that have visited (CPU)."""
        # popcount of bitmask
        v = self._visited
        count = torch.zeros(self.B, dtype=torch.long)
        for i in range(self.n):
            count = count + ((v >> i) & 1)
        return count
