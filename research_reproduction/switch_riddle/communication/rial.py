"""
RIAL — Reinforced Inter-Agent Learning
========================================
Foerster et al. 2016, Sec 5.1  [A]

RIAL treats inter-agent communication as a standard RL problem.
Each agent runs an independent DRQN.  Messages are discrete actions
selected via a separate ε-greedy policy alongside environment actions.

Key properties [A]:
  - DRQN (no experience replay)
  - Separate Q_u and Q_m outputs
  - Separate ε-greedy selection for u and m
  - Parameter-sharing variant (shared) and non-sharing (RIAL-NS)
  - No gradient flows through the communication channel (messages
    are treated as discrete env actions by the receiving agent)

Usage (called by Trainer):
    agent = RNNAgent(..., comm_mode='rial')
    rial  = RIALController(agent, n_agents, ...)
    q_u, q_m, hidden = rial.step(obs, prev_msg, prev_action, agent_id, hidden)
    action, message  = rial.select(q_u, q_m, training=True)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Optional

from switch_riddle.agents.rnn_agent import RNNAgent


class RIALController:
    """
    Thin wrapper around RNNAgent for the RIAL protocol.

    In the parameter-sharing (shared) variant, all agents use the
    same RNNAgent instance.  In RIAL-NS, each agent owns a separate
    RNNAgent; the caller is responsible for indexing by agent.      [A]

    Parameters
    ----------
    agent : RNNAgent
        The underlying network (comm_mode must be 'rial').
    n_messages : int
        Number of discrete message values (2 for 1-bit: {0,1}).
    epsilon : float
        ε for ε-greedy exploration.  Paper specifies 0.05.          [A]
    """

    def __init__(
        self,
        agent: RNNAgent,
        n_messages: int,
        epsilon: float = 0.05,
    ):
        assert agent.comm_mode == "rial", \
            "RIALController requires an RNNAgent with comm_mode='rial'."
        self.agent = agent
        self.n_messages = n_messages
        self.epsilon = epsilon                                       # [A]

    def step(
        self,
        obs: torch.Tensor,
        prev_msg: torch.Tensor,
        prev_action: torch.Tensor,
        agent_id: torch.Tensor,
        hidden: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the DRQN.

        Returns
        -------
        q_u    : (batch, n_actions)   — action Q-values
        q_m    : (batch, n_messages)  — message Q-values
        hidden : (2, batch, 128)      — updated GRU state
        """
        q_u, q_m, hidden_out = self.agent(
            obs, prev_msg, prev_action, agent_id, hidden
        )
        return q_u, q_m, hidden_out

    def select(
        self,
        q_u: torch.Tensor,
        q_m: torch.Tensor,
        training: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Independent ε-greedy selection for action and message.     [A]

        During evaluation (training=False) ε=0 (greedy).

        Returns
        -------
        actions  : (batch,) long — selected environment actions
        messages : (batch,) long — selected discrete messages
        """
        eps = self.epsilon if training else 0.0

        actions = self._epsilon_greedy(q_u, eps)                   # [A]
        messages = self._epsilon_greedy(q_m, eps)                  # [A]
        return actions, messages

    @staticmethod
    def _epsilon_greedy(q: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        ε-greedy selection over Q-values.

        Parameters
        ----------
        q       : (batch, n_choices)
        epsilon : exploration probability

        Returns
        -------
        choices : (batch,) long
        """
        batch, n = q.shape
        greedy = q.argmax(dim=-1)                                  # (batch,)
        if epsilon > 0.0:
            rand_mask = torch.rand(batch, device=q.device) < epsilon
            random_choices = torch.randint(0, n, (batch,), device=q.device)
            choices = torch.where(rand_mask, random_choices, greedy)
        else:
            choices = greedy
        return choices
