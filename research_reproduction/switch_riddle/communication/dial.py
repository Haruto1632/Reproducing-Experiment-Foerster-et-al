"""
DIAL — Differentiable Inter-Agent Learning
===========================================
Foerster et al. 2016, Sec 5.2  [A]

DIAL allows gradients to flow through the communication channel during
centralised training.  The sender outputs a real-valued message m_t^a;
the DRU regularises it during training (adds Gaussian noise + Logistic)
and discretises it during execution (hard threshold at 0).

Key properties [A]:
  - C-Net architecture (same backbone as RIAL but with continuous msg output)
  - Real-valued messages during training (gradient flows sender→receiver)
  - DRU(m, σ=2) during training
  - Binary 1{m>0} during evaluation
  - Parameter-sharing (DIAL) and non-sharing (DIAL-NS) variants

Usage (called by Trainer):
    agent = RNNAgent(..., comm_mode='dial')
    dru   = DRU(sigma=2.0)
    dial  = DIALController(agent, dru, n_agents, ...)
    q_u, m_processed, hidden = dial.step(obs, prev_msg, prev_action,
                                          agent_id, hidden, training=True)
    action = dial.select_action(q_u, training=True)
"""

from __future__ import annotations

import torch
from typing import Tuple, Optional

from switch_riddle.agents.rnn_agent import RNNAgent
from switch_riddle.communication.dru import DRU
from switch_riddle.communication.rial import RIALController


class DIALController:
    """
    Wrapper around RNNAgent for the DIAL / C-Net protocol.

    Parameters
    ----------
    agent   : RNNAgent with comm_mode='dial'
    dru     : DRU instance (σ=2)                                    [A]
    epsilon : ε for ε-greedy action selection on Q_u                [A]
    """

    def __init__(
        self,
        agent: RNNAgent,
        dru: DRU,
        epsilon: float = 0.05,
    ):
        assert agent.comm_mode == "dial", \
            "DIALController requires an RNNAgent with comm_mode='dial'."
        self.agent = agent
        self.dru = dru
        self.epsilon = epsilon                                       # [A]

    def step(
        self,
        obs: torch.Tensor,
        prev_msg: torch.Tensor,
        prev_action: torch.Tensor,
        agent_id: torch.Tensor,
        hidden: Optional[torch.Tensor],
        training: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the C-Net.

        The raw message output is passed through the DRU before it is
        fed to the next agent as `prev_msg`.  Gradient flows through
        the DRU's logistic during training (continuous path).        [A]

        Returns
        -------
        q_u         : (batch, n_actions)
        msg_out     : (batch, msg_dim)  — DRU-processed message
                      (differentiable during training; binary at eval)
        hidden_out  : (2, batch, 128)
        """
        q_u, m_raw, hidden_out = self.agent(
            obs, prev_msg, prev_action, agent_id, hidden
        )
        # Apply DRU — gradient flows through sigmoid in training mode [A]
        msg_out = self.dru(m_raw, training=training)
        return q_u, msg_out, hidden_out

    def select_action(
        self,
        q_u: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        """
        ε-greedy selection over environment actions Q_u.            [A]
        Messages are continuous and do NOT go through ε-greedy.     [A]

        Returns
        -------
        actions : (batch,) long
        """
        eps = self.epsilon if training else 0.0
        return RIALController._epsilon_greedy(q_u, eps)
