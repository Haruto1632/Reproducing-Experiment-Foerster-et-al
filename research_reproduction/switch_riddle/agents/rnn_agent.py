"""
Shared RNN Agent — Phase 2
==========================
Implements the agent backbone described in Foerster et al. 2016, Sec 6.1.

Architecture (paper-specified [A] unless noted):

  Inputs:  (obs, prev_message, prev_action, agent_id)
  ↓
  agent_id   → Embedding(n_agents, 128)                 [A]
  prev_action→ Embedding(n_actions, 128)                [A]
  prev_msg   → BatchNorm1d → Linear(msg_dim→128) → ReLU [A] / [C]
  obs        → Linear(obs_dim→128)                      [A spec, C arch]
  ↓
  z = element-wise sum of the four embeddings           [A]
  ↓
  2-layer GRU(128, 128)                                 [A]
  ↓
  Linear(128,128) → ReLU → Linear(128, n_out)           [A]

Implementation assumptions [C]:
  - TaskMLP: single Linear layer (paper says "task-specific network", depth unspecified)
  - BatchNorm applied to scalar message before MLP
  - Initial hidden state h_0 = zeros                    [A per Alg 1]
  - Initial message m_{-1} = zeros                      [C]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple


class RNNAgent(nn.Module):
    """
    Recurrent agent network shared across RIAL, DIAL, and NoComm.

    Parameters
    ----------
    n_agents : int
        Total number of agents (used for agent-index embedding).
    n_actions : int
        Number of environment actions (|U|).
    n_messages : int
        Number of discrete message values (|M|).  For DIAL this is 1
        (scalar continuous output); for RIAL this is 2 (discrete Q_m).
    obs_dim : int
        Dimensionality of the task observation (1 for Switch Riddle).
    msg_dim : int
        Dimensionality of the incoming message (1 for Switch Riddle).
    hidden_size : int
        GRU hidden size.  Paper specifies 128.                       [A]
    comm_mode : str
        'rial'  — produces Q_u and Q_m as separate discrete heads
        'dial'  — produces Q_u and a scalar continuous message output
        'nocomm'— produces Q_u only; message input is always zeroed
    """

    HIDDEN_SIZE = 128   # [A]

    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        n_messages: int,
        obs_dim: int = 1,
        msg_dim: int = 1,
        hidden_size: int = 128,
        comm_mode: str = "rial",
    ):
        super().__init__()
        assert comm_mode in ("rial", "dial", "nocomm"), \
            f"comm_mode must be 'rial', 'dial', or 'nocomm', got '{comm_mode}'"

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_messages = n_messages
        self.obs_dim = obs_dim
        self.msg_dim = msg_dim
        self.hidden_size = hidden_size
        self.comm_mode = comm_mode

        # ------------------------------------------------------------------
        # Embedding streams                                             [A]
        # ------------------------------------------------------------------

        # Agent-index embedding: a → 128
        self.agent_embed = nn.Embedding(n_agents, hidden_size)          # [A]

        # Previous-action embedding: u_{t-1} → 128
        self.action_embed = nn.Embedding(n_actions, hidden_size)        # [A]

        # Previous-message stream: m_{t-1} → BatchNorm → Linear → ReLU → 128
        # BatchNorm on the raw scalar message [A] / [C] (exact impl. assumed)
        self.msg_bn = nn.BatchNorm1d(msg_dim)                           # [C]
        self.msg_mlp = nn.Sequential(                                   # [A]
            nn.Linear(msg_dim, hidden_size),
            nn.ReLU(),
        )

        # Task-observation stream: o_t → 128
        # Paper: "task-specific network of same size". Depth unspecified. [C]
        self.obs_net = nn.Linear(obs_dim, hidden_size)                  # [C]

        # ------------------------------------------------------------------
        # Recurrent core: 2-layer GRU, 128 hidden units                [A]
        # ------------------------------------------------------------------
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )

        # ------------------------------------------------------------------
        # Output heads
        # ------------------------------------------------------------------
        # Shared first linear layer of the 2-layer output MLP          [A]
        self.out_fc1 = nn.Linear(hidden_size, hidden_size)

        # Action head: Q_u values (always present)
        self.q_u_head = nn.Linear(hidden_size, n_actions)

        if comm_mode == "rial":
            # Separate Q_m head for discrete message selection         [A]
            self.q_m_head = nn.Linear(hidden_size, n_messages)
        elif comm_mode == "dial":
            # Scalar continuous message output (DRU applied externally)[A]
            self.m_head = nn.Linear(hidden_size, msg_dim)
        # nocomm: no message output head

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: torch.Tensor,           # (batch, obs_dim)      — float
        prev_msg: torch.Tensor,      # (batch, msg_dim)      — float
        prev_action: torch.Tensor,   # (batch,)              — long
        agent_id: torch.Tensor,      # (batch,)              — long
        hidden: Optional[torch.Tensor] = None,  # (2, batch, hidden_size)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Parameters
        ----------
        obs          : task observation, shape (batch, obs_dim)
        prev_msg     : previous message received, shape (batch, msg_dim)
        prev_action  : previous environment action, shape (batch,)
        agent_id     : agent index, shape (batch,)
        hidden       : GRU hidden state, shape (n_layers, batch, hidden_size)
                       If None, zeros are used.                        [A]

        Returns
        -------
        q_u          : (batch, n_actions)   — action Q-values
        q_m / m_out  : (batch, n_messages) for RIAL,
                       (batch, msg_dim)    for DIAL,
                       None               for NoComm
        hidden_out   : (2, batch, hidden_size) — new GRU state
        """
        batch = obs.shape[0]

        # --- NoComm: zero out the message stream                      [C] ---
        if self.comm_mode == "nocomm":
            prev_msg = torch.zeros_like(prev_msg)

        # --- Embedding streams ------------------------------------------- #

        # Agent-index embedding                                         [A]
        e_agent = self.agent_embed(agent_id)                  # (B, 128)

        # Previous-action embedding                                     [A]
        e_action = self.action_embed(prev_action)             # (B, 128)

        # Previous-message embedding                                    [A/C]
        # BatchNorm expects (B, C) where C = msg_dim
        msg_normed = self.msg_bn(prev_msg.float())            # (B, msg_dim)
        e_msg = self.msg_mlp(msg_normed)                      # (B, 128)

        # Task-observation embedding                                    [A/C]
        e_obs = self.obs_net(obs.float())                     # (B, 128)

        # Element-wise sum                                              [A]
        z = e_agent + e_action + e_msg + e_obs                # (B, 128)

        # --- GRU --------------------------------------------------------- #
        if hidden is None:
            hidden = self.init_hidden(batch, obs.device)      # [A]

        # GRU expects (batch, seq_len, input_size) with batch_first=True
        z_seq = z.unsqueeze(1)                                # (B, 1, 128)
        gru_out, hidden_out = self.gru(z_seq, hidden)         # (B,1,128), (2,B,128)
        gru_feat = gru_out.squeeze(1)                         # (B, 128)

        # --- Output MLP -------------------------------------------------- #
        feat = torch.relu(self.out_fc1(gru_feat))             # (B, 128)  [A]

        q_u = self.q_u_head(feat)                             # (B, n_actions)

        if self.comm_mode == "rial":
            q_m = self.q_m_head(feat)                         # (B, n_messages)
            return q_u, q_m, hidden_out

        elif self.comm_mode == "dial":
            m_out = self.m_head(feat)                         # (B, msg_dim)
            return q_u, m_out, hidden_out

        else:  # nocomm
            return q_u, None, hidden_out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """
        Return zero initial hidden state.                              [A]
        Shape: (n_layers=2, batch_size, hidden_size)
        """
        return torch.zeros(
            2, batch_size, self.hidden_size, device=device
        )

    def copy_weights_from(self, other: "RNNAgent") -> None:
        """Hard-copy weights for target-network reset."""
        self.load_state_dict(other.state_dict())
