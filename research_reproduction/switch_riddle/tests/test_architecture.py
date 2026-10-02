"""
Unit Tests — Architecture, Communication, and Training Mechanics (Phases 2–5)
==============================================================================
Run with:
    pytest research_reproduction/switch_riddle/tests/test_architecture.py -v
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch
import torch.nn as nn

from switch_riddle.agents.rnn_agent import RNNAgent
from switch_riddle.communication.dru import DRU
from switch_riddle.communication.rial import RIALController
from switch_riddle.communication.dial import DIALController
from switch_riddle.training.trainer import NoCommController, TrainingConfig, Trainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_AGENTS = 3
N_ACTIONS = 2
N_MESSAGES = 2
BATCH = 4
HIDDEN = 128


def make_agent(mode="rial", n=N_AGENTS):
    return RNNAgent(
        n_agents=n, n_actions=N_ACTIONS, n_messages=N_MESSAGES,
        obs_dim=1, msg_dim=1, hidden_size=HIDDEN, comm_mode=mode
    )


def dummy_inputs(B=BATCH, device="cpu"):
    obs = torch.randint(0, 2, (B, 1)).float()
    prev_msg = torch.zeros(B, 1)
    prev_action = torch.zeros(B, dtype=torch.long)
    agent_id = torch.zeros(B, dtype=torch.long)
    return obs, prev_msg, prev_action, agent_id


# ===========================================================================
# Phase 2 — Shared Architecture
# ===========================================================================

class TestRNNAgentShape:
    """Verify output shapes for each comm_mode."""

    def test_rial_output_shapes(self):
        agent = make_agent("rial")
        obs, prev_msg, prev_action, agent_id = dummy_inputs()
        q_u, q_m, hidden = agent(obs, prev_msg, prev_action, agent_id)
        assert q_u.shape == (BATCH, N_ACTIONS)
        assert q_m.shape == (BATCH, N_MESSAGES)
        assert hidden.shape == (2, BATCH, HIDDEN)

    def test_dial_output_shapes(self):
        agent = make_agent("dial")
        obs, prev_msg, prev_action, agent_id = dummy_inputs()
        q_u, m_raw, hidden = agent(obs, prev_msg, prev_action, agent_id)
        assert q_u.shape == (BATCH, N_ACTIONS)
        assert m_raw.shape == (BATCH, 1)           # scalar continuous message
        assert hidden.shape == (2, BATCH, HIDDEN)

    def test_nocomm_output_shapes(self):
        agent = make_agent("nocomm")
        obs, prev_msg, prev_action, agent_id = dummy_inputs()
        q_u, m_out, hidden = agent(obs, prev_msg, prev_action, agent_id)
        assert q_u.shape == (BATCH, N_ACTIONS)
        assert m_out is None
        assert hidden.shape == (2, BATCH, HIDDEN)

    def test_invalid_comm_mode_raises(self):
        with pytest.raises(AssertionError):
            RNNAgent(n_agents=3, n_actions=2, n_messages=2,
                     obs_dim=1, msg_dim=1, comm_mode="ppo")


class TestHiddenStateInit:
    """h_0 = zeros  [A per Alg 1]"""

    def test_init_hidden_zeros(self):
        agent = make_agent("rial")
        h = agent.init_hidden(BATCH, torch.device("cpu"))
        assert h.shape == (2, BATCH, HIDDEN)
        assert h.sum().item() == 0.0

    def test_none_hidden_uses_zeros(self):
        agent = make_agent("rial")
        obs, prev_msg, prev_action, agent_id = dummy_inputs()
        # Pass hidden=None; should not raise and should use zeros internally
        q_u, q_m, h_out = agent(obs, prev_msg, prev_action, agent_id, hidden=None)
        assert q_u is not None


class TestNoCommZerosMessage:
    """NoComm must zero out the message stream  [C]"""

    def test_nocomm_ignores_nonzero_message(self):
        agent = make_agent("nocomm")
        obs, _, prev_action, agent_id = dummy_inputs()

        # With a non-zero message
        msg_nonzero = torch.ones(BATCH, 1)
        q_u_nonzero, _, _ = agent(obs, msg_nonzero, prev_action, agent_id)

        # Re-init and feed zero message — output should be same because
        # nocomm zeroes the message internally.
        # (We need a fresh identical agent to compare deterministically)
        agent2 = make_agent("nocomm")
        agent2.load_state_dict(agent.state_dict())
        msg_zero = torch.zeros(BATCH, 1)
        q_u_zero, _, _ = agent2(obs, msg_zero, prev_action, agent_id)

        # Both should produce identical Q-values since message is zeroed
        assert torch.allclose(q_u_nonzero, q_u_zero), \
            "NoComm must zero the message; non-zero input must not affect output."


class TestCopyWeights:
    """Target network hard-copy."""

    def test_copy_weights_identical(self):
        src = make_agent("rial")
        tgt = make_agent("rial")
        tgt.copy_weights_from(src)
        for p_src, p_tgt in zip(src.parameters(), tgt.parameters()):
            assert torch.allclose(p_src, p_tgt)

    def test_copy_weights_independent(self):
        src = make_agent("rial")
        tgt = make_agent("rial")
        tgt.copy_weights_from(src)
        # Modify src — tgt should not change
        with torch.no_grad():
            for p in src.parameters():
                p.add_(1.0)
        for p_src, p_tgt in zip(src.parameters(), tgt.parameters()):
            assert not torch.allclose(p_src, p_tgt)


# ===========================================================================
# Phase 3 — RIAL
# ===========================================================================

class TestRIALController:
    def test_separate_action_message_selection(self):
        """RIAL selects u and m independently  [A]"""
        agent = make_agent("rial")
        ctrl = RIALController(agent, n_messages=N_MESSAGES, epsilon=0.05)
        obs, prev_msg, prev_action, agent_id = dummy_inputs()
        q_u, q_m, _ = ctrl.step(obs, prev_msg, prev_action, agent_id, None)
        actions, messages = ctrl.select(q_u, q_m, training=True)
        assert actions.shape == (BATCH,)
        assert messages.shape == (BATCH,)
        # Actions and messages are selected independently — they can differ
        # (this is the whole point of separate ε-greedy)  [A]

    def test_greedy_at_eval(self):
        """ε=0 during evaluation → always greedy"""
        agent = make_agent("rial")
        ctrl = RIALController(agent, n_messages=N_MESSAGES, epsilon=0.05)
        # Force a very clear greedy choice
        q_u = torch.zeros(BATCH, N_ACTIONS)
        q_u[:, 0] = 1000.0   # action 0 dominates
        q_m = torch.zeros(BATCH, N_MESSAGES)
        q_m[:, 1] = 1000.0   # message 1 dominates

        acts, msgs = ctrl.select(q_u, q_m, training=False)  # eval → ε=0
        assert (acts == 0).all(), "Greedy should always pick action 0"
        assert (msgs == 1).all(), "Greedy should always pick message 1"

    def test_epsilon_greedy_explores(self):
        """With ε=1 all actions should be random (not always argmax)"""
        # Monkeypatch epsilon to 1.0
        agent = make_agent("rial")
        ctrl = RIALController(agent, n_messages=N_MESSAGES, epsilon=1.0)
        q_u = torch.zeros(100, N_ACTIONS)
        q_u[:, 0] = 1000.0   # greedy always 0
        acts, _ = ctrl.select(q_u, q_u, training=True)
        # With ε=1, should get some non-zero actions
        assert not (acts == 0).all(), "ε=1 must force random exploration"

    def test_rial_no_gradient_through_message(self):
        """
        RIAL treats messages as discrete actions — the message selection
        is not differentiable (argmax / randint).                    [A]
        """
        agent = make_agent("rial")
        ctrl = RIALController(agent, n_messages=N_MESSAGES, epsilon=0.0)
        obs, prev_msg, prev_action, agent_id = dummy_inputs()
        q_u, q_m, _ = ctrl.step(obs, prev_msg, prev_action, agent_id, None)
        acts, msgs = ctrl.select(q_u, q_m, training=False)
        # msgs should not carry grad (it's an integer index from argmax)
        assert not msgs.requires_grad


# ===========================================================================
# Phase 4 — DRU + DIAL
# ===========================================================================

class TestDRU:
    def test_train_mode_output_range(self):
        """During training, logistic output ∈ (0,1)  [A]"""
        dru = DRU(sigma=2.0)
        m = torch.randn(1000, 1)
        out = dru(m, training=True)
        assert (out > 0).all() and (out < 1).all(), \
            "Logistic output must be in (0,1) during training."

    def test_eval_mode_binary(self):
        """During eval, output ∈ {0.0, 1.0}  [A]"""
        dru = DRU(sigma=2.0)
        m = torch.randn(100, 1)
        out = dru(m, training=False)
        unique_vals = out.unique().tolist()
        assert set(unique_vals).issubset({0.0, 1.0}), \
            "Eval DRU must produce binary values only."

    def test_eval_threshold_at_zero(self):
        """1{m > 0}: positive → 1, negative → 0  [A]"""
        dru = DRU(sigma=2.0)
        m_pos = torch.tensor([[1.0], [0.1], [100.0]])
        m_neg = torch.tensor([[-1.0], [-0.1], [-100.0]])
        assert (dru(m_pos, training=False) == 1.0).all()
        assert (dru(m_neg, training=False) == 0.0).all()

    def test_sigma_controls_noise_scale(self):
        """Larger sigma → more variance in training output  [A]"""
        torch.manual_seed(0)
        dru_small = DRU(sigma=0.001)
        dru_large = DRU(sigma=100.0)
        m = torch.zeros(10000, 1)
        out_small = dru_small(m, training=True)
        out_large = dru_large(m, training=True)
        assert out_large.var() > out_small.var(), \
            "Larger sigma must produce higher output variance."

    def test_train_mode_is_differentiable(self):
        """Gradient must flow through DRU sigmoid during training  [A]"""
        dru = DRU(sigma=0.0)   # no noise to isolate the gradient test
        m = torch.randn(4, 1, requires_grad=True)
        out = dru(m, training=True)
        out.sum().backward()
        assert m.grad is not None, "Gradient must flow through DRU in training mode."

    def test_eval_mode_not_differentiable(self):
        """Eval DRU uses hard threshold — no gradient  [A]"""
        dru = DRU(sigma=2.0)
        m = torch.randn(4, 1)
        out = dru(m, training=False)
        assert not out.requires_grad


class TestDIALController:
    def test_step_output_shapes(self):
        agent = make_agent("dial")
        dru = DRU(sigma=2.0)
        ctrl = DIALController(agent, dru, epsilon=0.05)
        obs, prev_msg, prev_action, agent_id = dummy_inputs()
        q_u, msg_out, hidden = ctrl.step(
            obs, prev_msg, prev_action, agent_id, None, training=True
        )
        assert q_u.shape == (BATCH, N_ACTIONS)
        assert msg_out.shape == (BATCH, 1)
        assert hidden.shape == (2, BATCH, HIDDEN)

    def test_training_msg_continuous(self):
        """Training: DRU output is continuous (logistic)  [A]"""
        agent = make_agent("dial")
        dru = DRU(sigma=2.0)
        ctrl = DIALController(agent, dru, epsilon=0.0)
        obs, prev_msg, prev_action, agent_id = dummy_inputs(B=100)
        _, msg_out, _ = ctrl.step(
            obs, prev_msg, prev_action, agent_id, None, training=True
        )
        # Continuous: not all values should be 0 or 1
        unique = msg_out.detach().unique()
        assert len(unique) > 2, "Training message must be continuous, not binary."

    def test_eval_msg_binary(self):
        """Eval: DRU output is binary  [A]"""
        agent = make_agent("dial")
        dru = DRU(sigma=2.0)
        ctrl = DIALController(agent, dru, epsilon=0.0)
        obs, prev_msg, prev_action, agent_id = dummy_inputs(B=100)
        _, msg_out, _ = ctrl.step(
            obs, prev_msg, prev_action, agent_id, None, training=False
        )
        unique = set(msg_out.detach().numpy().flatten().tolist())
        assert unique.issubset({0.0, 1.0}), "Eval message must be binary."

    def test_gradient_flows_through_message(self):
        """
        Gradient must flow from Q-loss backward through the continuous
        message to the sender's parameters.                           [A]
        """
        agent_sender = make_agent("dial")
        agent_receiver = make_agent("dial")
        dru = DRU(sigma=0.0)   # no noise — purely logistic

        obs, prev_msg, prev_action, agent_id = dummy_inputs()

        # Sender forward pass — produces continuous message
        ctrl_sender = DIALController(agent_sender, dru, epsilon=0.0)
        q_u_s, msg_out, _ = ctrl_sender.step(
            obs, prev_msg, prev_action, agent_id, None, training=True
        )

        # msg_out fed to receiver as prev_msg
        q_u_r, _, _ = agent_receiver(
            obs, msg_out, prev_action, agent_id, None
        )

        # Loss on receiver's Q-values
        target = torch.zeros_like(q_u_r)
        loss = (q_u_r - target).pow(2).mean()
        loss.backward()

        # Check that sender's parameters received gradients
        sender_grads = [p.grad for p in agent_sender.parameters()
                        if p.grad is not None]
        assert len(sender_grads) > 0, \
            "Gradient must backprop through the message channel to the sender."


# ===========================================================================
# Phase 5 — NoComm
# ===========================================================================

class TestNoCommController:
    def test_nocomm_requires_nocomm_mode(self):
        agent = make_agent("rial")
        with pytest.raises(AssertionError):
            NoCommController(agent)

    def test_nocomm_forward(self):
        agent = make_agent("nocomm")
        ctrl = NoCommController(agent)
        obs, prev_msg, prev_action, agent_id = dummy_inputs()
        q_u, m_out, hidden = agent(obs, prev_msg, prev_action, agent_id, None)
        assert m_out is None
        assert q_u.shape == (BATCH, N_ACTIONS)


# ===========================================================================
# Training mechanics — counter and target-network tests
# ===========================================================================

class TestTrainingCounters:
    """Verify counters are tracked correctly during a short training run."""

    def _make_config(self, alg="nocomm", n=3, epochs=5, seed=0):
        return TrainingConfig(
            n_agents=n,
            algorithm=alg,
            param_sharing=True,
            max_epochs=epochs,
            batch_size=32,
            eval_every_epochs=5,
            eval_episodes=32,
            seed=seed,
            log_dir=os.path.join(
                os.path.dirname(__file__), "..", "results"
            ),
        )

    def test_completed_episodes_count(self):
        cfg = self._make_config(epochs=3)
        trainer = Trainer(cfg)
        trainer.train()
        c = trainer.counters
        # 3 epochs × 32 parallel = 96 completed episodes
        assert c.completed_episodes == 3 * 32

    def test_update_steps_count(self):
        cfg = self._make_config(epochs=4)
        trainer = Trainer(cfg)
        trainer.train()
        assert trainer.counters.update_steps == 4

    def test_plotted_x_axis_equals_epochs(self):
        cfg = self._make_config(epochs=6)
        trainer = Trainer(cfg)
        trainer.train()
        assert trainer.counters.plotted_x_axis == 6

    def test_target_update_fires_after_100_episodes(self):
        """
        Target network must reset when completed_episodes crosses 100.
        With batch_size=32 this happens at epoch 4 (128 episodes ≥ 100). [A/C]
        """
        cfg = self._make_config(epochs=5)
        trainer = Trainer(cfg)
        trainer.train()
        c = trainer.counters
        assert c.target_update_count >= 1, \
            "Target network must have been reset at least once in 5*32=160 episodes."
        assert c.last_target_update_episode >= 100

    def test_env_timesteps_positive(self):
        cfg = self._make_config(epochs=2)
        trainer = Trainer(cfg)
        trainer.train()
        assert trainer.counters.env_timesteps > 0
