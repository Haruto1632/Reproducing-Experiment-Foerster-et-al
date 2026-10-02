"""
tests/test_device.py
====================
Verifies that all model parameters, hidden states, and training tensors
are consistently placed on the selected device (CUDA when available,
otherwise CPU).

Run with:
    pytest research_reproduction/switch_riddle/tests/test_device.py -v
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch

from switch_riddle.agents.rnn_agent import RNNAgent
from switch_riddle.communication.dru import DRU
from switch_riddle.communication.rial import RIALController
from switch_riddle.communication.dial import DIALController
from switch_riddle.training.trainer import (
    Trainer, TrainingConfig, NoCommController, run_batch,
    _NSRIALController, _NSDIALController,
)
from switch_riddle.environment.switch_env import SwitchRiddleEnv


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 3
BATCH = 4


def make_agent(mode, device=DEVICE):
    return RNNAgent(
        n_agents=N, n_actions=2, n_messages=2,
        obs_dim=1, msg_dim=1, hidden_size=128, comm_mode=mode,
    ).to(device)


# ---------------------------------------------------------------------------
# 1. Model parameters on correct device
# ---------------------------------------------------------------------------

class TestModelDevice:
    @pytest.mark.parametrize("mode", ["rial", "dial", "nocomm"])
    def test_parameters_on_device(self, mode):
        agent = make_agent(mode)
        for name, p in agent.named_parameters():
            assert p.device.type == DEVICE.type, \
                f"Parameter '{name}' is on {p.device}, expected {DEVICE}"

    def test_target_network_on_device(self):
        agent = make_agent("rial")
        target = make_agent("rial")
        target.copy_weights_from(agent)
        for name, p in target.named_parameters():
            assert p.device.type == DEVICE.type, \
                f"Target param '{name}' on {p.device}, expected {DEVICE}"


# ---------------------------------------------------------------------------
# 2. Hidden state on correct device
# ---------------------------------------------------------------------------

class TestHiddenStateDevice:
    def test_init_hidden_on_device(self):
        agent = make_agent("rial")
        h = agent.init_hidden(BATCH, DEVICE)
        assert h.device.type == DEVICE.type, \
            f"init_hidden returned tensor on {h.device}, expected {DEVICE}"

    def test_gru_output_hidden_on_device(self):
        agent = make_agent("dial")
        obs = torch.zeros(BATCH, 1, device=DEVICE)
        prev_msg = torch.zeros(BATCH, 1, device=DEVICE)
        prev_action = torch.zeros(BATCH, dtype=torch.long, device=DEVICE)
        agent_id = torch.zeros(BATCH, dtype=torch.long, device=DEVICE)
        _, _, h_out = agent(obs, prev_msg, prev_action, agent_id, None)
        assert h_out.device.type == DEVICE.type, \
            f"GRU output hidden on {h_out.device}, expected {DEVICE}"


# ---------------------------------------------------------------------------
# 3. Training tensors on correct device
# ---------------------------------------------------------------------------

class TestTrainingTensorDevice:
    def _make_trainer(self, alg="nocomm", n=N):
        cfg = TrainingConfig(
            n_agents=n, algorithm=alg, param_sharing=True,
            max_epochs=2, batch_size=4,
            eval_every_epochs=2, eval_episodes=4,
            seed=0,
            log_dir=os.path.join(os.path.dirname(__file__), "..", "results"),
        )
        return Trainer(cfg)

    def test_trainer_device_is_cuda_when_available(self):
        trainer = self._make_trainer()
        assert trainer.device.type == DEVICE.type, \
            f"Trainer.device is {trainer.device}, expected {DEVICE}"

    def test_model_on_trainer_device(self):
        trainer = self._make_trainer()
        agent = trainer.controller.agent
        for name, p in agent.named_parameters():
            assert p.device.type == DEVICE.type, \
                f"Trainer model param '{name}' on {p.device}, expected {DEVICE}"

    def test_target_model_on_trainer_device(self):
        trainer = self._make_trainer()
        target_agent = trainer._target_ctrl.agent
        for name, p in target_agent.named_parameters():
            assert p.device.type == DEVICE.type, \
                f"Target param '{name}' on {p.device}, expected {DEVICE}"

    def test_run_batch_loss_on_device(self):
        """Loss tensor produced by run_batch must be on DEVICE."""
        trainer = self._make_trainer("nocomm")
        loss, _, _ = run_batch(
            trainer._train_envs, trainer.controller,
            trainer.config, training=True, device=DEVICE,
        )
        assert loss.device.type == DEVICE.type, \
            f"Loss on {loss.device}, expected {DEVICE}"

    def test_full_train_step_no_cpu_gpu_mismatch(self):
        """
        A full 2-epoch training run must complete without RuntimeError
        caused by device mismatches.
        """
        trainer = self._make_trainer("dial")
        # Should not raise
        trainer.train()


# ---------------------------------------------------------------------------
# 4. DRU tensors on correct device
# ---------------------------------------------------------------------------

class TestDRUDevice:
    def test_dru_output_on_same_device_as_input(self):
        dru = DRU(sigma=2.0)
        m = torch.randn(8, 1, device=DEVICE)
        out_train = dru(m, training=True)
        out_eval  = dru(m, training=False)
        assert out_train.device.type == DEVICE.type
        assert out_eval.device.type  == DEVICE.type


# ---------------------------------------------------------------------------
# 5. Device startup report
# ---------------------------------------------------------------------------

class TestStartupLog:
    def test_device_report(self, capsys):
        """Verify the startup print includes 'Device:' and GPU name."""
        from switch_riddle.training.trainer import print_device_info
        print_device_info()
        captured = capsys.readouterr()
        assert "Device:" in captured.out
        if DEVICE.type == "cuda":
            assert "GPU:" in captured.out
