# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def worker(monkeypatch):
    from vllm_ascend.worker import worker as worker_module

    worker = worker_module.NPUWorker.__new__(worker_module.NPUWorker)
    worker.model_config = SimpleNamespace(
        sleep_preserve_parameter_names=["weight"],
        sleep_mode_backend="camem",
    )
    worker.vllm_config = SimpleNamespace(model_config=worker.model_config)
    worker.model_runner = SimpleNamespace(
        model=torch.nn.Linear(2, 2),
        get_draft_model=lambda: None,
    )
    from vllm_ascend.device_allocator.sleep_mode_backend import CaMemBackend

    worker._extra_cleanup_pending = False
    worker._sleep_mode_backend = MagicMock(spec=CaMemBackend)
    worker.sleep_wakeup_manager = MagicMock()
    monkeypatch.setattr(worker, "synchronize_device", MagicMock())
    monkeypatch.setattr(worker_module.global_te, "unregister_buffer", MagicMock())
    monkeypatch.setattr(
        worker_module.torch.npu,
        "mem_get_info",
        MagicMock(side_effect=[(100, 200), (150, 200)]),
    )
    monkeypatch.setattr(
        worker_module,
        "get_ascend_config",
        lambda: SimpleNamespace(
            weight_nz_mode=0,
            rl_config=SimpleNamespace(enabled=False, sleep_mode_extra_cleanup=False),
        ),
    )
    return worker


def test_unmatched_pattern_fails_before_sleep(worker):
    worker.model_config.sleep_preserve_parameter_names = ["missing.*"]

    with pytest.raises(ValueError, match="No parameter matches sleep retention: missing"):
        worker.sleep(level=2)

    worker.sleep_mode_backend.suspend.assert_not_called()


def test_level_one_passes_empty_discardable(worker):
    worker.sleep(level=1)

    worker.sleep_mode_backend.suspend.assert_called_once_with(set())


def test_level_two_passes_discardable_storages(worker):
    model = worker.model_runner.model
    expected = worker._discardable_parameters(model)

    worker.sleep(level=2)

    worker.sleep_mode_backend.suspend.assert_called_once_with(expected)
