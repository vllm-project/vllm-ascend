# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.eplb.eplb_state import AscendEplbState
from vllm_ascend.worker import worker as worker_module


def _worker(monkeypatch):
    state = MagicMock(spec=AscendEplbState)
    state.model_states = {
        "model": SimpleNamespace(
            communicator=SimpleNamespace(_registered_regions=[(0, 4096)]),
            model=SimpleNamespace(set_eplb_state=MagicMock()),
            expert_load_pass_buffer=object(),
            logical_to_physical_map=object(),
            logical_replica_count=object(),
        )
    }
    worker = worker_module.NPUWorker.__new__(worker_module.NPUWorker)
    worker.model_runner = SimpleNamespace(eplb_state=state)
    allocator = MagicMock()
    allocator.pointer_to_data = {
        0: SimpleNamespace(tag="weights", handle=(0, 1024, 0, 0)),
        1024: SimpleNamespace(tag="kv_cache", handle=(0, 1024, 1024, 0)),
        2048: SimpleNamespace(tag="persistent", handle=(0, 1024, 2048, 0)),
    }
    monkeypatch.setattr(worker_module.CaMemAllocator, "get_instance", lambda: allocator)
    monkeypatch.setattr(worker_module.CaMemAllocator, "sleep_persistent_tag", "persistent")
    monkeypatch.setattr(torch.npu, "mem_get_info", lambda: (0, 4096))
    monkeypatch.setattr(
        worker_module,
        "get_ascend_config",
        lambda: SimpleNamespace(
            weight_nz_mode=0, rl_config=SimpleNamespace(enabled=False, sleep_mode_extra_cleanup=False)
        ),
    )
    return worker, state, allocator


def test_sleep_closes_before_unmap_and_partial_wake_waits_for_all_registered_tags(monkeypatch):
    worker, state, allocator = _worker(monkeypatch)
    allocator.sleep.side_effect = lambda **_kwargs: state.close.assert_called_once_with()
    worker.sleep()
    assert worker._eplb_pending_wake_tags == {"weights", "kv_cache"}
    worker.wake_up(tags=["weights"])
    state.resume.assert_not_called()
    worker.wake_up(tags=["kv_cache"])
    state.resume.assert_called_once_with()
    assert not hasattr(worker, "_eplb_pending_wake_tags")


def test_failed_close_blocks_sleep_unmap(monkeypatch):
    worker, state, allocator = _worker(monkeypatch)
    state.close.side_effect = RuntimeError("still bound")
    with pytest.raises(RuntimeError, match="still bound"):
        worker.sleep()
    allocator.sleep.assert_not_called()


def test_failed_close_blocks_wake_before_remap(monkeypatch):
    worker, state, allocator = _worker(monkeypatch)
    state.raise_if_close_failed.side_effect = RuntimeError("worker must terminate")
    with pytest.raises(RuntimeError, match="worker must terminate"):
        worker.wake_up()
    allocator.wake_up.assert_not_called()


def test_reload_weights_closes_old_registration_before_rebinding(monkeypatch):
    worker, state, _ = _worker(monkeypatch)
    worker.model_runner.reload_weights = MagicMock(side_effect=lambda **_kwargs: state.close.assert_called_once_with())
    worker.reload_weights(weights_path="replacement")
    worker.model_runner.reload_weights.assert_called_once_with(weights_path="replacement")
    ms = state.model_states["model"]
    ms.model.set_eplb_state.assert_called_once_with(
        ms.expert_load_pass_buffer, ms.logical_to_physical_map, ms.logical_replica_count
    )
    state.resume.assert_called_once_with()


def test_failed_reload_does_not_resume_migration(monkeypatch):
    worker, state, _ = _worker(monkeypatch)
    worker.model_runner.reload_weights = MagicMock(side_effect=RuntimeError("load failed"))
    with pytest.raises(RuntimeError, match="load failed"):
        worker.reload_weights()
    state.close.assert_called_once_with()
    state.resume.assert_not_called()
