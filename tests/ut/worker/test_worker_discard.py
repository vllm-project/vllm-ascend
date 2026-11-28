#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""NPUWorker.discard, the worker half of LLM.release_kv_cache_memory()."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest
from vllm.v1.worker import gpu_worker

from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import GlobalTE
from vllm_ascend.worker import worker as worker_module
from vllm_ascend.worker.worker import NPUWorker


@pytest.fixture
def calls():
    """One mock that records the device, Mooncake and backend calls in order."""
    return MagicMock()


@pytest.fixture
def worker(monkeypatch, calls):
    monkeypatch.setattr(gpu_worker, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(worker_module, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(worker_module.torch.npu, "mem_get_info", lambda: (0, 1), raising=False)
    monkeypatch.setattr(worker_module, "global_te", calls.global_te)
    monkeypatch.setattr(NPUWorker, "synchronize_device", lambda self: calls.synchronize_device())
    set_extra_cleanup(monkeypatch, enabled=False)

    worker = object.__new__(NPUWorker)
    worker._sleep_mode_backend = calls.backend
    worker._extra_cleanup_pending = False
    worker.sleep_wakeup_manager = calls.sleep_wakeup_manager
    worker.model_runner = SimpleNamespace(model=Mock(named_buffers=Mock(return_value=[])))
    return worker


def set_extra_cleanup(monkeypatch, enabled: bool, *, weight_nz_mode: int = 0) -> None:
    config = SimpleNamespace(
        weight_nz_mode=weight_nz_mode,
        rl_config=SimpleNamespace(enabled=enabled, sleep_mode_extra_cleanup=enabled),
    )
    monkeypatch.setattr(worker_module, "get_ascend_config", lambda: config)


def names(calls) -> list[str]:
    return [name for name, _, _ in calls.mock_calls]


def test_discarding_the_kv_cache_unregisters_mooncake_before_releasing_it(worker, calls):
    worker.discard(("kv_cache",))

    # Mooncake hands these addresses to the NIC, so a remote peer could read
    # them after the release if the unregister came second.
    assert names(calls) == ["synchronize_device", "global_te.unregister_buffer", "backend.discard"]
    calls.backend.discard.assert_called_once_with(("kv_cache",))


def test_discarding_other_tags_leaves_mooncake_registered(worker, calls):
    worker.discard(("weights",))

    assert names(calls) == ["backend.discard"]


def test_repeated_release_unregisters_once_and_kv_wake_registers_again(monkeypatch, worker):
    engine = Mock(register_memory=Mock(return_value=0), unregister_memory=Mock(return_value=0))
    transfer = GlobalTE()
    transfer.transfer_engine = engine
    transfer.register_buffer([0x1000], [64])
    monkeypatch.setattr(worker_module, "global_te", transfer)

    worker.discard(("kv_cache",))
    worker.discard(("kv_cache",))
    engine.unregister_memory.assert_called_once_with(0x1000)

    worker.wake_up(["kv_cache"])
    assert engine.register_memory.call_count == 2


def test_kv_wake_after_discard_skips_extra_cleanup(monkeypatch, worker, calls):
    set_extra_cleanup(monkeypatch, enabled=True)

    worker.discard(("kv_cache",))
    worker.wake_up(["kv_cache"])

    # A discard tears down no HCCL group and no ACL graph, so waking must not
    # restore groups or recapture graphs.
    calls.sleep_wakeup_manager.wakeup.assert_not_called()
    calls.backend.resume.assert_called_once_with(["kv_cache"])
    calls.global_te.reregister_buffer.assert_called_once_with()


def test_sleep_after_discard_runs_extra_cleanup_on_wake(monkeypatch, worker, calls):
    set_extra_cleanup(monkeypatch, enabled=True)

    worker.discard(("kv_cache",))
    worker.sleep(level=1)
    worker.wake_up()

    calls.sleep_wakeup_manager.sleep.assert_called_once_with()
    calls.sleep_wakeup_manager.wakeup.assert_called_once_with(None)


def test_kv_wake_recovers_a_discard_that_failed_part_way(worker, calls):
    calls.backend.discard.side_effect = RuntimeError("discard failed")

    with pytest.raises(RuntimeError, match="discard failed"):
        worker.discard(("kv_cache",))
    # The executor marks kv_cache asleep anyway; its wake-up remaps whatever
    # the allocator released and registers Mooncake again.
    worker.wake_up(["kv_cache"])

    calls.backend.resume.assert_called_once_with(["kv_cache"])
    calls.global_te.reregister_buffer.assert_called_once_with()


def test_extra_cleanup_stays_pending_when_the_teardown_raises(monkeypatch, worker, calls):
    set_extra_cleanup(monkeypatch, enabled=True)
    calls.sleep_wakeup_manager.sleep.side_effect = RuntimeError("HCCL teardown failed")

    with pytest.raises(RuntimeError, match="HCCL teardown failed"):
        worker.sleep(level=1)
    worker.wake_up()

    # Keep the worker-side pending flag set when teardown raises part-way so
    # a subsequent worker.wake_up() restores the destroyed groups.
    calls.sleep_wakeup_manager.wakeup.assert_called_once_with(None)


def test_weights_wake_leaves_the_restore_pending_for_the_kv_wake(monkeypatch, worker, calls):
    set_extra_cleanup(monkeypatch, enabled=True)

    worker.sleep(level=1)
    worker.wake_up(["weights"])
    assert worker._extra_cleanup_pending
    worker.wake_up(["kv_cache"])

    # The weights wake restores HCCL; graphs are recaptured on the KV wake.
    assert calls.sleep_wakeup_manager.wakeup.call_args_list == [call(["weights"]), call(["kv_cache"])]
    assert not worker._extra_cleanup_pending


def test_wake_without_a_sleep_skips_extra_cleanup(monkeypatch, worker, calls):
    set_extra_cleanup(monkeypatch, enabled=True)

    worker.wake_up()

    calls.sleep_wakeup_manager.wakeup.assert_not_called()


def test_sleep_and_kv_wake_pair_the_kv_connector_hooks(monkeypatch, worker, calls):
    monkeypatch.setattr(worker_module, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(worker_module, "get_kv_transfer_group", lambda: calls.connector)

    worker.sleep(level=1)
    assert names(calls) == [
        "synchronize_device",
        "synchronize_device",
        "global_te.unregister_buffer",
        "connector.unregister_kv_caches",
        "backend.suspend",
    ]
    calls.reset_mock()

    worker.wake_up(["weights"])
    calls.connector.restore_kv_caches.assert_not_called()
    calls.global_te.reregister_buffer.assert_not_called()
    calls.reset_mock()

    worker.wake_up(["kv_cache"])
    assert names(calls) == [
        "backend.resume",
        "connector.restore_kv_caches",
        "global_te.reregister_buffer",
        "synchronize_device",
    ]


def test_discard_and_kv_wake_align_mooncake_and_kv_connector_order(monkeypatch, worker, calls):
    monkeypatch.setattr(gpu_worker, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(gpu_worker, "get_kv_transfer_group", lambda: calls.connector)
    monkeypatch.setattr(worker_module, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(worker_module, "get_kv_transfer_group", lambda: calls.connector)

    worker.discard(("kv_cache",))
    assert names(calls) == [
        "synchronize_device",
        "global_te.unregister_buffer",
        "connector.unregister_kv_caches",
        "backend.discard",
    ]
    calls.reset_mock()

    worker.wake_up(["kv_cache"])
    assert names(calls) == [
        "backend.resume",
        "connector.restore_kv_caches",
        "global_te.reregister_buffer",
        "synchronize_device",
    ]


def test_kv_only_release_and_wake_work_with_nz_enabled(monkeypatch, worker, calls):
    set_extra_cleanup(monkeypatch, enabled=False, weight_nz_mode=1)

    worker.discard(("kv_cache",))
    worker.wake_up(["kv_cache"])

    assert names(calls) == [
        "synchronize_device",
        "global_te.unregister_buffer",
        "backend.discard",
        "backend.resume",
        "global_te.reregister_buffer",
        "synchronize_device",
    ]
    calls.backend.discard.assert_called_once_with(("kv_cache",))
    calls.backend.resume.assert_called_once_with(["kv_cache"])


def test_weights_discard_rejects_nz_before_changing_state(monkeypatch, worker, calls):
    set_extra_cleanup(monkeypatch, enabled=False, weight_nz_mode=1)

    with pytest.raises(ValueError, match="FRACTAL_NZ"):
        worker.discard(("weights", "kv_cache"))

    assert names(calls) == []


@pytest.mark.parametrize("tags", [None, ["weights"]])
def test_weights_wake_still_rejects_nz(monkeypatch, worker, calls, tags):
    set_extra_cleanup(monkeypatch, enabled=False, weight_nz_mode=1)

    with pytest.raises(ValueError, match="FRACTAL_NZ"):
        worker.wake_up(tags)

    assert names(calls) == []
