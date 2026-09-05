# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project


from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload import recompute_cpu_offload_connector as module
from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.metadata import (  # noqa: E402
    RecomputeCPUOffloadMetadata,
    RecomputeCPUOffloadWorkerMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.recompute_cpu_offload_connector import (  # noqa: E402
    RecomputeCPUOffloadConnectorV1,
)


def test_recompute_cpu_offload_connector_scheduler_methods_forward():
    connector = RecomputeCPUOffloadConnectorV1.__new__(RecomputeCPUOffloadConnectorV1)
    scheduler_manager = MagicMock()
    scheduler_manager.get_num_new_matched_tokens.return_value = (8, True)
    scheduler_manager.update_state_before_preempt.return_value = True
    scheduler_manager.has_pending_transfers.return_value = True
    scheduler_manager.has_preempted_request.return_value = True
    connector.scheduler_manager = scheduler_manager

    request = SimpleNamespace(request_id="req-1")
    blocks = MagicMock()
    block_ids = ([1, 2],)

    assert connector.get_num_new_matched_tokens(request, 4) == (8, True)
    connector.update_state_after_alloc(request, blocks, 8)
    assert connector.update_state_before_preempt(request, block_ids, 16) is True
    assert connector.has_pending_transfers() is True
    assert connector.has_preempted_request("req-1") is True

    scheduler_manager.get_num_new_matched_tokens.assert_called_once_with(request, 4)
    scheduler_manager.update_state_after_alloc.assert_called_once_with(request, blocks, 8)
    scheduler_manager.update_state_before_preempt.assert_called_once_with(request, block_ids, 16)


def test_recompute_cpu_offload_connector_worker_methods_forward():
    connector = RecomputeCPUOffloadConnectorV1.__new__(RecomputeCPUOffloadConnectorV1)
    worker_handler = MagicMock()
    worker_handler.get_finished.return_value = (None, {"req-1"})
    worker_handler.build_connector_worker_meta.return_value = RecomputeCPUOffloadWorkerMetadata(
        completed_store_events={3: 1}
    )
    connector.worker_handler = worker_handler

    metadata = RecomputeCPUOffloadMetadata(preempt_load_event=3)
    connector.bind_connector_metadata(metadata)
    connector.handle_preemptions(metadata)
    connector.start_load_kv(MagicMock())
    connector.wait_for_layer_load("layer.0")

    assert connector.get_finished(set()) == (None, {"req-1"})
    assert connector.build_connector_worker_meta().completed_store_events == {3: 1}

    worker_handler.bind_connector_metadata.assert_called_once_with(metadata)
    worker_handler.handle_preemptions.assert_called_once_with(metadata)
    worker_handler.start_load_kv.assert_called_once_with()
    worker_handler.wait_for_layer_load.assert_called_once_with()


def test_recompute_cpu_offload_connector_defaults_without_scheduler_manager():
    connector = RecomputeCPUOffloadConnectorV1.__new__(RecomputeCPUOffloadConnectorV1)
    connector.scheduler_manager = None

    assert connector.get_num_new_matched_tokens(MagicMock(), 0) == (0, False)
    assert connector.update_state_before_preempt(MagicMock(), ([],), 1) is False
    assert isinstance(
        connector.build_connector_meta(MagicMock()),
        RecomputeCPUOffloadMetadata,
    )
    assert connector.request_finished(MagicMock(), []) == (False, None)
    assert connector.request_finished_all_groups(MagicMock(), ([],)) == (
        False,
        None,
    )
    assert connector.has_pending_transfers() is False
    assert connector.has_preempted_request("req-1") is False
    assert connector.take_events() == []
    assert connector.reset_cache() is None


@pytest.mark.parametrize("role", [KVConnectorRole.WORKER, KVConnectorRole.SCHEDULER])
@pytest.mark.parametrize(
    ("extra", "capacity", "prefix"),
    [
        (None, 4 * 1024**3, False),
        ({"cpu_bytes_to_use": 32}, 16, False),
        ({"cpu_bytes_to_use": 32, "cpu_bytes_to_use_per_rank": 16}, 16, False),
        ({"cpu_bytes_to_use_per_rank": "8", "enable_offload_prefix_caching": True}, 8, True),
    ],
)
def test_constructor_selects_role_and_resolves_capacity(monkeypatch, role, extra, capacity, prefix):
    scheduler, worker = MagicMock(), MagicMock()
    monkeypatch.setattr(module, "RecomputeCPUOffloadScheduler", scheduler)
    monkeypatch.setattr(module, "RecomputeCPUOffloadWorker", worker)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        parallel_config=SimpleNamespace(world_size=2),
    )
    cache = SimpleNamespace()

    connector = RecomputeCPUOffloadConnectorV1(config, role, cache)

    assert connector.role == role
    if role == KVConnectorRole.SCHEDULER:
        scheduler.assert_called_once_with(config, cache, capacity, prefix)
        worker.assert_not_called()
        assert connector.scheduler_manager is scheduler.return_value
        assert connector.worker_handler is None
    else:
        worker.assert_called_once_with(config, cache, capacity)
        scheduler.assert_not_called()
        assert connector.worker_handler is worker.return_value
        assert connector.scheduler_manager is None


@pytest.mark.parametrize("value", ["true", 1, None])
def test_constructor_rejects_non_boolean_prefix_option(value):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"enable_offload_prefix_caching": value})
    )
    with pytest.raises(ValueError, match="must be a boolean"):
        RecomputeCPUOffloadConnectorV1(config, KVConnectorRole.SCHEDULER)


@pytest.mark.parametrize("present", [False, True])
@pytest.mark.parametrize(
    ("method", "args", "default"),
    [
        ("bind_gpu_block_pool", (object(),), None),
        ("update_state_after_alloc", (object(), object(), 3), None),
        ("update_connector_output", (object(),), None),
        ("build_connector_meta", (object(),), "metadata"),
        ("request_finished", (object(), [1]), (False, None)),
        ("request_finished_all_groups", (object(), ([1],)), (False, None)),
        ("take_events", (), []),
        ("reset_cache", (), None),
    ],
)
def test_scheduler_forwarding_and_missing_role_defaults(present, method, args, default):
    connector = RecomputeCPUOffloadConnectorV1.__new__(RecomputeCPUOffloadConnectorV1)
    handler = MagicMock()
    connector.scheduler_manager = handler if present else None
    result = getattr(connector, method)(*args)
    if present:
        getattr(handler, method).assert_called_once_with(*args)
        if method in {
            "build_connector_meta",
            "request_finished",
            "request_finished_all_groups",
            "take_events",
            "reset_cache",
        }:
            assert result is getattr(handler, method).return_value
    elif default == "metadata":
        assert result == RecomputeCPUOffloadMetadata()
    else:
        assert result == default


@pytest.mark.parametrize("present", [False, True])
def test_worker_registration_metadata_lifecycle_and_reserved_hooks(present):
    connector = RecomputeCPUOffloadConnectorV1.__new__(RecomputeCPUOffloadConnectorV1)
    handler = MagicMock()
    connector.worker_handler = handler if present else None
    metadata = RecomputeCPUOffloadMetadata()
    caches = {}
    connector.register_kv_caches(caches)
    connector.bind_connector_metadata(metadata)
    assert connector._connector_metadata is metadata
    connector.clear_connector_metadata()
    assert connector._connector_metadata is None
    connector.handle_preemptions(metadata)
    connector.start_load_kv(None)
    connector.wait_for_layer_load("a")
    assert connector.save_kv_layer("a", None, None) is None
    assert connector.wait_for_save() is None
    if present:
        handler.register_kv_caches.assert_called_once_with(caches)
        handler.clear_connector_metadata.assert_called_once_with()
    else:
        assert connector.get_finished(set()) == (None, None)
        assert connector.build_connector_worker_meta() is None
        assert handler.mock_calls == []
