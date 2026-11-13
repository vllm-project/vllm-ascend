# SPDX-License-Identifier: Apache-2.0
"""MRV2 graph-worker lifecycle contract; native worker is mocked here."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.worker.v2 import aclgraph_utils
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def worker(calls):
    def capture_status(error):
        if error is not None:
            raise error

    return SimpleNamespace(
        sparse_kv_offload_config=SimpleNamespace(selective_offload=True),
        _selective_native_experiment=True,
        mark_selective_native_failure=Mock(),
        start_selective_native_graphs=lambda graphs: calls.append("start"),
        check_selective_native_graph_step=lambda desc: None,
        check_selective_native_capture_status=capture_status,
        check_selective_native_phase_status=lambda phase, error: capture_status(error),
        begin_selective_native_graph=lambda graph, anchor: None,
        finish_selective_native_graph=lambda graph, epoch, anchor: None,
        stop_selective_native_graphs=lambda: calls.append("quiesce"),
        release_selective_native_graphs=lambda **kwargs: calls.append("release"),
    )


def graph_manager(offload):
    result = aclgraph_utils.ModelAclGraphManager.__new__(aclgraph_utils.ModelAclGraphManager)
    result.model_runner = SimpleNamespace(sparse_kv_offload_manager=offload)
    result.graphs = {}
    return result


def capture(manager, metadata=None):
    return manager.capture(torch.nn.Identity(), SimpleNamespace(device_metadata=metadata), None, None, None, [], None)


def test_capture_success_starts_worker_after_parent(monkeypatch):
    calls = []
    manager = graph_manager(worker(calls))
    monkeypatch.setattr(aclgraph_utils, "communicator_switch", nullcontext)
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "capture", lambda *a, **k: calls.append("capture"))
    capture(manager)
    assert calls == ["capture", "start"]


def test_capture_failure_quarantines_partial_graphs(monkeypatch):
    calls = []
    manager = graph_manager(worker(calls))
    monkeypatch.setattr(aclgraph_utils, "communicator_switch", nullcontext)

    def fail(*args, **kwargs):
        calls.append("capture")
        raise RuntimeError("capture failed")

    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "capture", fail)
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "release_graphs", lambda self: calls.append("drop"))
    with pytest.raises(RuntimeError, match="capture failed"):
        capture(manager)
    assert calls == ["capture"]


def test_start_failure_keeps_graph_resources_without_hiding_error(monkeypatch):
    calls = []
    offload = worker(calls)

    def fail(graphs):
        calls.append("start")
        raise RuntimeError("start failed")

    offload.start_selective_native_graphs = fail
    manager = graph_manager(offload)
    monkeypatch.setattr(aclgraph_utils, "communicator_switch", nullcontext)
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "capture", lambda *a, **k: calls.append("capture"))
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "release_graphs", lambda self: calls.append("drop"))
    with pytest.raises(RuntimeError, match="start failed"):
        capture(manager)
    assert calls == ["capture", "start"]


def test_capture_failure_still_retires_metadata(monkeypatch):
    calls = []
    manager = graph_manager(worker(calls))
    metadata = SimpleNamespace(finish=lambda: calls.append("finish_metadata"))
    monkeypatch.setattr(aclgraph_utils, "communicator_switch", nullcontext)

    def fail(*args, **kwargs):
        calls.append("capture")
        raise RuntimeError("capture failed")

    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "capture", fail)
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "release_graphs", lambda self: calls.append("drop"))
    with pytest.raises(RuntimeError, match="capture failed"):
        capture(manager, metadata)
    assert calls.count("finish_metadata") == 1
    assert calls == ["capture", "finish_metadata"]


def test_metadata_finish_failure_keeps_worker_and_graph_resources(monkeypatch):
    calls = []
    manager = graph_manager(worker(calls))

    def fail():
        calls.append("finish_metadata")
        raise RuntimeError("metadata finish failed")

    monkeypatch.setattr(aclgraph_utils, "communicator_switch", nullcontext)
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "capture", lambda *a, **k: calls.append("capture"))
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "release_graphs", lambda self: calls.append("drop"))
    with pytest.raises(RuntimeError, match="metadata finish failed"):
        capture(manager, SimpleNamespace(finish=fail))
    assert calls.count("finish_metadata") == 1
    assert calls == ["capture", "start", "finish_metadata"]


def test_release_graphs_quiesces_then_drops_then_releases(monkeypatch):
    calls = []
    manager = graph_manager(worker(calls))
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "release_graphs", lambda self: calls.append("drop"))
    manager.release_graphs()
    assert calls == ["quiesce", "drop", "release"]


def test_quiesce_failure_keeps_graph_and_worker_resources(monkeypatch):
    calls = []
    offload = worker(calls)
    offload.stop_selective_native_graphs = Mock(side_effect=RuntimeError("quiesce failed"))
    manager = graph_manager(offload)
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "release_graphs", lambda self: calls.append("drop"))
    with pytest.raises(RuntimeError, match="quiesce failed"):
        manager.release_graphs()
    assert calls == []


def test_parent_graph_release_failure_does_not_free_worker_resources(monkeypatch):
    calls = []
    manager = graph_manager(worker(calls))

    def fail(self):
        calls.append("drop")
        raise RuntimeError("drop failed")

    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "release_graphs", fail)
    with pytest.raises(RuntimeError, match="drop failed"):
        manager.release_graphs()
    assert calls == ["quiesce", "drop"]


def test_recapture_restarts_after_prior_graph_is_released(monkeypatch):
    calls = []
    manager = graph_manager(worker(calls))
    monkeypatch.setattr(aclgraph_utils, "communicator_switch", nullcontext)
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "capture", lambda *a, **k: calls.append("capture"))
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "release_graphs", lambda self: calls.append("drop"))
    capture(manager)
    manager.release_graphs()
    capture(manager)
    assert calls == ["capture", "start", "quiesce", "drop", "release", "capture", "start"]


def test_shutdown_quiesces_before_parent_and_releases_after(monkeypatch):
    calls = []
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.sparse_kv_offload_manager = worker(calls)
    runner.model = None
    monkeypatch.setattr(GPUModelRunner, "shutdown", lambda self: calls.append("shutdown"))
    runner.shutdown()
    assert calls == ["quiesce", "shutdown", "release"]


def test_shutdown_quiesce_failure_never_calls_parent(monkeypatch):
    calls = []
    runner = NPUModelRunner.__new__(NPUModelRunner)
    offload = worker(calls)
    offload.stop_selective_native_graphs = Mock(side_effect=RuntimeError("quiesce failed"))
    runner.sparse_kv_offload_manager = offload
    runner.model = None
    monkeypatch.setattr(GPUModelRunner, "shutdown", lambda self: calls.append("shutdown"))
    with pytest.raises(RuntimeError, match="quiesce failed"):
        runner.shutdown()
    assert calls == []


def test_shutdown_failure_keeps_worker_resources(monkeypatch):
    calls = []
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.sparse_kv_offload_manager = worker(calls)
    runner.model = None

    def fail(self):
        calls.append("shutdown")
        raise RuntimeError("shutdown failed")

    monkeypatch.setattr(GPUModelRunner, "shutdown", fail)
    with pytest.raises(RuntimeError, match="shutdown failed"):
        runner.shutdown()
    assert calls == ["quiesce", "shutdown"]


def test_absent_manager_preserves_capture_and_release(monkeypatch):
    calls = []
    manager = graph_manager(None)
    monkeypatch.setattr(aclgraph_utils, "communicator_switch", nullcontext)
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "capture", lambda *a, **k: calls.append("capture"))
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "release_graphs", lambda self: calls.append("drop"))
    capture(manager)
    manager.release_graphs()
    assert calls == ["capture", "drop"]


def test_nonselective_manager_does_not_require_worker_hooks(monkeypatch):
    calls = []
    offload = SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(selective_offload=False))
    manager = graph_manager(offload)
    monkeypatch.setattr(aclgraph_utils, "communicator_switch", nullcontext)
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "capture", lambda *a, **k: calls.append("capture"))
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "release_graphs", lambda self: calls.append("drop"))
    capture(manager)
    manager.release_graphs()
    assert calls == ["capture", "drop"]


def test_existing_selective_manager_without_worker_preserves_graph_behavior(monkeypatch):
    calls = []
    offload = SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(selective_offload=True))
    manager = graph_manager(offload)
    monkeypatch.setattr(aclgraph_utils, "communicator_switch", nullcontext)
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "capture", lambda *a, **k: calls.append("capture"))
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "release_graphs", lambda self: calls.append("drop"))
    capture(manager)
    manager.release_graphs()
    assert calls == ["capture", "drop"]


def test_incomplete_worker_contract_fails_before_capture(monkeypatch):
    calls = []
    offload = SimpleNamespace(_selective_native_experiment=True,
                              start_selective_native_graphs=lambda graphs: calls.append("start"))
    manager = graph_manager(offload)
    monkeypatch.setattr(aclgraph_utils, "communicator_switch", nullcontext)
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "capture", lambda *a, **k: calls.append("capture"))
    with pytest.raises(RuntimeError, match="worker"):
        capture(manager)
    assert calls == []
    assert offload._selective_native_failed


@pytest.mark.parametrize("updatable", [False, True])
def test_native_replay_retires_only_after_existing_graph_update(monkeypatch, updatable):
    calls = []
    offload = worker(calls)
    offload.topk_buffers_k = [object()]
    offload.begin_selective_native_graph = lambda graph, anchor: calls.append("begin") or 7
    offload.finish_selective_native_graph = lambda graph, epoch, anchor: calls.append("finish")
    manager = graph_manager(offload)
    manager.vllm_config = object()
    manager.update_stream = object()
    manager.model_runner.attn_groups = []
    manager.model_runner.model_state = SimpleNamespace(attn_metadata={}, device_metadata=None)
    # A real BatchExecutionDescriptor is hashable; use its identity here.
    class Descriptor:
        num_tokens = 1

    desc = Descriptor()
    manager.graphs = {desc: object()}
    monkeypatch.setattr(aclgraph_utils, "set_current_vllm_config", lambda *_: nullcontext())
    monkeypatch.setattr(aclgraph_utils, "_get_graph_update_backend", lambda *_: object())
    monkeypatch.setattr(aclgraph_utils, "use_updatable_graph", lambda *_: updatable)
    monkeypatch.setattr(manager, "_graph_relay", lambda *args: calls.append("replay_update") or "output")
    monkeypatch.setattr(manager, "_updatable_graph_replay", lambda *args: calls.append("replay_update") or "output")
    assert manager.run_fullgraph(desc) == "output"
    assert calls == ["begin", "replay_update", "finish"]


def test_failed_native_replay_does_not_record_completion(monkeypatch):
    calls = []
    offload = worker(calls)
    offload.topk_buffers_k = [object()]
    offload.begin_selective_native_graph = lambda graph, anchor: calls.append("begin") or 1
    offload.finish_selective_native_graph = lambda graph, epoch, anchor: calls.append("finish")
    manager = graph_manager(offload)
    manager.vllm_config = object()
    manager.update_stream = object()
    manager.model_runner.attn_groups = []
    manager.model_runner.model_state = SimpleNamespace(attn_metadata={}, device_metadata=None)

    class Descriptor:
        num_tokens = 1

    desc = Descriptor()
    manager.graphs = {desc: object()}
    monkeypatch.setattr(aclgraph_utils, "set_current_vllm_config", lambda *_: nullcontext())
    monkeypatch.setattr(aclgraph_utils, "_get_graph_update_backend", lambda *_: object())
    monkeypatch.setattr(aclgraph_utils, "use_updatable_graph", lambda *_: False)
    monkeypatch.setattr(manager, "_graph_relay", Mock(side_effect=RuntimeError("replay failed")))
    with pytest.raises(RuntimeError, match="replay failed"):
        manager.run_fullgraph(desc)
    assert calls == ["begin"]
    offload.mark_selective_native_failure.assert_called_once()


def test_native_backend_failure_marks_peers_before_replay(monkeypatch):
    offload = worker([])
    manager = graph_manager(offload)
    manager.vllm_config = object()
    manager.update_stream = object()
    manager.model_runner.attn_groups = []
    manager.model_runner.model_state = SimpleNamespace(attn_metadata={}, device_metadata=None)
    class Descriptor:
        num_tokens = 1
    monkeypatch.setattr(aclgraph_utils, "set_current_vllm_config", lambda *_: nullcontext())
    monkeypatch.setattr(aclgraph_utils, "_get_graph_update_backend", Mock(side_effect=RuntimeError("backend failed")))
    with pytest.raises(RuntimeError, match="backend failed"):
        manager.run_fullgraph(Descriptor())
    offload.mark_selective_native_failure.assert_called_once()
