# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU certificates/state-machine tests; these do not establish NPU correctness."""

import gc
import weakref
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.ascend_config import RlConfig
from vllm_ascend.device_allocator import graph_reuse as module
from vllm_ascend.device_allocator.camem import ACL_GRAPH_TAG
from vllm_ascend.device_allocator.graph_reuse import SleepGraphReuse, graph_resource_signature, reuse_unsupported_reason
from vllm_ascend.device_allocator.sleep_mem_optimized import SleepWakeupManager, serialized_model_execution


@pytest.fixture
def case(monkeypatch):
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_text_config=SimpleNamespace(model_type="qwen2"),
        is_multimodal_model=False,
        use_mla=False,
    )
    config = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=model_config,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY, cudagraph_capture_sizes=[1, 2, 4, 8]
        ),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=False,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        speculative_config=None,
        lora_config=None,
        kv_transfer_config=None,
    )
    backend = type("AscendAttentionBackend", (), {"__module__": "vllm_ascend.attention.attention_v1"})
    graph = SimpleNamespace(
        capture_resources={"workspace": torch.zeros(4)}, tasks=[SimpleNamespace(handle=object(), event=object())]
    )
    runner = SimpleNamespace(
        model=torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16),
        attn_groups=[[SimpleNamespace(backend=backend)]],
        cudagraph_manager=SimpleNamespace(
            graphs={1: graph},
            capture_sizes=[1, 2, 4, 8],
            sleep_reuse_capture_ready=True,
            hidden_states=None,
            aux_hidden_states=[],
        ),
    )
    allocator = SimpleNamespace(
        allocator_and_pools={ACL_GRAPH_TAG: (SimpleNamespace(id=(0, 7)), object())},
        pointer_to_data={},
        _check_memory_state=MagicMock(),
    )
    monkeypatch.setattr(module.CaMemAllocator, "get_instance", lambda: allocator)
    monkeypatch.setattr(
        module, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda capability: True)
    )
    monkeypatch.setattr(
        module,
        "get_ascend_config",
        lambda: SimpleNamespace(
            eplb_config=SimpleNamespace(dynamic_eplb=False),
            ascend_compilation_config=SimpleNamespace(enable_super_kernel=False),
        ),
    )
    monkeypatch.setattr(SleepGraphReuse, "unanimous", staticmethod(lambda local: local))
    group = SimpleNamespace(device_group=object())
    yield SimpleNamespace(config=config, runner=runner, graph=graph, allocator=allocator, group=group)


def controller(case):
    return SleepGraphReuse(case.config, lambda: case.runner)


def test_partial_wake_and_repeated_sleep_are_explicit(case):
    reuse = controller(case)
    assert reuse.allow_sleep(2)
    assert reuse.prepare_sleep(2, (case.group,))
    assert not reuse.allow_sleep(2)
    assert not reuse.complete_after(["weights"])
    reuse.note_restored(["weights"])
    assert reuse.state == "PARTIAL"
    assert reuse.complete_after(["kv_cache"])
    reuse.note_restored(["kv_cache"])
    assert reuse.can_replay((case.group,))
    reuse.finish(True)
    assert reuse.state == "READY" and reuse.reuses == 1
    assert reuse.allow_sleep(1)


def test_reverse_tag_order_waits_for_all_pools(case):
    reuse = controller(case)
    reuse.prepare_sleep(1, ())
    reuse.note_restored(["kv_cache"])
    assert reuse.pending_tags == {"weights"}
    assert not reuse.complete_after(["kv_cache"])
    assert reuse.complete_after(["weights"])


@pytest.mark.parametrize("change", ["graph", "event", "communicator", "pool", "capture_sizes", "backend"])
def test_changed_resource_cannot_replay(case, change):
    reuse = controller(case)
    reuse.prepare_sleep(1, (case.group,))
    if change == "graph":
        case.runner.cudagraph_manager.graphs = {1: SimpleNamespace(capture_resources={}, tasks=[])}
    elif change == "event":
        case.graph.tasks[0].event = object()
    elif change == "communicator":
        case.group.device_group = object()
    elif change == "pool":
        case.allocator.allocator_and_pools[ACL_GRAPH_TAG] = (SimpleNamespace(id=(0, 8)), object())
    elif change == "capture_sizes":
        case.config.compilation_config.cudagraph_capture_sizes = [1]
    else:
        case.runner.attn_groups[0][0].backend = type("OtherBackend", (), {})
    assert not reuse.can_replay((case.group,))


def test_workspace_replacement_invalidates_but_weight_values_may_change(case):
    # Read the actual CPU tensor's address/layout through a simulated NPU device
    # property, without running any NPU kernels.
    with patch.object(torch.Tensor, "device", new_callable=PropertyMock, return_value=SimpleNamespace(type="npu")):
        before = graph_resource_signature(case.config, case.runner, ())
        with torch.no_grad():
            case.runner.model.weight.add_(1)
        assert before == graph_resource_signature(case.config, case.runner, ())
        case.graph.capture_resources["workspace"] = torch.zeros(8)
        assert before != graph_resource_signature(case.config, case.runner, ())


def test_parameter_replacement_invalidates(case):
    with patch.object(torch.Tensor, "device", new_callable=PropertyMock, return_value=SimpleNamespace(type="npu")):
        before = graph_resource_signature(case.config, case.runner, ())
        case.runner.model.weight = torch.nn.Parameter(case.runner.model.weight.detach().clone())
        assert before != graph_resource_signature(case.config, case.runner, ())


def test_peer_rejection_forces_fallback_before_cleanup(case, monkeypatch):
    monkeypatch.setattr(SleepGraphReuse, "unanimous", staticmethod(lambda local: False))
    reuse = controller(case)
    assert not reuse.prepare_sleep(1, ())
    assert reuse.reason == "peer_not_reusable"


def test_inventory_exception_still_votes_for_collective_fallback(case, monkeypatch):
    reuse = controller(case)
    votes = []
    monkeypatch.setattr(module, "reuse_unsupported_reason", MagicMock(side_effect=RuntimeError("inventory")))
    monkeypatch.setattr(reuse, "unanimous", lambda vote: votes.append(vote) or vote)
    assert not reuse.prepare_sleep(1, (case.group,))
    assert votes == [False]
    assert reuse.reason == "resource_inventory_failed"


@pytest.mark.parametrize(
    "feature,reason",
    [
        ("ep", "expert_parallel_not_validated"),
        ("lora", "speculation_or_lora_not_validated"),
        ("mode", "requires_full_decode_only"),
    ],
)
def test_unsupported_combination_has_reason(case, feature, reason):
    if feature == "ep":
        case.config.parallel_config.enable_expert_parallel = True
    elif feature == "lora":
        case.config.lora_config = object()
    else:
        case.config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    assert reuse_unsupported_reason(case.config, case.runner) == reason


def test_manager_keeps_communicators_and_skips_capture(case):
    manager = SleepWakeupManager(case.config, object(), lambda: case.runner, graph_reuse_enabled=True)
    manager.hccl.iter_alive_group_coordinators = lambda: iter((case.group,))
    manager.hccl.sleep = MagicMock()
    manager.hccl.wakeup = MagicMock()
    manager.acl_graph.wakeup = MagicMock()
    manager.sleep()
    manager.wakeup(["weights"])
    manager.acl_graph.wakeup.assert_not_called()
    manager.wakeup(["kv_cache"])
    manager.hccl.sleep.assert_not_called()
    manager.hccl.wakeup.assert_not_called()
    manager.acl_graph.wakeup.assert_not_called()
    assert manager.graph_reuse.reuses == 1


def test_restore_failure_recaptures_without_replaying_old_graph(case):
    manager = SleepWakeupManager(case.config, object(), lambda: case.runner, graph_reuse_enabled=True)
    manager.hccl.iter_alive_group_coordinators = lambda: iter(())
    manager.hccl.release_lifecycle_anchor = MagicMock()
    manager._discard_sleep_graphs = MagicMock()
    manager.acl_graph.wakeup = MagicMock()
    manager.sleep()
    manager.graph_reuse.signature = None
    manager.graph_reuse.reason = "graph_pool_restore_failed"
    manager.wakeup()
    manager._discard_sleep_graphs.assert_called_once_with()
    manager.acl_graph.wakeup.assert_called_once_with(None)
    assert manager.graph_reuse.recaptures == 1


def test_failed_transaction_requires_restart(case):
    reuse = controller(case)
    reuse.state = "ERROR"
    with pytest.raises(RuntimeError, match="restart the worker"):
        reuse.allow_sleep(1)


def test_sleep_pins_communicator_identity_until_collective_recovery_finishes(case):
    class Communicator:
        pass

    owner = Communicator()
    reference = weakref.ref(owner)
    case.group.device_group = owner
    reuse = controller(case)
    assert reuse.prepare_sleep(1, (case.group,))
    case.group.device_group = Communicator()
    del owner
    gc.collect()
    assert reference() is not None
    reuse.note_restored(None)
    assert not reuse.can_replay((case.group,))
    reuse.finish(False)
    gc.collect()
    assert reference() is None


def test_shutdown_closes_reuse_and_retires_graphs(case):
    manager = SleepWakeupManager(case.config, object(), lambda: case.runner, graph_reuse_enabled=True)
    manager._discard_sleep_graphs = MagicMock()
    manager.shutdown()
    manager._discard_sleep_graphs.assert_called_once_with()
    assert manager.graph_reuse.state == "CLOSED"
    with pytest.raises(RuntimeError, match="worker is closed"):
        manager.graph_reuse.require_healthy()


@pytest.mark.parametrize("state", ["ASLEEP", "PARTIAL", "RESTORING", "ERROR", "CLOSED"])
def test_inference_guard_rejects_before_execution_and_allows_ready(case, state):
    manager = SleepWakeupManager(case.config, object(), lambda: case.runner, graph_reuse_enabled=True)

    class Worker:
        sleep_wakeup_manager = manager
        calls = 0

        @serialized_model_execution
        def execute(self):
            self.calls += 1
            return self.calls

    worker = Worker()
    manager.graph_reuse.state = state
    with pytest.raises(RuntimeError, match="Inference is unavailable"):
        worker.execute()
    assert worker.calls == 0
    assert manager.graph_reuse.state == state
    manager.graph_reuse.state = "READY"
    assert worker.execute() == 1


def test_model_execution_holds_sleep_transition_lock(case):
    manager = SleepWakeupManager(case.config, object(), lambda: case.runner, graph_reuse_enabled=True)
    entered, finish = Event(), Event()

    class Worker:
        sleep_wakeup_manager = manager

        @serialized_model_execution
        def execute(self):
            entered.set()
            assert finish.wait(5)

    worker = Worker()
    thread = Thread(target=worker.execute)
    thread.start()
    try:
        assert entered.wait(5)
        acquired = manager.transition_lock.acquire(blocking=False)
        if acquired:
            manager.transition_lock.release()
        assert not acquired
    finally:
        finish.set()
        thread.join(5)
    assert not thread.is_alive()


@pytest.mark.parametrize("fail_after_execution", [False, True])
@pytest.mark.parametrize("reuse_enabled", [False, True])
def test_native_free_failure_closes_inference_without_replay_or_recapture(
    case, monkeypatch, fail_after_execution, reuse_enabled
):
    manager = SleepWakeupManager(case.config, object(), lambda: case.runner, graph_reuse_enabled=reuse_enabled)
    monkeypatch.setattr(module.CaMemAllocator, "instance", case.allocator)
    calls = []
    failure = RuntimeError("CaMem free callback failed; restart the worker")
    case.allocator._check_memory_state.side_effect = [None, failure] if fail_after_execution else failure

    class Worker:
        sleep_wakeup_manager = manager

        @serialized_model_execution
        def execute(self):
            calls.append("execute")

    worker = Worker()
    with pytest.raises(RuntimeError, match="restart the worker"):
        worker.execute()
    assert calls == (["execute"] if fail_after_execution else [])
    if reuse_enabled:
        assert manager.graph_reuse.state == "ERROR"
        assert manager.graph_reuse.reuses == manager.graph_reuse.recaptures == 0
    case.allocator._check_memory_state.side_effect = failure
    with pytest.raises(RuntimeError, match="Inference is unavailable|restart the worker"):
        worker.execute()


def test_ordinary_inference_does_not_create_a_sleep_allocator(case, monkeypatch):
    monkeypatch.setattr(module.CaMemAllocator, "instance", None)
    create = MagicMock(side_effect=AssertionError("must not instantiate"))
    monkeypatch.setattr(module.CaMemAllocator, "get_instance", create)

    class Worker:
        @serialized_model_execution
        def execute(self):
            return "ordinary"

    assert Worker().execute() == "ordinary"
    create.assert_not_called()


@pytest.mark.parametrize("enabled,cleanup", [(False, False), (True, False), (False, True)])
def test_reuse_requires_rl_and_extra_cleanup(enabled, cleanup):
    with pytest.raises(ValueError, match="requires rl_config.enabled"):
        RlConfig(enabled=enabled, sleep_mode_extra_cleanup=cleanup, sleep_mode_graph_reuse=True)


def test_reuse_switch_defaults_off_and_accepts_lax_bool():
    assert RlConfig().sleep_mode_graph_reuse is False
    assert (
        RlConfig(enabled=True, sleep_mode_extra_cleanup=True, sleep_mode_graph_reuse="true").sleep_mode_graph_reuse
        is True
    )
