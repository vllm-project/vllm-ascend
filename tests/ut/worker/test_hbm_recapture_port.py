# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise recapture ordering and recovery failures with CPU boundaries."""

from collections import namedtuple
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from .test_fault_tolerance_port import load_definitions

MODES = NS(NONE="none", FULL="full", FULL_DECODE_ONLY="decode")
DESC = namedtuple("Desc", "cg_mode num_tokens num_ubatches num_active_loras uniform_token_count")


@pytest.fixture
def recovery():
    events = []
    ns = load_definitions(
        "vllm_ascend/worker/sentinel/graph_recovery.py",
        ["recover_acl_graphs", "_recover_v2_graphs", "_renew_graph_pool"],
        torch=NS(inference_mode=torch.inference_mode, npu=NS(synchronize=lambda: events.append("sync"))),
        CUDAGraphMode=MODES,
        logger=Mock(),
        set_current_vllm_config=lambda _: nullcontext(),
        set_cudagraph_capturing_enabled=Mock(side_effect=lambda value: events.append(("gate", value))),
        AclGraphSleepWakeupManager=Mock(return_value=NS(sleep=lambda: events.append("discard"))),
    )
    ns["_rebuild_mc2_after_recovery"] = Mock(side_effect=lambda *_: events.append("mc2"))
    pool_renew = ns["_renew_graph_pool"]
    ns["_renew_graph_pool"] = Mock(side_effect=lambda _: events.append("pool"))
    descs = [DESC(MODES.FULL, n, 1, 0, 1) for n in (8, 2)]
    graph = NS(_capture_descs={MODES.FULL: descs}, hidden_states=object(), aux_hidden_states=[object()])
    runner = NS(
        compilation_config=NS(cudagraph_mode=MODES.FULL_DECODE_ONLY),
        vllm_config=object(),
        speculator=None,
        model_state=NS(supports_mm_inputs=False),
        decode_query_len=1,
        cudagraph_manager=graph,
        capture_model=Mock(side_effect=lambda: events.append("capture")),
    )

    def replay(n, **kwargs):
        events.append(("replay", n))
        graph.recovery_replayed_graphs.add(next(d for d in descs if d.num_tokens == n))

    runner._dummy_run = Mock(side_effect=replay)
    return ns, NS(model_runner=runner, use_v2_model_runner=True, rank=0), events, pool_renew


def test_recapture_discards_old_graphs_renews_pool_and_replays_all_shapes(recovery):
    ns, worker, events, _ = recovery
    ns["recover_acl_graphs"](worker, "round")
    assert events == [
        "sync",
        "discard",
        "mc2",
        "pool",
        "capture",
        ("gate", False),
        ("replay", 8),
        "sync",
        ("replay", 2),
        "sync",
        ("gate", False),
    ]
    assert worker.model_runner.cudagraph_manager.recovery_replayed_graphs is None
    assert worker.model_runner.cudagraph_manager.hidden_states is None


@pytest.mark.parametrize("failure", ["capture", "replay", "fallback"])
def test_graph_failure_propagates_and_validation_cannot_succeed_eagerly(recovery, failure):
    ns, worker, _, _ = recovery
    runner = worker.model_runner
    if failure == "fallback":
        runner._dummy_run.side_effect = None
    else:
        target = runner.capture_model if failure == "capture" else runner._dummy_run
        target.side_effect = RuntimeError("graph failed")
    with pytest.raises(RuntimeError):
        ns["recover_acl_graphs"](worker, "round")
    assert ns["set_cudagraph_capturing_enabled"].call_args.args == (False,)
    assert runner.cudagraph_manager.recovery_replayed_graphs is None


@pytest.mark.parametrize("unsupported", ["draft", "encoder", "mixed", "ubatch", "lora", "empty"])
def test_unsupported_graphs_rejected_before_discard(recovery, unsupported):
    ns, worker, events, _ = recovery
    runner = worker.model_runner
    if unsupported == "draft":
        runner.speculator = object()
    elif unsupported == "encoder":
        runner.model_state = NS(supports_mm_inputs=True, encoder_runner=NS(has_cudagraph=lambda: True))
    elif unsupported == "empty":
        runner.cudagraph_manager._capture_descs = {}
    else:
        desc = DESC(
            "piecewise" if unsupported == "mixed" else MODES.FULL,
            8,
            2 if unsupported == "ubatch" else 1,
            1 if unsupported == "lora" else 0,
            1,
        )
        runner.cudagraph_manager._capture_descs = {MODES.FULL: [desc]}
    with pytest.raises(RuntimeError):
        ns["recover_acl_graphs"](worker, "round")
    assert events == []


def test_eager_retry_rebuilds_communication_without_graph_work(recovery):
    ns, worker, events, _ = recovery
    worker.model_runner.compilation_config.cudagraph_mode = MODES.NONE
    ns["recover_acl_graphs"](worker, "round")
    assert events == ["sync", "mc2"]


def test_new_pool_propagates_to_all_graph_owners(recovery):
    ns, worker, _, renew = recovery

    class Platform:
        _global_graph_pool = "old"

        def graph_pool_handle(self):
            return "new"

    wrappers = [NS(graph_pool="old"), NS(graph_pool="old")]
    ns.update(current_platform=Platform(), acl_graph=NS(_acl_graph_wrappers=wrappers))
    renew(worker.model_runner)
    assert Platform._global_graph_pool == "new"
    assert all(w.graph_pool == "new" for w in wrappers)
    assert worker.model_runner.cudagraph_manager.pool == "new"


@pytest.mark.parametrize("hbm,failure", [(False, False), (True, False), (True, True)])
def test_only_uce_recaptures_and_failure_keeps_worker_quarantined(hbm, failure):
    events = []

    class Base:
        def retry(self, request):
            events.append("groups")

    def recover(*args):
        events.append("graphs")
        if failure:
            raise RuntimeError("recapture failed")

    ns = load_definitions(
        "vllm_ascend/worker/sentinel/npu_worker_sentinel.py",
        ["WorkerSentinel"],
        GPUWorkerSentinel=Base,
        logger=Mock(),
        recover_acl_graphs=recover,
        torch=NS(npu=NS(synchronize=lambda: events.append("sync"))),
    )
    sentinel = ns["WorkerSentinel"].__new__(ns["WorkerSentinel"])
    sentinel.worker_faulted = True
    registry = NS(regions=[object()], invalidate_blocks=lambda _: events.append("zero"))
    sentinel.worker = NS(rank=0, model_runner=NS(kv_cache_region_registry=registry))
    sentinel.retry_cache_plan = {"blocks": [], "hbm_detected": hbm}
    sentinel._prepare_retry = Mock(side_effect=lambda _: events.append("classify"))
    sentinel._restart_device = Mock(side_effect=lambda: events.append("restart"))
    sentinel._release_fault_state = Mock(side_effect=lambda: events.append("release"))
    request = NS(instruction="retry", request_id="round")
    if failure:
        with pytest.raises(RuntimeError, match="recapture failed"):
            sentinel.retry(request)
        assert sentinel.worker_faulted
        sentinel._release_fault_state.assert_not_called()
    else:
        sentinel.retry(request)
        assert not sentinel.worker_faulted
        assert events == ["classify", "restart", "groups", "zero", "sync"] + (["graphs"] if hbm else []) + ["release"]


def test_metadata_reset_retires_failed_work_and_preserves_persistent_events():
    from contextlib import contextmanager

    ns = load_definitions(
        "vllm_ascend/worker/device_metadata.py",
        ["DeviceMetadataExecutor", "TargetDeviceMetadata"],
        contextmanager=contextmanager,
    )
    executor = ns["DeviceMetadataExecutor"].__new__(ns["DeviceMetadataExecutor"])
    event = object()
    executor.event = event
    executor._submission_in_flight = True
    executor._has_reuse_fence = True
    executor._batch_descriptor = object()
    executor._waited_stages = {1, 2}
    metadata = ns["TargetDeviceMetadata"].__new__(ns["TargetDeviceMetadata"])
    metadata.executor = executor
    metadata._tasks = (object(),)
    metadata._failed = True
    metadata.reset_after_recovery()
    assert metadata.executor is executor and executor.event is event
    assert not executor._submission_in_flight and not executor._has_reuse_fence
    assert executor._batch_descriptor is None and not executor._waited_stages
    assert metadata._tasks == () and not metadata._failed
    metadata.reset_after_recovery()


@pytest.mark.parametrize("failure", [None, "prepare", "destroy", "restore", "refresh"])
def test_mc2_recovery_order_and_failures(failure):
    events = []

    def step(name):
        def run(*args, **kwargs):
            events.append(name)
            if name == failure:
                if name == "refresh":
                    raise RuntimeError("refresh failed")
                return False
            return True

        return run

    group = NS(cpu_group=object(), destroy_hccl=step("destroy"), restore_hccl=step("restore"))
    ns = load_definitions(
        "vllm_ascend/worker/sentinel/graph_recovery.py",
        ["_rebuild_mc2_after_recovery"],
        HcclSleepWakeupManager=NS(
            prepare_moe_hccl_teardown=step("prepare"),
            refresh_moe_hccl_groups=step("refresh"),
            refresh_moe_hccl_runtime_state=step("runtime"),
        ),
        get_mc2_group=lambda: group,
        torch=NS(distributed=NS(barrier=step("barrier"))),
        logger=Mock(),
    )
    expected = ["prepare", "barrier", "destroy", "restore", "refresh", "runtime", "barrier"]
    if failure:
        with pytest.raises(RuntimeError):
            ns["_rebuild_mc2_after_recovery"](NS(rank=0), "round")
        assert events == expected[: expected.index(failure) + 1]
    else:
        ns["_rebuild_mc2_after_recovery"](NS(rank=0), "round")
        assert events == expected
