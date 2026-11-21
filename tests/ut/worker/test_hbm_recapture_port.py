# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise recapture ordering and recovery failures with CPU boundaries."""

import ast
from collections import namedtuple
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


def load_definitions(path, names, **namespace):
    # Load production definitions without importing torch/NPU in this CPU test.
    tree = ast.parse((Path(__file__).resolve().parents[3] / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


MODES = NS(NONE="none", FULL="full", PIECEWISE="piecewise", FULL_DECODE_ONLY="decode")
DESC = namedtuple("Desc", "cg_mode num_tokens num_ubatches num_active_loras uniform_token_count")


@pytest.fixture
def recovery():
    events = []
    ns = load_definitions(
        "vllm_ascend/worker/sentinel/graph_recovery.py",
        ["recover_acl_graphs", "_recover_v2_graphs", "_renew_graph_pool"],
        torch=NS(inference_mode=lambda: lambda fn: fn, npu=NS(synchronize=lambda: events.append("sync"))),
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
    graph = NS(
        _capture_descs={MODES.FULL: descs},
        hidden_states=object(),
        aux_hidden_states=[object()],
        graphs={desc: object() for desc in descs},
        _graphs_captured=True,
    )

    def release():
        events.append("release_graphs")
        graph.graphs.clear()
        graph._graphs_captured = False

    def capture():
        events.append("capture")
        graph.graphs = {
            desc: object() for group in graph._capture_descs.values() for desc in group if desc.cg_mode == MODES.FULL
        }
        graph._graphs_captured = True

    graph.release_graphs = Mock(side_effect=release)
    runner = NS(
        compilation_config=NS(cudagraph_mode=MODES.FULL_DECODE_ONLY),
        vllm_config=object(),
        speculator=None,
        model_state=NS(supports_mm_inputs=False),
        decode_query_len=1,
        cudagraph_manager=graph,
        capture_model=Mock(side_effect=capture),
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
        "release_graphs",
        "discard",
        "mc2",
        "pool",
        ("gate", True),
        "capture",
        ("gate", False),
        "sync",
        ("replay", 8),
        "sync",
        ("replay", 2),
        "sync",
        ("gate", False),
    ]
    assert worker.model_runner.cudagraph_manager.recovery_replayed_graphs is None
    assert worker.model_runner.cudagraph_manager.hidden_states is None


@pytest.mark.parametrize("mode", [MODES.FULL, MODES.PIECEWISE])
def test_consecutive_recoveries_replace_graphs_without_changing_plan(recovery, mode):
    ns, worker, events, _ = recovery
    runner = worker.model_runner
    graph = runner.cudagraph_manager
    desc = DESC(mode, 8, 1, 0, 1)
    graph._capture_descs = {mode: [desc]}
    for round_id in ("first", "second"):
        old_graphs = dict(graph.graphs)
        ns["recover_acl_graphs"](worker, round_id)
        assert graph._capture_descs == {mode: [desc]}
        assert graph._graphs_captured
        assert all(value is not old_graphs.get(key) for key, value in graph.graphs.items())
        assert graph.recovery_replayed_graphs is None
        assert ns["set_cudagraph_capturing_enabled"].call_args.args == (False,)
    assert runner.capture_model.call_count == 2
    assert events.count("release_graphs") == 2
    assert events.count("mc2") == 2


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


@pytest.mark.parametrize("unsupported", ["draft", "encoder"])
def test_secondary_graph_owners_rejected_before_discard(recovery, unsupported):
    ns, worker, events, _ = recovery
    runner = worker.model_runner
    if unsupported == "draft":
        runner.speculator = object()
    else:
        runner.model_state = NS(supports_mm_inputs=True, encoder_runner=NS(has_cudagraph=lambda: True))
    with pytest.raises(RuntimeError):
        ns["recover_acl_graphs"](worker, "round")
    assert events == []


@pytest.mark.parametrize("mode", ["piecewise", "mixed", "nonuniform", "ubatch", "lora", "empty"])
def test_native_capture_plan_is_preserved_without_forcing_uniform_decode(recovery, mode):
    ns, worker, events, _ = recovery
    runner = worker.model_runner
    graph = runner.cudagraph_manager
    desc = DESC(
        MODES.PIECEWISE if mode in ("piecewise", "mixed") else MODES.FULL,
        8,
        2 if mode == "ubatch" else 1,
        1 if mode == "lora" else 0,
        None if mode == "nonuniform" else 1,
    )
    plan = {} if mode == "empty" else {desc.cg_mode: [desc]}
    if mode == "mixed":
        plan[MODES.FULL] = [DESC(MODES.FULL, 2, 1, 0, 1)]
    graph._capture_descs = plan
    ns["recover_acl_graphs"](worker, "round")
    assert graph._capture_descs == plan
    assert events.index("release_graphs") < events.index("mc2") < events.index("capture")
    runner.capture_model.assert_called_once_with()
    runner._dummy_run.assert_not_called()
    assert graph.recovery_replayed_graphs is None
    assert ns["logger"].info.call_args.args[-1] is False


@pytest.mark.parametrize("failure", ["incomplete", "changed", "missing_full"])
def test_incomplete_recapture_cannot_report_success(recovery, failure):
    ns, worker, _, _ = recovery
    graph = worker.model_runner.cudagraph_manager

    def incomplete():
        graph._graphs_captured = failure != "incomplete"
        if failure == "changed":
            graph._capture_descs = {}

    worker.model_runner.capture_model.side_effect = incomplete
    with pytest.raises(RuntimeError, match="capture|recaptured"):
        ns["recover_acl_graphs"](worker, "round")
    worker.model_runner._dummy_run.assert_not_called()
    assert ns["set_cudagraph_capturing_enabled"].call_args.args == (False,)


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
    breakable = [NS(graph_pool="old-breakable")]
    ns.update(
        current_platform=Platform(),
        acl_graph=NS(_acl_graph_wrappers=wrappers),
        BreakableACLGraphWrapper=NS(_all_instances=breakable),
    )
    renew(worker.model_runner)
    assert Platform._global_graph_pool == "new"
    assert all(w.graph_pool == "new" for w in wrappers + breakable)
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
