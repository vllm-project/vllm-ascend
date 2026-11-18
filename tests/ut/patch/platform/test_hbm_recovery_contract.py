# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts for VA-only recovery and the unchanged Core FT vote."""

import ast
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS
from typing import cast
from unittest.mock import Mock
from weakref import WeakValueDictionary, ref

import pytest

ROOT = Path(__file__).parents[4]
STATUS = NS(PREEMPTED="preempted", WAITING="waiting", FINISHED_ABORTED="aborted")


def load_method(path, class_name, method):
    tree = ast.parse((ROOT / path).read_text())
    cls = tree if class_name is None else next(node for node in tree.body if getattr(node, "name", None) == class_name)
    node = next(node for node in cls.body if getattr(node, "name", None) == method)
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            node,
        ],
        type_ignores=[],
    )
    namespace = dict(cast=cast, Scheduler=object, RequestStatus=STATUS, logger=Mock())
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method]


@pytest.mark.parametrize(
    "existing,ready,expired,race,expected",
    [
        (b"rejected: peer UCE", True, False, None, b"rejected: peer UCE"),
        (None, True, False, None, b"approved"),
        (None, False, True, None, b"rejected: timeout"),
        (None, False, True, b"approved", b"approved"),
        (None, True, False, b"rejected: peer UCE", b"rejected: peer UCE"),
        (None, False, False, None, b"approved"),
    ],
)
def test_retry_consensus_keeps_first_decision(existing, ready, expired, race, expected):
    wait = load_method(
        "vllm_ascend/worker/sentinel/npu_worker_sentinel.py", "WorkerSentinel", "_wait_for_retry_decision"
    )
    values = {} if existing is None else {"decision": existing}
    readiness = [ready]

    def compare_set(key, old, new):
        if race is not None:
            values[key] = race
        return values.setdefault(key, new.encode())

    store = NS(
        check=lambda keys: "decision" in values if keys == ["decision"] else readiness[0],
        get=values.__getitem__,
        compare_set=Mock(side_effect=compare_set),
    )
    sleep = Mock(side_effect=lambda _: readiness.__setitem__(0, True))
    wait.__func__.__globals__.update(
        time=NS(monotonic=lambda: 2 if expired else 0, sleep=sleep), _RETRY_CLASSIFICATION_POLL_SEC=0.01
    )
    assert wait(store, "decision", ["ready"], 1, "timeout") == expected
    assert store.compare_set.call_count == (0 if existing else 1)
    assert sleep.call_count == (1 if not existing and not ready and not expired else 0)


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "classification",
        "aliases",
        "non_kv",
        "peer_non_kv",
        "peer_error",
        "restart",
        "repeated",
        "missing",
        "stop",
    ],
)
def test_retry_plan_uses_local_blocks_and_rejects_failed_phases(failure):
    import json
    import time
    from datetime import timedelta

    path = "vllm_ascend/worker/sentinel/npu_worker_sentinel.py"
    prepare = load_method(path, "WorkerSentinel", "_prepare_retry")
    wait = load_method(path, "WorkerSentinel", "_wait_for_retry_decision")
    helpers = {
        name: load_method(path, "WorkerSentinel", name)
        for name in (
            "_build_retry_context",
            "_classify_and_report_local_blocks",
            "_wait_for_classification",
            "_agree_dp_aliases",
            "_build_and_agree_cache_plan",
        )
    }
    prefix = "ascend_hbm_retry_classification/round"
    values = {
        f"{prefix}/ready/1/0": b"recoverable",
        f"{prefix}/cache_ready/1/0": b"1",
    }
    values[f"{prefix}/blocks/1/0"] = b"[7]"
    if failure == "peer_non_kv":
        values[f"{prefix}/ready/1/0"] = b"non_kv"
    if failure == "missing":
        del values[f"{prefix}/ready/1/0"]
    if failure == "peer_error":
        values[f"{prefix}/decision"] = b"rejected: peer_error"

    class NonKvFault(RuntimeError):
        pass

    class UnrecoverableFault(RuntimeError):
        def __init__(self, **kwargs):
            super().__init__(kwargs["reason"])

    def set_value(key, value):
        values[key] = value.encode() if isinstance(value, str) else value

    def compare_set(key, old, new):
        return values.setdefault(key, new.encode())

    store = NS(
        set=set_value,
        get=values.__getitem__,
        check=lambda keys: all(k in values for k in keys),
        compare_set=compare_set,
    )
    registry = Mock(regions=[object()], num_blocks=8)
    registry.partial_block_ids.return_value = {2}
    registry.expand_aliases.side_effect = (
        RuntimeError("aliases failed") if failure == "aliases" else lambda blocks: blocks | {3}
    )
    sentinel = NS(
        worker=NS(
            rank=0,
            parallel_config=NS(world_size=1, fault_tolerance_config=NS(engine_recovery_timeout_sec=1)),
            model_runner=NS(kv_cache_region_registry=registry),
        ),
        _stop_device=Mock(side_effect=RuntimeError("stop failed") if failure == "stop" else None),
        _classify_uce=Mock(
            side_effect=(
                NonKvFault("non-KV UCE")
                if failure in ("non_kv", "restart", "missing", "peer_error")
                else RuntimeError("classification failed")
                if failure == "classification"
                else None
            )
        ),
        _wait_for_retry_decision=wait,
        _non_kv_repair_attempted=failure == "repeated",
    )

    def restart():
        assert all(f"{prefix}/ready/{dp}/0" in values for dp in range(2))
        assert sentinel._non_kv_repair_attempted
        if failure == "restart":
            raise RuntimeError("restart failed")

    sentinel._restart_device = Mock(side_effect=restart)
    runtime = dict(
        json=json,
        time=time,
        timedelta=timedelta,
        torch=NS(distributed=NS(TCPStore=lambda *args, **kwargs: store)),
        _RetryContext=NS,
        _RETRY_CLASSIFICATION_RECOVERABLE=b"recoverable",
        _RETRY_CLASSIFICATION_NON_KV=b"non_kv",
        _RETRY_CLASSIFICATION_POLL_SEC=0.01,
        HbmUceNonKvFault=NonKvFault,
        HbmUceUnrecoverableFault=UnrecoverableFault,
    )
    prepare.__globals__.update(runtime)
    for name, helper in helpers.items():
        helper.__globals__.update(runtime)
        setattr(sentinel, name, helper.__get__(sentinel))
    wait.__func__.__globals__.update(time=time, _RETRY_CLASSIFICATION_POLL_SEC=0.01)
    request = NS(
        instruction="retry",
        request_id="round",
        params=dict(
            recovery_round="round",
            dp_group_size=2,
            dp_group_rank=0,
            dp_master_ip="localhost",
            recovery_store_port=1,
        ),
    )
    if failure:
        message = {
            "non_kv": "engine restart required",
            "peer_non_kv": "engine restart required",
            "repeated": "already attempted",
            "missing": "timed out",
        }.get(failure, failure)
        sentinel._prepare_retry = prepare.__get__(sentinel)
        retry = load_method(path, "WorkerSentinel", "retry")
        with pytest.raises(RuntimeError, match=message):
            retry(sentinel, request)
        assert sentinel.retry_cache_plan is None
        assert sentinel.worker_faulted
    else:
        prepare(sentinel, request)
        assert sentinel.retry_cache_plan == {"request_id": "round", "blocks": [2, 3], "hbm_detected": True}
    registry.invalidate_blocks.assert_not_called()
    if failure in ("non_kv", "peer_non_kv", "restart"):
        sentinel._restart_device.assert_called_once()
        # A new round must fail without repeating repair, including healthy peers.
        values.clear()
        request.request_id = "round-2"
        request.params["recovery_round"] = "round-2"
        with pytest.raises(RuntimeError, match="already attempted"):
            prepare(sentinel, request)
        sentinel._restart_device.assert_called_once()
    else:
        sentinel._restart_device.assert_not_called()


@pytest.mark.parametrize("kind", ["non_kv", "kv", "malformed", "query_failed", "non_pta"])
def test_only_classified_non_kv_ranges_allow_repair_before_restart(kind):
    path = "vllm_ascend/worker/sentinel/uce_fault.py"
    tree = ast.parse((ROOT / path).read_text())
    namespace = {"logger": Mock()}
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *classes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    classify = load_method(path, None, "classify_uce_after_stop")
    query_error = namespace["HbmUceUnrecoverableFault"]("unknown", None, "query failed")
    entries = [{"ptr": 4096, "size": 64} if kind != "malformed" else {"ptr": 4096}]
    registry = Mock(regions=[object()], num_blocks=8)
    registry.recover_uce_addresses.return_value = (
        NS(layer_name="kv", kind="backing", rank=0, device_id=0) if kind == "kv" else None
    )
    classify.__globals__.update(
        namespace,
        _UCE_NO_ADDRESS=0,
        _UCE_NON_PTA_MEMORY=1,
        _check_uce_in_memory=Mock(side_effect=query_error)
        if kind == "query_failed"
        else Mock(return_value=1 if kind == "non_pta" else 3),
        _get_uce_addr=lambda: entries,
        _first_uce_address=lambda entries: 4096,
    )
    if kind == "query_failed":
        with pytest.raises(namespace["HbmUceUnrecoverableFault"]):
            classify(registry, 0)
    else:
        fault = classify(registry, 0)
        assert isinstance(fault, namespace["HbmUceNonKvFault"]) == (kind == "non_kv")
        assert isinstance(fault, namespace["HbmUceKvCacheFault"]) == (kind == "kv")
    if kind in ("malformed", "query_failed", "non_pta"):
        registry.recover_uce_addresses.assert_not_called()


@pytest.fixture
def recovery():
    events = []
    config = NS(
        instance_id="engine",
        parallel_config=NS(data_parallel_rank=0, fault_tolerance_config=NS(request_failure_policy="fail")),
    )
    plan = {"request_id": "round", "blocks": [1], "hbm_detected": True}
    finished = NS(is_empty=lambda: False, finished_recving={"transfer"})
    poll = NS(kv_connector_output=finished)
    poll_outputs = [poll]

    class BaseExecutor:
        def __init__(self, config):
            self.vllm_config = config
            self.kv_output_aggregator = NS(aggregate=lambda outputs: outputs[0])

        def collective_rpc(self, method, timeout=None, args=(), kwargs=None, **options):
            if method == "get_kv_cache_recovery_plan":
                return [plan, plan]
            if callable(method):
                events.append("poll")
                return poll_outputs
            events.append(method)
            if options.get("non_block"):
                return Future()
            return ["original-result"]

    class BaseScheduler:
        def __init__(self, config):
            self.vllm_config = config
            self.parallel_config = config.parallel_config
            self.running = []
            self.requests = {}
            self.connector = object()
            self.finished_req_ids = {"transfer"}
            self.deferred_frees = deque([(3, ["block-ref"])])
            self.processed_step_seq = 1
            self.sched_step_seq = 3
            self.prev_step_scheduled_req_ids = {"old"}
            self.kv_cache_manager = NS(
                block_pool=NS(
                    blocks=[NS(ref_cnt=0, is_null=False) for _ in range(4)],
                    free_blocks=Mock(side_effect=lambda blocks: events.append("free")),
                ),
                evict_blocks=Mock(side_effect=lambda blocks: events.append(("evict", blocks))),
                get_block_ids=lambda _: ([],),
            )
            self.reset_connector_cache = Mock(return_value=True)

        def update_from_output(self, scheduled, output):
            events.append(("output", output.kv_connector_output))
            return {0: "result"}

    tree = ast.parse((ROOT / "vllm_ascend/core/hbm_recovery.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    namespace = dict(
        deque=deque,
        ref=ref,
        _SCHEDULERS=WeakValueDictionary(),
        _EXECUTORS=WeakValueDictionary(),
        MultiprocExecutor=BaseExecutor,
        Scheduler=BaseScheduler,
        AsyncScheduler=BaseScheduler,
        RecomputeScheduler=BaseScheduler,
        AsyncRecomputeScheduler=BaseScheduler,
        RequestStatus=STATUS,
        logger=Mock(),
        SchedulerOutput=NS(make_empty=lambda: "empty"),
        ModelRunnerOutput=NS(with_kv_conn_output_only=lambda kv: NS(kv_connector_output=kv)),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "hbm_recovery.py", "exec"), namespace)
    executor = namespace["HbmRecoveryExecutor"](config)
    scheduler = namespace["HbmRecoveryAsyncScheduler"](config)
    return NS(
        ns=namespace,
        executor=executor,
        scheduler=scheduler,
        plan=plan,
        poll=poll,
        poll_outputs=poll_outputs,
        config=config,
        events=events,
        request=NS(instruction="retry", request_id="round"),
    )


def test_recovery_finishes_locally_before_rpc_returns_and_only_evicts_damaged(recovery):
    r = recovery
    result = r.executor.collective_rpc("handle_ft_command", args=(r.request,))
    assert result == ["original-result"]
    assert r.events == ["handle_ft_command", "poll", ("output", r.poll.kv_connector_output), "free", ("evict", {1})]
    assert r.scheduler.processed_step_seq == 3
    assert not r.scheduler.prev_step_scheduled_req_ids
    assert not r.scheduler.deferred_frees


def test_batch_outputs_consumed_once_and_never_deliver_stale_tokens(recovery):
    r = recovery
    output = NS(kv_connector_output=NS(is_empty=lambda: False), stale_tokens=[100])
    future = r.executor.collective_rpc("sample_tokens", non_block=True)
    future.set_result(output)
    r.executor.collective_rpc("handle_ft_command", args=(r.request,))
    assert output.kv_connector_output is None  # unchanged Core drain now skips it
    assert future.result() is output
    assert sum(isinstance(e, tuple) and e[0] == "output" for e in r.events) == 2
    assert not r.executor._pending_hbm_outputs


def test_normally_consumed_outputs_are_not_delivered_again_at_recovery(recovery):
    r = recovery
    output = NS(kv_connector_output=NS(is_empty=lambda: False))
    future = r.executor.collective_rpc("sample_tokens", non_block=True)
    future.set_result(output)
    assert r.scheduler.update_from_output("normal", output) == {0: "result"}
    assert not r.executor._pending_hbm_outputs
    r.events.clear()
    r.executor.collective_rpc("handle_ft_command", args=(r.request,))
    assert sum(isinstance(e, tuple) and e[0] == "output" for e in r.events) == 1


def test_failed_future_is_salvaged_by_aggregator_not_replayed(recovery):
    r = recovery
    future = r.executor.collective_rpc("execute_model", non_block=True)
    future.set_exception(RuntimeError("failed old step"))
    r.executor.collective_rpc("handle_ft_command", args=(r.request,))
    assert not r.executor._pending_hbm_outputs
    assert r.scheduler.kv_cache_manager.evict_blocks.call_args.args == ({1},)


@pytest.mark.parametrize("instruction", ["query_mask", "scale_down", "activate_cpu_group_timeouts"])
def test_other_ft_commands_keep_original_executor_behavior(recovery, instruction):
    r = recovery
    r.request.instruction = instruction
    assert r.executor.collective_rpc("handle_ft_command", args=(r.request,)) == ["original-result"]
    assert r.events == ["handle_ft_command"]
    r.scheduler.reset_connector_cache.assert_not_called()


@pytest.mark.parametrize(
    "failure", ["stale", "missing_scheduler", "invalid_block", "connector", "live_request", "retained"]
)
def test_cleanup_failure_propagates_into_unmodified_core_recovery_vote(recovery, failure):
    r = recovery
    if failure == "stale":
        r.plan["request_id"] = "old"
    if failure == "missing_scheduler":
        r.ns["_SCHEDULERS"].clear()
    if failure == "invalid_block":
        r.plan["blocks"] = [8]
    if failure == "connector":
        r.scheduler.reset_connector_cache.return_value = False
    if failure == "live_request":
        r.scheduler.requests = {"live": NS(is_finished=lambda: False)}
    if failure == "retained":
        r.scheduler.kv_cache_manager.block_pool.blocks[0].ref_cnt = 1
    path = ROOT.parent / "vllm/vllm/v1/fault_tolerance/engine_core_sentinel.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineCoreSentinel")
    method = next(n for n in cls.body if getattr(n, "name", None) == "_recover_and_vote")
    votes = []

    def tensor(values, **kwargs):
        votes.extend(values)
        return NS(item=lambda: values[0])

    namespace = dict(torch=NS(tensor=tensor, int32="int32"))
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    drain = Mock()
    sentinel = NS(
        engine=NS(model_executor=r.executor), _reinit_groups=Mock(), _drain_batch_queue=drain, _initial_dp_size=1
    )
    with pytest.raises(RuntimeError):
        namespace["_recover_and_vote"](sentinel, r.request, terminal=False)
    assert votes == [False]
    drain.assert_not_called()


def test_non_hbm_recompute_preserves_preempted_requests(recovery):
    r = recovery
    r.config.parallel_config.fault_tolerance_config.request_failure_policy = "recompute"
    req = NS(is_finished=lambda: False, status=STATUS.PREEMPTED, num_in_flight_tokens=0, request_id="r")
    r.scheduler.requests = {"r": req}
    r.plan.update(blocks=[], hbm_detected=False)
    r.executor.collective_rpc("handle_ft_command", args=(r.request,))
    assert r.scheduler.requests == {"r": req}
    r.scheduler.reset_connector_cache.assert_not_called()
    r.scheduler.kv_cache_manager.evict_blocks.assert_not_called()


def test_hbm_recompute_without_framework_abort_is_rejected(recovery):
    r = recovery
    r.config.parallel_config.fault_tolerance_config.request_failure_policy = "recompute"
    r.scheduler.requests = {"r": NS(is_finished=lambda: False)}
    with pytest.raises(RuntimeError, match="abort=True"):
        r.executor.collective_rpc("handle_ft_command", args=(r.request,))
    r.scheduler.kv_cache_manager.evict_blocks.assert_not_called()


@pytest.mark.parametrize(
    "enabled,backend,custom,async_,error",
    [
        (False, "ray", None, False, False),
        (True, "mp", None, True, False),
        (True, "mp", None, False, False),
        (True, "ray", None, False, True),
        (True, "mp", "another.Scheduler", False, True),
    ],
)
def test_platform_selects_existing_extension_points_without_overwriting_custom_classes(
    enabled, backend, custom, async_, error
):
    setup = load_method("vllm_ascend/platform.py", None, "_setup_hbm_recovery_extensions")
    cfg = NS(
        parallel_config=NS(enable_fault_tolerance=enabled, distributed_executor_backend=backend),
        scheduler_config=NS(scheduler_cls=custom, async_scheduling=async_),
    )
    if error:
        with pytest.raises(ValueError):
            setup(cfg)
        assert cfg.parallel_config.distributed_executor_backend == backend
    else:
        setup(cfg)
        setup(cfg)  # repeated platform validation must be idempotent
        if enabled:
            assert cfg.parallel_config.distributed_executor_backend.endswith("HbmRecoveryExecutor")
            assert cfg.scheduler_config.scheduler_cls.endswith("AsyncScheduler" if async_ else "RecoveryScheduler")
        else:
            assert cfg.parallel_config.distributed_executor_backend == backend


@pytest.mark.parametrize("complete", [False, True])
def test_recovery_delivers_only_quorum_completed_transfers_to_scheduler(recovery, complete):
    from dataclasses import dataclass, field
    from itertools import chain

    # Use the production aggregation counters, including failed-step salvage.
    vllm_root = ROOT.parent / "vllm"
    namespace = {"dataclass": dataclass, "field": field, "chain": chain, "logger": Mock()}
    for relative, name in (
        ("vllm/v1/outputs.py", "KVConnectorOutput"),
        ("vllm/distributed/kv_transfer/kv_connector/utils.py", "KVOutputAggregator"),
    ):
        path = vllm_root / relative
        node = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == name)
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    output_type = namespace["KVConnectorOutput"]
    aggregator = namespace["KVOutputAggregator"](2)
    aggregator.ft_defer_kv_connector_outputs([output_type(finished_recving={"aborted"})])
    outputs = [
        NS(kv_connector_output=output_type()),
        NS(kv_connector_output=output_type(finished_recving={"aborted"} if complete else set())),
    ]
    r = recovery
    r.executor.kv_output_aggregator = aggregator
    r.poll_outputs[:] = outputs
    r.executor.collective_rpc("handle_ft_command", args=(r.request,))
    delivered = [event[1] for event in r.events if isinstance(event, tuple) and event[0] == "output"]
    if complete:
        assert len(delivered) == 1
        assert delivered[0].finished_recving == {"aborted"}
    else:
        assert not delivered


@pytest.mark.parametrize("async_", [False, True])
def test_platform_preserves_recompute_scheduler_across_revalidation(async_):
    setup = load_method("vllm_ascend/platform.py", None, "_setup_hbm_recovery_extensions")
    base = "vllm_ascend.core.recompute_scheduler." + ("AsyncRecomputeScheduler" if async_ else "RecomputeScheduler")
    expected = "vllm_ascend.core.hbm_recovery.HbmRecovery" + (
        "AsyncRecomputeScheduler" if async_ else "RecomputeScheduler"
    )
    cfg = NS(
        parallel_config=NS(enable_fault_tolerance=True, distributed_executor_backend="mp"),
        scheduler_config=NS(scheduler_cls=base, async_scheduling=async_),
    )
    for current in (base, expected, base):
        cfg.scheduler_config.scheduler_cls = current
        setup(cfg)
        assert cfg.scheduler_config.scheduler_cls == expected
        assert cfg.parallel_config.distributed_executor_backend.endswith("HbmRecoveryExecutor")


@pytest.mark.parametrize("async_", [False, True])
def test_recompute_output_and_schedule_survive_hbm_mixin(recovery, async_):
    # Execute the real recompute output method and actual inheritance declarations,
    # mocking only the upstream scheduler and output records (no NPU imports).
    r = recovery
    tree = ast.parse((ROOT / "vllm_ascend/core/recompute_scheduler.py").read_text())
    base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RecomputeScheduler")
    base.body = [n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "update_from_output"]
    async_base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AsyncRecomputeScheduler")
    ns = dict(
        r.ns,
        EngineCoreOutput=lambda **kw: NS(**kw),
        EngineCoreOutputs=lambda: NS(outputs=[]),
        FinishReason=NS(STOP="stop"),
    )
    ns["AsyncScheduler"] = type("AsyncScheduler", (ns["Scheduler"],), {})
    ns["Scheduler"].schedule = lambda self: "base-schedule"
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), base, async_base],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "recompute_scheduler.py", "exec"), ns)
    ns["RecomputeScheduler"].schedule = lambda self: "recompute-schedule"
    tree = ast.parse((ROOT / "vllm_ascend/core/hbm_recovery.py").read_text())
    name = "HbmRecovery" + ("AsyncRecomputeScheduler" if async_ else "RecomputeScheduler")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "hbm_recovery.py", "exec"), ns)
    scheduler = ns[name](r.executor.vllm_config)
    assert scheduler.schedule() == "recompute-schedule"
    ns["Scheduler"].update_from_output = lambda self, scheduled, output: {0: NS(outputs=[])}
    future = r.executor.collective_rpc("execute_model", non_block=True)
    output = NS(kv_connector_output=None)
    future.set_result(output)
    scheduled = NS(recomputed_reqs=[NS(request_id="recompute-me", client_index=0)])
    result = scheduler.update_from_output(scheduled, output)
    assert result[0].outputs[0].request_id == "recompute-me"
    assert result[0].outputs[0].stop_reason == "recomputed"
    assert not r.executor._pending_hbm_outputs
    scheduler.finish_hbm_recovery(r.plan)
    scheduler.kv_cache_manager.evict_blocks.assert_called_once_with({1})


@pytest.mark.parametrize("between_components", [False, True])
def test_recovery_pair_survives_handshake_config_revalidation(recovery, between_components):
    r = recovery
    # Execute the instance-id assignment from the real VllmConfig post-init.
    # Remaining platform/model validation needs NPU dependencies; the original
    # handshake invokes this method after constructing both components.
    path = ROOT.parent / "vllm/vllm/config/vllm.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VllmConfig")
    post_init = next(n for n in cls.body if getattr(n, "name", None) == "__post_init__")
    assignment = next(
        n
        for n in post_init.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "instance_id" for t in n.targets)
    )
    if between_components:
        r.ns["_SCHEDULERS"].clear()
    exec(
        compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"),
        {"self": r.config, "time": NS(time_ns=lambda: 123456789)},
    )
    r.config.parallel_config.data_parallel_rank = 3
    if between_components:
        r.scheduler = r.ns["HbmRecoveryAsyncScheduler"](r.config)
    output = NS(kv_connector_output=None)
    future = r.executor.collective_rpc("execute_model", non_block=True)
    future.set_result(output)
    r.scheduler.update_from_output("normal", output)
    assert not r.executor._pending_hbm_outputs
    assert r.executor.collective_rpc("handle_ft_command", args=(r.request,)) == ["original-result"]
    r.scheduler.kv_cache_manager.evict_blocks.assert_called_once_with({1})


def test_registry_isolates_equal_configs_and_releases_dead_components(recovery):
    import gc
    from copy import deepcopy

    r = recovery
    other_config = deepcopy(r.config)
    assert other_config == r.config
    other_executor = r.ns["HbmRecoveryExecutor"](other_config)
    other_scheduler = r.ns["HbmRecoveryAsyncScheduler"](other_config)
    other_executor.collective_rpc("handle_ft_command", args=(r.request,))
    other_scheduler.kv_cache_manager.evict_blocks.assert_called_once_with({1})
    r.scheduler.kv_cache_manager.evict_blocks.assert_not_called()
    executor_ref, scheduler_ref = ref(other_executor), ref(other_scheduler)
    del other_executor, other_scheduler
    gc.collect()
    assert executor_ref() is None and scheduler_ref() is None
    assert len(r.ns["_EXECUTORS"]) == len(r.ns["_SCHEDULERS"]) == 1
    r.executor.collective_rpc("handle_ft_command", args=(r.request,))
    r.scheduler.kv_cache_manager.evict_blocks.assert_called_once_with({1})
