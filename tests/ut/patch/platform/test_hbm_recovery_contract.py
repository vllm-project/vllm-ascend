# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts for Worker retry consensus and UCE classification."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
from typing import cast
from unittest.mock import Mock

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


@pytest.mark.parametrize(
    "start,size,expected",
    [(100, 100, [0]), (150, 100, [0, 1]), (50, 50, None), (99, 2, None), (299, 2, None), (100, 0, None)],
)
def test_uce_range_requires_complete_kv_coverage(start, size, expected):
    covered = load_method(
        "vllm_ascend/worker/sentinel/kv_cache_region_registry.py", "KvCacheRegionRegistry", "_covered_allocations"
    )
    registry = NS(regions=[NS(data_ptr=100, nbytes=100), NS(data_ptr=200, nbytes=100)])
    if expected is None:
        with pytest.raises(ValueError):
            covered(registry, start, size)
    else:
        assert covered(registry, start, size) == expected


def test_mixed_kv_and_non_kv_ranges_never_mark_partial_storage_safe():
    path = "vllm_ascend/worker/sentinel/kv_cache_region_registry.py"
    covered = load_method(path, "KvCacheRegionRegistry", "_covered_allocations")
    recover = load_method(path, "KvCacheRegionRegistry", "recover_uce_addresses")
    registry = NS(regions=[NS(data_ptr=100, nbytes=100)], uce_ranges=[], _update_tensor_to_safe=Mock())
    registry._covered_allocations = lambda start, size: covered(registry, start, size)
    assert recover(registry, [{"ptr": 110, "size": 1}, {"ptr": 199, "size": 2}]) is None
    registry._update_tensor_to_safe.assert_not_called()
    assert registry.uce_ranges == []
