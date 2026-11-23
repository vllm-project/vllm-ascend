# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression contracts for HBM-only prefix eviction before RPC return."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
from weakref import WeakValueDictionary

import pytest

ROOT = Path(__file__).parents[4]


def load_extensions():
    class Scheduler:
        def __init__(self, config):
            self.vllm_config = config

    class Executor:
        def collective_rpc(self, method, **kwargs):
            self.calls.append(method)
            return self.plans if callable(method) else "worker-complete"

    ns = dict(
        Scheduler=Scheduler,
        AsyncScheduler=Scheduler,
        RecomputeScheduler=Scheduler,
        AsyncRecomputeScheduler=Scheduler,
        MultiprocExecutor=Executor,
        WeakValueDictionary=WeakValueDictionary,
        logger=Mock(),
    )
    tree = ast.parse((ROOT / "vllm_ascend/core/hbm_recovery.py").read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, "hbm_recovery.py", "exec"), ns)
    config = NS()
    scheduler = ns["HbmRecoveryScheduler"](config)
    executor = ns["HbmRecoveryExecutor"]()
    executor.vllm_config = config
    executor.calls = []
    executor.plans = [dict(request_id="round", blocks=[3], hbm_detected=True)] * 2
    # Aborted requests already have zero references, but their hashes survive.
    hashes = {"damaged-prefix": 3, "damaged-partial-alias": 3, "healthy-prefix": 4}

    def evict(blocks):
        for key, block in list(hashes.items()):
            if block in blocks:
                del hashes[key]

    scheduler.kv_cache_manager = NS(block_pool=NS(blocks=[None] * 8), evict_blocks=Mock(side_effect=evict))
    return executor, scheduler, hashes


def request(instruction="retry", abort=True):
    return NS(instruction=instruction, request_id="round", params={"abort": abort})


def test_cleared_prefix_and_aliases_evicted_before_rpc_returns():
    executor, scheduler, hashes = load_extensions()
    assert executor.collective_rpc("handle_ft_command", args=(request(),)) == "worker-complete"
    assert hashes == {"healthy-prefix": 4}
    scheduler.kv_cache_manager.evict_blocks.assert_called_once_with({3})


@pytest.mark.parametrize("mode", ["link", "scale_down", "normal"])
def test_non_hbm_paths_do_not_evict_or_reset_connector(mode):
    executor, scheduler, hashes = load_extensions()
    scheduler.reset_connector_cache = Mock(side_effect=AssertionError("must not reset connector"))
    if mode == "link":
        executor.plans = [dict(request_id="round", blocks=[], hbm_detected=False)] * 2
    method = "execute_model" if mode == "normal" else "handle_ft_command"
    executor.collective_rpc(method, args=(request("scale_down" if mode == "scale_down" else "retry"),))
    scheduler.kv_cache_manager.evict_blocks.assert_not_called()
    scheduler.reset_connector_cache.assert_not_called()
    assert len(hashes) == 3


@pytest.mark.parametrize("failure", ["missing", "stale", "disagree", "invalid_id", "scheduler", "evict", "no_abort"])
def test_failure_prevents_successful_rpc_return(failure):
    executor, scheduler, _ = load_extensions()
    if failure == "missing":
        executor.plans = []
    elif failure == "stale":
        executor.plans[0]["request_id"] = "previous-round"
    elif failure == "disagree":
        executor.plans = [executor.plans[0], dict(executor.plans[0], blocks=[4])]
    elif failure == "invalid_id":
        executor.plans[0]["blocks"] = [8]
    elif failure == "scheduler":
        executor.vllm_config = NS()
    elif failure == "evict":
        scheduler.kv_cache_manager.evict_blocks.side_effect = RuntimeError("eviction failed")
    with pytest.raises(RuntimeError):
        executor.collective_rpc("handle_ft_command", args=(request(abort=failure != "no_abort"),))


def test_hbm_in_other_dp_does_not_evict_unrelated_local_blocks():
    executor, scheduler, hashes = load_extensions()
    executor.plans[0]["blocks"] = []
    executor.collective_rpc("handle_ft_command", args=(request(),))
    scheduler.kv_cache_manager.evict_blocks.assert_called_once_with(set())
    assert len(hashes) == 3


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_platform_preserves_scheduler_kind_on_revalidation(asynchronous, recompute):
    tree = ast.parse((ROOT / "vllm_ascend/platform.py").read_text())
    node = next(n for n in tree.body if getattr(n, "name", "") == "_setup_hbm_recovery_extensions")
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "platform.py", "exec"), ns)
    original = None
    if recompute:
        original = "vllm_ascend.core.recompute_scheduler." + (
            "AsyncRecomputeScheduler" if asynchronous else "RecomputeScheduler"
        )
    config = NS(
        parallel_config=NS(enable_fault_tolerance=True, distributed_executor_backend="mp"),
        scheduler_config=NS(async_scheduling=asynchronous, scheduler_cls=original),
    )
    setup = ns["_setup_hbm_recovery_extensions"]
    setup(config)
    selected = config.scheduler_config.scheduler_cls
    setup(config)
    assert config.scheduler_config.scheduler_cls == selected
    assert ("Async" in selected) == asynchronous
    assert ("Recompute" in selected) == recompute
