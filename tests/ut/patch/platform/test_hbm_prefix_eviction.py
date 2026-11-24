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


@pytest.mark.parametrize("damaged_index", [0, 1])
def test_warmed_prompt_recomputed_after_recovery(damaged_index):
    """Use the native cache index to reject a cleared page on the same prompt."""
    # Keep native dependencies isolated from the lightweight RPC contract tests.
    torch = pytest.importorskip("torch")
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.request import Request

    block_size = 16
    tokens = list(range(2 * block_size + 1))
    init_none_hash(sha256)

    def prompt(request_id):
        return Request(
            request_id=request_id,
            prompt_token_ids=tokens.copy(),
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_hasher=get_request_block_hasher(block_size, sha256),
        )

    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=16,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
        ),
        max_model_len=64,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=True,
    )
    warm = prompt("warm")
    assert manager.allocate_slots(warm, num_new_tokens=len(tokens)) is not None
    ids = manager.get_block_ids(warm.request_id)[0]
    kv_bytes = torch.ones((16, block_size), dtype=torch.uint8)
    manager.cache_blocks(warm, len(tokens))
    manager.free(warm)
    cached, num_cached, *_ = manager.get_computed_blocks(prompt("before"))
    assert num_cached == 2 * block_size
    damaged_id = ids[damaged_index]
    assert damaged_id in cached.get_block_ids()[0]

    kv_bytes[damaged_id].zero_()
    assert not kv_bytes[damaged_id].any()
    # Releasing requests and clearing bytes alone leaves a stale cache hit.
    assert manager.get_computed_blocks(prompt("stale"))[1] == 2 * block_size
    executor, scheduler, _ = load_extensions()
    scheduler.kv_cache_manager = manager
    executor.plans = [dict(request_id="round", blocks=[damaged_id], hbm_detected=True)] * 2
    executor.collective_rpc("handle_ft_command", args=(request(),))

    repeated = prompt("after")
    cached, num_cached, *_ = manager.get_computed_blocks(repeated)
    assert num_cached == damaged_index * block_size
    assert damaged_id not in cached.get_block_ids()[0]
    assert (
        manager.allocate_slots(
            repeated,
            num_new_tokens=len(tokens) - num_cached,
            num_new_computed_tokens=num_cached,
            new_computed_blocks=cached,
        )
        is not None
    )
    # The damaged suffix is scheduled for recomputation, not reused as KV.
    manager.cache_blocks(repeated, len(tokens))
    manager.free(repeated)
    assert manager.get_computed_blocks(prompt("rewarmed"))[1] == 2 * block_size
