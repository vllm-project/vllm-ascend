# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_ascend.distributed.memfabric.runtime import MemFabricRuntimeRegistry


def test_runtime_claims_reject_a_second_owner_before_native_initialization():
    registry = MemFabricRuntimeRegistry()
    lease = registry.acquire("offload", "sparse-kv", ("device0", "tp8", "pool-a"))

    with pytest.raises(RuntimeError, match="already claimed by owner='sparse-kv'"):
        registry.acquire("offload", "qwen-ple", ("device0", "tp8", "pool-a"))

    lease.release()


def test_runtime_claims_reference_count_same_owner_and_configuration():
    registry = MemFabricRuntimeRegistry()
    signature = ("device0", "tp8", "pool-a")
    first = registry.acquire("offload", "sparse-kv", signature)
    second = registry.acquire("offload", "sparse-kv", signature)

    first.release()
    with pytest.raises(RuntimeError, match="already claimed"):
        registry.acquire("offload", "qwen-ple", signature)

    second.release()
    with registry.acquire("offload", "qwen-ple", signature):
        pass


def test_runtime_claims_reject_same_owner_with_a_different_configuration():
    registry = MemFabricRuntimeRegistry()
    lease = registry.acquire("offload", "sparse-kv", ("device0", "tp8", "pool-a"))

    with pytest.raises(RuntimeError, match="already claimed"):
        registry.acquire("offload", "sparse-kv", ("device0", "tp4", "pool-a"))

    lease.release()


def test_runtime_claims_allow_independent_api_families():
    registry = MemFabricRuntimeRegistry()
    offload = registry.acquire("offload", "sparse-kv", ("device0", "tp8"))
    bm = registry.acquire("bm", "deepseek-engram", ("device0", "group-a"))

    bm.release()
    offload.release()


def test_runtime_claim_snapshot_is_sorted_and_reports_reference_count():
    registry = MemFabricRuntimeRegistry()
    first = registry.acquire("transfer_engine", "sfa", ("device0",))
    second = registry.acquire("transfer_engine", "sfa", ("device0",))

    assert registry.active_claims() == (("transfer_engine", "sfa", ("device0",), 2),)

    second.release()
    first.release()


def test_runtime_claim_release_is_idempotent():
    registry = MemFabricRuntimeRegistry()
    lease = registry.acquire("bm", "deepseek-engram", ("device0", "group-a"))

    lease.release()
    lease.release()
    next_lease = registry.acquire("bm", "qwen-ple", ("device0", "group-b"))
    next_lease.release()


def test_runtime_claim_rejects_unhashable_signatures():
    with pytest.raises(TypeError, match="signature must be hashable"):
        MemFabricRuntimeRegistry().acquire("offload", "sparse-kv", {"device": 0})
