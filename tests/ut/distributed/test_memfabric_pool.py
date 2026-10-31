# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

import pytest

from vllm_ascend.distributed.memfabric import OffloadPoolBinding, plan_offload_pools


def test_pool_plan_is_independent_of_owner_order_and_keeps_default_pool():
    stores = {
        "ple": "tcp://127.0.0.1:8502",
        "sparse_kv": "tcp://127.0.0.1:8500",
        "engram": "tcp://127.0.0.1:8501",
    }
    plan = plan_offload_pools(stores, default_pool_owner="sparse_kv")

    assert plan == plan_offload_pools(dict(reversed(stores.items())), default_pool_owner="sparse_kv")
    assert {owner: binding.pool_id for owner, binding in plan.items()} == {"sparse_kv": 0, "engram": 1, "ple": 2}
    assert plan["ple"] == OffloadPoolBinding("ple", 2, stores["ple"])
    assert plan_offload_pools({"ple": stores["ple"]})["ple"].pool_id == 1


def test_pool_plan_rejects_endpoint_conflicts_and_missing_default_owner():
    with pytest.raises(ValueError, match="distinct store_url"):
        plan_offload_pools({"kv": "tcp://127.0.0.1:8500", "ple": "tcp://127.0.0.1:8500"})
    with pytest.raises(ValueError, match="default_pool_owner"):
        plan_offload_pools({"ple": "tcp://127.0.0.1:8501"}, default_pool_owner="kv")


def test_pool_plan_respects_native_id_limit():
    stores = {f"owner-{i:02}": f"tcp://127.0.0.1:{8500 + i}" for i in range(64)}
    plan = plan_offload_pools(stores, default_pool_owner="owner-00")
    assert {binding.pool_id for binding in plan.values()} == set(range(64))
    with pytest.raises(ValueError, match="at most 63"):
        plan_offload_pools(stores)


def test_pool_binding_validates_and_normalizes_native_id():
    class PoolId(IntEnum):
        PLE = 1

    binding = OffloadPoolBinding("ple", PoolId.PLE, "tcp://127.0.0.1:8501")
    assert type(binding.pool_id) is int
    for invalid_id in (True, -1, 64):
        with pytest.raises(ValueError, match="pool_id"):
            OffloadPoolBinding("ple", invalid_id, binding.store_url)
