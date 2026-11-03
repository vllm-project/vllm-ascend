# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import pytest

from vllm_ascend.distributed.memfabric import OffloadInstanceBinding, plan_offload_instances


def test_instance_plan_is_independent_of_owner_order():
    stores = {
        "ple": "tcp://127.0.0.1:8502",
        "sparse_kv": "tcp://127.0.0.1:8500",
        "engram": "tcp://127.0.0.1:8501",
    }
    plan = plan_offload_instances(stores)

    assert list(plan) == ["engram", "ple", "sparse_kv"]
    assert {owner: binding.initialization_order for owner, binding in plan.items()} == {
        "engram": 0,
        "ple": 1,
        "sparse_kv": 2,
    }
    assert plan == plan_offload_instances(dict(reversed(stores.items())))
    assert plan["ple"] == OffloadInstanceBinding("ple", stores["ple"], 1)


def test_instance_plan_does_not_allocate_native_ids():
    plan = plan_offload_instances({"ple": "tcp://127.0.0.1:8501"})

    assert plan["ple"] == OffloadInstanceBinding("ple", "tcp://127.0.0.1:8501", 0)
    assert not hasattr(plan["ple"], "pool_id")


def test_instance_plan_rejects_endpoint_conflicts_and_invalid_owner():
    with pytest.raises(ValueError, match="distinct store_url"):
        plan_offload_instances({"kv": "tcp://127.0.0.1:8500", "ple": "tcp://127.0.0.1:8500"})
    with pytest.raises(ValueError, match="owner"):
        plan_offload_instances({"": "tcp://127.0.0.1:8501"})
    with pytest.raises(ValueError, match="store_url"):
        plan_offload_instances({"ple": "http://127.0.0.1:8501"})


def test_instance_plan_respects_native_instance_limit():
    stores = {f"owner-{i:02}": f"tcp://127.0.0.1:{8500 + i}" for i in range(64)}

    with pytest.raises(ValueError, match="at most 63"):
        plan_offload_instances(stores)
    with pytest.raises(ValueError, match="at most 63"):
        plan_offload_instances({"ple": "tcp://127.0.0.1:8501"}, reserved_instances=63)
    assert len(plan_offload_instances({"ple": "tcp://127.0.0.1:8501"}, reserved_instances=62)) == 1
    assert (
        plan_offload_instances({"ple": "tcp://127.0.0.1:8501"}, reserved_instances=62)["ple"].initialization_order == 62
    )
    with pytest.raises(ValueError, match="reserved_instances"):
        plan_offload_instances({}, reserved_instances=True)


def test_instance_binding_validates_owner_and_endpoint():
    class Owner(str, Enum):
        PLE = "ple"

    binding = OffloadInstanceBinding(Owner.PLE, "tcp://127.0.0.1:8501", 0)
    assert binding.owner == "ple"
    for invalid in ("", "http://127.0.0.1:8501"):
        with pytest.raises(ValueError):
            OffloadInstanceBinding("ple", invalid, 0)
    for invalid_order in (True, -1, 1.5, 63):
        with pytest.raises(ValueError, match="initialization_order"):
            OffloadInstanceBinding("ple", "tcp://127.0.0.1:8501", invalid_order)
