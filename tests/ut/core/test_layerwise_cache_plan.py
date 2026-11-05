# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace

import pytest

from vllm_ascend.core.layerwise_cache_plan import (
    AliasRule,
    BackingPlan,
    CacheOwnerDescription,
    LayerwisePolicy,
    OwnerViewPlan,
    PhysicalCachePlan,
    ReuseDependency,
    TransferRegionPlan,
    validate_physical_plan,
)
from vllm_ascend.core.layerwise_cache_planner import materialize_layerwise_plan, plan_layerwise_cache


class TwoBackingAdapter:
    """Unregistered test adapter with fixed overhead and stepped alignment."""

    adapter_id = "test-two-backing"
    version = 1

    def build_layout(self, groups, policy, num_blocks):
        first = 32 + num_blocks * 64
        second = 16 + ((num_blocks * 24 + 63) // 64) * 64
        owners = tuple(
            CacheOwnerDescription(name, i, 0, False, "test", name) for i, name in enumerate(("a", "b", "c", "d"))
        )
        # c joins both predecessor components; d shares a's first region but
        # must wait for c, which itself has already waited for a and b.
        placements = (
            ("a", "main", "first", 32, 64),
            ("b", "index", "second", 16, 24),
            ("c", "main", "first", 32, 64),
            ("c", "index", "second", 16, 24),
            ("d", "main", "first", 32, 64),
        )
        views = tuple(
            OwnerViewPlan(owner, component, backing, offset, "uint8", 1, (num_blocks, stride), (stride, 1))
            for owner, component, backing, offset, stride in placements
        )
        regions = tuple(
            TransferRegionPlan(owner, component, backing, offset, stride, stride, 8)
            for owner, component, backing, offset, stride in placements
        )
        return PhysicalCachePlan(
            self.adapter_id,
            self.version,
            num_blocks,
            policy,
            "groups",
            (BackingPlan("first", first, 16, first), BackingPlan("second", second, 16, second)),
            owners,
            views,
            regions,
            (AliasRule("a", "c"), AliasRule("b", "c"), AliasRule("a", "d"), AliasRule("c", "d")),
            (ReuseDependency("c", ("a", "b")), ReuseDependency("d", ("c",))),
        )


def test_generic_exact_solver_and_fixed_capacity_rebuild():
    adapter = TwoBackingAdapter()
    policy = LayerwisePolicy(2, 2, ())
    plan = plan_layerwise_cache(adapter, [], policy, 1024)
    assert plan.required_bytes == 1024
    assert adapter.build_layout([], policy, plan.num_blocks + 1).required_bytes > 1024
    assert (
        adapter.build_layout([], policy, plan.num_blocks).required_bytes
        != adapter.build_layout([], policy, 1).required_bytes * plan.num_blocks
    )
    assert materialize_layerwise_plan(adapter, [], policy, plan.num_blocks, 1024) == plan
    assert plan.dependencies[0].predecessors == ("a", "b")


@pytest.mark.parametrize("corrupt", ["alias", "bounds", "alignment", "component", "cycle", "stride", "unrelated"])
def test_physical_validation_rejects_corruption(corrupt):
    plan = TwoBackingAdapter().build_layout([], LayerwisePolicy(2, 2, ()), 3)
    validate_physical_plan(plan)
    if corrupt == "alias":
        plan = replace(plan, aliases=())
    elif corrupt == "bounds":
        plan = replace(plan, views=(replace(plan.views[0], offset_bytes=1000), *plan.views[1:]))
    elif corrupt == "alignment":
        plan = replace(plan, views=(replace(plan.views[0], element_size=4, offset_bytes=33), *plan.views[1:]))
    elif corrupt == "component":
        plan = replace(plan, transfer_regions=plan.transfer_regions[:-1])
    elif corrupt == "cycle":
        plan = replace(plan, dependencies=(ReuseDependency("a", ("d",)), *plan.dependencies))
    elif corrupt == "stride":
        plan = replace(plan, views=(replace(plan.views[0], strides=(65, 1)), *plan.views[1:]))
    elif corrupt == "unrelated":
        plan = replace(plan, dependencies=(ReuseDependency("c", ("b",)), ReuseDependency("d", ("c",))))
    with pytest.raises(ValueError):
        validate_physical_plan(plan)
