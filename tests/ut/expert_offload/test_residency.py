# SPDX-License-Identifier: Apache-2.0
from vllm_ascend.expert_offload.residency import ExpertResidency


def test_decode_replacement_protects_hits_and_commits_mapping():
    residency = ExpertResidency(8, 3)
    plan = residency.plan([4, 2, 4])
    assert plan.loads == ((2, 0), (4, 1))
    assert residency.slot_to_expert == (None, None, None)
    residency.commit(plan)
    plan = residency.plan([4, 6, 7])
    assert plan.loads == ((6, 2), (7, 0))
    assert plan.slot_to_expert == (7, 4, 6)
    residency.commit(plan)
    assert not residency.plan([7, 4, 6, 7]).loads
    assert residency.logical_to_physical() == (-1, -1, -1, -1, 1, -1, 2, 0)


def test_full_pool_fallback_preserves_decode_residency():
    residency = ExpertResidency(8, 2)
    residency.commit(residency.plan([1, 3]))
    before = residency.logical_to_physical()
    plan = residency.plan([1, 3, 5])
    assert plan.use_full_pool and not plan.loads
    residency.commit(plan)
    assert residency.logical_to_physical() == before
