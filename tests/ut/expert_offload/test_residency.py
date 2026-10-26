# SPDX-License-Identifier: Apache-2.0
import random

import pytest

from vllm_ascend.expert_offload.residency import ExpertResidency


def test_plan_is_transactional_and_hits_are_protected():
    residency = ExpertResidency(8, 3)
    plan = residency.plan([4, 2, 4])
    assert plan.loads == ((2, 0), (4, 1))
    assert residency.slot_to_expert == (None, None, None)
    residency.commit(plan)
    plan = residency.plan([4, 6, 7])
    assert plan.loads == ((6, 2), (7, 0))
    assert plan.slot_to_expert == (7, 4, 6)
    residency.commit(plan)
    assert residency.plan([7, 4, 6, 7]).loads == ()
    assert residency.logical_to_physical() == (-1, -1, -1, -1, 1, -1, 2, 0)


def test_overflow_leaves_decode_state_intact():
    residency = ExpertResidency(8, 2)
    residency.commit(residency.plan([1, 3]))
    before = residency.logical_to_physical()
    plan = residency.plan([1, 3, 5])
    assert plan.use_full_pool and not plan.loads
    residency.commit(plan)
    assert residency.logical_to_physical() == before


def test_stale_plan_is_rejected():
    residency = ExpertResidency(8, 2)
    first = residency.plan([1])
    stale = residency.plan([2])
    residency.commit(first)
    with pytest.raises(RuntimeError, match="stale"):
        residency.commit(stale)


@pytest.mark.parametrize("ids", [[-1], [8], [True], [1, True], [1.0], ["1"]])
def test_invalid_ids_are_rejected(ids):
    with pytest.raises(ValueError):
        ExpertResidency(8, 2).plan(ids)


@pytest.mark.parametrize("experts,slots", [(8, 0), (8, 8), (8, 9), (0, 1)])
def test_invalid_capacity(experts, slots):
    with pytest.raises(ValueError):
        ExpertResidency(experts, slots)


def test_randomized_placement_conserves_every_selected_expert():
    rng = random.Random(410)
    residency = ExpertResidency(32, 8)
    for _ in range(2000):
        selected = [rng.randrange(32) for _ in range(rng.randrange(13))]
        before = residency.slot_to_expert
        plan = residency.plan(selected)
        assert residency.slot_to_expert == before
        if len(set(selected)) > 8:
            assert plan.use_full_pool
        else:
            for eid in set(selected).intersection(before):
                assert plan.slot_to_expert.index(eid) == before.index(eid)
            residency.commit(plan)
            mapping = residency.logical_to_physical()
            assert all(mapping[eid] >= 0 for eid in selected)
            assert len({eid for eid in residency.slot_to_expert if eid is not None}) == sum(
                eid is not None for eid in residency.slot_to_expert
            )
