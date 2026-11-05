import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from vllm_ascend.eplb.core.eplb_worker import EplbWorker

INITIAL_EXPERT_MAPS = torch.tensor([[[0, 1, -1, -1], [-1, -1, 0, 1]]])
SWAPPED_PLACEMENT = [[[0, 2], [1, 3]]]
MOE_LOAD = torch.tensor([[[10.0, 1.0], [1.0, 1.0]]])


class StubPolicy:
    """Returns a fixed plan with a fixed `changed` flag."""

    def __init__(self, changed, new_placement=None):
        self.changed = changed
        self.new_placement = SWAPPED_PLACEMENT if new_placement is None else new_placement

    def rebalance_experts(self, current_expert_table, expert_workload):
        return self.changed, [0], self.new_placement


def make_worker(policy, shared_dict=None, policy_type=2):
    """Build a worker, with `policy` stubbed in unless it is None."""
    if shared_dict is None:
        shared_dict = {"expert_maps": INITIAL_EXPERT_MAPS.clone(), "moe_load": MOE_LOAD}
    with ExitStack() as stack:
        mock_ep_group = stack.enter_context(patch("vllm_ascend.eplb.core.eplb_worker.get_ep_group"))
        if policy is not None:
            stack.enter_context(
                patch("vllm_ascend.eplb.core.eplb_worker.PolicyFactory.generate_policy", return_value=policy)
            )
        mock_ep_group.return_value = MagicMock(rank_in_group=0)
        return EplbWorker(shared_dict=shared_dict, policy_type=policy_type)


def transfers(packed_update_info):
    """Every expert the packed plan asks this rank to send or receive."""
    moved = []
    for send_info, recv_info, _, _, _ in packed_update_info:
        moved += list(send_info) + list(recv_info)
    return moved


class TestEplbWorkerHonoursChangedFlag(unittest.TestCase):
    def test_plan_below_the_threshold_moves_nothing(self):
        worker = make_worker(StubPolicy(changed=0))

        packed_update_info = worker.do_update()

        self.assertEqual(transfers(packed_update_info), [])
        self.assertTrue(torch.equal(worker.old_expert_maps, INITIAL_EXPERT_MAPS))
        self.assertTrue(torch.equal(worker.shared_dict["expert_maps"], INITIAL_EXPERT_MAPS))

    def test_plan_above_the_threshold_is_applied(self):
        worker = make_worker(StubPolicy(changed=1))

        packed_update_info = worker.do_update()

        self.assertNotEqual(transfers(packed_update_info), [])
        self.assertFalse(torch.equal(worker.old_expert_maps, INITIAL_EXPERT_MAPS))

    def test_numpy_flags_are_honoured(self):
        self.assertEqual(transfers(make_worker(StubPolicy(changed=np.int64(0))).do_update()), [])
        self.assertEqual(transfers(make_worker(StubPolicy(changed=np.bool_(False))).do_update()), [])
        self.assertNotEqual(transfers(make_worker(StubPolicy(changed=np.int64(1))).do_update()), [])

    def test_unexpected_flag_is_treated_as_apply(self):
        worker = make_worker(StubPolicy(changed="unexpected"))

        packed_update_info = worker.do_update()

        self.assertNotEqual(transfers(packed_update_info), [])


class TestEplbWorkerWithRealPolicies(unittest.TestCase):
    """The same behaviour without a stub, so the flag is the one the shipped
    policies produce rather than the one the test asserts.

    Four ranks of two experts each. A balanced load gives DefaultEplb nothing
    worth 5%, and it still returns a *different* placement - which is exactly
    the plan that moves expert weights for no gain today. Putting both hot
    experts on one rank gives it a split worth making, and the flag is 1.
    """

    EXPERT_MAPS = torch.tensor(
        [
            [
                [0, 1, -1, -1, -1, -1, -1, -1],
                [-1, -1, 0, 1, -1, -1, -1, -1],
                [-1, -1, -1, -1, 0, 1, -1, -1],
                [-1, -1, -1, -1, -1, -1, 0, 1],
            ]
        ]
    )
    BALANCED_LOAD = torch.tensor([[[4.0, 4.0], [4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]])
    SPLITTABLE_LOAD = torch.tensor([[[10.0, 10.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]])

    def test_default_eplb_balanced_load_moves_nothing(self):
        worker = self._worker(self.BALANCED_LOAD, policy_type=1)

        self.assertEqual(transfers(worker.do_update()), [])
        self.assertTrue(torch.equal(worker.old_expert_maps, self.EXPERT_MAPS))

    def test_swift_balancer_balanced_load_moves_nothing(self):
        worker = self._worker(self.BALANCED_LOAD, policy_type=2)

        self.assertEqual(transfers(worker.do_update()), [])
        self.assertTrue(torch.equal(worker.old_expert_maps, self.EXPERT_MAPS))

    def test_default_eplb_still_rearranges_a_split_worth_making(self):
        worker = self._worker(self.SPLITTABLE_LOAD, policy_type=1)

        self.assertNotEqual(transfers(worker.do_update()), [])

    def test_swift_balancer_still_rearranges_a_split_worth_making(self):
        worker = self._worker(self.SPLITTABLE_LOAD, policy_type=2)

        self.assertNotEqual(transfers(worker.do_update()), [])

    def _worker(self, load, policy_type):
        return make_worker(
            None,
            shared_dict={"expert_maps": self.EXPERT_MAPS.clone(), "moe_load": load},
            policy_type=policy_type,
        )


class TestEplbWorkerKeepsRedundantPlacements(unittest.TestCase):
    """Keeping the placement round-trips it through global2local/local2global,
    which must be exact when an expert is replicated across ranks."""

    REDUNDANT_MAPS = torch.tensor([[[0, 1, -1], [0, -1, 1]]])

    def test_redundant_placement_is_preserved_exactly(self):
        self._assert_preserved(self.REDUNDANT_MAPS)

    def _assert_preserved(self, expert_maps):
        worker = make_worker(
            StubPolicy(changed=0, new_placement=SWAPPED_PLACEMENT),
            shared_dict={"expert_maps": expert_maps.clone(), "moe_load": torch.tensor([[[3.0, 1.0], [2.0, 1.0]]])},
        )

        packed_update_info = worker.do_update()

        self.assertEqual(transfers(packed_update_info), [])
        self.assertTrue(torch.equal(worker.old_expert_maps, expert_maps))
        self.assertTrue(torch.equal(worker.shared_dict["expert_maps"], expert_maps))


class TestEplbWorkerMissingLoad(unittest.TestCase):
    """Returning early queues None, which forward_before pops into an AttributeError."""

    def test_missing_load_keeps_the_placement_instead_of_returning_none(self):
        worker = make_worker(StubPolicy(changed=1), shared_dict={"expert_maps": INITIAL_EXPERT_MAPS.clone()})

        packed_update_info = worker.do_update()

        self.assertIsNotNone(packed_update_info)
        self.assertEqual(transfers(packed_update_info), [])
        self.assertTrue(torch.equal(worker.old_expert_maps, INITIAL_EXPERT_MAPS))


if __name__ == "__main__":
    unittest.main()
