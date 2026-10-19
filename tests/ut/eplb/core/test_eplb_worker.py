import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.eplb.core.eplb_worker import EplbWorker

INITIAL_EXPERT_MAPS = torch.tensor([[[0, 1, -1, -1], [-1, -1, 0, 1]]])
SWAPPED_PLACEMENT = [[[0, 2], [1, 3]]]
MOE_LOAD = torch.tensor([[[10.0, 1.0], [1.0, 1.0]]])


class StubPolicy:
    """Returns a fixed plan and records the placement it was asked to improve."""

    def __init__(self, changed, new_placement):
        self.changed = changed
        self.new_placement = new_placement
        self.seen_placements = []

    def rebalance_experts(self, current_expert_table, expert_workload):
        self.seen_placements.append(current_expert_table.clone())
        return self.changed, [0], self.new_placement


def make_worker(policy, record_only=False):
    with (
        patch("vllm_ascend.eplb.core.eplb_worker.get_ep_group") as mock_ep_group,
        patch("vllm_ascend.eplb.core.eplb_worker.PolicyFactory.generate_policy", return_value=policy),
    ):
        mock_ep_group.return_value = MagicMock(rank_in_group=0)
        worker = EplbWorker(
            shared_dict={"expert_maps": INITIAL_EXPERT_MAPS.clone(), "moe_load": MOE_LOAD},
            policy_type=2,
            record_only=record_only,
        )
    return worker


class TestEplbWorkerRecordOnly(unittest.TestCase):
    """While recording no weight moves, so the next plan must start from the
    installed placement rather than from the last plan."""

    def test_record_only_keeps_planning_from_installed_placement(self):
        policy = StubPolicy(changed=1, new_placement=SWAPPED_PLACEMENT)
        worker = make_worker(policy, record_only=True)

        worker.do_update()
        worker.do_update()

        self.assertEqual(len(policy.seen_placements), 2)
        self.assertTrue(torch.equal(policy.seen_placements[0], policy.seen_placements[1]))
        self.assertTrue(torch.equal(worker.old_expert_maps, INITIAL_EXPERT_MAPS))

    def test_record_only_still_publishes_the_plan_for_export(self):
        policy = StubPolicy(changed=1, new_placement=SWAPPED_PLACEMENT)
        worker = make_worker(policy, record_only=True)

        worker.do_update()

        published = worker.shared_dict["expert_maps"]
        self.assertFalse(torch.equal(published, INITIAL_EXPERT_MAPS))

    def test_without_record_only_the_plan_becomes_the_new_basis(self):
        policy = StubPolicy(changed=1, new_placement=SWAPPED_PLACEMENT)
        worker = make_worker(policy, record_only=False)

        worker.do_update()
        worker.do_update()

        self.assertEqual(len(policy.seen_placements), 2)
        self.assertFalse(torch.equal(policy.seen_placements[0], policy.seen_placements[1]))
        self.assertFalse(torch.equal(worker.old_expert_maps, INITIAL_EXPERT_MAPS))


if __name__ == "__main__":
    unittest.main()
