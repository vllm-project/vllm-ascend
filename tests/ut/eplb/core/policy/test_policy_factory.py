import unittest

import torch

from vllm_ascend.eplb.core.eplb_worker import EplbWorker
from vllm_ascend.eplb.core.policy.policy_factory import PolicyFactory
from vllm_ascend.eplb.core.policy.policy_flashlb import generate_layered_experts


class TestEplbRebalancePolicies(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.current_expert_table = generate_layered_experts()
        x = torch.rand(100, 58, 32, 9)
        x = x**10
        self.expert_workload = (x * 999 + 1).long()
        self.hotness = EplbWorker._calculate_hotness(self.current_expert_table, self.expert_workload.sum(0))

    @unittest.mock.patch("torch.npu.device_count", return_value=16)
    def test_swift_balance_rebalance_experts(self, mock_count):
        swift_policy = PolicyFactory.generate_policy(2)
        _, _, new_placement = swift_policy.rebalance_experts(self.current_expert_table, self.expert_workload.sum(0))
        update_mean, _ = EplbWorker._compute_imbalance(new_placement, self.hotness)

        self.assertLessEqual(update_mean, 1.08)

    def test_flashlb_rebalance_experts(self):
        flashlb_policy = PolicyFactory.generate_policy(3)
        _, _, new_placement = flashlb_policy.rebalance_experts(self.current_expert_table, self.expert_workload)
        update_mean, _ = EplbWorker._compute_imbalance(new_placement, self.hotness)

        self.assertLessEqual(update_mean, 1.1)

    def test_default_policy_rebalance_experts_with_new_ep_size(self):
        num_layers = 58
        default_policy = PolicyFactory.generate_policy(1)

        # Scale Down
        default_policy.set_new_ep_size(new_ep_size=30)
        _, _, new_placement = default_policy.rebalance_experts(self.current_expert_table, self.expert_workload.sum(0))
        self.assertTrue(all([len(new_placement[layer_idx]) == 30 for layer_idx in range(num_layers)]))

        # Scale Up
        default_policy.set_new_ep_size(new_ep_size=40)
        _, _, new_placement = default_policy.rebalance_experts(self.current_expert_table, self.expert_workload.sum(0))
        self.assertTrue(all([len(new_placement[layer_idx]) == 40 for layer_idx in range(num_layers)]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
