import os
import unittest
from unittest.mock import MagicMock, patch

# isort: off
import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.eplb.core.eplb_utils import generate_global_placement, generate_log2phy_map, init_eplb_config
# isort: on


class TestAscendConfig(unittest.TestCase):
    @patch("vllm.config.VllmConfig.__post_init__", MagicMock())
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def setUp(self, mock_fix_incompatible_config):
        vllm_config = VllmConfig()
        vllm_config.model_config = MagicMock()
        vllm_config.additional_config = {
            "refresh": True,
            "eplb_config": {"dynamic_eplb": True, "num_redundant_experts": 2},
        }
        from vllm.model_executor.layers.fused_moe.config import RoutingMethodType

        moe_parallel_config = FusedMoEParallelConfig(2, 0, 1, 2, 1, 1, 1, 1, 1, True, "hccl", enable_eplb=True)
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        moe_config = FusedMoEConfig(
            num_experts=8,
            experts_per_token=8,
            hidden_dim=8192,
            intermediate_size=10,
            num_local_experts=8,
            num_logical_experts=8,
            activation=MoEActivation.SILU,
            device="npu",
            routing_method=RoutingMethodType.Simulated,
            moe_parallel_config=moe_parallel_config,
            in_dtype=torch.float16,
        )
        moe_config.supports_eplb = True
        self.vllm_config = vllm_config
        self.moe_config = moe_config
        self.mock_npu_patcher = patch("torch.Tensor.npu", new=lambda self: self)
        self.mock_npu_patcher.start()
        os.environ["DYNAMIC_EPLB"] = "true"

    def tearDown(self):
        self.mock_npu_patcher.stop()
        os.environ.pop("DYNAMIC_EPLB", None)

    def test_init_eplb_config_with_eplb(self):
        eplb_config = init_ascend_config(self.vllm_config).eplb_config
        _, expert_map, log2phy, redundant_experts = init_eplb_config(eplb_config, 0, self.moe_config)
        gt_expert_map = torch.tensor([3, 4, -1, -1, -1, 0, 1, 2])
        gt_log2phy = torch.tensor([8, 9, 2, 3, 4, 5, 6, 7])
        self.assertTrue(torch.equal(expert_map, gt_expert_map))
        self.assertTrue(torch.equal(log2phy, gt_log2phy))
        self.assertEqual(redundant_experts, 2)

    def test_generate_global_placement_matches_vllm_physical_layout(self):
        placement = generate_global_placement(8, 2, 2, 0)

        self.assertTrue(
            torch.equal(
                placement,
                torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 0, 1]], dtype=torch.int32),
            )
        )

    def test_init_eplb_config_with_eplb_withmap(self):
        _TEST_DIR = os.path.dirname(__file__)
        self.vllm_config.additional_config["eplb_config"]["expert_map_path"] = _TEST_DIR + "/expert_map.json"
        eplb_config = init_ascend_config(self.vllm_config).eplb_config
        _, expert_map, log2phy, redundant_experts = init_eplb_config(eplb_config, 0, self.moe_config)
        gt_expert_map = torch.tensor([-1, 1, 4, -1, 2, -1, 0, 3])
        gt_log2phy = torch.tensor([2, 6, 9, 3, 7, 4, 5, 8])
        self.assertTrue(torch.equal(expert_map, gt_expert_map))
        self.assertTrue(torch.equal(log2phy, gt_log2phy))
        self.assertEqual(redundant_experts, 2)

    def test_generate_log2phy_map_rotates_tail_tp_rank_with_tp_size(self):
        global_expert_map = [
            torch.tensor([0, -1], dtype=torch.int32),
            torch.tensor([0, -1], dtype=torch.int32),
            torch.tensor([0, -1], dtype=torch.int32),
            torch.tensor([0, -1], dtype=torch.int32),
            torch.tensor([-1, 0], dtype=torch.int32),
            torch.tensor([-1, 0], dtype=torch.int32),
            torch.tensor([-1, 0], dtype=torch.int32),
            torch.tensor([-1, 0], dtype=torch.int32),
        ]

        fallback_tail_dp1 = generate_log2phy_map(global_expert_map, ep_rank=7)
        rotated_tail_dp0 = generate_log2phy_map(global_expert_map, ep_rank=3, tp_size=4)
        rotated_tail_dp1 = generate_log2phy_map(global_expert_map, ep_rank=7, tp_size=4)

        self.assertTrue(torch.equal(fallback_tail_dp1, torch.tensor([3, 7], dtype=torch.int32)))
        self.assertTrue(torch.equal(rotated_tail_dp0, torch.tensor([3, 4], dtype=torch.int32)))
        self.assertTrue(torch.equal(rotated_tail_dp1, torch.tensor([0, 5], dtype=torch.int32)))

    def test_init_eplb_config_without_eplb(self):
        self.vllm_config.additional_config = {"refresh": True}
        eplb_config = init_ascend_config(self.vllm_config).eplb_config
        _, expert_map, log2phy, redundant_experts = init_eplb_config(eplb_config, 0, self.moe_config)
        gt_expert_map = torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3])
        self.assertIsNone(log2phy)
        self.assertTrue(torch.equal(expert_map, gt_expert_map))
        self.assertEqual(redundant_experts, 0)
