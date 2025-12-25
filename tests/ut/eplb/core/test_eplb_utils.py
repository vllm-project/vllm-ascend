import os
import unittest
from unittest.mock import patch

# isort: off
import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.config import (FusedMoEConfig,
                                                         FusedMoEParallelConfig
                                                         )

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.eplb.core.eplb_utils import init_eplb_config
# isort: on


class TestAscendConfig(unittest.TestCase):

    def setUp(self):
        vllm_config = VllmConfig()
        vllm_config.additional_config = {
            "refresh": True,
            "eplb_config": {
                "dynamic_eplb": True,
                "num_redundant_experts": 2
            }
        }
        moe_parallel_config = FusedMoEParallelConfig(2, 0, 1, 2, 1, 1, 1, 1,
                                                     True, "hccl")
        moe_config = FusedMoEConfig(8, 8, 8192, 5, moe_parallel_config,
                                    torch.float16)
        moe_config.supports_eplb = True
        self.vllm_config = vllm_config
        self.moe_config = moe_config
        self.mock_npu = patch("torch.Tensor.npu",
                              new=lambda self: self).start()
        self.rank = 1

    def test_init_eplb_config_with_eplb(self):
        eplb_config = init_ascend_config(self.vllm_config).eplb_config
        expert_map, log2phy, redundant_experts = init_eplb_config(
            eplb_config, 0, self.moe_config)
        gt_expert_map = torch.tensor([4, -1, -1, -1, 0, 1, 2, 3])
        gt_log2phy = torch.tensor([9, 1, 2, 3, 5, 6, 7, 8])
        self.assertTrue(torch.equal(expert_map[self.rank], gt_expert_map))
        self.assertTrue(torch.equal(log2phy, gt_log2phy))
        self.assertEqual(redundant_experts, 2)

    def test_init_eplb_config_with_eplb_withmap(self):
        _TEST_DIR = os.path.dirname(__file__)
        self.vllm_config.additional_config["eplb_config"][
            "expert_map_path"] = _TEST_DIR + "/expert_map.json"
        eplb_config = init_ascend_config(self.vllm_config).eplb_config
        expert_map, log2phy, redundant_experts = init_eplb_config(
            eplb_config, 0, self.moe_config)
        gt_expert_map = torch.tensor([-1, 1, 4, -1, 2, -1, 0, 3])
        gt_log2phy = torch.tensor([2, 6, 9, 3, 7, 4, 5, 8])
        self.assertTrue(torch.equal(expert_map[self.rank], gt_expert_map))
        self.assertTrue(torch.equal(log2phy, gt_log2phy))
        self.assertEqual(redundant_experts, 2)

    def test_init_eplb_config_without_eplb(self):
        self.vllm_config.additional_config = {"refresh": True}
        eplb_config = init_ascend_config(self.vllm_config).eplb_config
        expert_map, log2phy, redundant_experts = init_eplb_config(
            eplb_config, 0, self.moe_config)
        gt_expert_map = torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3])
        print(expert_map, log2phy, redundant_experts)
        self.assertTrue(torch.equal(expert_map[self.rank], gt_expert_map))
        self.assertEqual(redundant_experts, 0)
