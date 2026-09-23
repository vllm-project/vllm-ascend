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
    @patch("vllm_ascend.platform._fix_incompatible_config")
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
        self.mock_npu_patcher = patch("torch.Tensor.npu", new=lambda self: self, create=True)
        self.mock_npu_patcher.start()
        os.environ["DYNAMIC_EPLB"] = "true"

    def tearDown(self):
        self.mock_npu_patcher.stop()
        os.environ.pop("DYNAMIC_EPLB", None)

    def test_init_eplb_config_with_eplb(self):
        eplb_config = init_ascend_config(self.vllm_config).eplb_config
        _, expert_map, log2phy, redundant_experts, phys_to_logical = init_eplb_config(eplb_config, 0, self.moe_config)
        # Full-length physical map: entry p is this rank's slot of physical
        # expert p (-1 when not owned), covering the redundant tail too.
        # rank 1 owns physical experts 5..9 in slots 0..4.
        gt_expert_map = torch.tensor([-1, -1, -1, -1, -1, 0, 1, 2, 3, 4])
        gt_log2phy = torch.tensor([8, 9, 2, 3, 4, 5, 6, 7])
        gt_phys_to_logical = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
        self.assertTrue(torch.equal(expert_map, gt_expert_map))
        self.assertTrue(torch.equal(log2phy, gt_log2phy))
        self.assertTrue(torch.equal(phys_to_logical, gt_phys_to_logical))
        self.assertEqual(redundant_experts, 2)

    def test_allgather_mask_indexing_covers_physical_topk_ids(self):
        # Regression test for issue #14080: TokenDispatcherWithAllGather
        # indexes expert_map with log2phy-mapped (physical) topk_ids, so the
        # map must cover the whole physical ID range [0, num_experts).
        eplb_config = init_ascend_config(self.vllm_config).eplb_config
        _, expert_map, log2phy, redundant_experts, _ = init_eplb_config(eplb_config, 0, self.moe_config)
        logical_topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        physical_topk_ids = log2phy[logical_topk_ids]

        self.assertTrue(torch.all(physical_topk_ids < expert_map.numel()))
        self.assertEqual(int(log2phy.max()), expert_map.numel() - 1)
        mask = expert_map[physical_topk_ids] != -1
        self.assertEqual(mask.shape, (2, 2))
        # rank 1 owns physical experts 5..9; logical 0,1 replicate there.
        self.assertTrue(torch.equal(mask, torch.tensor([[True, True], [False, False]])))
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
        _, expert_map, log2phy, redundant_experts, phys_to_logical = init_eplb_config(eplb_config, 0, self.moe_config)
        # Static EPLB placement [7,2,0,3,5] / [6,1,4,7,2]: the redundant
        # physical tail (IDs 8,9) replicates logical experts 7 and 2. The map
        # is positional (rank 1 owns physical 5..9 in slots 0..4); the layout
        # itself is carried by phys_to_logical.
        gt_expert_map = torch.tensor([-1, -1, -1, -1, -1, 0, 1, 2, 3, 4])
        gt_log2phy = torch.tensor([2, 6, 9, 3, 7, 4, 5, 8])
        gt_phys_to_logical = torch.tensor([7, 2, 0, 3, 5, 6, 1, 4, 7, 2])
        self.assertTrue(torch.equal(expert_map, gt_expert_map))
        self.assertTrue(torch.equal(log2phy, gt_log2phy))
        self.assertTrue(torch.equal(phys_to_logical, gt_phys_to_logical))
        self.assertEqual(redundant_experts, 2)

    def test_generate_log2phy_map_physical_layout_matches_logical_layout(self):
        # Legacy rows index logical experts; physical rows are full-length
        # (entry p = slot of physical expert p). Both must yield the same
        # logical-length map, and the output must stay logical-length.
        placement = torch.tensor([[7, 2, 0, 3, 5], [6, 1, 4, 7, 2]], dtype=torch.int32)
        phys_to_logical = placement.reshape(-1)
        logical_rows, physical_rows = [], []
        for rank, row in enumerate(placement):
            # Legacy layout: map entry l = slot of logical expert l.
            logical_map = torch.full((8,), -1, dtype=torch.int32)
            logical_map[row] = torch.arange(5, dtype=torch.int32)
            # Physical layout: positional, rank r owns phys [r*5, r*5+5).
            physical_map = torch.full((10,), -1, dtype=torch.int32)
            physical_map[rank * 5 : (rank + 1) * 5] = torch.arange(5, dtype=torch.int32)
            logical_rows.append(logical_map)
            physical_rows.append(physical_map)

        for ep_rank in range(2):
            legacy = generate_log2phy_map(logical_rows, ep_rank)
            physical = generate_log2phy_map(physical_rows, ep_rank, phys_to_logical=phys_to_logical)
            self.assertEqual(physical.shape[0], 8)
            self.assertTrue(torch.equal(legacy, physical))

    def test_generate_log2phy_map_physical_layout_with_tp_rotation(self):
        # tp_size replica rotation must agree with the legacy layout.
        # N=5, R=3, ep=4 -> 2 experts per rank.
        placement = torch.tensor([[0, 1], [2, 3], [4, 0], [1, 2]], dtype=torch.int32)
        phys_to_logical = placement.reshape(-1)
        legacy_rows, physical_rows = [], []
        for rank, row in enumerate(placement):
            legacy_map = torch.full((5,), -1, dtype=torch.int32)
            legacy_map[row] = torch.arange(2, dtype=torch.int32)
            physical_map = torch.full((8,), -1, dtype=torch.int32)
            physical_map[rank * 2 : (rank + 1) * 2] = torch.arange(2, dtype=torch.int32)
            legacy_rows.append(legacy_map)
            physical_rows.append(physical_map)

        legacy = generate_log2phy_map(legacy_rows, ep_rank=3, tp_size=4)
        physical = generate_log2phy_map(physical_rows, ep_rank=3, tp_size=4, phys_to_logical=phys_to_logical)

        self.assertEqual(physical.shape[0], 5)
        self.assertTrue(torch.equal(legacy, physical))

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
        _, expert_map, log2phy, redundant_experts, phys_to_logical = init_eplb_config(eplb_config, 0, self.moe_config)
        gt_expert_map = torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3])
        self.assertIsNone(log2phy)
        self.assertIsNone(phys_to_logical)
        self.assertTrue(torch.equal(expert_map, gt_expert_map))
        self.assertEqual(redundant_experts, 0)
