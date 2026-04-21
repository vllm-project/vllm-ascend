import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.eplb.eplb_updator import EplbUpdator


class TestEplbUpdatorComputeAndSetMoeLoad(unittest.TestCase):
    def setUp(self):
        # ====================== 1. Mock environment ======================
        self.rank = 0
        self.world_size = 4
        self.device = torch.device("cpu")

        # mock dist
        p1 = patch("torch.distributed.get_rank", return_value=self.rank)
        p2 = patch("torch.distributed.get_world_size", return_value=self.world_size)
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        p1.start()
        p2.start()

        # ====================== 2. Mock comm group ======================
        self.mock_comm_group = MagicMock()

        def mock_all_gather(tensor, dim):
            gathered = torch.cat([tensor for _ in range(self.world_size)], dim=dim)
            return gathered

        self.mock_comm_group.all_gather = mock_all_gather

        p3 = patch("vllm_ascend.eplb.eplb_updator.get_dynamic_eplb_group", return_value=self.mock_comm_group)
        self.addCleanup(p3.stop)
        p3.start()

        # ====================== 3. Mock EplbUpdator ======================
        self.eplb_config = MagicMock()
        self.loader = MagicMock()
        self.eplb_process = MagicMock()
        self.process = MagicMock()
        self.eplb_process.shared_dict = {}

        self.updator = EplbUpdator(
            eplb_config=self.eplb_config, loader=self.loader, eplb_process=self.eplb_process, process=self.process
        )

        # ====================== 4. Mock adaptor ======================
        self.adaptor = MagicMock()
        self.adaptor.num_moe_layers = 4
        self.adaptor.num_dense_layers = 2
        self.mock_local_load = torch.randn(58, 100, 8, device=self.device)
        self.adaptor.get_rank_expert_workload.return_value = self.mock_local_load

        self.updator.set_adaptor(self.adaptor)

    def test_compute_and_set_moe_load_normal(self):
        self.updator.multi_stage = False

        moe_load = self.updator.compute_and_set_moe_load()

        self.assertEqual(moe_load.shape, (58, self.world_size, 100, 8))
        self.assertTrue("moe_load" in self.updator.shared_dict)
        self.assertEqual(moe_load.device.type, "cpu")
        self.assertEqual(moe_load.shape[1], self.world_size)

    def test_compute_and_set_moe_load_multi_stage(self):
        self.updator.multi_stage = True

        moe_load = self.updator.compute_and_set_moe_load()

        self.assertEqual(moe_load.shape, (100, 58, self.world_size, 8))
        self.assertTrue("moe_load" in self.updator.shared_dict)
        self.assertEqual(moe_load.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
