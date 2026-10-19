import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.eplb.eplb_updator import EplbUpdator


class TestEplbUpdatorComputeAndSetMoeLoad(unittest.TestCase):
    def setUp(self):
        # Mock environment, comm group, eplb updator, adaptor
        self.rank = 0
        self.world_size = 4
        self.device = torch.device("cpu")

        p1 = patch("torch.distributed.get_rank", return_value=self.rank)
        p2 = patch("torch.distributed.get_world_size", return_value=self.world_size)
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        p1.start()
        p2.start()

        self.mock_comm_group = MagicMock()

        def mock_all_gather(tensor, dim):
            gathered = torch.cat([tensor for _ in range(self.world_size)], dim=dim)
            return gathered

        self.mock_comm_group.all_gather = mock_all_gather

        p3 = patch("vllm_ascend.eplb.eplb_updator.get_dynamic_eplb_group", return_value=self.mock_comm_group)
        self.addCleanup(p3.stop)
        p3.start()

        self.mock_pp = MagicMock()
        self.mock_pp.rank_in_group = 0
        p4 = patch("vllm.distributed.parallel_state._PP", self.mock_pp)
        self.addCleanup(p4.stop)
        p4.start()

        self.eplb_config = MagicMock()
        self.loader = MagicMock()
        self.eplb_process = MagicMock()
        self.process = MagicMock()
        self.eplb_process.shared_dict = {}

        self.updator = EplbUpdator(
            eplb_config=self.eplb_config, loader=self.loader, eplb_process=self.eplb_process, process=self.process
        )

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


class TestEplbUpdatorRecordMode(unittest.TestCase):
    """forward_before must not start a D2D transfer while recording."""

    HEAT_INTERVAL = 50
    ALGORITHM_INTERVAL = 50
    NUM_MOE_LAYERS = 4

    def _make_updator(self, record_path):
        p1 = patch("torch.distributed.get_rank", return_value=0)
        p2 = patch("torch.distributed.get_world_size", return_value=4)
        p3 = patch("vllm_ascend.eplb.eplb_updator.get_dynamic_eplb_group", return_value=MagicMock())
        mock_pp = MagicMock()
        mock_pp.rank_in_group = 0
        p4 = patch("vllm.distributed.parallel_state._PP", mock_pp)
        for p in (p1, p2, p3, p4):
            self.addCleanup(p.stop)
            p.start()

        eplb_config = MagicMock()
        eplb_config.expert_heat_collection_interval = self.HEAT_INTERVAL
        eplb_config.algorithm_execution_interval = self.ALGORITHM_INTERVAL
        eplb_config.expert_map_record_path = record_path
        eplb_config.eplb_policy_type = 2

        loader = MagicMock()
        eplb_process = MagicMock()
        eplb_process.shared_dict = {}

        updator = EplbUpdator(eplb_config=eplb_config, loader=loader, eplb_process=eplb_process, process=MagicMock())
        adaptor = MagicMock()
        adaptor.num_moe_layers = self.NUM_MOE_LAYERS
        adaptor.num_dense_layers = 2
        adaptor.get_rank_expert_workload.return_value = torch.randn(4, 8)
        updator.set_adaptor(adaptor)

        updator.cur_iterations = self.HEAT_INTERVAL + self.ALGORITHM_INTERVAL
        updator.update_info_all = [({}, {}, [0, 1], [0, 1], 0)]
        return updator, loader

    def test_forward_before_starts_no_transfer_when_recording(self):
        updator, loader = self._make_updator("/tmp/expert_map_record.json")

        updator.forward_before()

        loader.generate_expert_d2d_transfer_task.assert_not_called()
        loader.asyn_expert_weight_transfer.assert_not_called()
        self.assertEqual(updator.update_info_all, [])

    def test_forward_before_starts_transfer_when_not_recording(self):
        updator, loader = self._make_updator(None)

        updator.forward_before()

        loader.generate_expert_d2d_transfer_task.assert_called_once()
        loader.asyn_expert_weight_transfer.assert_called_once()
        self.assertEqual(updator.update_info_all, [])


if __name__ == "__main__":
    unittest.main()
