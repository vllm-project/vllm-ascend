import unittest
from queue import Queue
from types import SimpleNamespace
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

        # mock _PP in vllm.distributed.parallel_state (PP+EPLB support)
        # Patching the variable directly so that even the real get_pp_group()
        # (already imported into eplb_updator's namespace) reads a non-None _PP.
        self.mock_pp = MagicMock()
        self.mock_pp.rank_in_group = 0
        p4 = patch("vllm.distributed.parallel_state._PP", self.mock_pp)
        self.addCleanup(p4.stop)
        p4.start()

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


class TestEplbPlanReadiness(unittest.TestCase):
    def setUp(self):
        self.updator = EplbUpdator.__new__(EplbUpdator)
        self.updator.expert_heat_collection_interval = 3
        self.updator.algorithm_execution_interval = 2
        self.updator.num_moe_layers = 2
        self.updator.cur_iterations = 4
        self.updator.expert_map_record_path = None
        self.updator.adaptor = MagicMock()
        self.updator.eplb_loader = MagicMock()
        self.updator.eplb_process = SimpleNamespace(block_update_q=Queue())
        self.updator.comm_group = SimpleNamespace(cpu_group=object())
        self.updator.update_info_all = []
        self.updator._local_plan_ready = False
        self.updator._all_plans_ready = False
        self.updator._plan_ready = torch.zeros(1, dtype=torch.int32)
        self.plan = [([], [], [0, 1], [0, 1], layer) for layer in range(2)]

    @patch("vllm_ascend.eplb.eplb_updator.dist.all_reduce")
    def test_late_planner_does_not_block_or_advance_weight_update(self, all_reduce):
        for _ in range(3):
            self.updator.forward_before()
            self.updator.forward_end()
            self.assertEqual(self.updator.cur_iterations, 4)
            self.assertFalse(self.updator.update_expert_weight_flag())
        self.assertEqual(all_reduce.call_count, 3)
        self.updator.eplb_loader.asyn_expert_weight_transfer.assert_not_called()
        self.updator.adaptor.clear_all_moe_loads.assert_not_called()

        self.updator.eplb_process.block_update_q.put(self.plan)
        self.updator.forward_before()
        self.updator.forward_end()
        self.assertEqual(self.updator.cur_iterations, 5)
        self.assertTrue(self.updator.update_expert_weight_flag())

    @patch("vllm_ascend.eplb.eplb_updator.dist.all_reduce")
    def test_ready_rank_retains_plan_until_every_peer_is_ready(self, all_reduce):
        self.updator.eplb_process.block_update_q.put(self.plan)
        all_reduce.side_effect = lambda value, **kwargs: value.zero_()
        for _ in range(2):
            self.updator.forward_before()
            self.updator.forward_end()
            self.assertEqual(self.updator.cur_iterations, 4)
            self.assertIs(self.updator.update_info_all, self.plan)
        self.assertTrue(self.updator._local_plan_ready)
        self.updator.eplb_loader.asyn_expert_weight_transfer.assert_not_called()

        all_reduce.side_effect = None
        self.updator.forward_before()
        self.updator.forward_end()
        self.assertEqual(self.updator.cur_iterations, 5)
        all_reduce.assert_called_with(
            self.updator._plan_ready,
            op=torch.distributed.ReduceOp.MIN,
            group=self.updator.comm_group.cpu_group,
        )

        for layer in range(2):
            self.updator.forward_before()
            self.assertEqual(self.updator.eplb_loader.generate_expert_d2d_transfer_task.call_args.args[-1], layer)
            self.updator.forward_end()
        self.assertEqual(self.updator.eplb_loader.update_expert_map_and_weight.call_count, 2)
        self.assertEqual(self.updator.cur_iterations, 0)
        self.assertFalse(self.updator._local_plan_ready)
        self.assertFalse(self.updator._all_plans_ready)
        self.updator.adaptor.clear_all_moe_loads.assert_called_once()


if __name__ == "__main__":
    unittest.main()
