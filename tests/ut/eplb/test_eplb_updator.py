import multiprocessing
import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.eplb.eplb_updator import EplbUpdator


def _fetch_shared_dict_value(shared_dict, queue):
    """Module-level helper so the spawn context can pickle it."""
    try:
        value = shared_dict.get("phys_to_logical", None)
        if value is None:
            queue.put(("none",))
        else:
            queue.put(("ok", str(value.device), value.tolist()))
    except Exception as e:  # pragma: no cover - IPC failure path
        queue.put(("error", str(e)))


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

    def _run_warm_up_eplb(self):
        with (
            patch("torch.distributed.P2POp", MagicMock()),
            patch("torch.distributed.batch_isend_irecv", return_value=[]),
        ):
            self.updator.warm_up_eplb()

    def test_warm_up_eplb_stores_cpu_phys_to_logical(self):
        phys_to_logical = torch.arange(256, dtype=torch.int32)
        self.adaptor.phys_to_logical = phys_to_logical
        self._run_warm_up_eplb()

        stored = self.updator.shared_dict["phys_to_logical"]
        # torch_npu cannot re-share device tensors received from another
        # process, so the shared_dict contract is CPU-only.
        self.assertEqual(stored.device.type, "cpu")
        self.assertTrue(torch.equal(stored, phys_to_logical))

    def test_warm_up_eplb_shared_dict_survives_manager_ipc(self):
        # Regression: the EPLB worker reads phys_to_logical through the
        # Manager proxy from a subprocess. NPU tensors stored in the dict
        # cannot be sent back (torch_npu "_share_npu_" fails for tensors
        # received from another process), which killed the worker after the
        # first rebalance.
        manager = multiprocessing.Manager()
        self.addCleanup(manager.shutdown)
        self.updator.shared_dict = manager.dict()

        # Only real tensors may cross the Manager IPC; MagicMock values are
        # not picklable.
        self.adaptor.get_global_expert_map.return_value = torch.zeros(2, 4, 64, dtype=torch.int32)
        self.adaptor.phys_to_logical = torch.arange(256, dtype=torch.int32)
        self._run_warm_up_eplb()

        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(target=_fetch_shared_dict_value, args=(self.updator.shared_dict, queue))
        proc.start()
        try:
            result = queue.get(timeout=60)
        finally:
            proc.join(timeout=60)
        self.assertEqual(result[0], "ok", msg=f"shared_dict IPC failed: {result}")
        self.assertEqual(result[1], "cpu")


if __name__ == "__main__":
    unittest.main()
