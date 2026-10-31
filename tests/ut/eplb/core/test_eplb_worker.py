#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Unit tests for the EPLB worker (``vllm_ascend.eplb.core.eplb_worker``).

Covers the shared-dict data accessors, the global/local placement conversion,
placement sanity checks, the greedy update-info composition, hotness and
imbalance helpers and the end-to-end ``do_update`` cycle with the random
baseline policy.
"""

import unittest
from unittest import mock

import numpy as np
import torch

from vllm_ascend.eplb.core.eplb_worker import EplbProcess, EplbWorker


def make_worker(shared_dict, policy_type=0, tp_size=None) -> EplbWorker:
    return EplbWorker(shared_dict, policy_type, True, tp_size)


class TestEplbWorkerDataAccess(unittest.TestCase):
    def test_init_reads_expert_maps_from_shared_dict(self):
        expert_maps = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int32)
        worker = make_worker({"expert_maps": expert_maps})
        self.assertIsNone(worker.old_expert_maps)
        maps = worker.get_init_expert_maps()
        self.assertEqual(maps.shape, (1, 2, 2))
        self.assertEqual(worker.rank_id, 0)
        self.assertFalse(worker.multi_stage)

    def test_multi_stage_flag_for_flashlb_policy(self):
        with mock.patch("vllm_ascend.eplb.core.policy.policy_factory.warm_up"):
            worker = make_worker({}, policy_type=3)
        self.assertTrue(worker.multi_stage)
        self.assertEqual(worker.policy_type, 3)

    def test_fetch_and_sum_load_info(self):
        load = torch.tensor([[[10, 0], [0, 0]]], dtype=torch.int64)
        worker = make_worker({"moe_load": load})
        self.assertTrue(torch.equal(worker.fetch_and_sum_load_info(), load))

    def test_do_update_skips_without_load_info(self):
        expert_maps = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int32)
        worker = make_worker({"expert_maps": expert_maps})
        result = worker.do_update()
        self.assertIsNone(result)
        self.assertEqual(worker.num_local_experts, 4)

    def test_do_update_raises_without_expert_maps(self):
        worker = make_worker({})
        with self.assertRaises(ValueError):
            worker.do_update()

    def test_update_expert_map_writes_shared_dict(self):
        shared = {}
        worker = make_worker(shared)
        new_maps = torch.tensor([[[4, 5], [6, 7]]], dtype=torch.int32)
        worker.update_expert_map(new_maps)
        self.assertTrue(torch.equal(shared["expert_maps"], new_maps))


class TestPlacementConversion(unittest.TestCase):
    def test_global2local_records_slot_per_expert(self):
        worker = make_worker({})
        global_placement = torch.tensor([[[0, -1, 1], [2, -1, 3]]], dtype=torch.int32)
        local_placement = worker.global2local(global_placement, E_local=4)
        # The local table is expert -> slot: expert 0 lives in slot 0 of rank
        # 0, expert 1 in slot 2 of rank 0, and so on.
        self.assertEqual(local_placement.shape, (1, 2, 4))
        np.testing.assert_array_equal(local_placement[0].numpy(), np.array([[0, 2, -1, -1], [-1, -1, 0, 2]]))

    def test_local2global_recovers_per_rank_local_maps(self):
        worker = make_worker({})
        # With per-rank local ids (the production layout) the local table is
        # dense per rank, and local2global infers the width from the largest
        # local id: experts 0..S-1, one row per rank.
        global_placement = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int32)
        local_placement = worker.global2local(global_placement, E_local=4)
        np.testing.assert_array_equal(local_placement[0].numpy(), np.array([[0, 1, -1, -1], [-1, -1, 0, 1]]))
        restored = worker.local2global(local_placement)
        # The composition round-trips back to the original placement.
        self.assertEqual(restored.shape, global_placement.shape)
        np.testing.assert_array_equal(restored.numpy(), global_placement.numpy())

    def test_global2local_dense_with_per_rank_local_ids(self):
        worker = make_worker({})
        # Production-style map: every rank lists local ids 0..S-1.
        global_placement = torch.tensor([[[0, 1], [0, 1], [0, 1], [0, 1]]], dtype=torch.int32)
        local_placement = worker.global2local(global_placement, E_local=2)
        np.testing.assert_array_equal(local_placement[0].numpy(), np.array([[0, 1]] * 4))

    def test_local2global_all_missing_returns_empty(self):
        worker = make_worker({})
        local_placement = torch.full((1, 2, 3), -1, dtype=torch.int32)
        restored = worker.local2global(local_placement)
        self.assertEqual(restored.shape[2], 0)


class TestPlacementChecks(unittest.TestCase):
    def _worker(self):
        return make_worker({})

    def test_valid_movement_is_kept(self):
        worker = self._worker()
        old = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int32)
        new = torch.tensor([[[2, 1], [0, 3]]], dtype=torch.int32)
        worker.check_expert_placement(old, new)
        np.testing.assert_array_equal(new[0].numpy(), np.array([[2, 1], [0, 3]]))

    def test_missing_expert_falls_back_to_old(self):
        worker = self._worker()
        old = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int32)
        new = torch.tensor([[[0, 1], [2, 2]]], dtype=torch.int32)
        worker.check_expert_placement(old, new)
        np.testing.assert_array_equal(new[0].numpy(), old[0].numpy())

    def test_replicated_expert_on_one_rank_falls_back(self):
        worker = self._worker()
        old = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int32)
        new = torch.tensor([[[0, 0], [2, 3]]], dtype=torch.int32)
        worker.check_expert_placement(old, new)
        np.testing.assert_array_equal(new[0].numpy(), old[0].numpy())

    def test_intra_rank_reordering_falls_back(self):
        worker = self._worker()
        old = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int32)
        new = torch.tensor([[[1, 0], [2, 3]]], dtype=torch.int32)
        worker.check_expert_placement(old, new)
        np.testing.assert_array_equal(new[0].numpy(), old[0].numpy())


class TestUpdateInfoComposition(unittest.TestCase):
    def _yields(self, current, updated):
        worker = make_worker({})
        layers = list(worker.compose_expert_update_info_greedy(updated, current))
        self.assertEqual(len(layers), 1)
        return layers[0]

    def test_no_change_yields_empty_info(self):
        maps = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int32)
        send, recv, new_map, layer_id = self._yields(maps, maps.clone())
        self.assertEqual(send, {})
        self.assertEqual(recv, {})
        self.assertEqual(layer_id, 0)

    def test_slot_reorder_reports_nothing(self):
        current = torch.tensor([[[0, -1, 1], [2, -1, -1]]], dtype=torch.int32)
        updated = torch.tensor([[[1, -1, 0], [2, -1, -1]]], dtype=torch.int32)
        # No cell transitions between -1 and a value, so nothing is reported.
        send, recv, _, _ = self._yields(current, updated)
        self.assertEqual(send, {})
        self.assertEqual(recv, {})

    def test_fill_and_empty_transitions_are_reported(self):
        # Production maps use per-rank local ids (slot k holds expert k when
        # present), so -1 transitions identify both the receiving/emptying
        # rank and the slot; the reported tuples are (dst_rank, dst_slot) and
        # (src_rank, src_slot).
        current = torch.tensor([[[0, 1, 2], [-1, -1, 2]]], dtype=torch.int32)
        updated = torch.tensor([[[0, 1, -1], [-1, 2, 2]]], dtype=torch.int32)
        send, recv, _, _ = self._yields(current, updated)
        # Slot 2 empties on rank 0 and a value fills slot 1 on rank 1.
        self.assertEqual(send, {0: [(1, 1)]})
        self.assertEqual(recv, {1: [(0, 1)]})

    def test_pack_update_info(self):
        worker = make_worker({}, policy_type=1)
        new_map = torch.tensor([[[0, 1, 4, -1], [2, 4, -1, -1], [3, -1, 4, -1]]], dtype=torch.int32)
        send_info = {1: [(0, 4)]}
        recv_info = {0: [(1, 4)]}
        packed = worker.pack_update_info([(send_info, recv_info, new_map[0], 0)])
        self.assertEqual(len(packed), 1)
        send_all, recv_all, maps, log2phy, layer_id = packed[0]
        # Rank 0 (this worker) only receives; the send entry is empty.
        self.assertEqual(send_all, [])
        self.assertEqual(recv_all, [(1, 4)])
        self.assertEqual(maps, [0, 1, 4, -1])
        self.assertEqual(layer_id, 0)
        # Expert 0 lives on rank 0 only; experts 1 and 2 pick their
        # rank-0 replica from the (rank * valid_count) offset layout.
        np.testing.assert_array_equal(log2phy, [0, 1, 4])


class TestHotnessAndImbalance(unittest.TestCase):
    def test_calculate_hotness(self):
        deployment = np.array([[[0, 0], [1, 1]]])
        moe_load = torch.tensor([[[10, 20], [30, 40]]], dtype=torch.int64)
        hotness = EplbWorker._calculate_hotness(deployment, moe_load)
        # The hotness vector has num_ranks * slots entries; expert 0 takes the
        # sum of both of its slots.
        np.testing.assert_array_equal(hotness, np.array([[30, 70, 0, 0]]))

    def test_calculate_hotness_sums_replicas(self):
        deployment = np.array([[[0, 1], [0, 2]]])
        moe_load = torch.tensor([[[5, 5], [5, 5]]], dtype=torch.int64)
        hotness = EplbWorker._calculate_hotness(deployment, moe_load)
        np.testing.assert_array_equal(hotness, np.array([[10, 5, 5, 0]]))

    def test_compute_imbalance(self):
        deployment = np.array([[[0, 1], [2, 3]]])
        hotness = np.array([[10.0, 10.0, 0.0, 0.0]])
        mean_val, max_val = EplbWorker._compute_imbalance(deployment, hotness)
        # rank0: 10 + 10 = 20, rank1: 0 -> max 20, mean 10 -> 2.0.
        self.assertAlmostEqual(mean_val, 2.0)
        self.assertAlmostEqual(max_val, 2.0)

    def test_compute_imbalance_splits_replicated_hotness(self):
        deployment = np.array([[[0, 0], [1, 1]]])
        hotness = np.array([[10.0, 10.0, 0.0, 0.0]])
        mean_val, max_val = EplbWorker._compute_imbalance(deployment, hotness)
        # Each rank hosts one replica pair: (10 + 10) / 2 = 10 on both ranks.
        self.assertAlmostEqual(mean_val, 1.0)
        self.assertAlmostEqual(max_val, 1.0)

    def test_compute_imbalance_returns_per_layer_list(self):
        deployment = np.array([[[0, 1], [2, 3]], [[0, 1], [2, 3]]])
        hotness = np.array([[10.0, 10.0, 0.0, 0.0], [10.0, 10.0, 0.0, 0.0]])
        mean_val, max_val, per_layer = EplbWorker._compute_imbalance(deployment, hotness, return_list=True)
        self.assertEqual(len(per_layer), 2)
        for value in per_layer:
            self.assertAlmostEqual(value, 2.0)
        self.assertAlmostEqual(mean_val, 2.0)
        self.assertAlmostEqual(max_val, 2.0)


class TestDoUpdateEndToEnd(unittest.TestCase):
    def test_do_update_full_cycle_with_random_policy(self):
        # Production-style maps: each rank lists local ids 0..S-1, so the
        # local placement stays dense (no -1) for the policy and the
        # imbalance bookkeeping.
        expert_maps = torch.tensor([[[0, 1], [0, 1], [0, 1], [0, 1]]], dtype=torch.int32)
        moe_load = torch.tensor([[[10, 0], [0, 0], [0, 0], [0, 0]]], dtype=torch.int64)
        shared = {"expert_maps": expert_maps, "moe_load": moe_load}
        worker = make_worker(shared, policy_type=0)

        packed = worker.do_update()

        # The random policy swaps the last slot of ranks 1 and 3, which are
        # equal here, so the shared map comes back unchanged.
        np.testing.assert_array_equal(shared["expert_maps"].numpy(), expert_maps.numpy())
        self.assertEqual(len(packed), 1)
        send_all, recv_all, maps, log2phy, layer_id = packed[0]
        self.assertEqual(send_all, [])
        self.assertEqual(recv_all, [])
        self.assertEqual(maps, [0, 1])
        self.assertEqual(layer_id, 0)
        # 8 physical slots, valid_count = 8; rank-0 replicas are the first
        # entry of each logical expert's (rank * valid_count) offset list.
        np.testing.assert_array_equal(log2phy, [0, 1])

        # The worker publishes the hotness imbalance metrics for rank 0.
        # Local placement is dense [[0, 1]] per rank; expert 0 (hotness 10)
        # is split over 4 ranks -> 2.5 per rank, so every stage is equal.
        hotness_metrics = worker.latest_expert_hotness
        self.assertAlmostEqual(hotness_metrics["current_mean"], 1.0)
        self.assertAlmostEqual(hotness_metrics["update_max"], 1.0)
        self.assertEqual(len(hotness_metrics["current_imbalance_list"]), 1)

    def test_do_update_rejects_intra_rank_reorder(self):
        # A policy output that only reorders experts inside a rank is
        # detected by check_expert_placement and the original map is kept.
        expert_maps = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.int32)
        moe_load = torch.tensor([[[10, 0], [0, 0]]], dtype=torch.int64)
        shared = {"expert_maps": expert_maps, "moe_load": moe_load}
        worker = make_worker(shared, policy_type=0)
        new_table = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)
        with mock.patch.object(worker.policy, "rebalance_experts") as rebalance:
            rebalance.return_value = (1, [0], new_table)
            packed = worker.do_update()

        rebalance.assert_called_once()
        np.testing.assert_array_equal(shared["expert_maps"].numpy(), expert_maps.numpy())
        self.assertEqual(worker.old_expert_maps.tolist(), [[[0, 1], [0, 1]]])
        self.assertEqual(len(packed), 1)


class TestCalculateRebalanceAndProcess(unittest.TestCase):
    def test_calculate_rebalance_experts_delegates_to_policy(self):
        worker = make_worker({}, policy_type=1)
        old_maps = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int32)
        load = torch.tensor([[[100, 0], [0, 0]]], dtype=torch.int64)
        worker.old_expert_maps = old_maps
        new_table = torch.tensor([[[2, 0], [3, 1]]], dtype=torch.int32)
        with mock.patch.object(worker.policy, "rebalance_experts") as rebalance:
            rebalance.return_value = (1, [0], new_table)
            change, priority, result = worker.calculate_rebalance_experts(load, old_maps)
        rebalance.assert_called_once_with(old_maps, load)
        self.assertEqual(change, 1)
        self.assertEqual(priority, [0])
        self.assertTrue(torch.equal(result, new_table))

    def test_eplb_process_constructs_worker(self):
        # Queue() would create OS pipes, which the unit test environment does
        # not need: stub it out.
        with mock.patch("vllm_ascend.eplb.core.eplb_worker.Queue"):
            process = EplbProcess({}, policy_type=0)
        self.assertIsInstance(process.worker, EplbWorker)
        self.assertEqual(process.worker.policy_type, 0)
        self.assertFalse(process.worker.multi_stage)


if __name__ == "__main__":
    unittest.main(verbosity=2)
