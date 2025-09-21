from unittest.mock import patch

import numpy as np
import pytest

from vllm_ascend.eplb.core.policy.policy_dynamic_ep import DynamicEplb


class TestDynamicEplb:

    def test_add_redundant_basic(self):
        current_expert_table = np.array([[[0, 1], [1, 0]]])
        expert_workload = np.array([[[2, 3], [4, 1]]])
        num_original_expert = 2
        result = DynamicEplb.add_redundant(current_expert_table,
                                           expert_workload,
                                           num_original_expert)
        expected = np.array([[2 + 1, 3 + 4]])
        assert np.array_equal(result, expected)

    def test_get_redundant_num(self):
        counts = np.array([2, 1, 3])
        assert DynamicEplb.get_redundant_num(3, counts) == 3

    def test_calculate_max_heat_per_layer(self):
        workload_table = np.array([[[1, 2], [3, 4]], [[2, 2], [1, 1]]])
        max_heat = DynamicEplb.calculate_max_heat_per_layer(workload_table, 2)
        assert max_heat == [7, 4]

    def test_constraint_expert_local_exchange(self):
        current = [[[0, 1], [2, 3]]]
        global_dep = [[[1, 0], [3, 2]]]
        new_dep = DynamicEplb.constraint_expert_local_exchange(
            current, global_dep)
        assert new_dep == [[[0, 1], [2, 3]]]

    def test_compute_balanced_pack_redundancy_normal(self):
        origin_weights = [(0, 10), (1, 20)]
        result, boxes = DynamicEplb.compute_balanced_pack_redundancy(
            origin_weights, 2, 1)
        assert isinstance(result, list) and len(result) == 2

    def test_compute_balanced_pack_redundancy_card0(self):
        origin_weights = [(0, 10)]
        with pytest.raises(RuntimeError):
            DynamicEplb.compute_balanced_pack_redundancy(origin_weights, 0, 0)

    def test_compute_balanced_pack_normal(self):
        origin_weights = np.array([(0, 10), (1, 20)], dtype=object)
        result, boxes = DynamicEplb.compute_balanced_pack(origin_weights, 2)
        assert isinstance(result, list) and len(result) == 2

    def test_compute_balanced_pack_card0(self):
        origin_weights = np.array([(0, 10)], dtype=object)
        with pytest.raises(RuntimeError):
            DynamicEplb.compute_balanced_pack(origin_weights, 0)

    def test_original_compute_balanced_pack_redundancy(self):
        origin_weights = [(0, 5), (1, 10)]
        result, boxes = DynamicEplb.original_compute_balanced_pack_redundancy(
            origin_weights, 2, 1)
        assert isinstance(result, list) and len(result) == 2

    def test_rebalance_experts_normal(self):
        expert_table = np.array([[[0, 1], [1, 0]]])
        workload = np.array([[[2, 3], [4, 1]]])
        policy = DynamicEplb(config=None)
        change, priority, new_dep = policy.rebalance_experts(
            expert_table, workload)
        assert change in [0, 1]
        assert isinstance(priority, np.ndarray)
        assert isinstance(new_dep, list)
        assert np.array(new_dep).shape == expert_table.shape

    def test_rebalance_experts_exceptions(self):
        policy = DynamicEplb(config=None)

        # case1: num_original_expert != expert_num
        expert_table = np.array([[[0, 1], [1, 0]]])
        workload = np.array([[[2, 3], [4, 1]]])
        with patch.object(DynamicEplb,
                          'add_redundant',
                          return_value=np.array([[1, 2, 3]])):
            with pytest.raises(ValueError):
                policy.rebalance_experts(expert_table, workload)

        # case2: num_npus <= 0
        expert_table_zero = np.array([[]])  # 1 layer, 0 NPU, 0 experts
        workload_zero = np.array([[]])
        with pytest.raises(ValueError):
            policy.rebalance_experts(expert_table_zero, workload_zero)

        # case3: num_npus < num_redundancy_expert
        expert_table_small = np.array([[[0, 0]]])  # 1 layer, 1 NPU, 2 experts
        workload_small = np.array([[[1, 1]]])
        with patch.object(DynamicEplb, 'get_redundant_num', return_value=2):
            with pytest.raises(ValueError):
                policy.rebalance_experts(expert_table_small, workload_small)
