"""UT for vllm_ascend/eplb/core/policy/policy_default_eplb.py (DefaultEplb)."""

import numpy as np
import pytest

from vllm_ascend.eplb.core.policy.policy_default_eplb import DefaultEplb


class TestAddRedundant:
    def test_aggregates_workload_by_physical_expert_id(self):
        placement = np.array([[[0, 1], [1, 2]]], dtype=np.int64)
        workload = np.array([[[10, 20], [30, 40]]], dtype=np.int64)

        result = DefaultEplb.add_redundant(placement, workload, num_original_expert=3)

        assert result.tolist() == [[10, 50, 40]]

    def test_empty_placement_yields_zero_workloads(self):
        placement = np.array([[[1, 2], [3, 4]]], dtype=np.int64)
        workload = np.zeros((1, 2, 2), dtype=np.int64)

        result = DefaultEplb.add_redundant(placement, workload, num_original_expert=4)

        assert result.tolist() == [[0, 0, 0, 0]]


class TestCalculateMaxHeatPerLayer:
    def test_returns_per_layer_max_npu_heat(self):
        workload = np.array([[[10, 20], [5, 5]], [[1, 1], [100, 1]]], dtype=np.int64)

        result = DefaultEplb.calculate_max_heat_per_layer(workload, layer_num=2)

        assert result == [30, 101]


class TestGetRedundantNum:
    def test_sums_counts_minus_one(self):
        assert DefaultEplb.get_redundant_num(npu_num=4, counts=np.array([3, 2, 1])) == 3

    def test_no_redundancy(self):
        assert DefaultEplb.get_redundant_num(npu_num=4, counts=np.array([1, 1, 1])) == 0


class TestComputeBalancedPack:
    def test_zero_card_num_raises(self):
        weights = np.array([(0, 1)], dtype=object)
        with pytest.raises(RuntimeError, match="card_num can not be 0"):
            DefaultEplb.compute_balanced_pack(weights, card_num=0)

    def test_exact_placement_for_six_experts_on_four_cards(self):
        weights = np.array([(0, 10), (1, 20), (2, 30), (3, 40), (4, 50), (5, 60)], dtype=object)

        result, boxes = DefaultEplb.compute_balanced_pack(weights, card_num=4)

        assert [int(v) for v in [entry["box_index"] for entry in result]] == [1, 2, 3, 4]
        assert [[int(i) for i in entry["items"]] for entry in result] == [[5], [4], [3, 0], [2, 1]]
        assert [int(entry["item_count"]) for entry in result] == [1, 1, 2, 2]
        assert [int(entry["total_weight"]) for entry in result] == [60, 50, 50, 50]
        assert [[int(i) for i in box] for box in boxes] == [[5], [4], [3, 0], [2, 1]]
        assert sum(int(entry["total_weight"]) for entry in result) == 210

    def test_covers_every_expert_exactly_once(self):
        weights = np.array([(i, (i + 1) * 7) for i in range(7)], dtype=object)

        result, _ = DefaultEplb.compute_balanced_pack(weights, card_num=3)

        placed = [item for entry in result for item in entry["items"]]
        assert sorted(placed) == list(range(7))
        assert sum(entry["item_count"] for entry in result) == 7
        for entry in result:
            assert entry["item_count"] <= 3
            assert entry["total_weight"] == sum(w for i, w in weights if i in entry["items"])


class TestComputeBalancedPackRedundancy:
    def test_zero_card_num_raises(self):
        weights = np.array([(0, 1)], dtype=object)
        with pytest.raises(RuntimeError, match="card_num can not be 0"):
            DefaultEplb.compute_balanced_pack_redundancy(weights, card_num=0, num_redundancy_expert=0)

    def test_hot_expert_gets_redundant_copy_on_both_cards(self):
        weights = np.array([(0, 100), (1, 10), (2, 10), (3, 10)], dtype=object)

        result, boxes = DefaultEplb.compute_balanced_pack_redundancy(
            weights, card_num=2, num_redundancy_expert=1
        )

        # Hot expert 0 keeps one copy per card; the other experts are placed once.
        assert sorted(int(e) for box in boxes for e in box) == [0, 0, 1, 2, 3]
        assert sum(0 in box for box in boxes) == 2
        assert [entry["item_count"] for entry in result] == [3, 2]
        assert [int(entry["total_weight"]) for entry in result] == [70, 60]
        assert sum(entry["item_count"] for entry in result) == 5
        for expert_id in range(4):
            assert sum(expert_id in box for box in boxes) >= 1


class TestConstraintExpertLocalExchange:
    def test_preserves_current_positions_and_fills_new_experts(self):
        current = np.array([[[0, 1], [2, 3]]], dtype=np.int64)
        deployment = np.array([[[1, 0], [9, 3]]], dtype=np.int64)

        result = DefaultEplb.constraint_expert_local_exchange(current, deployment)

        # Expert 0/1 swap is undone (both stay on card 0); 2 leaves, 9 comes in.
        assert result is deployment
        assert result.tolist() == [[[0, 1], [9, 3]]]

    def test_identity_deployment_is_unchanged(self):
        current = np.array([[[4, 5], [6, 7]]], dtype=np.int64)
        deployment = np.array([[[4, 5], [6, 7]]], dtype=np.int64)

        result = DefaultEplb.constraint_expert_local_exchange(current, deployment)

        assert result.tolist() == [[[4, 5], [6, 7]]]


def _identity_placement(layer_num, npu_num, experts_per_npu):
    row = [[npu * experts_per_npu + expert for expert in range(experts_per_npu)] for npu in range(npu_num)]
    return np.tile(np.array([row], dtype=np.int64), (layer_num, 1, 1))


class TestRebalanceExperts:
    def test_redundancy_exceeding_npus_raises_value_error(self):
        # One NPU hosting expert 0 three times -> 2 redundant experts > 1 NPU.
        placement = np.array([[[0, 0, 0]]], dtype=np.int64)
        workload = np.ones((1, 1, 3), dtype=np.int64)

        with pytest.raises(ValueError, match="must be greater than or equal to the number of redundant experts"):
            DefaultEplb().rebalance_experts(placement.tolist(), workload.tolist())

    def test_balanced_workload_reports_no_change(self):
        placement = _identity_placement(1, 4, 2)
        workload = np.ones((1, 4, 2), dtype=np.int64)

        change, priority, deployment = DefaultEplb().rebalance_experts(placement.tolist(), workload.tolist())

        assert change == 0
        assert len(priority) == 1
        assert len(deployment) == 1
        assert sorted(item for card in deployment[0] for item in card) == list(range(8))

    def test_hot_experts_move_and_report_change(self):
        placement = _identity_placement(2, 4, 2)
        workload = np.ones((2, 4, 2), dtype=np.int64)
        # Layer 0: experts 0/1 (card 0) hot; layer 1: experts 2/3 (card 1) hot.
        workload[0, 0] = 100
        workload[1, 1] = 100

        change, priority, deployment = DefaultEplb().rebalance_experts(placement.tolist(), workload.tolist())

        assert change == 1
        assert [int(v) for v in priority] == [0, 1]
        assert len(deployment) == 2
        for layer in deployment:
            assert len(layer) == 4
            assert all(len(card) == 2 for card in layer)

        for layer_idx in range(2):
            assert sorted(int(e) for card in deployment[layer_idx] for e in card) == list(range(8))
            expert_heat = {
                int(placement[layer_idx, npu, slot]): workload[layer_idx, npu, slot]
                for npu in range(4)
                for slot in range(2)
            }
            before = max(int(workload[layer_idx, npu].sum()) for npu in range(4))
            after = max(sum(expert_heat[int(e)] for e in card) for card in deployment[layer_idx])
            assert after < before


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
