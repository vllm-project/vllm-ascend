from typing import Any

import numpy as np
import pytest

from vllm_ascend.eplb.core.policy.policy_dynamic_ep_v2 import (DynamicConfig,
                                                               DynamicEplbV2)


@pytest.fixture
def config():
    return DynamicConfig()


@pytest.fixture
def policy(config):
    return DynamicEplbV2(config)


def test_safe_operations(policy):
    # safe_divide
    assert policy.safe_divide(10, 2) == 5
    with pytest.raises(ZeroDivisionError):
        policy.safe_divide(1, 0)

    # safe_exact_divide
    assert policy.safe_exact_divide(10, 3) == 3
    with pytest.raises(ZeroDivisionError):
        policy.safe_exact_divide(1, 0)

    # safe_mod
    assert policy.safe_mod(10, 3) == 1
    with pytest.raises(ZeroDivisionError):
        policy.safe_mod(1, 0)


def test_add_redundant():
    workload = np.array([[[1, 2], [3, 4]]])
    placement = np.array([[[0, 1], [0, 1]]])
    result = DynamicEplbV2.add_redundant(placement, workload, 2)
    assert result.shape == (1, 2)
    assert np.all(result[0] == [4, 6])  # 0:1+3, 1:2+4


def test_get_redundant_num():
    counts = np.array([1, 2, 1])
    assert DynamicEplbV2.get_redundant_num(3, counts) == 1  # sum(counts-1)


def test_calculate_max_heat_per_layer():
    workload = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    result = DynamicEplbV2.calculate_max_heat_per_layer(workload, 2)
    assert result == [7, 15]


def test_calculate_initial_imbalance(policy):
    deployment = np.array([[[0, 1], [0, 1]]])
    workloads = np.array([[1, 1]])
    result = policy.calculate_initial_imbalance(deployment, workloads)
    assert isinstance(result, list)
    assert len(result) == 1


def test_compute_redundant_assignments(policy):
    base_experts = [(0, 10), (1, 5)]
    redundant, sorted_weights = policy.compute_redundant_assignments(
        base_experts, num_redundant_experts=2, num_experts=2)
    assert len(redundant) == 2
    assert len(sorted_weights) == 2


def test_prepare_expert_list():
    base_experts = [(0, 10), (1, 5)]
    redundant_assignments = [[2], []]
    result = DynamicEplbV2.prepare_expert_list(base_experts,
                                               redundant_assignments, 1)
    assert isinstance(result, list)
    assert len(result) == 1


def test_non_redundant_expert_information():
    origin_deployment = np.array([[0, 1]])
    updated_weights = [(0, 10), (1, 5)]
    rendun_pos: dict[Any, Any] = {}
    assignments, weights, loads, counts = DynamicEplbV2.non_redundant_expert_information(
        origin_deployment, updated_weights, rendun_pos)
    assert assignments[0] == [0, 1]
    assert loads[0] == 15


def test_recomputing_initial_weight(policy):
    layer_workloads = [10, 5]
    device_assignments = [[0, 1]]
    cur_layer_workload, num_all_experts = policy.recomputing_initial_weight(
        layer_workloads, device_assignments)
    assert cur_layer_workload[0] == 10
    assert num_all_experts[0] == 1


def test_safe_divide_zero_edge_case(policy):
    assert policy.safe_divide(0, 1) == 0
    assert policy.safe_divide(0, 5) == 0
