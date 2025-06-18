# fused moe ops test will hit the infer_schema error, we need add the patch
# here to make the test pass.
import vllm_ascend.patch.worker.patch_common.patch_utils  # type: ignore[import]  # isort: skip  # noqa

import json
from typing import List, TypedDict

import pytest
import torch

from vllm_ascend.ops.expert_load_balancer import ExpertLoadBalancer


class Device(TypedDict):
    device_id: int
    device_expert: List[int]


class Layer(TypedDict):
    layer_id: int
    device_count: int
    device_list: List[Device]


class MockData(TypedDict):
    moe_layer_count: int
    layer_list: List[Layer]


MOCK_DATA: MockData = {
    "moe_layer_count":
    1,
    "layer_list": [{
        "layer_id":
        0,
        "device_count":
        2,
        "device_list": [{
            "device_id": 0,
            "device_expert": [7, 2, 0, 3, 5]
        }, {
            "device_id": 1,
            "device_expert": [6, 1, 4, 7, 2]
        }]
    }]
}


@pytest.fixture
def mock_expert_load_balancer(tmp_path):
    json_file = tmp_path / "expert_map.json"
    with open(json_file, 'w') as f:
        json.dump(MOCK_DATA, f)

    return ExpertLoadBalancer(str(json_file), global_expert_num=8)


def test_init(mock_expert_load_balancer):
    assert isinstance(mock_expert_load_balancer.expert_map_tensor,
                      torch.Tensor)
    assert mock_expert_load_balancer.layers_num == MOCK_DATA["moe_layer_count"]
    assert mock_expert_load_balancer.ranks_num == MOCK_DATA["layer_list"][0][
        "device_count"]


def test_generate_index_dicts(mock_expert_load_balancer):
    tensor_2d = torch.tensor([[7, 2, 0, 3, 5], [6, 1, 4, 7, 2]])
    result = mock_expert_load_balancer.generate_index_dicts(tensor_2d)
    expected_result = [{
        7: 0,
        2: 1,
        0: 2,
        3: 3,
        5: 4
    }, {
        6: 5,
        1: 6,
        4: 7,
        7: 8,
        2: 9
    }]
    assert result == expected_result


def test_generate_expert_placement_map(mock_expert_load_balancer):
    expert_placement_map = mock_expert_load_balancer.generate_expert_placement_map(
    )
    assert expert_placement_map.shape == (mock_expert_load_balancer.layers_num,
                                          mock_expert_load_balancer.ranks_num,
                                          8)
    assert torch.all(expert_placement_map >= -1)


def test_generate_log2phy_expert_map(mock_expert_load_balancer):
    layer_id = 0
    log2phy_map = mock_expert_load_balancer.generate_log2phy_expert_map(
        layer_id)
    assert log2phy_map.shape == (mock_expert_load_balancer.ranks_num, 8)
    assert torch.all(log2phy_map >= -1)


def test_get_rank_placement_map(mock_expert_load_balancer, mocker):
    mocker.patch("torch_npu.npu._lazy_init")
    mocker.patch('torch.npu.current_device', return_value='cpu')
    layer_id = 0
    rank_id = 0
    rank_local_expert_num, rank_expert_map = mock_expert_load_balancer.get_rank_placement_map(
        layer_id, rank_id)
    assert rank_local_expert_num == 5
    expected_tensor = torch.tensor([2, -1, 1, 3, -1, 4, -1, 0],
                                   dtype=torch.int32).to(
                                       rank_expert_map.device)
    assert rank_expert_map.equal(expected_tensor)

    rank_id = 1
    rank_local_expert_num, rank_expert_map = mock_expert_load_balancer.get_rank_placement_map(
        layer_id, rank_id)
    expected_tensor = torch.tensor([-1, 1, 4, -1, 2, -1, 0, 3],
                                   dtype=torch.int32).to(
                                       rank_expert_map.device)
    assert rank_expert_map.equal(expected_tensor)


def test_get_rank_log2phy_map(mock_expert_load_balancer):
    layer_id = 0
    rank_id = 0
    log2phy_map = mock_expert_load_balancer.get_rank_log2phy_map(
        layer_id, rank_id)
    expected_tensor = torch.tensor([2, 6, 1, 3, 7, 4, 5, 0],
                                   dtype=torch.int32).to(log2phy_map.device)
    assert log2phy_map.equal(expected_tensor)

    rank_id = 1
    log2phy_map = mock_expert_load_balancer.get_rank_log2phy_map(
        layer_id, rank_id)
    expected_tensor = torch.tensor([2, 6, 9, 3, 7, 4, 5, 8],
                                   dtype=torch.int32).to(log2phy_map.device)
    assert log2phy_map.equal(expected_tensor)


def test_get_global_redundant_expert_num(mock_expert_load_balancer):
    redundant_expert_num = mock_expert_load_balancer.get_global_redundant_expert_num(
    )
    expected_redundant_expert_num = len(MOCK_DATA["layer_list"][0]["device_list"][0]["device_expert"]) * \
                                    MOCK_DATA["layer_list"][0]["device_count"] - 8
    assert redundant_expert_num == expected_redundant_expert_num
