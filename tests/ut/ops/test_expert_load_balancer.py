# fused moe ops test will hit the infer_schema error, we need add the patch
# here to make the test pass.
import vllm_ascend.patch.worker.patch_common.patch_utils  # type: ignore[import]  # isort: skip  # noqa

import json
import unittest
from typing import List, TypedDict
from unittest import mock

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


class TestExpertLoadBalancer(unittest.TestCase):

    def setUp(self):
        json_file = "expert_map.json"
        with open(json_file, 'w') as f:
            json.dump(MOCK_DATA, f)

        self.expert_load_balancer = ExpertLoadBalancer(json_file,
                                                       global_expert_num=8)

    def test_init(self):

        self.assertIsInstance(self.expert_load_balancer.expert_map_tensor,
                              torch.Tensor)
        self.assertEqual(self.expert_load_balancer.layers_num,
                         MOCK_DATA["moe_layer_count"])
        self.assertEqual(self.expert_load_balancer.ranks_num,
                         MOCK_DATA["layer_list"][0]["device_count"])

    def test_generate_index_dicts(self):
        tensor_2d = torch.tensor([[7, 2, 0, 3, 5], [6, 1, 4, 7, 2]])
        result = self.expert_load_balancer.generate_index_dicts(tensor_2d)
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
        self.assertEqual(result, expected_result)

    def test_generate_expert_placement_map(self):
        expert_placement_map = self.expert_load_balancer.generate_expert_placement_map(
        )
        self.assertEqual(expert_placement_map.shape,
                         (self.expert_load_balancer.layers_num,
                          self.expert_load_balancer.ranks_num, 8))
        self.assertTrue(torch.all(expert_placement_map >= -1))

    def test_generate_log2phy_expert_map(self):
        layer_id = 0
        log2phy_map = self.expert_load_balancer.generate_log2phy_expert_map(
            layer_id)
        self.assertEqual(log2phy_map.shape,
                         (self.expert_load_balancer.ranks_num, 8))
        self.assertTrue(torch.all(log2phy_map >= -1))

    @mock.patch("torch_npu.npu._lazy_init")
    @mock.patch("torch.npu.current_device", return_value="cpu")
    def test_get_rank_placement_map(self, mock_current_device, mock_lazy_init):
        layer_id = 0
        rank_id = 0
        rank_local_expert_num, rank_expert_map = self.expert_load_balancer.get_rank_placement_map(
            layer_id, rank_id)
        self.assertEqual(rank_local_expert_num, 5)
        expected_tensor = torch.tensor([2, -1, 1, 3, -1, 4, -1, 0],
                                       dtype=torch.int32).to(
                                           rank_expert_map.device)
        self.assertTrue(rank_expert_map.equal(expected_tensor))

        rank_id = 1
        rank_local_expert_num, rank_expert_map = self.expert_load_balancer.get_rank_placement_map(
            layer_id, rank_id)
        expected_tensor = torch.tensor([-1, 1, 4, -1, 2, -1, 0, 3],
                                       dtype=torch.int32).to(
                                           rank_expert_map.device)
        self.assertTrue(rank_expert_map.equal(expected_tensor))

    def test_get_rank_log2phy_map(self):
        layer_id = 0
        rank_id = 0
        log2phy_map = self.expert_load_balancer.get_rank_log2phy_map(
            layer_id, rank_id)
        expected_tensor = torch.tensor([2, 6, 1, 3, 7, 4, 5, 0],
                                       dtype=torch.int32).to(
                                           log2phy_map.device)
        self.assertTrue(log2phy_map.equal(expected_tensor))

        rank_id = 1
        log2phy_map = self.expert_load_balancer.get_rank_log2phy_map(
            layer_id, rank_id)
        expected_tensor = torch.tensor([2, 6, 9, 3, 7, 4, 5, 8],
                                       dtype=torch.int32).to(
                                           log2phy_map.device)
        self.assertTrue(log2phy_map.equal(expected_tensor))

    def test_get_global_redundant_expert_num(self):
        redundant_expert_num = self.expert_load_balancer.get_global_redundant_expert_num(
        )
        expected_redundant_expert_num = len(MOCK_DATA["layer_list"][0]["device_list"][0]["device_expert"]) * \
                                        MOCK_DATA["layer_list"][0]["device_count"] - 8
        self.assertEqual(redundant_expert_num, expected_redundant_expert_num)
