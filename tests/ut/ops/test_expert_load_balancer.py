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
# This file is a part of the vllm-ascend project.
#

import json
import os
from typing import List, TypedDict
from unittest import mock

import torch

from tests.ut.base import TestBase
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


class TestExpertLoadBalancer(TestBase):

    def setUp(self):
        _TEST_DIR = os.path.dirname(__file__)
        json_file = _TEST_DIR + "/expert_map.json"
        with open(json_file, 'r') as f:
            self.expert_map: MockData = json.load(f)

        self.expert_load_balancer = ExpertLoadBalancer(json_file, 8)

    def test_init(self):

        self.assertIsInstance(self.expert_load_balancer.expert_map_tensor,
                              torch.Tensor)
        self.assertEqual(self.expert_load_balancer.layers_num,
                         self.expert_map["moe_layer_count"])
        self.assertEqual(self.expert_load_balancer.ranks_num,
                         self.expert_map["layer_list"][0]["device_count"])

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
                          self.expert_load_balancer.ranks_num, 10))
        self.assertTrue(torch.all(expert_placement_map >= -1))

    def test_generate_log2phy_expert_map(self):
        layer_id = 0
        log2phy_map = self.expert_load_balancer.generate_log2phy_expert_map(
            layer_id)
        self.assertEqual(log2phy_map.shape,
                         (self.expert_load_balancer.ranks_num, 10))
        self.assertTrue(torch.all(log2phy_map >= -1))

    @mock.patch("torch_npu.npu._lazy_init")
    @mock.patch("torch.npu.current_device", return_value="cpu")
    def test_get_rank_placement_map(self, mock_current_device, mock_lazy_init):
        layer_id = 0
        rank_id = 0
        rank_local_expert_num, rank_expert_map = self.expert_load_balancer.get_rank_placement_map(
            layer_id, rank_id)
        self.assertEqual(rank_local_expert_num, 5)
        expected_tensor = torch.tensor([2, -1, 1, 3, -1, 4, -1, 0, -1, -1],
                                       dtype=torch.int32).to(
                                           rank_expert_map.device)
        self.assertTrue(rank_expert_map.equal(expected_tensor))

        rank_id = 1
        rank_local_expert_num, rank_expert_map = self.expert_load_balancer.get_rank_placement_map(
            layer_id, rank_id)
        expected_tensor = torch.tensor([-1, 1, 4, -1, 2, -1, 0, 3, -1, -1],
                                       dtype=torch.int32).to(
                                           rank_expert_map.device)
        self.assertTrue(rank_expert_map.equal(expected_tensor))

    def test_get_rank_log2phy_map(self):
        layer_id = 0
        rank_id = 0
        log2phy_map = self.expert_load_balancer.get_rank_log2phy_map(
            layer_id, rank_id)
        expected_tensor = torch.tensor([2, 6, 1, 3, 7, 4, 5, 0, -1, -1],
                                       dtype=torch.int32).to(
                                           log2phy_map.device)
        self.assertTrue(log2phy_map.equal(expected_tensor))

        rank_id = 1
        log2phy_map = self.expert_load_balancer.get_rank_log2phy_map(
            layer_id, rank_id)
        expected_tensor = torch.tensor([2, 6, 9, 3, 7, 4, 5, 8, -1, -1],
                                       dtype=torch.int32).to(
                                           log2phy_map.device)
        self.assertTrue(log2phy_map.equal(expected_tensor))

    def test_get_global_redundant_expert_num(self):
        redundant_expert_num = self.expert_load_balancer.get_global_redundant_expert_num(
        )
        expected_redundant_expert_num = len(self.expert_map["layer_list"][0]["device_list"][0]["device_expert"]) * \
                                        self.expert_map["layer_list"][0]["device_count"] - 8
        self.assertEqual(redundant_expert_num, expected_redundant_expert_num)
