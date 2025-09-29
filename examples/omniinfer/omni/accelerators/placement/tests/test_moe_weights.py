# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import torch
import torch_npu
import torch.nn as nn
import unittest
import ctypes
from typing import Dict
from omni.accelerators.placement.omni_placement import omni_placement
import multiprocessing as mp
import random
import time
import gc
from omni.accelerators.placement.omni_placement.utils import filter_dict_keys,convert_param_dict_to_list,convert_param_to_ctype,get_expert_ids
from unittest.mock import Mock, patch
from omni.accelerators.placement.omni_placement.placement_handler import init_dram_weights,deepseek_filter_func ,deepseek_get_layer_idx_func


def generate_name(layer_idx, weight_name,first_k_dense_replace=3):
    return f"model.layers.{layer_idx+first_k_dense_replace}.mlp.experts.{weight_name}"


def get_layer(key, layer_prefix="layer"):
    """
    For UnitTest
    从键名中提取layer索引
    """
    parts = key.split(".")
    for i, part in enumerate(parts):
        if part == layer_prefix and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                raise ValueError(f"Layer index in '{key}' after '{layer_prefix}' is not an integer")
    raise IndexError(f"Key '{key}' does not contain '{layer_prefix}' followed by an index")

class TestFilterDictKeys(unittest.TestCase):

    def test_basic_filtering(self):
        """测试基本的过滤功能"""
        param_dict = {'a': 1, 'bb': 2, 'ccc': 3, 'dddd': 4}

        def len_gt(key, threshold=2):
            return len(key) > threshold

        result = filter_dict_keys(param_dict, len_gt)
        expected = {'ccc': 3, 'dddd': 4}
        self.assertEqual(result, expected)

    def test_filter_with_custom_param(self):
        """测试使用自定义参数的过滤"""
        param_dict = {'a': 1, 'bb': 2, 'ccc': 3, 'dddd': 4}

        def len_gt(key, threshold=2):
            return len(key) > threshold

        result = filter_dict_keys(param_dict, len_gt, filter_param={"threshold":3})
        expected = {'dddd': 4}
        self.assertEqual(result, expected)

    def test_empty_dict(self):
        """测试空字典"""
        param_dict = {}

        def len_gt(key, threshold=2):
            return len(key) > threshold

        result = filter_dict_keys(param_dict, len_gt)
        expected = {}
        self.assertEqual(result, expected)

    def test_all_keys_filtered_out(self):
        """测试所有键都被过滤掉的情况"""
        param_dict = {'a': 1, 'bb': 2}

        def len_gt(key, threshold=2):
            return len(key) > threshold

        result = filter_dict_keys(param_dict, len_gt)
        expected = {}
        self.assertEqual(result, expected)

    def test_no_keys_filtered(self):
        """测试没有键被过滤的情况"""
        param_dict = {'ccc': 3, 'dddd': 4}

        def len_gt(key, threshold=2):
            return len(key) > threshold

        result = filter_dict_keys(param_dict, len_gt)
        expected = {'ccc': 3, 'dddd': 4}
        self.assertEqual(result, expected)

    def test_invalid_dict_type(self):
        """测试输入不是字典时抛出TypeError"""
        invalid_input = [1, 2, 3]

        def len_gt(key, threshold=2):
            return len(key) > threshold

        with self.assertRaises(TypeError) as context:
            filter_dict_keys(invalid_input, len_gt)
        self.assertEqual(str(context.exception), "param_dict must be a dictionary")

    def test_invalid_filter_func(self):
        """测试filter_func不可调用时抛出TypeError"""
        param_dict = {'a': 1, 'bb': 2}
        invalid_func = "not a function"

        with self.assertRaises(TypeError) as context:
            filter_dict_keys(param_dict, invalid_func)
        self.assertEqual(str(context.exception), "filter_func must be callable")

    def test_different_key_types(self):
        """测试不同类型的key"""
        param_dict = {1: 'one', 'two': 2, (3,): 'tuple'}

        def is_string(key, unused_param=None):
            return isinstance(key, str)

        result = filter_dict_keys(param_dict, is_string)
        expected = {'two': 2}
        self.assertEqual(result, expected)

class TestVllmNpuEnv(unittest.TestCase):
    def setUp(self):
        from vllm_npu import ENV
        self.ENV = ENV
    def test_enable_omni_placement(self):
        self.ENV.omni_placement_config_path = './config_test.yaml'
        self.assertTrue(self.ENV.use_omni_placement)

    def test_disable_omni_placement(self):
        self.ENV.omni_placement_config_path = ''
        self.assertFalse(self.ENV.use_omni_placement)

class TestConvertParamDictToList(unittest.TestCase):
    def test_basic_functionality(self):
        """测试基本功能"""
        param_dict = {
            "conv.layer.0.weight": torch.randn(2, 256),
            "conv.layer.0.bias": torch.randn(2, 256),
            "fc.layer.1.weight": torch.randn(2, 256)
        }
        result = convert_param_dict_to_list(param_dict, get_layer)
        self.assertEqual(len(result), 2)  # 2 layers
        self.assertEqual(len(result[0]), 2)  # 2 tensors in layer 0
        self.assertEqual(len(result[1]), 2)  # 1 tensor in layer 1 (after unbind)
        self.assertEqual(result[0][0][0].shape, torch.Size([256]))  # First tensor shape
        self.assertEqual(result[1][0][0].shape, torch.Size([256]))
        self.assertEqual(len(result[0][0]),2)  # First tensor shape
        self.assertEqual(len(result[1][0]),1)  # First tensor shape

    def test_multiple_tensors_same_layer(self):
        """测试同一层有多个张量"""
        param_dict = {
            "conv.layer.0.weight": torch.randn(2, 256),
            "conv.layer.0.bias": torch.randn(2, 256),
            "fc.layer.0.weight": torch.randn(2, 256)
        }
        result = convert_param_dict_to_list(param_dict, get_layer)
        self.assertEqual(len(result), 1)  # 1 layer
        self.assertEqual(len(result[0]), 2)  # 2 sublists after unbind
        self.assertEqual(len(result[0][0]), 3)  # 3 tensors in each sublist
        self.assertEqual(result[0][0][0].shape, torch.Size([256]))

    def test_empty_dict(self):
        """测试空字典"""
        param_dict = {}
        result = convert_param_dict_to_list(param_dict, get_layer)
        self.assertEqual(len(result), 0)

    def test_non_consecutive_layers(self):
        """测试非连续层索引"""
        param_dict = {
            "conv.layer.0.weight": torch.randn(2, 256),
            "conv.layer.2.bias": torch.randn(2, 256)
        }
        result = convert_param_dict_to_list(param_dict, get_layer)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 2)  # 2 sublists in layer 0
        self.assertEqual(len(result[1]), 0)  # Empty layer 1
        self.assertEqual(len(result[2]), 2)  # 2 sublists in layer 2
        self.assertEqual(result[0][0][0].shape, torch.Size([256]))

    def test_custom_layer_func_with_param(self):
        """测试带参数的自定义layer_func"""
        param_dict = {
            "conv.block.0.weight": torch.randn(2, 256),
            "conv.block.1.bias": torch.randn(2, 256)
        }
        custom_func = lambda key, prefix: get_layer(key, prefix)
        result = convert_param_dict_to_list(param_dict, custom_func, layer_func_param={"prefix" :"block"})
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)  # 2 sublists in layer 0
        self.assertEqual(len(result[1]), 2)  # 2 sublists in layer 1
        self.assertEqual(result[0][0][0].shape, torch.Size([256]))

    def test_invalid_param_dict(self):
        """测试非字典输入"""
        with self.assertRaises(TypeError) as context:
            convert_param_dict_to_list([1, 2, 3], get_layer)
        self.assertEqual(str(context.exception), "param_dict must be a dictionary")

    def test_invalid_layer_func(self):
        """测试不可调用的layer_func"""
        param_dict = {"conv.layer.0.weight": torch.randn(2, 256)}
        with self.assertRaises(TypeError) as context:
            convert_param_dict_to_list(param_dict, "not_callable")
        self.assertEqual(str(context.exception), "layer_func must be callable")

    def test_invalid_key_format(self):
        """测试键格式错误"""
        param_dict = {"conv.weight": torch.randn(2, 256)}
        with self.assertRaises(IndexError) as context:
            convert_param_dict_to_list(param_dict, get_layer)
        self.assertEqual(str(context.exception), "Key 'conv.weight' does not contain 'layer' followed by an index")

    def test_invalid_tensor_list(self):
        """测试combine_tensors_to_nested_list的错误处理"""
        param_dict = {
            "conv.layer.0.weight": torch.randn(2, 256),
            "conv.layer.0.bias": "not_a_tensor"  # 非张量
        }
        with self.assertRaises(TypeError) as context:
            convert_param_dict_to_list(param_dict, get_layer)
        self.assertEqual(str(context.exception), "All elements in tensor_list must be torch.Tensor")

    def test_mismatched_tensor_shapes(self):
        """测试张量形状不匹配也能concat起来"""
        param_dict = {
            "conv.layer.0.weight": torch.randn(2, 256),
            "conv.layer.0.bias": torch.randn(3, 256)  # 形状不同
        }
        convert_param_dict_to_list(param_dict, get_layer)

# 单元测试
class TestConvertParamToCtype(unittest.TestCase):

    def setUp(self):
        # 在每个测试前设置模拟的omni_placement.Tensor
        self.mock_tensor_class = Mock()
        self.patcher = patch('omni_placement.omni_placement.Tensor', self.mock_tensor_class)
        self.patcher.start()

    def tearDown(self):
        # 在每个测试后停止patch
        self.patcher.stop()

    def test_basic_functionality(self):
        """测试基本功能"""
        param_list = [
            [  # layer 0
                [torch.randn(256), torch.randn(256)],  # expert 0
                [torch.randn(256)]                     # expert 1
            ],
            [  # layer 1
                [torch.randn(256), torch.randn(256)]   # expert 0
            ]
        ]
        result = convert_param_to_ctype(param_list)

        # 检查结构
        self.assertEqual(len(result), 2)  # 2 layers
        self.assertEqual(len(result[0]), 2)  # 2 experts in layer 0
        self.assertEqual(len(result[0][0]), 2)  # 2 params in layer 0, expert 0
        self.assertEqual(len(result[1][0]), 2)  # 2 params in layer 1, expert 0

        # 检查调用
        self.mock_tensor_class.assert_called()
        # 检查第一个张量的调用参数
        first_call_args = self.mock_tensor_class.call_args_list[0][1]
        self.assertEqual(first_call_args['length'], 256)  # numel
        self.assertEqual(first_call_args['name'], "layer.0.expert.0.weight.0")
        self.assertIsInstance(first_call_args['data_ptr'], int)

    def test_empty_list(self):
        """测试空列表"""
        param_list = []
        result = convert_param_to_ctype(param_list)
        self.assertEqual(len(result), 0)
        self.mock_tensor_class.assert_not_called()

    def test_nested_empty_lists(self):
        """测试嵌套空列表"""
        param_list = [[[]], []]  # layer 0: empty expert, layer 1: no experts
        result = convert_param_to_ctype(param_list)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[0][0]), 0)
        self.assertEqual(len(result[1]), 0)
        self.mock_tensor_class.assert_not_called()

    def test_invalid_input_type(self):
        """测试非列表输入"""
        with self.assertRaises(TypeError) as context:
            convert_param_to_ctype("not_a_list")
        self.assertEqual(str(context.exception), "param_list must be a list")
        self.mock_tensor_class.assert_not_called()

    def test_invalid_tensor_type(self):
        """测试非张量元素"""
        param_list = [
            [[torch.randn(256), "not_a_tensor"]]  # 包含非张量元素
        ]
        with self.assertRaises(TypeError) as context:
            convert_param_to_ctype(param_list)
        self.assertEqual(str(context.exception), "All elements must be torch.Tensor")

    def test_different_tensor_shapes(self):
        """测试不同形状的张量"""
        param_list = [
            [  # layer 0
                [torch.randn(256), torch.randn(128)]  # 不同大小的张量
            ]
        ]
        result = convert_param_to_ctype(param_list)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[0][0]), 2)
        # 检查调用次数
        self.assertEqual(self.mock_tensor_class.call_count, 2)
        # 检查参数
        self.assertEqual(self.mock_tensor_class.call_args_list[0][1]['length'], 256)
        self.assertEqual(self.mock_tensor_class.call_args_list[1][1]['length'], 128)

    def test_nested_structure_integrity(self):
        """测试嵌套结构的完整性"""
        param_list = [
            [  # layer 0
                [torch.randn(256), torch.randn(256)],  # expert 0
                [torch.randn(256)]                     # expert 1
            ],
            [  # layer 1
                [torch.randn(256), torch.randn(256), torch.randn(256)]  # expert 0
            ]
        ]
        result = convert_param_to_ctype(param_list)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(len(result[0][0]), 2)
        self.assertEqual(len(result[0][1]), 1)
        self.assertEqual(len(result[1][0]), 3)
        self.assertEqual(self.mock_tensor_class.call_count, 6)  # 2 + 1 + 3 tensors

class TestGetExpertIds(unittest.TestCase):
    def test_basic_functionality(self):
        """测试基本功能"""
        pattern = torch.tensor([
            [True, False, True],   # layer 0: experts 0, 2
            [False, True, False]   # layer 1: expert 1
        ], dtype=torch.bool)
        result = get_expert_ids(pattern)
        expected = [[0, 2], [1]]
        self.assertEqual(result, expected)

    def test_no_experts_in_layer(self):
        """测试某层没有expert的情况"""
        pattern = torch.tensor([
            [True, False, True],   # layer 0: experts 0, 2
            [False, False, False]  # layer 1: no experts
        ], dtype=torch.bool)
        result = get_expert_ids(pattern)
        expected = [[0, 2], []]
        self.assertEqual(result, expected)

    def test_all_experts_in_layer(self):
        """测试某层所有expert都被选中"""
        pattern = torch.tensor([
            [True, True, True],    # layer 0: all experts
            [False, True, False]   # layer 1: expert 1
        ], dtype=torch.bool)
        result = get_expert_ids(pattern)
        expected = [[0, 1, 2], [1]]
        self.assertEqual(result, expected)

    def test_single_layer(self):
        """测试单层情况"""
        pattern = torch.tensor([[True, False, True]], dtype=torch.bool)  # layer 0: experts 0, 2
        result = get_expert_ids(pattern)
        expected = [[0, 2]]
        self.assertEqual(result, expected)

    def test_empty_pattern(self):
        """测试空张量（0层）"""
        pattern = torch.tensor([], dtype=torch.bool).reshape(0, 3)
        result = get_expert_ids(pattern)
        expected = []
        self.assertEqual(result, expected)

    def test_invalid_input_type(self):
        """测试非torch.Tensor输入"""
        with self.assertRaises(TypeError) as context:
            get_expert_ids([[True, False], [False, True]])
        self.assertEqual(str(context.exception), "placement_pattern_current_rank must be a torch.Tensor")

    def test_invalid_dtype(self):
        """测试非bool类型的张量"""
        pattern = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
        with self.assertRaises(ValueError) as context:
            get_expert_ids(pattern)
        self.assertEqual(str(context.exception), "placement_pattern_current_rank must have dtype torch.bool")

    def test_invalid_dimension(self):
        """测试非2D张量"""
        pattern = torch.tensor([True, False, True], dtype=torch.bool)  # 1D
        with self.assertRaises(ValueError) as context:
            get_expert_ids(pattern)
        self.assertEqual(str(context.exception), "placement_pattern_current_rank must be a 2D tensor")

        pattern = torch.tensor([[[True, False]]], dtype=torch.bool)  # 3D
        with self.assertRaises(ValueError) as context:
            get_expert_ids(pattern)
        self.assertEqual(str(context.exception), "placement_pattern_current_rank must be a 2D tensor")

class TestMoeWeightsWrapper(unittest.TestCase):
    def setUp(self):
        # 创建测试模型
        self.first_k_dense_replace=3
        self.num_layers = 58
        self.num_expert = 64
        self.device = torch.device("npu:0")
        NAMES = ["w13_weight","w2_weight","input"]
        self.weights = torch.arange(1,1+self.num_expert,dtype=torch.int8).to(self.device).expand(len(NAMES),self.num_layers, -1)
        self.param_dict = {}
        for name_idx,name in enumerate(NAMES):
            for layer_idx in range(self.num_layers):
                self.param_dict[generate_name(layer_idx,name)] = self.weights[name_idx,layer_idx]
        self.local_rank_pattern = self.weights[0].bool()


    def test_single_process_init_dram_weights(self):
        world_size = 1
        moeweights = omni_placement.MoEWeights(self.num_expert,world_size)
        init_dram_weights(moeweights,self.param_dict,self.local_rank_pattern,self.first_k_dense_replace)
        self.assertTrue(moeweights.isShmInitialized())

    def test_multi_process_init(self):
        """测试多进程初始化"""
        world_size = 4
        moeweights = omni_placement.MoEWeights(self.num_expert, world_size)
        init_dram_weights(moeweights, self.param_dict, self.local_rank_pattern,self.first_k_dense_replace)
        self.assertFalse(moeweights.isShmInitialized())

    def test_zero_experts(self):
        """测试专家数量为0的情况"""
        world_size = 1
        num_expert = 0
        moeweights = omni_placement.MoEWeights(num_expert, world_size)
        # 假设 init_dram_weights 可以处理空专家的情况
        empty_param_dict = {}
        empty_pattern = torch.zeros(self.num_layers, num_expert, dtype=torch.bool).to(self.device)
        with self.assertRaises(RuntimeError) as context:
            init_dram_weights(moeweights, empty_param_dict, empty_pattern,self.first_k_dense_replace)
        self.assertFalse(moeweights.isShmInitialized())  # 确保未初始化

    def test_zero_layers(self):
        """测试层数为0的情况"""
        world_size = 1
        num_layers = 0
        weights = torch.arange(1, 1 + self.num_expert, dtype=torch.int8).to(self.device).expand(3, num_layers, -1)
        param_dict = {}
        local_rank_pattern = weights[0].bool()
        moeweights = omni_placement.MoEWeights(self.num_expert, world_size)
        with self.assertRaises(RuntimeError) as context:
            init_dram_weights(moeweights, param_dict, local_rank_pattern,self.first_k_dense_replace)
        self.assertFalse(moeweights.isShmInitialized())  # 确保未初始化

    def test_invalid_param_dict(self):
        """测试无效的param_dict"""
        world_size = 1
        moeweights = omni_placement.MoEWeights(self.num_expert, world_size)
        invalid_param_dict = "not_a_dict"
        with self.assertRaises(TypeError):  # 假设 init_dram_weights 检查类型
            init_dram_weights(moeweights, invalid_param_dict, self.local_rank_pattern,self.first_k_dense_replace)

    def test_mismatched_local_rank_pattern(self):
        """测试local_rank_pattern形状不匹配"""
        world_size = 1
        moeweights = omni_placement.MoEWeights(self.num_expert, world_size)
        mismatched_pattern = torch.ones(self.num_layers + 1, self.num_expert, dtype=torch.bool).to(self.device)  # 多一层
        with self.assertRaises(IndexError):  # 假设 init_dram_weights 检查形状
            init_dram_weights(moeweights, self.param_dict, mismatched_pattern,first_k_dense_replace=self.first_k_dense_replace)

    def test_uninitialized_state(self):
        """测试未初始化的状态"""
        world_size = 1
        moeweights = omni_placement.MoEWeights(self.num_expert, world_size)
        self.assertFalse(moeweights.isShmInitialized())  # 未调用 init_dram_weights


if __name__ == "__main__":
    unittest.main()
    # tmp = TestMoeWeightsWrapper()
    # tmp.setUp()
    # tmp.test_mismatched_local_rank_pattern()
