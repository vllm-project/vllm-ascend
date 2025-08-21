import os
import unittest
from unittest import mock

import torch

from vllm_ascend.ops.linear import (AscendMlpColumnParallelLinear,
                                    AscendMlpMergedColumnParallelLinear,
                                    AscendMlpRowParallelLinear, LinearBase,
                                    QuantizationConfig)


class TestAscendMlpRowParallelLinear(unittest.TestCase):

    def setUp(self):
        os.environ["VLLM_ASCEND_ENABLE_MLP_OPTIMIZE"] = "1"
        self.tensor_parallel_world_size = 2
        self.tensor_parallel_rank = 0
        self.mlp_tensor_parallel_world_size = 2
        self.mlp_tensor_parallel_rank = 1

        self.get_tensor_model_parallel_world_size_patch = mock.patch(
            'vllm_ascend.ops.linear.get_tensor_model_parallel_world_size',
            return_value=self.tensor_parallel_world_size)
        self.get_tensor_model_parallel_rank_patch = mock.patch(
            'vllm_ascend.ops.linear.get_tensor_model_parallel_rank',
            return_value=self.tensor_parallel_rank)
        self.get_mlp_tensor_model_parallel_world_size_patch = mock.patch(
            'vllm_ascend.ops.linear.get_mlp_tensor_model_parallel_world_size',
            return_value=self.mlp_tensor_parallel_world_size)
        self.get_mlp_tensor_model_parallel_rank_patch = mock.patch(
            'vllm_ascend.ops.linear.get_mlp_tensor_model_parallel_rank',
            return_value=self.mlp_tensor_parallel_rank)

        self.get_tensor_model_parallel_world_size_mock = \
            self.get_tensor_model_parallel_world_size_patch.start()
        self.get_tensor_model_parallel_rank_mock = \
            self.get_tensor_model_parallel_rank_patch.start()
        self.get_mlp_tensor_model_parallel_world_size_mock = \
            self.get_mlp_tensor_model_parallel_world_size_patch.start()
        self.get_mlp_tensor_model_parallel_rank_mock = \
            self.get_mlp_tensor_model_parallel_rank_patch.start()

        self.split_tensor_along_last_dim_patch = mock.patch(
            'vllm_ascend.ops.linear.split_tensor_along_last_dim',
            return_value=(torch.randn(10, 8), torch.randn(10, 8)))
        self.tensor_model_parallel_all_reduce_patch = mock.patch(
            'vllm_ascend.ops.linear.tensor_model_parallel_all_reduce',
            return_value=torch.randn(10, 8))
        self.tensor_model_parallel_all_reduce_mock = \
            self.tensor_model_parallel_all_reduce_patch.start()
        self.split_tensor_along_last_dim_mock = \
            self.split_tensor_along_last_dim_patch.start()
        self.get_mlp_tp_group_patch = \
            mock.patch('vllm_ascend.ops.linear.get_mlp_tp_group')
        self.get_mlp_tp_group_mock = self.get_mlp_tp_group_patch.start()
        self.get_mlp_tp_group_mock.return_value = mock.MagicMock()
        self.get_mlp_tp_group_mock.return_value.reduce_scatter = \
            mock.MagicMock()

    def tearDown(self):
        self.get_tensor_model_parallel_world_size_patch.stop()
        self.get_tensor_model_parallel_rank_patch.stop()
        self.get_mlp_tensor_model_parallel_world_size_patch.stop()
        self.get_mlp_tensor_model_parallel_rank_patch.stop()
        self.split_tensor_along_last_dim_patch.stop()
        self.tensor_model_parallel_all_reduce_patch.stop()
        self.get_mlp_tp_group_patch.stop()

    def test_init_with_down_proj_prefix(self):
        layer = AscendMlpRowParallelLinear(input_size=16,
                                           output_size=8,
                                           prefix="down_proj")
        self.assertEqual(layer.tp_size, self.mlp_tensor_parallel_world_size)
        self.assertEqual(layer.tp_rank, self.mlp_tensor_parallel_rank)
        self.assertTrue(layer.enable_mlp_optimze)

    def test_forward_with_mlp_optimize(self):
        layer = AscendMlpRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="down_proj",
            input_is_parallel=False,
        )
        input_tensor = torch.randn(16, 8)  # (batch_size, input_size)
        layer(input_tensor)

        self.split_tensor_along_last_dim_mock.assert_called_once_with(
            input_tensor, num_partitions=layer.tp_size)

    def test_forward_without_mlp_optimize(self):
        layer = AscendMlpRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="other",
            input_is_parallel=False,
        )
        input_tensor = torch.randn(16, 8)
        layer(input_tensor)

        self.split_tensor_along_last_dim_mock.assert_called_once_with(
            input_tensor, num_partitions=layer.tp_size)
        self.tensor_model_parallel_all_reduce_mock.assert_called_once()

    def test_skip_bias_add(self):
        layer = AscendMlpRowParallelLinear(
            input_size=16,
            output_size=8,
            skip_bias_add=True,
        )
        input_tensor = torch.randn(16, 8)
        output, bias = layer(input_tensor)

        self.assertIsNotNone(bias)

    def test_no_reduce_results(self):
        layer = AscendMlpRowParallelLinear(input_size=16,
                                           output_size=8,
                                           reduce_results=False,
                                           bias=False)
        input_tensor = torch.randn(16, 8)
        layer(input_tensor)

        self.tensor_model_parallel_all_reduce_mock.assert_not_called()

    def test_input_not_parallel(self):
        layer = AscendMlpRowParallelLinear(input_size=16,
                                           output_size=8,
                                           input_is_parallel=False)
        input_tensor = torch.randn(16, 8)
        layer(input_tensor)

        self.split_tensor_along_last_dim_mock.assert_called_once()

    def test_exception_when_reduce_false_and_bias(self):
        with self.assertRaises(ValueError):
            AscendMlpRowParallelLinear(input_size=16,
                                       output_size=8,
                                       reduce_results=False,
                                       bias=True,
                                       skip_bias_add=False)


class TestAscendMlpColumnParallelLinear(unittest.TestCase):

    def setUp(self):
        os.environ["VLLM_ASCEND_ENABLE_MLP_OPTIMIZE"] = "1"
        # Mock distributed functions
        self.mlp_tp_size_patch = \
            mock.patch('vllm_ascend.ops.linear.get_mlp_tensor_model_parallel_world_size')
        self.mlp_tp_size_mock = self.mlp_tp_size_patch.start()
        self.mlp_tp_size_mock.return_value = 2  # Simulate 2 GPUs in MLP TP group

        self.mlp_tp_rank_patch = \
            mock.patch('vllm_ascend.ops.linear.get_mlp_tensor_model_parallel_rank')
        self.mlp_tp_rank_mock = self.mlp_tp_rank_patch.start()
        self.mlp_tp_rank_mock.return_value = 0  # Current GPU rank

        self.tp_size_patch = \
            mock.patch('vllm_ascend.ops.linear.get_tensor_model_parallel_world_size')
        self.tp_size_mock = self.tp_size_patch.start()
        self.tp_size_mock.return_value = 4  # Simulate 4 GPUs in regular TP group

        self.tp_rank_patch = \
            mock.patch('vllm_ascend.ops.linear.get_tensor_model_parallel_rank')
        self.tp_rank_mock = self.tp_rank_patch.start()
        self.tp_rank_mock.return_value = 1  # Current GPU rank

        # Mock divide function (assumed to be in your module)
        self.divide_patch = mock.patch('vllm_ascend.ops.linear.divide')
        self.divide_mock = self.divide_patch.start()
        self.divide_mock.side_effect = lambda x, y: x // y  # Simulate division

        # Mock QuantizationConfig and QuantMethod
        self.quant_config_mock = mock.MagicMock(spec=QuantizationConfig)

        # Mock LinearBase initialization
        self.linear_base_init_patch = mock.patch.object(
            LinearBase, "__init__", side_effect=self.mock_linear_base_init)
        self.linear_base_init_patch.start()

        self.quant_method_mock = mock.MagicMock()

    def mock_linear_base_init(self, instance, *args, **kwargs):
        instance.quant_method = self.quant_method_mock
        instance.params_dtype = mock.MagicMock()

        instance.input_size = 16
        instance.output_size = 8
        instance.output_size_per_partition = 4
        instance.params_dtype = torch.float32

    def tearDown(self):
        self.mlp_tp_size_patch.stop()
        self.mlp_tp_rank_patch.stop()
        self.tp_size_patch.stop()
        self.tp_rank_patch.stop()
        self.divide_patch.stop()
        self.linear_base_init_patch.stop()

    def test_mlp_optimize_initialization(self):
        # Test when prefix contains "gate_up_proj"
        with mock.patch.object(torch.nn.Module, 'register_parameter'):
            layer = AscendMlpColumnParallelLinear(
                input_size=16,
                output_size=8,
                prefix="model.layers.0.gate_up_proj",
                bias=False,
            )

        # Verify MLP optimization flags
        self.assertTrue(layer.enable_mlp_optimze)
        self.assertEqual(layer.tp_size, 2)
        self.assertEqual(layer.tp_rank, 0)
        self.assertEqual(layer.input_size_per_partition, 16)
        self.assertEqual(layer.output_size_per_partition, 4)

        # Check quant_method.create_weights was called
        self.quant_method_mock.create_weights.assert_called_once()

    def test_regular_parallel_initialization(self):
        # Test when prefix does NOT contain "gate_up_proj"
        with mock.patch.object(torch.nn.Module, 'register_parameter'):
            layer = AscendMlpColumnParallelLinear(
                input_size=16,
                output_size=8,
                prefix="model.layers.0.q_proj",
                quant_config=self.quant_config_mock,
                bias=False,
            )

        # Verify regular TP flags
        self.assertFalse(layer.enable_mlp_optimze)
        self.assertEqual(layer.tp_size, 4)
        self.assertEqual(layer.tp_rank, 1)
        self.assertEqual(layer.input_size_per_partition, 16)
        self.assertEqual(layer.output_size_per_partition, 4)
        # Check quant_method.create_weights was called
        self.quant_method_mock.create_weights.assert_called_once()

    def test_output_sizes_handling(self):
        # Test when output_sizes is provided
        with mock.patch.object(torch.nn.Module, 'register_parameter'):
            layer = AscendMlpColumnParallelLinear(
                input_size=16,
                output_size=8,
                output_sizes=[4, 4],
                prefix="model.layers.0.qkv_proj",
                quant_config=self.quant_config_mock,
                bias=False,
            )

        # Verify output_partition_sizes
        self.assertEqual(layer.output_partition_sizes, [2])


class TestAscendMlpMergedColumnParallelLinear(unittest.TestCase):

    def setUp(self):
        os.environ["VLLM_ASCEND_ENABLE_MLP_OPTIMIZE"] = "1"
        # Mock get_mlp_tensor_model_parallel_world_size and get_tensor_model_parallel_world_size
        self.mlp_world_size_patch = \
            mock.patch("vllm_ascend.ops.linear.get_mlp_tensor_model_parallel_world_size", return_value=2)
        self.tensor_world_size_patch = \
            mock.patch("vllm_ascend.ops.linear.get_tensor_model_parallel_world_size", return_value=2)
        self.mlp_world_size_patch.start()
        self.tensor_world_size_patch.start()

        # Mock get_mlp_tensor_model_parallel_rank and get_tensor_model_parallel_rank
        self.mlp_rank_patch = \
            mock.patch("vllm_ascend.ops.linear.get_mlp_tensor_model_parallel_rank", return_value=0)
        self.tensor_rank_patch = \
            mock.patch("vllm_ascend.ops.linear.get_tensor_model_parallel_rank", return_value=0)
        self.mlp_rank_patch.start()
        self.tensor_rank_patch.start()

        # Mock all_gather methods
        self.get_mlp_tp_group_patch = \
            mock.patch('vllm_ascend.ops.linear.get_mlp_tp_group')
        self.get_mlp_tp_group_mock = self.get_mlp_tp_group_patch.start()
        self.get_mlp_tp_group_mock.return_value = mock.MagicMock()
        self.get_mlp_tp_group_mock.return_value.all_gather = mock.MagicMock()
        self.tensor_model_parallel_all_gather_patch = mock.patch(
            'vllm_ascend.ops.linear.tensor_model_parallel_all_gather',
            return_value=torch.randn(10, 8))
        self.tensor_model_parallel_all_gather_mock = \
            self.tensor_model_parallel_all_gather_patch.start()

        # Mock AscendMlpColumnParallelLinear's __init__
        self.linear_init_patch = mock.patch.object(
            AscendMlpColumnParallelLinear,
            "__init__",
            side_effect=self.mock_linear_init)
        self.linear_init_patch.start()

        # Create mock objects
        self.quant_method_mock = mock.MagicMock()
        self.apply_output = torch.randn(2, 8)

        self.quant_method_mock.apply.return_value = self.apply_output

    def mock_linear_init(self, instance, *args, **kwargs):
        torch.nn.Module.__init__(instance)
        # Set quant_method and other attributes
        instance.quant_method = self.quant_method_mock
        instance.bias = torch.nn.Parameter(torch.randn(8))  # Example bias
        instance.input_size = 16
        instance.output_size = 8
        instance.gather_output = False
        instance.skip_bias_add = False
        instance.return_bias = True

    def test_forward_with_enable_mlp_optimze(self):
        # Setup input
        input_tensor = torch.randn(1, 16)

        # Create instance with prefix "gate_up_proj" to trigger enable_mlp_optimze = True
        layer = AscendMlpMergedColumnParallelLinear(input_size=16,
                                                    output_sizes=[8],
                                                    bias=True,
                                                    gather_output=False,
                                                    skip_bias_add=False,
                                                    params_dtype=torch.float32,
                                                    quant_config=None,
                                                    prefix="other_proj")

        # Call forward
        output, bias = layer(input_tensor)

        # Validate calls
        self.assertEqual(output.shape, self.apply_output.shape)

    def test_forward_without_enable_mlp_optimze(self):
        # Setup input
        input_tensor = torch.randn(1, 16)

        # Create instance with prefix not containing "gate_up_proj"
        layer = AscendMlpMergedColumnParallelLinear(input_size=16,
                                                    output_sizes=[8],
                                                    bias=True,
                                                    gather_output=False,
                                                    skip_bias_add=False,
                                                    params_dtype=torch.float32,
                                                    quant_config=None,
                                                    prefix="other_proj")

        # Call forward
        output, bias = layer(input_tensor)

        # Validate calls
        self.quant_method_mock.apply.assert_called_once_with(
            layer, input_tensor, layer.bias)
        self.tensor_model_parallel_all_gather_mock.assert_not_called()
        self.assertEqual(output.shape, self.apply_output.shape)

    def tearDown(self):
        self.linear_init_patch.stop()
        self.mlp_world_size_patch.stop()
        self.tensor_world_size_patch.stop()
        self.mlp_rank_patch.stop()
        self.tensor_rank_patch.stop()
        self.get_mlp_tp_group_mock.stop()
        self.tensor_model_parallel_all_gather_mock.stop()
