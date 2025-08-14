import unittest
from unittest import mock
import torch
from typing import Any, Optional, Union
from torch.nn import Parameter
from vllm_ascend.ops.linear import AttnColumnParallelLinear, AttnRowParallelLinear
from vllm.model_executor.parallel_utils import get_mlp_tensor_model_parallel_world_size, get_mlp_tensor_model_parallel_rank
from vllm.quantization import QuantizationConfig, QuantMethodBase
from vllm.model_executor.layers import split_tensor_along_last_dim

class TestAttnColumnParallelLinear(unittest.TestCase):
    def setUp(self):
        # Mock tensor parallel world size and rank
        self.patcher_get_mlp_tp_size = mock.patch('your_module.get_mlp_tensor_model_parallel_world_size')
        self.mock_get_mlp_tp_size = self.patcher_get_mlp_tp_size.start()
        self.mock_get_mlp_tp_size.return_value = 2  # Simulate TP size = 2

        self.patcher_get_mlp_rank = mock.patch('your_module.get_mlp_tensor_model_parallel_rank')
        self.mock_get_mlp_rank = self.patcher_get_mlp_rank.start()
        self.mock_get_mlp_rank.return_value = 0  # Simulate TP rank = 0

        # Mock quantization method
        self.patcher_quant_method = mock.patch('your_module.QuantMethodBase')
        self.mock_quant_method = self.patcher_quant_method.start()
        self.mock_quant_method.return_value = mock.Mock(spec=QuantMethodBase)

    def tearDown(self):
        self.patcher_get_mlp_tp_size.stop()
        self.patcher_get_mlp_rank.stop()
        self.patcher_quant_method.stop()

    def test_init_with_bias_and_output_sizes(self):
        input_size = 128
        output_size = 256
        output_sizes = [64, 64, 128]
        bias = True
        quant_config = mock.Mock(spec=QuantizationConfig)

        layer = AttnColumnParallelLinear(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            output_sizes=output_sizes,
            quant_config=quant_config,
            prefix="test"
        )

        # Verify output partition sizes
        self.assertEqual(layer.output_size_per_partition, 128)
        self.assertEqual(layer.output_partition_sizes, [64, 64, 64])
        self.assertIsNotNone(layer.bias)

    def test_init_without_bias(self):
        input_size = 128
        output_size = 256
        bias = False

        layer = AttnColumnParallelLinear(
            input_size=input_size,
            output_size=output_size,
            bias=bias
        )

        self.assertIsNone(layer.bias)

    def test_weight_loader_with_sharded_weight(self):
        input_size = 128
        output_size = 256
        layer = AttnColumnParallelLinear(input_size, output_size, bias=False)
        param = Parameter(torch.empty(128, 128))  # Simulated weight
        param.is_sharded_weight = True  # Simulate sharded weight

        loaded_weight = torch.randn(256, 128)  # Full weight

        layer.weight_loader(param, loaded_weight)

        # Sharded weight should not be sliced
        self.assertTrue(torch.equal(param.data, loaded_weight))

    def test_weight_loader_without_sharding(self):
        input_size = 128
        output_size = 256
        layer = AttnColumnParallelLinear(input_size, output_size, bias=False)
        param = Parameter(torch.empty(128, 128))  # Simulated weight

        loaded_weight = torch.randn(256, 128)  # Full weight

        layer.weight_loader(param, loaded_weight)

        # Verify that the weight is sliced to TP rank's portion
        expected_weight = loaded_weight[128:256, :]  # TP rank 0
        self.assertTrue(torch.equal(param.data, expected_weight))

    def test_weight_loader_for_gguf_weight(self):
        input_size = 128
        output_size = 256
        layer = AttnColumnParallelLinear(input_size, output_size, bias=False)
        param = Parameter(torch.empty(128, 128))
        param.is_gguf_weight = True  # Simulate GGUF weight

        loaded_weight = torch.randn(256, 128)

        layer.weight_loader(param, loaded_weight)

        # Verify that GGUF weight is materialized with TP slicing
        expected_shape = (128, 128)
        self.assertEqual(param.data.shape, expected_shape)

    def test_forward_with_padding_and_all_gather(self):
        input_size = 128
        output_size = 256
        num_tokens = 3
        max_num_tokens_across_dp = 5

        # Mock forward context
        forward_context = mock.Mock()
        forward_context.max_tokens_across_dp = max_num_tokens_across_dp
        self.patcher_forward_context = mock.patch('your_module.get_forward_context')
        self.mock_forward_context = self.patcher_forward_context.start()
        self.mock_forward_context.return_value = forward_context

        # Mock all_gather
        self.patcher_all_gather = mock.patch('your_module.get_mlp_tp_group().all_gather')
        self.mock_all_gather = self.patcher_all_gather.start()
        self.mock_all_gather.return_value = torch.randn(5, 128)

        # Mock quant_method.apply
        self.mock_quant_method.apply.return_value = torch.randn(5, 128)

        layer = AttnColumnParallelLinear(input_size, output_size, bias=False)
        input_ = torch.randn(3, 128)

        output, output_bias = layer(input_, num_tokens)

        # Verify padding and all_gather
        self.assertEqual(input_.shape[0], 3)
        self.mock_all_gather.assert_called_once_with(input_, 0)
        self.assertEqual(output.shape, (5, 128))
        self.assertIsNone(output_bias)

    def test_forward_with_skip_bias_add(self):
        input_size = 128
        output_size = 256
        layer = AttnColumnParallelLinear(input_size, output_size, skip_bias_add=True, bias=True)
        input_ = torch.randn(5, 128)
        num_tokens = 5

        output, output_bias = layer(input_, num_tokens)

        # Verify output and bias are separated
        self.assertIsNotNone(output_bias)
        self.assertEqual(output.shape, (5, 128))

    def test_extra_repr(self):
        input_size = 128
        output_size = 256
        layer = AttnColumnParallelLinear(input_size, output_size, bias=True, gather_output=True)
        expected_repr = (
            f"in_features={input_size}, "
            f"output_features={128}, "
            f"bias=True, "
            f"tp_size=2, "
            f"gather_output=True"
        )
        self.assertEqual(str(layer), expected_repr)


class TestAttnRowParallelLinear(unittest.TestCase):
    def setUp(self):
        # Mock tensor parallel world size and rank
        self.patcher_get_mlp_tp_size = mock.patch('your_module.get_mlp_tensor_model_parallel_world_size')
        self.mock_get_mlp_tp_size = self.patcher_get_mlp_tp_size.start()
        self.mock_get_mlp_tp_size.return_value = 2  # Simulate TP size = 2

        self.patcher_get_mlp_rank = mock.patch('your_module.get_mlp_tensor_model_parallel_rank')
        self.mock_get_mlp_rank = self.patcher_get_mlp_rank.start()
        self.mock_get_mlp_rank.return_value = 0  # Simulate TP rank = 0

        # Mock split_tensor_along_last_dim
        self.patcher_split_tensor = mock.patch('your_module.split_tensor_along_last_dim')
        self.mock_split_tensor = self.patcher_split_tensor.start()

        # Mock reduce_scatter
        self.patcher_reduce_scatter = mock.patch('your_module.get_mlp_tp_group().reduce_scatter')
        self.mock_reduce_scatter = self.patcher_reduce_scatter.start()
        self.mock_reduce_scatter.return_value = torch.randn(5, 128)

        # Mock quantization method
        self.patcher_quant_method = mock.patch('your_module.QuantMethodBase')
        self.mock_quant_method = self.patcher_quant_method.start()
        self.mock_quant_method.return_value = mock.Mock(spec=QuantMethodBase)
        self.mock_quant_method.return_value.apply = mock.Mock(return_value=torch.randn(5, 128))

    def tearDown(self):
        self.patcher_get_mlp_tp_size.stop()
        self.patcher_get_mlp_rank.stop()
        self.patcher_split_tensor.stop()
        self.patcher_reduce_scatter.stop()
        self.patcher_quant_method.stop()

    def test_init_with_bias_and_input_parallel(self):
        input_size = 256
        output_size = 128
        layer = AttnRowParallelLinear(
            input_size=input_size,
            output_size=output_size,
            bias=True,
            input_is_parallel=True,
            prefix="test"
        )

        self.assertEqual(layer.input_size_per_partition, 128)
        self.assertEqual(layer.output_size_per_partition, 128)
        self.assertIsNotNone(layer.bias)

    def test_init_without_bias(self):
        input_size = 256
        output_size = 128
        layer = AttnRowParallelLinear(
            input_size=input_size,
            output_size=output_size,
            bias=False,
            input_is_parallel=False
        )

        self.assertIsNone(layer.bias)

    def test_init_reduce_results_false_with_bias(self):
        input_size = 256
        output_size = 128
        with self.assertRaises(ValueError):
            AttnRowParallelLinear(
                input_size=input_size,
                output_size=output_size,
                bias=True,
                reduce_results=False,
                skip_bias_add=False
            )

    def test_weight_loader_with_sharded_weight(self):
        input_size = 256
        output_size = 128
        layer = AttnRowParallelLinear(input_size, output_size, bias=False)
        param = Parameter(torch.empty(128, 128))  # Simulated weight
        param.is_sharded_weight = True  # Simulate sharded weight

        loaded_weight = torch.randn(256, 128)  # Full weight

        layer.weight_loader(param, loaded_weight)

        # Sharded weight should not be sliced
        self.assertTrue(torch.equal(param.data, loaded_weight))

    def test_weight_loader_without_sharding(self):
        input_size = 256
        output_size = 128
        layer = AttnRowParallelLinear(input_size, output_size, bias=False)
        param = Parameter(torch.empty(128, 128))  # Simulated weight

        loaded_weight = torch.randn(256, 128)  # Full weight

        layer.weight_loader(param, loaded_weight)

        # Verify that the weight is sliced to TP rank's portion
        expected_weight = loaded_weight[:128, :]  # TP rank 0
        self.assertTrue(torch.equal(param.data, expected_weight))

    def test_weight_loader_for_gguf_weight(self):
        input_size = 256
        output_size = 128
        layer = AttnRowParallelLinear(input_size, output_size, bias=False)
        param = Parameter(torch.empty(128, 128))
        param.is_gguf_weight = True  # Simulate GGUF weight

        loaded_weight = torch.randn(256, 128)

        layer.weight_loader(param, loaded_weight)

        # Verify that GGUF weight is materialized with TP slicing
        expected_shape = (128, 128)
        self.assertEqual(param.data.shape, expected_shape)

    def test_forward_input_parallel_true(self):
        input_size = 256
        output_size = 128
        num_tokens = 3
        layer = AttnRowParallelLinear(input_size, output_size, input_is_parallel=True, bias=False)
        input_ = torch.randn(3, 128)

        output, output_bias = layer(input_, num_tokens)

        # Verify no splitting and reduce_scatter
        self.mock_split_tensor.assert_not_called()
        self.mock_reduce_scatter.assert_called_once()
        self.assertEqual(output.shape, (3, 128))
        self.assertIsNone(output_bias)

    def test_forward_input_parallel_false(self):
        input_size = 256
        output_size = 128
        num_tokens = 3
        layer = AttnRowParallelLinear(input_size, output_size, input_is_parallel=False, bias=False)
        input_ = torch.randn(3, 256)

        output, output_bias = layer(input_, num_tokens)

        # Verify input was split and reduce_scatter
        self.mock_split_tensor.assert_called_once_with(input_, num_partitions=2)
        self.mock_reduce_scatter.assert_called_once()
        self.assertEqual(output.shape, (3, 128))

    def test_forward_with_skip_bias_add(self):
        input_size = 256
        output_size = 128
        layer = AttnRowParallelLinear(input_size, output_size, skip_bias_add=True, bias=True)
        input_ = torch.randn(5, 128)
        num_tokens = 5

        output, output_bias = layer(input_, num_tokens)

        # Verify output and bias are separated
        self.assertIsNotNone(output_bias)
        self.assertEqual(output.shape, (5, 128))

    def test_forward_output_truncation(self):
        input_size = 256
        output_size = 128
        num_tokens = 3
        layer = AttnRowParallelLinear(input_size, output_size, input_is_parallel=True, bias=False)
        input_ = torch.randn(5, 128)  # Simulate padded input

        output, output_bias = layer(input_, num_tokens)

        # Ensure output is truncated to num_tokens
        self.assertEqual(output.shape, (3, 128))

