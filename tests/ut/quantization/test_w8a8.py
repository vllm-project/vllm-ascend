import unittest
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.quantization.w8a8 import (AscendC8KVCacheMethod,
                                           AscendW8A8LinearMethod,
                                           quant_per_tensor)
from vllm_ascend.utils import AscendDeviceType


class TestQuantPerTensor(TestBase):

    @patch("torch_npu.npu_quantize")
    def test_quant_per_tensor(self, mock_npu_quantize):
        in_tensor = torch.randn(32, 128)
        input_scale = torch.tensor(0.1)
        input_offset = torch.tensor(0)

        expected_output = torch.randint(-128, 127, (32, 128), dtype=torch.int8)
        mock_npu_quantize.return_value = expected_output

        output = quant_per_tensor(in_tensor, input_scale, input_offset)

        mock_npu_quantize.assert_called_once_with(
            in_tensor,
            input_scale,
            input_offset,
            torch.qint8,
            -1,
            False,
        )

        self.assertTrue(torch.equal(output, expected_output))


class TestAscendW8A8LinearMethod(TestBase):

    def setUp(self):
        self.method = AscendW8A8LinearMethod()

    def test_get_weight(self):
        weight = self.method.get_weight(10, 20)
        self.assertEqual(weight['weight'].dtype, torch.int8)
        self.assertEqual(weight['weight'].shape, (20, 10))

    def test_get_pertensor_param(self):
        params = self.method.get_pertensor_param(torch.bfloat16)
        self.assertEqual(params['input_scale'].dtype, torch.bfloat16)
        self.assertEqual(params['input_offset'].dtype, torch.int8)
        self.assertEqual(params['input_scale'].shape, (1, ))
        self.assertEqual(params['input_offset'].shape, (1, ))

    def test_get_perchannel_param(self):
        params = self.method.get_perchannel_param(10, torch.bfloat16)

        self.assertEqual(params['quant_bias'].dtype, torch.int32)
        self.assertEqual(params['deq_scale'].dtype, torch.float32)
        self.assertEqual(params['weight_scale'].dtype, torch.bfloat16)
        self.assertEqual(params['weight_offset'].dtype, torch.bfloat16)
        self.assertEqual(params['quant_bias'].shape, (10, ))
        self.assertEqual(params['deq_scale'].shape, (10, ))
        self.assertEqual(params['weight_scale'].shape, (10, 1))
        self.assertEqual(params['weight_offset'].shape, (10, 1))

    @patch("vllm_ascend.quantization.w8a8.get_forward_context")
    @patch("vllm_ascend.quantization.w8a8.quant_per_tensor")
    @patch("torch_npu.npu_quant_matmul")
    def test_apply_with_x_not_int8(self, mock_npu_quant_matmul,
                                   mock_quant_per_tensor,
                                   mock_get_forward_context):
        layer = MagicMock()
        layer.aclnn_input_scale = 0.1
        layer.aclnn_input_offset = 0.2
        layer.weight = torch.randn(128, 256)
        layer.deq_scale = 0.3

        mock_forward_context = MagicMock()
        mock_get_forward_context.return_value = mock_forward_context
        mock_weight_prefetch_method = MagicMock()
        mock_forward_context.weight_prefetch_method = mock_weight_prefetch_method

        x = torch.randn(32, 128)
        bias = torch.randn(256)
        mock_quant_per_tensor.return_value = torch.randint(-128,
                                                           127,
                                                           x.shape,
                                                           dtype=torch.int8)

        expected_y_output = torch.randn(32, 256)
        mock_npu_quant_matmul.return_value = expected_y_output

        output = self.method.apply(layer, x, bias)

        expected_y_output += bias
        self.assertTrue(torch.equal(output, expected_y_output))

    @patch("torch_npu.npu_quant_matmul")
    def test_apply_with_x_is_int8(self, mock_npu_quant_matmul):
        layer = MagicMock()
        layer.aclnn_input_scale = 0.1
        layer.aclnn_input_offset = 0.2
        layer.weight = torch.randn(128, 256)
        layer.deq_scale = 0.3

        x = torch.randint(-128, 127, (32, 128), dtype=torch.int8)
        bias = torch.randn(256)

        expected_y_output = torch.randn(32, 256)
        mock_npu_quant_matmul.return_value = expected_y_output

        output = self.method.apply(layer, x, bias)
        expected_y_output += bias
        self.assertTrue(torch.equal(output, expected_y_output))

    @patch('vllm_ascend.utils.get_ascend_device_type',
           return_value=AscendDeviceType._310P)
    @patch("torch_npu.npu_quant_matmul")
    def test_apply_with_x_is_310p(self, mock_npu_quant_matmul,
                                  mock_soc_version):
        layer = MagicMock()
        layer.aclnn_input_scale = 0.1
        layer.aclnn_input_offset = 0.2
        layer.weight = torch.randn(128, 256)
        layer.deq_scale = 0.3

        x = torch.randint(-128, 127, (32, 128), dtype=torch.int8)
        bias = torch.randn(256)

        expected_y_output = torch.randn(32, 256)
        mock_npu_quant_matmul.return_value = expected_y_output

        output = self.method.apply(layer, x, bias)
        expected_y_output += bias
        self.assertTrue(torch.equal(output, expected_y_output))

    @patch("vllm_ascend.quantization.w8a8.is_enable_nz")
    @patch('torch_npu.npu_format_cast')
    def test_process_weights_after_loading_not_nz(self, mock_npu_format_cast,
                                                  mock_is_nz):
        layer = MagicMock()

        layer.weight.data = torch.randn(128, 256)
        layer.input_scale.data = torch.tensor([0.1])
        layer.input_offset.data = torch.tensor([0])
        layer.deq_scale = torch.tensor([0.5])
        layer.weight_scale.data = torch.randn(128, 1)
        layer.weight_offset.data = torch.randn(128, 1)

        mock_is_nz.return_value = 0
        mock_npu_format_cast.return_value = MagicMock
        self.method.process_weights_after_loading(layer)

        expected_offset = torch.tensor([0]).repeat(256).to(torch.int8)
        self.assertTrue(
            torch.equal(layer.aclnn_input_offset.data, expected_offset))
        self.assertFalse(layer.aclnn_input_offset.requires_grad)

        self.assertFalse(layer.deq_scale.requires_grad)

        self.assertEqual(layer.weight_scale.data.shape, (128, ))
        self.assertEqual(layer.weight_offset.data.shape, (128, ))
        mock_npu_format_cast.assert_not_called()

    @patch("vllm_ascend.quantization.w8a8.is_enable_nz")
    @patch('torch_npu.npu_format_cast')
    def test_process_weights_after_loading_nz(self, mock_npu_format_cast,
                                              mock_is_nz):
        layer = MagicMock()

        layer.weight.data = torch.randn(128, 256)
        layer.input_scale.data = torch.tensor([0.1])
        layer.input_offset.data = torch.tensor([0])
        layer.deq_scale = torch.tensor([0.5])
        layer.weight_scale.data = torch.randn(128, 1)
        layer.weight_offset.data = torch.randn(128, 1)

        mock_is_nz.return_value = 1
        mock_npu_format_cast.return_value = MagicMock
        self.method.process_weights_after_loading(layer)

        expected_offset = torch.tensor([0]).repeat(256).to(torch.int8)
        self.assertTrue(
            torch.equal(layer.aclnn_input_offset.data, expected_offset))
        self.assertFalse(layer.aclnn_input_offset.requires_grad)

        self.assertFalse(layer.deq_scale.requires_grad)

        self.assertEqual(layer.weight_scale.data.shape, (128, ))
        self.assertEqual(layer.weight_offset.data.shape, (128, ))
        mock_npu_format_cast.assert_called_once()


class TestAscendC8KVCacheMethod(TestBase):

    def setUp(self):
        self.layer = MagicMock()
        self.layer.num_kv_heads = 4
        self.layer.head_size = 64
        self.layer.num_heads = 8
        self.layer._k_scale_float = 1.0
        self.layer._v_scale_float = 1.0
        self.method = AscendC8KVCacheMethod()

        self.attention_type = MagicMock()
        self.attention_type.DECODER = "decoder"
        self.attention_type.ENCODER = "encoder"

    def test_create_weights(self):
        AscendC8KVCacheMethod.create_weights(self.layer)

        self.layer.register_parameter.assert_any_call("key_antiquant_scale",
                                                      unittest.mock.ANY)
        self.layer.register_parameter.assert_any_call("value_antiquant_scale",
                                                      unittest.mock.ANY)

        calls = self.layer.register_parameter.call_args_list

        for call in calls:
            args, kwargs = call
            param = kwargs.get('parameter', args[1] if len(args) > 1 else None)

            expected_shape = (self.layer.num_kv_heads * self.layer.head_size, )
            self.assertEqual(param.shape, expected_shape)

    @patch('vllm_ascend.utils.get_ascend_device_type',
           return_value=AscendDeviceType.A3)
    def test_process_weights_after_loading_not_310p(self, mock_soc_version):
        key_data = torch.ones(4 * 64)
        value_data = torch.ones(4 * 64) * 2

        self.layer.key_antiquant_scale.data = key_data
        self.layer.value_antiquant_scale.data = value_data

        self.method.process_weights_after_loading(self.layer)

        self.assertEqual(self.method.antiquant_scale_comb.shape, (2, 256))
        self.assertTrue(torch.all(self.method.antiquant_scale_comb[0] == 1))
        self.assertTrue(torch.all(self.method.antiquant_scale_comb[1] == 2))

    @patch('vllm_ascend.utils.get_ascend_device_type',
           return_value=AscendDeviceType._310P)
    def test_process_weights_after_loading_is_310p(self, mock_soc_version):
        key_data = torch.ones(4 * 64)
        value_data = torch.ones(4 * 64) * 2

        self.layer.key_antiquant_scale.data = key_data
        self.layer.value_antiquant_scale.data = value_data

        self.method.process_weights_after_loading(self.layer)

        self.assertEqual(self.method.antiquant_scale_comb.shape, (2, 256))
        self.assertTrue(torch.all(self.method.antiquant_scale_comb[0] == 1))
        self.assertTrue(torch.all(self.method.antiquant_scale_comb[1] == 2))

    @patch('torch_npu.npu_scatter_nd_update_')
    @patch("vllm_ascend.quantization.w8a8.quant_per_tensor")
    def test_apply_decode_only(self, mock_quant, mock_scatter):

        num_tokens = 2
        query = torch.randn(num_tokens,
                            self.layer.num_heads * self.layer.head_size)
        key = torch.randn(num_tokens,
                          self.layer.num_kv_heads * self.layer.head_size)
        value = torch.randn(num_tokens,
                            self.layer.num_kv_heads * self.layer.head_size)
        output = torch.empty_like(query)

        attn_metadata = MagicMock()
        attn_metadata.attn_state = AscendAttentionState.DecodeOnly
        attn_metadata.seq_lens = [10, 10]
        attn_metadata.block_tables = torch.tensor([[0, 1], [1, 2]])
        attn_metadata.slot_mapping = torch.tensor([0, 1])
        attn_metadata.attn_mask = None

        block_size = 16
        key_cache = torch.empty(2, block_size, self.layer.num_kv_heads,
                                self.layer.head_size)
        value_cache = torch.empty(2, block_size, self.layer.num_kv_heads,
                                  self.layer.head_size)
        kv_cache = (key_cache, value_cache)

        mock_quant.side_effect = [key, value]

        self.layer.key_antiquant_scale.data = torch.ones(
            self.layer.num_kv_heads * self.layer.head_size)
        self.layer.value_antiquant_scale.data = torch.ones(
            self.layer.num_kv_heads * self.layer.head_size)
        self.method.process_weights_after_loading(self.layer)

        expected_output = torch.randn(
            num_tokens, self.layer.num_heads * self.layer.head_size)
        with patch('torch_npu.npu_incre_flash_attention',
                   return_value=expected_output):
            result = self.method.apply(self.layer, query, key, value, kv_cache,
                                       attn_metadata,
                                       self.attention_type.DECODER, 1.0,
                                       output)

            self.assertEqual(mock_quant.call_count, 2)
            self.assertEqual(mock_scatter.call_count, 2)
            self.assertTrue(torch.equal(result, expected_output))

    @patch('torch_npu.npu_scatter_nd_update_')
    @patch("vllm_ascend.quantization.w8a8.quant_per_tensor")
    def test_apply_attn_metadata_without_decode(self, mock_quant,
                                                mock_scatter):

        num_tokens = 2
        query = torch.randn(num_tokens,
                            self.layer.num_heads * self.layer.head_size)
        key = torch.randn(num_tokens,
                          self.layer.num_kv_heads * self.layer.head_size)
        value = torch.randn(num_tokens,
                            self.layer.num_kv_heads * self.layer.head_size)
        output = torch.empty_like(query)

        attn_metadata = MagicMock(spec=[
            'attn_state', 'seq_lens', 'block_tables', 'slot_mapping',
            'attn_mask'
        ])
        attn_metadata.attn_state = AscendAttentionState.DecodeOnly
        attn_metadata.seq_lens = [10, 10]
        attn_metadata.block_tables = torch.tensor([[0, 1], [1, 2]])
        attn_metadata.slot_mapping = torch.tensor([0, 1])
        attn_metadata.attn_mask = None

        block_size = 16
        key_cache = torch.empty(2, block_size, self.layer.num_kv_heads,
                                self.layer.head_size)
        value_cache = torch.empty(2, block_size, self.layer.num_kv_heads,
                                  self.layer.head_size)
        kv_cache = (key_cache, value_cache)

        mock_quant.side_effect = [key, value]

        self.layer.key_antiquant_scale.data = torch.ones(
            self.layer.num_kv_heads * self.layer.head_size)
        self.layer.value_antiquant_scale.data = torch.ones(
            self.layer.num_kv_heads * self.layer.head_size)
        self.method.process_weights_after_loading(self.layer)

        expected_output = torch.randn(
            num_tokens, self.layer.num_heads * self.layer.head_size)
        with patch('torch_npu.npu_incre_flash_attention',
                   return_value=expected_output):
            result = self.method.apply(self.layer, query, key, value, kv_cache,
                                       attn_metadata,
                                       self.attention_type.DECODER, 1.0,
                                       output)

            self.assertEqual(mock_quant.call_count, 2)
            self.assertEqual(mock_scatter.call_count, 2)
            self.assertTrue(torch.equal(result, expected_output))

    @patch("vllm_ascend.quantization.w8a8.quant_per_tensor")
    @patch('torch_npu._npu_flash_attention')
    def test_apply_prefill_no_cache(self, mock_flash, mock_quant):
        """Test apply method in prefill no-cache mode"""

        num_tokens = 2
        query = torch.randn(num_tokens,
                            self.layer.num_heads * self.layer.head_size)
        key = torch.randn(num_tokens,
                          self.layer.num_kv_heads * self.layer.head_size)
        value = torch.randn(num_tokens,
                            self.layer.num_kv_heads * self.layer.head_size)
        output = torch.empty_like(query)

        attn_metadata = MagicMock()
        attn_metadata.attn_state = AscendAttentionState.PrefillNoCache
        attn_metadata.seq_lens = [10, 10]
        attn_metadata.attn_mask = torch.ones(2, 2)

        kv_cache = (torch.tensor([]), torch.tensor([]))
        mock_quant.return_value = key

        result = self.method.apply(self.layer, query, key, value, kv_cache,
                                   attn_metadata, self.attention_type.DECODER,
                                   1.0, output)

        # Check that flash attention was called
        mock_flash.assert_called_once()

        # Check output shape
        self.assertEqual(
            result.shape,
            (num_tokens, self.layer.num_heads * self.layer.head_size))

    @patch("vllm_ascend.quantization.w8a8.quant_per_tensor")
    def test_apply_unsupported_attention_type(self, mock_quant):

        query = torch.randn(1, self.layer.num_heads * self.layer.head_size)
        key = torch.randn(1, self.layer.num_kv_heads * self.layer.head_size)
        value = torch.randn(1, self.layer.num_kv_heads * self.layer.head_size)
        output = torch.empty_like(query)

        mock_quant.return_value = key

        attn_metadata = MagicMock()
        attn_metadata.attn_state = AscendAttentionState.PrefillNoCache

        with self.assertRaises(NotImplementedError) as cm:
            self.method.apply(self.layer, query, key, value, (None, None),
                              attn_metadata, self.attention_type.ENCODER, 1.0,
                              output)

        assert "Encoder self-attention" in str(
            cm.exception), f"Encoder self-attention not in {str(cm.exception)}"
        assert "not implemented" in str(
            cm.exception), f"not implemented not in{str(cm.exception)}"

        mock_quant.assert_not_called()

    @patch("vllm_ascend.quantization.w8a8.quant_per_tensor")
    def test_apply_unsupported_attention_state(self, mock_quant):
        """Test apply with unsupported attention state"""
        query = torch.randn(1, self.layer.num_heads * self.layer.head_size)
        key = torch.randn(1, self.layer.num_kv_heads * self.layer.head_size)
        value = torch.randn(1, self.layer.num_kv_heads * self.layer.head_size)
        output = torch.empty_like(query)

        attn_metadata = MagicMock()
        attn_metadata.attn_state = AscendAttentionState.PrefillCacheHit
        mock_quant.return_value = key
        kv_cache = (torch.tensor([]), torch.tensor([]))

        with self.assertRaises(NotImplementedError):
            self.method.apply(self.layer, query, key, value, kv_cache,
                              attn_metadata, self.attention_type.DECODER, 1.0,
                              output)
