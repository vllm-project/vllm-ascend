import unittest
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.moe.experts_selector import (_native_grouped_topk,
                                                  select_experts)
from vllm_ascend.quantization.w8a8 import (AscendC8KVCacheMethod,
                                           AscendW8A8FusedMoEMethod,
                                           AscendW8A8LinearMethod,
                                           fused_experts, fused_experts_310p,
                                           quant_per_tensor)


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

    @patch("vllm_ascend.quantization.w8a8.quant_per_tensor")
    @patch("torch_npu.npu_quant_matmul")
    def test_apply_with_x_not_int8(self, mock_npu_quant_matmul,
                                   mock_quant_per_tensor):
        layer = MagicMock()
        layer.aclnn_input_scale = 0.1
        layer.aclnn_input_offset = 0.2
        layer.weight = torch.randn(128, 256)
        layer.deq_scale = 0.3

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

    @patch("vllm_ascend.quantization.w8a8.is_310p", return_value=True)
    @patch("torch_npu.npu_quant_matmul")
    def test_apply_with_x_is_310p(self, mock_npu_quant_matmul, mock_is_310p):
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

    @patch('torch_npu.npu_format_cast')
    def test_process_weights_after_loading(self, mock_npu_format_cast):
        layer = MagicMock()

        layer.weight.data = torch.randn(128, 256)
        layer.input_scale.data = torch.tensor([0.1])
        layer.input_offset.data = torch.tensor([0])
        layer.deq_scale = torch.tensor([0.5])
        layer.weight_scale.data = torch.randn(128, 1)
        layer.weight_offset.data = torch.randn(128, 1)

        mock_npu_format_cast.return_value = MagicMock
        self.method.process_weights_after_loading(layer)

        expected_offset = torch.tensor([0]).repeat(256).to(torch.int8)
        self.assertTrue(
            torch.equal(layer.aclnn_input_offset.data, expected_offset))
        self.assertFalse(layer.aclnn_input_offset.requires_grad)

        self.assertFalse(layer.deq_scale.requires_grad)

        self.assertEqual(layer.weight_scale.data.shape, (128, ))
        self.assertEqual(layer.weight_offset.data.shape, (128, ))


class TestAscendW8A8FusedMoEMethod(TestBase):

    def setUp(self):
        self.moe_method = AscendW8A8FusedMoEMethod()
        self.num_experts = 4
        self.intermediate_size = 64
        self.hidden_size = 128
        self.dtype = torch.float32

    def test_init(self):
        self.assertTrue(self.moe_method.transpose_weight)

    def test_get_weight(self):
        weights = self.moe_method.get_weight(
            num_experts=self.num_experts,
            intermediate_size_per_partition=self.intermediate_size,
            hidden_sizes=self.hidden_size,
            params_dtype=self.dtype)

        assert "w13_weight" in weights, f"w13_weight not in {weights}"
        assert "w2_weight" in weights, f"w2_weight not in {weights}"
        self.assertEqual(
            weights["w13_weight"].shape,
            (self.num_experts, 2 * self.intermediate_size, self.hidden_size))
        self.assertEqual(
            weights["w2_weight"].shape,
            (self.num_experts, self.hidden_size, self.intermediate_size))
        self.assertEqual(weights["w13_weight"].dtype, torch.int8)
        self.assertEqual(weights["w2_weight"].dtype, torch.int8)
        self.assertFalse(weights["w13_weight"].requires_grad)
        self.assertFalse(weights["w2_weight"].requires_grad)

    def test_get_dynamic_quant_param(self):
        quant_params = self.moe_method.get_dynamic_quant_param(
            num_experts=self.num_experts,
            intermediate_size_per_partition=self.intermediate_size,
            hidden_sizes=self.hidden_size,
            params_dtype=self.dtype)

        expected_params = [
            "w13_weight_scale", "w13_weight_offset", "w2_weight_scale",
            "w2_weight_offset", "w2_deq_scale", "w13_deq_scale",
            "w2_input_scale", "w13_input_scale", "w2_input_offset",
            "w13_input_offset", "quant_bias"
        ]

        for param in expected_params:
            assert param in quant_params, f"{param} not in {quant_params}"

        # Check some sample shapes
        self.assertEqual(quant_params["w13_weight_scale"].shape,
                         (self.num_experts, 2 * self.intermediate_size, 1))
        self.assertEqual(quant_params["w2_input_offset"].shape,
                         (self.num_experts, 1))
        self.assertEqual(quant_params["quant_bias"].shape,
                         (self.num_experts, self.hidden_size))

    @patch('vllm_ascend.quantization.w8a8.select_experts')
    @patch('vllm_ascend.quantization.w8a8.fused_experts')
    def test_apply_with_other_expert_count(self, mock_fused_experts,
                                           mock_select_experts):
        # Setup
        mock_layer = MagicMock()
        x = torch.randn(32, self.hidden_size)
        router_logits = torch.randn(32, 128)  # 128 experts
        top_k = 2

        # Mock return values
        mock_select_experts.return_value = (torch.randn(32, top_k),
                                            torch.randint(0, 128, (32, top_k)))
        mock_fused_experts.return_value = torch.randn(32, self.hidden_size)

        # Test
        result = self.moe_method.apply(layer=mock_layer,
                                       x=x,
                                       router_logits=router_logits,
                                       top_k=top_k,
                                       renormalize=True,
                                       global_num_experts=128)

        # Assertions
        mock_select_experts.assert_called_once()
        mock_fused_experts.assert_called_once()
        self.assertEqual(result.shape, (32, self.hidden_size))

    @patch("vllm_ascend.quantization.w8a8.is_310p", return_value=True)
    @patch('vllm_ascend.quantization.w8a8.select_experts')
    @patch('vllm_ascend.quantization.w8a8.fused_experts_310p')
    def test_apply_is_310p(self, mock_fused_experts_310p, mock_select_experts,
                           mock_is_310p):
        # Setup
        mock_layer = MagicMock()
        x = torch.randn(32, self.hidden_size)
        router_logits = torch.randn(32, 128)  # 128 experts
        top_k = 2

        # Mock return values
        mock_select_experts.return_value = (torch.randn(32, top_k),
                                            torch.randint(0, 128, (32, top_k)))
        mock_fused_experts_310p.return_value = torch.randn(
            32, self.hidden_size)

        # Test
        result = self.moe_method.apply(layer=mock_layer,
                                       x=x,
                                       router_logits=router_logits,
                                       top_k=top_k,
                                       renormalize=True,
                                       global_num_experts=128)

        # Assertions
        mock_select_experts.assert_called_once()
        mock_fused_experts_310p.assert_called_once()
        self.assertEqual(result.shape, (32, self.hidden_size))


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
        """测试 create_weights 是否正确注册参数"""
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

    @patch("vllm_ascend.quantization.w8a8.is_310p", return_value=False)
    def test_process_weights_after_loading_not_310p(self, mock_is_310p):
        key_data = torch.ones(4 * 64)
        value_data = torch.ones(4 * 64) * 2

        self.layer.key_antiquant_scale.data = key_data
        self.layer.value_antiquant_scale.data = value_data

        self.method.process_weights_after_loading(self.layer)

        self.assertEqual(self.method.antiquant_scale_comb.shape, (2, 256))
        self.assertTrue(torch.all(self.method.antiquant_scale_comb[0] == 1))
        self.assertTrue(torch.all(self.method.antiquant_scale_comb[1] == 2))

    @patch("vllm_ascend.quantization.w8a8.is_310p", return_value=True)
    def test_process_weights_after_loading_is_310p(self, mock_is_310p):
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


class TestFusedExperts(TestBase):

    @patch("vllm_ascend.quantization.w8a8.quant_per_tensor")
    @patch('vllm_ascend.quantization.w8a8.get_ep_group')
    @patch('torch_npu.npu_moe_init_routing_v2')
    @patch('torch_npu.npu_grouped_matmul')
    @patch('torch_npu.npu_swiglu')
    @patch('torch_npu.npu_moe_finalize_routing')
    def test_fused_experts_with_expert_map(self, mock_finalize, mock_swiglu,
                                           mock_group_matmul,
                                           mock_init_routing,
                                           mock_get_ep_group,
                                           mock_quant_per_tensor):
        num_tokens = 32
        hidden_size = 128
        intermediate_size = 256
        num_experts = 4
        top_k = 2

        hidden_states = torch.randn(num_tokens, hidden_size)

        w1 = torch.randn(num_experts, intermediate_size * 2, hidden_size)
        w1_scale = torch.tensor([0.1])
        w1_input_scale = torch.tensor([[0.2, 0.2], [0.2, 0.2]])
        w1_input_offset = torch.tensor([0])

        w2 = torch.randn(num_experts, hidden_size, intermediate_size)
        w2_scale = torch.tensor([0.1])
        w2_input_scale = torch.tensor([0.2])
        w2_input_offset = torch.tensor([0])

        topk_weights = torch.rand(num_tokens, top_k)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k))
        expert_map = torch.arange(num_experts)

        mock_get_ep_group.return_value.world_size = 8

        mock_quant_per_tensor.return_value = torch.randint(-128,
                                                           127,
                                                           hidden_states.shape,
                                                           dtype=torch.int8)

        mock_init_routing.return_value = (torch.randn(num_tokens * top_k,
                                                      hidden_size),
                                          torch.arange(num_tokens * top_k),
                                          torch.tensor([num_tokens // 2] * 2),
                                          torch.tensor(1.0))

        mock_group_matmul.side_effect = [[
            torch.randn(num_tokens * top_k, intermediate_size * 2)
        ], [torch.randn(num_tokens * top_k, hidden_size)]]

        mock_swiglu.return_value = torch.randn(num_tokens * top_k,
                                               intermediate_size)

        expected_output = torch.randn(num_tokens, hidden_size)
        mock_finalize.return_value = expected_output

        output = fused_experts(
            hidden_states=hidden_states,
            w1=w1,
            w1_scale=w1_scale,
            w1_input_scale=w1_input_scale,
            w1_input_offset=w1_input_offset,
            w2=w2,
            w2_scale=w2_scale,
            w2_input_scale=w2_input_scale,
            w2_input_offset=w2_input_offset,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=top_k,
            global_num_experts=num_experts,
            expert_map=expert_map,
        )

        mock_init_routing.assert_called_once()

        self.assertEqual(mock_group_matmul.call_count, 2)

        self.assertEqual(output.shape, (num_tokens, hidden_size))

        mock_finalize.assert_called_once()

    @patch("vllm_ascend.quantization.w8a8.quant_per_tensor")
    @patch('vllm_ascend.quantization.w8a8.get_ep_group')
    @patch('torch_npu.npu_grouped_matmul')
    @patch('torch_npu.npu_swiglu')
    def test_fused_experts_without_expert_map(self, mock_swiglu,
                                              mock_group_matmul,
                                              mock_get_ep_group,
                                              mock_quant_per_tensor):
        num_tokens = 16
        hidden_size = 64
        intermediate_size = 128
        num_experts = 8
        top_k = 1

        hidden_states = torch.randn(num_tokens, hidden_size)
        w1 = torch.randn(num_experts, intermediate_size * 2, hidden_size)
        w2 = torch.randn(num_experts, hidden_size, intermediate_size)
        topk_weights = torch.rand(num_tokens, top_k)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k))

        mock_get_ep_group.return_value.world_size = 8

        mock_quant_per_tensor.return_value = torch.randint(-128,
                                                           127,
                                                           hidden_states.shape,
                                                           dtype=torch.int8)
        mock_group_matmul.side_effect = [[
            torch.randn(num_tokens * top_k, intermediate_size * 2)
        ], [torch.randn(num_tokens * top_k, hidden_size)]]
        mock_swiglu.return_value = torch.randn(num_tokens * top_k,
                                               intermediate_size)
        with self.assertRaises(NotImplementedError):
            fused_experts(
                hidden_states=hidden_states,
                w1=w1,
                w1_scale=torch.tensor([0.1]),
                w1_input_scale=torch.tensor([[0.2, 0.2], [0.2, 0.2]]),
                w1_input_offset=torch.tensor([0]),
                w2=w2,
                w2_scale=torch.tensor([0.1]),
                w2_input_scale=torch.tensor([0.1]),
                w2_input_offset=torch.tensor([0]),
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                top_k=top_k,
                global_num_experts=num_experts,
                expert_map=None,
            )


class TestFusedExperts310(TestBase):

    @patch('torch_npu.npu_quant_grouped_matmul_dequant')
    @patch("vllm_ascend.quantization.w8a8.quant_per_tensor")
    @patch('vllm_ascend.quantization.w8a8.get_ep_group')
    @patch('torch_npu.npu_swiglu')
    def test_fused_experts_310p_with_expert_map(self, mock_swiglu,
                                                mock_get_ep_group,
                                                mock_quant_per_tensor,
                                                mock_matmul_dequant):
        num_tokens = 32
        hidden_size = 128
        intermediate_size = 256
        num_experts = 4
        top_k = 1

        hidden_states = torch.randn(num_tokens, hidden_size)

        w1 = torch.randn(num_experts, intermediate_size * 2, hidden_size)
        w1_scale = torch.tensor([0.1])
        w1_input_scale = torch.tensor([[0.2, 0.2], [0.2, 0.2]])

        w2 = torch.randn(num_experts, hidden_size, intermediate_size)
        w2_scale = torch.tensor([0.1])
        w2_input_scale = torch.tensor([0.2])

        topk_weights = torch.rand(num_tokens, top_k)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k))
        expert_map = torch.arange(num_experts)

        mock_get_ep_group.return_value.world_size = 1

        mock_quant_per_tensor.return_value = torch.randint(-128,
                                                           127,
                                                           hidden_states.shape,
                                                           dtype=torch.int8)

        mock_swiglu.return_value = torch.randn(num_tokens * top_k,
                                               intermediate_size)

        mock_matmul_dequant.return_value = hidden_states

        output = fused_experts_310p(
            hidden_states=hidden_states,
            w1=w1,
            w1_scale=w1_scale,
            w1_input_scale=w1_input_scale,
            w2=w2,
            w2_scale=w2_scale,
            w2_input_scale=w2_input_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=top_k,
            global_num_experts=num_experts,
            expert_map=expert_map,
        )

        self.assertEqual(output.shape, (num_tokens, hidden_size))
        self.assertEqual(mock_matmul_dequant.call_count, 2)


class TestSelectExperts(TestBase):

    def setUp(self):
        # Common test data
        self.num_tokens = 10
        self.hidden_size = 32
        self.num_experts = 8
        self.top_k = 2

        self.hidden_states = torch.randn(self.num_tokens, self.hidden_size)
        self.router_logits = torch.randn(self.num_tokens, self.num_experts)

    @patch('torch_npu.npu_moe_gating_top_k_softmax')
    def test_softmax_scoring(self, mock_topk):
        """Test softmax scoring function"""
        mock_topk.return_value = (torch.ones(self.num_tokens, self.top_k),
                                  torch.zeros(self.num_tokens,
                                              self.top_k,
                                              dtype=torch.long),
                                  torch.arange(0,
                                               self.num_tokens * self.top_k,
                                               dtype=torch.int32).view(
                                                   self.top_k,
                                                   -1).permute(1,
                                                               0).contiguous())

        weights, ids, _ = select_experts(hidden_states=self.hidden_states,
                                         router_logits=self.router_logits,
                                         top_k=self.top_k,
                                         use_grouped_topk=False,
                                         renormalize=False,
                                         scoring_func="softmax")

        self.assertEqual(weights.shape, (self.num_tokens, self.top_k))
        self.assertEqual(ids.shape, (self.num_tokens, self.top_k))

    def test_sigmoid_scoring(self):
        """Test sigmoid scoring function"""

        weights, ids, _ = select_experts(hidden_states=self.hidden_states,
                                         router_logits=self.router_logits,
                                         top_k=self.top_k,
                                         use_grouped_topk=False,
                                         renormalize=False,
                                         scoring_func="sigmoid")

        self.assertEqual(weights.shape, (self.num_tokens, self.top_k))
        self.assertEqual(ids.shape, (self.num_tokens, self.top_k))

    def test_invalid_scoring_func(self):
        """Test invalid scoring function raises ValueError"""
        with self.assertRaises(ValueError):
            select_experts(hidden_states=self.hidden_states,
                           router_logits=self.router_logits,
                           top_k=self.top_k,
                           use_grouped_topk=False,
                           renormalize=False,
                           scoring_func="invalid_func")

    @patch('torch.topk')
    def test_grouped_topk(self, mock_topk):
        """Test grouped topk functionality"""
        mock_topk.return_value = (torch.ones(self.num_tokens, self.top_k),
                                  torch.zeros(self.num_tokens,
                                              self.top_k,
                                              dtype=torch.long))

        weights, ids, _ = select_experts(hidden_states=self.hidden_states,
                                         router_logits=self.router_logits,
                                         top_k=self.top_k,
                                         use_grouped_topk=True,
                                         renormalize=False,
                                         topk_group=4,
                                         num_expert_group=2)

        mock_topk.assert_called()
        self.assertEqual(weights.shape, (self.num_tokens, self.top_k))
        self.assertEqual(ids.shape, (self.num_tokens, self.top_k))
        self.assertEqual(ids.dtype, torch.int32)

    @patch('vllm_ascend.ops.moe.experts_selector._native_grouped_topk')
    def test_grouped_topk_with_correction_bias(self, mock_grouped_topk):
        """Test grouped topk with expert score correction bias"""
        mock_grouped_topk.return_value = torch.ones(self.num_tokens,
                                                    self.num_experts)

        e_score_correction_bias = torch.randn(self.num_experts)
        weights, ids, _ = select_experts(
            hidden_states=self.hidden_states,
            router_logits=self.router_logits,
            top_k=self.top_k,
            use_grouped_topk=True,
            renormalize=False,
            topk_group=4,
            num_expert_group=2,
            e_score_correction_bias=e_score_correction_bias)

        mock_grouped_topk.assert_called_once()
        self.assertEqual(weights.shape, (self.num_tokens, self.top_k))
        self.assertEqual(ids.shape, (self.num_tokens, self.top_k))

    def test_custom_routing_function(self):
        """Test custom routing function"""
        mock_custom_routing = MagicMock()
        mock_custom_routing.return_value = (torch.ones(self.num_tokens,
                                                       self.top_k),
                                            torch.zeros(self.num_tokens,
                                                        self.top_k,
                                                        dtype=torch.int32))

        weights, ids, _ = select_experts(
            hidden_states=self.hidden_states,
            router_logits=self.router_logits,
            top_k=self.top_k,
            use_grouped_topk=False,
            renormalize=False,
            custom_routing_function=mock_custom_routing)

        mock_custom_routing.assert_called_once()
        self.assertEqual(weights.shape, (self.num_tokens, self.top_k))
        self.assertEqual(ids.shape, (self.num_tokens, self.top_k))
        self.assertEqual(ids.dtype, torch.int32)

    @patch('torch_npu.npu_moe_gating_top_k_softmax')
    def test_renormalize(self, mock_topk):
        """Test renormalization"""
        mock_topk.return_value = (torch.ones(self.num_tokens, self.top_k),
                                  torch.zeros(self.num_tokens,
                                              self.top_k,
                                              dtype=torch.long),
                                  torch.arange(0,
                                               self.num_tokens * self.top_k,
                                               dtype=torch.int32).view(
                                                   self.top_k,
                                                   -1).permute(1,
                                                               0).contiguous())

        weights, ids, _ = select_experts(
            hidden_states=self.hidden_states,
            router_logits=self.router_logits,
            top_k=self.top_k,
            use_grouped_topk=False,
            renormalize=True,
        )

        # Check if weights are normalized (sum to 1 for each token)
        sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums)))

    @patch('torch_npu.npu_moe_gating_top_k_softmax')
    def test_output_dtypes(self, mock_topk):
        """Test output dtypes"""
        mock_topk.return_value = (torch.ones(self.num_tokens, self.top_k),
                                  torch.zeros(self.num_tokens,
                                              self.top_k,
                                              dtype=torch.long),
                                  torch.arange(0,
                                               self.num_tokens * self.top_k,
                                               dtype=torch.int32).view(
                                                   self.top_k,
                                                   -1).permute(1,
                                                               0).contiguous())

        weights, ids, _ = select_experts(
            hidden_states=self.hidden_states,
            router_logits=self.router_logits,
            top_k=self.top_k,
            use_grouped_topk=False,
            renormalize=False,
        )

        self.assertEqual(weights.dtype, self.hidden_states.dtype)
        self.assertEqual(ids.dtype, torch.int32)


class TestNativeGroupedTopkPartialMock(TestBase):

    def test_basic_group_selection(self):
        topk_weights = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
                                     [0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1],
                                     [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                                     [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]],
                                    dtype=torch.float32)

        expected_topk_indices = torch.tensor([[0, 1], [1, 0], [0, 1], [0, 1]])

        with patch('torch.topk',
                   return_value=(None, expected_topk_indices)) as mock_topk:
            result = _native_grouped_topk(topk_weights=topk_weights,
                                          num_expert_group=2,
                                          topk_group=2)

            mock_topk.assert_called_once()

            expected_result = topk_weights
            self.assertTrue(torch.allclose(result, expected_result))

    def test_partial_group_selection(self):

        topk_weights = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
                                     [0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1]])

        expected_topk_indices = torch.tensor([[0], [1]])

        with patch('torch.topk', return_value=(None, expected_topk_indices)):
            result = _native_grouped_topk(topk_weights=topk_weights,
                                          num_expert_group=2,
                                          topk_group=1)

            expected_result = torch.tensor(
                [[0.1, 0.9, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.9, 0.1]])
            self.assertTrue(torch.allclose(result, expected_result))

    def test_single_group(self):
        topk_weights = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.3, 0.7]])

        expected_topk_indices = torch.tensor([[0], [0]])

        with patch('torch.topk', return_value=(None, expected_topk_indices)):
            result = _native_grouped_topk(topk_weights=topk_weights,
                                          num_expert_group=1,
                                          topk_group=1)
            self.assertTrue(result.numel() > 0)
