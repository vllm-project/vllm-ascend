from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm_ascend.quantization.method_adapters import (
    AscendLinearMethod,
    AscendKVCacheMethod,
    AscendFusedMoEMethod, AscendEmbeddingMethod,
)
from vllm_ascend.quantization.methods.base import AscendLinearScheme, AscendAttentionScheme, AscendMoEScheme


class TestAscendLinearMethod(TestBase):

    def setUp(self):
        self.mock_scheme = MagicMock(spec=AscendLinearScheme)
        self.mock_scheme.get_weight.return_value = {
            "weight": torch.empty(128, 256, dtype=torch.int8),
            "_packed_dim": 0,
            "_packed_factor": 0.1,
        }
        self.mock_scheme.get_pertensor_param.return_value = {
            "weight_scale_pertensor": torch.empty(1, 1, dtype=torch.int8),
        }
        self.mock_scheme.get_perchannel_param.return_value = {
            "weight_scale_perchannel": torch.empty(128, 1, dtype=torch.int8),
        }
        self.mock_scheme.get_pergroup_param.return_value = {
            "weight_scale_second": torch.empty(128, 2, dtype=torch.int8),
            "weight_offset_second": torch.empty(128, 2, dtype=torch.int8),
            "weight_scale_pergroup": torch.empty(128, 2, dtype=torch.int8),
        }
        self.method = AscendLinearMethod(self.mock_scheme)

    @patch("vllm_ascend.quantization.method_adapters.PerTensorScaleParameter")
    def test_create_weights(self, mock_parameter):
        mock_parameter.return_value = torch.nn.Parameter(torch.empty(1, 1, dtype=torch.int8), requires_grad=False)
        layer = torch.nn.Module()
        weight_loader = MagicMock()
        self.method.create_weights(
            layer,
            input_size_per_partition=256,
            output_partition_sizes=[128],
            input_size=256,
            output_size=128,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )
        # Check get_weight method
        self.mock_scheme.get_weight.assert_called_once_with(256, 128, torch.bfloat16)
        self.assertIn("weight", dict(layer.named_parameters()))
        self.assertNotIn("_packed_dim", dict(layer.named_parameters()))
        self.assertNotIn("_packed_factor", dict(layer.named_parameters()))
        self.assertEqual(layer.weight.input_dim, 1)
        self.assertEqual(layer.weight.output_dim, 0)
        self.assertEqual(layer.weight.packed_dim, 0)
        self.assertEqual(layer.weight.packed_factor, 0.1)

        # Check per tensor param
        self.mock_scheme.get_pertensor_param.assert_called_once_with(torch.bfloat16)
        self.assertTrue(layer.weight_scale_pertensor.ignore_warning)
        self.assertEqual(layer.weight_scale_pertensor.weight_loader, weight_loader)

        # Check per channel param
        self.mock_scheme.get_perchannel_param.assert_called_once_with(128, torch.bfloat16)
        self.assertEqual(layer.weight_scale_perchannel.output_dim, 0)
        self.assertEqual(layer.weight_scale_perchannel.weight_loader, weight_loader)

        # Check per group param
        self.mock_scheme.get_pergroup_param.assert_called_once()
        self.assertEqual(layer.weight_scale_pergroup.output_dim, 0)
        self.assertFalse(hasattr(layer.weight_scale_pergroup, "input_dim"))
        self.assertEqual(layer.weight_scale_second.input_dim, 1)
        self.assertEqual(layer.weight_offset_second.input_dim, 1)

    def test_process_weights_after_loading_delegates(self):
        layer = torch.nn.Module()
        self.mock_scheme.process_weights_after_loading.return_value = None
        self.method.process_weights_after_loading(layer)
        self.mock_scheme.process_weights_after_loading.assert_called_once_with(layer)

    def test_process_weights_after_loading_skips_when_not_present(self):
        del self.mock_scheme.process_weights_after_loading
        layer = torch.nn.Module()
        self.method.process_weights_after_loading(layer)

    def test_get_computed_params(self):
        computed = self.method.get_computed_params()
        self.assertIn("weight_offset", computed)
        self.assertIn("quant_bias", computed)
        self.assertIn("deq_scale", computed)
        self.assertIn("weight_scale", computed)
        self.assertEqual(len(computed), 4)

    def test_apply_delegates_to_scheme(self):
        layer = MagicMock(spec=ColumnParallelLinear)
        x = torch.randn(4, 256)
        self.mock_scheme.apply.return_value = torch.randn(4, 128)
        output = self.method.apply(layer, x)
        self.mock_scheme.apply.assert_called_once()
        self.assertEqual(output.shape, (4, 128))


class TestAscendKVCacheMethod(TestBase):

    def setUp(self):
        self.mock_scheme = MagicMock(spec=AscendAttentionScheme)
        self.mock_scheme.create_weights.return_value = None
        self.mock_scheme.process_weights_after_loading.return_value = None
        self.method = AscendKVCacheMethod(self.mock_scheme)

    def test_create_weights_delegates(self):
        layer = torch.nn.Module()
        self.method.create_weights(layer)
        self.mock_scheme.create_weights.assert_called_once_with(layer)

    def test_process_weights_after_loading_delegates(self):
        layer = torch.nn.Module()
        self.method.process_weights_after_loading(layer)
        self.mock_scheme.process_weights_after_loading.assert_called_once_with(layer)

    def test_apply_delegates(self):
        layer = torch.nn.Module()
        query = torch.randn(4, 8, 64)
        key = torch.randn(4, 8, 64)
        value = torch.randn(4, 8, 64)
        self.mock_scheme.apply.return_value = torch.randn(4, 8, 64)
        self.method.apply(
            layer, query, key, value,
            kv_cache=None, attn_metadata=None,
            attn_type=None, scale=1.0, output=None,
        )
        self.mock_scheme.apply.assert_called_once()


class TestAscendFusedMoEMethod(TestBase):

    def setUp(self):
        self.mock_scheme = MagicMock(spec=AscendMoEScheme)
        self.mock_scheme.group_size = 0
        self.mock_moe_config = MagicMock()
        self.method = AscendFusedMoEMethod(self.mock_scheme, self.mock_moe_config)

    def test_process_weights_after_loading_delegates(self):
        layer = torch.nn.Module()
        self.mock_scheme.process_weights_after_loading.return_value = None
        self.method.process_weights_after_loading(layer)
        self.mock_scheme.process_weights_after_loading.assert_called_once_with(layer)

    def test_process_weights_after_loading_skips_when_not_present(self):
        del self.mock_scheme.process_weights_after_loading
        layer = torch.nn.Module()
        self.method.process_weights_after_loading(layer)

    def test_create_weights_registers_parameters(self):
        self.mock_scheme.get_weight.return_value = {
            "w13_weight": torch.empty(8, 256, 128, dtype=torch.int8),
            "w2_weight": torch.empty(8, 128, 256, dtype=torch.int8),
        }
        self.mock_scheme.get_dynamic_quant_param.return_value = {
            "w13_weight_scale_second": torch.empty(8, 256, 1, dtype=torch.bfloat16),
            "w2_weight_offset_second": torch.empty(8, 128, 1, dtype=torch.bfloat16),
            "w2_scale_bias": torch.empty(8, 128, 1, dtype=torch.bfloat16),
            "w13_weight_scale": torch.empty(8, 256, 1, dtype=torch.bfloat16),
            "w2_weight_offset": torch.empty(8, 128, 1, dtype=torch.bfloat16),
        }
        # per channel quantization
        layer = torch.nn.Module()
        self.method.create_weights(
            layer,
            num_experts=8,
            hidden_size=128,
            intermediate_size_per_partition=256,
            params_dtype=torch.bfloat16,
        )
        self.assertIn("w13_weight", dict(layer.named_parameters()))
        self.assertIn("w2_weight", dict(layer.named_parameters()))

        self.assertEqual(layer.w13_weight_scale_second.quant_method, FusedMoeWeightScaleSupported.GROUP.value)
        self.assertEqual(layer.w2_weight_offset_second.quant_method, FusedMoeWeightScaleSupported.GROUP.value)
        self.assertEqual(layer.w2_scale_bias.quant_method, FusedMoeWeightScaleSupported.GROUP.value)
        self.assertEqual(layer.w13_weight_scale.quant_method, FusedMoeWeightScaleSupported.CHANNEL.value)
        self.assertEqual(layer.w2_weight_offset.quant_method, FusedMoeWeightScaleSupported.CHANNEL.value)

        # per group quantization
        self.mock_scheme.group_size = 128
        layer = torch.nn.Module()
        self.method.create_weights(
            layer,
            num_experts=8,
            hidden_size=128,
            intermediate_size_per_partition=256,
            params_dtype=torch.bfloat16,
        )
        self.assertEqual(layer.w13_weight_scale_second.quant_method, FusedMoeWeightScaleSupported.GROUP.value)
        self.assertEqual(layer.w2_weight_offset_second.quant_method, FusedMoeWeightScaleSupported.GROUP.value)
        self.assertEqual(layer.w2_scale_bias.quant_method, FusedMoeWeightScaleSupported.GROUP.value)
        self.assertEqual(layer.w13_weight_scale.quant_method, FusedMoeWeightScaleSupported.GROUP.value)
        self.assertEqual(layer.w2_weight_offset.quant_method, FusedMoeWeightScaleSupported.GROUP.value)



    def test_apply_method(self):
        layer = torch.nn.Module()
        x = torch.randn(8, 64)
        router_logits = torch.randn(8, 64)
        top_k = 3
        renormalize = True
        self.mock_scheme.apply.return_value = None
        self.method.apply(
            layer, x, router_logits, top_k, renormalize
        )
        self.mock_scheme.apply.assert_called_once()

    def test_supports_eplb_default_false(self):
        self.assertFalse(self.method.supports_eplb)

    def test_supports_eplb_when_scheme_supports(self):
        self.mock_scheme.supports_eplb = True
        self.assertTrue(self.method.supports_eplb)


class TestAscendEmbeddingMethod(TestBase):

    def test_init(self):
        layer = MagicMock(spec=AscendLinearScheme)
        method = AscendEmbeddingMethod(layer)
        self.assertTrue(isinstance(method, AscendLinearMethod))
        self.assertEqual(method.quant_method, layer)
