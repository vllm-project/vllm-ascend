from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.utils.math_utils import cdiv

from tests.ut.base import TestBase
from vllm_ascend.quantization.method_adapters import (
    AscendFusedMoEMethod,
    AscendKVCacheMethod,
    AscendLinearMethod,
)
from vllm_ascend.quantization.methods import AscendW8A8MXFP8DynamicLinearMethod
from vllm_ascend.quantization.methods.base import AscendAttentionScheme, AscendLinearScheme, AscendMoEScheme


class TestAscendLinearMethod(TestBase):
    @patch("vllm_ascend.quantization.method_adapters.enable_dsa_cp_with_layer_shard")
    def setUp(self, mock_enable_dsa_cp_with_layer_shard):
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
        self.mock_scheme.get_pertensor_param.assert_called_once()
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
            layer,
            query,
            key,
            value,
            kv_cache=None,
            attn_metadata=None,
            attn_type=None,
            scale=1.0,
            output=None,
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
        layer = self.create_moe_weights()
        self.assertIn("w13_weight", dict(layer.named_parameters()))
        self.assertIn("w2_weight", dict(layer.named_parameters()))

        self.assertEqual(layer.w13_weight_scale_second.quant_method, FusedMoeWeightScaleSupported.GROUP.value)
        self.assertEqual(layer.w2_weight_offset_second.quant_method, FusedMoeWeightScaleSupported.GROUP.value)
        self.assertEqual(layer.w2_scale_bias.quant_method, FusedMoeWeightScaleSupported.GROUP.value)
        self.assertEqual(layer.w13_weight_scale.quant_method, FusedMoeWeightScaleSupported.CHANNEL.value)
        self.assertEqual(layer.w2_weight_offset.quant_method, FusedMoeWeightScaleSupported.CHANNEL.value)

        # per group quantization
        self.mock_scheme.group_size = 128
        layer = self.create_moe_weights()
        self.assertEqual(layer.w13_weight_scale.quant_method, FusedMoeWeightScaleSupported.GROUP.value)
        self.assertEqual(layer.w2_weight_offset.quant_method, FusedMoeWeightScaleSupported.GROUP.value)

    def create_moe_weights(self):
        layer = torch.nn.Module()
        self.method.create_weights(
            layer,
            num_experts=8,
            hidden_size=128,
            intermediate_size_per_partition=256,
            params_dtype=torch.bfloat16,
        )
        return layer

    def test_apply_method(self):
        layer = torch.nn.Module()
        x = torch.randn(8, 64)
        router_logits = torch.randn(8, 64)
        top_k = 3
        renormalize = True
        self.mock_scheme.apply.return_value = None
        self.method.apply(layer, x, router_logits, top_k, renormalize)
        self.mock_scheme.apply.assert_called_once()


def _make_layer(base_cls, **attrs):
    """Build a Row/ColumnParallelLinear without running its heavy __init__.

    ``isinstance`` still holds (real base class) and ``register_parameter`` works
    (Module is initialized), while only the attributes create_weights reads/writes
    are injected.
    """
    layer = base_cls.__new__(base_cls)
    torch.nn.Module.__init__(layer)
    for k, v in attrs.items():
        setattr(layer, k, v)
    return layer


class TestAscendLinearMethodGroupAlign(TestBase):
    """MXFP8 group-aligned TP sharding in AscendLinearMethod.create_weights.

    Uses intermediate_size=4304 (Qwen3-VL ViT MLP), which is not a multiple of
    group_size(32) * tp_size(2): the group-aligned split is [2176, 2128] elements /
    [68, 67] groups.
    """

    GROUP = 32

    def _scheme(self, spec):
        scheme = MagicMock(spec=spec)
        scheme.group_size = self.GROUP
        scheme.get_weight.side_effect = lambda in_p, out_p, dt: {"weight": torch.empty(out_p, in_p, dtype=torch.int8)}
        scheme.get_pertensor_param.return_value = {}
        scheme.get_perchannel_param.return_value = {}
        scheme.get_pergroup_param.side_effect = lambda in_p, out_p, dt, layer_type=None: {
            "weight_scale": torch.empty(out_p, cdiv(in_p, self.GROUP), dtype=torch.uint8)
        }
        return AscendLinearMethod(scheme)

    def _mxfp8(self):
        return self._scheme(AscendW8A8MXFP8DynamicLinearMethod)

    def test_row_uneven_partition(self):
        # MXFP8 RowParallel splits the grouped input dim on group boundaries.
        for rank, exp_elem, exp_groups in [(0, 2176, 68), (1, 2128, 67)]:
            method = self._mxfp8()
            weight_loader = MagicMock()
            layer = _make_layer(RowParallelLinear, tp_size=2, tp_rank=rank, prefix="fc2")
            method.create_weights(
                layer,
                input_size_per_partition=2152,  # vLLM's even split, overridden below
                output_partition_sizes=[512],
                input_size=4304,
                output_size=512,
                params_dtype=torch.bfloat16,
                weight_loader=weight_loader,
            )
            self.assertEqual(layer.input_size_per_partition, exp_elem)
            self.assertEqual(tuple(layer.weight.shape), (512, exp_elem))
            self.assertEqual(tuple(layer.weight_scale.shape), (512, exp_groups))
            self.assertIsNot(layer.weight.weight_loader, weight_loader)
            self.assertIsNot(layer.weight_scale.weight_loader, weight_loader)
            # full checkpoint narrowed on the aligned boundaries
            layer.weight.weight_loader(layer.weight, torch.zeros(512, 4304, dtype=torch.int8))
            layer.weight_scale.weight_loader(layer.weight_scale, torch.zeros(512, 135, dtype=torch.uint8))
            self.assertEqual(tuple(layer.weight.shape), (512, exp_elem))
            self.assertEqual(tuple(layer.weight_scale.shape), (512, exp_groups))

    def test_column_uneven_partition(self):
        # MXFP8 single-shard ColumnParallel splits the output dim on the same
        # boundaries so fc1 output shards match the paired fc2 input shards.
        for rank, exp_elem in [(0, 2176), (1, 2128)]:
            method = self._mxfp8()
            weight_loader = MagicMock()
            layer = _make_layer(ColumnParallelLinear, tp_size=2, tp_rank=rank, gather_output=False, prefix="fc1")
            method.create_weights(
                layer,
                input_size_per_partition=512,
                output_partition_sizes=[2152],
                input_size=512,
                output_size=4304,
                params_dtype=torch.bfloat16,
                weight_loader=weight_loader,
            )
            self.assertEqual(layer.output_size_per_partition, exp_elem)
            self.assertEqual(layer.output_partition_sizes, [exp_elem])
            self.assertEqual(tuple(layer.weight.shape), (exp_elem, 512))
            self.assertEqual(tuple(layer.weight_scale.shape), (exp_elem, cdiv(512, self.GROUP)))
            self.assertIsNot(layer.weight.weight_loader, weight_loader)
            full_scale = torch.zeros(4304, cdiv(512, self.GROUP), dtype=torch.uint8)
            layer.weight.weight_loader(layer.weight, torch.zeros(4304, 512, dtype=torch.int8))
            layer.weight_scale.weight_loader(layer.weight_scale, full_scale)
            self.assertEqual(tuple(layer.weight.shape), (exp_elem, 512))

    def test_divisible_no_align(self):
        # 4096 % (32*2) == 0 -> no override, native loader kept.
        method = self._mxfp8()
        weight_loader = MagicMock()
        layer = _make_layer(RowParallelLinear, tp_size=2, tp_rank=0, prefix="x")
        method.create_weights(
            layer,
            input_size_per_partition=2048,
            output_partition_sizes=[512],
            input_size=4096,
            output_size=512,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )
        self.assertIs(layer.weight.weight_loader, weight_loader)
        self.assertIs(layer.weight_scale.weight_loader, weight_loader)

    def test_non_mxfp8_no_align(self):
        # Non-MXFP8 scheme never group-aligns even on an unaligned size.
        method = self._scheme(AscendLinearScheme)
        weight_loader = MagicMock()
        layer = _make_layer(RowParallelLinear, tp_size=2, tp_rank=0, prefix="x")
        method.create_weights(
            layer,
            input_size_per_partition=2152,
            output_partition_sizes=[512],
            input_size=4304,
            output_size=512,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )
        self.assertIs(layer.weight.weight_loader, weight_loader)

    def test_tp1_no_align(self):
        method = self._mxfp8()
        weight_loader = MagicMock()
        layer = _make_layer(RowParallelLinear, tp_size=1, tp_rank=0, prefix="x")
        method.create_weights(
            layer,
            input_size_per_partition=4304,
            output_partition_sizes=[512],
            input_size=4304,
            output_size=512,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )
        self.assertIs(layer.weight.weight_loader, weight_loader)

    def test_merged_column_no_align(self):
        # Multi-shard column (gate_up/qkv) is out of scope -> no override.
        method = self._mxfp8()
        weight_loader = MagicMock()
        layer = _make_layer(ColumnParallelLinear, tp_size=2, tp_rank=0, gather_output=False, prefix="gate_up")
        method.create_weights(
            layer,
            input_size_per_partition=512,
            output_partition_sizes=[2152, 2152],
            input_size=512,
            output_size=4304,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )
        self.assertIs(layer.weight.weight_loader, weight_loader)

    def test_gather_output_no_align(self):
        method = self._mxfp8()
        weight_loader = MagicMock()
        layer = _make_layer(ColumnParallelLinear, tp_size=2, tp_rank=0, gather_output=True, prefix="x")
        method.create_weights(
            layer,
            input_size_per_partition=512,
            output_partition_sizes=[2152],
            input_size=512,
            output_size=4304,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )
        self.assertIs(layer.weight.weight_loader, weight_loader)

    def test_column_bias_no_align(self):
        # Single-shard column with bias=True is out of scope: the bias is built
        # after create_weights with the native even split, so group-aligning the
        # weight would desync it. The native loader must be kept.
        method = self._mxfp8()
        weight_loader = MagicMock()
        layer = _make_layer(
            ColumnParallelLinear, tp_size=2, tp_rank=0, gather_output=False, has_bias=True, prefix="fc1"
        )
        method.create_weights(
            layer,
            input_size_per_partition=512,
            output_partition_sizes=[2152],
            input_size=512,
            output_size=4304,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )
        self.assertIs(layer.weight.weight_loader, weight_loader)

    def test_loader_fallback_on_presharded(self):
        # An already-sharded tensor (input dim != full) falls back to orig loader.
        method = self._mxfp8()
        weight_loader = MagicMock()
        layer = _make_layer(RowParallelLinear, tp_size=2, tp_rank=0, prefix="fc2")
        method.create_weights(
            layer,
            input_size_per_partition=2152,
            output_partition_sizes=[512],
            input_size=4304,
            output_size=512,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )
        presharded = torch.zeros(512, 2176, dtype=torch.int8)
        layer.weight.weight_loader(layer.weight, presharded)
        weight_loader.assert_called_once_with(layer.weight, presharded)
