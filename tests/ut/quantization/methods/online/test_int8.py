#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.model_executor.layers.linear import LinearBase, RowParallelLinear
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticChannelSym,
    kInt8StaticChannelSym,
)

from tests.ut.base import TestBase
from vllm_ascend.quantization.configs.online_config import (
    AscendOnlineQuantizationConfig,
    build_ascend_online_quant_config,
)
from vllm_ascend.quantization.methods.online.adapters import AscendOnlineInt8LinearMethod
from vllm_ascend.quantization.methods.online.int8 import (
    AscendOnlineInt8W8A8FusedMoEMethod,
    AscendOnlineInt8W8A8LinearMethod,
    _linear_reduce_group,
    _moe_reduce_group,
    _zero_expert_padding,
    quantize_last_dim_to_int8,
)

MODULE = "vllm_ascend.quantization.methods.online.int8"
CONFIG_MODULE = "vllm_ascend.quantization.configs.online_config"
PROCESSED_FLAG = "_already_called_process_weights_after_loading"


def _int8_moe_config(**kwargs):
    from vllm.config.quantization import QuantizationConfigArgs

    return AscendOnlineQuantizationConfig(QuantizationConfigArgs(moe=QuantSpec(weight=kInt8StaticChannelSym), **kwargs))


class _LinearLayer(LinearBase):
    """Minimal LinearBase subclass.

    vLLM resolves online targets from the layer *type*, so the tests need a real
    LinearBase subclass instead of a ``MagicMock(spec=LinearBase)``.
    """

    def __init__(self, prefix: str = "model.layers.0.mlp.down_proj") -> None:
        torch.nn.Module.__init__(self)
        self.prefix = prefix
        self.tp_size = 1
        self.input_size = 64
        self.input_size_per_partition = 64


class TestQuantizeLastDimToInt8(TestBase):
    def test_reconstruction_is_within_int8_resolution(self):
        torch.manual_seed(0)
        weight = torch.randn(16, 64, dtype=torch.bfloat16)

        qweight, scale = quantize_last_dim_to_int8(weight, scale_dtype=torch.float32)

        self.assertEqual(qweight.dtype, torch.int8)
        self.assertEqual(qweight.shape, weight.shape)
        self.assertEqual(scale.shape, (16, 1))

        dequant = qweight.float() * scale
        # Symmetric per-output-channel int8: the error is bounded by half a
        # quantization step, i.e. amax / 254 per row.
        row_amax = weight.float().abs().amax(dim=1, keepdim=True)
        max_error = (dequant - weight.float()).abs().amax(dim=1, keepdim=True)
        self.assertTrue(torch.all(max_error <= row_amax / 254 + 1e-6))

    def test_scale_is_amax_over_127(self):
        weight = torch.tensor([[1.0, -0.5, 0.25], [2.0, 0.0, -2.0]], dtype=torch.float32)

        qweight, scale = quantize_last_dim_to_int8(weight)

        self.assertTrue(torch.allclose(scale.flatten(), torch.tensor([1.0 / 127, 2.0 / 127])))
        self.assertTrue(torch.allclose(qweight[0].float(), torch.tensor([127.0, -64.0, 32.0]), atol=1))
        self.assertTrue(torch.allclose(qweight[1].float(), torch.tensor([127.0, 0.0, -127.0]), atol=1))

    def test_zero_row_stays_zero(self):
        weight = torch.zeros(2, 8, dtype=torch.bfloat16)
        weight[1] = 1.0

        qweight, scale = quantize_last_dim_to_int8(weight)

        self.assertTrue(torch.all(qweight[0] == 0))
        self.assertEqual(scale[0].item(), 0.0)
        self.assertGreater(scale[1].item(), 0.0)

    def test_scale_dtype_follows_request(self):
        weight = torch.randn(4, 8, dtype=torch.bfloat16)

        _, bf16_scale = quantize_last_dim_to_int8(weight, scale_dtype=torch.bfloat16)
        _, fp32_scale = quantize_last_dim_to_int8(weight, scale_dtype=torch.float32)

        self.assertEqual(bf16_scale.dtype, torch.bfloat16)
        self.assertEqual(fp32_scale.dtype, torch.float32)

    def test_expert_stack_keeps_leading_dims(self):
        weight = torch.randn(3, 5, 8, dtype=torch.bfloat16)

        qweight, scale = quantize_last_dim_to_int8(weight)

        self.assertEqual(qweight.shape, (3, 5, 8))
        self.assertEqual(scale.shape, (3, 5, 1))

    def test_rejects_already_quantized_weight(self):
        with self.assertRaises(NotImplementedError):
            quantize_last_dim_to_int8(torch.zeros(4, 8, dtype=torch.int8))

    def test_all_reduces_amax_over_the_given_group(self):
        weight = torch.ones(4, 8, dtype=torch.float32)
        group = MagicMock()
        group.world_size = 2

        with patch("torch.distributed.all_reduce") as all_reduce:
            _, scale = quantize_last_dim_to_int8(weight, reduce_group=group)

        all_reduce.assert_called_once()
        self.assertEqual(all_reduce.call_args.kwargs["op"], torch.distributed.ReduceOp.MAX)
        self.assertAlmostEqual(scale[0].item(), 1.0 / 127, places=7)


class TestOnlineInt8LinearMethod(TestBase):
    def _layer(self, weight=None):
        layer = SimpleNamespace(prefix="model.layers.0.mlp.down_proj", params_dtype=torch.bfloat16)
        layer.weight = torch.nn.Parameter(
            torch.randn(32, 64, dtype=torch.bfloat16) if weight is None else weight,
            requires_grad=False,
        )
        return layer

    def test_get_weight_is_meta_when_layerwise_enabled(self):
        scheme = AscendOnlineInt8W8A8LinearMethod()
        with patch(f"{MODULE}.ascend_envs.VLLM_ASCEND_ONLINE_QUANT_LAYERWISE", True):
            params = scheme.get_weight(64, 32, torch.bfloat16)

        self.assertEqual(list(params), ["weight"])
        self.assertTrue(params["weight"].is_meta)
        self.assertEqual(tuple(params["weight"].shape), (32, 64))
        self.assertEqual(params["weight"].dtype, torch.bfloat16)

    def test_get_weight_is_materialized_when_layerwise_disabled(self):
        scheme = AscendOnlineInt8W8A8LinearMethod()
        with patch(f"{MODULE}.ascend_envs.VLLM_ASCEND_ONLINE_QUANT_LAYERWISE", False):
            params = scheme.get_weight(64, 32, torch.bfloat16)

        self.assertFalse(params["weight"].is_meta)

    def test_no_scale_parameters_are_created_up_front(self):
        """Scale parameters must be absent while loading or the layerwise
        "is this layer fully loaded" accounting never completes."""
        scheme = AscendOnlineInt8W8A8LinearMethod()
        self.assertEqual(scheme.get_perchannel_param(32, torch.bfloat16), {})
        self.assertEqual(scheme.get_pertensor_param(torch.bfloat16), {})
        self.assertEqual(scheme.get_pergroup_param(64, 32, torch.bfloat16), {})

    def test_process_weights_installs_scale_and_delegates_the_layout(self):
        scheme = AscendOnlineInt8W8A8LinearMethod()
        layer = self._layer()

        scale_buffer = torch.empty(32, 1, dtype=torch.bfloat16)
        offset_buffer = torch.empty(32, 1, dtype=torch.bfloat16)
        with (
            patch(
                f"{MODULE}.AscendW8A8DynamicLinearMethod.get_perchannel_param",
                return_value={"weight_scale": scale_buffer, "weight_offset": offset_buffer},
            ),
            patch(f"{MODULE}.AscendW8A8DynamicLinearMethod.process_weights_after_loading") as layout,
            patch(f"{MODULE}._linear_reduce_group", return_value=None),
        ):
            scheme.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertEqual(tuple(layer.weight.shape), (32, 64))
        self.assertEqual(tuple(layer.weight_scale.shape), (32, 1))
        self.assertTrue(torch.all(layer.weight_offset == 0))
        layout.assert_called_once_with(scheme, layer)
        self.assertTrue(getattr(layer, PROCESSED_FLAG))

    def test_process_weights_is_idempotent(self):
        scheme = AscendOnlineInt8W8A8LinearMethod()
        layer = self._layer()
        setattr(layer, PROCESSED_FLAG, True)

        with patch(f"{MODULE}.quantize_last_dim_to_int8") as quantize:
            scheme.process_weights_after_loading(layer)

        quantize.assert_not_called()

    def test_process_weights_rejects_non_2d_weight(self):
        scheme = AscendOnlineInt8W8A8LinearMethod()
        layer = self._layer(weight=torch.randn(2, 8, 16, dtype=torch.bfloat16))

        with self.assertRaises(ValueError):
            scheme.process_weights_after_loading(layer)


class TestOnlineInt8MoeMethod(TestBase):
    def _scheme(self):
        # Bypass __init__: it probes the MC2 communication group, which does not
        # exist in a unit-test process.
        return AscendOnlineInt8W8A8FusedMoEMethod.__new__(AscendOnlineInt8W8A8FusedMoEMethod)

    def test_no_scale_parameters_are_created_up_front(self):
        self.assertEqual(self._scheme().get_dynamic_quant_param(2, 16, 32, torch.bfloat16), {})

    def test_get_weight_uses_the_fused_gate_up_layout(self):
        with patch(f"{MODULE}.ascend_envs.VLLM_ASCEND_ONLINE_QUANT_LAYERWISE", True):
            params = self._scheme().get_weight(4, 16, 32, torch.bfloat16)

        self.assertEqual(tuple(params["w13_weight"].shape), (4, 32, 32))
        self.assertEqual(tuple(params["w2_weight"].shape), (4, 32, 16))
        self.assertTrue(params["w13_weight"].is_meta)

    def test_process_weights_quantizes_both_projections_and_delegates_the_layout(self):
        scheme = self._scheme()
        layer = SimpleNamespace(
            prefix="model.layers.0.mlp.experts",
            moe_config=SimpleNamespace(tp_size=1, dp_size=1, pcp_size=1),
        )
        layer.w13_weight = torch.nn.Parameter(torch.randn(2, 32, 16, dtype=torch.bfloat16), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.randn(2, 16, 16, dtype=torch.bfloat16), requires_grad=False)

        buffers = {
            "w13_weight_scale": torch.empty(2, 32, 1, dtype=torch.bfloat16),
            "w13_weight_offset": torch.empty(2, 32, 1, dtype=torch.bfloat16),
            "w2_weight_scale": torch.empty(2, 16, 1, dtype=torch.bfloat16),
            "w2_weight_offset": torch.empty(2, 16, 1, dtype=torch.bfloat16),
        }
        with (
            patch(
                f"{MODULE}.AscendW8A8DynamicFusedMoEMethod.get_dynamic_quant_param",
                return_value=dict(buffers),
            ) as make_buffers,
            patch(f"{MODULE}.AscendW8A8DynamicFusedMoEMethod.process_weights_after_loading") as layout,
        ):
            scheme.process_weights_after_loading(layer)

        make_buffers.assert_called_once_with(scheme, 2, 16, 16, torch.bfloat16)
        self.assertEqual(layer.w13_weight.dtype, torch.int8)
        self.assertEqual(layer.w2_weight.dtype, torch.int8)
        self.assertEqual(tuple(layer.w13_weight.shape), (2, 32, 16))
        self.assertTrue(torch.all(layer.w13_weight_offset == 0))
        self.assertTrue(torch.all(layer.w2_weight_offset == 0))
        layout.assert_called_once_with(scheme, layer)

    def test_tensor_sharded_experts_spanning_dp_are_rejected(self):
        with self.assertRaises(NotImplementedError):
            _moe_reduce_group(SimpleNamespace(tp_size=8, dp_size=2, pcp_size=1))


class TestZeroExpertPadding(TestBase):
    def _layer(self, w2_contraction=8):
        return SimpleNamespace(
            w13_weight=torch.nn.Parameter(torch.ones(2, 8, 6, dtype=torch.bfloat16), requires_grad=False),
            w2_weight=torch.nn.Parameter(torch.ones(2, 6, w2_contraction, dtype=torch.bfloat16), requires_grad=False),
        )

    def test_padded_lanes_are_zeroed_before_quantization(self):
        layer = self._layer()
        moe_config = SimpleNamespace(
            hidden_dim_unpadded=4,
            intermediate_size_per_partition_unpadded=3,
            w13_num_shards=2,
        )

        _zero_expert_padding(moe_config, layer)

        # w13 holds two 4-row shards whose last row (3 and 7) is padding, and
        # columns 4..5 are the padded hidden dim.
        self.assertTrue(torch.all(layer.w13_weight[0, 3] == 0))
        self.assertTrue(torch.all(layer.w13_weight[0, 7] == 0))
        self.assertTrue(torch.all(layer.w13_weight[0, :, 4:] == 0))
        self.assertTrue(torch.all(layer.w13_weight[0, 0:3, 0:4] == 1))
        # w2 is padded on both the hidden rows and the intermediate columns.
        self.assertTrue(torch.all(layer.w2_weight[0, 4:, :] == 0))
        self.assertTrue(torch.all(layer.w2_weight[0, :, 3:] == 0))
        self.assertTrue(torch.all(layer.w2_weight[0, 0:4, 0:3] == 1))

    def test_unpadded_weights_are_left_alone(self):
        layer = self._layer(w2_contraction=4)
        moe_config = SimpleNamespace(
            hidden_dim_unpadded=6,
            intermediate_size_per_partition_unpadded=4,
            w13_num_shards=2,
        )

        _zero_expert_padding(moe_config, layer)

        self.assertTrue(torch.all(layer.w13_weight == 1))
        self.assertTrue(torch.all(layer.w2_weight == 1))

    def test_missing_unpadded_metadata_is_a_noop(self):
        layer = self._layer()

        _zero_expert_padding(SimpleNamespace(), layer)

        self.assertTrue(torch.all(layer.w13_weight == 1))
        self.assertTrue(torch.all(layer.w2_weight == 1))


class TestReduceGroupSelection(TestBase):
    def test_no_reduction_for_replicated_layers(self):
        layer = MagicMock(spec=LinearBase)
        layer.tp_size = 1
        layer.input_size = 64
        layer.input_size_per_partition = 64
        layer.prefix = "qkv_proj"
        self.assertIsNone(_linear_reduce_group(layer))

    def test_no_reduction_for_column_parallel_layers(self):
        layer = MagicMock(spec=LinearBase)
        layer.tp_size = 8
        layer.input_size = 64
        layer.input_size_per_partition = 64
        layer.prefix = "qkv_proj"
        self.assertIsNone(_linear_reduce_group(layer))

    def test_row_parallel_layer_reduces_over_the_tp_group(self):
        layer = MagicMock(spec=RowParallelLinear)
        layer.tp_size = 8
        layer.input_size = 64
        layer.input_size_per_partition = 8
        layer.prefix = "lm_head"
        tp_group = object()
        with (
            patch(f"{MODULE}.oproj_tp_enable", return_value=False),
            patch("vllm.distributed.get_tp_group", return_value=tp_group),
        ):
            self.assertIs(_linear_reduce_group(layer), tp_group)

    def test_reduction_follows_the_fine_grained_o_proj_group(self):
        layer = MagicMock(spec=RowParallelLinear)
        layer.tp_size = 8
        layer.input_size = 64
        layer.input_size_per_partition = 8
        layer.prefix = "model.layers.0.self_attn.o_proj"
        otp_group = object()
        with (
            patch(f"{MODULE}.oproj_tp_enable", return_value=True),
            patch("vllm_ascend.distributed.parallel_state.get_otp_group", return_value=otp_group),
        ):
            self.assertIs(_linear_reduce_group(layer), otp_group)

    def test_reduction_follows_the_fine_grained_down_proj_group(self):
        layer = MagicMock(spec=RowParallelLinear)
        layer.tp_size = 8
        layer.input_size = 64
        layer.input_size_per_partition = 8
        layer.prefix = "model.layers.0.mlp.down_proj"
        mlp_tp_group = object()
        with (
            patch(f"{MODULE}.oproj_tp_enable", return_value=False),
            patch(f"{MODULE}.mlp_tp_enable", return_value=True),
            patch("vllm_ascend.distributed.parallel_state.get_mlp_tp_group", return_value=mlp_tp_group),
        ):
            self.assertIs(_linear_reduce_group(layer), mlp_tp_group)


class TestAscendOnlineQuantizationConfig(TestBase):
    def test_supports_the_int8_weight_key(self):
        method_cls = _int8_moe_config()._get_method_cls(QuantSpec(weight=kInt8StaticChannelSym), {}, LinearBase)

        self.assertIs(method_cls, AscendOnlineInt8LinearMethod)

    def test_rejects_unsupported_weight_keys_with_a_hardware_hint(self):
        with self.assertRaises(NotImplementedError) as raised:
            _int8_moe_config()._get_method_cls(QuantSpec(weight=kFp8StaticChannelSym), {}, LinearBase)

        self.assertIn("INT8", str(raised.exception))

    def test_rejects_activation_overrides(self):
        spec = QuantSpec(weight=kInt8StaticChannelSym, activation=kInt8StaticChannelSym)

        with self.assertRaises(ValueError):
            _int8_moe_config()._get_method_cls(spec, {}, LinearBase)

    def test_accepts_both_a_layer_and_a_layer_type(self):
        # vLLM #55686 changed the third argument of _get_method_cls from the
        # layer instance to its type. Both spellings must keep resolving.
        config = _int8_moe_config()

        for layer_arg in (LinearBase, _LinearLayer()):
            self.assertIs(
                config._get_method_cls(QuantSpec(weight=kInt8StaticChannelSym), {}, layer_arg),
                AscendOnlineInt8LinearMethod,
            )

    def test_override_still_matches_the_upstream_signature(self):
        # _get_method_cls is a private vLLM hook with a single upstream caller,
        # so a reshape of it must fail here rather than in production.
        import inspect

        from vllm.model_executor.layers.quantization.online.base import (
            OnlineQuantizationConfig as UpstreamOnlineQuantizationConfig,
        )

        ours = inspect.signature(AscendOnlineQuantizationConfig._get_method_cls).parameters
        theirs = inspect.signature(UpstreamOnlineQuantizationConfig._get_method_cls).parameters

        self.assertEqual(
            [parameter.kind for parameter in ours.values()],
            [parameter.kind for parameter in theirs.values()],
            "vLLM changed the shape of _get_method_cls; update the Ascend override in "
            "vllm_ascend/quantization/configs/online_config.py.",
        )

    def test_moe_only_shorthand_leaves_linears_on_the_ascend_unquantized_path(self):
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        config = _int8_moe_config()
        layer = _LinearLayer("model.layers.0.mlp.gate_proj")

        self.assertIsInstance(
            config.get_quant_method(layer, "model.layers.0.mlp.gate_proj"),
            AscendUnquantizedLinearMethod,
        )

    def test_ignored_layers_also_use_the_ascend_unquantized_path(self):
        from vllm.config.quantization import QuantizationConfigArgs

        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        config = AscendOnlineQuantizationConfig(
            QuantizationConfigArgs(
                moe=QuantSpec(weight=kInt8StaticChannelSym),
                ignore=["model.layers.0.mlp.gate_proj"],
            )
        )
        layer = _LinearLayer("model.layers.0.mlp.gate_proj")

        self.assertIsInstance(
            config.get_quant_method(layer, "model.layers.0.mlp.gate_proj"),
            AscendUnquantizedLinearMethod,
        )

    def test_matched_linear_layer_is_wrapped_in_the_online_adapter(self):
        from vllm.config.quantization import QuantizationConfigArgs

        from vllm_ascend.quantization.method_adapters import AscendOnlineLinearMethod

        config = AscendOnlineQuantizationConfig(QuantizationConfigArgs(linear=QuantSpec(weight=kInt8StaticChannelSym)))
        layer = _LinearLayer("model.layers.0.mlp.down_proj")
        vllm_config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))

        # OnlineLinearBase.__init__ records the model dtype.
        with patch(
            "vllm.model_executor.layers.quantization.online.fp8.get_current_vllm_config",
            return_value=vllm_config,
        ):
            method = config.get_quant_method(layer, "model.layers.0.mlp.down_proj")

        self.assertIsInstance(method, AscendOnlineLinearMethod)
        self.assertIn("model.layers.0.mlp.down_proj", config.quantized_layers)


class TestAscendOnlineQuantConfig(TestBase):
    def _profile(self, family):
        return patch(
            "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
            return_value=SimpleNamespace(quantization_backend_family=family),
        )

    def test_standard_backend_builds_the_config(self):
        from vllm_ascend.device.hardware_profile import QuantizationBackendFamily

        args = QuantizationConfigArgs(moe=QuantSpec(weight=kInt8StaticChannelSym))
        with self._profile(QuantizationBackendFamily.STANDARD):
            config = AscendOnlineQuantizationConfig(args)

        self.assertIs(config.args, args)
        self.assertEqual(AscendOnlineQuantizationConfig.get_min_capability(), 0)

    def test_non_standard_quantization_backend_is_rejected(self):
        from vllm_ascend.device.hardware_profile import QuantizationBackendFamily

        args = QuantizationConfigArgs(moe=QuantSpec(weight=kInt8StaticChannelSym))
        with self._profile(QuantizationBackendFamily.COMPATIBILITY), self.assertRaises(NotImplementedError):
            AscendOnlineQuantizationConfig(args)

    def test_inherits_upstream_argument_validation(self):
        """vLLM's own check still applies: a config must select something."""
        from vllm_ascend.device.hardware_profile import QuantizationBackendFamily

        with self._profile(QuantizationBackendFamily.STANDARD), self.assertRaises(ValueError):
            AscendOnlineQuantizationConfig(QuantizationConfigArgs())


class TestBuildAscendOnlineQuantConfig(TestBase):
    def _standard_profile(self):
        from vllm_ascend.device.hardware_profile import QuantizationBackendFamily

        return patch(
            "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
            return_value=SimpleNamespace(quantization_backend_family=QuantizationBackendFamily.STANDARD),
        )

    def test_shorthand_builds_the_ascend_config(self):
        model_config = SimpleNamespace(
            quantization="int8_per_channel_weight_only",
            quantization_config=None,
            dtype=torch.bfloat16,
        )

        with self._standard_profile():
            config = build_ascend_online_quant_config(model_config)

        self.assertIsInstance(config, AscendOnlineQuantizationConfig)
        self.assertIsNotNone(config.args.moe)
        self.assertIsNone(config.args.linear)

    def test_dict_quantization_config_is_normalized(self):
        model_config = SimpleNamespace(
            quantization="online",
            quantization_config={"moe": "int8_per_channel_static"},
            dtype=torch.bfloat16,
        )

        with self._standard_profile():
            config = build_ascend_online_quant_config(model_config)

        self.assertEqual(config.args.moe, QuantSpec(weight=kInt8StaticChannelSym))

    def test_explicit_online_requires_a_quantization_config(self):
        model_config = SimpleNamespace(quantization="online", quantization_config=None, dtype=torch.bfloat16)

        with self._standard_profile(), self.assertRaises(ValueError) as raised:
            build_ascend_online_quant_config(model_config)

        self.assertIn("--quantization-config", str(raised.exception))

    def test_unsupported_dtype_is_rejected(self):
        model_config = SimpleNamespace(
            quantization="int8_per_channel_weight_only",
            quantization_config=None,
            dtype=torch.float32,
        )

        with self._standard_profile(), self.assertRaises(ValueError) as raised:
            build_ascend_online_quant_config(model_config)

        self.assertIn("not supported", str(raised.exception))

    def test_non_standard_quantization_backend_is_rejected(self):
        from vllm_ascend.device.hardware_profile import QuantizationBackendFamily

        model_config = SimpleNamespace(
            quantization="int8_per_channel_weight_only",
            quantization_config=None,
            dtype=torch.bfloat16,
        )
        with (
            patch(
                "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
                return_value=SimpleNamespace(quantization_backend_family=QuantizationBackendFamily.COMPATIBILITY),
            ),
            self.assertRaises(NotImplementedError),
        ):
            build_ascend_online_quant_config(model_config)


class TestQuantizationConfigPatch(TestBase):
    def test_ascend_online_names_are_diverted(self):
        from vllm.config.vllm import VllmConfig

        patched = VllmConfig._get_quantization_config
        self.assertTrue(getattr(patched, "_vllm_ascend_online_quant_patched", False))

        model_config = SimpleNamespace(quantization="int8_per_channel_weight_only", quantization_config=None)
        sentinel = object()
        with patch(f"{CONFIG_MODULE}.build_ascend_online_quant_config", return_value=sentinel) as builder:
            self.assertIs(patched(model_config, None), sentinel)

        builder.assert_called_once_with(model_config)

    def test_other_quantization_methods_keep_using_upstream(self):
        from vllm.config.vllm import VllmConfig

        model_config = SimpleNamespace(quantization="compressed-tensors", quantization_config=None)
        # The fake ModelConfig cannot drive the upstream path; what matters here
        # is that the Ascend builder is never consulted.
        with (
            patch(f"{CONFIG_MODULE}.build_ascend_online_quant_config") as builder,
            contextlib.suppress(Exception),
        ):
            VllmConfig._get_quantization_config(model_config, None)

        builder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
