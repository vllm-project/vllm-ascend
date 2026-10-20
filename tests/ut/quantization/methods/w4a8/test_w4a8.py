from types import SimpleNamespace
from unittest.mock import Mock, patch

import regex as re
import torch

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import identity
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily
from vllm_ascend.quantization.methods.w4a8.w4a8 import (
    AscendW4A8DynamicFusedMoEMethod,
    _expand_a5_w4a8_scale,
)
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD, COMPRESSED_TENSORS_METHOD


class TestAscendW4A8DynamicFusedMoEMethod(TestBase):
    experts = 8
    input_size = 16
    output_size = 56

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_ascend_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_mc2_group")
    @patch("torch.distributed.get_rank", return_value=0)
    def setUp(self, mock_get_rank, mock_get_mc2_group, get_current_vllm_config, mock_get_ascend_config):
        # Mock ascend config
        mock_ascend_config = Mock()
        mock_ascend_config.eplb_config.dynamic_eplb = False
        mock_get_ascend_config.return_value = mock_ascend_config

        mock_vllm_config = Mock()
        mock_vllm_config.quant_config = Mock(quant_description={"group_size": 0})
        mock_vllm_config.quant_config.get_name.return_value = ASCEND_QUANTIZATION_METHOD
        mock_vllm_config.parallel_config = Mock(enable_expert_parallel=True, enable_eplb=False)
        mock_vllm_config.use_v2_model_runner = False
        mock_vllm_config.scheduler_config = Mock(
            max_num_batched_tokens=2048, max_model_len=2048, enable_chunked_prefill=False
        )
        get_current_vllm_config.return_value = mock_vllm_config
        self.quant_method = AscendW4A8DynamicFusedMoEMethod()

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_current_vllm_config")
    def test_init_raise_error(self, get_current_vllm_config, mock_tp_size):
        # test init rejects per-group quantization
        mock_vllm_config = Mock()
        mock_vllm_config.quant_config = Mock(quant_description={"group_size": 256})
        mock_vllm_config.parallel_config = Mock(enable_expert_parallel=True)
        mock_vllm_config.use_v2_model_runner = False
        get_current_vllm_config.return_value = mock_vllm_config
        with self.assertRaisesRegex(ValueError, "no longer supported"):
            AscendW4A8DynamicFusedMoEMethod()

        # test init rejects tp>16 when use modelslim weights
        mock_vllm_config.use_v2_model_runner = True
        mock_vllm_config.quant_config = Mock(quant_description={})
        mock_vllm_config.quant_config.get_name.return_value = ASCEND_QUANTIZATION_METHOD
        mock_vllm_config.parallel_config = Mock(enable_expert_parallel=False)
        mock_tp_size.return_value = 32
        with self.assertRaisesRegex(ValueError, "tp>16"):
            AscendW4A8DynamicFusedMoEMethod()

    def test_get_weight(self):
        param_dict = self.quant_method.get_weight(self.experts, self.input_size, self.output_size, torch.bfloat16)
        self.assertEqual(param_dict["w13_weight"].dtype, torch.int8)
        self.assertEqual(param_dict["w13_weight"].shape, (self.experts, self.input_size, self.output_size))
        self.assertEqual(param_dict["w2_weight"].dtype, torch.int8)
        self.assertEqual(param_dict["w2_weight"].shape, (self.experts, self.output_size // 2, self.input_size))

    def test_get_weight_compressed_tensors(self):
        self.quant_method.quant_method = COMPRESSED_TENSORS_METHOD
        result = self.quant_method.get_weight(self.experts, self.input_size, self.output_size, torch.bfloat16)
        self.assertEqual(result["w13_weight"].dtype, torch.int8)
        self.assertEqual(result["w13_weight"].shape, (self.experts, 2 * self.input_size, self.output_size))
        self.assertEqual(result["w2_weight"].shape, (self.experts, self.output_size, self.input_size))

    def test_get_dynamic_quant_param(self):
        param_dict = self.quant_method.get_dynamic_quant_param(
            self.experts, self.input_size, self.output_size, torch.bfloat16
        )
        self.assertEqual(param_dict["w13_weight_scale"].dtype, torch.float32)
        self.assertEqual(param_dict["w13_weight_scale"].shape, (self.experts, 2 * self.input_size, 1))
        self.assertEqual(param_dict["w13_weight_offset"].dtype, torch.float32)
        self.assertEqual(param_dict["w13_weight_offset"].shape, (self.experts, 2 * self.input_size, 1))
        self.assertEqual(param_dict["w2_weight_scale"].dtype, torch.float32)
        self.assertEqual(param_dict["w2_weight_scale"].shape, (self.experts, self.output_size, 1))
        self.assertEqual(param_dict["w2_weight_offset"].dtype, torch.float32)
        self.assertEqual(param_dict["w2_weight_offset"].shape, (self.experts, self.output_size, 1))
        self.assertEqual(param_dict["w13_scale_bias"].dtype, torch.float32)
        self.assertEqual(param_dict["w13_scale_bias"].shape, (self.experts, 2 * self.input_size, 1))
        self.assertEqual(param_dict["w2_scale_bias"].dtype, torch.float32)
        self.assertEqual(
            param_dict["w2_scale_bias"].shape, (self.experts, self.output_size, 16 // self.quant_method.tp_size)
        )
        pergroup_param = [
            "w13_weight_scale_second",
            "w13_weight_offset_second",
            "w2_weight_scale_second",
            "w2_weight_offset_second",
        ]
        is_contains = any(key in param_dict for key in pergroup_param)
        self.assertFalse(is_contains)

    def test_get_dynamic_quant_param_compressed_tensors(self):
        self.quant_method.quant_method = COMPRESSED_TENSORS_METHOD
        result = self.quant_method.get_dynamic_quant_param(
            self.experts, self.input_size, self.output_size, torch.bfloat16
        )
        self.assertIn("w13_weight_scale", result)
        self.assertIn("w2_weight_scale", result)
        self.assertEqual(result["w13_weight_scale"].dtype, torch.bfloat16)
        self.assertEqual(result["w13_weight_scale"].shape, (self.experts, 2 * self.input_size, 1))
        self.assertEqual(result["w2_weight_scale"].dtype, torch.bfloat16)
        self.assertEqual(result["w2_weight_scale"].shape, (self.experts, self.output_size, 1))

    def build_layer(self):
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros((self.experts, self.input_size, self.output_size), dtype=torch.int8), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.zeros((self.experts, self.output_size // 2, self.input_size), dtype=torch.int8),
            requires_grad=False,
        )
        w13_scale_bias = torch.zeros((self.experts, 2 * self.input_size, 1), dtype=torch.float32)
        layer.w13_scale_bias = torch.nn.Parameter(w13_scale_bias, requires_grad=False)
        w2_scale_bias = torch.zeros(
            (self.experts, self.output_size, 16 // self.quant_method.tp_size), dtype=torch.float32
        )
        layer.w2_scale_bias = torch.nn.Parameter(w2_scale_bias, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.ones((self.experts, 2 * self.input_size, 1), dtype=torch.float32), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.ones((self.experts, self.output_size, 1), dtype=torch.float32), requires_grad=False
        )
        return layer

    def build_layer_compressed_tensors(self):
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros((self.experts, 2 * self.input_size, self.output_size), dtype=torch.int8), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.zeros((self.experts, self.output_size, self.input_size), dtype=torch.int8), requires_grad=False
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.ones((self.experts, 2 * self.input_size, 1), dtype=torch.bfloat16), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.ones((self.experts, self.output_size, 1), dtype=torch.bfloat16), requires_grad=False
        )
        return layer

    def test_get_eplb_weight_views_matches_routed_compute_inputs(self):
        layer = self.build_layer()

        weight_views = self.quant_method.get_eplb_weight_views(layer)

        expected = [
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            layer.w13_scale_bias,
            layer.w2_scale_bias,
        ]
        self.assertEqual(len(weight_views), len(expected))
        for actual, expected_view in zip(weight_views, expected):
            self.assertIs(actual, expected_view)

    def test_get_eplb_weight_views_rejects_incomplete_scale_bias(self):
        layer = self.build_layer()
        del layer.w2_scale_bias

        with self.assertRaisesRegex(RuntimeError, "w13_scale_bias and w2_scale_bias to be present or absent together"):
            self.quant_method.get_eplb_weight_views(layer)

    def test_get_eplb_weight_views_rejects_incomplete_scale_bias_lists(self):
        layer = torch.nn.Module()
        layer.w13_weight_list = [torch.empty(2, 3) for _ in range(self.experts)]
        layer.w2_weight_list = [torch.empty(3, 2) for _ in range(self.experts)]
        layer.w13_weight_scale_list = [torch.empty(3) for _ in range(self.experts)]
        layer.w2_weight_scale_list = [torch.empty(2) for _ in range(self.experts)]
        layer.w13_scale_bias_list = [torch.empty(3) for _ in range(self.experts)]
        layer.w2_scale_bias_list = None

        with self.assertRaisesRegex(
            RuntimeError,
            "w13_scale_bias_list and w2_scale_bias_list to be present or absent together",
        ):
            self.quant_method.get_eplb_weight_views(layer)

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_current_vllm_config", new=lambda: None)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.use_cann_megamoe", new=lambda _: False)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_ascend_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.maybe_trans_nz")
    @patch("torch.Tensor.npu", new=lambda self: self, create=True)
    def test_process_weights_after_loading(self, mock_maybe_trans_nz, mock_get_ascend_config):
        mock_maybe_trans_nz.side_effect = identity
        mock_get_ascend_config.return_value.enable_fused_mc2 = 0
        layer = self.build_layer()
        self.quant_method.process_weights_after_loading(layer)
        self.assertEqual(layer.w13_scale_bias.data.shape, (self.experts, 2 * self.input_size))
        self.assertEqual(layer.w13_scale_bias.data.dtype, torch.float32)
        self.assertEqual(layer.w2_scale_bias.data.shape, (self.experts, self.output_size))
        self.assertEqual(layer.w2_scale_bias.data.dtype, torch.float32)

        self.quant_method.use_expert_weight_list = True
        list_layer = self.build_layer()
        self.quant_method.process_weights_after_loading(list_layer)
        self.assertFalse(hasattr(list_layer, "w13_weight"))
        self.assertEqual(len(list_layer.w13_weight_list), self.experts)
        self.assertTrue(all(weight.storage_offset() == 0 for weight in list_layer.w13_weight_list))
        weight_views = self.quant_method.get_eplb_weight_views(list_layer)
        self.assertIs(weight_views[0], list_layer.w13_weight_list)
        self.assertIs(weight_views[-1], list_layer.w2_scale_bias_list)

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_current_vllm_config", new=lambda: None)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.use_cann_megamoe", new=lambda _: False)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_ascend_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.maybe_trans_nz")
    @patch("torch.Tensor.npu", new=lambda self: self, create=True)
    def test_process_weights_after_loading_compressed_tensors(self, mock_maybe_trans_nz, mock_get_ascend_config):
        mock_maybe_trans_nz.side_effect = identity
        mock_get_ascend_config.return_value.enable_fused_mc2 = 0
        self.quant_method.quant_method = COMPRESSED_TENSORS_METHOD
        layer = self.build_layer_compressed_tensors()
        self.quant_method.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, "w13_scale_bias"))
        self.assertEqual(layer.w13_scale_bias.data.shape, (self.experts, 2 * self.input_size))
        self.assertEqual(layer.w13_scale_bias.data.dtype, torch.float32)
        self.assertEqual(layer.w2_scale_bias.data.shape, (self.experts, self.output_size))

        self.quant_method.use_expert_weight_list = True
        list_layer = self.build_layer_compressed_tensors()
        self.quant_method.process_weights_after_loading(list_layer)
        self.assertFalse(hasattr(list_layer, "w13_weight"))
        self.assertEqual(len(list_layer.w13_weight_list), self.experts)
        self.assertEqual(list_layer.w13_weight_list[0].dtype, torch.int8)
        self.assertTrue(all(weight.storage_offset() == 0 for weight in list_layer.w13_weight_list))

    def test_pack_to_int32_asserts_packed_dim(self):
        weight = torch.zeros((self.experts, self.output_size, 10), dtype=torch.int8)
        expected_message = f"the last dim of weight needs to be divided by 4 but got shape {weight.shape}"

        with self.assertRaisesRegex(AssertionError, re.escape(expected_message)):
            self.quant_method._pack_to_int32(weight)

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8._EXTRA_CTX")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.build_fused_experts_input")
    def test_apply_comprehensive(self, mock_build_input, mock_ctx):
        tokens = 4
        num_experts = self.experts
        hidden_size = self.output_size
        top_k = 2

        layer = self.build_layer()
        layer.swiglu_limit = 1000000
        x = torch.randn(tokens, hidden_size, dtype=torch.bfloat16)
        topk_weights = torch.randn(tokens, top_k, dtype=torch.float32)
        topk_ids = torch.randint(0, num_experts, (tokens, top_k), dtype=torch.int64)
        expert_map = torch.randint(0, num_experts, (num_experts,), dtype=torch.int64)
        mc2_mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        pertoken_scale = torch.randn(tokens, dtype=torch.float32)
        log2phy = torch.randint(0, num_experts, (num_experts,), dtype=torch.int64)
        layer.ascend_expert_map = expert_map
        layer.global_redundant_expert_num = 0
        layer.log2phy = log2phy
        layer.ascend_mc2_mask = mc2_mask
        layer.ascend_pertoken_scale = pertoken_scale
        layer.activation = "silu"
        layer.apply_router_weight_on_input = False
        layer.swiglu_alpha = 1.0
        layer.swiglu_beta = 0.0

        mock_fused_input = Mock()
        mock_fused_input.hidden_states = x
        mock_fused_input.topk_weights = topk_weights
        mock_fused_input.topk_ids = topk_ids
        mock_fused_input.activation = "silu"
        mock_build_input.return_value = mock_fused_input

        mock_comm = Mock()
        expected_output = torch.randn(tokens, hidden_size, dtype=torch.bfloat16)
        mock_comm.fused_experts.return_value = expected_output
        mock_ctx.moe_comm_method = mock_comm
        mock_ctx.use_mega_moe = False

        output = self.quant_method.apply(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )

        mock_build_input.assert_called_once()
        build_kwargs = mock_build_input.call_args.kwargs
        self.assertTrue(torch.equal(build_kwargs["hidden_states"], x))
        self.assertEqual(build_kwargs["quant_type"], self.quant_method.quant_type)
        self.assertTrue(build_kwargs["is_per_channel_weight"])
        self.assertEqual(build_kwargs["activation"], "silu")
        self.assertEqual(build_kwargs["apply_router_weight_on_input"], False)

        mock_comm.fused_experts.assert_called_once()
        self.assertEqual(mock_comm.fused_experts.call_args.kwargs["fused_experts_input"], mock_fused_input)
        self.assertTrue(torch.equal(output, expected_output))


class TestAscendW4A8A5Fallback(TestBase):
    def setUp(self):
        self.config = SimpleNamespace(
            quant_config=SimpleNamespace(quant_description={}, get_name=lambda: ASCEND_QUANTIZATION_METHOD),
            parallel_config=SimpleNamespace(enable_expert_parallel=True, enable_eplb=False),
            use_v2_model_runner=True,
        )
        self.ascend_config = SimpleNamespace(enable_fused_mc2=0, eplb_config=SimpleNamespace(dynamic_eplb=False))
        self.profile = SimpleNamespace(device_adaptor_family=DeviceAdaptorFamily.FP8_OPTIMIZED)
        module = "vllm_ascend.quantization.methods.w4a8.w4a8"
        for target, options in (
            ("get_current_vllm_config", {"return_value": self.config}),
            ("get_ascend_config", {"return_value": self.ascend_config}),
            ("get_current_hardware_profile", {"return_value": self.profile}),
            ("get_tensor_model_parallel_world_size", {"return_value": 8}),
            ("get_mc2_group", {"side_effect": AttributeError("CPU test has no device group")}),
        ):
            patcher = patch(f"{module}.{target}", **options)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_scale_expansion_preserves_encoded_bits_and_dequantized_weights(self):
        encoded = torch.tensor([[0, 0xFFFFFFFF, 0x3D800000, 0x80000000] * 8] * 2, dtype=torch.int64)
        generator = torch.Generator().manual_seed(3108)
        for input_size, groups in ((4096, 16), (1280, 5)):
            weight = torch.empty((2, input_size, 16), dtype=torch.int8)
            for scale in (encoded, encoded.unsqueeze(1)):
                with self.subTest(input_size=input_size, scale_shape=scale.shape):
                    expanded = _expand_a5_w4a8_scale(scale, weight)
                    self.assertEqual(expanded.shape, (2, groups, 32))
                    self.assertEqual(expanded.dtype, torch.int64)
                    self.assertTrue(expanded.is_contiguous())
                    for group in range(groups):
                        torch.testing.assert_close(expanded[:, group], encoded, rtol=0, atol=0)

            quantized_weight = torch.randint(-8, 8, (2, input_size, 32), dtype=torch.int8, generator=generator)
            channel_scale = torch.rand((2, 32), generator=generator) / 16
            encoded_scale = channel_scale.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
            expanded = _expand_a5_w4a8_scale(encoded_scale, weight).to(torch.int32).view(torch.float32)
            per_group = quantized_weight.reshape(2, groups, 256, 32).float() * expanded[:, :, None, :]
            per_channel = quantized_weight.float() * channel_scale[:, None, :]
            torch.testing.assert_close(per_group.reshape_as(per_channel), per_channel, rtol=0, atol=0)

    def test_scale_expansion_rejects_invalid_layouts(self):
        encoded = torch.zeros((2, 32), dtype=torch.int64)
        valid_weight = torch.empty((2, 4096, 16), dtype=torch.int8)
        for scale, weight in (
            (encoded, torch.empty((2, 1281, 16), dtype=torch.int8)),
            (encoded, torch.empty((2, 0, 16), dtype=torch.int8)),
            (encoded, torch.empty((4096, 16), dtype=torch.int8)),
            (encoded, torch.empty((2, 4096, 4), dtype=torch.int32)),
            (encoded.float(), valid_weight),
            (encoded[:, :16], valid_weight),
            (encoded[:1], valid_weight),
            (encoded.reshape(2, 2, 16), valid_weight),
        ):
            with (
                self.subTest(scale_shape=scale.shape, weight_shape=weight.shape, scale_dtype=scale.dtype),
                self.assertRaises(ValueError),
            ):
                _expand_a5_w4a8_scale(scale, weight)

    def test_fallback_is_limited_to_a5_modelslim_and_does_not_change_class_activation(self):
        original_activations = AscendW4A8DynamicFusedMoEMethod.fused_activations
        for family in DeviceAdaptorFamily:
            for quantization in (ASCEND_QUANTIZATION_METHOD, COMPRESSED_TENSORS_METHOD):
                with self.subTest(family=family, quantization=quantization):
                    self.profile.device_adaptor_family = family
                    self.config.quant_config.get_name = Mock(return_value=quantization)
                    method = AscendW4A8DynamicFusedMoEMethod()
                    expected = (
                        family == DeviceAdaptorFamily.FP8_OPTIMIZED and quantization == ASCEND_QUANTIZATION_METHOD
                    )
                    self.assertEqual(method._use_a5_unfused_w4a8, expected)
                    self.assertEqual(method.fused_activations, frozenset() if expected else original_activations)
        self.assertEqual(AscendW4A8DynamicFusedMoEMethod.fused_activations, original_activations)

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.logger")
    def test_fallback_is_reported_only_on_a5(self, logger):
        self.profile.device_adaptor_family = DeviceAdaptorFamily.STANDARD
        AscendW4A8DynamicFusedMoEMethod()
        logger.warning_once.assert_not_called()
        self.profile.device_adaptor_family = DeviceAdaptorFamily.FP8_OPTIMIZED
        AscendW4A8DynamicFusedMoEMethod()
        logger.warning_once.assert_called_once()
        self.assertIn("Fused GMM-activation-quantization is disabled", logger.warning_once.call_args.args[0])

    def test_non_a5_keeps_fused_parallel_configurations(self):
        self.profile.device_adaptor_family = DeviceAdaptorFamily.STANDARD
        self.config.parallel_config.enable_expert_parallel = False
        self.config.parallel_config.enable_eplb = True
        self.ascend_config.enable_fused_mc2 = 1
        method = AscendW4A8DynamicFusedMoEMethod()
        self.assertFalse(method._use_a5_unfused_w4a8)
        self.assertEqual(method.fused_activations, frozenset({"silu", "situ"}))
        self.assertTrue(method.use_expert_weight_list)

    def test_fallback_rejects_unsupported_parallel_paths(self):
        for owner, attribute, value in (
            (self.config.parallel_config, "enable_expert_parallel", False),
            (self.config.parallel_config, "enable_eplb", True),
            (self.ascend_config, "enable_fused_mc2", 1),
            (self.ascend_config.eplb_config, "dynamic_eplb", True),
        ):
            with self.subTest(attribute=attribute):
                previous = getattr(owner, attribute)
                setattr(owner, attribute, value)
                self.config.use_v2_model_runner = attribute != "dynamic_eplb"
                try:
                    with self.assertRaisesRegex(ValueError, "requires EP without EPLB"):
                        AscendW4A8DynamicFusedMoEMethod()
                finally:
                    setattr(owner, attribute, previous)
                    self.config.use_v2_model_runner = True

    @patch("torch.Tensor.npu", new=lambda tensor: tensor, create=True)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.maybe_trans_nz")
    def test_load_preserves_nd_packed_weights_msd_assist_and_reuses_scales(self, trans_nz):
        method = AscendW4A8DynamicFusedMoEMethod()
        experts, hidden, intermediate = 2, 4096, 1280
        generator = torch.Generator().manual_seed(3108)
        layer = torch.nn.Module()
        tensors = {
            "w13_weight": torch.randint(
                -128, 128, (experts, intermediate, hidden), dtype=torch.int8, generator=generator
            ),
            "w2_weight": torch.randint(
                -128, 128, (experts, hidden // 2, intermediate), dtype=torch.int8, generator=generator
            ),
            "w13_weight_scale": torch.rand((experts, 2 * intermediate, 1), generator=generator) / 16,
            "w2_weight_scale": torch.rand((experts, hidden, 1), generator=generator) / 16,
            "w13_scale_bias": (
                torch.randint(-128, 128, (experts, 2 * intermediate, 1), generator=generator).float() / 16
            ),
            "w2_scale_bias": torch.randint(-128, 128, (experts, hidden, 16), generator=generator).float() / 16,
        }
        for name, tensor in tensors.items():
            layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
        method.process_weights_after_loading(layer)
        trans_nz.assert_not_called()
        self.assertEqual(layer.w13_weight.shape, (experts, hidden, intermediate // 4))
        self.assertEqual(layer.w2_weight.shape, (experts, intermediate, hidden // 8))
        self.assertEqual(layer.w13_weight_scale.shape, (experts, 16, 2 * intermediate))
        self.assertEqual(layer.w2_weight_scale.shape, (experts, 5, hidden))
        for name in ("w13_weight", "w2_weight"):
            packed = getattr(layer, name)
            self.assertEqual(packed.dtype, torch.int32)
            self.assertTrue(packed.is_contiguous())
            torch.testing.assert_close(packed.view(torch.int8), tensors[name].transpose(1, 2), rtol=0, atol=0)
        for name in ("w13_weight_scale", "w2_weight_scale"):
            expanded = getattr(layer, name)
            self.assertTrue(expanded.is_contiguous())
            decoded = expanded.to(torch.int32).view(torch.float32)
            expected = tensors[name].squeeze(-1)
            for group in range(decoded.shape[1]):
                torch.testing.assert_close(decoded[:, group], expected, rtol=0, atol=0)
        for name in ("w13_scale_bias", "w2_scale_bias"):
            torch.testing.assert_close(getattr(layer, name), tensors[name].sum(dim=-1), rtol=0, atol=0)

        # Forward must pass the prepared scales and MSD corrections unchanged;
        # repeated graph calls must not expand or copy weight-scale buffers.
        tokens = torch.ones((4, hidden), dtype=torch.int8)
        token_scale = torch.ones(4)
        compute_input = SimpleNamespace(
            layer=layer,
            hidden_states=tokens,
            dynamic_scale=token_scale,
            group_list=torch.tensor([2, 2]),
            group_list_type=1,
        )
        module = "vllm_ascend.quantization.methods.w4a8.w4a8"
        with (
            patch.object(method, "_quant_hidden_states", return_value=(tokens, token_scale)),
            patch(f"{module}.dispose_tensor"),
            patch(f"{module}.torch_npu.npu_grouped_matmul", return_value=[tokens]) as grouped_matmul,
        ):
            for _ in range(2):
                method.apply_gmm1(compute_input)
                first = grouped_matmul.call_args.kwargs
                self.assertEqual(first["scale"][0].data_ptr(), layer.w13_weight_scale.data_ptr())
                self.assertIs(first["bias"][0], layer.w13_scale_bias)
                method.apply_gmm2(compute_input, tokens, token_scale)
                second = grouped_matmul.call_args.kwargs
                self.assertIs(second["scale"][0], layer.w2_weight_scale)
                self.assertIs(second["bias"][0], layer.w2_scale_bias)

        # Reject lists before a second transformation can corrupt loaded data.
        original_weight = layer.w13_weight
        layer.w13_weight_list = []
        with self.assertRaisesRegex(ValueError, "does not support expert weight lists"):
            method.process_weights_after_loading(layer)
        self.assertIs(layer.w13_weight, original_weight)
