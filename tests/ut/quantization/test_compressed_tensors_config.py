from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import RowParallelLinear, UnquantizedLinearMethod

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import (
    COMPRESSED_TENSORS_W8A8_CONFIG,
    create_mock_ascend_config,
    create_mock_vllm_config,
)
from vllm_ascend.quantization.compressed_tensors_config import AscendCompressedTensorsConfig
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods import (
    AscendW4A8MXFPDynamicFusedMoEMethod,
    AscendW8A8DynamicLinearMethod,
)
from vllm_ascend.utils import COMPRESSED_TENSORS_METHOD

KIMI_K3_MXFP4_CONFIG = {
    "config_groups": {
        "group_0": {
            "format": "mxfp4-pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": 32,
                "num_bits": 4,
                "observer": None,
                "observer_kwargs": {},
                "scale_dtype": "torch.uint8",
                "strategy": "group",
                "symmetric": True,
                "type": "float",
            },
        }
    },
    "format": "mxfp4-pack-quantized",
    "ignore": [
        "re:.*self_attn.*",
        "re:.*shared_experts.*",
        r"re:.*mlp\.(gate|up|gate_up|down)_proj.*",
        "re:.*lm_head.*",
        "re:.*vision_tower.*",
        "re:.*mm_projector.*",
    ],
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
}


class TestAscendCompressedTensorsQuanType(TestBase):
    def setUp(self):
        self.config = AscendCompressedTensorsConfig(
            target_scheme_map={"Linear": {}},
            ignore=["lm_head"],
            quant_format="",
            config={},
        )

    def _make_weight_quant(self, num_bits=8, strategy="channel", dynamic=False, symmetric=True, group_size=None):
        mock = MagicMock()
        mock.num_bits = num_bits
        mock.strategy = strategy
        mock.dynamic = dynamic
        mock.symmetric = symmetric
        mock.group_size = group_size
        return mock

    def _make_input_quant(self, num_bits=8, strategy="tensor", dynamic=False, symmetric=True):
        mock = MagicMock()
        mock.num_bits = num_bits
        mock.strategy = strategy
        mock.dynamic = dynamic
        mock.symmetric = symmetric
        return mock

    def test_detect_w8a8_static(self):
        weight = self._make_weight_quant(num_bits=8, strategy="channel", dynamic=False, symmetric=True)
        input_q = self._make_input_quant(num_bits=8, strategy="tensor", dynamic=False, symmetric=True)
        result = self.config._detect_quant_type(weight, input_q, "int-quantized")
        self.assertEqual(result, "W8A8")

    def test_detect_w8a8_dynamic(self):
        weight = self._make_weight_quant(num_bits=8, strategy="channel", dynamic=False, symmetric=True)
        input_q = self._make_input_quant(num_bits=8, strategy="token", dynamic=True, symmetric=True)
        result = self.config._detect_quant_type(weight, input_q, "int-quantized")
        self.assertEqual(result, "W8A8_DYNAMIC")

    def test_detect_w4a8_dynamic(self):
        weight = self._make_weight_quant(num_bits=4, strategy="channel", dynamic=False, symmetric=True)
        input_q = self._make_input_quant(num_bits=8, strategy="token", dynamic=True, symmetric=True)
        result = self.config._detect_quant_type(weight, input_q, "int-quantized")
        self.assertEqual(result, "W4A8_DYNAMIC")

    def test_detect_w4a16(self):
        from compressed_tensors.quantization import QuantizationType

        weight = MagicMock()
        weight.num_bits = 4
        weight.strategy = "group"
        weight.dynamic = False
        weight.type = QuantizationType.INT
        result = self.config._detect_quant_type(weight, None, None)
        self.assertEqual(result, "W4A16")

    def test_detect_explicit_packed_mxfp4_as_w4a8_mxfp(self):
        from compressed_tensors.quantization import QuantizationType

        weight = self._make_weight_quant(num_bits=4, strategy="group", dynamic=False)
        weight.type = QuantizationType.FLOAT
        weight.group_size = 32

        result = self.config._detect_quant_type(weight, None, "mxfp4-pack-quantized")

        self.assertEqual(result, "W4A8_MXFP")

    def test_float4_is_not_mxfp4_without_exact_packed_format(self):
        from compressed_tensors.quantization import QuantizationType

        weight = self._make_weight_quant(num_bits=4, strategy="group", dynamic=False)
        weight.type = QuantizationType.FLOAT
        weight.group_size = 32

        for format in (None, "float-quantized", "mxfp4"):
            with self.subTest(format=format), self.assertRaises(NotImplementedError):
                self.config._detect_quant_type(weight, None, format)

    def test_packed_mxfp4_moe_scheme_uses_dynamic_mxfp_activation(self):
        from compressed_tensors.quantization import QuantizationType

        weight = self._make_weight_quant(num_bits=4, strategy="group", dynamic=False)
        weight.type = QuantizationType.FLOAT
        weight.group_size = 32
        with (
            patch(
                "vllm_ascend.quantization.methods.w4a8_mxfp4.get_current_vllm_config",
                return_value=create_mock_vllm_config(),
            ),
            patch(
                "vllm_ascend.quantization.methods.w4a8_mxfp4.get_ascend_config",
                return_value=create_mock_ascend_config(),
            ),
            patch("vllm_ascend.quantization.methods.w4a8_mxfp4.get_ep_group", return_value=MagicMock()),
        ):
            scheme = self.config._create_scheme_for_layer_type(
                weight_quant=weight,
                input_quant=None,
                format="mxfp4-pack-quantized",
                layer_type="moe",
            )

        self.assertIsInstance(scheme, AscendW4A8MXFPDynamicFusedMoEMethod)
        self.assertTrue(scheme.use_weight_packed)
        weights = scheme.get_weight(
            num_experts=2,
            intermediate_size_per_partition=64,
            hidden_sizes=128,
            params_dtype=torch.bfloat16,
        )
        scales = scheme.get_dynamic_quant_param(
            num_experts=2,
            intermediate_size_per_partition=64,
            hidden_sizes=128,
            params_dtype=torch.bfloat16,
        )
        self.assertEqual(set(weights), {"w13_weight_packed", "w2_weight_packed"})
        self.assertEqual(set(scales), {"w13_weight_scale", "w2_weight_scale"})

    def test_kimi_k3_real_targets_and_ignore_select_only_routed_experts(self):
        config = AscendCompressedTensorsConfig.from_config(KIMI_K3_MXFP4_CONFIG)
        linear = MagicMock(spec=RowParallelLinear)
        ignored_prefixes = (
            "model.layers.0.self_attn.q_proj",
            "model.layers.1.mlp.shared_experts.gate_proj",
            "model.layers.0.mlp.gate_proj",
            "lm_head",
            "vision_tower.blocks.0.mlp.gate_proj",
            "mm_projector.proj",
        )
        for prefix in ignored_prefixes:
            with self.subTest(prefix=prefix):
                self.assertIsNone(config.get_scheme_dict(linear, prefix))

        # FusedMoE is synthesized from the Linear target. Its expert prefix
        # deliberately does not match the dense/shared MLP ignore patterns.
        config._add_fused_moe_to_target_scheme_map()
        fused_moe = RoutedExperts.__new__(RoutedExperts)
        torch.nn.Module.__init__(fused_moe)
        expert_scheme = config.get_scheme_dict(
            fused_moe,
            "model.layers.1.mlp.experts.0.gate_proj",
        )
        assert expert_scheme is not None
        self.assertEqual(expert_scheme["format"], "mxfp4-pack-quantized")

    def test_detect_unsupported_raises(self):
        weight = self._make_weight_quant(num_bits=2, strategy="channel", dynamic=False, symmetric=True)
        input_q = self._make_input_quant(num_bits=2, strategy="tensor", dynamic=False, symmetric=True)
        with self.assertRaises(NotImplementedError):
            self.config._detect_quant_type(weight, input_q, "int_quantized")


class TestAscendCompressedTensorsConfigGetQuantMethod(TestBase):
    def setUp(self):
        self.config = AscendCompressedTensorsConfig.from_config(COMPRESSED_TENSORS_W8A8_CONFIG)

    @patch("vllm_ascend.quantization.method_adapters.AscendLinearMethod.__init__")
    def test_get_linear_quant_method(self, mock_method):
        mock_method.return_value = None
        layer = MagicMock(spec=RowParallelLinear)
        result = self.config.get_quant_method(layer, "model.layers.0.self_attn.q_proj")
        self.assertEqual(layer.ascend_quant_method, COMPRESSED_TENSORS_METHOD)
        self.assertTrue(isinstance(result, AscendLinearMethod))
        self.assertTrue(isinstance(layer.scheme, AscendW8A8DynamicLinearMethod))

    def test_get_linear_unquantized_method(self):
        layer = MagicMock(spec=RowParallelLinear)
        result = self.config.get_quant_method(layer, "lm_head")
        self.assertEqual(layer.ascend_quant_method, COMPRESSED_TENSORS_METHOD)
        self.assertTrue(isinstance(result, UnquantizedLinearMethod))

    def test_adds_routed_experts_target_for_linear_scheme(self):
        linear_scheme = self.config.target_scheme_map["Linear"]

        self.config._add_fused_moe_to_target_scheme_map()

        self.assertIs(self.config.target_scheme_map["RoutedExperts"], linear_scheme)

    @patch("vllm_ascend.quantization.compressed_tensors_config.find_matched_target", return_value=None)
    def test_get_scheme_dict_returns_none_for_unmatched_target(self, _mock_find_target):
        layer = MagicMock(spec=Attention)

        result = self.config.get_scheme_dict(layer, "model.layers.0.self_attn.attn")

        self.assertIsNone(result)

    @patch(
        "vllm_ascend.quantization.compressed_tensors_config.find_matched_target",
        return_value="Linear",
    )
    def test_get_scheme_dict_returns_none_for_none_scheme(self, _mock_find_target):
        self.config.target_scheme_map["Linear"] = None
        layer = MagicMock(spec=Attention)

        result = self.config.get_scheme_dict(layer, "model.layers.0.self_attn.attn")

        self.assertIsNone(result)

    @patch("vllm_ascend.quantization.method_adapters.AscendFusedMoEMethod")
    def test_get_routed_experts_quant_method(self, mock_method):
        layer = RoutedExperts.__new__(RoutedExperts)
        torch.nn.Module.__init__(layer)
        layer.moe_config = MagicMock()
        moe_scheme = MagicMock()

        with patch.object(self.config, "_get_moe_scheme", return_value=moe_scheme):
            result = self.config.get_quant_method(layer, "model.layers.0.mlp.experts")

        self.assertIs(result, mock_method.return_value)
        self.assertEqual(layer.ascend_quant_method, COMPRESSED_TENSORS_METHOD)
        self.assertIs(layer.scheme, moe_scheme)
        mock_method.assert_called_once_with(moe_scheme, layer.moe_config, None)

    def test_no_quant_method(self):
        layer = MagicMock(spec=Attention)
        result = self.config.get_quant_method(layer, "attn")
        self.assertIsNone(result)
